//! Concurrent analysis jobs.
//!
//! A job runs the tracking pipeline on a worker thread, streams annotated
//! frames (JPEG) into a [`FrameBus`] consumed by the browser, and finally
//! scores the 0.5 s windows. All mutable state lives behind `RwLock`s so the
//! HTTP handlers stay short and non-blocking.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use image::codecs::jpeg::JpegEncoder;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, watch};

use crate::pipeline::{self, AnalyzeReport, PipelineConfig, RunInfo, ScoredWindow};
use crate::web::telemetry::LogRecord;

/// Annotated frame handed to the UI.
#[derive(Clone, Debug, Serialize)]
pub struct FrameMeta {
    pub index: usize,
    pub t: f64,
    pub total: usize,
}

/// Decoded-and-annotated frame as JPEG bytes, plus where it sits in time.
#[derive(Clone)]
pub struct FramePacket {
    pub meta: FrameMeta,
    pub jpeg: Arc<Vec<u8>>,
}

/// Latest frame + a broadcast channel for live streaming.
///
/// The UI first fetches `/frame` (latest state, no replay), then subscribes to
/// `/frames` for every subsequent frame. Encoding is throttled by
/// [`FrameBus::should_encode`] so a 60 fps video at 640 px does not saturate the
/// CPU while the tracker is also running.
pub struct FrameBus {
    last: RwLock<Option<FramePacket>>,
    tx: broadcast::Sender<FramePacket>,
    last_emit: Mutex<Instant>,
    /// Minimum wall-clock gap between published frames.
    min_interval: Duration,
    /// Annotated frames are only produced when at least one viewer is attached.
    viewers: std::sync::atomic::AtomicUsize,
    jpeg_quality: u8,
}

impl FrameBus {
    pub fn new(fps_publish: f32, jpeg_quality: u8) -> Self {
        let (tx, _) = broadcast::channel(8);
        let min_interval = Duration::from_secs_f32(1.0 / fps_publish.max(1.0));
        Self {
            last: RwLock::new(None),
            tx,
            last_emit: Mutex::new(Instant::now() - Duration::from_secs(60)),
            min_interval,
            viewers: std::sync::atomic::AtomicUsize::new(0),
            jpeg_quality,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<FramePacket> {
        let n = self.viewers.fetch_add(1, Ordering::SeqCst);
        tracing::debug!(viewers = n + 1, "frame bus subscriber added");
        self.tx.subscribe()
    }

    pub fn unsubscribe(&self) {
        let n = self.viewers.fetch_sub(1, Ordering::SeqCst);
        tracing::debug!(
            viewers = n.saturating_sub(1),
            "frame bus subscriber removed"
        );
    }

    /// Whether a viewer is attached (skip encoding entirely otherwise).
    pub fn has_viewer(&self) -> bool {
        self.viewers.load(Ordering::SeqCst) > 0
    }

    /// Guard object: keeps the viewer count raised for as long as it is alive.
    ///
    /// The count must be raised from *inside* the response body stream, not in
    /// the axum handler: handlers are dropped as soon as the body is handed to
    /// hyper (the stream outlives them).
    pub fn attach(self: &Arc<Self>) -> ViewerGuard {
        self.viewers.fetch_add(1, Ordering::SeqCst);
        let n = self.viewers.load(Ordering::SeqCst);
        tracing::debug!(viewers = n, "frame bus subscriber added");
        ViewerGuard { bus: self.clone() }
    }

    pub fn last(&self) -> Option<FramePacket> {
        self.last.read().ok().and_then(|g| g.clone())
    }

    /// Rate-limited publish. Returns true when the frame was emitted.
    pub fn publish(
        &self,
        index: usize,
        t: f64,
        total: usize,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> bool {
        if !self.has_viewer() || !self.acquire_slot() {
            return false;
        }
        // Encode *outside* the rate-limit lock: two viewers joining at the same
        // instant must not serialize JPEG encoding on the pipeline thread.
        let jpeg = match encode_jpeg(rgb, width, height, self.jpeg_quality) {
            Ok(b) => b,
            Err(e) => {
                tracing::debug!(error = %e, "frame encode failed");
                return false;
            }
        };
        let packet = FramePacket {
            meta: FrameMeta { index, t, total },
            jpeg: Arc::new(jpeg),
        };
        if let Ok(mut slot) = self.last.write() {
            *slot = Some(packet.clone());
        }
        let _ = self.tx.send(packet);
        true
    }

    /// Take the frame-rate token; false means "skip this frame".
    fn acquire_slot(&self) -> bool {
        let mut last = match self.last_emit.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        if last.elapsed() < self.min_interval {
            return false;
        }
        *last = Instant::now();
        true
    }
}

/// Keeps the viewer count raised while an SSE body is alive.
pub struct ViewerGuard {
    bus: Arc<FrameBus>,
}

impl Drop for ViewerGuard {
    fn drop(&mut self) {
        let n = self.bus.viewers.fetch_sub(1, Ordering::SeqCst);
        tracing::debug!(
            viewers = n.saturating_sub(1),
            "frame bus subscriber removed"
        );
    }
}

fn encode_jpeg(rgb: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity((width * height / 4) as usize);
    let mut enc = JpegEncoder::new_with_quality(&mut out, quality);
    enc.encode(rgb, width, height, image::ExtendedColorType::Rgb8)
        .context("jpeg encode")?;
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Done | Self::Failed | Self::Cancelled)
    }
}

/// A finished (or in-flight) analysis job.
pub struct Job {
    pub id: String,
    pub input: PathBuf,
    pub input_name: String,
    /// Time window of `input` that is being analysed (`None` = whole file).
    pub clip: Option<crate::video_io::Clip>,
    /// Directory holding this job's artifacts (`out/web/<id>`), where the
    /// restart-recovery files (`job.json`, `tracking.json`, `report.json`)
    /// are written.
    pub dir: PathBuf,
    pub status: RwLock<JobStatus>,
    pub error: RwLock<Option<String>>,
    pub created_ms: u64,
    pub frames_done: RwLock<usize>,
    pub total_frames: RwLock<usize>,
    /// Tracking records, available as soon as the tracking pass finishes.
    pub run: RwLock<Option<Arc<RunInfo>>>,
    pub report: RwLock<Option<Arc<AnalyzeReport>>>,
    /// Windows scored so far, growing while the job runs (live curve/table).
    pub windows: RwLock<Vec<ScoredWindow>>,
    pub feature_names: RwLock<Vec<String>>,
    pub video: RwLock<Option<PathBuf>>,
    pub frames: Arc<FrameBus>,
    pub logs: Arc<RwLock<Vec<LogRecord>>>,
    pub cancel: Arc<AtomicBool>,
    /// Sent when the job reaches a terminal state.
    pub finished: watch::Sender<bool>,
    pub duration_s: RwLock<Option<f64>>,
}

impl Job {
    pub fn status(&self) -> JobStatus {
        *self.status.read().unwrap_or_else(|e| e.into_inner())
    }

    pub fn set_status(&self, s: JobStatus) {
        *self.status.write().unwrap_or_else(|e| e.into_inner()) = s;
        self.save_meta();
    }

    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    /// The part of a job that is persisted as `job.json`.
    pub fn meta(&self) -> JobMeta {
        let clip = self.clip;
        JobMeta {
            id: self.id.clone(),
            input: self.input.display().to_string(),
            input_name: self.input_name.clone(),
            start_s: clip.map(|c| c.start_s).unwrap_or(0.0),
            end_s: clip.and_then(|c| c.end_s),
            created_ms: self.created_ms,
            status: self.status(),
            error: self.error.read().unwrap_or_else(|e| e.into_inner()).clone(),
            duration_s: *self.duration_s.read().unwrap_or_else(|e| e.into_inner()),
            has_video: self.video.read().map(|v| v.is_some()).unwrap_or(false),
        }
    }

    /// Best-effort rewrite of `<job dir>/job.json`.
    ///
    /// Called from every status transition: losing this file only costs restart
    /// recovery for this one job, so failures are logged, not propagated.
    pub fn save_meta(&self) {
        if let Err(e) = write_json_atomic(&self.dir.join("job.json"), &self.meta()) {
            tracing::warn!(job = %self.id, error = %e, "cannot persist job metadata");
        }
    }
}

/// Persisted job header (`out/web/<id>/job.json`).
///
/// Holds exactly what cannot be recomputed from the artifacts on disk: which
/// media file was analysed, which clip, when, and how the job ended. The
/// per-frame records (`tracking.json`) and scored windows (`report.json`) live
/// in separate files because they are much larger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobMeta {
    pub id: String,
    pub input: String,
    pub input_name: String,
    #[serde(default)]
    pub start_s: f64,
    #[serde(default)]
    pub end_s: Option<f64>,
    pub created_ms: u64,
    pub status: JobStatus,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub duration_s: Option<f64>,
    #[serde(default)]
    pub has_video: bool,
}

/// Write JSON through a temp file, so a crash can never leave a truncated
/// `job.json` behind (the loader would silently skip that job).
fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let tmp = path.with_extension("json.tmp");
    {
        let f = std::fs::File::create(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        let mut w = std::io::BufWriter::new(f);
        serde_json::to_writer_pretty(&mut w, value).context("serialising job json")?;
        std::io::Write::flush(&mut w)?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("renaming {}", path.display()))?;
    Ok(())
}

/// Read a JSON artifact, treating a missing or corrupt file as `None`.
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    let bytes = std::fs::read(path).ok()?;
    match serde_json::from_slice(&bytes) {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "ignoring corrupt job artifact");
            None
        }
    }
}

/// Everything the HTTP layer needs to know about a job, in one payload.
#[derive(Serialize)]
pub struct JobView {
    pub id: String,
    pub input: String,
    pub input_name: String,
    /// Source-time offset of the first analysed frame (seconds).
    pub start_s: f64,
    /// Requested source-time end of the analysed window (seconds).
    pub end_s: Option<f64>,
    pub status: JobStatus,
    pub error: Option<String>,
    pub created_ms: u64,
    pub frames_done: usize,
    pub total_frames: usize,
    pub progress: f64,
    pub duration_s: Option<f64>,
    pub has_video: bool,
    pub summary: Option<SummaryView>,
}

#[derive(Serialize)]
pub struct SummaryView {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub frames: usize,
    /// Duration of the analysed clip.
    pub duration_s: f64,
    /// Source-time offset of the first analysed frame (seconds).
    pub start_s: f64,
    pub end_s: Option<f64>,
    pub source_duration_s: f64,
    pub n_tracks: usize,
    pub n_windows: usize,
    pub immobile_windows: usize,
    pub mobile_windows: usize,
    pub mean_immobile_prob: f32,
    /// Mean RII over all windows (published TST criterion).
    pub mean_rii: f32,
    pub per_track: Vec<TrackView>,
}

#[derive(Serialize)]
pub struct TrackView {
    pub track_id: usize,
    pub n_windows: usize,
    pub immobile_fraction: f32,
}

impl SummaryView {
    fn from_report(run: &RunInfo, report: &AnalyzeReport) -> Self {
        let mut s = Self::from_windows(run, &report.windows);
        s.n_tracks = report.n_tracks;
        s.mean_immobile_prob = report.mean_immobile_prob;
        s.per_track = report
            .per_track_immobile_fraction
            .iter()
            .map(|(id, frac)| TrackView {
                track_id: *id,
                n_windows: report.windows.iter().filter(|w| w.track_id == *id).count(),
                immobile_fraction: *frac,
            })
            .collect();
        s
    }

    /// Stats from the windows scored so far (live view).
    fn from_windows(run: &RunInfo, windows: &[ScoredWindow]) -> Self {
        use crate::classify::ActClass;
        let n_windows = windows.len();
        let mut by_track: std::collections::BTreeMap<usize, (usize, usize)> =
            std::collections::BTreeMap::new();
        for w in windows {
            let e = by_track.entry(w.track_id).or_insert((0, 0));
            e.1 += 1;
            if w.label == ActClass::Immobile {
                e.0 += 1;
            }
        }
        Self {
            width: run.width,
            height: run.height,
            fps: run.fps,
            frames: run.frames,
            duration_s: run.duration_s,
            start_s: run.start_s,
            end_s: run.end_s,
            source_duration_s: run.source_duration_s,
            n_tracks: by_track.len(),
            n_windows,
            immobile_windows: windows
                .iter()
                .filter(|w| w.label == ActClass::Immobile)
                .count(),
            mobile_windows: windows
                .iter()
                .filter(|w| w.label == ActClass::Mobile)
                .count(),
            mean_immobile_prob: if n_windows == 0 {
                0.0
            } else {
                windows.iter().map(|w| w.immobile_prob).sum::<f32>() / n_windows as f32
            },
            mean_rii: if n_windows == 0 {
                0.0
            } else {
                windows.iter().map(|w| w.rii).sum::<f32>() / n_windows as f32
            },
            per_track: by_track
                .into_iter()
                .map(|(id, (imm, total))| TrackView {
                    track_id: id,
                    n_windows: total,
                    immobile_fraction: if total > 0 {
                        imm as f32 / total as f32
                    } else {
                        0.0
                    },
                })
                .collect(),
        }
    }
}

impl Job {
    pub fn view(&self) -> JobView {
        let frames_done = *self.frames_done.read().unwrap_or_else(|e| e.into_inner());
        let total = *self.total_frames.read().unwrap_or_else(|e| e.into_inner());
        let report = self
            .report
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let run = self.run.read().unwrap_or_else(|e| e.into_inner()).clone();
        let live_windows = self
            .windows
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        // Live stats: the summary is derived from whatever has been scored so
        // far, so the header tiles update while the job is still running and
        // are identical to the final report once `report` is set.
        let summary = match (run, report) {
            (Some(run), Some(report)) => Some(SummaryView::from_report(&run, &report)),
            (Some(run), None) if !live_windows.is_empty() => {
                Some(SummaryView::from_windows(&run, &live_windows))
            }
            _ => None,
        };
        let progress = if total == 0 {
            if self.status() == JobStatus::Done {
                1.0
            } else {
                0.0
            }
        } else {
            (frames_done as f64 / total as f64).clamp(0.0, 1.0)
        };
        JobView {
            id: self.id.clone(),
            input: self.input.display().to_string(),
            input_name: self.input_name.clone(),
            start_s: self.clip.map(|c| c.start_s).unwrap_or(0.0),
            end_s: self.clip.and_then(|c| c.end_s),
            status: self.status(),
            error: self.error.read().unwrap_or_else(|e| e.into_inner()).clone(),
            created_ms: self.created_ms,
            frames_done,
            total_frames: total,
            progress,
            duration_s: *self.duration_s.read().unwrap_or_else(|e| e.into_inner()),
            has_video: self.video.read().map(|v| v.is_some()).unwrap_or(false),
            summary,
        }
    }
}

/// Registry of all jobs started since boot.
pub struct JobManager {
    jobs: RwLock<HashMap<String, Arc<Job>>>,
    order: RwLock<Vec<String>>,
    out_dir: PathBuf,
    media_dir: PathBuf,
}

impl JobManager {
    pub fn new(out_dir: PathBuf, media_dir: PathBuf) -> Self {
        Self {
            jobs: RwLock::new(HashMap::new()),
            order: RwLock::new(Vec::new()),
            out_dir,
            media_dir,
        }
    }

    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }

    pub fn media_dir(&self) -> &Path {
        &self.media_dir
    }

    pub fn get(&self, id: &str) -> Option<Arc<Job>> {
        self.jobs.read().ok().and_then(|m| m.get(id).cloned())
    }

    pub fn list(&self) -> Vec<Arc<Job>> {
        let jobs = self.jobs.read().unwrap_or_else(|e| e.into_inner());
        self.order
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter_map(|id| jobs.get(id).cloned())
            .collect()
    }

    pub fn insert(&self, job: Arc<Job>) {
        if let Ok(mut m) = self.jobs.write() {
            m.insert(job.id.clone(), job.clone());
        }
        if let Ok(mut o) = self.order.write() {
            o.insert(0, job.id.clone());
        }
    }

    /// Reserve the output directory of a job.
    pub fn job_dir(&self, id: &str) -> PathBuf {
        self.out_dir.join("web").join(id)
    }

    /// Rebuild the in-memory registry from `out_dir/web/*/job.json`.
    ///
    /// Called once at startup: jobs are in-memory objects, so without this the
    /// whole history disappears whenever the process restarts. Jobs that were
    /// still queued/running at shutdown cannot be resumed and are marked
    /// `failed`; finished jobs are reopened from `tracking.json` +
    /// `report.json` so their records, curve and summary are available again.
    ///
    /// Directories without a `job.json` (written by older builds) are left
    /// alone. Returns the number of restored jobs.
    pub fn load_existing(&self) -> usize {
        let root = self.out_dir.join("web");
        let Ok(entries) = std::fs::read_dir(&root) else {
            return 0;
        };
        let mut found: Vec<(PathBuf, JobMeta)> = Vec::new();
        for entry in entries.flatten() {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            match read_json::<JobMeta>(&dir.join("job.json")) {
                Some(meta) => found.push((dir, meta)),
                None => continue,
            }
        }
        // Oldest first: `insert` pushes onto the front, so the newest job ends
        // up first exactly like a freshly created one.
        found.sort_by_key(|(_, meta)| meta.created_ms);
        let total = found.len();
        let mut interrupted = 0;
        for (dir, meta) in found {
            let job = Arc::new(Self::restore(&dir, meta));
            if !job.status().is_terminal() {
                interrupted += 1;
                if let Ok(mut g) = job.error.write() {
                    *g = Some("interrupted by server restart".to_string());
                }
                job.set_status(JobStatus::Failed);
            }
            self.insert(job);
        }
        if total > 0 {
            tracing::info!(
                restored = total,
                interrupted,
                path = %root.display(),
                "restored analysis jobs from disk"
            );
        }
        total
    }

    /// Reopen one job directory. No worker is started: the job is already
    /// finished (or was interrupted, in which case it is marked failed by the
    /// caller).
    fn restore(dir: &Path, meta: JobMeta) -> Job {
        let run = read_json::<RunInfo>(&dir.join("tracking.json")).map(Arc::new);
        let report = read_json::<AnalyzeReport>(&dir.join("report.json")).map(Arc::new);
        let windows = report
            .as_ref()
            .map(|r| r.windows.clone())
            .unwrap_or_default();
        let frames = run.as_ref().map(|r| r.frames).unwrap_or(0);
        let done = report.is_some();
        // Rebuild the clip the same way `create_job` would have.
        let clip = if meta.start_s > 1e-3 || meta.end_s.is_some() {
            Some(crate::video_io::Clip::new(meta.start_s, meta.end_s))
        } else {
            None
        };
        let video_path = dir.join("annotated.mp4");
        let video = (meta.has_video && video_path.is_file()).then_some(video_path);
        let (finished_tx, _) = watch::channel(meta.status.is_terminal());
        Job {
            id: meta.id,
            input: PathBuf::from(meta.input),
            input_name: meta.input_name,
            clip,
            dir: dir.to_path_buf(),
            status: RwLock::new(meta.status),
            error: RwLock::new(meta.error),
            created_ms: meta.created_ms,
            frames_done: RwLock::new(if done { frames } else { 0 }),
            total_frames: RwLock::new(frames),
            run: RwLock::new(run),
            report: RwLock::new(report),
            windows: RwLock::new(windows),
            feature_names: RwLock::new(
                crate::features::FEATURE_NAMES
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
            video: RwLock::new(video),
            frames: Arc::new(FrameBus::new(15.0, 72)),
            logs: Arc::new(RwLock::new(Vec::new())),
            cancel: Arc::new(AtomicBool::new(false)),
            finished: finished_tx,
            duration_s: RwLock::new(meta.duration_s),
        }
    }
}

/// Snapshot of what the pipeline produced, shared with the browser.
pub struct CompletedRun {
    pub run: Arc<RunInfo>,
    pub report: Arc<AnalyzeReport>,
    pub video: Option<PathBuf>,
}

/// Append-only `windows.csv` writer used by the live job.
///
/// The header is written once at job start, then one line per scored window as
/// soon as the window completes - so `out/web/<job>/windows.csv` is readable
/// (and tail-able) during the run, not just at the end.
struct LiveCsv {
    path: PathBuf,
    file: Option<std::io::BufWriter<std::fs::File>>,
}

impl LiveCsv {
    fn create(path: &Path) -> Result<Self> {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).ok();
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        {
            use std::io::Write as _;
            write!(
                f,
                "track_id,start_frame,end_frame,start_t,end_t,immobile_prob,label,rii"
            )?;
            for n in crate::features::FEATURE_NAMES {
                write!(f, ",{n}")?;
            }
            writeln!(f)?;
            f.flush()?;
        }
        Ok(Self {
            path: path.to_path_buf(),
            file: Some(f),
        })
    }

    fn append(&mut self, windows: &[ScoredWindow]) {
        let Some(f) = self.file.as_mut() else {
            return;
        };
        if pipeline::write_window_rows(f, windows).is_err() {
            return;
        }
        // Flush per batch (not per row): the UI polls the file rarely, and a
        // flush per window would be measurable at 0.5 s windows.
        let _ = std::io::Write::flush(f);
    }

    fn finish(&mut self) {
        if let Some(mut f) = self.file.take() {
            let _ = std::io::Write::flush(&mut f);
        }
    }

    /// Replace the file with the authoritative batch report.
    fn rewrite(&mut self, report: &AnalyzeReport) -> Result<()> {
        self.finish();
        pipeline::write_window_csv(&self.path, &report.feature_names, &report.windows)
    }
}

/// Blocking worker: runs the pipeline and scoring for one job.
///
/// Called from `tokio::task::spawn_blocking`; every channel send is
/// non-blocking, and cancellation is polled once per frame.
pub fn execute(
    job: Arc<Job>,
    cfg: PipelineConfig,
    analyze_cfg: pipeline::AnalyzeConfig,
    csv_path: PathBuf,
) -> Result<CompletedRun> {
    let started = Instant::now();
    job.set_status(JobStatus::Running);
    let frames = job.frames.clone();
    let cancel = job.cancel_flag();
    let progress_job = job.clone();
    // Windows scored *during* tracking are appended here, so the UI can show
    // the curve/table live instead of waiting for the whole video.
    let csv_sink = LiveCsv::create(&csv_path)?;
    if let Ok(mut g) = job.feature_names.write() {
        *g = crate::features::FEATURE_NAMES
            .iter()
            .map(|s| s.to_string())
            .collect();
    }

    struct WebHook {
        frames: Arc<FrameBus>,
        job: Arc<Job>,
        cancel: Arc<AtomicBool>,
        csv: std::sync::Mutex<LiveCsv>,
    }

    impl pipeline::FrameHook for WebHook {
        fn on_frame(&mut self, frame: &pipeline::FrameView<'_>) {
            if let Some(total) = frame.total_frames
                && let Ok(mut g) = self.job.total_frames.write()
            {
                *g = total;
            }
            if let Ok(mut g) = self.job.frames_done.write() {
                *g = frame.index + 1;
            }
            if let Some(rgb) = frame.overlay {
                self.frames.publish(
                    frame.index,
                    frame.t,
                    frame.total_frames.unwrap_or(0),
                    rgb,
                    frame.width,
                    frame.height,
                );
            }
        }

        fn on_windows(&mut self, windows: &[ScoredWindow]) {
            if !windows.is_empty()
                && let Ok(mut csv) = self.csv.lock()
            {
                csv.append(windows);
            }
            if let Ok(mut g) = self.job.windows.write() {
                g.extend_from_slice(windows);
            }
        }

        fn cancelled(&self) -> bool {
            self.cancel.load(Ordering::Relaxed)
        }
    }

    let mut hook = WebHook {
        frames: frames.clone(),
        job: progress_job,
        cancel,
        csv: std::sync::Mutex::new(csv_sink),
    };

    let mut run_cfg = cfg.clone();
    run_cfg.live_scoring = Some(analyze_cfg.clone());
    // Persist the tracking result next to the other artifacts: the in-memory
    // `RunInfo` is what `/api/jobs/{id}/records` serves, and it has to survive
    // a restart too.
    run_cfg.output_json = Some(job.dir.join("tracking.json"));
    let run = pipeline::run_with_hook(&run_cfg, Some(&mut hook)).context("tracking pipeline")?;
    if job.cancel.load(Ordering::Relaxed) {
        job.set_status(JobStatus::Cancelled);
        let _ = job.finished.send(true);
        // Cancellation is a normal outcome, not a failure.
        return Err(anyhow::Error::new(crate::web::error::Cancelled));
    }

    job.set_status(JobStatus::Running);
    // `run` breaks out of the loop on `max_frames`, so the live windows are
    // already complete: recomputing would only duplicate work. The batch
    // report is still built from the same records to keep one code path for
    // the summary (means, per-track fractions).
    let report = pipeline::score_tracks(&run, &analyze_cfg).context("window scoring")?;
    if let Ok(mut g) = job.windows.write()
        && (g.len() != report.windows.len()
            || g.iter()
                .zip(&report.windows)
                .any(|(a, b)| a.start_frame != b.start_frame || a.track_id != b.track_id))
    {
        // Defensive: fall back to the authoritative batch result.
        *g = report.windows.clone();
        if let Ok(mut csv) = hook.csv.lock() {
            csv.rewrite(&report)?;
        }
    }
    hook.csv.lock().map(|mut c| c.finish()).ok();

    let video = cfg
        .output_video
        .clone()
        .filter(|p| cfg.write_video && p.exists());

    let duration = started.elapsed().as_secs_f64();
    if let Ok(mut g) = job.duration_s.write() {
        *g = Some(duration);
    }
    let run = Arc::new(run);
    let report = Arc::new(report);
    if let Ok(mut g) = job.run.write() {
        *g = Some(run.clone());
    }
    if let Ok(mut g) = job.report.write() {
        *g = Some(report.clone());
    }
    if let Ok(mut g) = job.video.write() {
        *g = video.clone();
    }
    // Authoritative scored windows, so a restarted server can rebuild the
    // curve/table without re-running the classifier.
    if let Err(e) = write_json_atomic(&job.dir.join("report.json"), &*report) {
        tracing::warn!(job = %job.id, error = %e, "cannot persist window report");
    }
    job.set_status(JobStatus::Done);
    let _ = job.finished.send(true);
    tracing::info!(
        job = %job.id,
        frames = run.frames,
        windows = report.windows.len(),
        elapsed_s = duration,
        "analysis job finished"
    );
    Ok(CompletedRun { run, report, video })
}

/// Fail a job and notify subscribers.
pub fn fail(job: &Job, err: &anyhow::Error) {
    let msg = format!("{err:#}");
    // A user cancellation already set the status; do not log or flag it as a
    // pipeline failure.
    if job.status() == JobStatus::Cancelled {
        let _ = job.finished.send(true);
        return;
    }
    tracing::error!(job = %job.id, error = %msg, "analysis job failed");
    if let Ok(mut g) = job.error.write() {
        *g = Some(msg);
    }
    job.set_status(JobStatus::Failed);
    let _ = job.finished.send(true);
}
