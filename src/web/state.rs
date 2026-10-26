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
use serde::Serialize;
use tokio::sync::{broadcast, watch};

use crate::pipeline::{self, AnalyzeReport, PipelineConfig, RunInfo};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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
    pub status: RwLock<JobStatus>,
    pub error: RwLock<Option<String>>,
    pub created_ms: u64,
    pub frames_done: RwLock<usize>,
    pub total_frames: RwLock<usize>,
    /// Tracking records, available as soon as the tracking pass finishes.
    pub run: RwLock<Option<Arc<RunInfo>>>,
    pub report: RwLock<Option<Arc<AnalyzeReport>>>,
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
    }

    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }
}

/// Everything the HTTP layer needs to know about a job, in one payload.
#[derive(Serialize)]
pub struct JobView {
    pub id: String,
    pub input: String,
    pub input_name: String,
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
    pub duration_s: f64,
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
        let summary = match (run, report) {
            (Some(run), Some(report)) => Some(SummaryView {
                width: run.width,
                height: run.height,
                fps: run.fps,
                frames: run.frames,
                duration_s: run.duration_s,
                n_tracks: report.n_tracks,
                n_windows: report.windows.len(),
                immobile_windows: report
                    .windows
                    .iter()
                    .filter(|w| w.label == crate::classify::ActClass::Immobile)
                    .count(),
                mobile_windows: report
                    .windows
                    .iter()
                    .filter(|w| w.label == crate::classify::ActClass::Mobile)
                    .count(),
                mean_immobile_prob: report.mean_immobile_prob,
                mean_rii: if report.windows.is_empty() {
                    0.0
                } else {
                    report.windows.iter().map(|w| w.rii).sum::<f32>() / report.windows.len() as f32
                },
                per_track: report
                    .per_track_immobile_fraction
                    .iter()
                    .map(|(id, frac)| TrackView {
                        track_id: *id,
                        n_windows: report.windows.iter().filter(|w| w.track_id == *id).count(),
                        immobile_fraction: *frac,
                    })
                    .collect(),
            }),
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
}

/// Snapshot of what the pipeline produced, shared with the browser.
pub struct CompletedRun {
    pub run: Arc<RunInfo>,
    pub report: Arc<AnalyzeReport>,
    pub video: Option<PathBuf>,
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

    struct WebHook {
        frames: Arc<FrameBus>,
        job: Arc<Job>,
        cancel: Arc<AtomicBool>,
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

        fn cancelled(&self) -> bool {
            self.cancel.load(Ordering::Relaxed)
        }
    }

    let mut hook = WebHook {
        frames: frames.clone(),
        job: progress_job,
        cancel,
    };

    let run = pipeline::run_with_hook(&cfg, Some(&mut hook)).context("tracking pipeline")?;
    if job.cancel.load(Ordering::Relaxed) {
        job.set_status(JobStatus::Cancelled);
        let _ = job.finished.send(true);
        // Cancellation is a normal outcome, not a failure.
        return Err(anyhow::Error::new(crate::web::error::Cancelled));
    }

    job.set_status(JobStatus::Running);
    let report = pipeline::score_tracks(&run, &analyze_cfg).context("window scoring")?;
    pipeline::write_window_csv(&csv_path, &report.feature_names, &report.windows).ok();

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
