//! End-to-end pipeline: decode -> motion trigger -> YOLO (on ROIs) -> track ->
//! annotate -> encode.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::common::{Bbox, Detection};
use crate::detect::blob::BlobDetector;
use crate::detect::device::{ActiveDevice, Device};
use crate::detect::motion::MotionTrigger;
use crate::detect::{COCO_NAMES, YoloDetector};
use crate::draw::{self, DrawTrack, FrameOverlay};
use crate::preprocess::LetterboxCfg;
use crate::track::{MultiTracker, TrackerConfig};
use crate::video_io::{Clip, VideoReader};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum DetectorKind {
    /// Background-subtraction connected components (fast, needs a clean floor).
    Blob,
    /// YOLO11 ONNX model (needed for cluttered scenes / fine-tuned models).
    Yolo,
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub input: PathBuf,
    /// Decode only this time window of `input` (no intermediate file is
    /// written; frame indices and timestamps restart at the clip start).
    pub clip: Option<Clip>,
    pub output_video: Option<PathBuf>,
    pub output_json: Option<PathBuf>,
    pub output_csv: Option<PathBuf>,
    pub detector: DetectorKind,
    pub model: PathBuf,
    pub max_side: u32,
    pub score_thr: f32,
    pub iou_thr: f32,
    /// Blob detector: gray difference threshold.
    pub blob_threshold: u8,
    /// Blob detector: minimum object area as a fraction of the frame.
    pub blob_min_area: f32,
    /// Blob detector: reference for the *empty enclosure* (animals removed),
    /// either a frame index inside the video or a still image.
    /// Strongly recommended for TST - it makes every animal visible from the
    /// very first frame instead of relying on motion.
    pub reference: Option<Reference>,
    /// Class name filter, e.g. `mouse`. `any` keeps every class.
    pub class_name: String,
    /// If the requested class never fires, fall back to generic animal classes
    /// (cat/dog/bear...) - useful with a stock COCO model.
    pub fallback_animal: bool,
    pub detect_every: usize,
    /// Re-run detection at least every N frames even when nothing moves.
    pub refresh_every: usize,
    pub tracker: TrackerConfig,
    pub write_video: bool,
    pub show_rois: bool,
    pub show_trails: bool,
    pub threads: usize,
    /// Compute device for the YOLO backend (`cpu` / `cuda` / `auto`).
    /// Ignored by the blob backend.
    pub device: Device,
    /// CUDA device ordinal used when `device` resolves to CUDA.
    pub gpu_device_id: i32,
    pub max_frames: Option<usize>,
    pub preview_every: Option<usize>,
    /// Score 0.5 s windows *while* tracking and hand them to the hook, so the
    /// web UI can draw the curve live. `None` (CLI) leaves scoring to the
    /// caller after the run.
    pub live_scoring: Option<AnalyzeConfig>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            input: PathBuf::new(),
            clip: None,
            output_video: None,
            output_json: None,
            output_csv: None,
            detector: DetectorKind::Blob,
            model: PathBuf::from("models/yolo11n.onnx"),
            max_side: 640,
            score_thr: 0.20,
            iou_thr: 0.45,
            blob_threshold: 22,
            blob_min_area: 0.0015,
            reference: None,
            class_name: "mouse".to_string(),
            fallback_animal: true,
            detect_every: 1,
            refresh_every: 90,
            tracker: TrackerConfig::default(),
            write_video: true,
            show_rois: false,
            show_trails: true,
            threads: 2,
            device: Device::Auto,
            gpu_device_id: 0,
            max_frames: None,
            preview_every: None,
            live_scoring: None,
        }
    }
}

/// Per-track summary written to JSON, ready for feature extraction.
#[derive(Debug, Serialize, Deserialize)]
pub struct TrackSummary {
    pub id: usize,
    pub class_id: usize,
    pub start_frame: usize,
    pub end_frame: usize,
    pub n_points: usize,
    pub n_measured: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DetectionRecord {
    pub frame: usize,
    pub track_id: usize,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub measured: bool,
    /// Fraction of foreground cells inside the box (pixel-level activity).
    /// Only available with the blob backend; 0 with YOLO.
    #[serde(default)]
    pub motion_energy: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RunInfo {
    pub input: String,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub frames: usize,
    /// Duration of the analysed clip (== `frames / fps`).
    pub duration_s: f64,
    /// Source-time offset of the first analysed frame. Frame indices, window
    /// times and `duration_s` are all relative to this offset, so a UI playing
    /// the original video must add it when seeking.
    #[serde(default)]
    pub start_s: f64,
    /// Requested end of the analysed clip (source seconds), if any.
    #[serde(default)]
    pub end_s: Option<f64>,
    /// Duration of the source file, ignoring the clip.
    #[serde(default)]
    pub source_duration_s: f64,
    pub detect_calls: usize,
    pub detections: usize,
    /// Execution provider the detector ran on (`cpu`, `cuda:0`).
    /// Always `cpu` for the blob backend.
    #[serde(default)]
    pub device: String,
    pub tracks: Vec<TrackSummary>,
    pub records: Vec<DetectionRecord>,
}

pub struct PipelineStats {
    pub frames: usize,
    pub detect_calls: usize,
    pub detections: usize,
    pub elapsed_s: f64,
}

/// One decoded (and optionally annotated) frame handed to a [`FrameHook`].
pub struct FrameView<'a> {
    pub index: usize,
    /// Timestamp in seconds.
    pub t: f64,
    pub fps: f64,
    pub width: u32,
    pub height: u32,
    /// Annotated RGB buffer when the caller asked for annotation.
    pub overlay: Option<&'a [u8]>,
    /// Decoded frame count (unknown for some containers).
    pub total_frames: Option<usize>,
}

/// Optional observer called once per decoded frame.
///
/// The callback runs on the pipeline thread, so it must be cheap: hand the data
/// to a channel or a worker rather than blocking here.
pub trait FrameHook: Send {
    fn on_frame(&mut self, frame: &FrameView<'_>);

    /// Called when one or more windows finished scoring during the same frame
    /// (only when [`PipelineConfig::live_scoring`] is set).
    fn on_windows(&mut self, _windows: &[ScoredWindow]) {}

    /// Return `true` to stop the pipeline after the current frame
    /// (used by the web UI's cancel button). Defaults to never stopping.
    fn cancelled(&self) -> bool {
        false
    }
}

/// Run the whole pipeline and report what happened.
pub fn run(cfg: &PipelineConfig) -> Result<RunInfo> {
    run_with_hook(cfg, None)
}

/// [`run`] variant used by the web server to stream frames to the browser.
pub fn run_with_hook(
    cfg: &PipelineConfig,
    mut hook: Option<&mut dyn FrameHook>,
) -> Result<RunInfo> {
    let started = Instant::now();
    let mut reader = VideoReader::open_clipped(&cfg.input, cfg.max_side, cfg.clip)?;
    let (w, h) = (reader.width, reader.height);
    let fps = reader.fps;
    let clip_start_s = reader.start_s;
    let source_duration_s = reader.source_duration_s;
    info!(
        width = w,
        height = h,
        fps,
        start_s = clip_start_s,
        duration_s = reader.duration_s,
        "decoding input"
    );

    // --- detector backend -------------------------------------------------
    let mut yolo = if cfg.detector == DetectorKind::Yolo {
        let mut d = YoloDetector::load_with_device(
            &cfg.model,
            cfg.score_thr,
            cfg.iou_thr,
            100,
            cfg.threads,
            cfg.device,
            cfg.gpu_device_id,
        )?;
        if d.class_names.is_empty() {
            d.class_names = COCO_NAMES.iter().map(|s| s.to_string()).collect();
        }
        let want = d.resolve_class(&cfg.class_name).with_context(|| {
            format!(
                "class `{}` not found in model (available: {}). \
                 Add --class-name any to accept every class.",
                cfg.class_name,
                d.class_names.join(", ")
            )
        })?;
        info!(
            backend = "yolo",
            classes = d.declared_classes(),
            class = %cfg.class_name,
            class_id = want,
            device = %d.device(),
            "detector ready"
        );
        Some((d, want))
    } else {
        info!(
            backend = "blob",
            threshold = cfg.blob_threshold,
            "detector ready"
        );
        None
    };
    // `usize::MAX` = accept every class (blob backend and `--class-name any`).
    let mut want_class = if cfg.detector == DetectorKind::Blob {
        usize::MAX
    } else {
        yolo.as_ref().map(|(_, c)| *c).unwrap_or(usize::MAX)
    };

    // Execution provider actually in use; the blob backend never touches ORT.
    let active_device = yolo
        .as_ref()
        .map(|(d, _)| d.device())
        .unwrap_or(ActiveDevice::Cpu);
    if active_device == ActiveDevice::Cpu && cfg.device != Device::Cpu {
        info!(
            requested = %cfg.device,
            "detector running on CPU (no usable GPU execution provider)"
        );
    }

    // The network input size comes from the ONNX graph (Ultralytics exports a
    // static shape = the `imgsz` used at export time), falling back to 640.
    let (net_w, net_h) = yolo.as_ref().map(|(d, _)| d.net_wh()).unwrap_or((640, 640));
    let lb = LetterboxCfg::new(w, h, net_w, net_h);
    if cfg.detector == DetectorKind::Yolo {
        info!(net_w, net_h, "network input size");
    }

    let mut blob: Option<BlobDetector> = if cfg.detector == DetectorKind::Blob {
        let mut b = BlobDetector::new(w, h);
        b.diff_threshold = cfg.blob_threshold;
        b.min_area_ratio = cfg.blob_min_area;
        Some(b)
    } else {
        None
    };
    // A reference frame replaces the temporal background, so load it *before*
    // consumption starts (VideoReader supports skipping from the current
    // position, so we reopen the decoder for this one-off seek).
    if let (Some(b), Some(reference)) = (blob.as_mut(), cfg.reference.as_ref()) {
        match reference {
            Reference::Frame(idx) => {
                // `idx` is relative to the analysed clip's first frame.
                let mut ref_reader = VideoReader::open_clipped(&cfg.input, cfg.max_side, cfg.clip)?;
                if let Some(f) = ref_reader.skip_to(*idx)? {
                    b.set_reference(&f.data);
                    info!(frame = idx, "using empty-enclosure reference frame");
                } else {
                    warn!(
                        frame = idx,
                        "reference frame beyond end of video; falling back to temporal background"
                    );
                }
            }
            Reference::Image(path) => {
                let img = image::open(path)
                    .with_context(|| format!("reading reference image {}", path.display()))?;
                // Match VideoReader's geometry: longest side = max_side,
                // dimensions rounded to even numbers.
                let (cw, ch) = fit_even(w, h);
                let resized = img
                    .resize_exact(cw, ch, image::imageops::FilterType::Triangle)
                    .to_rgb8();
                b.set_reference(resized.as_raw());
                info!(path = %path.display(), "using reference image");
            }
        }
    }

    let mut motion = MotionTrigger::new(w, h);
    let mut tracker = MultiTracker::new(cfg.tracker.clone());

    let mut writer = if cfg.write_video {
        let path = cfg
            .output_video
            .clone()
            .unwrap_or_else(|| PathBuf::from("out/tracked.mp4"));
        Some(crate::video_io::VideoWriter::create(&path, w, h, fps)?)
    } else {
        None
    };

    let pb = ProgressBar::new(reader.n_frames_est.max(1) as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({per_sec}) {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    let mut all_records = Vec::new();
    let mut detect_calls = 0usize;
    let mut n_detections = 0usize;
    let mut frame_count = 0usize;
    let mut summary: HashMap<usize, TrackSummary> = HashMap::new();
    // Hook consumers want a percentage, which needs the decoded frame total:
    // the smallest of the requested cap and the container's frame count.
    let total_frames = match (cfg.max_frames, reader.n_frames_est) {
        (Some(cap), est) if est > 0 => Some(cap.min(est)),
        (Some(cap), _) => Some(cap),
        (None, est) if est > 0 => Some(est),
        _ => None,
    };

    // Keep only tracks of the requested class in the output.
    let mut first_refresh_done = false;
    // Live window scoring for observers (web UI): fills the trailing records
    // of `all_records` that this frame produced, so the incremental result
    // matches the batch report exactly.
    let mut live = match (cfg.live_scoring.as_ref(), hook.as_ref()) {
        (Some(analyze_cfg), Some(_)) => Some(LiveScorer::new(analyze_cfg, w, h, fps)?),
        _ => None,
    };

    while let Some(mut frame) = reader.next_frame()? {
        if let Some(max) = cfg.max_frames
            && frame.index >= max
        {
            break;
        }
        if hook.as_ref().is_some_and(|h| h.cancelled()) {
            info!(frame = frame.index, "pipeline cancelled by caller");
            break;
        }
        frame_count += 1;
        pb.inc(1);

        // --- 1. motion gate ------------------------------------------------
        let moving = motion.update(&frame.data);
        let force_refresh = frame_count % cfg.refresh_every.max(1) == 0;
        let periodic = frame_count % cfg.detect_every.max(1) == 0;
        // NOTE: gating on `moving` is a *performance* optimisation for busy
        // scenes, but it is wrong for tail-suspension: the animal is still for
        // most of the trial, and "still" is exactly what we must measure. A
        // motion gate would then skip almost every frame and the tracker would
        // only see a handful of detections, producing short, fragmented tracks.
        //
        // So: always run the detector when the motion gate is disabled
        // (`detect_every = 1`), and only use the gate when the caller
        // explicitly asked for subsampling.
        let gated = cfg.detect_every > 1;
        let should_detect = if gated {
            (moving && periodic) || force_refresh
        } else {
            true
        };

        // --- 2. ROI proposal + detection -----------------------------------
        let rois: Vec<Bbox> = if moving {
            motion.candidates().into_iter().map(|d| d.bbox).collect()
        } else {
            Vec::new()
        };

        let mut detections: Vec<Detection> = Vec::new();
        let mut full_dets: Vec<Detection> = Vec::new();
        let keep = |cls: usize| accept_class(want_class, cls);

        if let Some(bd) = blob.as_mut() {
            // Blob backend: every frame is cheap, so there is no gating.
            bd.update(&frame.data);
            detect_calls += 1;
            detections = bd.detections();
            n_detections += detections.len();
        } else if should_detect {
            detect_calls += 1;
            let Some((detector, _)) = yolo.as_mut() else {
                unreachable!("yolo backend without a detector")
            };
            if moving && !rois.is_empty() && cfg.defer_features_only_on_roi() {
                detections = detect_in_rois(detector, &frame.data, w, h, &lb, &rois, want_class)?;
            } else {
                // full-frame detection: exact, used for periodic refresh
                full_dets = detector.detect(&frame.data, &lb)?;
            }
            detections.extend(full_dets.drain(..).filter(|d| keep(d.class_id)));
            n_detections += detections.len();

            // Generic COCO models rarely fire `mouse` on lab footage; if the
            // first full-frame pass finds nothing, widen to animal classes.
            let is_full_frame = !(moving && !rois.is_empty() && cfg.defer_features_only_on_roi());
            if is_full_frame && !first_refresh_done {
                first_refresh_done = true;
                if detections.is_empty()
                    && cfg.fallback_animal
                    && cfg.class_name.eq_ignore_ascii_case("mouse")
                {
                    warn!(
                        "class `mouse` produced no detections with the stock COCO model; \
                         falling back to generic animal classes (cat/dog/bear/...) - \
                         fine-tune a mouse detector for production use"
                    );
                    want_class = usize::MAX;
                }
            }
        }
        let keep = |cls: usize| accept_class(want_class, cls);

        // --- 3. tracking ---------------------------------------------------
        let tracks = tracker.update(&detections);

        if let Some(peek) = cfg.preview_every {
            if frame_count % peek == 0 {
                draw::print_ascii_preview(&frame.data, w, h, 64)?;
            }
        }

        // --- 4. annotate ----------------------------------------------------
        if writer.is_some() || cfg.preview_every.is_some() {
            let draws: Vec<DrawTrack> = tracks
                .iter()
                .filter(|t| keep(t.class_id))
                .map(|t| DrawTrack {
                    track: t,
                    color: draw::id_color(t.id),
                    label: format!("ID {} {:.2}", t.id, t.last_score),
                    trail_len: 120,
                })
                .collect();
            let overlay = FrameOverlay {
                frame_index: frame.index,
                timestamp_s: frame.index as f64 / fps,
                fps,
                rois: &rois,
                show_rois: cfg.show_rois,
                show_trails: cfg.show_trails,
                show_skeleton_axes: false,
            };
            let empty = HashMap::new();
            draw::draw_frame(&mut frame.data, w, h, &draws, &overlay, &empty);
        }

        if let Some(wtr) = writer.as_mut() {
            wtr.write_frame(&frame.data)?;
        }

        // --- 4a. live window scoring (web UI) -------------------------------
        // Runs before the hook callback so the freshly appended records are
        // exactly the ones `LiveScorer` consumes.
        if let (Some(scorer), Some(obs)) = (live.as_mut(), hook.as_mut()) {
            let tracks: Vec<_> = tracker
                .tracks
                .iter()
                .filter(|t| t.is_confirmed() && keep(t.class_id))
                .cloned()
                .collect();
            let mut energy: HashMap<usize, f32> = HashMap::new();
            if let Some(bd) = blob.as_ref() {
                for t in &tracks {
                    energy.insert(t.id, mask_ratio_in(bd, &t.bbox(), w, h));
                }
            }
            let fresh = scorer.push(frame.index, &tracks, &energy);
            if !fresh.is_empty() {
                obs.on_windows(&fresh);
            }
        }

        // --- 4b. stream the annotated frame to observers (web UI) -----------
        if let Some(obs) = hook.as_mut() {
            obs.on_frame(&FrameView {
                index: frame.index,
                t: frame.index as f64 / fps,
                fps,
                width: w,
                height: h,
                overlay: Some(&frame.data),
                total_frames,
            });
        }

        // --- 5. bookkeeping -------------------------------------------------
        for t in &tracks {
            if !keep(t.class_id) {
                continue;
            }
            let entry = summary.entry(t.id).or_insert_with(|| TrackSummary {
                id: t.id,
                class_id: t.class_id,
                start_frame: frame.index,
                end_frame: frame.index,
                n_points: 0,
                n_measured: 0,
            });
            entry.end_frame = frame.index;
            entry.n_points += 1;
            if t.time_since_update == 0 {
                entry.n_measured += 1;
            }
            if let Some(p) = t.history.back() {
                let b = p.bbox;
                all_records.push(DetectionRecord {
                    frame: frame.index,
                    track_id: t.id,
                    x1: b.x1,
                    y1: b.y1,
                    x2: b.x2,
                    y2: b.y2,
                    score: p.score,
                    measured: p.measured,
                    motion_energy: blob
                        .as_ref()
                        .map(|bd| mask_ratio_in(bd, &b, w, h))
                        .unwrap_or(0.0),
                });
            }
        }
    }

    pb.finish_with_message("done");

    // Score the trailing windows that only became complete at the end.
    if let (Some(mut scorer), Some(obs)) = (live.take(), hook.as_mut()) {
        let tail = scorer.flush();
        if !tail.is_empty() {
            obs.on_windows(&tail);
        }
    }

    if let Some(wtr) = writer {
        let path = wtr.path.clone();
        wtr.finish()?;
        info!("wrote {}", path.display());
    }

    let info_out = RunInfo {
        input: cfg.input.display().to_string(),
        width: w,
        height: h,
        fps,
        frames: frame_count,
        duration_s: frame_count as f64 / fps,
        start_s: clip_start_s,
        end_s: cfg.clip.and_then(|c| c.end_s),
        source_duration_s,
        detect_calls,
        detections: n_detections,
        device: active_device.to_string(),
        tracks: {
            let mut v: Vec<_> = summary.into_values().collect();
            v.sort_by_key(|t| t.id);
            v
        },
        records: all_records,
    };

    if let Some(path) = &cfg.output_json {
        write_json(path, &info_out)?;
    }
    if let Some(path) = &cfg.output_csv {
        write_csv(path, &info_out)?;
    }

    let elapsed = started.elapsed().as_secs_f64();
    info!(
        frames = frame_count,
        detect_calls,
        detections = n_detections,
        tracks = info_out.tracks.len(),
        device = %active_device,
        elapsed_s = elapsed,
        "pipeline finished"
    );
    if detect_calls == 0 {
        warn!("detector never ran — check motion thresholds or use --refresh-every 1");
    }
    if cfg.detector == DetectorKind::Yolo && info_out.tracks.is_empty() {
        if n_detections == 0 {
            warn!(
                score_thr = cfg.score_thr,
                class = %cfg.class_name,
                "yolo produced no detections at all - lower the score threshold \
                 (this fine-tuned model peaks around 0.2-0.4) or check the class name"
            );
        } else {
            warn!(
                detections = n_detections,
                min_hits = cfg.tracker.min_hits,
                score_thr = cfg.score_thr,
                "yolo found sparse detections but no track survived min_hits - \
                 lower the score threshold or min_hits"
            );
        }
    }
    Ok(info_out)
}

impl PipelineConfig {
    /// When true, motion-triggered frames only analyse the proposed ROIs.
    fn defer_features_only_on_roi(&self) -> bool {
        true
    }
}

/// Run YOLO on cropped ROIs (padded to square) and map boxes back.
fn detect_in_rois(
    det: &mut YoloDetector,
    rgb: &[u8],
    w: u32,
    h: u32,
    lb: &LetterboxCfg,
    rois: &[Bbox],
    class_id: usize,
) -> Result<Vec<Detection>> {
    let mut out = Vec::new();
    for roi in rois {
        let r = roi.clip(w, h);
        if r.wh().0 < 16.0 || r.wh().1 < 16.0 {
            continue;
        }
        let crop = crop_rgba(rgb, w, h, &r);
        let (cw, ch) = (r.wh().0.round() as u32, r.wh().1.round() as u32);
        let crop_lb = LetterboxCfg::new(cw, ch, lb.dst_w, lb.dst_h);
        let mut dets = det.detect(&crop, &crop_lb)?;
        for d in dets.iter_mut() {
            d.bbox = Bbox::new(
                d.bbox.x1 + r.x1,
                d.bbox.y1 + r.y1,
                d.bbox.x2 + r.x1,
                d.bbox.y2 + r.y1,
            );
        }
        // `usize::MAX` is the "accept every class" sentinel (`--class-name
        // any`); comparing it directly would drop every detection.
        out.extend(
            dets.into_iter()
                .filter(|d| accept_class(class_id, d.class_id)),
        );
    }
    Ok(out)
}

/// Class filter shared by the full-frame and ROI detection paths.
///
/// `wanted == usize::MAX` is the sentinel for "accept every class"
/// (`--class-name any`, blob backend).
fn accept_class(wanted: usize, class_id: usize) -> bool {
    wanted == usize::MAX || wanted == class_id
}

fn crop_rgba(rgb: &[u8], w: u32, _h: u32, r: &Bbox) -> Vec<u8> {
    let [x1, y1, x2, y2] = r.rounded();
    let (cw, chh) = (x2.saturating_sub(x1), y2.saturating_sub(y1));
    let mut buf = Vec::with_capacity((cw * chh * 3) as usize);
    for y in y1..y2 {
        let start = ((y * w + x1) * 3) as usize;
        let end = start + (cw * 3) as usize;
        buf.extend_from_slice(&rgb[start..end]);
    }
    buf
}

fn write_json(path: &Path, info: &RunInfo) -> Result<()> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let f = std::fs::File::create(path).context("creating json output")?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(f), info)?;
    info!("wrote {}", path.display());
    Ok(())
}

fn write_csv(path: &Path, info: &RunInfo) -> Result<()> {
    use std::io::Write as _;
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "frame,track_id,x1,y1,x2,y2,score,measured")?;
    for r in &info.records {
        writeln!(
            f,
            "{},{},{:.2},{:.2},{:.2},{:.2},{:.4},{}",
            r.frame, r.track_id, r.x1, r.y1, r.x2, r.y2, r.score, r.measured as u8
        )?;
    }
    info!("wrote {}", path.display());
    Ok(())
}

/// Fraction of foreground mask cells inside `b` (pixel-level activity proxy).
fn mask_ratio_in(bd: &BlobDetector, b: &Bbox, frame_w: u32, frame_h: u32) -> f32 {
    let (gw, gh) = bd.coarse_grid();
    let cell = 4.0f32;
    let clip = b.clip(frame_w, frame_h);
    let x1 = (clip.x1 / cell).floor().max(0.0) as usize;
    let y1 = (clip.y1 / cell).floor().max(0.0) as usize;
    let x2 = ((clip.x2 / cell).ceil() as usize).min(gw);
    let y2 = ((clip.y2 / cell).ceil() as usize).min(gh);
    if x1 >= x2 || y1 >= y2 {
        return 0.0;
    }
    let mut on = 0usize;
    let mut total = 0usize;
    for y in y1..y2 {
        for x in x1..x2 {
            total += 1;
            if bd.mask[y * gw + x] {
                on += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        on as f32 / total as f32
    }
}

// --------------------------------------------------------------------------- //
// Window analysis (0.5 s immobility scoring)
// --------------------------------------------------------------------------- //

#[derive(Debug, Clone)]
pub struct AnalyzeConfig {
    /// Optional XGBoost JSON model; a speed-threshold baseline otherwise.
    pub classifier: Option<PathBuf>,
    pub window_s: f64,
    pub stride_s: f64,
}

impl Default for AnalyzeConfig {
    fn default() -> Self {
        Self {
            classifier: None,
            window_s: crate::features::DEFAULT_WINDOW_S,
            stride_s: crate::features::DEFAULT_WINDOW_S * 0.5,
        }
    }
}

/// One scored window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredWindow {
    pub track_id: usize,
    pub start_frame: usize,
    pub end_frame: usize,
    pub start_t: f64,
    pub end_t: f64,
    pub immobile_prob: f32,
    pub label: crate::classify::ActClass,
    /// Classic RII score, for comparison with published TST criteria.
    pub rii: f32,
    pub values: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeReport {
    pub n_tracks: usize,
    pub feature_names: Vec<String>,
    pub mean_immobile_prob: f32,
    /// Fraction of windows labelled immobile, per track.
    pub per_track_immobile_fraction: Vec<(usize, f32)>,
    pub windows: Vec<ScoredWindow>,
}

/// Run tracking, then score every window of every track.
///
/// Thin wrapper around [`score_tracks`] for the CLI; the web server uses the
/// split form because it also needs the per-frame records for the UI.
pub fn analyze(cfg: &PipelineConfig, analyze_cfg: &AnalyzeConfig) -> Result<AnalyzeReport> {
    let info = run(cfg)?;
    score_tracks(&info, analyze_cfg)
}

/// Score 0.5 s windows from an already-computed tracking run.
pub fn score_tracks(info: &RunInfo, cfg: &AnalyzeConfig) -> Result<AnalyzeReport> {
    let fps = info.fps;
    let (frame_w, frame_h) = (info.width, info.height);

    // 1) regroup the exported per-frame records by track
    let mut by_track: HashMap<usize, Vec<(usize, Bbox, f32)>> = HashMap::new();
    for r in &info.records {
        by_track.entry(r.track_id).or_default().push((
            r.frame,
            Bbox::new(r.x1, r.y1, r.x2, r.y2),
            r.motion_energy,
        ));
    }

    let clf = crate::classify::load_or_baseline(cfg.classifier.as_deref())?;
    let names: Vec<String> = crate::features::FEATURE_NAMES
        .iter()
        .map(|s| s.to_string())
        .collect();
    let frame_diag = ((frame_w * frame_w + frame_h * frame_h) as f32).sqrt();

    let mut windows: Vec<ScoredWindow> = Vec::new();
    let mut per_track: Vec<(usize, f32)> = Vec::new();

    for (tid, recs) in by_track.iter() {
        if recs.len() < 4 {
            continue;
        }
        // reference size: median body diagonal of this track
        let mut diags: Vec<f32> = recs
            .iter()
            .map(|(_, b, _)| {
                let (w, h) = b.wh();
                (w * w + h * h).sqrt()
            })
            .collect();
        diags.sort_by(|a, b| a.total_cmp(b));
        let ref_size = diags[diags.len() / 2].max(1.0);

        let n = ((cfg.window_s * fps).round() as usize).max(2);
        let stride = ((cfg.stride_s * fps).round() as usize).max(1);

        let mut immobile = 0usize;
        let mut total = 0usize;
        let mut i = 0usize;
        while i + n <= recs.len() {
            let slice = &recs[i..i + n];
            let samples: Vec<(f64, Bbox)> = slice
                .iter()
                .map(|(f, b, _)| (*f as f64 / fps, *b))
                .collect();

            let mut w = crate::features::compute_window(
                *tid,
                slice[0].0,
                slice[slice.len() - 1].0,
                &samples,
                ref_size,
                fps,
            );

            // features that need context outside the window
            let energy: f32 = slice.iter().map(|(_, _, e)| *e).sum::<f32>() / slice.len() as f32;
            if let Some(idx) = w.names.iter().position(|s| s == "motion_energy") {
                w.values[idx] = energy;
            }
            if let Some(idx) = w.names.iter().position(|s| s == "size_norm") {
                w.values[idx] = (ref_size / frame_diag).clamp(0.0, 1.0);
            }

            let pred = clf.predict(&w.values)?;
            let rii = crate::features::rii_score(&samples, crate::features::RII_PIXEL_THRESHOLD);

            if pred.label == crate::classify::ActClass::Immobile {
                immobile += 1;
            }
            total += 1;
            windows.push(ScoredWindow {
                track_id: *tid,
                start_frame: w.start_frame,
                end_frame: w.end_frame,
                start_t: w.start_t,
                end_t: w.end_t,
                immobile_prob: pred.immobile_prob,
                label: pred.label,
                rii,
                values: w.values,
            });
            i += stride;
        }
        if total > 0 {
            per_track.push((*tid, immobile as f32 / total as f32));
        }
    }

    windows.sort_by(|a, b| (a.track_id, a.start_frame).cmp(&(b.track_id, b.start_frame)));
    per_track.sort_by_key(|(id, _)| *id);

    let mean_immobile_prob = if windows.is_empty() {
        0.0
    } else {
        windows.iter().map(|w| w.immobile_prob).sum::<f32>() / windows.len() as f32
    };

    Ok(AnalyzeReport {
        n_tracks: by_track.len(),
        feature_names: names,
        mean_immobile_prob,
        per_track_immobile_fraction: per_track,
        windows,
    })
}

/// Incremental version of [`score_tracks`] for the live web view.
///
/// Windows are emitted while the video is still being tracked, so the UI can
/// show the immobility curve and the window table as they grow. The final
/// report produced by [`score_tracks`] is identical to the concatenation of
/// everything emitted here (same window grid, same reference size, same
/// features), so the end state of the live view matches the persisted CSV.
pub struct LiveScorer {
    clf: Box<dyn crate::classify::Classifier>,
    fps: f64,
    /// Window length / stride in frames.
    n: usize,
    stride: usize,
    frame_diag: f32,
    /// Per-frame boxes of every track seen so far.
    tracks: HashMap<usize, Vec<(usize, Bbox, f32)>>,
    /// Next window index to emit, per track.
    next: HashMap<usize, usize>,
    /// Once a track ends its reference size is frozen, so windows already
    /// emitted keep a consistent normalisation.
    frozen: std::collections::HashSet<usize>,
}

impl LiveScorer {
    pub fn new(cfg: &AnalyzeConfig, frame_w: u32, frame_h: u32, fps: f64) -> Result<Self> {
        let clf = crate::classify::load_or_baseline(cfg.classifier.as_deref())?;
        let n = ((cfg.window_s * fps).round() as usize).max(2);
        let stride = ((cfg.stride_s * fps).round() as usize).max(1);
        Ok(Self {
            clf,
            fps,
            n,
            stride,
            frame_diag: ((frame_w * frame_w + frame_h * frame_h) as f32).sqrt(),
            tracks: HashMap::new(),
            next: HashMap::new(),
            frozen: std::collections::HashSet::new(),
        })
    }

    /// Append one frame of tracking output and return every window that just
    /// became complete.
    ///
    /// `energy` maps track id -> mean blob foreground ratio of this frame
    /// (empty for the YOLO backend, whose records carry 0.0 like `run`).
    pub fn push(
        &mut self,
        frame: usize,
        tracks: &[crate::track::Track],
        energy: &HashMap<usize, f32>,
    ) -> Vec<ScoredWindow> {
        for t in tracks {
            self.tracks.entry(t.id).or_default().push((
                frame,
                t.bbox(),
                energy.get(&t.id).copied().unwrap_or(0.0),
            ));
        }
        let mut out = Vec::new();
        for t in tracks {
            let Some(recs) = self.tracks.get(&t.id) else {
                continue;
            };
            let mut i = *self.next.get(&t.id).unwrap_or(&0);
            while i + self.n <= recs.len() {
                if let Some(w) = self.score_window(t.id, &recs[i..i + self.n]) {
                    out.push(w);
                }
                i += self.stride;
            }
            self.next.insert(t.id, i);
        }
        out
    }

    /// Score whatever tail is still pending (call once tracking finished).
    ///
    /// Window `i` is scored when the history covers `i + stride` frames, so
    /// the frames the tracker's ring buffer may drop are never needed.
    pub fn flush(&mut self) -> Vec<ScoredWindow> {
        let ids: Vec<usize> = self.tracks.keys().copied().collect();
        let mut out = Vec::new();
        for tid in ids {
            self.frozen.insert(tid);
            let Some(recs) = self.tracks.get(&tid) else {
                continue;
            };
            let len = recs.len();
            let start = *self.next.get(&tid).unwrap_or(&0);
            let mut i = start;
            while i < len {
                let end = (i + self.n).min(len);
                if end.saturating_sub(i) >= 2
                    && let Some(w) = self.score_window(tid, &recs[i..end])
                {
                    out.push(w);
                }
                i += self.stride;
            }
        }
        out
    }

    fn score_window(&self, tid: usize, slice: &[(usize, Bbox, f32)]) -> Option<ScoredWindow> {
        if slice.len() < 2 {
            return None;
        }
        let ref_size = if self.frozen.contains(&tid) {
            median_diag(slice)
        } else {
            self.tracks.get(&tid).map(|r| median_diag(r)).unwrap_or(1.0)
        };
        let samples: Vec<(f64, Bbox)> = slice
            .iter()
            .map(|(f, b, _)| (*f as f64 / self.fps, *b))
            .collect();
        let mut w = crate::features::compute_window(
            tid,
            slice[0].0,
            slice[slice.len() - 1].0,
            &samples,
            ref_size,
            self.fps,
        );
        let energy: f32 = slice.iter().map(|(_, _, e)| *e).sum::<f32>() / slice.len() as f32;
        if let Some(idx) = w.names.iter().position(|s| s == "motion_energy") {
            w.values[idx] = energy;
        }
        if let Some(idx) = w.names.iter().position(|s| s == "size_norm") {
            w.values[idx] = (ref_size / self.frame_diag).clamp(0.0, 1.0);
        }
        let pred = self.clf.predict(&w.values).ok()?;
        let rii = crate::features::rii_score(&samples, crate::features::RII_PIXEL_THRESHOLD);
        Some(ScoredWindow {
            track_id: tid,
            start_frame: w.start_frame,
            end_frame: w.end_frame,
            start_t: w.start_t,
            end_t: w.end_t,
            immobile_prob: pred.immobile_prob,
            label: pred.label,
            rii,
            values: w.values,
        })
    }
}

/// Median body diagonal over a track's boxes (same rule as `score_tracks`).
fn median_diag(recs: &[(usize, Bbox, f32)]) -> f32 {
    if recs.is_empty() {
        return 1.0;
    }
    let mut diags: Vec<f32> = recs
        .iter()
        .map(|(_, b, _)| {
            let (w, h) = b.wh();
            (w * w + h * h).sqrt()
        })
        .collect();
    diags.sort_by(|a, b| a.total_cmp(b));
    diags[diags.len() / 2].max(1.0)
}

/// Write scored windows as CSV rows (no header) into `w`.
///
/// Shared by the batch [`write_window_csv`] and the web server's live writer,
/// so both emit byte-identical rows.
pub fn write_window_rows<W: std::io::Write>(w: &mut W, windows: &[ScoredWindow]) -> Result<()> {
    for win in windows {
        write!(
            w,
            "{},{},{},{:.3},{:.3},{:.4},{},{:.3}",
            win.track_id,
            win.start_frame,
            win.end_frame,
            win.start_t,
            win.end_t,
            win.immobile_prob,
            match win.label {
                crate::classify::ActClass::Immobile => "immobile",
                crate::classify::ActClass::Mobile => "mobile",
            },
            win.rii
        )?;
        for v in &win.values {
            write!(w, ",{v:.5}")?;
        }
        writeln!(w)?;
    }
    Ok(())
}

/// Write a scored report as CSV (one row per window).
pub fn write_window_csv(path: &Path, names: &[String], windows: &[ScoredWindow]) -> Result<()> {
    use std::io::Write as _;
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p).ok();
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(
        f,
        "track_id,start_frame,end_frame,start_t,end_t,immobile_prob,label,rii"
    )?;
    for n in names {
        write!(f, ",{n}")?;
    }
    writeln!(f)?;
    write_window_rows(&mut f, windows)?;
    info!("wrote {}", path.display());
    Ok(())
}

/// Reference used by the blob backend to separate animals from the enclosure.
#[derive(Debug, Clone)]
pub enum Reference {
    /// Frame index inside the analysed clip (empty enclosure). Index 0 is the
    /// first frame of the clip, not of the source file.
    Frame(usize),
    /// Path to a still image of the empty enclosure.
    Image(PathBuf),
}

/// Even-rounded copy of a size (ffmpeg/encoder friendly).
fn fit_even(w: u32, h: u32) -> (u32, u32) {
    (((w + 1) / 2) * 2, ((h + 1) / 2) * 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_class_sentinel_accepts_everything() {
        // `--class-name any` / blob backend.
        assert!(accept_class(usize::MAX, 0));
        assert!(accept_class(usize::MAX, 7));
    }

    #[test]
    fn accept_class_filters_exact_id() {
        assert!(accept_class(0, 0));
        assert!(!accept_class(0, 1));
        // Regression: the ROI path used to compare against the sentinel and
        // therefore dropped every detection in `any` mode.
        assert!(!accept_class(2, 3));
    }
}
