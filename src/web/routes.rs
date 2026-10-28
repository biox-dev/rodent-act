//! Axum router: REST API, SSE streams and (optionally) the built SPA.

use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Context;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path as UrlPath, Query, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::Stream;
use image::ImageEncoder as _;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tower_http::cors::CorsLayer;

use crate::dataset;
use crate::pipeline::{AnalyzeConfig, DetectorKind, PipelineConfig, Reference};
use crate::track::TrackerConfig;
use crate::web::state::{self, Job, JobManager, JobStatus, JobView};
use crate::web::telemetry::{self, LogRecord};

#[derive(Clone)]
pub struct AppState {
    pub jobs: Arc<JobManager>,
    pub log_tx: broadcast::Sender<LogRecord>,
}

// --------------------------------------------------------------------------- //
// request / response DTOs
// --------------------------------------------------------------------------- //

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateJob {
    /// File name inside the media directory (see `/api/media`).
    pub media: String,
    #[serde(default)]
    pub detector: Option<String>,
    /// Empty-enclosure reference: `empty_ref.png`, `frame:0`, or omitted.
    #[serde(default)]
    pub reference: Option<String>,
    #[serde(default)]
    pub blob_threshold: Option<u8>,
    #[serde(default)]
    pub blob_min_area: Option<f32>,
    #[serde(default)]
    pub max_side: Option<u32>,
    #[serde(default)]
    pub min_hits: Option<usize>,
    #[serde(default)]
    pub max_age: Option<usize>,
    #[serde(default)]
    pub max_frames: Option<usize>,
    /// Source-time start of the analysed segment, in seconds (default 0).
    #[serde(default)]
    pub start_s: Option<f64>,
    /// Source-time end of the analysed segment, in seconds (default: EOF).
    /// The segment is decoded in place - no clipped file is written.
    #[serde(default)]
    pub end_s: Option<f64>,
    #[serde(default)]
    pub score_thr: Option<f32>,
    #[serde(default)]
    pub class_name: Option<String>,
    /// Model path relative to the working directory (YOLO backend).
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub window: Option<f64>,
    #[serde(default)]
    pub stride: Option<f64>,
    /// Decision threshold on `immobile_prob` (default 0.5).
    #[serde(default)]
    pub threshold: Option<f32>,
    #[serde(default)]
    pub clf: Option<String>,
    /// Write an H.264 preview next to the frames (extra ffmpeg pass).
    #[serde(default)]
    pub write_video: Option<bool>,
}

#[derive(Serialize)]
struct MediaFile {
    name: String,
    size: u64,
    modified_ms: u64,
}
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn api_error(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(serde_json::json!({ "error": msg.into() }))).into_response()
}

/// Output of `/api/info`, mirroring what the CLI can do.
#[derive(Serialize)]
struct ServerInfo {
    version: &'static str,
    media_dir: String,
    out_dir: String,
    ffmpeg: bool,
    /// Whether this binary was compiled with `--features cuda`.
    gpu_build: bool,
    /// Whether a CUDA execution provider can actually be registered.
    gpu_available: bool,
    /// Human readable device summary (`Describe()` output).
    device: String,
    features: Vec<String>,
    default_window_s: f64,
}

/// Resolve a user-supplied file name inside `dir`, refusing path traversal.
fn safe_join(dir: &Path, name: &str) -> Option<PathBuf> {
    let candidate = Path::new(name);
    if candidate.is_absolute()
        || candidate
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return None;
    }
    let joined = dir.join(candidate);
    joined.starts_with(dir).then_some(joined)
}

fn has_ffmpeg() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Directory scanned for ONNX models, relative to the server's working
/// directory (i.e. the repository root when started with `cargo run`).
const MODELS_DIR: &str = "models";

/// Model used when the request does not name one. Picked ahead of the stock
/// `yolo11n.onnx` because COCO has no rodent class at all.
const DEFAULT_MODEL: &str = "models/mouse.onnx";

/// An `.onnx` file offered to the UI.
#[derive(Serialize)]
struct ModelFile {
    /// Path handed back to `POST /api/jobs` (already relative to the CWD).
    path: String,
    name: String,
    size: u64,
}

/// `GET /api/models` - list the ONNX files under `models/`.
///
/// Only the file list is reported: the model's class names are resolved by the
/// pipeline itself from the ONNX metadata, so the UI does not need to know
/// them. `--class-name any` accepts every class regardless.
async fn list_models() -> Json<Vec<ModelFile>> {
    let mut out: Vec<ModelFile> = std::fs::read_dir(MODELS_DIR)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            e.path().is_file()
                && e.path()
                    .extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("onnx"))
        })
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            ModelFile {
                path: format!("{MODELS_DIR}/{name}"),
                size: e.metadata().map(|m| m.len()).unwrap_or(0),
                name,
            }
        })
        .collect();
    // `mouse.onnx` first, then alphabetical.
    out.sort_by_key(|m| (m.name != "mouse.onnx", m.name.clone()));
    Json(out)
}

async fn info(State(st): State<AppState>) -> Json<ServerInfo> {
    Json(ServerInfo {
        version: env!("CARGO_PKG_VERSION"),
        media_dir: st.jobs.media_dir().display().to_string(),
        out_dir: st.jobs.out_dir().display().to_string(),
        ffmpeg: has_ffmpeg(),
        gpu_build: crate::detect::device::Device::compiled_with_gpu(),
        gpu_available: crate::detect::device::device_available(),
        device: crate::detect::device::describe(),
        features: crate::features::FEATURE_NAMES
            .iter()
            .map(|s| s.to_string())
            .collect(),
        default_window_s: crate::features::DEFAULT_WINDOW_S,
    })
}

async fn list_media(State(st): State<AppState>) -> Json<Vec<MediaFile>> {
    let dir = st.jobs.media_dir().to_path_buf();
    let mut files = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !matches!(
                ext.as_str(),
                "mp4" | "avi" | "mov" | "mkv" | "webm" | "png" | "jpg"
            ) {
                continue;
            }
            let meta = entry.metadata().ok();
            files.push(MediaFile {
                name: entry.file_name().to_string_lossy().to_string(),
                size: meta.as_ref().map(|m| m.len()).unwrap_or(0),
                modified_ms: meta
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
            });
        }
    }
    files.sort_by_key(|f| std::cmp::Reverse(f.modified_ms));
    Json(files)
}

/// Serve a media file with HTTP range support (needed for `<video>` seeking).
async fn media_file(
    State(st): State<AppState>,
    UrlPath(name): UrlPath<String>,
    headers: axum::http::HeaderMap,
) -> Response {
    let Some(path) = safe_join(st.jobs.media_dir(), &name) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid media name");
    };
    serve_file_range(&path, &headers).await
}

async fn job_video(
    State(st): State<AppState>,
    UrlPath(id): UrlPath<String>,
    headers: axum::http::HeaderMap,
) -> Response {
    let Some(job) = st.jobs.get(&id) else {
        return api_error(StatusCode::NOT_FOUND, "unknown job");
    };
    let path = job
        .video
        .read()
        .ok()
        .and_then(|g| g.clone())
        .unwrap_or_default();
    serve_file_range(&path, &headers).await
}

/// Minimal range-aware file responder (single `bytes=start-end` range).
async fn serve_file_range(path: &Path, headers: &axum::http::HeaderMap) -> Response {
    let Ok(meta) = tokio::fs::metadata(path).await else {
        return api_error(StatusCode::NOT_FOUND, "file not found");
    };
    let len = meta.len();
    let mime = match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        _ => "application/octet-stream",
    };
    let range = headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_range);

    let build = |status: StatusCode, start: u64, end: u64| -> Response {
        let mut resp = Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, mime)
            .header(header::ACCEPT_RANGES, "bytes")
            .header(header::CONTENT_LENGTH, (end - start + 1).to_string());
        if status == StatusCode::PARTIAL_CONTENT {
            resp = resp.header(header::CONTENT_RANGE, format!("bytes {start}-{end}/{len}"));
        }
        let body = file_stream(path.to_path_buf(), start, end);
        resp.body(Body::from_stream(body))
            .unwrap_or_else(|_| api_error(StatusCode::INTERNAL_SERVER_ERROR, "body error"))
    };

    match range {
        Some((start, end)) if start < len => {
            let end = end.min(len - 1);
            build(StatusCode::PARTIAL_CONTENT, start, end)
        }
        Some(_) => api_error(
            StatusCode::RANGE_NOT_SATISFIABLE,
            format!("range outside file (len {len})"),
        ),
        None => build(StatusCode::OK, 0, len.saturating_sub(1)),
    }
}

fn file_stream(
    path: PathBuf,
    start: u64,
    end: u64,
) -> impl Stream<Item = std::io::Result<axum::body::Bytes>> + Send + 'static {
    async_stream::stream! {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};
        let mut f = match tokio::fs::File::open(&path).await {
            Ok(f) => f,
            Err(e) => {
                yield Err(e);
                return;
            }
        };
        if f.seek(std::io::SeekFrom::Start(start)).await.is_err() {
            return;
        }
        let mut left = end - start + 1;
        let mut buf = vec![0u8; 64 * 1024];
        while left > 0 {
            let want = buf.len().min(left as usize);
            match f.read(&mut buf[..want]).await {
                Ok(0) => break,
                Ok(n) => {
                    left -= n as u64;
                    yield Ok(axum::body::Bytes::copy_from_slice(&buf[..n]));
                }
                Err(e) => {
                    yield Err(e);
                    break;
                }
            }
        }
    }
}

/// Parse `bytes=0-1023` / `bytes=-1024` / `bytes=1024-`.
fn parse_range(value: &str) -> Option<(u64, u64)> {
    let spec = value.strip_prefix("bytes=")?;
    let (a, b) = spec.split_once('-')?;
    match (a.trim(), b.trim()) {
        ("", "") => None,
        ("", suffix) => {
            let n: u64 = suffix.parse().ok()?;
            Some((u64::MAX.saturating_sub(n - 1), u64::MAX))
        }
        (start, "") => Some((start.parse().ok()?, u64::MAX)),
        (start, end) => Some((start.parse().ok()?, end.parse().ok()?)),
    }
}

async fn create_job(State(st): State<AppState>, Json(req): Json<CreateJob>) -> Response {
    let Some(input) = safe_join(st.jobs.media_dir(), &req.media) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid media name");
    };
    if !input.exists() {
        return api_error(
            StatusCode::NOT_FOUND,
            format!("media file not found: {}", req.media),
        );
    }

    // Resolve the clip window before touching the job directory: an invalid
    // range is a client error, and the probe gives the frame rate needed to
    // reject windows shorter than a single frame.
    let probe = match crate::video_io::probe(&input) {
        Ok(p) => p,
        Err(e) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("cannot probe media file: {e}"),
            );
        }
    };
    let source_duration = probe.duration_s;
    let start_s = req.start_s.unwrap_or(0.0);
    if !start_s.is_finite() || start_s < 0.0 {
        return api_error(
            StatusCode::BAD_REQUEST,
            format!("startS must be a non-negative number of seconds (got {start_s})"),
        );
    }
    let start_s = if source_duration > 0.0 {
        start_s.min(source_duration)
    } else {
        start_s
    };
    let end_s = match req.end_s {
        Some(e) if !e.is_finite() => {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("endS must be a finite number of seconds (got {e})"),
            );
        }
        Some(e) if source_duration > 0.0 => Some(e.min(source_duration)),
        other => other,
    };
    if let Some(e) = end_s
        && e <= start_s
    {
        return api_error(
            StatusCode::BAD_REQUEST,
            format!("endS ({e:.2}s) must be after startS ({start_s:.2}s)"),
        );
    }
    // `None` when the window covers the whole file, so the reader keeps the
    // container's exact frame count and skips the extra ffmpeg seek options.
    let clip = match end_s {
        Some(e) if e < source_duration - 1e-3 => Some(crate::video_io::Clip::new(start_s, Some(e))),
        _ if start_s > 1e-3 => Some(crate::video_io::Clip::new(start_s, None)),
        _ => None,
    };
    if let Some(c) = clip
        && source_duration > 0.0
    {
        let len = c.len(source_duration);
        if len * probe.fps < 1.0 {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "clip is shorter than one frame ({len:.3}s at {:.2} fps)",
                    probe.fps
                ),
            );
        }
    }

    let id = format!("j{}-{:04x}", now_ms(), rand_suffix());
    let job_dir = st.jobs.job_dir(&id);
    if let Err(e) = std::fs::create_dir_all(&job_dir) {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot create job dir: {e}"),
        );
    }

    let detector = match req.detector.as_deref() {
        None | Some("blob") => DetectorKind::Blob,
        Some("yolo") => DetectorKind::Yolo,
        Some(other) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("unknown detector `{other}` (use blob|yolo)"),
            );
        }
    };
    let reference = req.reference.as_deref().and_then(|spec| {
        if let Some(idx) = spec.strip_prefix("frame:") {
            idx.trim().parse::<usize>().ok().map(Reference::Frame)
        } else if spec.is_empty() {
            None
        } else {
            safe_join(st.jobs.media_dir(), spec).map(Reference::Image)
        }
    });

    let window = req
        .window
        .unwrap_or(crate::features::DEFAULT_WINDOW_S)
        .clamp(0.1, 10.0);
    let video_path = req
        .write_video
        .unwrap_or(false)
        .then(|| job_dir.join("annotated.mp4"));

    // Resolve the model path up front: a missing/corrupt model would otherwise
    // only fail inside the worker thread, long after the job was accepted.
    let model = req
        .model
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_MODEL);
    if detector == DetectorKind::Yolo {
        // Accept both `mouse.onnx` (bare file name) and `models/mouse.onnx`
        // (what `/api/models` returns); strip any `models/` prefix first.
        let bare = model
            .trim_start_matches("./")
            .strip_prefix(&format!("{MODELS_DIR}/"))
            .unwrap_or(model);
        let exists = safe_join(Path::new(MODELS_DIR), bare).is_some_and(|p| p.is_file());
        if !exists {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!(
                    "model not found: {model} (paths are relative to the server's \
                     working directory `{}`; see GET /api/models)",
                    std::env::current_dir()
                        .map(|d| d.display().to_string())
                        .unwrap_or_else(|_| "?".into())
                ),
            );
        }
    }

    // Empty/"any" means "accept every class"; otherwise use the given name.
    let class_name = match req.class_name.map(|s| s.trim().to_string()) {
        Some(c) if !c.is_empty() => c,
        _ => "any".to_string(),
    };

    let cfg = PipelineConfig {
        input: input.clone(),
        clip,
        output_video: video_path,
        output_json: None,
        output_csv: None,
        detector,
        model: PathBuf::from(model),
        max_side: req.max_side.unwrap_or(640).clamp(160, 1920),
        score_thr: req.score_thr.unwrap_or(0.2),
        iou_thr: 0.45,
        blob_threshold: req.blob_threshold.unwrap_or(22),
        blob_min_area: req.blob_min_area.unwrap_or(0.0015),
        reference,
        class_name,
        fallback_animal: true,
        detect_every: 1,
        refresh_every: 90,
        tracker: TrackerConfig {
            min_hits: req.min_hits.unwrap_or(5).max(1),
            max_age: req.max_age.unwrap_or(30).max(1),
            ..Default::default()
        },
        write_video: req.write_video.unwrap_or(false),
        show_rois: false,
        show_trails: true,
        threads: 2,
        // The web UI cannot choose a device yet; `auto` picks the GPU whenever
        // the binary was built with `--features cuda` and one is usable.
        device: crate::detect::device::Device::Auto,
        gpu_device_id: 0,
        max_frames: req.max_frames.filter(|n| *n > 0),
        preview_every: None,
        live_scoring: None,
    };
    let analyze_cfg = AnalyzeConfig {
        classifier: req.clf.filter(|s| !s.is_empty()).map(PathBuf::from),
        window_s: window,
        stride_s: req.stride.unwrap_or(window * 0.5).max(0.05),
        threshold: req
            .threshold
            .unwrap_or(crate::pipeline::DEFAULT_THRESHOLD)
            .clamp(0.01, 0.99),
    };

    let (finished_tx, _) = tokio::sync::watch::channel(false);
    let (log_tx, _) = broadcast::channel(256);
    let job = Arc::new(Job {
        id: id.clone(),
        input_name: req.media.clone(),
        input,
        clip,
        dir: job_dir.clone(),
        status: std::sync::RwLock::new(JobStatus::Queued),
        error: std::sync::RwLock::new(None),
        created_ms: now_ms(),
        frames_done: std::sync::RwLock::new(0),
        total_frames: std::sync::RwLock::new(0),
        run: std::sync::RwLock::new(None),
        report: std::sync::RwLock::new(None),
        analyze: std::sync::RwLock::new(analyze_cfg.clone()),
        windows: std::sync::RwLock::new(Vec::new()),
        feature_names: std::sync::RwLock::new(
            crate::features::FEATURE_NAMES
                .iter()
                .map(|s| s.to_string())
                .collect(),
        ),
        video: std::sync::RwLock::new(None),
        frames: Arc::new(state::FrameBus::new(15.0, 72)),
        logs: Arc::new(std::sync::RwLock::new(Vec::new())),
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        finished: finished_tx,
        duration_s: std::sync::RwLock::new(None),
    });
    st.jobs.insert(job.clone());
    // Persist the header before the worker starts so the job is visible again
    // (as interrupted) even if the server dies right away.
    job.save_meta();

    // Bridge pipeline logs into the per-job buffer + live SSE stream.
    let mut log_rx = st.log_tx.subscribe();
    let job_logs = job.logs.clone();
    let job_log_tx = log_tx.clone();
    tokio::spawn(async move {
        while let Ok(rec) = log_rx.recv().await {
            if let Ok(mut buf) = job_logs.write() {
                if buf.len() >= 2000 {
                    buf.drain(..500);
                }
                buf.push(rec.clone());
            }
            let _ = job_log_tx.send(rec);
        }
    });

    let csv_path = job_dir.join("windows.csv");
    let worker_job = job.clone();
    let job_id = id.clone();
    tokio::task::spawn_blocking(move || {
        if let Err(e) = state::execute(worker_job.clone(), cfg, analyze_cfg, csv_path) {
            state::fail(&worker_job, &e);
        }
    });

    tracing::info!(job = %job_id, media = %req.media, "analysis job queued");
    (StatusCode::CREATED, Json(job.view())).into_response()
}

fn rand_suffix() -> u16 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(now_ms());
    (h.finish() & 0xffff) as u16
}

async fn list_jobs(State(st): State<AppState>) -> Json<Vec<JobView>> {
    Json(st.jobs.list().iter().map(|j| j.view()).collect())
}

async fn get_job(State(st): State<AppState>, UrlPath(id): UrlPath<String>) -> Response {
    match st.jobs.get(&id) {
        Some(job) => Json(job.view()).into_response(),
        None => api_error(StatusCode::NOT_FOUND, "unknown job"),
    }
}

async fn cancel_job(State(st): State<AppState>, UrlPath(id): UrlPath<String>) -> Response {
    let Some(job) = st.jobs.get(&id) else {
        return api_error(StatusCode::NOT_FOUND, "unknown job");
    };
    if !job.status().is_terminal() {
        job.cancel.store(true, Ordering::Relaxed);
        job.set_status(JobStatus::Cancelled);
        let _ = job.finished.send(true);
    }
    Json(job.view()).into_response()
}

/// `POST /api/jobs/{id}/rescore` parameters.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RescoreJob {
    /// Window length in seconds (omitted = keep the job's current value).
    #[serde(default)]
    pub window: Option<f64>,
    /// Window stride in seconds (omitted = keep the job's current value).
    #[serde(default)]
    pub stride: Option<f64>,
    /// Decision threshold on `immobile_prob` (omitted = keep the current one).
    #[serde(default)]
    pub threshold: Option<f32>,
    /// XGBoost model path; `""` falls back to the speed baseline, `null` or
    /// omitted keeps the model used by the previous run.
    #[serde(default)]
    pub clf: Option<String>,
}

/// Re-score the windows of a finished job from its stored tracking records.
///
/// No ffmpeg, no detection, no tracking: only feature extraction and the
/// classifier run, so changing `window` / `stride` / model costs milliseconds
/// instead of re-analysing the whole video.
async fn rescore_job(
    State(st): State<AppState>,
    UrlPath(id): UrlPath<String>,
    Json(req): Json<RescoreJob>,
) -> Response {
    let Some(job) = st.jobs.get(&id) else {
        return api_error(StatusCode::NOT_FOUND, "unknown job");
    };
    if !job.status().is_terminal() {
        return api_error(
            StatusCode::CONFLICT,
            "job is still running - wait for it to finish (or cancel it) first",
        );
    }

    let current = job.analyze.read().map(|g| g.clone()).unwrap_or_default();
    let window = req.window.unwrap_or(current.window_s).clamp(0.1, 10.0);
    let stride = req.stride.unwrap_or(current.stride_s).max(0.05);
    let classifier = match req.clf {
        Some(s) => {
            let s = s.trim();
            (!s.is_empty()).then(|| PathBuf::from(s))
        }
        None => current.classifier.clone(),
    };
    let cfg = AnalyzeConfig {
        classifier,
        window_s: window,
        stride_s: stride,
        threshold: req.threshold.unwrap_or(current.threshold).clamp(0.01, 0.99),
    };

    let worker = job.clone();
    let csv_path = job.dir.join("windows.csv");
    let result =
        tokio::task::spawn_blocking(move || state::rescore(&worker, &cfg, &csv_path)).await;
    match result {
        Ok(Ok(report)) => {
            let r = &*report;
            Json(serde_json::json!({
                "feature_names": &r.feature_names,
                "n_windows": r.windows.len(),
                "threshold": r.threshold,
                "mean_immobile_prob": r.mean_immobile_prob,
                "per_track_immobile_fraction": &r.per_track_immobile_fraction,
                "per_track_stats": &r.per_track_stats,
                "windows": &r.windows,
                "job": job.view(),
            }))
            .into_response()
        }
        Ok(Err(e)) => api_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
        Err(e) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("rescore task failed: {e}"),
        ),
    }
}

/// Per-frame tracking records (boxes for every frame of every track).
///
/// The UI uses these to draw an overlay that follows the video element, so
/// boxes stay visible even for previously downloaded videos.
async fn job_records(State(st): State<AppState>, UrlPath(id): UrlPath<String>) -> Response {
    let Some(job) = st.jobs.get(&id) else {
        return api_error(StatusCode::NOT_FOUND, "unknown job");
    };
    let run = job.run.read().ok().and_then(|g| g.clone());
    let Some(run) = run else {
        return api_error(StatusCode::CONFLICT, "job still tracking");
    };
    Json(serde_json::json!({
        "width": run.width,
        "height": run.height,
        "fps": run.fps,
        "frames": run.frames,
        // Frame indices/timestamps below are relative to the clip start; the
        // UI adds `start_s` when it seeks the source video.
        "start_s": run.start_s,
        "end_s": run.end_s,
        "source_duration_s": run.source_duration_s,
        "tracks": run.tracks,
        "records": run.records,
    }))
    .into_response()
}

/// Scored windows. Available *while the job runs*: `complete` is false until
/// the final report exists, and `progress` tells the UI how much of the video
/// has been scored so the curve can be drawn live.
async fn job_windows(State(st): State<AppState>, UrlPath(id): UrlPath<String>) -> Response {
    let Some(job) = st.jobs.get(&id) else {
        return api_error(StatusCode::NOT_FOUND, "unknown job");
    };
    let report = job.report.read().ok().and_then(|g| g.clone());
    let windows = job.windows.read().map(|g| g.clone()).unwrap_or_default();
    let names = job
        .feature_names
        .read()
        .map(|g| g.clone())
        .unwrap_or_default();
    let done = *job.frames_done.read().unwrap_or_else(|e| e.into_inner());
    let total = *job.total_frames.read().unwrap_or_else(|e| e.into_inner());
    let status = job.status();
    // An idle job with no window yet is "not started" rather than "running";
    // the UI uses this to keep polling instead of showing an empty chart.
    let complete = report.is_some() || status.is_terminal();
    Json(serde_json::json!({
        "feature_names": names,
        "threshold": report.as_ref().map(|r| r.threshold),
        "mean_immobile_prob": report.as_ref().map(|r| r.mean_immobile_prob),
        "per_track_immobile_fraction": report.as_ref().map(|r| &r.per_track_immobile_fraction),
        "per_track_stats": report.as_ref().map(|r| &r.per_track_stats),
        "windows": windows,
        "complete": complete,
        "status": status,
        "progress": if total > 0 { (done as f64 / total as f64).clamp(0.0, 1.0) } else { 0.0 },
        "frames_done": done,
        "total_frames": total,
    }))
    .into_response()
}

/// Live window updates. Each `windows` event carries the windows scored since
/// the previous event, so the client appends instead of re-downloading.
async fn job_window_events(State(st): State<AppState>, UrlPath(id): UrlPath<String>) -> Response {
    let Some(job) = st.jobs.get(&id) else {
        return api_error(StatusCode::NOT_FOUND, "unknown job");
    };
    let mut finished_rx = job.finished.subscribe();
    let job_poll = job.clone();

    let stream = async_stream::stream! {
        let mut sent = 0usize;
        let mut ticker = tokio::time::interval(Duration::from_millis(700));
        loop {
            tokio::select! {
                biased;
                changed = finished_rx.changed() => {
                    if changed.is_err() {
                        break;
                    }
                    // Flush whatever is left before closing the stream.
                    let pending: Vec<_> = job_poll
                        .windows
                        .read()
                        .map(|g| g.get(sent..).map(|s| s.to_vec()).unwrap_or_default())
                        .unwrap_or_default();
                    if !pending.is_empty() {
                        yield Ok::<Event, Infallible>(window_event(&pending, job_poll.status().is_terminal()));
                    }
                    yield Ok(Event::default().event("done").data("{}"));
                    return;
                }
                _ = ticker.tick() => {
                    let pending: Vec<_> = job_poll
                        .windows
                        .read()
                        .map(|g| g.get(sent..).map(|s| s.to_vec()).unwrap_or_default())
                        .unwrap_or_default();
                    if !pending.is_empty() {
                        sent += pending.len();
                        yield Ok(window_event(&pending, job_poll.status().is_terminal()));
                    } else if job_poll.status().is_terminal() {
                        yield Ok(Event::default().event("done").data("{}"));
                        return;
                    }
                }
            }
        }
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

fn window_event(windows: &[crate::pipeline::ScoredWindow], complete: bool) -> Event {
    let data = serde_json::json!({
        "complete": complete,
        "windows": windows,
    });
    Event::default()
        .event("windows")
        .data(serde_json::to_string(&data).unwrap_or_else(|_| "{}".into()))
}

async fn job_logs(
    State(st): State<AppState>,
    UrlPath(id): UrlPath<String>,
    Query(q): Query<LogQuery>,
) -> Response {
    let Some(job) = st.jobs.get(&id) else {
        return api_error(StatusCode::NOT_FOUND, "unknown job");
    };
    let since = q.since.unwrap_or(0);
    let limit = q.limit.unwrap_or(500).clamp(1, 5000);
    let logs = job.logs.read().ok();
    let mut records: Vec<LogRecord> = logs
        .as_ref()
        .map(|g| g.iter().skip(since).cloned().collect())
        .unwrap_or_default();
    let total = since + records.len();
    if records.len() > limit {
        let drop = records.len() - limit;
        records.drain(..drop);
    }
    let next = total;
    Json(serde_json::json!({ "total": total, "next": next, "records": records })).into_response()
}

#[derive(Debug, Deserialize)]
struct LogQuery {
    since: Option<usize>,
    limit: Option<usize>,
}

// ------------------------------- SSE: status ------------------------------- //

async fn job_events(State(st): State<AppState>, UrlPath(id): UrlPath<String>) -> Response {
    let Some(job) = st.jobs.get(&id) else {
        return api_error(StatusCode::NOT_FOUND, "unknown job");
    };
    // Subscribe *before* building the initial snapshot so no update is lost.
    let mut log_rx = st.log_tx.subscribe();
    let mut finished_rx = job.finished.subscribe();

    let initial_logs: Vec<LogRecord> = job
        .logs
        .read()
        .map(|g| g.iter().cloned().collect())
        .unwrap_or_default();
    let job_for_poll = job.clone();

    let stream = async_stream::stream! {
        for rec in initial_logs {
            yield Ok::<Event, Infallible>(telemetry::log_event(&rec));
        }
        yield Ok(telemetry::status_event(status_name(job_for_poll.status())));
        if job_for_poll.status().is_terminal() {
            yield Ok(Event::default().event("done").data("{}"));
            return;
        }

        let mut ticker = tokio::time::interval(Duration::from_millis(500));
        loop {
            tokio::select! {
                biased;
                rec = log_rx.recv() => {
                    match rec {
                        Ok(rec) => {
                            yield Ok(telemetry::log_event(&rec));
                        }
                        Err(broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(_) => break,
                    }
                }
                changed = finished_rx.changed() => {
                    if changed.is_err() {
                        break;
                    }
                    yield Ok(telemetry::status_event(status_name(job_for_poll.status())));
                    yield Ok(Event::default().event("done").data("{}"));
                    return;
                }
                _ = ticker.tick() => {
                    let (done, total) = {
                        let d = job_for_poll.frames_done.read().map(|g| *g).unwrap_or(0);
                        let t = job_for_poll.total_frames.read().map(|g| *g).unwrap_or(0);
                        (d, t)
                    };
                    if total > 0 {
                        yield Ok(telemetry::progress_event(
                            done as f64 / total as f64,
                            &format!("{done}/{total} frames"),
                        ));
                    }
                    if job_for_poll.status().is_terminal() {
                        yield Ok(telemetry::status_event(status_name(job_for_poll.status())));
                        yield Ok(Event::default().event("done").data("{}"));
                        return;
                    }
                }
            }
        }
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

fn status_name(s: JobStatus) -> &'static str {
    match s {
        JobStatus::Queued => "queued",
        JobStatus::Running => "running",
        JobStatus::Done => "done",
        JobStatus::Failed => "failed",
        JobStatus::Cancelled => "cancelled",
    }
}

// ------------------------------- SSE: frames ------------------------------- //

/// Latest annotated frame (JPEG). 204 when no frame was published yet.
async fn latest_frame(State(st): State<AppState>, UrlPath(id): UrlPath<String>) -> Response {
    let Some(job) = st.jobs.get(&id) else {
        return api_error(StatusCode::NOT_FOUND, "unknown job");
    };
    match job.frames.last() {
        Some(p) => jpeg_response(&p.jpeg, &p.meta),
        None => StatusCode::NO_CONTENT.into_response(),
    }
}

fn jpeg_response(bytes: &[u8], meta: &state::FrameMeta) -> Response {
    let mut resp = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/jpeg")
        .header(header::CACHE_CONTROL, "no-store");
    if let Ok(v) = HeaderValue::from_str(&meta.index.to_string()) {
        resp = resp.header("X-Frame-Index", v);
    }
    if let Ok(v) = HeaderValue::from_str(&format!("{:.4}", meta.t)) {
        resp = resp.header("X-Frame-Time", v);
    }
    resp.body(Body::from(bytes.to_vec()))
        .unwrap_or_else(|_| api_error(StatusCode::INTERNAL_SERVER_ERROR, "body error"))
}

/// Live frame stream. Each SSE `data:` is base64-encoded JPEG.
async fn frame_stream(State(st): State<AppState>, UrlPath(id): UrlPath<String>) -> Response {
    let Some(job) = st.jobs.get(&id) else {
        return api_error(StatusCode::NOT_FOUND, "unknown job");
    };
    let job_frames = job.frames.clone();
    let job_poll = job.clone();

    let stream = async_stream::stream! {
        // Attach from *inside* the future: the handler is dropped as soon as
        // the body is handed to hyper, but this stream lives for the whole
        // connection. While attached, the pipeline encodes annotated frames.
        let _viewer = job_frames.attach();
        let mut rx = job_frames.subscribe();
        let frame_event = |packet: &state::FramePacket| {
            let b64 = base64_encode(&packet.jpeg);
            Ok::<Event, Infallible>(
                Event::default()
                    .event("frame")
                    .data(format!(
                        "{{\"index\":{},\"t\":{:.4},\"total\":{},\"jpeg\":\"{}\"}}",
                        packet.meta.index, packet.meta.t, packet.meta.total, b64
                    )),
            )
        };
        // Seed with the most recent frame: the pipeline may already have
        // dropped frames for this viewer, and a `<img>` needs one immediately.
        if let Some(last) = job_frames.last() {
            yield frame_event(&last);
        }
        loop {
            match rx.recv().await {
                Ok(packet) => yield frame_event(&packet),
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
            if job_poll.status().is_terminal() && job_poll.frames.last().is_none() {
                break;
            }
        }
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

/// Small, dependency-free base64 encoder (RFC 4648, no line breaks).
fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Dependency-free base64 decoder (RFC 4648). Whitespace is ignored and both
/// the standard and URL-safe alphabets are accepted, so `canvas.toDataURL`
/// payloads decode without a third-party crate.
fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for c in text.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => return None,
        } as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

// --------------------------------------------------------------------------- //
// dataset endpoints (extract / autolabel / review / split / check)
// --------------------------------------------------------------------------- //

/// Datasets live under `<out>/datasets/<name>/raw`, next to job outputs.
fn datasets_root(st: &AppState) -> PathBuf {
    st.jobs.out_dir().join("datasets")
}

fn dataset_dir(st: &AppState, name: &str) -> Option<PathBuf> {
    dataset::is_safe_name(name).then(|| datasets_root(st).join(name))
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AutolabelRequest {
    threshold: Option<u8>,
    min_area: Option<f32>,
    max_objects: Option<usize>,
    channels: Option<usize>,
    /// `[x1, y1, x2, y2]`, normalised 0..1.
    roi: Option<[f32; 4]>,
    class_id: Option<usize>,
    only_empty: Option<bool>,
}

impl AutolabelRequest {
    fn into_options(self) -> dataset::AutolabelOptions {
        let d = dataset::AutolabelOptions::default();
        dataset::AutolabelOptions {
            threshold: self.threshold.unwrap_or(d.threshold).min(200),
            min_area: self.min_area.unwrap_or(d.min_area).clamp(0.0, 0.9),
            max_objects: self.max_objects.unwrap_or(d.max_objects).clamp(1, 200),
            channels: self.channels.unwrap_or(0),
            roi: self.roi,
            class_id: self.class_id.unwrap_or(0),
            only_empty: self.only_empty.unwrap_or(false),
            nms_iou: d.nms_iou,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExtractDatasetRequest {
    /// File name inside the media directory.
    media: String,
    /// Dataset directory name (default: the video file stem).
    #[serde(default)]
    dataset: Option<String>,
    #[serde(default)]
    frames: Option<usize>,
    #[serde(default)]
    fps: Option<f64>,
    #[serde(default)]
    max_side: Option<u32>,
    #[serde(default)]
    start_s: Option<f64>,
    #[serde(default)]
    end_s: Option<f64>,
    #[serde(default)]
    quality: Option<u8>,
    #[serde(default)]
    overwrite: Option<bool>,
    #[serde(default)]
    autolabel: Option<AutolabelRequest>,
}

async fn list_datasets(State(st): State<AppState>) -> Json<Vec<dataset::DatasetSummary>> {
    let root = datasets_root(&st);
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&root) {
        for entry in rd.flatten() {
            let path = entry.path();
            if !path.is_dir() || !path.join("raw").is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            out.push(dataset::summarize(&path, &name));
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Json(out)
}

async fn dataset_detail(
    State(st): State<AppState>,
    UrlPath(name): UrlPath<String>,
) -> Response {
    let Some(root) = dataset_dir(&st, &name) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid dataset name");
    };
    if !root.join("raw").is_dir() {
        return api_error(StatusCode::NOT_FOUND, "unknown dataset");
    }
    Json(dataset::summarize(&root, &name)).into_response()
}

/// `POST /api/datasets/extract` - sample frames and (optionally) pre-label.
async fn dataset_extract(
    State(st): State<AppState>,
    Json(req): Json<ExtractDatasetRequest>,
) -> Response {
    let Some(video) = safe_join(st.jobs.media_dir(), &req.media) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid media name");
    };
    if !video.is_file() {
        return api_error(
            StatusCode::NOT_FOUND,
            format!("media not found: {}", req.media),
        );
    }
    let stem = video
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let name = req
        .dataset
        .as_deref()
        .map(dataset::sanitize_name)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| dataset::sanitize_name(&stem));
    let root = datasets_root(&st).join(&name);
    let raw = root.join("raw");

    let opts = dataset::ExtractOptions {
        video,
        out_dir: raw.clone(),
        frames: req.frames.unwrap_or(300).clamp(1, 100_000),
        fps: req.fps.unwrap_or(1.0).max(0.0),
        max_side: req.max_side.unwrap_or(960).clamp(64, 3840),
        start_s: req.start_s.filter(|v| v.is_finite() && *v >= 0.0),
        end_s: req.end_s.filter(|v| v.is_finite()),
        quality: req.quality.unwrap_or(92).clamp(1, 100),
        prefix: None,
        overwrite: req.overwrite.unwrap_or(false),
    };
    let auto = req.autolabel.map(AutolabelRequest::into_options);

    let worker = tokio::task::spawn_blocking(
        move || -> anyhow::Result<(dataset::ExtractReport, Option<dataset::AutolabelReport>)> {
            let extract = dataset::extract::extract_frames(&opts)?;
            let autolabel = match auto {
                Some(o) => Some(dataset::autolabel::autolabel_dir(&raw, &o)?),
                None => None,
            };
            Ok((extract, autolabel))
        },
    )
    .await;

    match worker {
        Ok(Ok((extract, autolabel))) => {
            tracing::info!(
                dataset = %name,
                frames = extract.written,
                boxes = autolabel.as_ref().map(|a| a.boxes).unwrap_or(0),
                "dataset frames extracted"
            );
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "dataset": dataset::summarize(&root, &name),
                    "extract": extract,
                    "autolabel": autolabel,
                })),
            )
                .into_response()
        }
        Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("extract worker failed: {e}"),
        ),
    }
}

async fn dataset_frames(
    State(st): State<AppState>,
    UrlPath(name): UrlPath<String>,
) -> Response {
    let Some(root) = dataset_dir(&st, &name) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid dataset name");
    };
    let raw = root.join("raw");
    if !raw.is_dir() {
        return api_error(StatusCode::NOT_FOUND, "unknown dataset");
    }
    let has_split = root.join("data.yaml").is_file();
    let root_for_list = root.clone();
    match tokio::task::spawn_blocking(move || dataset::list_frames(&root_for_list)).await {
        Ok(Ok(frames)) => Json(serde_json::json!({
            "name": name,
            "dir": root.display().to_string(),
            "hasSplit": has_split,
            "frames": frames,
        }))
        .into_response(),
        Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("list worker failed: {e}"),
        ),
    }
}

async fn dataset_image(
    State(st): State<AppState>,
    UrlPath((name, file)): UrlPath<(String, String)>,
) -> Response {
    let Some(root) = dataset_dir(&st, &name) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid dataset name");
    };
    let Some(path) = safe_join(&root.join("raw"), &file) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid file name");
    };
    if !dataset::is_image_path(&path) {
        return api_error(StatusCode::BAD_REQUEST, "not an image file");
    }
    let mime = if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("png"))
    {
        "image/png"
    } else {
        "image/jpeg"
    };
    match tokio::fs::read(&path).await {
        Ok(bytes) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, mime),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            bytes,
        )
            .into_response(),
        Err(_) => api_error(StatusCode::NOT_FOUND, "frame not found"),
    }
}

async fn dataset_get_labels(
    State(st): State<AppState>,
    UrlPath((name, file)): UrlPath<(String, String)>,
) -> Response {
    let Some(root) = dataset_dir(&st, &name) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid dataset name");
    };
    if !file.to_ascii_lowercase().ends_with(".txt") {
        return api_error(StatusCode::BAD_REQUEST, "not a label file");
    }
    let Some(path) = safe_join(&root.join("raw"), &file) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid file name");
    };
    match dataset::read_labels(&path) {
        Ok(boxes) => Json(serde_json::json!({ "boxes": boxes })).into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
struct SaveLabelsRequest {
    boxes: Vec<dataset::LabelBox>,
}

async fn dataset_put_labels(
    State(st): State<AppState>,
    UrlPath((name, file)): UrlPath<(String, String)>,
    Json(body): Json<SaveLabelsRequest>,
) -> Response {
    let Some(root) = dataset_dir(&st, &name) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid dataset name");
    };
    if !file.to_ascii_lowercase().ends_with(".txt") {
        return api_error(StatusCode::BAD_REQUEST, "not a label file");
    }
    let Some(path) = safe_join(&root.join("raw"), &file) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid file name");
    };
    if body.boxes.len() > 500 {
        return api_error(StatusCode::BAD_REQUEST, "too many boxes (max 500)");
    }
    for (i, b) in body.boxes.iter().enumerate() {
        if !b.is_valid() {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("box {i} has invalid values: {b:?}"),
            );
        }
    }
    match dataset::write_labels(&path, &body.boxes) {
        Ok(()) => Json(serde_json::json!({
            "ok": true,
            "boxes": body.boxes.len(),
        }))
        .into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `POST /api/datasets/{name}/capture` - one frame from a job's player.
///
/// The browser sends the JPEG it grabbed from the `<video>` plus the tracked
/// boxes of that frame, so the reviewer can grow a dataset with the exact
/// frames the current model gets wrong (fine-tuning). The image is re-encoded
/// to a plain JPEG and stored next to a YOLO `.txt`, exactly like `extract`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CaptureFrameRequest {
    /// Base64 JPEG/PNG, either bare or as a `data:image/...;base64,` URL.
    image: String,
    #[serde(default)]
    boxes: Vec<dataset::LabelBox>,
    /// File-name prefix (defaults to `frame`).
    #[serde(default)]
    prefix: Option<String>,
    /// Source frame index written into the file name.
    #[serde(default)]
    frame: Option<usize>,
    /// Replace an existing image/label pair with the same name.
    #[serde(default)]
    overwrite: Option<bool>,
}

async fn dataset_capture(
    State(st): State<AppState>,
    UrlPath(name): UrlPath<String>,
    Json(req): Json<CaptureFrameRequest>,
) -> Response {
    let Some(root) = dataset_dir(&st, &name) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid dataset name");
    };
    if req.boxes.len() > 200 {
        return api_error(StatusCode::BAD_REQUEST, "too many boxes (max 200)");
    }
    for (i, b) in req.boxes.iter().enumerate() {
        if !b.is_valid() {
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("box {i} has invalid values: {b:?}"),
            );
        }
    }

    // `canvas.toDataURL("image/jpeg")` prefixes a data URL; accept both forms.
    let payload = match req.image.split_once(',') {
        Some((head, tail)) if head.starts_with("data:") => tail,
        _ => req.image.as_str(),
    };
    if payload.len() > 16 * 1024 * 1024 {
        return api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "image is too large (max 16 MB of base64)",
        );
    }
    let Some(bytes) = base64_decode(payload) else {
        return api_error(StatusCode::BAD_REQUEST, "image is not valid base64");
    };

    let prefix = dataset::sanitize_name(req.prefix.as_deref().unwrap_or("frame"));
    let index = req.frame.unwrap_or(0);
    let raw = root.join("raw");
    if let Err(e) = std::fs::create_dir_all(&raw) {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("creating {}: {e}", raw.display()),
        );
    }
    let image_path = raw.join(format!("{prefix}_{index:06}.jpg"));
    let label_path = dataset::label_path_for(&image_path);
    let existed = image_path.exists();
    let file_name = image_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    if existed && !req.overwrite.unwrap_or(false) {
        return api_error(
            StatusCode::CONFLICT,
            format!("{file_name} 已存在，勾选「覆盖同名帧」后重试"),
        );
    }

    // Re-encode instead of storing the upload verbatim: the browser picks the
    // format, but a dataset must stay a uniform JPEG set.
    let decoded = match image::load_from_memory(&bytes) {
        Ok(img) => img.to_rgb8(),
        Err(e) => return api_error(StatusCode::BAD_REQUEST, format!("unsupported image: {e}")),
    };
    let (width, height) = (decoded.width(), decoded.height());
    let mut jpeg = Vec::new();
    if let Err(e) = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 92).write_image(
        decoded.as_raw(),
        width,
        height,
        image::ExtendedColorType::Rgb8,
    ) {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("encoding jpeg: {e}"),
        );
    }
    if let Err(e) = std::fs::write(&image_path, &jpeg) {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("writing {}: {e}", image_path.display()),
        );
    }
    if let Err(e) = dataset::write_labels(&label_path, &req.boxes) {
        // Keep the pair consistent: an image without its label file would
        // silently turn into a negative sample.
        let _ = std::fs::remove_file(&image_path);
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("writing labels: {e}"),
        );
    }

    tracing::info!(
        dataset = %name,
        frame = index,
        boxes = req.boxes.len(),
        overwrite = existed,
        "frame captured into dataset"
    );
    Json(serde_json::json!({
        "dataset": dataset::summarize(&root, &name),
        "image": file_name,
        "label": label_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default(),
        "boxes": req.boxes.len(),
        "width": width,
        "height": height,
        "overwritten": existed,
    }))
    .into_response()
}

async fn dataset_autolabel(
    State(st): State<AppState>,
    UrlPath(name): UrlPath<String>,
    Json(req): Json<AutolabelRequest>,
) -> Response {
    let Some(root) = dataset_dir(&st, &name) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid dataset name");
    };
    let raw = root.join("raw");
    if !raw.is_dir() {
        return api_error(StatusCode::NOT_FOUND, "unknown dataset");
    }
    let opts = req.into_options();
    let root_for_summary = root.clone();
    match tokio::task::spawn_blocking(move || dataset::autolabel::autolabel_dir(&raw, &opts)).await
    {
        Ok(Ok(report)) => {
            tracing::info!(
                dataset = %name,
                images = report.images,
                boxes = report.boxes,
                "dataset auto-labelled"
            );
            Json(serde_json::json!({
                "report": report,
                "dataset": dataset::summarize(&root_for_summary, &name),
            }))
            .into_response()
        }
        Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("autolabel worker failed: {e}"),
        ),
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SplitRequest {
    val_frac: Option<f32>,
    seed: Option<u64>,
    class_name: Option<String>,
}

async fn dataset_split(
    State(st): State<AppState>,
    UrlPath(name): UrlPath<String>,
    Json(req): Json<SplitRequest>,
) -> Response {
    let Some(root) = dataset_dir(&st, &name) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid dataset name");
    };
    if !root.join("raw").is_dir() {
        return api_error(StatusCode::NOT_FOUND, "unknown dataset");
    }
    let opts = dataset::SplitOptions {
        raw_dir: root.join("raw"),
        out_dir: root.clone(),
        val_frac: req.val_frac.unwrap_or(0.2).clamp(0.01, 0.9),
        seed: req.seed.unwrap_or(0),
        class_name: req.class_name.unwrap_or_else(|| "mouse".to_string()),
    };
    let check_root = root.clone();
    let worker = tokio::task::spawn_blocking(
        move || -> anyhow::Result<(dataset::SplitReport, dataset::CheckReport)> {
            let split = dataset::split::split_dataset(&opts)?;
            let check = dataset::split::check_dataset(&check_root)?;
            Ok((split, check))
        },
    )
    .await;
    match worker {
        Ok(Ok((split, check))) => {
            tracing::info!(dataset = %name, train = split.train, val = split.val, "dataset split");
            Json(serde_json::json!({ "split": split, "check": check })).into_response()
        }
        Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("split worker failed: {e}"),
        ),
    }
}

/// `POST /api/datasets/{name}/assign` - move frames into train/val (or out).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AssignSplitRequest {
    /// Image file names inside `raw/` (`TST_000012.jpg`).
    frames: Vec<String>,
    /// `"train"`, `"val"` or `"none"` (remove from both subsets).
    target: String,
}

async fn dataset_assign(
    State(st): State<AppState>,
    UrlPath(name): UrlPath<String>,
    Json(req): Json<AssignSplitRequest>,
) -> Response {
    let Some(root) = dataset_dir(&st, &name) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid dataset name");
    };
    if !root.join("raw").is_dir() {
        return api_error(StatusCode::NOT_FOUND, "unknown dataset");
    }
    if req.frames.len() > 500 {
        return api_error(StatusCode::BAD_REQUEST, "too many frames (max 500)");
    }
    let target = match dataset::split::AssignTarget::parse(&req.target) {
        Ok(t) => t,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let summary_root = root.clone();
    let worker = tokio::task::spawn_blocking(move || {
        dataset::split::assign_frames(&root, &req.frames, target)
    })
    .await;
    match worker {
        Ok(Ok(report)) => {
            tracing::info!(
                dataset = %name,
                target = %report.split,
                files = report.files,
                "dataset frames reassigned"
            );
            Json(serde_json::json!({
                "assign": report,
                "dataset": dataset::summarize(&summary_root, &name),
            }))
            .into_response()
        }
        Ok(Err(e)) => api_error(StatusCode::BAD_REQUEST, e.to_string()),
        Err(e) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("assign worker failed: {e}"),
        ),
    }
}

async fn dataset_check(
    State(st): State<AppState>,
    UrlPath(name): UrlPath<String>,
) -> Response {
    let Some(root) = dataset_dir(&st, &name) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid dataset name");
    };
    match tokio::task::spawn_blocking(move || dataset::split::check_dataset(&root)).await {
        Ok(Ok(report)) => Json(report).into_response(),
        Ok(Err(e)) => api_error(StatusCode::BAD_REQUEST, e.to_string()),
        Err(e) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("check worker failed: {e}"),
        ),
    }
}

async fn dataset_delete(
    State(st): State<AppState>,
    UrlPath(name): UrlPath<String>,
) -> Response {
    let Some(root) = dataset_dir(&st, &name) else {
        return api_error(StatusCode::BAD_REQUEST, "invalid dataset name");
    };
    if !root.is_dir() {
        return api_error(StatusCode::NOT_FOUND, "unknown dataset");
    }
    match std::fs::remove_dir_all(&root) {
        Ok(()) => {
            tracing::info!(dataset = %name, "dataset deleted");
            Json(serde_json::json!({ "ok": true })).into_response()
        }
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

// --------------------------------------------------------------------------- //
// router
// --------------------------------------------------------------------------- //

pub fn router(state: AppState, frontend_dist: Option<&Path>) -> Router {
    let api = Router::new()
        .route("/info", get(info))
        .route("/models", get(list_models))
        .route("/media", get(list_media))
        .route("/media/{name}", get(media_file))
        .route("/datasets", get(list_datasets))
        .route("/datasets/extract", post(dataset_extract))
        .route("/datasets/{name}", get(dataset_detail).delete(dataset_delete))
        .route("/datasets/{name}/frames", get(dataset_frames))
        // A captured frame is a base64 JPEG, so allow more than axum's 2 MB
        // default JSON body.
        .route(
            "/datasets/{name}/capture",
            post(dataset_capture).layer(DefaultBodyLimit::max(24 * 1024 * 1024)),
        )
        .route("/datasets/{name}/image/{file}", get(dataset_image))
        .route(
            "/datasets/{name}/labels/{file}",
            get(dataset_get_labels).put(dataset_put_labels),
        )
        .route("/datasets/{name}/autolabel", post(dataset_autolabel))
        .route("/datasets/{name}/split", post(dataset_split))
        .route("/datasets/{name}/assign", post(dataset_assign))
        .route("/datasets/{name}/check", post(dataset_check))
        .route("/jobs", get(list_jobs).post(create_job))
        .route("/jobs/{id}", get(get_job).delete(cancel_job))
        .route("/jobs/{id}/cancel", post(cancel_job))
        .route("/jobs/{id}/rescore", post(rescore_job))
        .route("/jobs/{id}/events", get(job_events))
        .route("/jobs/{id}/frame", get(latest_frame))
        .route("/jobs/{id}/frames", get(frame_stream))
        .route("/jobs/{id}/records", get(job_records))
        .route("/jobs/{id}/windows", get(job_windows))
        .route("/jobs/{id}/window-events", get(job_window_events))
        .route("/jobs/{id}/logs", get(job_logs))
        .route("/jobs/{id}/video", get(job_video))
        .with_state(state);

    let mut app = Router::new().nest("/api", api);

    // Priority: an explicit on-disk build (`--frontend <dist>`) wins, so UI
    // tweaks can be shipped without recompiling; otherwise use the frontend
    // embedded into the binary by `build.rs`.
    let on_disk = frontend_dist.filter(|dist| dist.join("index.html").is_file());
    match on_disk {
        Some(dist) => {
            tracing::info!(path = %dist.display(), "serving frontend from disk");
            app = app
                .nest_service(
                    "/assets",
                    tower_http::services::ServeDir::new(dist.join("assets")),
                )
                // Client-side routes (React Router) fall back to index.html.
                .fallback_service(
                    tower_http::services::ServeDir::new(dist)
                        .append_index_html_on_directories(true)
                        .fallback(axum::routing::any_service(
                            tower_http::services::ServeFile::new(dist.join("index.html")),
                        )),
                );
        }
        None if crate::web::assets::is_embedded() => {
            tracing::info!(
                files = crate::web::assets::count(),
                "serving frontend embedded at compile time"
            );
            app = app.fallback(embedded_asset);
        }
        None => {
            tracing::warn!(
                "frontend build missing - run `npm --prefix frontend install && \
                 npm --prefix frontend run build`, then rebuild; serving API only"
            );
        }
    }

    app.layer(
        CorsLayer::new()
            .allow_origin(tower_http::cors::Any)
            .allow_methods(tower_http::cors::Any)
            .allow_headers(tower_http::cors::Any),
    )
}

/// Serve a file from the frontend embedded at compile time.
///
/// Unknown paths fall back to `index.html` so client-side routes work on a
/// hard refresh. Hashed `/assets/*` files are immutable; `index.html` must be
/// revalidated so an upgraded binary is picked up.
async fn embedded_asset(uri: axum::http::Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    let (body, served, immutable) = match crate::web::assets::get(path) {
        Some(bytes) => (bytes, path, path.starts_with("assets/")),
        None => match crate::web::assets::get("index.html") {
            Some(bytes) => (bytes, "index.html", false),
            None => return StatusCode::NOT_FOUND.into_response(),
        },
    };

    let cache = if immutable {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, crate::web::assets::mime_type(served)),
            (header::CACHE_CONTROL, cache),
        ],
        Body::from(body),
    )
        .into_response()
}

/// Start the HTTP server (blocking until shutdown).
pub async fn serve(
    addr: std::net::SocketAddr,
    state: AppState,
    frontend_dist: Option<PathBuf>,
) -> anyhow::Result<()> {
    let app = router(state, frontend_dist.as_deref());
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    // Report the address the OS actually bound (resolves `:0` to a real port).
    let local = listener.local_addr().unwrap_or(addr);
    tracing::info!("listening on http://{local}");
    axum::serve(listener, app).await.context("axum server")?;
    Ok(())
}
