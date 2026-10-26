//! `rodent-act` — mouse tail-suspension (TST) video analysis.
//!
//! Step 1 (this milestone): multi-object detection + tracking + annotated
//! preview video. Later steps add 0.5 s window features and XGBoost
//! classification.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

use rodent_act::{classify, draw, pipeline, pipeline::PipelineConfig, track, video_io};

#[derive(Parser, Debug)]
#[command(
    name = "rodent-act",
    version,
    about = "Mouse tail-suspension test analyser (detection + tracking)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Detect and track mice in a video, writing an annotated preview.
    Track {
        /// Input video file.
        #[arg(short, long)]
        input: PathBuf,

        /// Annotated output video (mp4/h264).
        #[arg(short, long, default_value = "out/tracked.mp4")]
        output: PathBuf,

        /// Optional JSON dump of all track points.
        #[arg(long)]
        json: Option<PathBuf>,

        /// Optional CSV dump of all track points.
        #[arg(long)]
        csv: Option<PathBuf>,

        /// YOLO ONNX model path (only used with `--detector yolo`).
        #[arg(long, default_value = "models/yolo11n.onnx")]
        model: PathBuf,

        /// Detection backend.
        #[arg(long, value_enum, default_value_t = pipeline::DetectorKind::Blob)]
        detector: pipeline::DetectorKind,

        /// Blob backend: gray contrast threshold (raise for dim footage).
        #[arg(long, default_value_t = 22)]
        blob_threshold: u8,

        /// Blob backend: minimum object area as a fraction of the frame.
        #[arg(long, default_value_t = 0.0015)]
        blob_min_area: f32,

        /// Class name to keep, matched against the model's class list
        /// (e.g. `mouse`, `animal` for a fine-tuned model, or `any` for all).
        /// Only used with `--detector yolo`.
        #[arg(long, default_value = "mouse")]
        class_name: String,

        /// Blob backend: EMPTY enclosure (no animals) for background
        /// subtraction. Can be a still image (recommended) or `frame:N` to use
        /// frame N of the analysed video. Without it the temporal background
        /// is used, which cannot see animals that never move.
        #[arg(long)]
        reference: Option<String>,

        /// Longest side used for processing/preview (frames are resized).
        #[arg(long, default_value_t = 640)]
        max_side: u32,

        /// Detection score threshold (YOLO backend).
        #[arg(long, default_value_t = 0.2)]
        score_thr: f32,

        /// NMS IoU threshold (YOLO backend).
        #[arg(long, default_value_t = 0.45)]
        iou_thr: f32,

        /// If `--class-name mouse` yields nothing (stock COCO model), widen to
        /// generic animal classes instead of producing zero tracks.
        #[arg(long, default_value_t = true)]
        fallback_animal: bool,

        /// Minimum frames between CNN evaluations while the scene moves.
        #[arg(long, default_value_t = 1)]
        detect_every: usize,

        /// Force a full-frame detection every N frames.
        #[arg(long, default_value_t = 90)]
        refresh_every: usize,

        /// Frames a track survives without a detection.
        #[arg(long, default_value_t = 30)]
        max_age: usize,

        /// Detections needed before a track is reported.
        #[arg(long, default_value_t = 3)]
        min_hits: usize,

        /// Draw motion ROIs (debug).
        #[arg(long)]
        show_rois: bool,

        /// Disable trajectory trails.
        #[arg(long)]
        no_trails: bool,

        /// Do not write the annotated video.
        #[arg(long)]
        no_video: bool,

        /// Only process the first N frames (quick tests).
        #[arg(long)]
        max_frames: Option<usize>,

        /// Print an ASCII preview every N frames.
        #[arg(long)]
        preview_every: Option<usize>,

        /// ONNX Runtime intra-op threads.
        #[arg(long, default_value_t = 2)]
        threads: usize,
    },

    /// Info about a video (resolution, fps, frame count).
    Probe { input: PathBuf },

    /// Detect + track, then score 0.5 s immobility windows.
    Analyze {
        /// Input video file.
        #[arg(short, long)]
        input: PathBuf,

        /// Where to write per-window CSV (features + prediction).
        #[arg(long, default_value = "out/windows.csv")]
        windows: PathBuf,

        /// Where to write the annotated preview video.
        #[arg(short, long, default_value = "out/analyzed.mp4")]
        output: PathBuf,

        /// Optional XGBoost JSON model. Without it a speed-threshold baseline
        /// is used.
        #[arg(long)]
        clf: Option<PathBuf>,

        /// Window length in seconds.
        #[arg(long, default_value_t = 0.5)]
        window: f64,

        /// Window stride in seconds (defaults to half the window => 50 % overlap).
        #[arg(long)]
        stride: Option<f64>,

        /// Detection backend.
        #[arg(long, value_enum, default_value_t = pipeline::DetectorKind::Blob)]
        detector: pipeline::DetectorKind,

        /// YOLO ONNX model path (only with `--detector yolo`).
        #[arg(long, default_value = "models/yolo11n.onnx")]
        model: PathBuf,

        /// Class name to keep with `--detector yolo`.
        #[arg(long, default_value = "mouse")]
        class_name: String,

        /// Longest side used for processing.
        #[arg(long, default_value_t = 640)]
        max_side: u32,

        /// Blob backend: gray contrast threshold.
        #[arg(long, default_value_t = 22)]
        blob_threshold: u8,

        /// Blob backend: EMPTY enclosure reference, either an image path or
        /// `frame:N` to use frame N of the analysed video.
        #[arg(long)]
        reference: Option<String>,

        /// Detections needed before a track is reported.
        #[arg(long, default_value_t = 5)]
        min_hits: usize,

        /// Frames a track survives without a detection.
        #[arg(long, default_value_t = 30)]
        max_age: usize,

        /// Only process the first N frames.
        #[arg(long)]
        max_frames: Option<usize>,

        /// Draw the analysis (per-window state) into the video.
        #[arg(long, default_value_t = true)]
        annotate: bool,
    },

    /// Start the web UI (Axum API + React frontend) for interactive analysis.
    Serve {
        /// Address to bind.
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,

        /// Directory scanned for input videos (shown in the picker).
        #[arg(long, default_value = "media")]
        media_dir: PathBuf,

        /// Directory for job outputs (frames, CSV, optional preview video).
        #[arg(long, default_value = "out")]
        out_dir: PathBuf,

        /// Built frontend (Vite `dist`). Skipped when missing - use the dev
        /// server (`npm run dev`, proxying /api here) instead.
        #[arg(long, default_value = "frontend/dist")]
        frontend: PathBuf,
    },

    /// Write a PNG preview frame from an input video (no tracking).
    Snapshot {
        input: PathBuf,
        #[arg(long, default_value_t = 0)]
        frame: usize,
        #[arg(long, default_value = "out/snapshot.png")]
        output: PathBuf,
        #[arg(long, default_value_t = 960)]
        max_side: u32,
    },
}

/// `frame:12` -> use video frame 12, anything else is treated as an image path.
fn parse_reference(spec: String) -> pipeline::Reference {
    if let Some(idx) = spec.strip_prefix("frame:") {
        if let Ok(n) = idx.trim().parse::<usize>() {
            return pipeline::Reference::Frame(n);
        }
    }
    pipeline::Reference::Image(PathBuf::from(spec))
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // `serve` installs its own registry (fmt + SSE log layer) before any
    // pipeline code logs, so it must not go through the CLI subscriber below.
    if let Command::Serve {
        addr,
        media_dir,
        out_dir,
        frontend,
    } = cli.command
    {
        return run_server(addr, media_dir, out_dir, frontend);
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .compact()
        .init();

    match cli.command {
        Command::Track {
            input,
            output,
            json,
            csv,
            model,
            max_side,
            score_thr,
            iou_thr,
            class_name,
            fallback_animal,
            detector,
            blob_threshold,
            blob_min_area,
            reference,
            detect_every,
            refresh_every,
            max_age,
            min_hits,
            show_rois,
            no_trails,
            no_video,
            max_frames,
            preview_every,
            threads,
        } => {
            let cfg = PipelineConfig {
                input,
                output_video: Some(output),
                output_json: json,
                output_csv: csv,
                model,
                max_side,
                score_thr,
                iou_thr,
                detector,
                blob_threshold,
                blob_min_area,
                reference: reference.map(parse_reference),
                class_name,
                fallback_animal,
                detect_every,
                refresh_every,
                tracker: track::TrackerConfig {
                    max_age,
                    min_hits,
                    ..Default::default()
                },
                write_video: !no_video,
                show_rois,
                show_trails: !no_trails,
                threads,
                max_frames,
                preview_every,
            };
            let info = pipeline::run(&cfg)?;
            println!(
                "frames={} detect_calls={} detections={} tracks={}",
                info.frames,
                info.detect_calls,
                info.detections,
                info.tracks.len()
            );
        }
        Command::Probe { input } => {
            let (w, h, fps, n) = video_io::probe_video(&input)?;
            println!(
                "{w}x{h} @ {fps:.3} fps, frames~{n}, duration~{:.2}s",
                n as f64 / fps
            );
        }
        Command::Analyze {
            input,
            windows,
            output,
            clf,
            window,
            stride,
            detector,
            model,
            class_name,
            max_side,
            blob_threshold,
            reference,
            min_hits,
            max_age,
            max_frames,
            annotate,
        } => {
            let track = PipelineConfig {
                input,
                output_video: annotate.then_some(output),
                output_json: None,
                output_csv: None,
                detector,
                model,
                max_side,
                blob_threshold,
                reference: reference.map(parse_reference),
                class_name,
                detect_every: 1,
                refresh_every: 90,
                tracker: track::TrackerConfig {
                    min_hits,
                    max_age,
                    ..Default::default()
                },
                write_video: annotate,
                threads: 2,
                max_frames,
                ..Default::default()
            };
            let analyze_cfg = pipeline::AnalyzeConfig {
                classifier: clf,
                window_s: window,
                stride_s: stride.unwrap_or(window * 0.5),
            };
            let report = pipeline::analyze(&track, &analyze_cfg)?;
            pipeline::write_window_csv(&windows, &report.feature_names, &report.windows)?;
            println!(
                "tracks={} windows={} immobile_windows={} mean_immobile_prob={:.3}",
                report.n_tracks,
                report.windows.len(),
                report
                    .windows
                    .iter()
                    .filter(|w| w.label == classify::ActClass::Immobile)
                    .count(),
                report.mean_immobile_prob
            );
            for (id, frac) in &report.per_track_immobile_fraction {
                println!("  track {id}: immobile {:.1}%", frac * 100.0);
            }
        }
        Command::Serve { .. } => unreachable!("handled before the CLI subscriber"),
        Command::Snapshot {
            input,
            frame,
            output,
            max_side,
        } => {
            let mut reader = video_io::VideoReader::open(&input, max_side)?;
            let f = reader
                .skip_to(frame)?
                .ok_or_else(|| anyhow::anyhow!("frame {frame} beyond end of video"))?;
            if let Some(p) = output.parent() {
                std::fs::create_dir_all(p).ok();
            }
            draw::save_preview_png(
                output.to_str().unwrap_or("out/snapshot.png"),
                &f.data,
                f.width,
                f.height,
            )?;
            println!("wrote {}", output.display());
        }
    }
    Ok(())
}

/// Start the Axum server (web UI). Installs the log layer *before* any
/// pipeline code runs, so the browser receives every tracing event.
fn run_server(
    addr: String,
    media_dir: PathBuf,
    out_dir: PathBuf,
    frontend: PathBuf,
) -> anyhow::Result<()> {
    let addr: std::net::SocketAddr = addr
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid --addr `{addr}`: {e}"))?;
    let (log_tx, _) = tokio::sync::broadcast::channel(1024);
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,ort=warn")))
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_writer(std::io::stderr)
                .compact(),
        )
        .with(rodent_act::web::telemetry::SseLayer::new(
            log_tx.clone(),
            tracing::Level::INFO,
        ))
        .init();

    std::fs::create_dir_all(&out_dir).ok();

    let state = rodent_act::web::AppState {
        jobs: std::sync::Arc::new(rodent_act::web::JobManager::new(out_dir, media_dir)),
        log_tx,
    };
    let frontend = frontend.exists().then_some(frontend);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(rodent_act::web::serve(addr, state, frontend))
}
