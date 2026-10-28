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

use rodent_act::{
    classify, dataset, detect, draw, pipeline, pipeline::PipelineConfig, track, video_io,
};

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

        /// Start of the analysed segment, in seconds.
        #[arg(long)]
        start: Option<f64>,

        /// End of the analysed segment, in seconds (defaults to end of file).
        /// The segment is decoded in place; no clipped file is written.
        #[arg(long)]
        end: Option<f64>,

        /// Print an ASCII preview every N frames.
        #[arg(long)]
        preview_every: Option<usize>,

        /// ONNX Runtime intra-op threads.
        #[arg(long, default_value_t = 2)]
        threads: usize,

        /// Compute device for `--detector yolo`: `cpu`, `cuda` or `auto`.
        /// `cuda` requires a build with `--features cuda` and a GPU ONNX
        /// Runtime library; it errors out instead of silently using the CPU.
        #[arg(long, default_value = "auto")]
        device: detect::device::Device,

        /// CUDA device ordinal (only with `--device cuda`).
        #[arg(long, default_value_t = 0)]
        gpu_device: i32,
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

        /// Decision threshold on `immobile_prob` (default 0.5): windows above it
        /// are labelled immobile.
        #[arg(long, default_value_t = 0.5)]
        threshold: f32,

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

        /// YOLO detection score threshold. Fine-tuned TST models are usually
        /// under-confident (peaks around 0.2-0.4), so lower this (e.g. 0.05)
        /// if tracks come out fragmented or empty.
        #[arg(long, default_value_t = 0.2)]
        score_thr: f32,

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

        /// Compute device for `--detector yolo`: `cpu`, `cuda` or `auto`.
        #[arg(long, default_value = "auto")]
        device: detect::device::Device,

        /// CUDA device ordinal (only with `--device cuda`).
        #[arg(long, default_value_t = 0)]
        gpu_device: i32,

        /// Only process the first N frames.
        #[arg(long)]
        max_frames: Option<usize>,

        /// Start of the analysed segment, in seconds.
        #[arg(long)]
        start: Option<f64>,

        /// End of the analysed segment, in seconds (defaults to end of file).
        #[arg(long)]
        end: Option<f64>,

        /// Draw the analysis (per-window state) into the video.
        #[arg(long, default_value_t = true)]
        annotate: bool,
    },

    /// Re-score windows from an existing tracking JSON (no video decoding).
    ///
    /// Only feature extraction + classification run, so `--window` / `--stride`
    /// / `--clf` can be explored without re-running detection and tracking.
    Rescore {
        /// JSON written by `track --json` (or `out/web/<job>/tracking.json`).
        #[arg(long)]
        json: PathBuf,

        /// Where to write the per-window CSV (features + prediction).
        #[arg(long, default_value = "out/windows.csv")]
        windows: PathBuf,

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

        /// Decision threshold on `immobile_prob` (default 0.5).
        #[arg(long, default_value_t = 0.5)]
        threshold: f32,
    },

    /// Start the web UI (Axum API + React frontend) for interactive analysis.
    Serve {
        /// Address to bind (0.0.0.0 listens on all interfaces).
        #[arg(long, default_value = "0.0.0.0:8080")]
        addr: String,

        /// Directory scanned for input videos (shown in the picker).
        #[arg(long, default_value = "media")]
        media_dir: PathBuf,

        /// Directory for job outputs (frames, CSV, optional preview video).
        #[arg(long, default_value = "out")]
        out_dir: PathBuf,

        /// Optional override for the built frontend (Vite `dist`) served from
        /// disk. By default the build embedded at compile time is used, so the
        /// binary is self-contained; pass this to test a UI change without
        /// recompiling.
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

    /// Build a YOLO dataset from a video (extract -> autolabel -> split).
    ///
    /// Pure-Rust replacement for `analytics/make_dataset.py`; the same folders
    /// can then be reviewed in the web UI (`rodent-act serve` -> 数据集).
    Dataset {
        #[command(subcommand)]
        cmd: DatasetCommand,
    },
}

#[derive(Subcommand, Debug)]
enum DatasetCommand {
    /// Sample frames from a video into a YOLO-format folder.
    Extract {
        /// Input video.
        #[arg(long)]
        video: PathBuf,

        /// Output `raw/` folder (images + empty .txt labels).
        #[arg(long)]
        out: PathBuf,

        /// Maximum number of sampled frames.
        #[arg(long, default_value_t = 300)]
        frames: usize,

        /// Sampling rate in frames per second (0 = spread `--frames` evenly).
        #[arg(long, default_value_t = 1.0)]
        fps: f64,

        /// Longest side of the stored JPEGs.
        #[arg(long, default_value_t = 960)]
        max_side: u32,

        /// Start of the sampled segment, in source seconds.
        #[arg(long)]
        start: Option<f64>,

        /// End of the sampled segment, in source seconds.
        #[arg(long)]
        end: Option<f64>,

        /// JPEG quality (1-100).
        #[arg(long, default_value_t = 92)]
        quality: u8,

        /// File-name prefix (default: the video file stem).
        #[arg(long)]
        prefix: Option<String>,

        /// Re-encode images that already exist (labels are always kept).
        #[arg(long)]
        overwrite: bool,

        /// Run `autolabel` on the extracted frames afterwards.
        #[arg(long)]
        autolabel: bool,

        /// With `--autolabel`: keep at most N boxes per frame (0 = unlimited).
        #[arg(long, default_value_t = 0)]
        channels: usize,

        /// With `--autolabel`: keep only boxes centred inside x1,y1,x2,y2
        /// (normalised 0..1).
        #[arg(long)]
        roi: Option<String>,
    },

    /// Pre-fill YOLO labels with dark-region detection.
    Autolabel {
        /// Folder with images (e.g. `dataset/raw`).
        #[arg(long)]
        raw: PathBuf,

        /// Darkness threshold below the frame median.
        #[arg(long, default_value_t = 45)]
        thresh: u8,

        /// Minimum blob area as a fraction of the frame.
        #[arg(long, default_value_t = 0.0008)]
        min_area: f32,

        /// Maximum candidate boxes per frame.
        #[arg(long, default_value_t = 12)]
        max_objects: usize,

        /// Keep at most N boxes per frame (0 = unlimited).
        #[arg(long, default_value_t = 0)]
        channels: usize,

        /// Keep only boxes centred inside x1,y1,x2,y2 (normalised 0..1).
        #[arg(long)]
        roi: Option<String>,

        /// Class index written to the labels.
        #[arg(long, default_value_t = 0)]
        class_id: usize,

        /// Skip frames that already have boxes.
        #[arg(long)]
        only_empty: bool,
    },

    /// Copy a reviewed `raw/` folder into images/labels/{train,val} + data.yaml.
    Split {
        #[arg(long)]
        raw: PathBuf,

        /// Dataset root (receives `images/`, `labels/`, `data.yaml`).
        #[arg(long)]
        out: PathBuf,

        /// Fraction of frames used for validation.
        #[arg(long, default_value_t = 0.2)]
        val_frac: f32,

        #[arg(long, default_value_t = 0)]
        seed: u64,

        /// Class name written to `data.yaml`.
        #[arg(long, default_value = "mouse")]
        class_name: String,
    },

    /// Validate a dataset folder (split layout if present, else `raw/`).
    Check {
        #[arg(long)]
        dataset: PathBuf,
    },
}

/// Print a one-line summary plus per-track window counts for an analysis report.
///
/// `immobile` / `mobile` counts follow the report's decision threshold, so
/// `--threshold` is visible in the output instead of being silently applied.
fn print_analysis(report: &pipeline::AnalyzeReport) {
    let immobile = report
        .windows
        .iter()
        .filter(|w| w.label == classify::ActClass::Immobile)
        .count();
    println!(
        "tracks={} windows={} immobile_windows={} mobile_windows={} threshold={:.2} \
         mean_immobile_prob={:.3}",
        report.n_tracks,
        report.windows.len(),
        immobile,
        report.windows.len() - immobile,
        report.threshold,
        report.mean_immobile_prob
    );
    for s in &report.per_track_stats {
        println!(
            "  track {}: immobile {}/{} ({:.1}%, {:.2}s, longest {:.2}s), mobile {} ({:.2}s), \
             covered {:.2}s",
            s.track_id,
            s.immobile_windows,
            s.n_windows,
            s.immobile_fraction * 100.0,
            s.immobile_s,
            s.longest_immobile_s,
            s.mobile_windows,
            s.mobile_s,
            s.covered_s
        );
    }
}

/// Build a clip from `--start`/`--end`, rejecting inverted or empty windows.
fn parse_clip(start: Option<f64>, end: Option<f64>) -> anyhow::Result<Option<video_io::Clip>> {
    let start = start.unwrap_or(0.0);
    if !start.is_finite() || start < 0.0 {
        anyhow::bail!("--start must be a non-negative number of seconds (got {start})");
    }
    match end {
        Some(end) if !end.is_finite() => {
            anyhow::bail!("--end must be a finite number of seconds (got {end})")
        }
        Some(end) if end <= start => {
            anyhow::bail!("--end ({end}s) must be after --start ({start}s)")
        }
        Some(end) => Ok(Some(video_io::Clip {
            start_s: start,
            end_s: Some(end),
        })),
        None if start > 0.0 => Ok(Some(video_io::Clip {
            start_s: start,
            end_s: None,
        })),
        None => Ok(None),
    }
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
            start,
            end,
            preview_every,
            threads,
            device,
            gpu_device,
        } => {
            let cfg = PipelineConfig {
                input,
                clip: parse_clip(start, end)?,
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
                device,
                gpu_device_id: gpu_device,
                max_frames,
                preview_every,
                live_scoring: None,
            };
            let info = pipeline::run(&cfg)?;
            println!(
                "frames={} detect_calls={} detections={} tracks={} device={}",
                info.frames,
                info.detect_calls,
                info.detections,
                info.tracks.len(),
                info.device
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
            threshold,
            detector,
            model,
            class_name,
            max_side,
            score_thr,
            blob_threshold,
            reference,
            min_hits,
            max_age,
            device,
            gpu_device,
            max_frames,
            start,
            end,
            annotate,
        } => {
            let track = PipelineConfig {
                input,
                clip: parse_clip(start, end)?,
                output_video: annotate.then_some(output),
                output_json: None,
                output_csv: None,
                detector,
                model,
                max_side,
                score_thr,
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
                device,
                gpu_device_id: gpu_device,
                max_frames,
                ..Default::default()
            };
            let analyze_cfg = pipeline::AnalyzeConfig {
                classifier: clf,
                window_s: window,
                stride_s: stride.unwrap_or(window * 0.5),
                threshold,
            };
            let report = pipeline::analyze(&track, &analyze_cfg)?;
            pipeline::write_window_csv(&windows, &report.feature_names, &report.windows)?;
            print_analysis(&report);
        }
        Command::Rescore {
            json,
            windows,
            clf,
            window,
            stride,
            threshold,
        } => {
            let file = std::fs::File::open(&json)
                .map_err(|e| anyhow::anyhow!("opening {}: {e}", json.display()))?;
            let info: pipeline::RunInfo = serde_json::from_reader(std::io::BufReader::new(file))
                .map_err(|e| anyhow::anyhow!("parsing {}: {e}", json.display()))?;
            let analyze_cfg = pipeline::AnalyzeConfig {
                classifier: clf,
                window_s: window,
                stride_s: stride.unwrap_or(window * 0.5),
                threshold,
            };
            let report = pipeline::score_tracks(&info, &analyze_cfg)?;
            pipeline::write_window_csv(&windows, &report.feature_names, &report.windows)?;
            print_analysis(&report);
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
        Command::Dataset { cmd } => match cmd {
            DatasetCommand::Extract {
                video,
                out,
                frames,
                fps,
                max_side,
                start,
                end,
                quality,
                prefix,
                overwrite,
                autolabel,
                channels,
                roi,
            } => {
                let opts = dataset::ExtractOptions {
                    video,
                    out_dir: out.clone(),
                    frames,
                    fps,
                    max_side,
                    start_s: start,
                    end_s: end,
                    quality,
                    prefix,
                    overwrite,
                };
                let report = dataset::extract::extract_frames(&opts)?;
                println!(
                    "wrote {} frames to {} ({}x{}, {} fps source, step={} frames, skipped={})",
                    report.written,
                    report.out_dir,
                    report.width,
                    report.height,
                    report.fps,
                    report.step,
                    report.skipped
                );
                if start.is_some() || end.is_some() {
                    println!(
                        "clip: {} .. {}",
                        start
                            .map(|s| format!("{s:.2}s"))
                            .unwrap_or_else(|| "0.00s".into()),
                        end.map(|e| format!("{e:.2}s"))
                            .unwrap_or_else(|| "end".into())
                    );
                }
                if autolabel {
                    let mut auto = dataset::AutolabelOptions {
                        channels,
                        ..Default::default()
                    };
                    if let Some(spec) = roi {
                        auto.roi = Some(dataset::parse_roi(&spec)?);
                    }
                    let rep = dataset::autolabel::autolabel_dir(&out, &auto)?;
                    println!(
                        "pre-labelled {} images, {} boxes ({} empty, {} skipped)",
                        rep.images, rep.boxes, rep.empty, rep.skipped
                    );
                }
                println!("next: review the labels in `rodent-act serve` (数据集), then split");
            }
            DatasetCommand::Autolabel {
                raw,
                thresh,
                min_area,
                max_objects,
                channels,
                roi,
                class_id,
                only_empty,
            } => {
                let opts = dataset::AutolabelOptions {
                    threshold: thresh,
                    min_area,
                    max_objects,
                    channels,
                    roi: roi.as_deref().map(dataset::parse_roi).transpose()?,
                    class_id,
                    only_empty,
                    ..Default::default()
                };
                let rep = dataset::autolabel::autolabel_dir(&raw, &opts)?;
                println!(
                    "pre-labelled {} images, {} boxes ({} empty, {} skipped)",
                    rep.images, rep.boxes, rep.empty, rep.skipped
                );
                println!("pre-labels are a starting point: correct them before training");
            }
            DatasetCommand::Split {
                raw,
                out,
                val_frac,
                seed,
                class_name,
            } => {
                let opts = dataset::SplitOptions {
                    raw_dir: raw,
                    out_dir: out,
                    val_frac,
                    seed,
                    class_name,
                };
                let rep = dataset::split::split_dataset(&opts)?;
                println!("train={} val={}", rep.train, rep.val);
                println!("wrote {}", rep.data_yaml);
            }
            DatasetCommand::Check { dataset } => {
                let rep = dataset::split::check_dataset(&dataset)?;
                println!(
                    "scope={} images={} labels={} boxes={} empty={} problems={}",
                    rep.scope,
                    rep.images,
                    rep.labeled,
                    rep.boxes,
                    rep.empty_images,
                    rep.problems.len()
                );
                for p in rep.problems.iter().take(50) {
                    println!("  {p}");
                }
                for w in &rep.warnings {
                    println!("WARNING: {w}");
                }
                if !rep.ok {
                    anyhow::bail!("dataset check failed");
                }
            }
        },
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
        jobs: {
            let jobs = std::sync::Arc::new(rodent_act::web::JobManager::new(out_dir, media_dir));
            // Jobs live in memory; reopen the ones persisted by a previous run
            // so a restart no longer empties the history.
            jobs.load_existing();
            jobs
        },
        log_tx,
    };
    // The embedded copy is the default; `--frontend` only overrides it when the
    // directory actually contains a build.
    let frontend = frontend.join("index.html").is_file().then_some(frontend);
    if frontend.is_none() {
        if rodent_act::web::assets::is_embedded() {
            tracing::info!(
                files = rodent_act::web::assets::count(),
                "using frontend embedded in the binary"
            );
        } else {
            tracing::warn!(
                "no embedded frontend and no `--frontend` build - the UI is unavailable"
            );
        }
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(rodent_act::web::serve(addr, state, frontend))
}
