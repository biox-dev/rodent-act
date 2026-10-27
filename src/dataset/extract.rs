//! Step 1 of the dataset workflow: sample frames out of a video.
//!
//! Frames are decoded once through [`VideoReader`] (ffmpeg -> raw rgb24) and
//! written as JPEGs, one empty YOLO label file next to each image so the
//! folder is immediately a valid dataset. Existing label files are never
//! truncated, which makes re-running the extraction safe after hand-editing.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use image::ImageEncoder as _;
use serde::Serialize;

use crate::video_io::{Clip, VideoReader};

use super::{is_image_path, label_path_for, sanitize_name};

#[derive(Debug, Clone)]
pub struct ExtractOptions {
    pub video: PathBuf,
    /// Dataset `raw/` directory (created if missing).
    pub out_dir: PathBuf,
    /// Maximum number of sampled frames.
    pub frames: usize,
    /// Sampling rate in frames per second (`0` = spread `frames` evenly).
    pub fps: f64,
    /// Longest side of the stored images (never upscales).
    pub max_side: u32,
    /// Source-time start of the sampled segment, in seconds.
    pub start_s: Option<f64>,
    /// Source-time end of the sampled segment, in seconds.
    pub end_s: Option<f64>,
    /// JPEG quality (1..=100).
    pub quality: u8,
    /// File-name prefix (default: the video file stem).
    pub prefix: Option<String>,
    /// Re-encode images that already exist.
    pub overwrite: bool,
}

impl ExtractOptions {
    pub fn clip(&self) -> Option<Clip> {
        let start = self.start_s.unwrap_or(0.0).max(0.0);
        match self.end_s {
            Some(end) if end > start => Some(Clip::new(start, Some(end))),
            _ if start > 0.0 => Some(Clip::new(start, None)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtractReport {
    pub out_dir: String,
    pub written: usize,
    /// Existing images skipped because `overwrite` was false.
    pub skipped: usize,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    /// Frames between two samples.
    pub step: usize,
    /// Decoded frames seen while sampling.
    pub scanned: usize,
}

/// Write one rgb24 frame as a JPEG.
fn save_jpeg(path: &Path, rgb: Vec<u8>, width: u32, height: u32, quality: u8) -> Result<()> {
    let buf = image::RgbImage::from_raw(width, height, rgb)
        .ok_or_else(|| anyhow::anyhow!("frame buffer size mismatch"))?;
    let file = fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    let encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut writer, quality.clamp(1, 100));
    encoder
        .write_image(buf.as_raw(), width, height, image::ExtendedColorType::Rgb8)
        .with_context(|| format!("encoding {}", path.display()))?;
    writer.flush()?;
    Ok(())
}

/// Sample frames from `opts.video` into `opts.out_dir`.
pub fn extract_frames(opts: &ExtractOptions) -> Result<ExtractReport> {
    if opts.frames == 0 {
        bail!("--frames must be at least 1");
    }
    if opts.fps < 0.0 {
        bail!("--fps must be >= 0 (use 0 to spread --frames evenly)");
    }
    if let (Some(start), Some(end)) = (opts.start_s, opts.end_s)
        && end <= start
    {
        bail!("--end ({end}s) must be after --start ({start}s)");
    }
    if !opts.video.is_file() {
        bail!("input video not found: {}", opts.video.display());
    }

    fs::create_dir_all(&opts.out_dir)
        .with_context(|| format!("creating {}", opts.out_dir.display()))?;

    let clip = opts.clip();
    let mut reader = VideoReader::open_clipped(&opts.video, opts.max_side, clip)?;
    let fps = if reader.fps.is_finite() && reader.fps > 0.0 {
        reader.fps
    } else {
        25.0
    };

    // How many decoded frames to advance between two samples.
    let step = if opts.fps > 0.0 {
        (fps / opts.fps).round().max(1.0) as usize
    } else {
        (reader.n_frames_est.max(1) / opts.frames.max(1)).max(1)
    };

    let prefix = opts
        .prefix
        .as_deref()
        .map(sanitize_name)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            opts.video
                .file_stem()
                .map(|s| sanitize_name(&s.to_string_lossy()))
        })
        .unwrap_or_else(|| "frame".to_string());

    // Source frame index of the clip's first frame, so names stay aligned with
    // the original video when `--start` is used.
    let index_base = (reader.start_s * fps).round().max(0.0) as usize;

    let mut written = 0usize;
    let mut skipped = 0usize;
    let mut sampled = 0usize;
    let mut scanned = 0usize;
    let mut out_w = 0u32;
    let mut out_h = 0u32;

    while sampled < opts.frames {
        let Some(frame) = reader.next_frame()? else {
            break;
        };
        scanned += 1;
        if frame.index % step != 0 || frame.index / step >= opts.frames {
            continue;
        }
        sampled += 1;
        let source_index = index_base + frame.index;
        let image_path = opts.out_dir.join(format!("{prefix}_{source_index:06}.jpg"));
        // Never silently overwrite hand-corrected frames.
        if image_path.exists() && !opts.overwrite {
            skipped += 1;
            continue;
        }
        let (w, h) = (frame.width, frame.height);
        save_jpeg(&image_path, frame.data, w, h, opts.quality)?;
        // Empty sibling label so the folder is valid YOLO layout right away.
        let label = label_path_for(&image_path);
        if !label.exists() {
            fs::write(&label, b"").with_context(|| format!("creating {}", label.display()))?;
        }
        out_w = w;
        out_h = h;
        written += 1;
    }

    if sampled == 0 {
        bail!(
            "no frames written from {} (clip empty or shorter than one sample step)",
            opts.video.display()
        );
    }

    // Report the real size from the first image when every frame was skipped.
    if out_w == 0
        && let Some(first) = super::list_images(&opts.out_dir)?
            .into_iter()
            .find(|p| is_image_path(p))
        && let Ok((w, h)) = image::image_dimensions(&first)
    {
        out_w = w;
        out_h = h;
    }

    Ok(ExtractReport {
        out_dir: opts.out_dir.display().to_string(),
        written,
        skipped,
        width: out_w,
        height: out_h,
        fps,
        step,
        scanned,
    })
}
