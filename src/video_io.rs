//! Video I/O through `ffmpeg` child processes (no C bindings, no build deps).
//!
//! * [`VideoReader`] decodes the stream to raw `rgb24` frames at a fixed
//!   output size (letterboxed/padded) so that every downstream consumer sees
//!   square pixels.
//! * [`VideoWriter`] encodes raw `rgb24` frames with libx264.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, bail};
use tracing::debug;

/// True when the letterbox padding changed the geometry (i.e. aspect ratio was
/// not preserved by the resize itself).
#[derive(Debug, Clone, Copy)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    /// Content area inside the padded frame, in output pixels.
    pub content_w: f32,
    pub content_h: f32,
}

impl Letterbox {
    /// Map a box in *content* coordinates (out-pixel space, before padding) to
    /// padded-frame coordinates.
    pub fn to_padded(&self, b: crate::common::Bbox) -> crate::common::Bbox {
        crate::common::Bbox::new(
            b.x1 + self.pad_x,
            b.y1 + self.pad_y,
            b.x2 + self.pad_x,
            b.y2 + self.pad_y,
        )
    }

    /// Map a box in padded-frame coordinates back to content coordinates.
    pub fn from_padded(&self, b: crate::common::Bbox) -> crate::common::Bbox {
        crate::common::Bbox::new(
            b.x1 - self.pad_x,
            b.y1 - self.pad_y,
            b.x2 - self.pad_x,
            b.y2 - self.pad_y,
        )
    }
}

/// Half-open time window `[start_s, end_s)` of the *source* video that should
/// be analysed, in seconds.
///
/// Clipping happens in the decoder (`ffmpeg -ss ... -i ... -t ...`), so no
/// intermediate re-encoded file is produced: the pipeline simply sees a video
/// that starts at `start_s` and ends at `end_s`, with frame indices rebased to
/// 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clip {
    pub start_s: f64,
    /// `None` means "until the end of the source video".
    pub end_s: Option<f64>,
}

impl Clip {
    pub fn new(start_s: f64, end_s: Option<f64>) -> Self {
        Self {
            start_s: start_s.max(0.0),
            end_s: end_s.filter(|e| e.is_finite()),
        }
    }

    /// End of the clip, clamped to the real duration of the source.
    pub fn end_or(&self, source_duration: f64) -> f64 {
        match self.end_s {
            Some(e) => e.min(source_duration).max(self.start_s),
            None => source_duration.max(self.start_s),
        }
    }

    /// Length of the clip in seconds.
    pub fn len(&self, source_duration: f64) -> f64 {
        (self.end_or(source_duration) - self.start_s).max(0.0)
    }

    pub fn is_empty(&self, source_duration: f64) -> bool {
        self.len(source_duration) <= 0.0
    }
}

fn compute_letterbox(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Letterbox {
    let scale = (dst_w as f32 / src_w as f32).min(dst_h as f32 / src_h as f32);
    let content_w = (src_w as f32 * scale).round();
    let content_h = (src_h as f32 * scale).round();
    Letterbox {
        scale,
        pad_x: ((dst_w as f32 - content_w) * 0.5).round(),
        pad_y: ((dst_h as f32 - content_h) * 0.5).round(),
        content_w,
        content_h,
    }
}

/// Raw frame produced by ffmpeg (`rgb24`, `padded_w * padded_h * 3` bytes).
pub struct Frame {
    pub index: usize,
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Source geometry (the padded output size is `width x height`).
    pub letterbox: Letterbox,
}

impl Frame {
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * self.width + x) * 3) as usize;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }
}

pub struct VideoReader {
    child: Child,
    stdout: BufReader<ChildStdout>,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub n_frames_est: usize,
    pub frame_bytes: usize,
    letterbox: Letterbox,
    next_index: usize,
    /// Source-time offset of the first frame this reader yields.
    pub start_s: f64,
    /// Duration of the decoded window (the clip, or the whole source).
    pub duration_s: f64,
    /// Duration of the source file, ignoring the clip.
    pub source_duration_s: f64,
}

impl VideoReader {
    pub fn open(path: &Path, max_side: u32) -> Result<Self> {
        Self::open_clipped(path, max_side, None)
    }

    /// Open `path`, decoded from `clip.start_s` up to `clip.end_s`.
    ///
    /// Frame indices are rebased so the first decoded frame has index 0, and
    /// timestamps are relative to the clip start (frame 0 == t 0.0 s).
    pub fn open_clipped(path: &Path, max_side: u32, clip: Option<Clip>) -> Result<Self> {
        if !path.exists() {
            bail!("input video not found: {}", path.display());
        }
        let p = probe(path)?;
        let (src_w, src_h, fps) = (p.width, p.height, p.fps);
        let (dst_w, dst_h) = fit_size(src_w, src_h, max_side);
        let letterbox = compute_letterbox(src_w, src_h, dst_w, dst_h);

        let start_s = clip
            .map(|c| c.start_s.max(0.0))
            .unwrap_or(0.0)
            .min(p.duration_s);
        let duration_s = match clip {
            Some(c) => c.len(p.duration_s),
            None => (p.duration_s - start_s).max(0.0),
        };
        // A clipped stream has fewer frames than the container reports, so
        // derive the estimate from the requested window instead.
        let nb = match clip {
            Some(_) if fps > 0.0 => (duration_s * fps).round() as usize,
            _ if p.n_frames > 0 => p.n_frames,
            _ if fps > 0.0 => (duration_s * fps).round() as usize,
            _ => 0,
        };

        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-hide_banner", "-loglevel", "error", "-nostdin"]);
        if start_s > 1e-4 {
            // Input seek: ffmpeg decodes from the previous keyframe and drops
            // the leading frames, so this is frame-accurate without a second
            // encode pass (unlike `-c copy` cutting).
            cmd.args(["-ss", &format!("{start_s:.3}")]);
        }
        cmd.arg("-i").arg(path);
        if clip.is_some() {
            // Output-side duration caps the stream exactly at the clip end.
            cmd.args(["-t", &format!("{duration_s:.3}")]);
        }
        cmd.args(["-an", "-sn"])
            .args(["-vf", &format!("scale={dst_w}:{dst_h},format=rgb24")])
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .context("failed to spawn ffmpeg (is it on PATH?)")?;

        let stdout = BufReader::with_capacity(
            (dst_w * dst_h * 3) as usize,
            child.stdout.take().context("ffmpeg stdout not captured")?,
        );

        debug!(
            src_w,
            src_h, dst_w, dst_h, fps, start_s, duration_s, "video opened"
        );
        Ok(Self {
            child,
            stdout,
            width: dst_w,
            height: dst_h,
            fps,
            n_frames_est: nb,
            frame_bytes: (dst_w * dst_h * 3) as usize,
            letterbox,
            next_index: 0,
            start_s,
            duration_s,
            source_duration_s: p.duration_s,
        })
    }

    /// Read the next frame. Returns `None` at end of stream.
    pub fn next_frame(&mut self) -> Result<Option<Frame>> {
        let mut data = vec![0u8; self.frame_bytes];
        let mut filled = 0usize;
        while filled < self.frame_bytes {
            match self.stdout.read(&mut data[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e).context("failed reading ffmpeg output"),
            }
        }
        if filled == 0 {
            return Ok(None);
        }
        if filled < self.frame_bytes {
            bail!(
                "truncated frame from ffmpeg ({filled}/{} bytes)",
                self.frame_bytes
            );
        }
        let index = self.next_index;
        self.next_index += 1;
        Ok(Some(Frame {
            index,
            data,
            width: self.width,
            height: self.height,
            letterbox: self.letterbox,
        }))
    }

    /// Read frame `target` (skipping frames if needed). Used for random access
    /// inside a sliding window; callers should mostly consume sequentially.
    pub fn skip_to(&mut self, target: usize) -> Result<Option<Frame>> {
        while self.next_index < target {
            if self.next_frame()?.is_none() {
                return Ok(None);
            }
        }
        self.next_frame()
    }
}

impl Drop for VideoReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct VideoWriter {
    child: Child,
    stdin: Option<ChildStdin>,
    pub path: PathBuf,
    width: u32,
    height: u32,
    frame_bytes: usize,
}

impl VideoWriter {
    pub fn create(path: &Path, width: u32, height: u32, fps: f64) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let mut child = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y", "-nostdin"])
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{width}x{height}")])
            .args(["-r", &format!("{fps}")])
            .args(["-i", "-"])
            // Explicit mapping: the rawvideo demuxer sometimes exposes the
            // stdin pipe as more than one stream, which would make ffmpeg
            // write a second moov atom ("Found duplicated MOOV Atom") and a
            // file no player can decode.
            .args(["-map", "0:v:0"])
            .args(["-an", "-sn", "-dn"])
            .args(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
            .args(["-pix_fmt", "yuv420p"])
            // `+faststart` rewrites the file after encoding; give it room to
            // and keep the temporary file in the same directory as the target
            // so the rename stays on one filesystem.
            .args(["-movflags", "+faststart"])
            .arg(path)
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("failed to spawn ffmpeg encoder")?;
        // Drain stderr on a thread so a chatty encoder can never block on a
        // full pipe (which would stall our writer and eventually break the
        // stream with a partial frame).
        if let Some(err) = child.stderr.take() {
            std::thread::spawn(move || {
                use std::io::Read;
                let mut buf = Vec::new();
                let mut err = err;
                let _ = err.read_to_end(&mut buf);
                let text = String::from_utf8_lossy(&buf);
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    tracing::error!(target: "ffmpeg", "{line}");
                }
            });
        }
        let stdin = child.stdin.take().context("ffmpeg stdin not captured")?;
        Ok(Self {
            child,
            stdin: Some(stdin),
            path: path.to_path_buf(),
            width,
            height,
            frame_bytes: (width as usize) * (height as usize) * 3,
        })
    }

    pub fn write_frame(&mut self, rgb: &[u8]) -> Result<()> {
        if rgb.len() != self.frame_bytes {
            bail!(
                "frame size mismatch: got {} bytes, encoder expects {} ({}x{} rgb24)",
                rgb.len(),
                self.frame_bytes,
                self.width,
                self.height
            );
        }
        let stdin = self.stdin.as_mut().context("ffmpeg stdin already closed")?;
        match stdin.write_all(rgb) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {
                bail!("ffmpeg encoder closed its input early (encode failed); see ffmpeg log above")
            }
            Err(e) => Err(e).context("writing frame to ffmpeg"),
        }
    }

    /// Flush and finalize the file. Must be called for a valid output.
    pub fn finish(mut self) -> Result<()> {
        if let Some(mut stdin) = self.stdin.take() {
            stdin.flush()?;
            drop(stdin); // EOF -> ffmpeg finalises the container
        }
        let status = self.child.wait()?;
        if !status.success() {
            bail!("ffmpeg encoder exited with {status}");
        }
        Ok(())
    }
}

impl Drop for VideoWriter {
    fn drop(&mut self) {
        // If finish() was not called the encoder never wrote a valid trailer,
        // so the file on disk is undecodable. Kill the encoder and remove the
        // half-written output instead of leaving a corrupt MP4 behind.
        if self.stdin.take().is_some() {
            let _ = self.child.kill();
            let _ = self.child.wait();
            if self.path.exists() {
                let _ = std::fs::remove_file(&self.path);
                tracing::warn!(
                    path = %self.path.display(),
                    "removed incomplete video (pipeline stopped before finishing)"
                );
            }
        } else {
            let _ = self.child.try_wait();
        }
    }
}

/// Resize geometry keeping aspect ratio, longest side == `max_side`, both
/// dimensions rounded to even numbers (encoder friendly).
fn fit_size(w: u32, h: u32, max_side: u32) -> (u32, u32) {
    let longest = w.max(h) as f32;
    let s = (max_side as f32 / longest).min(1.0);
    let mut nw = (w as f32 * s).round() as u32;
    let mut nh = (h as f32 * s).round() as u32;
    nw = ((nw + 1) / 2) * 2;
    nh = ((nh + 1) / 2) * 2;
    (nw.max(2), nh.max(2))
}

/// Geometry/timing facts about a video file.
#[derive(Debug, Clone, Copy, Default)]
pub struct VideoProbe {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub n_frames: usize,
    pub duration_s: f64,
}

/// Probe geometry, frame rate, frame count and duration.
pub fn probe(path: &Path) -> Result<VideoProbe> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args([
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
        ])
        .args(["-show_entries", "format=duration"])
        .args(["-of", "json"])
        .arg(path)
        .output()
        .context("failed to run ffprobe")?;
    if !out.status.success() {
        bail!("ffprobe failed for {}", path.display());
    }
    let json: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let stream = json["streams"].get(0).context("no video stream in input")?;

    let width = stream["width"].as_u64().unwrap_or(0) as u32;
    let height = stream["height"].as_u64().unwrap_or(0) as u32;
    if width == 0 || height == 0 {
        bail!("could not determine video dimensions");
    }

    let fps = stream["r_frame_rate"]
        .as_str()
        .and_then(parse_rational)
        .filter(|v| *v > 1.0 && *v < 1000.0)
        .unwrap_or(25.0);

    let duration_s = json["format"]["duration"]
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|d| d.is_finite() && *d > 0.0)
        .unwrap_or(0.0);

    let nb = stream["nb_frames"]
        .as_str()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .or_else(|| (duration_s > 0.0).then(|| (duration_s * fps).round() as usize))
        .unwrap_or(0);

    let duration_s = if duration_s > 0.0 {
        duration_s
    } else if fps > 0.0 {
        nb as f64 / fps
    } else {
        0.0
    };

    Ok(VideoProbe {
        width,
        height,
        fps,
        n_frames: nb,
        duration_s,
    })
}

/// Returns `(width, height, fps, n_frames)`.
pub fn probe_video(path: &Path) -> Result<(u32, u32, f64, usize)> {
    let p = probe(path)?;
    Ok((p.width, p.height, p.fps, p.n_frames))
}

fn parse_rational(s: &str) -> Option<f64> {
    if let Some((num, den)) = s.split_once('/') {
        let n: f64 = num.parse().ok()?;
        let d: f64 = den.parse().ok()?;
        if d == 0.0 { None } else { Some(n / d) }
    } else {
        s.parse().ok()
    }
}

/// Count frames by decoding (fallback when container lacks frame count).
pub fn count_frames_lazy(path: &Path) -> Result<usize> {
    let mut r = VideoReader::open(path, 320)?;
    let mut n = 0;
    while r.next_frame()?.is_some() {
        n += 1;
    }
    Ok(n)
}

/// `-list_devices` helper for environments with multiple cameras.
pub fn list_capture_devices() -> Result<Vec<String>> {
    let out = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-list_devices",
            "true",
            "-f",
            "v4l2",
            "-i",
            "",
        ])
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .output()?;
    let mut devices = Vec::new();
    if let Ok(text) = String::from_utf8(out.stderr) {
        for line in text.lines() {
            if line.contains("/dev/video") {
                devices.push(line.trim().to_string());
            }
        }
    }
    Ok(devices)
}

/// Read a single line from stdin (used by the interactive preview prompt).
pub fn read_line_trimmed() -> String {
    let mut s = String::new();
    std::io::stdin().lock().read_line(&mut s).ok();
    s.trim().to_string()
}

/// Convenience: ensure a line-buffered stdout for progress bars on Windows.
pub fn flush_stdout() {
    let _ = std::io::stdout().flush();
}

#[allow(dead_code)]
fn _assert_bufread(_: &dyn BufRead) {}
