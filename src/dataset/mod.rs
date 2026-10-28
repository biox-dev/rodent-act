//! YOLO dataset tooling: frame sampling, pre-labelling, manual-review helpers
//! and train/val export.
//!
//! This is the pure-Rust equivalent of `analytics/make_dataset.py`:
//!
//! ```text
//! extract   video -> raw/<stem>_000123.jpg (+ empty .txt)
//! autolabel raw/*  -> dark-region boxes written as YOLO labels
//! (review)  web UI canvas editor reads/writes the same .txt files
//! split     raw/*  -> images/{train,val}, labels/{train,val}, data.yaml
//! check     validate labels (missing files, bad lines, out-of-range values)
//! ```
//!
//! Label format (one box per line, all normalised to 0..1):
//!
//! ```text
//! <class_idx> <x_center> <y_center> <width> <height>
//! ```

pub mod autolabel;
pub mod extract;
pub mod split;

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub use autolabel::{AutolabelOptions, AutolabelReport};
pub use extract::{ExtractOptions, ExtractReport};
pub use split::{CheckReport, SplitOptions, SplitReport};

/// Image extensions recognised inside a dataset folder.
pub const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "bmp"];

/// One YOLO box, in normalised image coordinates.
///
/// Serialised as camelCase so the web UI can use it directly
/// (`classId`, `xc`, `yc`, `w`, `h`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LabelBox {
    pub class_id: usize,
    /// Box centre, 0..1.
    pub xc: f32,
    pub yc: f32,
    /// Box size, 0..1.
    pub w: f32,
    pub h: f32,
}

impl LabelBox {
    pub fn new(class_id: usize, xc: f32, yc: f32, w: f32, h: f32) -> Self {
        Self {
            class_id,
            xc,
            yc,
            w,
            h,
        }
    }

    /// True when every value is finite and inside the image.
    pub fn is_valid(&self) -> bool {
        [self.xc, self.yc, self.w, self.h]
            .iter()
            .all(|v| v.is_finite())
            && (0.0..=1.0).contains(&self.xc)
            && (0.0..=1.0).contains(&self.yc)
            && self.w > 0.0
            && self.w <= 1.0
            && self.h > 0.0
            && self.h <= 1.0
    }
}

/// Which exported subset folder currently contains a frame.
///
/// A frame can be in both when the split was produced externally (or copied
/// by hand); the annotation editor shows that explicitly so it can be fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SplitMember {
    Train,
    Val,
    Both,
}

/// One image + its label file, as shown in the annotation editor.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetFrame {
    /// Image file name (`TST_000012.jpg`).
    pub name: String,
    /// Label file name (`TST_000012.txt`).
    pub label: String,
    pub boxes: usize,
    pub width: u32,
    pub height: u32,
    /// `"train"`, `"val"`, `"both"`, or `null` when not exported yet.
    pub split: Option<SplitMember>,
}

/// Aggregate counters for a dataset directory (`<root>/raw`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetSummary {
    pub name: String,
    pub path: String,
    pub n_images: usize,
    pub n_boxes: usize,
    pub empty_images: usize,
    pub has_split: bool,
    pub train: usize,
    pub val: usize,
    pub modified_ms: u64,
}

// --------------------------------------------------------------------------- //
// names / paths
// --------------------------------------------------------------------------- //

/// True when `name` is a single safe path component (dataset or file name).
pub fn is_safe_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 96
        && name != "."
        && name != ".."
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Make an arbitrary string usable as a dataset directory name.
pub fn sanitize_name(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            out.push(c);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "dataset".to_string()
    } else {
        out.chars().take(64).collect()
    }
}

/// `foo.jpg` -> `foo.txt` next to it.
pub fn label_path_for(image: &Path) -> PathBuf {
    image.with_extension("txt")
}

pub fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTS.iter().any(|x| x.eq_ignore_ascii_case(e)))
        .unwrap_or(false)
}

/// Sorted image files in `dir` (empty when the directory does not exist).
pub fn list_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let rd = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
    };
    let mut out: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_image_path(p))
        .collect();
    out.sort();
    Ok(out)
}

fn count_files(dir: &Path) -> usize {
    fs::read_dir(dir)
        .map(|rd| rd.flatten().filter(|e| e.path().is_file()).count())
        .unwrap_or(0)
}

// --------------------------------------------------------------------------- //
// label I/O
// --------------------------------------------------------------------------- //

/// Read a YOLO label file. Blank lines are ignored; malformed values are
/// clamped instead of aborting so a single bad line cannot block the editor.
pub fn read_labels(path: &Path) -> Result<Vec<LabelBox>> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let mut out = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 5 {
            continue;
        }
        let Ok(class_id) = parts[0].parse::<usize>() else {
            continue;
        };
        let mut v = [0f32; 4];
        let mut ok = true;
        for (dst, s) in v.iter_mut().zip(&parts[1..]) {
            match s.parse::<f32>() {
                Ok(x) if x.is_finite() => *dst = x.clamp(0.0, 1.0),
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let [xc, yc, w, h] = v;
        if w <= 0.0 || h <= 0.0 {
            continue;
        }
        out.push(LabelBox::new(class_id, xc, yc, w, h));
    }
    Ok(out)
}

/// Overwrite a label file with `boxes` (creates parent dirs).
pub fn write_labels(path: &Path, boxes: &[LabelBox]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let tmp = path.with_extension("txt.tmp");
    let mut file = fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    for b in boxes {
        writeln!(
            file,
            "{} {:.6} {:.6} {:.6} {:.6}",
            b.class_id, b.xc, b.yc, b.w, b.h
        )?;
    }
    file.flush()?;
    drop(file);
    // Atomic replace: the editor auto-saves while a request may be reading.
    fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

// --------------------------------------------------------------------------- //
// summaries
// --------------------------------------------------------------------------- //

/// Exported subset folders that contain `name` (checked by file name).
fn split_member(root: &Path, name: &str) -> Option<SplitMember> {
    let in_train = root.join("images/train").join(name).is_file();
    let in_val = root.join("images/val").join(name).is_file();
    match (in_train, in_val) {
        (true, true) => Some(SplitMember::Both),
        (true, false) => Some(SplitMember::Train),
        (false, true) => Some(SplitMember::Val),
        (false, false) => None,
    }
}

/// Frames in `root/raw`, with per-frame box counts, dimensions and whether
/// each one is already copied into `images/{train,val}`.
pub fn list_frames(root: &Path) -> Result<Vec<DatasetFrame>> {
    let images = list_images(&root.join("raw"))?;
    let mut out = Vec::with_capacity(images.len());
    for img in images {
        let label = label_path_for(&img);
        let boxes = read_labels(&label).map(|b| b.len()).unwrap_or(0);
        let (width, height) = image::image_dimensions(&img).unwrap_or((0, 0));
        let name = img
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let split = split_member(root, &name);
        out.push(DatasetFrame {
            label: label
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            split,
            name,
            boxes,
            width,
            height,
        });
    }
    Ok(out)
}

/// Counters for the dataset rooted at `root` (expects `root/raw`).
pub fn summarize(root: &Path, name: &str) -> DatasetSummary {
    let raw = root.join("raw");
    let images = list_images(&raw).unwrap_or_default();
    let mut n_boxes = 0usize;
    let mut empty_images = 0usize;
    for img in &images {
        let n = read_labels(&label_path_for(img))
            .map(|b| b.len())
            .unwrap_or(0);
        if n == 0 {
            empty_images += 1;
        }
        n_boxes += n;
    }
    let modified_ms = fs::metadata(&raw)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
        .unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
        });
    DatasetSummary {
        name: name.to_string(),
        path: root.display().to_string(),
        n_images: images.len(),
        n_boxes,
        empty_images,
        has_split: root.join("data.yaml").is_file(),
        train: count_files(&root.join("images/train")),
        val: count_files(&root.join("images/val")),
        modified_ms,
    }
}

/// Parse `x1,y1,x2,y2` (normalised), rejecting empty or inverted rectangles.
pub fn parse_roi(spec: &str) -> Result<[f32; 4]> {
    let vals: Vec<f32> = spec
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<f32>()
                .with_context(|| format!("invalid ROI value `{s}`"))
        })
        .collect::<Result<_>>()?;
    if vals.len() != 4 {
        bail!("ROI expects x1,y1,x2,y2 (got {} value(s))", vals.len());
    }
    let [x1, y1, x2, y2] = [vals[0], vals[1], vals[2], vals[3]];
    if [x1, y1, x2, y2].iter().any(|v| !(0.0..=1.0).contains(v)) || x2 <= x1 || y2 <= y1 {
        bail!("ROI must be 0..1 with x2>x1 and y2>y1 (got {spec})");
    }
    Ok([x1, y1, x2, y2])
}
