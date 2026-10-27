//! Step 2: pre-fill YOLO labels from "darker than the frame median" blobs.
//!
//! The TST rig is white walls + dark mice, so a simple darkness mask finds the
//! animals without any model. This is a starting point for hand correction,
//! not ground truth: lights, cables and cage hardware can appear as boxes, so
//! an ROI filter is strongly recommended.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::common::Bbox;

use super::{LabelBox, label_path_for, list_images, read_labels, write_labels};

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AutolabelOptions {
    /// How much darker than the frame median a pixel must be (0..=200).
    pub threshold: u8,
    /// Minimum connected-component area as a fraction of the frame.
    pub min_area: f32,
    /// Maximum candidates considered before ROI/channel truncation.
    pub max_objects: usize,
    /// Keep at most this many boxes per frame (`0` = no limit).
    pub channels: usize,
    /// Keep only boxes whose centre is inside `[x1, y1, x2, y2]` (0..1).
    pub roi: Option<[f32; 4]>,
    pub class_id: usize,
    /// Skip frames that already have boxes (does not overwrite them).
    pub only_empty: bool,
    /// NMS IoU threshold.
    pub nms_iou: f32,
}

impl Default for AutolabelOptions {
    fn default() -> Self {
        Self {
            threshold: 45,
            min_area: 0.0008,
            max_objects: 12,
            channels: 0,
            roi: None,
            class_id: 0,
            only_empty: false,
            nms_iou: 0.25,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AutolabelReport {
    /// Frames that were labelled.
    pub images: usize,
    pub boxes: usize,
    /// Frames skipped because `only_empty` was set and they had boxes.
    pub skipped: usize,
    /// Labelled frames where no candidate was found.
    pub empty: usize,
}

/// Pre-label every image in `dir`, writing one `.txt` per image.
pub fn autolabel_dir(dir: &Path, opts: &AutolabelOptions) -> Result<AutolabelReport> {
    if !(0.0..1.0).contains(&opts.min_area) {
        bail!("min_area must be in 0..1 (fraction of the frame)");
    }
    if let Some([x1, y1, x2, y2]) = opts.roi
        && (!(0.0..=1.0).contains(&x1)
            || !(0.0..=1.0).contains(&y1)
            || !(0.0..=1.0).contains(&x2)
            || !(0.0..=1.0).contains(&y2)
            || x2 <= x1
            || y2 <= y1)
    {
        bail!("roi must be [x1, y1, x2, y2] in 0..1 with x2 > x1 and y2 > y1");
    }

    let images = list_images(dir)?;
    if images.is_empty() {
        bail!("no images in {} - extract frames first", dir.display());
    }

    let mut report = AutolabelReport {
        images: 0,
        boxes: 0,
        skipped: 0,
        empty: 0,
    };

    for path in images {
        let label = label_path_for(&path);
        if opts.only_empty && !read_labels(&label)?.is_empty() {
            report.skipped += 1;
            continue;
        }
        let img = image::open(&path)
            .with_context(|| format!("reading image {}", path.display()))?
            .to_rgb8();
        let boxes = detect_boxes(img.as_raw(), img.width(), img.height(), opts);
        if boxes.is_empty() {
            report.empty += 1;
        }
        report.boxes += boxes.len();
        write_labels(&label, &boxes)?;
        report.images += 1;
    }
    Ok(report)
}

/// Detect dark blobs in one rgb24 frame and return normalised YOLO boxes.
pub fn detect_boxes(rgb: &[u8], width: u32, height: u32, opts: &AutolabelOptions) -> Vec<LabelBox> {
    let (w, h) = (width as usize, height as usize);
    if w < 8 || h < 8 || rgb.len() < w * h * 3 {
        return Vec::new();
    }
    let gray = gray_from_rgb(rgb, w, h);
    let median = median_gray(&gray);
    let thr = opts.threshold as i32;

    let mut mask = vec![0u8; w * h];
    let mut any = false;
    for i in 0..w * h {
        if median as i32 - gray[i] as i32 > thr {
            mask[i] = 1;
            any = true;
        }
    }
    if !any {
        return Vec::new();
    }

    // 9x9 structuring element, like the OpenCV reference implementation.
    const RADIUS: usize = 4;
    mask = open(&mask, w, h, RADIUS);
    mask = close(&mask, w, h, RADIUS, 3);

    let comps = connected_components(&mask, w, h);
    let frame_area = (w * h) as f32;
    let mut candidates: Vec<(f32, Bbox)> = comps
        .iter()
        .filter_map(|c| {
            let bw = (c.x2 - c.x1 + 1) as f32;
            let bh = (c.y2 - c.y1 + 1) as f32;
            if (c.area as f32) < opts.min_area * frame_area || bw.min(bh) < 6.0 {
                return None;
            }
            Some((
                c.area as f32,
                Bbox::new(
                    c.x1 as f32,
                    c.y1 as f32,
                    c.x2 as f32 + 1.0,
                    c.y2 as f32 + 1.0,
                ),
            ))
        })
        .collect();

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept: Vec<Bbox> = Vec::new();
    for (_, b) in candidates {
        if kept.iter().any(|k| b.iou(k) > opts.nms_iou) {
            continue;
        }
        kept.push(b);
        if kept.len() >= opts.max_objects.max(1) {
            break;
        }
    }

    let mut out = Vec::with_capacity(kept.len());
    for b in kept {
        let xc = (b.x1 + b.x2) * 0.5 / w as f32;
        let yc = (b.y1 + b.y2) * 0.5 / h as f32;
        if let Some([rx1, ry1, rx2, ry2]) = opts.roi
            && !(rx1 <= xc && xc <= rx2 && ry1 <= yc && yc <= ry2)
        {
            continue;
        }
        out.push(LabelBox::new(
            opts.class_id,
            xc,
            yc,
            (b.x2 - b.x1) / w as f32,
            (b.y2 - b.y1) / h as f32,
        ));
    }
    if opts.channels > 0 {
        out.truncate(opts.channels);
    }
    out
}

// --------------------------------------------------------------------------- //
// image helpers
// --------------------------------------------------------------------------- //

fn gray_from_rgb(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut gray = vec![0u8; w * h];
    for (i, px) in rgb.chunks_exact(3).take(w * h).enumerate() {
        // Rec. 601 luma, integer form (same weights the drawing code uses).
        gray[i] = ((px[0] as u32 * 77 + px[1] as u32 * 150 + px[2] as u32 * 29) / 256) as u8;
    }
    gray
}

fn median_gray(gray: &[u8]) -> u8 {
    let mut hist = [0u32; 256];
    for &g in gray {
        hist[g as usize] += 1;
    }
    let half = (gray.len() / 2) as u32;
    let mut acc = 0u32;
    for (v, &n) in hist.iter().enumerate() {
        acc += n;
        if acc > half {
            return v as u8;
        }
    }
    255
}

/// Separable line min/max (rectangle structuring element).
fn line_filter(
    src: &[u8],
    w: usize,
    h: usize,
    r: usize,
    horizontal: bool,
    dilate: bool,
) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if horizontal {
        for y in 0..h {
            let row = &src[y * w..(y + 1) * w];
            let orow = &mut out[y * w..(y + 1) * w];
            for (x, o) in orow.iter_mut().enumerate() {
                let lo = x.saturating_sub(r);
                let hi = (x + r).min(w - 1);
                let mut v = u8::from(!dilate);
                for &s in &row[lo..=hi] {
                    if dilate {
                        if s != 0 {
                            v = 1;
                            break;
                        }
                    } else if s == 0 {
                        v = 0;
                        break;
                    }
                }
                *o = v;
            }
        }
    } else {
        for x in 0..w {
            for y in 0..h {
                let lo = y.saturating_sub(r);
                let hi = (y + r).min(h - 1);
                let mut v = u8::from(!dilate);
                for yy in lo..=hi {
                    let s = src[yy * w + x];
                    if dilate {
                        if s != 0 {
                            v = 1;
                            break;
                        }
                    } else if s == 0 {
                        v = 0;
                        break;
                    }
                }
                out[y * w + x] = v;
            }
        }
    }
    out
}

fn erode(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    let tmp = line_filter(src, w, h, r, true, false);
    line_filter(&tmp, w, h, r, false, false)
}

fn dilate(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    let tmp = line_filter(src, w, h, r, true, true);
    line_filter(&tmp, w, h, r, false, true)
}

fn open(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    dilate(&erode(src, w, h, r), w, h, r)
}

fn close(src: &[u8], w: usize, h: usize, r: usize, iters: usize) -> Vec<u8> {
    let mut cur = src.to_vec();
    for _ in 0..iters {
        cur = dilate(&cur, w, h, r);
    }
    for _ in 0..iters {
        cur = erode(&cur, w, h, r);
    }
    cur
}

#[derive(Debug, Clone, Copy)]
struct Component {
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    area: usize,
}

/// 8-connected components of a binary mask (iterative flood fill).
fn connected_components(mask: &[u8], w: usize, h: usize) -> Vec<Component> {
    let mut seen = vec![false; w * h];
    let mut comps = Vec::new();
    let mut stack: Vec<usize> = Vec::with_capacity(1024);
    for start in 0..w * h {
        if mask[start] == 0 || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.clear();
        stack.push(start);
        let mut c = Component {
            x1: w,
            y1: h,
            x2: 0,
            y2: 0,
            area: 0,
        };
        while let Some(p) = stack.pop() {
            let (x, y) = (p % w, p / w);
            c.area += 1;
            c.x1 = c.x1.min(x);
            c.x2 = c.x2.max(x);
            c.y1 = c.y1.min(y);
            c.y2 = c.y2.max(y);
            for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let n = yy * w + xx;
                    if mask[n] != 0 && !seen[n] {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
        }
        comps.push(c);
    }
    comps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_dark_rectangle() {
        let (w, h) = (80u32, 60u32);
        let mut rgb = vec![230u8; (w * h * 3) as usize];
        // 20x14 dark block around (30..50, 20..34)
        for y in 20..34 {
            for x in 30..50 {
                let i = ((y * w + x) * 3) as usize;
                rgb[i] = 20;
                rgb[i + 1] = 20;
                rgb[i + 2] = 20;
            }
        }
        let boxes = detect_boxes(&rgb, w, h, &AutolabelOptions::default());
        assert_eq!(boxes.len(), 1, "{boxes:?}");
        let b = boxes[0];
        assert!((b.xc - 40.0 / 80.0).abs() < 0.02, "{b:?}");
        assert!((b.yc - 27.0 / 60.0).abs() < 0.02, "{b:?}");
        assert!(b.is_valid());
    }

    #[test]
    fn roi_and_channels() {
        let (w, h) = (100u32, 100u32);
        let mut rgb = vec![240u8; (w * h * 3) as usize];
        for (cx, cy) in [(20u32, 20u32), (70, 70)] {
            for y in cy..cy + 10 {
                for x in cx..cx + 10 {
                    let i = ((y * w + x) * 3) as usize;
                    rgb[i] = 10;
                    rgb[i + 1] = 10;
                    rgb[i + 2] = 10;
                }
            }
        }
        let opts = AutolabelOptions {
            threshold: 45,
            min_area: 0.001,
            roi: Some([0.4, 0.4, 1.0, 1.0]),
            ..Default::default()
        };
        let boxes = detect_boxes(&rgb, w, h, &opts);
        assert_eq!(boxes.len(), 1, "{boxes:?}");
        assert!(boxes[0].xc > 0.5);

        let opts = AutolabelOptions {
            channels: 1,
            ..AutolabelOptions::default()
        };
        let boxes = detect_boxes(&rgb, w, h, &opts);
        assert_eq!(boxes.len(), 1);
    }
}
