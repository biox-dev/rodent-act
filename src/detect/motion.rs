//! Background-subtraction based motion candidate generator.
//!
//! Purpose: *triggered* detection. Running a CNN on every frame is wasteful for
//! a mouse that sits still most of the time. Instead we compute cheap motion
//! energy with a temporal median background + frame differencing, and only call
//! the detector when the scene actually changes (plus a low-rate periodic
//! "refresh" so slow/dark objects are still found eventually).

use crate::common::{Bbox, Detection};

/// Downscale factor used internally; the mask lives on a coarse grid.
pub const DOWNSCALE: u32 = 8;
/// Max number of candidate regions emitted per frame.
const MAX_CANDIDATES: usize = 8;

pub struct MotionTrigger {
    width: u32,
    height: u32,
    gw: usize,
    gh: usize,
    gray: Vec<u8>,
    /// Exponential moving average of the background.
    background: Vec<f32>,
    diff: Vec<u8>,
    warmup: usize,
    /// `diff` value (0..=255) above which a coarse cell is "moving".
    diff_threshold: u8,
    /// Minimum number of moving coarse cells to trigger.
    min_cells: usize,
    min_area_ratio: f32,
    /// Last measured motion ratio (0..1) — reused as a feature later.
    pub last_motion_ratio: f32,
}

impl MotionTrigger {
    pub fn new(width: u32, height: u32) -> Self {
        let gw = (width / DOWNSCALE) as usize;
        let gh = (height / DOWNSCALE) as usize;
        let n = gw * gh;
        Self {
            width,
            height,
            gw,
            gh,
            gray: vec![0; (width * height) as usize],
            background: vec![0.0; n],
            diff: vec![0; n],
            warmup: 0,
            diff_threshold: 25,
            min_cells: 24,
            min_area_ratio: 0.004,
            last_motion_ratio: 0.0,
        }
    }

    /// Convert to luma and box-downsample to the coarse grid.
    fn update_gray(&mut self, rgb: &[u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                let i = (row + x) * 3;
                // Rec.601 luma, integer approximation
                self.gray[row + x] =
                    ((rgb[i] as u32 * 77 + rgb[i + 1] as u32 * 150 + rgb[i + 2] as u32 * 29) >> 8)
                        as u8;
            }
        }

        // mean-pool to coarse grid
        let d = DOWNSCALE as usize;
        for gy in 0..self.gh {
            for gx in 0..self.gw {
                let mut sum = 0u32;
                for dy in 0..d {
                    let y = gy * d + dy;
                    if y >= h {
                        break;
                    }
                    let row = y * w;
                    for dx in 0..d {
                        let x = gx * d + dx;
                        if x >= w {
                            break;
                        }
                        sum += self.gray[row + x] as u32;
                    }
                }
                self.diff[gy * self.gw + gx] = (sum / (d * d) as u32) as u8;
            }
        }
    }

    /// Compute the motion mask for `rgb` and return whether the detector
    /// should run on this frame.
    pub fn update(&mut self, rgb: &[u8]) -> bool {
        self.update_gray(rgb);
        // `diff` now holds the coarse luma image; reuse `gray` buffer tmp
        let coarse = std::mem::take(&mut self.diff);
        let n = coarse.len();

        // warm-up: initialise background with first few frames
        if self.warmup < 5 {
            for i in 0..n {
                self.background[i] += (coarse[i] as f32 - self.background[i]) / (self.warmup + 1) as f32;
            }
            self.warmup += 1;
            let mut mask = vec![0u8; n];
            mask.copy_from_slice(&coarse);
            self.diff = mask;
            self.last_motion_ratio = 0.0;
            return true; // always analyse the first frames
        }

        let mut moving = 0usize;
        let alpha = 0.02f32; // slow background adaptation
        for i in 0..n {
            let v = coarse[i] as f32;
            let d = (v - self.background[i]).abs();
            let motion = d > self.diff_threshold as f32;
            // only update background where nothing moves (avoid ghosts)
            if !motion {
                self.background[i] += (v - self.background[i]) * alpha;
            }
            if motion {
                moving += 1;
            }
        }
        self.last_motion_ratio = moving as f32 / n as f32;
        self.diff = coarse;
        moving >= self.min_cells && self.last_motion_ratio >= self.min_area_ratio
    }

    /// Coarse motion mask bounding boxes in original-frame coordinates.
    /// These are *candidate* regions (not detections).
    pub fn candidates(&self) -> Vec<Detection> {
        // Convert the coarse diff to a boolean mask (background already in
        // `background`, `diff` holds the current coarse image).
        let n = self.gw * self.gh;
        let mut mask = vec![false; n];
        for i in 0..n {
            mask[i] = (self.diff[i] as f32 - self.background[i]).abs() > self.diff_threshold as f32;
        }

        let mut regions = connected_boxes(&mask, self.gw, self.gh);
        regions.sort_by(|a, b| b.area().total_cmp(&a.area()));
        regions.truncate(MAX_CANDIDATES);

        let s = DOWNSCALE as f32;
        let (fw, fh) = (self.width as f32, self.height as f32);
        regions
            .into_iter()
            .map(|r| {
                // pad generously so full body (incl. tail) is inside the ROI
                let pad = 6.0 * s;
                let b = Bbox::new(
                    (r.x1 * s - pad).max(0.0),
                    (r.y1 * s - pad).max(0.0),
                    (r.x2 * s + pad).min(fw),
                    (r.y2 * s + pad).min(fh),
                );
                Detection {
                    bbox: b,
                    score: 1.0,
                    class_id: usize::MAX, // marker: candidate, not a detected class
                }
            })
            .collect()
    }
}

/// Union of coarse cells into bounding boxes via a coarse 2x2 dilation +
/// row-band grouping (cheap, good enough for ROI proposal).
fn connected_boxes(mask: &[bool], gw: usize, gh: usize) -> Vec<Bbox> {
    // dilate once so nearby blobs merge
    let mut d = vec![false; mask.len()];
    for y in 0..gh {
        for x in 0..gw {
            if !mask[y * gw + x] {
                continue;
            }
            for dy in 0..=2 {
                for dx in 0..=2 {
                    let ny = y + dy;
                    let nx = x + dx;
                    if ny < gh && nx < gw {
                        d[ny * gw + nx] = true;
                    }
                }
            }
        }
    }

    // stack-based flood fill on the dilated mask
    let mut visited = vec![false; d.len()];
    let mut boxes = Vec::new();
    let mut stack = Vec::new();
    for start in 0..d.len() {
        if !d[start] || visited[start] {
            continue;
        }
        let (mut x1, mut y1, mut x2, mut y2) = (usize::MAX, usize::MAX, 0usize, 0usize);
        let mut cells = 0usize;
        stack.push(start);
        visited[start] = true;
        while let Some(idx) = stack.pop() {
            let x = idx % gw;
            let y = idx / gw;
            cells += 1;
            x1 = x1.min(x);
            y1 = y1.min(y);
            x2 = x2.max(x + 1);
            y2 = y2.max(y + 1);
            let neighbours = [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ];
            for (nx, ny) in neighbours {
                if nx < gw && ny < gh {
                    let ni = ny * gw + nx;
                    if d[ni] && !visited[ni] {
                        visited[ni] = true;
                        stack.push(ni);
                    }
                }
            }
        }
        if cells >= 4 {
            boxes.push(Bbox::new(x1 as f32, y1 as f32, x2 as f32, y2 as f32));
        }
    }
    boxes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_motion_no_trigger() {
        let w = 64;
        let h = 64;
        let mut mt = MotionTrigger::new(w, h);
        let frame = vec![10u8; (w * h * 3) as usize];
        for _ in 0..10 {
            mt.update(&frame);
        }
        assert!(!mt.update(&frame));
        assert!(mt.candidates().is_empty());
    }

    #[test]
    fn motion_triggers_and_finds_region() {
        let w = 128;
        let h = 128;
        let mut mt = MotionTrigger::new(w, h);
        let frame = vec![10u8; (w * h * 3) as usize];
        for _ in 0..8 {
            mt.update(&frame);
        }
        let mut moving = frame.clone();
        for y in 40..80 {
            for x in 40..80 {
                let i = ((y * w + x) * 3) as usize;
                moving[i] = 250;
                moving[i + 1] = 250;
                moving[i + 2] = 250;
            }
        }
        assert!(mt.update(&moving));
        let c = mt.candidates();
        assert!(!c.is_empty(), "expected at least one candidate");
        let b = c[0].bbox;
        assert!(b.x1 < 60.0 && b.x2 > 60.0 && b.y1 < 60.0 && b.y2 > 60.0, "{b:?}");
    }
}
