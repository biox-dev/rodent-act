//! Background-modelling "blob" object detector.
//!
//! When the animal is much darker (or brighter) than the enclosure floor, we do
//! not need a CNN at all: a running background estimate plus a per-pixel
//! threshold yields the animal mask, and connected components give the boxes.
//!
//! This is the default detector for the tail-suspension pipeline:
//!
//! * it is ~200x cheaper than a YOLO forward pass on CPU,
//! * it finds *every* animal in the arena (multi-object by construction),
//! * it gives a free per-object motion mask that the 0.5 s feature extractor
//!   reuses as ``motion_energy``.
//!
//! Tuning knobs: `diff_threshold` (contrast), `min_area` / `max_area`
//! (in pixels, relative to the frame), `max_objects`.

use crate::common::{Bbox, Detection};

/// How recently a cell must have *changed* to be considered background.
///
/// This is the key modelling decision for behaviour analysis on a static
/// camera. A pixel that has **never** changed within this window is frozen:
/// either an animal body (which differs from the floor and stays put) or a
/// perfectly uniform region. Real bedding always jitters from sensor noise, so
/// the floor keeps refreshing itself, while an animal's interior goes quiet.
///
/// The foreground mask therefore combines two cues:
/// * **contrast** - the cell differs from the temporal-median background, and
/// * **frozen** - the cell has not changed for `CHANGE_WINDOW` frames.
const CHANGE_WINDOW: u32 = 60;

/// Number of frames in the temporal-median window used to suppress moving
/// objects from the background estimate.
const HISTORY: usize = 25;

pub struct BlobDetector {
    width: u32,
    height: u32,
    /// Number of pixels collapsed per coarse cell on each axis.
    cell: u32,
    gw: usize,
    gh: usize,
    /// Background model (gray, coarse grid).
    bg: Vec<f32>,
    /// Current coarse gray image.
    cur: Vec<u8>,
    /// Current per-cell "object" mask (true = object).
    pub mask: Vec<bool>,
    /// Sliding-window ring buffer of recent coarse frames (temporal median).
    ring: Vec<u8>,
    ring_pos: usize,
    ring_filled: usize,
    /// Scratch buffer for the median computation.
    scratch: Vec<u8>,
    /// Long-term static background (slow EMA of the *median* background);
    /// this is what the scene looks like with no animal present.
    pub static_bg: Vec<f32>,
    /// Frames since each cell last changed by more than `change_threshold`.
    /// `>= CHANGE_WINDOW` means "frozen" (never/long-ago changed).
    since_change: Vec<u32>,
    /// Monotonic frame counter (used to separate ghosts from new objects).
    frame_no: u32,
    /// Frame index at which each cell was last *released* from the foreground
    /// mask (i.e. the moment an animal moved away from it).
    vacated_at: Vec<u32>,
    /// Explicit empty-scene reference (see [`BlobDetector::set_reference`]).
    reference: Option<Vec<u8>>,
    /// Previous frame (median over a short window, updated every frame).
    prev: Vec<u8>,
    /// Per-cell change threshold (adaptive, from local noise).
    pub change_threshold: u8,
    /// Absolute gray difference threshold.
    pub diff_threshold: u8,
    /// Minimum/maximum object area as a fraction of the frame.
    pub min_area_ratio: f32,
    pub max_area_ratio: f32,
    /// Background adaptation rate (`0` = frozen background).

    ///
    /// This trades responsiveness to lighting changes against absorbing a
    /// stationary animal into the background. At 25 fps, `0.0008` needs about
    /// 50 s for a stationary object to fade - long enough for a 30 s TST trial.
    pub alpha: f32,
    pub max_objects: usize,
    /// Minimum "fill" score for a blob to be reported. Low-fill blobs are
    /// usually leftover bedding debris rather than animals.
    pub min_score: f32,
    /// Overlap above which two blobs are considered the same object.
    pub nms_iou: f32,
    /// Frames during which a vacated region is suppressed as a ghost.
    pub ghost_frames: u32,
    pub invert: bool,
    /// Gap (in coarse cells) across which fragments are merged.
    pub merge_gap: usize,
    warmup: usize,
    /// Label id for all produced detections (single-class "animal").
    pub class_id: usize,
}

impl BlobDetector {
    pub fn new(width: u32, height: u32) -> Self {
        let cell = 4u32;
        let gw = (width / cell) as usize;
        let gh = (height / cell) as usize;
        Self {
            width,
            height,
            cell,
            gw,
            gh,
            bg: vec![-1.0; gw * gh],
            cur: vec![0; gw * gh],
            mask: vec![false; gw * gh],
            ring: vec![0; gw * gh * HISTORY],
            ring_pos: 0,
            ring_filled: 0,
            scratch: vec![0; HISTORY],
            static_bg: vec![0.0; gw * gh],
            since_change: vec![0; gw * gh],
            frame_no: 0,
            vacated_at: vec![u32::MAX; gw * gh],
            reference: None,
            prev: vec![0; gw * gh],
            change_threshold: 6,
            diff_threshold: 22,
            min_area_ratio: 0.0015,
            max_area_ratio: 0.60,
            alpha: 0.0008,
            max_objects: 4,
            min_score: 0.13,
            nms_iou: 0.25,
            ghost_frames: 90,
            invert: false,
            merge_gap: 5,
            warmup: 0,
            class_id: 0,
        }
    }

    fn gray_coarse(&mut self, rgb: &[u8]) {
        let w = self.width as usize;
        let n = self.gw * self.gh;
        // Downsample with a stride: cheap, and boxes get refined later.
        let s = self.cell as usize;
        for gy in 0..self.gh {
            for gx in 0..self.gw {
                let mut sum = 0u32;
                let mut cnt = 0u32;
                for dy in 0..s {
                    let y = gy * s + dy;
                    if y >= self.height as usize {
                        break;
                    }
                    let row = y * w;
                    for dx in 0..s {
                        let x = gx * s + dx;
                        if x >= w {
                            break;
                        }
                        let i = (row + x) * 3;
                        sum += (rgb[i] as u32 * 77 + rgb[i + 1] as u32 * 150 + rgb[i + 2] as u32 * 29)
                            >> 8;
                        cnt += 1;
                    }
                }
                self.cur[gy * self.gw + gx] = if cnt == 0 { 0 } else { (sum / cnt) as u8 };
            }
        }
        debug_assert_eq!(self.cur.len(), n);
    }

    /// Update the background model and object mask. Returns `true` when at
    /// least one object is present.
    /// Provide one frame of the **empty enclosure** (animals removed).
    ///
    /// The temporal background cannot see an object that is present from the
    /// very first frame, which is exactly the TST situation (the animal hangs
    /// there before the recording starts). With a reference frame the detector
    /// compares against that instead, and *every* animal is found.
    pub fn set_reference(&mut self, rgb: &[u8]) {
        self.gray_coarse(rgb);
        self.reference = Some(self.cur.clone());
    }

    /// True when an explicit reference frame is in use.
    pub fn has_reference(&self) -> bool {
        self.reference.is_some()
    }

    pub fn update(&mut self, rgb: &[u8]) -> bool {
        self.gray_coarse(rgb);
        let n = self.bg.len();

        // With a reference frame the "background" is simply that frame.
        if let Some(r) = self.reference.clone() {
            let thr = self.diff_threshold as f32;
            let mut n_obj = 0usize;
            for i in 0..n {
                let obj = (self.cur[i] as f32 - r[i] as f32).abs() > thr;
                self.mask[i] = obj;
                if obj {
                    n_obj += 1;
                }
            }
            self.prev.copy_from_slice(&self.cur);
            self.frame_no = self.frame_no.wrapping_add(1);
            return n_obj > 0;
        }

        // Push the new frame into the temporal window.
        let base = self.ring_pos * n;
        self.ring[base..base + n].copy_from_slice(&self.cur);
        self.ring_pos = (self.ring_pos + 1) % HISTORY;
        self.ring_filled = (self.ring_filled + 1).min(HISTORY);

        if self.ring_filled < 2 {
            self.mask.iter_mut().for_each(|m| *m = false);
            self.prev.copy_from_slice(&self.cur);
            self.warmup = self.ring_filled;
            return false;
        }

        // --- 1. static background estimate (median of the last HISTORY) ----
        if self.ring_filled >= HISTORY {
            self.recompute_background();
        } else {
            // during warm-up use the running mean of what we have
            for i in 0..n {
                self.bg[i] += (self.cur[i] as f32 - self.bg[i]) / self.ring_filled as f32;
            }
        }

        // --- 1b. long-term static background (slow EMA of the median) ------
        if self.ring_filled >= HISTORY {
            let a = 0.002f32;
            for i in 0..n {
                if self.static_bg[i] == 0.0 {
                    self.static_bg[i] = self.bg[i];
                } else {
                    self.static_bg[i] += (self.bg[i] - self.static_bg[i]) * a;
                }
            }
        }

        // --- 2. motion history: when did each cell last change? ------------
        let thr = self.change_threshold;
        for i in 0..n {
            let d = (self.cur[i] as i16 - self.prev[i] as i16).unsigned_abs();
            if d > thr as u16 {
                self.since_change[i] = 0;
            } else {
                self.since_change[i] = self.since_change[i].saturating_add(1);
            }
        }
        self.prev.copy_from_slice(&self.cur);
        self.frame_no = self.frame_no.wrapping_add(1);

        // --- 3. foreground = differs from the static background OR has been
        //        frozen for a long time (=> it is an object, not the floor) --
        let diff_thr = self.diff_threshold as f32;
        let mut n_obj = 0usize;
        for i in 0..n {
            let differs = (self.cur[i] as f32 - self.bg[i]).abs() > diff_thr;
            let frozen = self.since_change[i] >= CHANGE_WINDOW;
            // A *frozen* cell is an object only if it also differs from the
            // long-term static background (`static_bg`) - i.e. it is a still
            // object sitting on the floor, not the floor itself.
            let was = self.mask[i];
            let obj = differs || (frozen && self.frozen_is_object(i));
            if was && !obj {
                // Released this frame. Only record a ghost when the cell now
                // matches the *static* background - i.e. the animal really
                // moved away. A cell that merely stopped moving but is still
                // darker than the floor is an object that should stay visible.
                let matches_floor =
                    (self.cur[i] as f32 - self.static_bg[i]).abs() <= self.diff_threshold as f32 * 0.6;
                self.vacated_at[i] = if matches_floor { self.frame_no } else { u32::MAX };
            }
            self.mask[i] = obj;
            if obj {
                n_obj += 1;
            }
        }
        self.warmup = self.ring_filled;
        n_obj > 0
    }

    /// A frozen cell counts as an object when it does not belong to the
    /// long-term static background (the part of the scene that never moved).
    fn frozen_is_object(&self, i: usize) -> bool {
        (self.cur[i] as f32 - self.static_bg[i]).abs() > self.diff_threshold as f32 * 0.6
    }

    /// Per-cell median over the temporal window.
    fn recompute_background(&mut self) {
        let n = self.bg.len();
        let filled = self.ring_filled;
        let mid = filled / 2;
        for i in 0..n {
            for slot in 0..filled {
                self.scratch[slot] = self.ring[slot * n + i];
            }
            let buf = &mut self.scratch[..filled];
            // `select_nth_unstable` is O(n) and we only need the median.
            let (_, med, _) = buf.select_nth_unstable(mid);
            self.bg[i] = *med as f32;
        }
    }

    /// Connected components of the object mask -> boxes in frame pixels.
    ///
    /// Animals are low-contrast gradient objects, so a raw threshold splits a
    /// single animal into fragments (eye, ear, body). We therefore:
    /// 1. apply a 3x3 opening to drop salt noise,
    /// 2. dilate each component by `merge_gap` cells so nearby parts join,
    /// 3. re-run connected components on the dilated mask,
    /// 4. filter by area and keep the `max_objects` largest blobs.
    pub fn detections(&self) -> Vec<Detection> {
        // Ghost suppression: cells that were foreground very recently and have
        // just been released are bedding the animal just left behind. They are
        // still "frozen" (they changed once, then stopped), which would
        // otherwise create a spurious stationary object.
        let mut cleaned = self.mask.clone();
        let ghost_window = self.ghost_frames;
        if ghost_window > 0 {
            for i in 0..cleaned.len() {
                if cleaned[i] {
                    let age = self.frame_no.wrapping_sub(self.vacated_at[i]);
                    if self.vacated_at[i] != u32::MAX && age < ghost_window {
                        cleaned[i] = false;
                    }
                }
            }
        }
        let opened = morph_open(&cleaned, self.gw, self.gh);
        let (merged, dilated) = morph_close(&opened, self.gw, self.gh, self.merge_gap);
        let (boxes, areas) = components(&merged, self.gw, self.gh);
        let s = self.cell as f32;
        let frame_area = (self.width * self.height) as f32;
        let min_area = self.min_area_ratio * frame_area;
        let max_area = self.max_area_ratio * frame_area;

        let mut out: Vec<Detection> = boxes
            .into_iter()
            .zip(areas)
            .filter_map(|(b, cells)| {
                let area = cells as f32 * s * s;
                if area < min_area || area > max_area {
                    return None;
                }
                let bbox = Bbox::new(
                    b.x1 * s,
                    b.y1 * s,
                    (b.x2 * s).min(self.width as f32),
                    (b.y2 * s).min(self.height as f32),
                );
                // Reject very thin structures (tails, cables, bedding edges).
                let (bw, bh) = bbox.wh();
                if bw.min(bh) < 6.0 * s {
                    return None;
                }
                // Count real (undilated) mask cells inside the merged box so
                // the score reflects actual object pixels rather than the
                // dilation padding.
                let real = count_mask(&opened, self.gw, b);
                let fill = (real as f32 * s * s) / (bw * bh).max(1.0);
                if fill < self.min_score {
                    return None;
                }
                let _ = dilated;
                Some(Detection {
                    bbox,
                    // pseudo-score: plausibility of the blob as an animal body
                    score: fill.clamp(0.1, 1.0),
                    class_id: self.class_id,
                })
            })
            .collect();

        // NMS: a body split into two blobs (head vs torso) would otherwise
        // produce two tracks for one animal.
        out.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(b.bbox.area().total_cmp(&a.bbox.area()))
        });
        let mut kept: Vec<Detection> = Vec::with_capacity(out.len());
        for d in out {
            if kept.iter().any(|k| k.bbox.iou(&d.bbox) > self.nms_iou) {
                continue;
            }
            kept.push(d);
            if kept.len() >= self.max_objects {
                break;
            }
        }
        kept
    }

    /// Fraction of masked cells (used as a global activity signal).
    pub fn masked_ratio(&self) -> f32 {
        let n = self.mask.iter().filter(|m| **m).count();
        n as f32 / self.mask.len().max(1) as f32
    }

    /// Grayscale copy of the current frame on the coarse grid.
    pub fn coarse_gray(&self) -> &[u8] {
        &self.cur
    }

    pub fn coarse_grid(&self) -> (usize, usize) {
        (self.gw, self.gh)
    }
}

/// 3x3 erosion followed by 3x3 dilation (removes isolated cells).
fn morph_open(mask: &[bool], gw: usize, gh: usize) -> Vec<bool> {
    let eroded = erode(mask, gw, gh);
    dilate(&eroded, gw, gh, 1)
}

fn erode(mask: &[bool], gw: usize, gh: usize) -> Vec<bool> {
    let mut out = vec![false; mask.len()];
    for y in 0..gh {
        for x in 0..gw {
            if !mask[y * gw + x] {
                continue;
            }
            let mut all = true;
            'nb: for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= gw as i32 || ny >= gh as i32 || !mask[ny as usize * gw + nx as usize] {
                        all = false;
                        break 'nb;
                    }
                }
            }
            out[y * gw + x] = all;
        }
    }
    out
}

/// Square dilation with a `r`-cell radius (Chebyshev distance).
fn dilate(mask: &[bool], gw: usize, gh: usize, r: usize) -> Vec<bool> {
    if r == 0 {
        return mask.to_vec();
    }
    // separable: horizontal pass then vertical pass, each O(n * r)
    let mut tmp = vec![false; mask.len()];
    for y in 0..gh {
        let row = y * gw;
        for x in 0..gw {
            if !mask[row + x] {
                continue;
            }
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(gw - 1);
            for xx in lo..=hi {
                tmp[row + xx] = true;
            }
        }
    }
    let mut out = vec![false; mask.len()];
    for y in 0..gh {
        for x in 0..gw {
            if !tmp[y * gw + x] {
                continue;
            }
            let lo = y.saturating_sub(r);
            let hi = (y + r).min(gh - 1);
            for yy in lo..=hi {
                out[yy * gw + x] = true;
            }
        }
    }
    out
}

/// Morphological closing with radius `r`: dilate then erode, which joins
/// fragments that are at most `2r` cells apart while keeping the outer
/// boundary at roughly the true object extent.
fn morph_close(mask: &[bool], gw: usize, gh: usize, r: usize) -> (Vec<bool>, Vec<bool>) {
    let d = dilate(mask, gw, gh, r);
    let closed = if r == 0 { d.clone() } else { erode(&d, gw, gh) };
    (closed, d)
}

/// Number of `true` cells of `mask` inside the coarse box `b`.
fn count_mask(mask: &[bool], gw: usize, b: Bbox) -> usize {
    let x1 = b.x1.max(0.0) as usize;
    let y1 = b.y1.max(0.0) as usize;
    let x2 = (b.x2.max(0.0) as usize).min(gw);
    let y2 = (b.y2.max(0.0) as usize).max(y1);
    let mut n = 0;
    let gh = mask.len() / gw.max(1);
    for y in y1..y2.min(gh) {
        for x in x1..x2 {
            if mask[y * gw + x] {
                n += 1;
            }
        }
    }
    n
}

/// 4-connected component labelling returning bounding boxes (coarse units)
/// and the number of cells per component.
fn components(mask: &[bool], gw: usize, gh: usize) -> (Vec<Bbox>, Vec<usize>) {    let mut visited = vec![false; mask.len()];
    let mut boxes = Vec::new();
    let mut areas = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || visited[start] {
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
            let nb = [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ];
            for (nx, ny) in nb {
                if nx < gw && ny < gh {
                    let ni = ny * gw + nx;
                    if mask[ni] && !visited[ni] {
                        visited[ni] = true;
                        stack.push(ni);
                    }
                }
            }
        }
        if cells >= 3 {
            boxes.push(Bbox::new(x1 as f32, y1 as f32, x2 as f32, y2 as f32));
            areas.push(cells);
        }
    }
    (boxes, areas)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_with_blob(w: u32, h: u32, bx: u32, by: u32, bw: u32, bh: u32) -> Vec<u8> {
        let mut f = vec![200u8; (w * h * 3) as usize];
        for y in by..(by + bh) {
            for x in bx..(bx + bw) {
                let i = ((y * w + x) * 3) as usize;
                f[i] = 40;
                f[i + 1] = 40;
                f[i + 2] = 40;
            }
        }
        f
    }

    #[test]
    fn finds_dark_blob() {
        let (w, h) = (320, 240);
        let mut d = BlobDetector::new(w, h);
        let bg = frame_with_blob(w, h, 0, 0, 0, 0);
        for _ in 0..HISTORY + 1 {
            d.update(&bg);
        }
        let f = frame_with_blob(w, h, 100, 60, 60, 40);
        d.update(&f);
        let dets = d.detections();
        assert_eq!(dets.len(), 1, "expected a single blob, got {}", dets.len());
        // The morphological closing pads the box by up to `merge_gap` cells
        // on each side; require that the true object is fully contained and
        // that the box is not wildly larger.
        let pad = 2.0 * d.merge_gap as f32 * d.cell as f32;
        let b = dets[0].bbox;
        assert!(b.x1 <= 100.0 && b.y1 <= 60.0, "box must contain the object: {b:?}");
        assert!(
            b.x2 >= 160.0 && b.y2 >= 100.0,
            "box must contain the object: {b:?}"
        );
        assert!(
            b.x1 > 100.0 - pad && b.y1 > 60.0 - pad,
            "box padded too much: {b:?}"
        );
    }

    #[test]
    fn finds_two_blobs() {
        let (w, h) = (320, 240);
        let mut d = BlobDetector::new(w, h);
        let bg = frame_with_blob(w, h, 0, 0, 0, 0);
        for _ in 0..HISTORY + 1 {
            d.update(&bg);
        }
        let mut f = frame_with_blob(w, h, 40, 40, 50, 40);
        for y in 150..200 {
            for x in 200..260 {
                let i = ((y * w + x) * 3) as usize;
                f[i] = 40;
                f[i + 1] = 40;
                f[i + 2] = 40;
            }
        }
        d.update(&f);
        let dets = d.detections();
        assert_eq!(dets.len(), 2, "{dets:?}");
    }

    #[test]
    fn empty_scene_no_detection() {
        let (w, h) = (320, 240);
        let mut d = BlobDetector::new(w, h);
        let bg = frame_with_blob(w, h, 0, 0, 0, 0);
        // A perfectly noise-free static scene does report foreground during
        // warm-up (the later "frozen" cue needs a few frames to settle); what
        // matters is that nothing survives filtering into a detection.
        for _ in 0..HISTORY * 4 {
            d.update(&bg);
        }
        assert!(d.detections().is_empty());
    }
}

#[cfg(test)]
mod median_tests {
    use super::*;

    fn scene(w: u32, h: u32, blob: Option<(u32, u32)>) -> Vec<u8> {
        let mut f = vec![210u8; (w * h * 3) as usize];
        if let Some((bx, by)) = blob {
            for y in by..(by + 50).min(h) {
                for x in bx..(bx + 60).min(w) {
                    let i = ((y * w + x) * 3) as usize;
                    f[i] = 50;
                    f[i + 1] = 50;
                    f[i + 2] = 50;
                }
            }
        }
        f
    }

    #[test]
    fn object_that_enters_and_stays_is_detected() {
        // The temporal median deliberately cannot see an object that is
        // present from the very first frame (it is indistinguishable from the
        // background). It must, however, keep seeing an object that enters
        // the empty scene afterwards and then stops moving.
        let (w, h) = (320u32, 240u32);
        let mut d = BlobDetector::new(w, h);
        for _ in 0..HISTORY + 2 {
            d.update(&scene(w, h, None));
        }
        let still = scene(w, h, Some((200, 120)));
        for _ in 0..HISTORY * 4 {
            d.update(&still);
        }
        let dets = d.detections();
        assert_eq!(dets.len(), 1, "stationary object was absorbed: {dets:?}");
        let b = dets[0].bbox;
        assert!((b.x1 - 200.0).abs() < 20.0, "{b:?}");
    }

    #[test]
    fn empty_scene_stays_clean() {
        let (w, h) = (320u32, 240u32);
        let mut d = BlobDetector::new(w, h);
        for _ in 0..HISTORY * 3 {
            d.update(&scene(w, h, None));
        }
        assert!(d.detections().is_empty());
        assert!(d.masked_ratio() < 0.01);
    }
}
