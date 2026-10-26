//! YOLO11 ONNX detector decoding.
//!
//! Output layout (`output0`) is `[1, 84, N]` where the first 4 rows are
//! `cx, cy, w, h` in **network input pixels** and the remaining 80 rows are
//! class scores (already sigmoid-normalised by the export).

use crate::common::{Bbox, Detection};
use crate::preprocess::LetterboxCfg;

/// Minimal non-maximum suppression.
pub fn nms(mut dets: Vec<Detection>, iou_thr: f32, max_out: usize) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Detection> = Vec::with_capacity(max_out.min(dets.len()));
    for d in dets {
        if keep.len() >= max_out {
            break;
        }
        let suppressed = keep
            .iter()
            .any(|k| k.class_id == d.class_id && k.bbox.iou(&d.bbox) > iou_thr);
        if !suppressed {
            keep.push(d);
        }
    }
    keep
}

/// Decode a raw `[84, anchors]` output into detections in original-frame
/// coordinates.
pub fn decode(
    out: &[f32],
    rows: usize,
    anchors: usize,
    lb: &LetterboxCfg,
    score_thr: f32,
    iou_thr: f32,
    max_det: usize,
) -> Vec<Detection> {
    debug_assert_eq!(out.len(), rows * anchors);
    let num_classes = rows.saturating_sub(4);
    let mut raw = Vec::new();

    // Row-major [rows, anchors]: out[row * anchors + a]
    for a in 0..anchors {
        // fast class pre-filter
        let mut best_c = 0usize;
        let mut best_s = 0.0f32;
        for c in 0..num_classes {
            let s = out[(4 + c) * anchors + a];
            if s > best_s {
                best_s = s;
                best_c = c;
            }
        }
        if best_s < score_thr {
            continue;
        }
        let cx = out[a];
        let cy = out[anchors + a];
        let w = out[2 * anchors + a];
        let h = out[3 * anchors + a];
        if !(w.is_finite() && h.is_finite()) || w <= 0.0 || h <= 0.0 {
            continue;
        }
        let bbox = lb.to_source(Bbox::new(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0));
        if bbox.area() < 4.0 {
            continue;
        }
        raw.push(Detection {
            bbox,
            score: best_s,
            class_id: best_c,
        });
    }

    nms(raw, iou_thr, max_det)
}

/// Row count is `4 + num_classes`; sanity-check against the model.
pub fn expected_rows(num_classes: usize) -> usize {
    4 + num_classes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_single_box() {
        // 1 class model: 5 rows, 2 anchors
        let lb = LetterboxCfg::new(100, 100, 100, 100);
        let mut out = vec![0.0f32; 5 * 2];
        // anchor 0: box at (50,50) size 20x20 score 0.9
        out[0] = 50.0;
        out[2] = 50.0;
        out[4] = 20.0;
        out[6] = 20.0;
        out[8] = 0.9;
        let dets = decode(&out, 5, 2, &lb, 0.25, 0.45, 100);
        assert_eq!(dets.len(), 1);
        assert!((dets[0].bbox.x1 - 40.0).abs() < 1e-3);
    }

    #[test]
    fn nms_keeps_best() {
        let a = Detection { bbox: Bbox::new(0., 0., 10., 10.), score: 0.9, class_id: 0 };
        let b = Detection { bbox: Bbox::new(1., 1., 11., 11.), score: 0.5, class_id: 0 };
        let c = Detection { bbox: Bbox::new(50., 50., 60., 60.), score: 0.4, class_id: 0 };
        let keep = nms(vec![b, a, c], 0.45, 100);
        assert_eq!(keep.len(), 2);
        assert_eq!(keep[0].score, 0.9);
    }
}
