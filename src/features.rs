//! Behaviour features per 0.5 s window.
//!
//! The tail-suspension test (TST) is scored, in short time slices, as
//! **immobile** or **mobile**. This module turns a track's per-frame boxes into
//! fixed-length windows and computes the descriptor consumed by the classifier.
//!
//! Differences from the classic "activity score" (a.k.a. RII - *refined index of
//! immobility*, the 1-pixel/frame threshold published for ANADE): every
//! displacement here is normalised by the animal's own median body diagonal, so
//! the features are invariant to camera distance and resolution, and a 0.5 s
//! window can be judged either by the classic *fraction of frames with
//! displacement > threshold* or by the learned classifier.
//!
//! Keep [`FEATURE_NAMES`] in sync with `analytics/features.py` and with
//! `analytics/train_xgb.py` - the classifier indexes features positionally.

use serde::{Deserialize, Serialize};

use crate::common::Bbox;
use crate::track::Track;

/// Feature order. **Do not reorder** once a model has been trained.
pub const FEATURE_NAMES: &[&str] = &[
    "speed_mean",
    "speed_std",
    "speed_max",
    "path_length",
    "displacement",
    "path_efficiency",
    "turn_abs_mean",
    "turn_abs_max",
    "stationary_fraction",
    "micro_move_ratio",
    "bbox_area_cv",
    "aspect_mean",
    "aspect_std",
    "motion_energy",
    "velocity_autocorr1",
    "n_jumps",
    "valid_ratio",
    "size_norm",
];

/// Default window length for TST scoring.
pub const DEFAULT_WINDOW_S: f64 = 0.5;

/// Speed (body lengths / s) below which a frame counts as fully stationary.
pub const MICRO_SPEED: f32 = 0.08;

/// Speed (body lengths / s) below which a frame counts as (nearly) stationary.
pub const STATIONARY_SPEED: f32 = 0.30;

/// Version of the *reference* RII criterion: fraction of frames whose centroid
/// moves less than 1 pixel within a 0.5 s window (data-driven fallback below).
pub const RII_PIXEL_THRESHOLD: f32 = 1.0;

/// One window's feature vector plus metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowFeatures {
    pub track_id: usize,
    pub start_frame: usize,
    pub end_frame: usize,
    pub start_t: f64,
    pub end_t: f64,
    pub names: Vec<String>,
    pub values: Vec<f32>,
}

impl WindowFeatures {
    pub fn get(&self, name: &str) -> Option<f32> {
        self.names
            .iter()
            .position(|n| n == name)
            .and_then(|i| self.values.get(i).copied())
    }
}

/// Reference size of a track: median body diagonal, used to normalise distance.
pub fn reference_size(track: &Track) -> f32 {
    let mut diags: Vec<f32> = track
        .history
        .iter()
        .map(|p| {
            let (w, h) = p.bbox.wh();
            (w * w + h * h).sqrt()
        })
        .collect();
    if diags.is_empty() {
        return 1.0;
    }
    diags.sort_by(|a, b| a.total_cmp(b));
    let m = diags.len() / 2;
    let med = if diags.len() % 2 == 0 {
        0.5 * (diags[m - 1] + diags[m])
    } else {
        diags[m]
    };
    med.max(1.0)
}

/// Compute the feature vector for a slice of boxes.
///
/// `samples` are `(timestamp_s, bbox)` pairs covering the window, in order.
pub fn compute_window(
    track_id: usize,
    start_frame: usize,
    end_frame: usize,
    samples: &[(f64, Bbox)],
    ref_size: f32,
    fps: f64,
) -> WindowFeatures {
    let names: Vec<String> = FEATURE_NAMES.iter().map(|s| s.to_string()).collect();
    if samples.len() < 2 {
        return WindowFeatures {
            track_id,
            start_frame,
            end_frame,
            start_t: samples.first().map(|s| s.0).unwrap_or(0.0),
            end_t: samples.last().map(|s| s.0).unwrap_or(0.0),
            names,
            values: vec![0.0; FEATURE_NAMES.len()],
        };
    }

    let dt = 1.0 / fps.max(1e-6);
    let ref_size = ref_size.max(1e-6);

    // --- kinematics -------------------------------------------------------
    let mut speeds = Vec::with_capacity(samples.len() - 1);
    let mut headings = Vec::with_capacity(samples.len() - 1);
    let mut jumps = 0usize;
    for w in samples.windows(2) {
        let (c0, c1) = (w[0].1.center(), w[1].1.center());
        let dx = c1.0 - c0.0;
        let dy = c1.1 - c0.1;
        let d = (dx * dx + dy * dy).sqrt();
        if d > 2.0 * ref_size {
            jumps += 1;
            speeds.push(0.0);
            continue;
        }
        speeds.push(d / ref_size / dt as f32);
        if d > 1e-3 {
            headings.push(dy.atan2(dx));
        }
    }

    let turns: Vec<f32> = headings
        .windows(2)
        .map(|h| {
            let mut d = h[1] - h[0];
            while d > std::f32::consts::PI {
                d -= 2.0 * std::f32::consts::PI;
            }
            while d < -std::f32::consts::PI {
                d += 2.0 * std::f32::consts::PI;
            }
            d.abs()
        })
        .collect();

    let path_length: f32 = speeds.iter().sum::<f32>() * dt as f32;
    let (c_first, c_last) = (samples[0].1.center(), samples[samples.len() - 1].1.center());
    let displacement =
        ((c_last.0 - c_first.0).powi(2) + (c_last.1 - c_first.1).powi(2)).sqrt() / ref_size;

    let sp = if speeds.is_empty() { vec![0.0] } else { speeds };
    let mean = sp.iter().sum::<f32>() / sp.len() as f32;
    let var = sp.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / sp.len() as f32;

    // The two fractions must *partition* the window: a frame with 0.15 body
    // lengths/s is neither fully stationary nor micro-moving.
    let stationary = sp.iter().filter(|s| **s < MICRO_SPEED).count() as f32 / sp.len() as f32;
    let micro = sp
        .iter()
        .filter(|s| **s >= MICRO_SPEED && **s < STATIONARY_SPEED)
        .count() as f32
        / sp.len() as f32;

    let areas: Vec<f32> = samples.iter().map(|s| s.1.area()).collect();
    let area_mean = areas.iter().sum::<f32>() / areas.len() as f32;
    let area_sd = (areas.iter().map(|a| (a - area_mean).powi(2)).sum::<f32>()
        / areas.len() as f32)
        .sqrt();
    let area_cv = if area_mean > 1e-6 { area_sd / area_mean } else { 0.0 };

    let aspects: Vec<f32> = samples.iter().map(|s| s.1.aspect()).collect();
    let asp_mean = aspects.iter().sum::<f32>() / aspects.len() as f32;
    let asp_sd = (aspects.iter().map(|a| (a - asp_mean).powi(2)).sum::<f32>()
        / aspects.len() as f32)
        .sqrt();

    let autocorr = if sp.len() > 2 {
        let num: f32 = sp.windows(2).map(|w| (w[0] - mean) * (w[1] - mean)).sum();
        let den: f32 = sp.iter().map(|s| (s - mean).powi(2)).sum();
        if den > 1e-9 { num / den } else { 0.0 }
    } else {
        0.0
    };

    let values = vec![
        mean,
        var.sqrt(),
        sp.iter().cloned().fold(0.0f32, f32::max),
        path_length,
        displacement,
        if path_length > 1e-9 { displacement / path_length } else { 0.0 },
        if turns.is_empty() { 0.0 } else { turns.iter().sum::<f32>() / turns.len() as f32 },
        turns.iter().cloned().fold(0.0f32, f32::max),
        stationary,
        micro,
        area_cv,
        asp_mean,
        asp_sd,
        0.0, // motion_energy: filled in by the pipeline from the pixel mask
        autocorr,
        jumps as f32,
        1.0, // valid_ratio: pipeline overwrites with measured-frame fraction
        (ref_size / (640.0f32.powi(2) + 360.0f32.powi(2)).sqrt()).clamp(0.0, 1.0),
    ];

    WindowFeatures {
        track_id,
        start_frame,
        end_frame,
        start_t: samples[0].0,
        end_t: samples[samples.len() - 1].0,
        names,
        values,
    }
}

/// Split a track's history into windows and compute features for each.
pub fn windows_for_track(
    track: &Track,
    fps: f64,
    window_s: f64,
    stride_s: f64,
    frame_w: u32,
    frame_h: u32,
) -> Vec<WindowFeatures> {
    let hist: Vec<_> = track.history.iter().collect();
    if hist.len() < 2 {
        return Vec::new();
    }
    let ref_size = reference_size(track);
    let frame_diag = ((frame_w * frame_w + frame_h * frame_h) as f32).sqrt();
    let n = ((window_s * fps).round() as usize).max(2);
    let stride = ((stride_s * fps).round() as usize).max(1);

    let mut out = Vec::new();
    let mut i = 0usize;
    while i + n <= hist.len() {
        let slice = &hist[i..i + n];
        let samples: Vec<(f64, Bbox)> = slice
            .iter()
            .map(|p| (p.frame as f64 / fps, p.bbox))
            .collect();
        let mut w = compute_window(
            track.id,
            slice[0].frame,
            slice[slice.len() - 1].frame,
            &samples,
            ref_size,
            fps,
        );
        // overwrite the two features that need extra context
        if let Some(idx) = w.names.iter().position(|n| n == "valid_ratio") {
            let measured = slice.iter().filter(|p| p.measured).count() as f32;
            w.values[idx] = measured / slice.len() as f32;
        }
        if let Some(idx) = w.names.iter().position(|n| n == "size_norm") {
            w.values[idx] = (ref_size / frame_diag).clamp(0.0, 1.0);
        }
        out.push(w);
        i += stride;
    }
    out
}

/// Classic RII score for a window: fraction of frames with < threshold motion.
///
/// This is the *reference* metric TST papers report, kept for validation and
/// for producing pseudo-labels before a trained classifier exists.
pub fn rii_score(samples: &[(f64, Bbox)], pixel_threshold: f32) -> f32 {
    if samples.len() < 2 {
        return 1.0;
    }
    let still = samples
        .windows(2)
        .filter(|w| {
            let (c0, c1) = (w[0].1.center(), w[1].1.center());
            ((c1.0 - c0.0).powi(2) + (c1.1 - c0.1).powi(2)).sqrt() < pixel_threshold
        })
        .count();
    still as f32 / (samples.len() - 1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bbox(cx: f32, cy: f32) -> Bbox {
        Bbox::new(cx - 20.0, cy - 10.0, cx + 20.0, cy + 10.0)
    }

    #[test]
    fn still_object_has_no_speed() {
        let samples: Vec<(f64, Bbox)> = (0..25).map(|i| (i as f64 * 0.04, bbox(100.0, 100.0))).collect();
        let w = compute_window(1, 0, 24, &samples, 45.0, 25.0);
        assert!(w.get("speed_mean").unwrap() < 1e-6);
        assert!((w.get("stationary_fraction").unwrap() - 1.0).abs() < 1e-6);
        assert!((w.get("displacement").unwrap() - 0.0).abs() < 1e-6);
        assert_eq!(rii_score(&samples, 1.0), 1.0);
    }

    #[test]
    fn moving_object_has_speed_and_path() {
        // 2 px per frame at 25 fps, ref size 45 px -> 2/45*25 = 1.11 body/s
        let samples: Vec<(f64, Bbox)> = (0..25)
            .map(|i| (i as f64 * 0.04, bbox(100.0 + i as f32 * 2.0, 100.0)))
            .collect();
        let w = compute_window(1, 0, 24, &samples, 45.0, 25.0);
        let speed = w.get("speed_mean").unwrap();
        assert!((speed - 1.111).abs() < 0.05, "speed={speed}");
        assert!(w.get("path_length").unwrap() > 0.5);
        assert!((w.get("path_efficiency").unwrap() - 1.0).abs() < 1e-3);
        assert_eq!(rii_score(&samples, 1.0), 0.0);
    }

    #[test]
    fn feature_vector_matches_names() {
        let samples: Vec<(f64, Bbox)> = (0..5).map(|i| (i as f64 * 0.04, bbox(10.0, 10.0))).collect();
        let w = compute_window(1, 0, 4, &samples, 45.0, 25.0);
        assert_eq!(w.names.len(), FEATURE_NAMES.len());
        assert_eq!(w.values.len(), FEATURE_NAMES.len());
    }

    #[test]
    fn stationary_and_micro_fractions_partition() {
        // half still, half micro-moving
        let mut samples = Vec::new();
        let mut x = 100.0f32;
        for i in 0..25 {
            if i >= 12 {
                // 0.5 px per frame at 25 fps with a 45 px body
                // => 0.5 * 25 / 45 = 0.28 body lengths/s (the "micro" band)
                x += 0.5;
            }
            samples.push((i as f64 * 0.04, bbox(x, 100.0)));
        }
        let w = compute_window(1, 0, 24, &samples, 45.0, 25.0);
        let s = w.get("stationary_fraction").unwrap();
        let m = w.get("micro_move_ratio").unwrap();
        assert!(s > 0.3 && s < 0.7, "stationary={s}");
        assert!(m > 0.2, "micro={m}");
    }
}
