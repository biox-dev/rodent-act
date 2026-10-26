//! Multi-object tracker: constant-velocity Kalman filter + Hungarian
//! assignment + IoU/distance gating + track lifecycle management.
//!
//! The design follows a simplified SORT/ByteTrack hybrid:
//! * detections are associated to existing tracks by a mixed
//!   IoU / centre-distance cost,
//! * unmatched tracks coast (predict-only) for up to `max_age` frames,
//! * unmatched detections spawn new tracks once they are `min_hits` consistent,
//! * tracks carry a `TrackPoint` history so features can be extracted later.

use std::collections::VecDeque;

pub mod hungarian;
pub mod kalman;

use crate::common::{Bbox, Detection};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Lost,
}

#[derive(Debug, Clone)]
pub struct TrackPoint {
    pub frame: usize,
    /// Smoothed (Kalman) box.
    pub bbox: Bbox,
    /// Matched detection box (`None` when coasting).
    pub measured: bool,
    pub score: f32,
    pub class_id: usize,
}

#[derive(Debug, Clone)]
pub struct Track {
    pub id: usize,
    pub class_id: usize,
    pub state: TrackState,
    pub kalman: crate::track::kalman::KalmanState,
    pub hits: usize,
    pub age: usize,
    pub time_since_update: usize,
    pub history: VecDeque<TrackPoint>,
    pub last_score: f32,
}

impl Track {
    pub fn bbox(&self) -> Bbox {
        crate::track::kalman::z_to_bbox(&self.kalman.x)
    }

    pub fn velocity(&self) -> (f32, f32) {
        (self.kalman.x[4], self.kalman.x[5])
    }

    pub fn is_confirmed(&self) -> bool {
        self.state != TrackState::Tentative
    }
}

#[derive(Debug, Clone)]
pub struct TrackerConfig {
    /// Minimum detections before a track becomes `Confirmed`.
    pub min_hits: usize,
    /// Frames a track survives without measurements.
    pub max_age: usize,
    /// Cost (1 - IoU) above which a match is rejected.
    pub max_iou_cost: f32,
    /// Maximum normalised centre distance for a valid match. The distance is
    /// divided by the *average* of the two box diagonals, so a fast, small
    /// animal is tracked as well as a slow, large one.
    pub max_dist_in_heights: f32,
    /// How many historical points to keep per track.
    pub history_len: usize,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            min_hits: 3,
            max_age: 30,
            max_iou_cost: 0.85,
            max_dist_in_heights: 1.6,
            history_len: 900,
        }
    }
}

pub struct MultiTracker {
    pub tracks: Vec<Track>,
    next_id: usize,
    kf: crate::track::kalman::KalmanFilter,
    cfg: TrackerConfig,
    frame: usize,
}

impl MultiTracker {
    pub fn new(cfg: TrackerConfig) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            kf: crate::track::kalman::KalmanFilter::new(),
            cfg,
            frame: 0,
        }
    }

    pub fn config(&self) -> &TrackerConfig {
        &self.cfg
    }

    /// Advance the tracker by one frame.
    /// `detections` are expected in original-frame coordinates.
    pub fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        self.frame += 1;

        // 1) predict all existing tracks
        for t in self.tracks.iter_mut() {
            t.kalman = self.kf.predict(&t.kalman);
            t.age += 1;
            t.time_since_update += 1;
        }

        // 2) build cost matrix (rows = tracks, cols = detections)
        let track_pred: Vec<Bbox> = self.tracks.iter().map(|t| t.bbox()).collect();
        let mut cost = vec![vec![1e6f32; detections.len()]; self.tracks.len()];
        for (ti, tb) in track_pred.iter().enumerate() {
            for (di, d) in detections.iter().enumerate() {
                if self.tracks[ti].class_id != d.class_id {
                    continue;
                }
                let iou = tb.iou(&d.bbox);
                let (tcx, tcy) = tb.center();
                let (dcx, dcy) = d.bbox.center();
                let dist = ((tcx - dcx).powi(2) + (tcy - dcy).powi(2)).sqrt();
                let (tw, th) = tb.wh();
                let (dw, dh) = d.bbox.wh();
                // Scale-normalised distance: 0 = identical centre, 1 = the
                // centres are one body-diagonal apart.
                let tdiag = (tw * tw + th * th).sqrt().max(1.0);
                let ddiag = (dw * dw + dh * dh).sqrt().max(1.0);
                let scale = 0.5 * (tdiag + ddiag);
                let ndist = dist / scale;
                if ndist > self.cfg.max_dist_in_heights {
                    continue;
                }
                // Cost blends overlap (shape consistency) with normalised
                // distance (motion continuity). Distance dominates because a
                // small, fast object barely overlaps its previous box.
                let c = 0.25 * (1.0 - iou) + 0.75 * (ndist / self.cfg.max_dist_in_heights);
                if c > self.cfg.max_iou_cost {
                    continue;
                }
                cost[ti][di] = c;
            }
        }

        // 3) solve assignment
        let assignment = if !self.tracks.is_empty() && !detections.is_empty() {
            hungarian::assign(&cost)
        } else {
            vec![None; self.tracks.len()]
        };

        let mut det_used = vec![false; detections.len()];
        let mut matched_tracks = vec![false; self.tracks.len()];

        for (ti, maybe_di) in assignment.iter().enumerate() {
            let Some(di) = *maybe_di else { continue };
            det_used[di] = true;
            matched_tracks[ti] = true;
            let d = &detections[di];
            let t = &mut self.tracks[ti];
            t.kalman = self
                .kf
                .update(&t.kalman, crate::track::kalman::bbox_to_z(&d.bbox));
            t.hits += 1;
            t.time_since_update = 0;
            t.last_score = d.score;
            if t.state == TrackState::Tentative && t.hits >= self.cfg.min_hits {
                t.state = TrackState::Confirmed;
            } else if t.state == TrackState::Lost {
                t.state = TrackState::Confirmed;
            }
            push_point(t, self.frame, d.bbox, true, d.score, d.class_id, self.cfg.history_len);
        }

        // 4) coast unmatched, but still record the predicted box
        for (ti, t) in self.tracks.iter_mut().enumerate() {
            if matched_tracks[ti] {
                continue;
            }
            if t.time_since_update > self.cfg.max_age {
                t.state = TrackState::Lost;
            }
            let predicted = t.bbox();
            let class_id = t.class_id;
            let score = t.last_score;
            push_point(t, self.frame, predicted, false, score, class_id, self.cfg.history_len);
        }

        // 5) spawn tracks for unmatched detections
        for (di, d) in detections.iter().enumerate() {
            if det_used[di] {
                continue;
            }
            let kalman = self.kf.initiate(crate::track::kalman::bbox_to_z(&d.bbox));
            let mut t = Track {
                id: self.next_id,
                class_id: d.class_id,
                state: if self.cfg.min_hits <= 1 {
                    TrackState::Confirmed
                } else {
                    TrackState::Tentative
                },
                kalman,
                hits: 1,
                age: 1,
                time_since_update: 0,
                history: VecDeque::with_capacity(self.cfg.history_len),
                last_score: d.score,
            };
            self.next_id += 1;
            push_point(
                &mut t,
                self.frame,
                d.bbox,
                true,
                d.score,
                d.class_id,
                self.cfg.history_len,
            );
            self.tracks.push(t);
        }

        // 6) drop dead tracks
        self.tracks
            .retain(|t| t.state != TrackState::Lost && t.time_since_update <= self.cfg.max_age);

        self.tracks
            .iter()
            .filter(|t| t.is_confirmed())
            .cloned()
            .collect()
    }

    /// Force-confirm everything (useful while a camera is being tuned).
    pub fn confirm_all(&mut self) {
        for t in self.tracks.iter_mut() {
            if t.state == TrackState::Tentative {
                t.state = TrackState::Confirmed;
            }
        }
    }

    pub fn frame_index(&self) -> usize {
        self.frame
    }
}

fn push_point(
    t: &mut Track,
    frame: usize,
    bbox: Bbox,
    measured: bool,
    score: f32,
    class_id: usize,
    cap: usize,
) {
    if t.history.len() >= cap {
        t.history.pop_front();
    }
    t.history.push_back(TrackPoint {
        frame,
        bbox,
        measured,
        score,
        class_id,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, w: f32, h: f32) -> Detection {
        Detection {
            bbox: Bbox::new(x, y, x + w, y + h),
            score: 0.9,
            class_id: 64,
        }
    }

    #[test]
    fn keeps_single_identity() {
        let mut mt = MultiTracker::new(TrackerConfig::default());
        let mut last = Vec::new();
        for i in 0..20 {
            last = mt.update(&[det(10.0 + i as f32 * 2.0, 10.0, 30.0, 20.0)]);
        }
        assert_eq!(last.len(), 1);
        assert_eq!(mt.tracks.len(), 1);
    }

    #[test]
    fn tracks_two_objects() {
        let mut mt = MultiTracker::new(TrackerConfig::default());
        let mut out = Vec::new();
        for i in 0..15 {
            out = mt.update(&[
                det(10.0 + i as f32, 10.0, 20.0, 20.0),
                det(200.0, 10.0 - i as f32, 20.0, 20.0),
            ]);
        }
        assert_eq!(out.len(), 2);
        let ids: std::collections::HashSet<_> = out.iter().map(|t| t.id).collect();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn coasts_through_gap() {
        let mut mt = MultiTracker::new(TrackerConfig::default());
        for i in 0..10 {
            mt.update(&[det(10.0 + i as f32 * 3.0, 10.0, 20.0, 20.0)]);
        }
        let id_before = mt.tracks[0].id;
        for _ in 0..5 {
            mt.update(&[]); // gap
        }
        let out = mt.update(&[det(60.0, 10.0, 20.0, 20.0)]);
        assert_eq!(out[0].id, id_before, "identity must survive the gap");
    }
}
