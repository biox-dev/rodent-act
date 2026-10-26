//! Small shared types used across modules.

use serde::{Deserialize, Serialize};

/// Axis aligned bounding box in pixel coordinates of the *original* frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bbox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl Bbox {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            x1: x1.min(x2),
            y1: y1.min(y2),
            x2: x1.max(x2),
            y2: y1.max(y2),
        }
    }

    /// Width/height. Never negative.
    pub fn wh(&self) -> (f32, f32) {
        ((self.x2 - self.x1).max(0.0), (self.y2 - self.y1).max(0.0))
    }

    pub fn area(&self) -> f32 {
        let (w, h) = self.wh();
        w * h
    }

    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }

    /// Width / height (guard against zero).
    pub fn aspect(&self) -> f32 {
        let (w, h) = self.wh();
        w / h.max(1e-3)
    }

    pub fn iou(&self, other: &Bbox) -> f32 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);
        let iw = (ix2 - ix1).max(0.0);
        let ih = (iy2 - iy1).max(0.0);
        let inter = iw * ih;
        let union = self.area() + other.area() - inter;
        if union <= 0.0 { 0.0 } else { inter / union }
    }

    pub fn clip(&self, width: u32, height: u32) -> Bbox {
        Bbox::new(
            self.x1.clamp(0.0, width as f32),
            self.y1.clamp(0.0, height as f32),
            self.x2.clamp(0.0, width as f32),
            self.y2.clamp(0.0, height as f32),
        )
    }

    pub fn scaled(&self, s: f32) -> Bbox {
        Bbox::new(self.x1 * s, self.y1 * s, self.x2 * s, self.y2 * s)
    }

    pub fn rounded(&self) -> [u32; 4] {
        [
            self.x1.round().max(0.0) as u32,
            self.y1.round().max(0.0) as u32,
            self.x2.round().max(0.0) as u32,
            self.y2.round().max(0.0) as u32,
        ]
    }
}

/// A single object detection in original frame coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub bbox: Bbox,
    pub score: f32,
    /// Class index as produced by the network (COCO class id for YOLO).
    pub class_id: usize,
}
