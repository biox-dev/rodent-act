//! Pre-processing shared by the detector and the tracker.
//!
//! The network is a YOLO11 detector exported through Ultralytics, which
//! expects a **letterboxed** (aspect preserved, grey padding) float32 NCHW
//! tensor with values in `[0, 1]`. The same geometry is used for every frame,
//! so we can compute the transform once at start-up.

use crate::common::Bbox;

/// Geometry mapping original frames <-> network input.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxCfg {
    pub src_w: u32,
    pub src_h: u32,
    pub dst_w: u32,
    pub dst_h: u32,
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

impl LetterboxCfg {
    pub fn new(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Self {
        let scale = (dst_w as f32 / src_w as f32).min(dst_h as f32 / src_h as f32);
        let new_w = src_w as f32 * scale;
        let new_h = src_h as f32 * scale;
        Self {
            src_w,
            src_h,
            dst_w,
            dst_h,
            scale,
            pad_x: (dst_w as f32 - new_w) * 0.5,
            pad_y: (dst_h as f32 - new_h) * 0.5,
        }
    }

    /// Build the NCHW tensor (`[1, 3, dst_h, dst_w]`) from a packed rgb24 frame.
    pub fn to_tensor(&self, rgb: &[u8]) -> Vec<f32> {
        let (dw, dh) = (self.dst_w as usize, self.dst_h as usize);
        let mut out = vec![114.0 / 255.0; 3 * dw * dh]; // grey padding
        let plane = dw * dh;

        for y in 0..dh {
            // inverse map: network pixel -> source pixel (nearest neighbour)
            let sy = ((y as f32 - self.pad_y) / self.scale) as i64;
            // precompute channel offsets
            let row_net = y * dw;
            for x in 0..dw {
                let sx = ((x as f32 - self.pad_x) / self.scale) as i64;
                if sx < 0 || sy < 0 || sx >= self.src_w as i64 || sy >= self.src_h as i64 {
                    continue;
                }
                let src = ((sy as usize * self.src_w as usize) + sx as usize) * 3;
                let dst = row_net + x;
                out[dst] = rgb[src] as f32 / 255.0;
                out[plane + dst] = rgb[src + 1] as f32 / 255.0;
                out[2 * plane + dst] = rgb[src + 2] as f32 / 255.0;
            }
        }
        out
    }

    /// Network-input box -> original frame box.
    pub fn to_source(&self, b: Bbox) -> Bbox {
        Bbox::new(
            ((b.x1 - self.pad_x) / self.scale).clamp(0.0, self.src_w as f32),
            ((b.y1 - self.pad_y) / self.scale).clamp(0.0, self.src_h as f32),
            ((b.x2 - self.pad_x) / self.scale).clamp(0.0, self.src_w as f32),
            ((b.y2 - self.pad_y) / self.scale).clamp(0.0, self.src_h as f32),
        )
    }

    /// Original frame box -> network-input box.
    pub fn to_net(&self, b: Bbox) -> Bbox {
        Bbox::new(
            b.x1 * self.scale + self.pad_x,
            b.y1 * self.scale + self.pad_y,
            b.x2 * self.scale + self.pad_x,
            b.y2 * self.scale + self.pad_y,
        )
    }
}
