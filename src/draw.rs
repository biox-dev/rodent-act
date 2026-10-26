//! Frame annotation and preview output.
//!
//! Only plain RGB buffer operations here (no image crate in the hot path).

use std::collections::HashMap;
use std::f32::consts::PI;
use std::io::Write;

use anyhow::Result;
use image::{ImageBuffer, Rgb};

use crate::common::Bbox;
use crate::track::Track;

pub type Rgb3 = [u8; 3];

/// Deterministic, high-contrast colour per track id.
pub fn id_color(id: usize) -> Rgb3 {
    // golden-angle hue stepping in HSV, converted to RGB
    let hue = (id as f32 * 137.508) % 360.0;
    hsv_to_rgb(hue, 0.85, 1.0)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Rgb3 {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - ((hp % 2.0) - 1.0).abs());
    let (r, g, b) = match hp as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

pub struct Canvas<'a> {
    pub data: &'a mut [u8],
    pub width: u32,
    pub height: u32,
}

impl<'a> Canvas<'a> {
    pub fn new(data: &'a mut [u8], width: u32, height: u32) -> Self {
        Self { data, width, height }
    }

    #[inline]
    fn put(&mut self, x: i32, y: i32, c: Rgb3, alpha: f32) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let i = ((y as u32 * self.width + x as u32) * 3) as usize;
        if alpha >= 1.0 {
            self.data[i] = c[0];
            self.data[i + 1] = c[1];
            self.data[i + 2] = c[2];
        } else {
            for k in 0..3 {
                let old = self.data[i + k] as f32;
                self.data[i + k] = (old * (1.0 - alpha) + c[k] as f32 * alpha).clamp(0.0, 255.0) as u8;
            }
        }
    }

    pub fn hline(&mut self, x0: i32, x1: i32, y: i32, c: Rgb3, t: i32) {
        for dy in 0..t.max(1) {
            for x in x0.min(x1)..=x0.max(x1) {
                self.put(x, y + dy, c, 1.0);
            }
        }
    }

    pub fn vline(&mut self, y0: i32, y1: i32, x: i32, c: Rgb3, t: i32) {
        for dx in 0..t.max(1) {
            for y in y0.min(y1)..=y0.max(y1) {
                self.put(x + dx, y, c, 1.0);
            }
        }
    }

    pub fn line(&mut self, a: (i32, i32), b: (i32, i32), c: Rgb3, t: i32) {
        let (x0, y0) = a;
        let (x1, y1) = b;
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let (mut x, mut y) = (x0, y0);
        let mut err = dx + dy;
        loop {
            for oy in 0..t.max(1) {
                for ox in 0..t.max(1) {
                    self.put(x + ox, y + oy, c, 1.0);
                }
            }
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Anti-aliased-ish circle via four arcs.
    pub fn circle(&mut self, cx: i32, cy: i32, r: i32, c: Rgb3, t: i32) {
        let (mut x, mut y, mut err) = (r, 0, 1 - r);
        while x >= y {
            for (px, py) in [
                (cx + x, cy + y),
                (cx + y, cy + x),
                (cx - y, cy + x),
                (cx - x, cy + y),
                (cx - x, cy - y),
                (cx - y, cy - x),
                (cx + y, cy - x),
                (cx + x, cy - y),
            ] {
                for oy in 0..t {
                    for ox in 0..t {
                        self.put(px + ox, py + oy, c, 1.0);
                    }
                }
            }
            y += 1;
            if err < 0 {
                err += 2 * y + 1;
            } else {
                x -= 1;
                err += 2 * (y - x) + 1;
            }
        }
    }

    pub fn dashed_rect(&mut self, b: Bbox, c: Rgb3, t: i32) {
        let [x1, y1, x2, y2] = b.rounded();
        dash_line(self, (x1 as i32, y1 as i32), (x2 as i32, y1 as i32), c, t);
        dash_line(self, (x2 as i32, y1 as i32), (x2 as i32, y2 as i32), c, t);
        dash_line(self, (x2 as i32, y2 as i32), (x1 as i32, y2 as i32), c, t);
        dash_line(self, (x1 as i32, y2 as i32), (x1 as i32, y1 as i32), c, t);
    }

    pub fn rect(&mut self, b: Bbox, c: Rgb3, t: i32) {
        let [x1, y1, x2, y2] = b.rounded();
        self.hline(x1 as i32, x2 as i32, y1 as i32, c, t);
        self.hline(x1 as i32, x2 as i32, y2 as i32 - t, c, t);
        self.vline(y1 as i32, y2 as i32, x1 as i32, c, t);
        self.vline(y1 as i32, y2 as i32, x2 as i32 - t, c, t);
    }

    /// Semi-transparent fill, used to tint ROIs / heat overlays.
    pub fn fill_rect(&mut self, b: Bbox, c: Rgb3, alpha: f32) {
        let [x1, y1, x2, y2] = b.rounded();
        for y in y1 as i32..y2 as i32 {
            for x in x1 as i32..x2 as i32 {
                self.put(x, y, c, alpha);
            }
        }
    }

    /// Tiny 5x7 bitmap font, uppercase only (digits, letters, `# : . - / %`).
    pub fn text(&mut self, s: &str, x: i32, y: i32, c: Rgb3, scale: i32) {
        // dark backing box for legibility
        let w = (s.chars().count() as i32) * 6 * scale;
        let h = 8 * scale;
        for yy in (y - scale)..(y + h) {
            for xx in (x - scale)..(x + w) {
                self.put(xx, yy, [0, 0, 0], 0.55);
            }
        }

        let mut cx = x;
        for ch in s.chars() {
            let glyph = glyph_of(ch);
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if bits & (1 << (4 - col)) != 0 {
                        for sy in 0..scale {
                            for sx in 0..scale {
                                self.put(
                                    cx + col * scale + sx,
                                    y + row as i32 * scale + sy,
                                    c,
                                    1.0,
                                );
                            }
                        }
                    }
                }
            }
            cx += 6 * scale;
        }
    }
}

fn dash_line(c: &mut Canvas, a: (i32, i32), b: (i32, i32), color: Rgb3, t: i32) {
    let dx = (b.0 - a.0) as f32;
    let dy = (b.1 - a.1) as f32;
    let len = (dx * dx + dy * dy).sqrt().max(1.0);
    let steps = len as i32;
    for i in 0..steps {
        if (i / 6) % 2 == 0 {
            let x = a.0 + (dx * i as f32 / len) as i32;
            let y = a.1 + (dy * i as f32 / len) as i32;
            for oy in 0..t {
                for ox in 0..t {
                    c.put(x + ox, y + oy, color, 1.0);
                }
            }
        }
    }
}

/// 5x7 glyphs, one byte per row (low 5 bits).
fn glyph_of(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        'A' => [0b00100, 0b01010, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        '#' => [0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010],
        ' ' => [0; 7],
        _ => [0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111],
    }
}

/// What to draw for one tracked object.
pub struct DrawTrack<'a> {
    pub track: &'a Track,
    pub color: Rgb3,
    pub label: String,
    pub trail_len: usize,
}

pub struct FrameOverlay<'a> {
    pub frame_index: usize,
    pub timestamp_s: f64,
    pub fps: f64,
    /// Regions the detector was asked to analyse (for debugging).
    pub rois: &'a [Bbox],
    pub show_rois: bool,
    pub show_trails: bool,
    pub show_skeleton_axes: bool,
}

/// Draw tracks (boxes + labels + trails) and the HUD.
pub fn draw_frame(
    rgb: &mut [u8],
    width: u32,
    height: u32,
    tracks: &[DrawTrack<'_>],
    overlay: &FrameOverlay<'_>,
    _labels: &HashMap<usize, String>,
) {
    let mut c = Canvas::new(rgb, width, height);

    if overlay.show_rois {
        for r in overlay.rois {
            c.dashed_rect(*r, [0, 200, 255], 1);
        }
    }

    if overlay.show_skeleton_axes {
        let cx = (width / 2) as i32;
        let cy = (height / 2) as i32;
        c.line((cx - 30, cy), (cx + 30, cy), [255, 255, 255], 1);
        c.line((cx, cy - 30), (cx, cy + 30), [255, 255, 255], 1);
    }

    for dt in tracks {
        let b = dt.track.bbox();
        c.rect(b, dt.color, 2);
        let (cx, cy) = b.center();
        c.circle(cx as i32, cy as i32, 3, dt.color, 1);

        if overlay.show_trails {
            let hist = &dt.track.history;
            let n = hist.len().min(dt.trail_len);
            let pts: Vec<(i32, i32)> = hist
                .iter()
                .skip(hist.len() - n)
                .filter(|p| p.measured)
                .map(|p| {
                    let (x, y) = p.bbox.center();
                    (x as i32, y as i32)
                })
                .collect();
            for w in pts.windows(2) {
                c.line(w[0], w[1], dt.color, 2);
            }
        }

        let [x1, y1, ..] = b.rounded();
        let ly = (y1 as i32 - 20).max(2);
        c.text(&dt.label, x1 as i32, ly, dt.color, 2);
    }

    // HUD
    let hud = format!(
        "F {}  T {:.2}S  {} TRACKS",
        overlay.frame_index,
        overlay.frame_index as f64 / overlay.fps,
        tracks.len()
    );
    c.text(&hud, 8, 8, [255, 255, 0], 2);
    c.text("RODENT-ACT", 8, height as i32 - 24, [0, 255, 128], 2);
}

/// Convert to a half-size preview and write a PNG (for quick inspection).
pub fn save_preview_png(path: &str, rgb: &[u8], width: u32, height: u32) -> Result<()> {
    let img: ImageBuffer<Rgb<u8>, _> =
        ImageBuffer::from_raw(width, height, rgb.to_vec()).expect("buffer size mismatch");
    img.save(path)?;
    Ok(())
}

/// Write an ASCII-art preview to stderr (works over SSH without X11).
pub fn print_ascii_preview(rgb: &[u8], width: u32, height: u32, cols: u32) -> Result<()> {
    const RAMP: &[u8] = b" .:-=+*#%@";
    let cols = cols.min(width);
    let rows = (cols as f32 * height as f32 / width as f32 * 0.5) as u32;
    let mut out = String::with_capacity((cols * rows * 2) as usize);
    for r in 0..rows {
        for cidx in 0..cols {
            let x = (cidx as f32 * width as f32 / cols as f32) as u32;
            let y = (r as f32 * height as f32 / rows as f32) as u32;
            let i = ((y.min(height - 1) * width + x.min(width - 1)) * 3) as usize;
            let luma = (rgb[i] as u32 * 77 + rgb[i + 1] as u32 * 150 + rgb[i + 2] as u32 * 29) / 256;
            let idx = (luma as usize * (RAMP.len() - 1)) / 255;
            out.push(RAMP[idx] as char);
            out.push(RAMP[idx] as char);
        }
        out.push('\n');
    }
    std::io::stderr().write_all(out.as_bytes())?;
    std::io::stderr().flush()?;
    Ok(())
}

const _: () = {
    // keep PI import meaningful for future angle annotations
    let _ = PI;
};
