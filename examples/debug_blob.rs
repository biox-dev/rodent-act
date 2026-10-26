//! Debug helper: dump blob detections for a frame as JSON lines.
//!
//! cargo run --example debug_blob -- media/mouse_cage_20s.mp4 150 22

use rodent_act::debug_api::*;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let video = args.get(1).cloned().unwrap_or_else(|| "media/mouse_cage_20s.mp4".into());
    let frame_idx: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(150);
    let threshold: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(22);
    let min_area: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0015);

    let mut reader = video_io::VideoReader::open(std::path::Path::new(&video), 640)?;
    let mut det = detect::blob::BlobDetector::new(reader.width, reader.height);
    det.diff_threshold = threshold;
    det.min_area_ratio = min_area;

    let mut count = 0;
    while let Some(f) = reader.next_frame()? {
        let any = det.update(&f.data);
        if f.index == frame_idx {
            let dets = det.detections();
            println!("frame {} any={} dets={} mask_ratio={:.4}", f.index, any, dets.len(), det.masked_ratio());
            for d in &dets {
                println!(
                    "  box {:>5.0},{:>5.0} {:>4.0}x{:<4.0} score {:.2}",
                    d.bbox.x1,
                    d.bbox.y1,
                    d.bbox.wh().0,
                    d.bbox.wh().1,
                    d.score
                );
            }
            image::save_buffer("out/blob_debug.png", &f.data, f.width, f.height, image::ColorType::Rgb8)?;
            break;
        }
        count += 1;
        if count > frame_idx + 10 {
            break;
        }
    }
    Ok(())
}
