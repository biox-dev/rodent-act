//! Debug helper: run the detector on one frame from a video and print the
//! best candidates for every class, so thresholds/class ids can be checked.
//!
//! cargo run --example debug_detect -- <video> <frame> <score_thr> [model.onnx]
//!
//! Works with the stock COCO model and with a fine-tuned model (class names are
//! read from the ONNX metadata, so a custom model prints its own labels).

use rodent_act::debug_api::*;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let video = args.get(1).cloned().unwrap_or_else(|| "media/synthetic_mice.mp4".into());
    let frame_idx: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);
    let score_thr: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.01);
    let model = args
        .get(4)
        .cloned()
        .unwrap_or_else(|| "models/yolo11n.onnx".into());

    let mut reader = video_io::VideoReader::open(std::path::Path::new(&video), 640)?;
    let f = reader.skip_to(frame_idx)?.expect("frame not found");
    std::fs::create_dir_all("out")?;
    image::save_buffer(
        "out/debug_frame.png",
        &f.data,
        f.width,
        f.height,
        image::ColorType::Rgb8,
    )?;

    let mut det = detect::YoloDetector::load(
        std::path::Path::new(&model),
        score_thr,
        0.45,
        300,
        4,
    )?;
    let (net_w, net_h) = det.net_wh();
    let lb = preprocess::LetterboxCfg::new(f.width, f.height, net_w, net_h);
    println!("network input: {net_w}x{net_h}");
    let names: Vec<String> = if det.class_names.is_empty() {
        detect::COCO_NAMES.iter().map(|s| s.to_string()).collect()
    } else {
        det.class_names.clone()
    };
    println!("model: {model}  classes: {} ({})", names.len(), names.join(", "));

    let mut dets = det.detect(&f.data, &lb)?;
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    println!("{} candidates (score >= {score_thr}) on frame {frame_idx} of {video}", dets.len());
    for d in dets.iter().take(20) {
        println!(
            "  class {:>2} {:<14} score {:.3}  box {:>4.0},{:>4.0} {:>4.0}x{:<4.0}",
            d.class_id,
            names.get(d.class_id).map(|s| s.as_str()).unwrap_or("?"),
            d.score,
            d.bbox.x1,
            d.bbox.y1,
            d.bbox.wh().0,
            d.bbox.wh().1
        );
    }
    Ok(())
}
