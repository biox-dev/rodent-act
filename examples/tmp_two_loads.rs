//! Temporary check: two detectors in one process (regression for the
//! "only one environment is allowed per process" error).

use std::path::Path;

fn main() -> anyhow::Result<()> {
    for i in 0..2 {
        let d = rodent_act::detect::YoloDetector::load(
            Path::new("models/yolo11n.onnx"),
            0.25,
            0.45,
            300,
            2,
        )?;
        println!("load {i}: ok, device = {}, net = {:?}", d.device(), d.net_wh());
    }
    Ok(())
}
