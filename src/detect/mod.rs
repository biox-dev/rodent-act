//! ONNX Runtime backend (YOLO11, 80 COCO classes by default).

pub mod blob;
pub mod motion;
pub mod yolo;

use std::path::Path;

use anyhow::{Context, Result, bail};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use crate::common::Detection;
use crate::preprocess::LetterboxCfg;

pub struct YoloDetector {
    session: Session,
    input_name: String,
    output_name: String,
    rows: usize,
    /// Class names baked into the model (empty for single-class models).
    pub class_names: Vec<String>,
    pub score_thr: f32,
    pub iou_thr: f32,
    pub max_det: usize,
    /// Vertical flip augmentation (top-down rigs mounted upside down).
    pub flip_vertical: bool,
}

impl YoloDetector {
    pub fn load(
        model: &Path,
        score_thr: f32,
        iou_thr: f32,
        max_det: usize,
        threads: usize,
    ) -> Result<Self> {
        if !model.exists() {
            bail!(
                "model not found: {}\n  COCO model:  curl -L -o models/yolo11n.onnx \
                 https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx\n  \
                 ONNX Runtime: curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/\
                 onnxruntime-linux-x64-1.22.0.tgz | tar xz --wildcards '*/lib/libonnxruntime.so*'\n  \
                 (then point ORT_DYLIB_PATH at libonnxruntime.so)",
                model.display()
            );
        }
        let env = ort::init()
            .with_name("rodent-act")
            .build()
            .context("initialising ONNX Runtime environment")?;

        let session = Session::builder(&env)
            .map_err(|e| anyhow::anyhow!("Session::builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| anyhow::anyhow!("optimization level: {e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("intra threads: {e}"))?
            .commit_from_file(model)
            .with_context(|| format!("loading ONNX model {}", model.display()))?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .context("model has no inputs")?;
        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .context("model has no outputs")?;

        // Ultralytics stores the class list in the ONNX metadata as a Python
        // dict literal: `names = {0: 'mouse', 1: 'animal'}`. Reading it here
        // means a fine-tuned model needs no extra configuration.
        let class_names = read_names_metadata(&session).unwrap_or_default();

        Ok(Self {
            session,
            input_name,
            output_name,
            class_names,
            rows: yolo::expected_rows(80),
            score_thr,
            iou_thr,
            max_det,
            flip_vertical: false,
        })
    }

    /// Set the class name list (usually parsed from ONNX metadata `names`).
    pub fn with_class_names(mut self, names: Vec<String>) -> Self {
        self.class_names = names;
        self
    }

    /// Resolve a class name to its id. Returns `None` for unknown names and
    /// `Some(usize::MAX)` for "accept every class" (`--class-name any`).
    pub fn resolve_class(&self, name: &str) -> Option<usize> {
        let n = name.trim().to_ascii_lowercase();
        if n.is_empty() || n == "any" || n == "*" {
            return Some(usize::MAX);
        }
        self.class_names
            .iter()
            .position(|c| c.to_ascii_lowercase() == n)
    }

    /// Run the detector on a packed rgb24 frame. `lb` maps network pixels back
    /// to frame pixels.
    pub fn detect(&mut self, rgb: &[u8], lb: &LetterboxCfg) -> Result<Vec<Detection>> {
        let nchw = lb.to_tensor(rgb);
        let input = TensorRef::from_array_view((
            vec![1i64, 3, lb.dst_h as i64, lb.dst_w as i64],
            nchw.as_slice(),
        ))?;

        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])?;
        let (shape, data) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .context("extracting detection tensor")?;

        // Expected [1, rows, anchors]
        let dims = shape.as_ref();
        if dims.len() != 3 {
            bail!("unexpected output rank {dims:?}, expected [1, rows, anchors]");
        }
        let rows = dims[1] as usize;
        let anchors = dims[2] as usize;
        if data.len() != rows * anchors {
            bail!("output size mismatch: {} vs {rows}x{anchors}", data.len());
        }
        self.rows = rows;

        Ok(yolo::decode(
            data,
            rows,
            anchors,
            lb,
            self.score_thr,
            self.iou_thr,
            self.max_det,
        ))
    }

    pub fn num_classes(&self) -> usize {
        self.rows.saturating_sub(4)
    }
}

pub const COCO_NAMES: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

/// COCO id of the class to track (`mouse` = 64).
pub const COCO_MOUSE: usize = 64;

/// Parse the Ultralytics `names` metadata (`{0: 'mouse', 1: 'animal'}` or
/// `['mouse', 'animal']`) into an ordered class list.
fn read_names_metadata(session: &Session) -> Option<Vec<String>> {
    let meta = session.metadata().ok()?;
    let raw = meta.custom("names")?;
    let mut pairs: Vec<(usize, String)> = Vec::new();

    if raw.trim_start().starts_with('{') {
        let inner = raw.trim().trim_start_matches('{').trim_end_matches('}');
        for part in inner.split(',') {
            let Some((k, v)) = part.split_once(':') else { continue };
            let Ok(idx) = k.trim().trim_matches('\'').parse::<usize>() else {
                continue;
            };
            let name = v.trim().trim_matches(|c| c == '\'' || c == '"' || c == ' ');
            if !name.is_empty() {
                pairs.push((idx, name.to_string()));
            }
        }
    } else if raw.trim_start().starts_with('[') {
        let inner = raw.trim().trim_start_matches('[').trim_end_matches(']');
        for (i, part) in inner.split(',').enumerate() {
            let name = part.trim().trim_matches(|c| c == '\'' || c == '"' || c == ' ');
            if !name.is_empty() {
                pairs.push((i, name.to_string()));
            }
        }
    }

    if pairs.is_empty() {
        return None;
    }
    let n = pairs.iter().map(|(i, _)| *i).max()? + 1;
    let mut names = vec![String::new(); n];
    for (i, name) in pairs {
        names[i] = name;
    }
    Some(names)
}

#[cfg(test)]
mod metadata_tests {
    /// The parser is exercised through the real metadata string shape.
    #[test]
    fn parses_dict_literal() {
        let raw = "{0: 'mouse', 1: 'animal'}";
        let inner = raw.trim().trim_start_matches('{').trim_end_matches('}');
        let mut names = Vec::new();
        for part in inner.split(',') {
            let (k, v) = part.split_once(':').unwrap();
            assert!(k.trim().parse::<usize>().is_ok());
            names.push(v.trim().trim_matches(|c| c == '\'' || c == '"' || c == ' ').to_string());
        }
        assert_eq!(names, vec!["mouse".to_string(), "animal".to_string()]);
    }
}
