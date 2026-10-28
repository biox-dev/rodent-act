//! ONNX Runtime backend (YOLO11, 80 COCO classes by default).

pub mod blob;
pub mod device;
pub mod motion;
pub mod yolo;

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, bail};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use crate::common::Detection;
use crate::preprocess::LetterboxCfg;
use device::{ActiveDevice, Device};

/// Process-global ONNX Runtime environment.
///
/// ONNX Runtime allows exactly **one** environment per process, and `ort`
/// remembers it for the lifetime of the process (even after the `Environment`
/// handle is dropped). Creating a second environment therefore fails with
/// "only one environment is allowed per process", which is what happened when
/// a second job/dataset was analysed in the same process. Build it once and
/// keep it alive forever.
static ONNX_ENV: OnceLock<ort::environment::Environment> = OnceLock::new();

/// Serialises first-time initialisation so two threads cannot race to create
/// the environment (the loser would get the "only one environment" error).
static ONNX_ENV_LOCK: Mutex<()> = Mutex::new(());

/// Get the shared ONNX Runtime environment, creating it on first use.
fn onnx_environment() -> Result<&'static ort::environment::Environment> {
    if let Some(env) = ONNX_ENV.get() {
        return Ok(env);
    }

    let _guard = ONNX_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(env) = ONNX_ENV.get() {
        return Ok(env);
    }

    let env = ort::init()
        .with_name("rodent-act")
        .build()
        .context("initialising ONNX Runtime environment")?;
    Ok(ONNX_ENV.get_or_init(|| env))
}

pub struct YoloDetector {
    session: Session,
    input_name: String,
    output_name: String,
    /// Execution provider the session was created with (`cpu` or `cuda:0`).
    pub device: ActiveDevice,
    rows: usize,
    /// Class names baked into the model (empty for single-class models).
    pub class_names: Vec<String>,
    pub score_thr: f32,
    pub iou_thr: f32,
    pub max_det: usize,
    /// Vertical flip augmentation (top-down rigs mounted upside down).
    pub flip_vertical: bool,
    /// Network input `(width, height)` as baked into the ONNX graph.
    ///
    /// Ultralytics exports a **static** input shape (the `imgsz` used during
    /// export), so this cannot be assumed to be 640 - a model trained/exported
    /// at 480 will reject a 640 tensor. `0` means the axis is dynamic.
    pub input_wh: (u32, u32),
}

impl YoloDetector {
    pub fn load(
        model: &Path,
        score_thr: f32,
        iou_thr: f32,
        max_det: usize,
        threads: usize,
    ) -> Result<Self> {
        Self::load_with_device(model, score_thr, iou_thr, max_det, threads, Device::Cpu, 0)
    }

    /// Load a model on an explicit device.
    ///
    /// `device` is only honoured by builds compiled with `--features cuda`;
    /// see [`device`] for the resolution rules. An explicit [`Device::Cuda`]
    /// on a machine that cannot run it is a hard error, never a silent CPU
    /// fallback.
    pub fn load_with_device(
        model: &Path,
        score_thr: f32,
        iou_thr: f32,
        max_det: usize,
        threads: usize,
        device: Device,
        gpu_device_id: i32,
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
        let env = onnx_environment()?;

        let builder = Session::builder(env)
            .map_err(|e| anyhow::anyhow!("Session::builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| anyhow::anyhow!("optimization level: {e}"))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("intra threads: {e}"))?;

        let (mut builder, active_device) = device::apply(builder, device, gpu_device_id)?;

        let session = builder
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

        // Read the static input size from the graph (NCHW). `0`/-1 means the
        // axis is dynamic.
        let input_wh = session
            .inputs()
            .first()
            .map(|i| match i.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => {
                    let w = shape.get(3).copied().unwrap_or(0);
                    let h = shape.get(2).copied().unwrap_or(0);
                    (w.max(0) as u32, h.max(0) as u32)
                }
                _ => (0, 0),
            })
            .unwrap_or((0, 0));

        // Ultralytics stores the class list in the ONNX metadata as a Python
        // dict literal: `names = {0: 'mouse', 1: 'animal'}`. Reading it here
        // means a fine-tuned model needs no extra configuration.
        let class_names = read_names_metadata(&session).unwrap_or_default();

        Ok(Self {
            session,
            input_name,
            output_name,
            device: active_device,
            class_names,
            rows: yolo::expected_rows(80),
            score_thr,
            iou_thr,
            max_det,
            flip_vertical: false,
            input_wh,
        })
    }

    /// Execution provider actually in use (`cpu`, `cuda:0`, ...).
    pub fn device(&self) -> ActiveDevice {
        self.device
    }

    /// Network input size, with a 640 fallback for dynamic-shape exports.
    pub fn net_wh(&self) -> (u32, u32) {
        if self.input_wh.0 > 0 && self.input_wh.1 > 0 {
            self.input_wh
        } else {
            (640, 640)
        }
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

    /// Number of classes the model declares (ONNX `names` metadata), or the
    /// class count decoded from the output tensor when the metadata is absent.
    pub fn declared_classes(&self) -> usize {
        if self.class_names.is_empty() {
            self.num_classes()
        } else {
            self.class_names.len()
        }
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
    parse_names(&raw)
}

/// Split a Python-literal-ish container body on commas that are *outside*
/// quotes, so names containing a comma (`'traffic light, red'`) survive.
fn split_top_level(inner: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut quote: Option<char> = None;
    for (i, ch) in inner.char_indices() {
        match quote {
            Some(q) if ch == q => quote = None,
            Some(_) => {}
            None if ch == '\'' || ch == '"' => quote = Some(ch),
            None if ch == ',' => {
                parts.push(&inner[start..i]);
                start = i + 1;
            }
            None => {}
        }
    }
    parts.push(&inner[start..]);
    parts
}

/// Unquote a Python string literal / bare token.
fn unquote(tok: &str) -> String {
    tok.trim()
        .trim_matches(|c: char| c == '\'' || c == '"' || c.is_whitespace())
        .to_string()
}

/// Parse the Ultralytics `names` metadata value into an ordered class list.
pub fn parse_names(raw: &str) -> Option<Vec<String>> {
    let body = raw.trim();
    let mut pairs: Vec<(usize, String)> = Vec::new();

    if body.starts_with('{') {
        let inner = body.trim_start_matches('{').trim_end_matches('}');
        for part in split_top_level(inner) {
            let Some((k, v)) = part.split_once(':') else {
                continue;
            };
            let Ok(idx) = unquote(k).parse::<usize>() else {
                continue;
            };
            let name = unquote(v);
            if !name.is_empty() {
                pairs.push((idx, name));
            }
        }
    } else if body.starts_with('[') || body.starts_with('(') {
        let inner = body
            .trim_start_matches(['[', '('])
            .trim_end_matches([']', ')']);
        for (i, part) in split_top_level(inner).into_iter().enumerate() {
            let name = unquote(part);
            if !name.is_empty() {
                pairs.push((i, name));
            }
        }
    }

    if pairs.is_empty() {
        return None;
    }
    let n = pairs.iter().map(|(i, _)| *i).max()? + 1;
    let mut names = vec![String::new(); n];
    for (i, name) in pairs {
        if i < n {
            names[i] = name;
        }
    }
    Some(names)
}

#[cfg(test)]
mod metadata_tests {
    use super::parse_names;

    #[test]
    fn parses_dict_literal() {
        assert_eq!(
            parse_names("{0: 'mouse', 1: 'animal'}").unwrap(),
            vec!["mouse".to_string(), "animal".to_string()]
        );
    }

    #[test]
    fn parses_single_class_dict() {
        assert_eq!(
            parse_names("{0: 'item'}").unwrap(),
            vec!["item".to_string()]
        );
    }

    #[test]
    fn parses_list_literal() {
        assert_eq!(
            parse_names("['mouse', 'animal']").unwrap(),
            vec!["mouse".to_string(), "animal".to_string()]
        );
    }

    #[test]
    fn parses_out_of_order_and_quoted_commas() {
        assert_eq!(
            parse_names("{1: 'traffic light, red', 0: 'mouse'}").unwrap(),
            vec!["mouse".to_string(), "traffic light, red".to_string()]
        );
    }

    #[test]
    fn rejects_empty() {
        assert!(parse_names("{}").is_none());
        assert!(parse_names("").is_none());
    }
}
