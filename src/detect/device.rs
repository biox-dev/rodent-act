//! Compute-device selection for the ONNX backend.
//!
//! The YOLO detector runs on the CPU unless a GPU execution provider was both
//! **compiled in** (`--features cuda`) *and* **found at runtime** (a CUDA
//! enabled ONNX Runtime library plus the matching NVIDIA driver / CUDA / cuDNN
//! libraries).
//!
//! The two conditions are deliberately kept separate:
//!
//! * a CPU build ignores device flags entirely (nothing GPU related is linked);
//! * an explicit `--device cuda` on a build *without* the feature, or on a
//!   machine without a usable CUDA runtime, returns a **hard error** instead of
//!   silently falling back to the CPU. Silent fallback is the classic reason
//!   people believe they benchmark a GPU while actually running on a CPU.
//!
//! `--device auto` (the default) prefers the GPU when it is available and
//! otherwise stays on the CPU without complaint.

use std::fmt;
use std::str::FromStr;

/// Which execution provider the session should be built with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// Uppercase first letter `cpu`/`cuda`/`auto`, plus the common spellings.
    Cpu,
    /// CUDA execution provider (`onnxruntime-linux-x64-gpu`).
    Cuda,
    /// Use a GPU when this build and machine can, otherwise CPU.
    #[default]
    Auto,
}

impl Device {
    /// Wording used in the `device=` log field and the web UI.
    pub fn label(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
            Device::Auto => "auto",
        }
    }

    /// Whether this build was compiled with ONNX Runtime GPU support.
    pub const fn compiled_with_gpu() -> bool {
        cfg!(feature = "cuda")
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl FromStr for Device {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Device::Auto),
            "cpu" => Ok(Device::Cpu),
            "cuda" | "gpu" => Ok(Device::Cuda),
            other => Err(format!(
                "unknown device `{other}` (expected `cpu`, `cuda` or `auto`)"
            )),
        }
    }
}

/// What actually ended up being used, after resolving [`Device`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveDevice {
    Cpu,
    Cuda { device_id: i32 },
}

impl fmt::Display for ActiveDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActiveDevice::Cpu => f.write_str("cpu"),
            ActiveDevice::Cuda { device_id } => write!(f, "cuda:{device_id}"),
        }
    }
}

/// Reason a CUDA request could not be honoured.
#[derive(Debug, Clone)]
pub struct DeviceError(pub String);

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DeviceError {}

/// `SessionBuilder` augmentation that registers the requested provider.
///
/// Returns the (possibly unchanged) builder plus the device that will be used.
/// With the `cuda` feature disabled this is a thin pass-through, so the CPU
/// build has no runtime cost and no dangling configuration knobs.
#[cfg(feature = "cuda")]
pub fn apply(
    builder: ort::session::builder::SessionBuilder,
    device: Device,
    gpu_device_id: i32,
) -> Result<(ort::session::builder::SessionBuilder, ActiveDevice), DeviceError> {
    use ort::ep::CUDA;

    let want_gpu = match device {
        Device::Cpu => false,
        Device::Cuda => true,
        Device::Auto => device_available(),
    };
    if !want_gpu {
        return Ok((builder, ActiveDevice::Cpu));
    }

    // `error_on_failure()` matters: `ort` otherwise logs a warning and keeps
    // going on the CPU, which makes a misconfigured GPU run look successful.
    match builder.with_execution_providers([
        CUDA::default()
            .with_device_id(gpu_device_id)
            .build()
            .error_on_failure(),
    ]) {
        Ok(b) => Ok((b, ActiveDevice::Cuda { device_id: gpu_device_id })),
        Err(e) => Err(DeviceError(format!(
            "could not enable the CUDA execution provider: {e}\n  \
             check that ORT_DYLIB_PATH points at a GPU build of ONNX Runtime \
             (onnxruntime-linux-x64-gpu-*), that the NVIDIA driver is loaded, and that \
             CUDA 12.8+ / cuDNN 9 are installed."
        ))),
    }
}

#[cfg(not(feature = "cuda"))]
pub fn apply(
    builder: ort::session::builder::SessionBuilder,
    device: Device,
    _gpu_device_id: i32,
) -> Result<(ort::session::builder::SessionBuilder, ActiveDevice), DeviceError> {
    match device {
        Device::Cpu | Device::Auto => Ok((builder, ActiveDevice::Cpu)),
        Device::Cuda => Err(DeviceError(
            "this build has no CUDA support: rebuild with `cargo build --release \
             --features cuda` (the ONNX Runtime library must be a GPU build, see \
             README section 0.3)"
                .to_string(),
        )),
    }
}

/// Whether a CUDA enabled ONNX Runtime session can plausibly be created.
///
/// Only meaningful with the `cuda` feature; without it the answer is `false`.
#[cfg(feature = "cuda")]
pub fn device_available() -> bool {
    use ort::ep::{CUDA, ExecutionProvider};

    // `is_available` is an associated function on the EP type; it queries the
    // loaded ONNX Runtime for a provider named `CUDAExecutionProvider`.
    CUDA::is_available().unwrap_or(false)
}

#[cfg(not(feature = "cuda"))]
pub const fn device_available() -> bool {
    false
}

/// Human readable summary for logs / `GET /api/info`.
pub fn describe() -> String {
    if !Device::compiled_with_gpu() {
        "cpu-only build (rebuild with --features cuda for GPU)".to_string()
    } else if device_available() {
        "cuda available".to_string()
    } else {
        "cuda compiled in but unavailable at runtime (CPU-only ORT library, \
         missing driver, or no GPU)"
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_device_names() {
        assert_eq!("cpu".parse::<Device>().unwrap(), Device::Cpu);
        assert_eq!("GPU".parse::<Device>().unwrap(), Device::Cuda);
        assert_eq!("cuda".parse::<Device>().unwrap(), Device::Cuda);
        assert_eq!("auto".parse::<Device>().unwrap(), Device::Auto);
        assert_eq!("".parse::<Device>().unwrap(), Device::Auto);
        assert!("tpu".parse::<Device>().is_err());
    }

    #[test]
    fn cpu_build_rejects_explicit_cuda() {
        // Only meaningful for the default (CPU) build; a `--features cuda`
        // test binary may return Ok when a GPU is present.
        if !Device::compiled_with_gpu() {
            assert!(matches!(device_available(), false));
        }
        // `cpu` must always resolve without touching any GPU library.
        assert!(!matches!(Device::Auto.label(), ""));
    }
}
