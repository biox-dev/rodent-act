//! Library facade: exposes the internals so examples/tools and (later) unit
//! tests can reuse the same building blocks as the CLI.
//!
//! `src/lib.rs` is the canonical home of the module tree; `main.rs` only wires
//! the CLI on top of it.

pub mod classify;
pub mod common;
pub mod detect;
pub mod draw;
pub mod features;
pub mod pipeline;
pub mod preprocess;
pub mod track;
pub mod video_io;

/// Stable re-export surface used by `examples/` (avoids deep paths).
pub mod debug_api {
    pub use crate::common;
    pub use crate::detect;
    pub use crate::preprocess;
    pub use crate::video_io;
}
