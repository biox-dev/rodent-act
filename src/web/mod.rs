//! Optional web layer: Axum REST + SSE server and the React frontend build.

pub mod assets;
pub mod error;
pub mod routes;
pub mod state;
pub mod telemetry;

pub use routes::{AppState, router, serve};
pub use state::JobManager;
