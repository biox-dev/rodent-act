//! Bridge from `tracing` to the web UI.
//!
//! The pipeline already logs via `tracing`; installing [`SseLayer`] forwards
//! every event into a broadcast channel that the `/api/jobs/{id}/events`
//! endpoint streams to the browser as Server-Sent Events (SSE).
//!
//! Progress bars (`indicatif`) are a terminal concept, so the layer is only
//! wired in when the CLI runs as `rodent-act serve`.

use axum::response::sse::Event;
use tokio::sync::broadcast;
use tracing::field::{Field, Visit};
use tracing::{Event as TraceEvent, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;

/// One log line as the UI sees it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LogRecord {
    pub ts_ms: u64,
    pub level: String,
    pub target: String,
    pub message: String,
}

impl LogRecord {
    pub fn level_rank(&self) -> u8 {
        match self.level.as_str() {
            "ERROR" => 5,
            "WARN" => 4,
            "INFO" => 3,
            "DEBUG" => 2,
            _ => 1,
        }
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Collects `message` + every other field into one readable line.
struct MsgVisitor {
    message: String,
    extra: Vec<(String, String)>,
}

impl Visit for MsgVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        } else {
            self.extra
                .push((field.name().to_string(), value.to_string()));
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let text = format!("{value:?}");
        if field.name() == "message" {
            self.message.push_str(&text);
        } else {
            self.extra.push((field.name().to_string(), text));
        }
    }
}

/// `tracing` layer broadcasting formatted events to the SSE channel.
pub struct SseLayer {
    tx: broadcast::Sender<LogRecord>,
    min_level: tracing::Level,
}

impl SseLayer {
    pub fn new(tx: broadcast::Sender<LogRecord>, min_level: tracing::Level) -> Self {
        Self { tx, min_level }
    }
}

impl<S> Layer<S> for SseLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &TraceEvent<'_>, _ctx: Context<'_, S>) {
        if *event.metadata().level() > self.min_level {
            return;
        }
        let mut v = MsgVisitor {
            message: String::new(),
            extra: Vec::new(),
        };
        event.record(&mut v);
        let mut message = v.message;
        if !v.extra.is_empty() {
            let fields = v
                .extra
                .iter()
                .map(|(k, val)| format!("{k}={val}"))
                .collect::<Vec<_>>()
                .join(" ");
            if message.is_empty() {
                message = fields;
            } else {
                message = format!("{message} ({fields})");
            }
        }
        let rec = LogRecord {
            ts_ms: now_ms(),
            level: event.metadata().level().to_string(),
            target: event.metadata().target().to_string(),
            message,
        };
        // A slow/absent browser must never block the analysis thread.
        let _ = self.tx.send(rec);
    }
}

/// JSON payload of an SSE `log` event.
pub fn log_event(rec: &LogRecord) -> Event {
    Event::default()
        .event("log")
        .json_data(rec)
        .unwrap_or_else(|_| Event::default().comment("encode error"))
}

/// JSON payload of an SSE `progress` event.
pub fn progress_event(progress: f64, message: &str) -> Event {
    Event::default()
        .event("progress")
        .json_data(serde_json::json!({ "progress": progress, "message": message }))
        .unwrap_or_else(|_| Event::default().comment("encode error"))
}

/// JSON payload of an SSE `status` event.
pub fn status_event(status: &str) -> Event {
    Event::default()
        .event("status")
        .json_data(serde_json::json!({ "status": status }))
        .unwrap_or_else(|_| Event::default().comment("encode error"))
}
