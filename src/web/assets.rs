//! Compile-time embedding of the built Vite frontend (`frontend/dist`).
//!
//! `build.rs` walks `frontend/dist` and emits `&[(&str, &[u8])]` pairs via
//! `include!`, so a release binary serves the UI with no files next to it.
//!
//! When `frontend/dist/index.html` does not exist at compile time, the table is
//! empty and the server falls back to the `--frontend` directory (or to an
//! API-only mode plus a hint in the logs).

/// `(path relative to dist, file contents)`, sorted for binary search.
pub static FILES: &[(&str, &[u8])] = include!(concat!(env!("OUT_DIR"), "/frontend_assets.rs"));

/// Whether a frontend build was embedded at compile time.
pub fn is_embedded() -> bool {
    !FILES.is_empty()
}

/// Look up a file by its dist-relative path (e.g. `assets/index-abc.js`).
///
/// Leading slashes and `.`/`..` segments are rejected so a request can never
/// escape the embedded tree.
pub fn get(path: &str) -> Option<&'static [u8]> {
    let key = path.trim_start_matches('/');
    if key.is_empty() || key.split('/').any(|seg| seg.is_empty() || seg == "." || seg == "..") {
        return None;
    }
    FILES
        .binary_search_by(|(name, _)| name.cmp(&key))
        .ok()
        .map(|idx| FILES[idx].1)
}

/// Number of embedded files (0 when the dist directory was absent).
pub fn count() -> usize {
    FILES.len()
}

/// Guessed MIME type from the file extension. Vite emits hashed files of these
/// kinds only, so a small table is enough.
pub fn mime_type(path: &str) -> &'static str {
    let ext = path.rsplit_once('.').map(|(_, e)| e).unwrap_or("");
    match ext {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "ttf" => "font/ttf",
        "wasm" => "application/wasm",
        "txt" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_traversal() {
        assert!(get("").is_none());
        assert!(get("/").is_none());
        assert!(get("../Cargo.toml").is_none());
        assert!(get("assets/../../etc/passwd").is_none());
        assert!(get("assets//index.js").is_none());
    }

    #[test]
    fn table_is_sorted_and_paths_are_clean() {
        for (name, _) in FILES {
            assert!(!name.starts_with('/'), "absolute path in asset table: {name}");
            assert!(!name.contains('\\'), "backslash in asset path: {name}");
        }
        assert!(
            FILES.windows(2).all(|w| w[0].0 < w[1].0),
            "asset table must be sorted for binary_search"
        );
    }
}
