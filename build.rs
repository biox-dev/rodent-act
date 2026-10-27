//! Build script: embed the compiled Vite frontend (`frontend/dist`) into the
//! binary so `cargo build --release` produces a single self-contained server.
//!
//! The directory is optional: when `npm run build` has not been run yet, the
//! generated table is empty and `rodent-act serve` still works (API only, or
//! with `--frontend <dir>` pointing at a build made elsewhere).
//!
//! Nothing is read from `node_modules`; the script only walks `dist`, and a
//! missing dist is not an error - it is reported as a `cargo:warning`.

use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let dist = manifest_dir.join("frontend").join("dist");

    // Re-run only when the build output actually changes.
    println!("cargo:rerun-if-changed=frontend/dist");
    println!("cargo:rerun-if-changed=build.rs");

    let dest = out_dir.join("frontend_assets.rs");

    if !dist.join("index.html").is_file() {
        println!(
            "cargo:warning=frontend/dist missing - the binary will serve no UI. \
             Run `npm --prefix frontend install && npm --prefix frontend run build` first."
        );
        std::fs::write(&dest, "&[]\n").expect("write frontend_assets.rs");
        return;
    }

    let mut files = Vec::new();
    collect(&dist, &dist, &mut files);
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut code = String::from("&[\n");
    let mut total = 0u64;
    for (rel, abs) in &files {
        total += std::fs::metadata(abs).map(|m| m.len()).unwrap_or(0);
        code.push_str(&format!(
            "    ({:?}, include_bytes!({:?}).as_slice()),\n",
            rel,
            abs.to_string_lossy()
        ));
    }
    code.push_str("]\n");

    std::fs::write(&dest, code).expect("write frontend_assets.rs");
    println!(
        "cargo:warning=embedded {} frontend files ({:.1} KiB) from {}",
        files.len(),
        total as f64 / 1024.0,
        dist.display()
    );
}

/// Recursively gather `(dist-relative path, absolute path)` for every file.
fn collect(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            println!("cargo:warning=cannot read {}: {e}", dir.display());
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            collect(root, &path, out);
        } else if file_type.is_file() {
            let Ok(rel) = path.strip_prefix(root) else {
                continue;
            };
            let rel = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            out.push((rel, path));
        }
    }
}
