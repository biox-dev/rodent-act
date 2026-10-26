"""Convert the Rust tracker's JSON output into window feature CSVs.

Usage
-----
    python analytics/extract_features.py out/tracks.json out/windows.csv \
        --fps 25 --width 640 --height 360 --window 0.5 --overlap 0.5

The output CSV has one row per (track, 0.5 s window) and is the input for
``train_xgb.py``. Labels can be added later with ``label_windows.py``.
"""
from __future__ import annotations

import argparse
import json

from features import FEATURE_NAMES, build_dataset, write_feature_matrix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", help="JSON produced by `rodent-act track --json`")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--width", type=float, required=True)
    ap.add_argument("--height", type=float, required=True)
    ap.add_argument("--window", type=float, default=0.5)
    ap.add_argument("--overlap", type=float, default=0.5)
    args = ap.parse_args()

    rows = build_dataset(
        args.records,
        args.fps,
        args.width,
        args.height,
        window_s=args.window,
        overlap=args.overlap,
    )
    write_feature_matrix(rows, args.out)

    n_tracks = len(json.load(open(args.records))["tracks"])
    print(f"{len(rows)} windows from {n_tracks} tracks -> {args.out}")
    print(f"{len(FEATURE_NAMES)} features: {', '.join(FEATURE_NAMES)}")


if __name__ == "__main__":
    main()
