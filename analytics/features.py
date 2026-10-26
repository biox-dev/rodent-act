"""Building blocks for the 0.5 s motion/mobility windows.

The tail-suspension test (TST) is scored by judging, in short time slices,
whether the animal is **immobile** or **mobile**. This module turns a track's
per-frame boxes into fixed-length windows and computes the features consumed by
the XGBoost classifier.

Design notes
------------
* Window length is 0.5 s by default (``fps * 0.5`` frames), with an optional
  50 % overlap so a behaviour that straddles a boundary is not lost.
* All distances are normalised by the animal's own reference size (median box
  diagonal over the whole track), so they are scale invariant between videos
  and camera setups.
* Missing frames (track lost) are forward filled but flagged with
  ``valid_ratio`` so the classifier can discount unreliable windows.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

FEATURE_NAMES: list[str] = [
    "speed_mean",            # mean centroid speed (body length / s)
    "speed_std",
    "speed_max",
    "path_length",           # total travelled distance (body lengths)
    "displacement",          # net start->end distance (body lengths)
    "path_efficiency",       # displacement / path_length  (0..1)
    "turn_abs_mean",         # mean |heading change| per frame (rad)
    "turn_abs_max",
    "stationary_fraction",   # fraction of frames below the speed threshold
    "micro_move_ratio",      # fraction of frames with tiny but non-zero motion
    "bbox_area_cv",          # coefficient of variation of box area
    "aspect_mean",
    "aspect_std",
    "motion_energy",         # mean frame-to-frame pixel diff inside the box
    "velocity_autocorr1",    # lag-1 autocorrelation of speed
    "n_jumps",               # count of implausible centroid jumps
    "valid_ratio",           # fraction of frames with a real detection
    "size_norm",             # median box diagonal relative to the frame
]


@dataclass
class FrameSample:
    frame: int
    t: float
    cx: float
    cy: float
    w: float
    h: float
    measured: bool = True


@dataclass
class Window:
    track_id: int
    start_frame: int
    end_frame: int
    start_t: float
    end_t: float
    features: dict[str, float] = field(default_factory=dict)
    label: int | None = None  # 1 = immobile, 0 = mobile (filled by the labeller)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_track_records(path: str | Path, fps: float) -> dict[int, list[FrameSample]]:
    """Read the JSON written by ``rodent-act track --json``."""
    data = json.loads(Path(path).read_text())
    tracks: dict[int, list[FrameSample]] = {}
    for r in data["records"]:
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        tracks.setdefault(r["track_id"], []).append(
            FrameSample(
                frame=r["frame"],
                t=r["frame"] / fps,
                cx=(x1 + x2) / 2,
                cy=(y1 + y2) / 2,
                w=max(1e-3, x2 - x1),
                h=max(1e-3, y2 - y1),
                measured=bool(r.get("measured", True)),
            )
        )
    for v in tracks.values():
        v.sort(key=lambda s: s.frame)
    return tracks


def make_windows(
    samples: Sequence[FrameSample],
    fps: float,
    window_s: float = 0.5,
    overlap: float = 0.5,
) -> list[list[FrameSample]]:
    """Split a track into (optionally overlapping) fixed-duration chunks."""
    n = max(1, int(round(window_s * fps)))
    step = max(1, int(round(n * (1.0 - overlap))))
    out = []
    i = 0
    while i + n <= len(samples):
        out.append(list(samples[i : i + n]))
        i += step
    return out


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def _median(xs: Iterable[float]) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def compute_features(
    samples: Sequence[FrameSample],
    fps: float,
    ref_size: float,
    frame_w: float,
    frame_h: float,
    *,
    stationary_speed: float = 0.30,
    micro_speed: float = 0.08,
    jump_threshold: float = 2.0,
) -> dict[str, float]:
    """Per-window descriptor.

    ``ref_size`` is the track's median box diagonal; every distance is divided
    by it so features are in *body lengths*.
    """
    ref = max(1e-6, ref_size)
    dt = 1.0 / max(1e-6, fps)

    # per-frame kinematics
    speeds: list[float] = []
    headings: list[float] = []
    jumps = 0
    for a, b in zip(samples, samples[1:]):
        dx, dy = b.cx - a.cx, b.cy - a.cy
        d = math.hypot(dx, dy)
        if d > jump_threshold * ref:
            jumps += 1
            speeds.append(0.0)
            continue
        speeds.append(d / ref / dt)
        if d > 1e-3:
            headings.append(math.atan2(dy, dx))

    turns = []
    for h1, h2 in zip(headings, headings[1:]):
        d = abs((h2 - h1 + math.pi) % (2 * math.pi) - math.pi)
        turns.append(d)

    path_len = sum(speeds) * dt
    first, last = samples[0], samples[-1]
    displacement = math.hypot(last.cx - first.cx, last.cy - first.cy) / ref

    sp = speeds or [0.0]
    stationary = sum(1 for s in sp if s < stationary_speed) / len(sp)
    micro = sum(1 for s in sp if micro_speed <= s < stationary_speed) / len(sp)

    areas = [s.w * s.h for s in samples]
    aspects = [s.w / max(1e-6, s.h) for s in samples]
    area_mean = sum(areas) / len(areas)
    area_cv = (_std(areas) / area_mean) if area_mean > 0 else 0.0

    # lag-1 autocorrelation of speed (regularity of the movement)
    if len(sp) > 2:
        m = sum(sp) / len(sp)
        num = sum((sp[i] - m) * (sp[i + 1] - m) for i in range(len(sp) - 1))
        den = sum((x - m) ** 2 for x in sp)
        autocorr = num / den if den > 1e-9 else 0.0
    else:
        autocorr = 0.0

    valid = sum(1 for s in samples if s.measured) / len(samples)

    return {
        "speed_mean": sum(sp) / len(sp),
        "speed_std": _std(sp),
        "speed_max": max(sp),
        "path_length": path_len,
        "displacement": displacement,
        "path_efficiency": displacement / path_len if path_len > 1e-9 else 0.0,
        "turn_abs_mean": sum(turns) / len(turns) if turns else 0.0,
        "turn_abs_max": max(turns) if turns else 0.0,
        "stationary_fraction": stationary,
        "micro_move_ratio": micro,
        "bbox_area_cv": area_cv,
        "aspect_mean": sum(aspects) / len(aspects),
        "aspect_std": _std(aspects),
        # filled in from the video when available; 0.0 otherwise
        "motion_energy": 0.0,
        "velocity_autocorr1": autocorr,
        "n_jumps": float(jumps),
        "valid_ratio": valid,
        "size_norm": ref / max(1.0, math.hypot(frame_w, frame_h)),
    }


def track_reference_size(samples: Sequence[FrameSample]) -> float:
    return _median(math.hypot(s.w, s.h) for s in samples)


def build_dataset(
    records_path: str | Path,
    fps: float,
    frame_w: float,
    frame_h: float,
    *,
    window_s: float = 0.5,
    overlap: float = 0.5,
) -> list[dict]:
    """Track records -> one dict per window, ready for ``pandas.DataFrame``."""
    tracks = load_track_records(records_path, fps)
    rows: list[dict] = []
    for tid, samples in tracks.items():
        ref = track_reference_size(samples)
        for chunk in make_windows(samples, fps, window_s, overlap):
            feats = compute_features(chunk, fps, ref, frame_w, frame_h)
            rows.append(
                {
                    "track_id": tid,
                    "start_frame": chunk[0].frame,
                    "end_frame": chunk[-1].frame,
                    "start_t": chunk[0].t,
                    "end_t": chunk[-1].t,
                    **feats,
                }
            )
    return rows


def write_feature_matrix(rows: Sequence[dict], out_csv: str | Path) -> None:
    import csv

    cols = ["track_id", "start_frame", "end_frame", "start_t", "end_t", *FEATURE_NAMES]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
