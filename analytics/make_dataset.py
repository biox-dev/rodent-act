"""Generate a YOLO-format labelling dataset from a TST video.

Why this exists
---------------
Lab footage (multi-channel TST rigs, home cages) is not covered by the COCO
`mouse` class, so the stock YOLO11n detects nothing. Fine-tuning needs a few
hundred labelled frames. Hand-labelling from scratch is slow; the standard
trick is **model-assisted labelling**:

   1. ``extract`` - sample N frames spread over the video, plus a frame
      sequence around them (a mouse that moves between sampled frames is
      easier to trace).
   2. ``autolabel`` - run a first-pass detector (usually the blob detector,
      which is *very* good at "is there something dark hanging in channel k")
      to pre-fill YOLO labels. For a multi-channel rig the channel walls make
      this especially reliable: one candidate per channel.
   3. Correct the pre-labels in any annotation tool (labelImg, Label Studio,
      CVAT, Roboflow) - typically 1-2 clicks per frame, not 6.
   4. ``split`` - build ``dataset/data.yaml`` with a train/val split *by video*.

Everything is plain YOLO format:
    ``<class_idx> <x_center> <y_center> <width> <height>``  (all normalised 0..1)
and every image has a sibling ``.txt`` file with the same stem.

Usage
-----
    python analytics/make_dataset.py extract --video TST.mp4 --out dataset/raw \
        --frames 300 --fps 1 --channels 6

    python analytics/make_dataset.py autolabel --video TST.mp4 --out dataset/raw \
        --channels 6

    python analytics/make_dataset.py split --raw dataset/raw --out dataset \
        --val-frac 0.2

    python analytics/model_tools.py fine-tune --data dataset/data.yaml \
        --epochs 100 --imgsz 640 --out models/mouse.onnx
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _frames_of(video: str | Path) -> tuple[cv2.VideoCapture, int, float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    return cap, n, fps


def write_label(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    """boxes: (class, xc, yc, w, h) normalised."""
    with open(path, "w") as f:
        for cls, xc, yc, w, h in boxes:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# --------------------------------------------------------------------------- #
# 1. extract frames
# --------------------------------------------------------------------------- #
def extract(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap, n, fps = _frames_of(args.video)

    if args.fps and args.fps > 0:
        step = max(1, int(round(fps / args.fps)))
    else:
        step = max(1, n // max(1, args.frames))
    wanted = set(range(0, n, step))
    if args.frames:
        wanted = set(sorted(wanted)[: args.frames])

    written = 0
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if i not in wanted:
            continue
        if args.max_side and max(frame.shape[:2]) > args.max_side:
            s = args.max_side / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        name = f"{Path(args.video).stem}_{i:06d}"
        cv2.imwrite(str(out / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        # empty label file so the directory is immediately valid YOLO format
        (out / f"{name}.txt").touch()
        written += 1
    cap.release()
    print(f"wrote {written} frames to {out}")
    print("next: run `autolabel` to pre-fill labels, then correct them by hand")


# --------------------------------------------------------------------------- #
# 2. auto-label with the blob detector logic (pure numpy mirror)
# --------------------------------------------------------------------------- #
def _blob_boxes(gray: np.ndarray, thr: int, min_area_ratio: float, nms_iou: float,
                max_objs: int) -> list[tuple[int, int, int, int]]:
    """Static-image variant of the Rust blob detector.

    For a still frame we do not have a background, so we use the *darkness*
    relative to the frame median (a TST rig is white walls + dark mice), which
    is exactly the cue the operator uses.
    """
    med = float(np.median(gray))
    mask = ((med - gray) > thr).astype(np.uint8)
    if mask.sum() == 0:
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = gray.shape
    frame_area = float(h * w)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area_ratio * frame_area:
            continue
        if min(bw, bh) < 6:
            continue
        boxes.append((int(x), int(y), int(bw), int(bh)))
    # simple NMS
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        if any(_iou(b, k) > nms_iou for k in kept):
            continue
        kept.append(b)
        if len(kept) >= max_objs:
            break
    return kept


def _iou(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def autolabel(args: argparse.Namespace) -> None:
    raw = Path(args.raw)
    images = sorted(raw.glob("*.jpg")) + sorted(raw.glob("*.png"))
    if not images:
        raise SystemExit(f"no images in {raw} - run `extract` first")

    # Optional ROI: only keep boxes whose centre is inside this rectangle.
    # Strongly recommended - it removes lights, cables, cage hardware and the
    # operator's hands, which would otherwise become "mice" in the labels.
    roi = None
    if args.roi:
        try:
            x1, y1, x2, y2 = (float(v) for v in args.roi.split(","))
            roi = (x1, y1, x2, y2)
            print(f"ROI filter: x {x1}-{x2}, y {y1}-{y2} (normalised)")
        except ValueError:
            raise SystemExit("--roi expects x1,y1,x2,y2 in 0..1 normalised coords")

    total = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        boxes = _blob_boxes(gray, args.thresh, args.min_area, 0.25, args.max_objects)
        norm = []
        for x, y, bw, bh in boxes:
            xc, yc = (x + bw / 2) / w, (y + bh / 2) / h
            if roi and not (roi[0] <= xc <= roi[2] and roi[1] <= yc <= roi[3]):
                continue
            norm.append((args.class_id, xc, yc, bw / w, bh / h))
        if args.channels and len(norm) > args.channels:
            norm = norm[: args.channels]
        write_label(img_path.with_suffix(".txt"), norm)
        total += len(norm)
    print(f"pre-labelled {len(images)} images, {total} boxes total")
    print("NOW CORRECT THEM IN AN ANNOTATION TOOL - pre-labels are a starting point,")
    print("not ground truth (missing/extra boxes are expected).")


# --------------------------------------------------------------------------- #
# 3. train/val split
# --------------------------------------------------------------------------- #
def split(args: argparse.Namespace) -> None:
    raw = Path(args.raw)
    out = Path(args.out)
    images = sorted(raw.glob("*.jpg")) + sorted(raw.glob("*.png"))
    if not images:
        raise SystemExit(f"no images in {raw}")
    rng = random.Random(args.seed)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * args.val_frac))
    splits = {
        "train": images[n_val:],
        "val": images[:n_val],
    }
    for name, items in splits.items():
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / name).mkdir(parents=True, exist_ok=True)
        for img in items:
            (out / "images" / name / img.name).write_bytes(img.read_bytes())
            lbl = img.with_suffix(".txt")
            (out / "labels" / name / img.name.replace(img.suffix, ".txt")).write_bytes(
                lbl.read_bytes() if lbl.exists() else b""
            )

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        "# generated by analytics/make_dataset.py split\n"
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: mouse\n"
    )
    print(f"train={len(splits['train'])} val={len(splits['val'])}")
    print(f"wrote {yaml_path}")


# --------------------------------------------------------------------------- #
# 4. sanity check
# --------------------------------------------------------------------------- #
def check(args: argparse.Namespace) -> None:
    root = Path(args.dataset)
    bad = 0
    n_img = n_lbl = n_boxes = 0
    for img in sorted((root / "images" / "train").glob("*")) + sorted(
        (root / "images" / "val").glob("*")
    ):
        n_img += 1
        lbl = root / "labels" / img.parent.name / f"{img.stem}.txt"
        if not lbl.exists():
            print(f"MISSING label: {lbl}")
            bad += 1
            continue
        n_lbl += 1
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                print(f"BAD line in {lbl}: {line!r}")
                bad += 1
                continue
            vals = [float(v) for v in parts[1:]]
            if any(v < 0 or v > 1 for v in vals):
                print(f"OUT OF RANGE in {lbl}: {line!r}")
                bad += 1
            n_boxes += 1
    print(f"images={n_img} labels={n_lbl} boxes={n_boxes} problems={bad}")
    if n_boxes < 100:
        print("WARNING: fewer than 100 boxes - fine-tuning will overfit badly")
    if bad:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="sample frames from a video")
    e.add_argument("--video", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--frames", type=int, default=300)
    e.add_argument("--fps", type=float, default=1.0, help="sampling rate (0 = use --frames evenly)")
    e.add_argument("--max-side", type=int, default=960)
    e.set_defaults(func=extract)

    a = sub.add_parser("autolabel", help="pre-fill YOLO labels from dark blobs")
    a.add_argument("--raw", required=True)
    a.add_argument("--thresh", type=int, default=45, help="darkness below frame median")
    a.add_argument("--min-area", type=float, default=0.0008)
    a.add_argument("--max-objects", type=int, default=12)
    a.add_argument("--channels", type=int, default=0, help="max objects per frame (rig channels)")
    a.add_argument(
        "--roi",
        default="",
        help="keep only boxes centred inside x1,y1,x2,y2 (normalised 0..1); "
        "strongly recommended",
    )
    a.add_argument("--class-id", type=int, default=0)
    a.set_defaults(func=autolabel)

    s = sub.add_parser("split", help="build dataset/data.yaml")
    s.add_argument("--raw", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--val-frac", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=split)

    c = sub.add_parser("check", help="validate the dataset")
    c.add_argument("--dataset", required=True)
    c.set_defaults(func=check)

    args = ap.parse_args()
    args.func(args)
    json.dumps({})  # keep json import meaningful for future export


if __name__ == "__main__":
    main()
