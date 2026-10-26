"""Detector model tooling for rodent-act.

Subcommands
-----------
convert   Re-export an Ultralytics ``.pt`` checkpoint to ONNX with a fixed
          image size and ``nms=False`` (the Rust decoder does NMS itself).
fine-tune Fine-tune ``yolo11n.pt`` on a small mouse dataset, then convert.
names     Print the class names baked into an ONNX/PT model so the Rust CLI
          ``--class-name`` argument can be set correctly.

The COCO checkpoint shipped with Ultralytics knows the ``mouse`` class, but it
only fires on hand-held/posed mice. Overhead lab footage (TST rigs, home cages)
needs a few hundred labelled frames - ``fine-tune`` is the fast path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def convert(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(args.weights)
    out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
        nms=False,
        half=False,
    )
    out_path = Path(out)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if out_path != dst:
        dst.write_bytes(out_path.read_bytes())
    names = model.names
    print(f"exported {out_path} -> {dst}")
    print(f"classes ({len(names)}): {names if len(names) <= 12 else list(names.values())[:12]}")


def fine_tune(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        single_cls=True,
        patience=30,
        exist_ok=True,
    )
    # best checkpoint lives in <project>/<name>/weights/best.pt
    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.exists():
        sys.exit(f"training finished but {best} is missing")
    ns = argparse.Namespace(
        weights=str(best),
        out=args.out,
        imgsz=args.imgsz,
        opset=args.opset,
    )
    convert(ns)


def names(args: argparse.Namespace) -> None:
    p = Path(args.model)
    if p.suffix == ".onnx":
        import onnx
        from onnx import TensorProto

        m = onnx.load(str(p))
        found = False
        for prop in m.metadata_props:
            if prop.key == "names":
                print(prop.value)
                found = True
        if not found:
            print("no 'names' metadata - the model is single- or generic-class. "
                  "Use --class-name -1 to track every detection.")
        return
    from ultralytics import YOLO

    model = YOLO(str(p))
    print(model.names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="export .pt -> .onnx")
    c.add_argument("weights")
    c.add_argument("--out", default="models/mouse.onnx")
    c.add_argument("--imgsz", type=int, default=640)
    c.add_argument("--opset", type=int, default=17)
    c.set_defaults(func=convert)

    f = sub.add_parser("fine-tune", help="train on a dataset then export")
    f.add_argument("--data", required=True, help="data.yaml with train/val image lists")
    f.add_argument("--base", default="yolo11n.pt")
    f.add_argument("--epochs", type=int, default=80)
    f.add_argument("--imgsz", type=int, default=640)
    f.add_argument("--batch", type=int, default=16)
    f.add_argument("--device", default="cpu")
    f.add_argument("--project", default="runs")
    f.add_argument("--name", default="mouse_yolo11n")
    f.add_argument("--out", default="models/mouse.onnx")
    f.add_argument("--opset", type=int, default=17)
    f.set_defaults(func=fine_tune)

    n = sub.add_parser("names", help="print class names of a model")
    n.add_argument("model")
    n.set_defaults(func=names)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
