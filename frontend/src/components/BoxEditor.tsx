import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import type { LabelBox } from '../api';

/**
 * Minimal YOLO box editor.
 *
 * - drag on empty space: draw a new box
 * - click a box: select it (drag to move, corner handles to resize)
 * - right-click a box: delete it
 *
 * Boxes are normalised (0..1) so they are independent of the display size.
 */

type Corner = 'nw' | 'ne' | 'sw' | 'se';

type Drag =
  | { kind: 'new'; x0: number; y0: number; x1: number; y1: number }
  | {
      kind: 'move';
      index: number;
      startX: number;
      startY: number;
      dx: number;
      dy: number;
      orig: LabelBox;
    }
  | {
      kind: 'resize';
      index: number;
      corner: Corner;
      startX: number;
      startY: number;
      dx: number;
      dy: number;
      orig: LabelBox;
    };

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);

function toRect(b: LabelBox, w: number, h: number) {
  const x1 = (b.xc - b.w / 2) * w;
  const y1 = (b.yc - b.h / 2) * h;
  return { x1, y1, x2: x1 + b.w * w, y2: y1 + b.h * h };
}

function toBox(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  w: number,
  h: number,
  classId: number,
): LabelBox {
  return {
    classId,
    xc: (x1 + x2) / 2 / w,
    yc: (y1 + y2) / 2 / h,
    w: Math.max(0, x2 - x1) / w,
    h: Math.max(0, y2 - y1) / h,
  };
}

/** Result of the in-progress gesture, in normalised coordinates. */
function applyDrag(d: Drag, w: number, h: number): LabelBox {
  if (d.kind === 'new') {
    return toBox(
      clamp(Math.min(d.x0, d.x1), 0, w),
      clamp(Math.min(d.y0, d.y1), 0, h),
      clamp(Math.max(d.x0, d.x1), 0, w),
      clamp(Math.max(d.y0, d.y1), 0, h),
      w,
      h,
      0,
    );
  }
  const r = toRect(d.orig, w, h);
  if (d.kind === 'move') {
    const bw = r.x2 - r.x1;
    const bh = r.y2 - r.y1;
    const x1 = clamp(r.x1 + d.dx, 0, w - bw);
    const y1 = clamp(r.y1 + d.dy, 0, h - bh);
    return toBox(x1, y1, x1 + bw, y1 + bh, w, h, d.orig.classId);
  }
  // Resize: move only the dragged corner.
  let x1 = r.x1;
  let y1 = r.y1;
  let x2 = r.x2;
  let y2 = r.y2;
  if (d.corner === 'nw' || d.corner === 'ne') y1 = clamp(r.y1 + d.dy, 0, h);
  else y2 = clamp(r.y2 + d.dy, 0, h);
  if (d.corner === 'nw' || d.corner === 'sw') x1 = clamp(r.x1 + d.dx, 0, w);
  else x2 = clamp(r.x2 + d.dx, 0, w);
  if (Math.abs(x2 - x1) < 2) {
    if (d.corner === 'nw' || d.corner === 'sw') x1 = Math.max(0, x2 - 2);
    else x2 = Math.min(w, x1 + 2);
  }
  if (Math.abs(y2 - y1) < 2) {
    if (d.corner === 'nw' || d.corner === 'ne') y1 = Math.max(0, y2 - 2);
    else y2 = Math.min(h, y1 + 2);
  }
  return toBox(
    Math.min(x1, x2),
    Math.min(y1, y2),
    Math.max(x1, x2),
    Math.max(y1, y2),
    w,
    h,
    d.orig.classId,
  );
}

interface Props {
  src: string;
  /** Natural image size (0 = read it from the loaded image). */
  width: number;
  height: number;
  boxes: LabelBox[];
  selected: number | null;
  onSelect: (index: number | null) => void;
  onChange: (boxes: LabelBox[]) => void;
}

export default function BoxEditor({
  src,
  width,
  height,
  boxes,
  selected,
  onSelect,
  onChange,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const dragRef = useRef<Drag | null>(null);
  const [, bump] = useReducer((x: number) => x + 1, 0);
  const [size, setSize] = useState({ w: Math.max(width, 1), h: Math.max(height, 1) });

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const { w, h } = size;
    const rect = canvas.getBoundingClientRect();
    const scale = rect.width > 0 ? w / rect.width : 1;

    ctx.clearRect(0, 0, w, h);
    const img = imgRef.current;
    if (img && img.complete && img.naturalWidth > 0) {
      ctx.drawImage(img, 0, 0, w, h);
    } else {
      ctx.fillStyle = '#0b0f14';
      ctx.fillRect(0, 0, w, h);
    }

    const drag = dragRef.current;
    const line = Math.max(1.5, Math.max(w, h) / 500);
    const font = Math.max(11, Math.max(w, h) / 55);
    const handle = Math.max(3, 4 * scale);

    boxes.forEach((b, i) => {
      const live =
        drag && drag.kind !== 'new' && drag.index === i ? applyDrag(drag, w, h) : b;
      const r = toRect(live, w, h);
      const isSel = i === selected;
      ctx.lineWidth = isSel ? line * 2 : line;
      ctx.strokeStyle = isSel ? '#4aa8ff' : '#29d398';
      ctx.strokeRect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);

      // Index tag above the top-left corner.
      ctx.font = `${font}px ui-sans-serif, system-ui, sans-serif`;
      ctx.textBaseline = 'bottom';
      ctx.fillStyle = isSel ? '#4aa8ff' : '#29d398';
      ctx.fillText(String(i + 1), r.x1 + 2, Math.max(font, r.y1 - 2));

      if (isSel) {
        ctx.fillStyle = '#4aa8ff';
        for (const [hx, hy] of [
          [r.x1, r.y1],
          [r.x2, r.y1],
          [r.x1, r.y2],
          [r.x2, r.y2],
        ]) {
          ctx.fillRect(hx - handle, hy - handle, handle * 2, handle * 2);
        }
      }
    });

    if (drag && drag.kind === 'new') {
      const x1 = Math.min(drag.x0, drag.x1);
      const y1 = Math.min(drag.y0, drag.y1);
      const x2 = Math.max(drag.x0, drag.x1);
      const y2 = Math.max(drag.y0, drag.y1);
      ctx.save();
      ctx.setLineDash([6 * scale, 5 * scale]);
      ctx.lineWidth = line;
      ctx.strokeStyle = '#f0b429';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.restore();
    }
  }, [boxes, selected, size]);

  const drawRef = useRef(draw);
  drawRef.current = draw;

  // Load the frame image; the natural size wins over the metadata we sent.
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      setSize({
        w: img.naturalWidth || Math.max(width, 1),
        h: img.naturalHeight || Math.max(height, 1),
      });
      drawRef.current();
    };
    img.onerror = () => {
      imgRef.current = null;
      drawRef.current();
    };
    img.src = src;
    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [src, width, height]);

  useEffect(() => {
    draw();
  }, [draw]);

  const point = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const sx = rect.width > 0 ? size.w / rect.width : 1;
    const sy = rect.height > 0 ? size.h / rect.height : 1;
    return {
      x: (e.clientX - rect.left) * sx,
      y: (e.clientY - rect.top) * sy,
      scale: sx,
    };
  };

  const hitTest = (x: number, y: number): number | null => {
    for (let i = boxes.length - 1; i >= 0; i--) {
      const r = toRect(boxes[i], size.w, size.h);
      if (x >= r.x1 && x <= r.x2 && y >= r.y1 && y <= r.y2) return i;
    }
    return null;
  };

  const handleHit = (x: number, y: number, scale: number): Corner | null => {
    if (selected == null || !boxes[selected]) return null;
    const r = toRect(boxes[selected], size.w, size.h);
    const hs = Math.max(6, 9 * scale);
    const corners: [Corner, number, number][] = [
      ['nw', r.x1, r.y1],
      ['ne', r.x2, r.y1],
      ['sw', r.x1, r.y2],
      ['se', r.x2, r.y2],
    ];
    for (const [corner, hx, hy] of corners) {
      if (Math.abs(x - hx) <= hs && Math.abs(y - hy) <= hs) return corner;
    }
    return null;
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    const { x, y, scale } = point(e);
    e.currentTarget.setPointerCapture(e.pointerId);

    const corner = handleHit(x, y, scale);
    if (corner != null && selected != null) {
      dragRef.current = {
        kind: 'resize',
        index: selected,
        corner,
        startX: x,
        startY: y,
        dx: 0,
        dy: 0,
        orig: boxes[selected],
      };
      bump();
      return;
    }

    const hit = hitTest(x, y);
    if (hit != null) {
      onSelect(hit);
      dragRef.current = {
        kind: 'move',
        index: hit,
        startX: x,
        startY: y,
        dx: 0,
        dy: 0,
        orig: boxes[hit],
      };
      bump();
      return;
    }

    onSelect(null);
    dragRef.current = { kind: 'new', x0: x, y0: y, x1: x, y1: y };
    bump();
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = dragRef.current;
    if (!d) return;
    const { x, y } = point(e);
    if (d.kind === 'new') {
      d.x1 = x;
      d.y1 = y;
    } else {
      d.dx = x - d.startX;
      d.dy = y - d.startY;
    }
    bump();
  };

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const d = dragRef.current;
    dragRef.current = null;
    if (!d) return;
    const { scale } = point(e);
    const { w, h } = size;

    if (d.kind === 'new') {
      const ok =
        Math.abs(d.x1 - d.x0) >= 3 * scale && Math.abs(d.y1 - d.y0) >= 3 * scale;
      if (ok) {
        const box = applyDrag(d, w, h);
        onChange([...boxes, box]);
        onSelect(boxes.length);
      }
    } else {
      onChange(boxes.map((b, i) => (i === d.index ? applyDrag(d, w, h) : b)));
      onSelect(d.index);
    }
    bump();
  };

  const onContextMenu = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const sx = rect.width > 0 ? size.w / rect.width : 1;
    const sy = rect.height > 0 ? size.h / rect.height : 1;
    const x = (e.clientX - rect.left) * sx;
    const y = (e.clientY - rect.top) * sy;
    const hit = hitTest(x, y);
    if (hit != null) {
      onChange(boxes.filter((_, i) => i !== hit));
      onSelect(null);
    }
  };

  return (
    <div className="box-editor">
      <canvas
        ref={canvasRef}
        width={size.w}
        height={size.h}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onContextMenu={onContextMenu}
      />
    </div>
  );
}
