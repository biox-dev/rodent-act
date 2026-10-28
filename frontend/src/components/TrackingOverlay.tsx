import { useEffect, useRef } from 'react';
import type { DetectionRecord } from '../api';

/** Golden-angle hue stepping, mirroring `draw::id_color` in Rust. */
export function trackColor(id: number): string {
  const hue = (id * 137.508) % 360;
  return `hsl(${hue.toFixed(1)} 85% 60%)`;
}

interface Props {
  records: DetectionRecord[] | undefined;
  /** Source video geometry (box coordinates are in this space). */
  videoWidth: number;
  videoHeight: number;
  /** Median body diagonal per track, for the confidence ramp. */
  visible: boolean;
}

/**
 * Bounding-box overlay drawn on a canvas that exactly covers the letterboxed
 * video area, so boxes line up at any player size.
 */
export function TrackingOverlay({ records, videoWidth, videoHeight, visible }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const cssW = parent.clientWidth;
      const cssH = parent.clientHeight;
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
      canvas.style.width = `${cssW}px`;
      canvas.style.height = `${cssH}px`;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);
      if (!visible || !records || !videoWidth || !videoHeight) return;

      // object-fit: contain -> letterbox inside the stage
      const scale = Math.min(cssW / videoWidth, cssH / videoHeight);
      const offX = (cssW - videoWidth * scale) / 2;
      const offY = (cssH - videoHeight * scale) / 2;

      ctx.lineWidth = 2;
      ctx.font = '600 12px ui-monospace, monospace';
      ctx.textBaseline = 'bottom';

      for (const r of records) {
        const color = trackColor(r.track_id);
        const x = offX + r.x1 * scale;
        const y = offY + r.y1 * scale;
        const w = (r.x2 - r.x1) * scale;
        const h = (r.y2 - r.y1) * scale;
        ctx.strokeStyle = color;
        ctx.globalAlpha = r.measured ? 1 : 0.45;
        ctx.strokeRect(x, y, w, h);

        const label = `M-${r.track_id} ${r.score.toFixed(2)}`;
        const textW = ctx.measureText(label).width + 8;
        ctx.globalAlpha = 0.85;
        ctx.fillStyle = color;
        ctx.fillRect(x, Math.max(0, y - 16), textW, 16);
        ctx.globalAlpha = 1;
        ctx.fillStyle = '#04120f';
        ctx.fillText(label, x + 4, Math.max(12, y - 2));
      }
      ctx.globalAlpha = 1;
    };

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(parent);
    return () => ro.disconnect();
  }, [records, videoWidth, videoHeight, visible]);

  return <canvas ref={canvasRef} className="overlay" />;
}
