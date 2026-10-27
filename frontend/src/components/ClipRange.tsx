import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

export function fmtClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00.0';
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
}

interface Props {
  /** Total duration of the source video, in seconds. */
  duration: number;
  start: number;
  end: number;
  onChange: (start: number, end: number) => void;
  /** Called while dragging with the time under the pointer (preview seek). */
  onScrub?: (t: number) => void;
}

/**
 * Draggable two-handle range over the video duration.
 *
 * The handles are plain divs driven by pointer events rather than nested
 * `<input type=range>` elements: overlapping native thumbs make it impossible
 * to grab the one underneath, and the visual selection needs to stay exact
 * when the window is very short.
 */
export default function ClipRange({ duration, start, end, onChange, onScrub }: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<'start' | 'end' | null>(null);
  // Latest props, read from the window pointer listener without re-binding it
  // on every mousemove.
  const latest = useRef({ duration, start, end, onChange, onScrub });
  latest.current = { duration, start, end, onChange, onScrub };

  const timeAt = useCallback((clientX: number): number => {
    const el = trackRef.current;
    const d = latest.current.duration;
    if (!el || d <= 0) return 0;
    const r = el.getBoundingClientRect();
    const f = r.width > 0 ? (clientX - r.left) / r.width : 0;
    return Math.min(Math.max(f, 0), 1) * d;
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (ev: PointerEvent) => {
      const { duration: d, start: s0, end: e0, onChange: emit, onScrub: scrub } =
        latest.current;
      if (d <= 0) return;
      const t = timeAt(ev.clientX);
      // Keep at least one 30 fps frame between the handles.
      const gap = Math.min(1 / 30, d);
      let s = s0;
      let en = e0;
      if (dragging === 'start') s = Math.min(t, en - gap);
      else en = Math.max(t, s + gap);
      emit(Math.min(Math.max(s, 0), d), Math.min(Math.max(en, 0), d));
      scrub?.(t);
    };
    const onUp = () => setDragging(null);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    };
  }, [dragging, timeAt]);

  const { startPct, endPct, spanPct } = useMemo(() => {
    if (duration <= 0) return { startPct: 0, endPct: 100, spanPct: 100 };
    const s = Math.min(Math.max(start / duration, 0), 1) * 100;
    const e = Math.min(Math.max(end / duration, 0), 1) * 100;
    return { startPct: s, endPct: e, spanPct: Math.max(e - s, 0) };
  }, [duration, start, end]);

  return (
    <div className={`clip-range${dragging ? ' dragging' : ''}`}>
      <div className="clip-track" ref={trackRef}>
        <div className="clip-selection" style={{ left: `${startPct}%`, width: `${spanPct}%` }} />
        <div
          className="clip-handle start"
          style={{ left: `${startPct}%` }}
          title="拖动设置起点"
          onPointerDown={(e) => {
            e.preventDefault();
            setDragging('start');
          }}
        >
          <span />
        </div>
        <div
          className="clip-handle end"
          style={{ left: `${endPct}%` }}
          title="拖动设置终点"
          onPointerDown={(e) => {
            e.preventDefault();
            setDragging('end');
          }}
        >
          <span />
        </div>
      </div>
      <div className="clip-ticks mono">
        <span>0:00.0</span>
        <span>{duration > 0 ? `选区 ${fmtClock(Math.max(end - start, 0))}` : '等待视频加载…'}</span>
        <span>{fmtClock(duration)}</span>
      </div>
    </div>
  );
}
