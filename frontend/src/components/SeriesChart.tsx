import { useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import type { ScoredWindow } from '../api';
import { trackColor } from './TrackingOverlay';
import TrackFilter from './TrackFilter';

interface Props {
  windows: ScoredWindow[];
  /** Playback time in seconds; draws a cursor line. */
  current?: number;
  height?: number;
  mode: 'prob' | 'speed';
  /** Feature index for `speed` mode (e.g. speed_mean). */
  featureIndex?: number;
  /**
   * Total video duration. While scoring is still running the data only
   * covers part of the video; anchoring the axis to the full duration keeps
   * the curve growing left-to-right instead of rescaling on every frame.
   */
  duration?: number;
  /**
   * Track ids to hide (controlled by the caller so several charts can share
   * one filter). Undefined or empty shows every track.
   */
  hiddenTracks?: ReadonlySet<number>;
  /** Click on a legend chip; omit to render a non-interactive legend. */
  onToggleTrack?: (trackId: number) => void;
  /** Reset the filter; renders a "全部" chip once anything is hidden. */
  onShowAllTracks?: () => void;
  /** Hide every track at once; renders a "取消全部" chip. */
  onHideAllTracks?: () => void;
  /**
   * Click or drag on the plot seeks the video to that clip-relative time.
   * Omit to keep the chart read-only.
   */
  onSeek?: (t: number) => void;
  /**
   * Decision threshold drawn as a dashed line in `prob` mode. Defaults to 0.5;
   * the job page passes the threshold that actually produced the labels.
   */
  threshold?: number;
}

/**
 * Dependency-free SVG line chart: immobility probability (or a raw feature)
 * per window, one polyline per track.
 */
export default function SeriesChart({
  windows,
  current,
  height = 180,
  mode,
  featureIndex = 0,
  duration,
  hiddenTracks,
  onToggleTrack,
  onShowAllTracks,
  onHideAllTracks,
  onSeek,
  threshold = 0.5,
}: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  // Time under the mouse (or the dragged cursor); null = no vertical line.
  const [hoverT, setHoverT] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);

  const series = useMemo(() => {
    const byTrack = new Map<number, { t: number; v: number }[]>();
    for (const w of windows) {
      const v = mode === 'prob' ? w.immobile_prob : (w.values[featureIndex] ?? 0);
      const t = (w.start_t + w.end_t) / 2;
      const list = byTrack.get(w.track_id);
      if (list) list.push({ t, v });
      else byTrack.set(w.track_id, [{ t, v }]);
    }
    return [...byTrack.entries()].sort((a, b) => a[0] - b[0]);
  }, [windows, mode, featureIndex]);

  // Only draw the tracks the caller lets through, but keep the full list for
  // the legend so hidden tracks can be switched back on.
  const visible = useMemo(
    () =>
      hiddenTracks && hiddenTracks.size > 0
        ? series.filter(([trackId]) => !hiddenTracks.has(trackId))
        : series,
    [series, hiddenTracks],
  );

  const W = 800;
  const H = height;
  const pad = { l: 38, r: 10, t: 10, b: 22 };

  const { maxT, maxV } = useMemo(() => {
    let mt = 0.001;
    let mv = mode === 'prob' ? 1 : 0.001;
    for (const [, pts] of visible) {
      for (const p of pts) {
        mt = Math.max(mt, p.t);
        mv = Math.max(mv, p.v);
      }
    }
    return { maxT: Math.max(mt, duration ?? 0), maxV: mv * 1.05 };
  }, [visible, mode, duration]);

  if (series.length === 0) return <div className="empty">暂无数据</div>;

  const legend = (
    <TrackFilter
      trackIds={series.map(([trackId]) => trackId)}
      hiddenTracks={hiddenTracks}
      onToggleTrack={onToggleTrack}
      onShowAllTracks={onShowAllTracks}
      onHideAllTracks={onHideAllTracks}
      minTracks={1}
    />
  );

  if (visible.length === 0) {
    return (
      <div className="chart-block">
        {legend}
        <div className="empty">已隐藏全部轨迹，点击图例恢复</div>
      </div>
    );
  }

  const x = (t: number) => pad.l + (t / maxT) * (W - pad.l - pad.r);
  const y = (v: number) => H - pad.b - (v / maxV) * (H - pad.t - pad.b);

  const ticks = mode === 'prob' ? [0, 0.25, 0.5, 0.75, 1] : [0, maxV / 2, maxV];
  const plotW = W - pad.l - pad.r;

  // The SVG is stretched non-uniformly (preserveAspectRatio="none"), so map
  // client x through the element box into viewBox units, then into seconds.
  const timeAt = (clientX: number): number => {
    const svg = svgRef.current;
    if (!svg || plotW <= 0) return 0;
    const rect = svg.getBoundingClientRect();
    if (rect.width <= 0) return 0;
    const vx = ((clientX - rect.left) / rect.width) * W;
    const frac = (vx - pad.l) / plotW;
    return Math.min(Math.max(frac, 0), 1) * maxT;
  };

  const handlePointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!onSeek) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
    const t = timeAt(e.clientX);
    setHoverT(t);
    onSeek(t);
  };

  const handlePointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    const t = timeAt(e.clientX);
    setHoverT(t);
    // While dragging, scrub the video in step with the cursor.
    if (dragging) onSeek?.(t);
  };

  const handlePointerUp = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    setDragging(false);
  };

  return (
    <div className="chart-block">
      {legend}
      <svg
        ref={svgRef}
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        role="img"
        style={{
          cursor: onSeek ? (dragging ? 'grabbing' : 'crosshair') : 'default',
          touchAction: 'none',
        }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onPointerLeave={() => {
          if (!dragging) setHoverT(null);
        }}
      >
        {ticks.map((v) => (
          <g key={v}>
            <line
              x1={pad.l}
              x2={W - pad.r}
              y1={y(v)}
              y2={y(v)}
              stroke="var(--border)"
              strokeDasharray={v === 0 ? undefined : '3 4'}
            />
            <text x={4} y={y(v) + 4} fill="var(--muted)" fontSize={10}>
              {mode === 'prob' ? v.toFixed(2) : v.toFixed(2)}
            </text>
          </g>
        ))}
        {mode === 'prob' && (
          <g>
            <line
              x1={pad.l}
              x2={W - pad.r}
              y1={y(threshold)}
              y2={y(threshold)}
              stroke="var(--warn)"
              strokeWidth={1}
              strokeDasharray="6 4"
              opacity={0.8}
            />
            <text
              x={W - pad.r - 4}
              y={y(threshold) - 4}
              textAnchor="end"
              fill="var(--warn)"
              fontSize={10}
            >
              {threshold.toFixed(2)}
            </text>
          </g>
        )}
        {visible.map(([trackId, pts]) => (
          <polyline
            key={trackId}
            fill="none"
            stroke={trackColor(trackId)}
            strokeWidth={1.8}
            vectorEffect="non-scaling-stroke"
            points={pts.map((p) => `${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(' ')}
          />
        ))}
        {current !== undefined && current > 0 && (
          <line
            x1={x(Math.min(current, maxT))}
            x2={x(Math.min(current, maxT))}
            y1={pad.t}
            y2={H - pad.b}
            stroke="#fff"
            strokeWidth={1}
            opacity={0.6}
          />
        )}
        {hoverT !== null && (
          <g pointerEvents="none">
            <line
              x1={x(hoverT)}
              x2={x(hoverT)}
              y1={pad.t}
              y2={H - pad.b}
              stroke="var(--accent)"
              strokeWidth={1}
              strokeDasharray="4 3"
            />
            <text
              x={Math.min(Math.max(x(hoverT), pad.l + 22), W - pad.r - 22)}
              y={pad.t + 10}
              fill="var(--text)"
              fontSize={10}
              textAnchor="middle"
            >
              {hoverT.toFixed(2)}s
            </text>
          </g>
        )}
        <text x={pad.l} y={H - 6} fill="var(--muted)" fontSize={10}>
          0s
        </text>
        <text x={W - pad.r} y={H - 6} fill="var(--muted)" fontSize={10} textAnchor="end">
          {maxT.toFixed(1)}s
        </text>
      </svg>
    </div>
  );
}
