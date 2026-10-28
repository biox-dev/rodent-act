import { useMemo } from 'react';
import type { ScoredWindow } from '../api';
import { trackColor } from './TrackingOverlay';

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
}: Props) {
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

  const legend =
    series.length > 1 ? (
      <div className="chart-legend">
        {series.map(([trackId]) => {
          const off = hiddenTracks?.has(trackId) ?? false;
          const color = trackColor(trackId);
          return (
            <button
              key={trackId}
              type="button"
              className={`legend-chip${off ? ' off' : ''}`}
              disabled={!onToggleTrack}
              onClick={() => onToggleTrack?.(trackId)}
              title={onToggleTrack ? '点击显示/隐藏该轨迹' : undefined}
            >
              <span
                className="dot"
                style={{ background: off ? 'transparent' : color, borderColor: color }}
              />
              ID {trackId}
            </button>
          );
        })}
        {hiddenTracks && hiddenTracks.size > 0 && onShowAllTracks && (
          <button type="button" className="legend-chip all" onClick={onShowAllTracks}>
            全部
          </button>
        )}
      </div>
    ) : null;

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

  return (
    <div className="chart-block">
      {legend}
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img">
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
          <line
            x1={pad.l}
            x2={W - pad.r}
            y1={y(0.5)}
            y2={y(0.5)}
            stroke="var(--warn)"
            strokeWidth={1}
            strokeDasharray="6 4"
            opacity={0.8}
          />
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
