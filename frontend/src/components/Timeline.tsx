import { useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import type { ScoredWindow } from '../api';
import { trackColor } from './TrackingOverlay';
import TrackFilter from './TrackFilter';

interface Props {
  windows: ScoredWindow[];
  duration: number;
  /** Current playback time in seconds. */
  current: number;
  onSeek: (t: number) => void;
  /**
   * Track ids to hide; shared with the charts so one filter drives every view.
   * Undefined or empty shows every track.
   */
  hiddenTracks?: ReadonlySet<number>;
  /** Click on a filter chip; omit to render a non-interactive legend. */
  onToggleTrack?: (trackId: number) => void;
  /** Reset the filter; renders a "全部" chip once anything is hidden. */
  onShowAllTracks?: () => void;
  /** Hide every track at once; renders a "取消全部" chip. */
  onHideAllTracks?: () => void;
}

/** Lane pitch in px; must match the `top`/`height` used for the bars. */
const LANE_H = 24;

/**
 * Per-track lane of 0.5 s windows: green = immobile, blue = mobile.
 * Hovering shows a vertical cursor, dragging scrubs, clicking seeks the video
 * to that exact time.
 */
export default function Timeline({
  windows,
  duration,
  current,
  onSeek,
  hiddenTracks,
  onToggleTrack,
  onShowAllTracks,
  onHideAllTracks,
}: Props) {
  const trackRef = useRef<HTMLDivElement | null>(null);
  // Time under the mouse (or the dragged cursor); null = no vertical line.
  const [hoverT, setHoverT] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);

  const lanes = useMemo(() => {
    const byTrack = new Map<number, ScoredWindow[]>();
    for (const w of windows) {
      const list = byTrack.get(w.track_id);
      if (list) list.push(w);
      else byTrack.set(w.track_id, [w]);
    }
    return [...byTrack.entries()].sort((a, b) => a[0] - b[0]);
  }, [windows]);

  // Hidden tracks keep their chip (so they can be switched back on) but lose
  // their lane.
  const visibleLanes = useMemo(
    () =>
      hiddenTracks && hiddenTracks.size > 0
        ? lanes.filter(([trackId]) => !hiddenTracks.has(trackId))
        : lanes,
    [lanes, hiddenTracks],
  );

  const span = Math.max(duration, ...windows.map((w) => w.end_t), 0.001);

  if (windows.length === 0) {
    return <div className="empty">暂无窗口数据</div>;
  }

  const filter = (
    <TrackFilter
      trackIds={lanes.map(([trackId]) => trackId)}
      hiddenTracks={hiddenTracks}
      onToggleTrack={onToggleTrack}
      onShowAllTracks={onShowAllTracks}
      onHideAllTracks={onHideAllTracks}
    />
  );

  if (visibleLanes.length === 0) {
    return (
      <div className="chart-block">
        {filter}
        <div className="empty">已隐藏全部轨迹，点击图例恢复</div>
      </div>
    );
  }

  const height = Math.min(LANE_H * visibleLanes.length + 8, 220);

  // Map a viewport x coordinate to a clip-relative time.
  const timeAt = (clientX: number): number => {
    const el = trackRef.current;
    if (!el) return 0;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0) return 0;
    const frac = (clientX - rect.left) / rect.width;
    return Math.min(Math.max(frac, 0), 1) * span;
  };

  const handlePointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
    const t = timeAt(e.clientX);
    setHoverT(t);
    onSeek(t);
  };

  const handlePointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const t = timeAt(e.clientX);
    setHoverT(t);
    // While dragging, scrub the video in step with the cursor.
    if (dragging) onSeek(t);
  };

  const handlePointerUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    setDragging(false);
  };

  // Keep the time badge inside the lane strip: near the edges it anchors to
  // the line instead of centring on it.
  const hoverPct = hoverT !== null ? (hoverT / span) * 100 : null;
  const hoverAnchor = hoverPct === null || hoverPct > 88 ? -100 : hoverPct < 12 ? 0 : -50;

  return (
    <div className="chart-block">
      {filter}
      <div>
        <div className="timeline-row">
          {/* Track ids in a gutter, so each lane can be identified without
              hovering the bars. Colours match the SeriesChart legend. */}
          <div className="timeline-labels" style={{ height }}>
            {visibleLanes.map(([trackId], lane) => (
              <div
                key={trackId}
                className="lane-label"
                style={{ top: lane * LANE_H + 4, color: trackColor(trackId) }}
                title={`track ${trackId}`}
              >
                {trackId}
              </div>
            ))}
          </div>
          <div
            ref={trackRef}
            className="timeline"
            style={{
              height,
              cursor: dragging ? 'grabbing' : 'crosshair',
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
            {visibleLanes.map(([trackId, list], lane) => (
              <div
                key={trackId}
                style={{ position: 'absolute', left: 0, right: 0, top: lane * LANE_H + 4, height: 20 }}
              >
                {list.map((w) => {
                  const left = (w.start_t / span) * 100;
                  const width = Math.max(((w.end_t - w.start_t) / span) * 100, 0.2);
                  const color = w.label === 'immobile' ? 'var(--immobile)' : 'var(--mobile)';
                  const title = `track ${w.track_id} · ${w.start_t.toFixed(2)}–${w.end_t.toFixed(2)}s · ${w.label} · p(immobile)=${w.immobile_prob.toFixed(3)} · RII=${w.rii.toFixed(2)}`;
                  return (
                    <div
                      key={`${w.track_id}-${w.start_frame}`}
                      className="bar"
                      title={title}
                      style={{
                        left: `${left}%`,
                        width: `${width}%`,
                        background: color,
                        opacity: trackId % 2 === 0 ? 0.95 : 0.8,
                      }}
                    />
                  );
                })}
              </div>
            ))}
            <div className="cursor" style={{ left: `${(current / span) * 100}%` }} />
            {hoverPct !== null && (
              <div className="hover-cursor" style={{ left: `${hoverPct}%` }}>
                <span className="hover-time" style={{ transform: `translateX(${hoverAnchor}%)` }}>
                  {hoverT?.toFixed(2)}s
                </span>
              </div>
            )}
          </div>
        </div>
        <div className="legend" style={{ marginTop: 8 }}>
          <span>
            <i className="dot" style={{ background: 'var(--immobile)' }} />
            不动 (immobile)
          </span>
          <span>
            <i className="dot" style={{ background: 'var(--mobile)' }} />
            活动 (mobile)
          </span>
          <span className="muted">
            {visibleLanes.length === lanes.length
              ? `${lanes.length} 条轨迹`
              : `${visibleLanes.length} / ${lanes.length} 条轨迹`}{' '}
            · 每条一行
          </span>
        </div>
      </div>
    </div>
  );
}
