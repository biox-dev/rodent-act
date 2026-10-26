import { useMemo } from 'react';
import type { ScoredWindow } from '../api';

interface Props {
  windows: ScoredWindow[];
  duration: number;
  /** Current playback time in seconds. */
  current: number;
  onSeek: (t: number) => void;
}

/**
 * Per-track lane of 0.5 s windows: green = immobile, blue = mobile.
 * Clicking seeks the video to that window's midpoint.
 */
export default function Timeline({ windows, duration, current, onSeek }: Props) {
  const lanes = useMemo(() => {
    const byTrack = new Map<number, ScoredWindow[]>();
    for (const w of windows) {
      const list = byTrack.get(w.track_id);
      if (list) list.push(w);
      else byTrack.set(w.track_id, [w]);
    }
    return [...byTrack.entries()].sort((a, b) => a[0] - b[0]);
  }, [windows]);

  const span = Math.max(duration, ...windows.map((w) => w.end_t), 0.001);

  if (windows.length === 0) {
    return <div className="empty">暂无窗口数据</div>;
  }

  return (
    <div>
      <div className="timeline" style={{ height: `${Math.min(24 * lanes.length + 8, 220)}px` }}>
        {lanes.map(([trackId, list], lane) => (
          <div
            key={trackId}
            style={{ position: 'absolute', left: 0, right: 0, top: lane * 24 + 4, height: 20 }}
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
                  onClick={() => onSeek((w.start_t + w.end_t) / 2)}
                />
              );
            })}
          </div>
        ))}
        <div className="cursor" style={{ left: `${(current / span) * 100}%` }} />
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
        <span className="muted">{lanes.length} 条轨迹 · 每条一行</span>
      </div>
    </div>
  );
}
