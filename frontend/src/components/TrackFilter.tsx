import { trackColor } from './TrackingOverlay';

interface Props {
  /** Every track id the caller knows about, in display order. */
  trackIds: readonly number[];
  /**
   * Track ids to hide. Undefined or empty shows every track.
   */
  hiddenTracks?: ReadonlySet<number>;
  /** Click on a chip; omit to render a non-interactive legend. */
  onToggleTrack?: (trackId: number) => void;
  /** Reset the filter; renders a "全部" chip once anything is hidden. */
  onShowAllTracks?: () => void;
  /** Hide every track at once; renders a "取消全部" chip. */
  onHideAllTracks?: () => void;
  /** Show the chips only when there is more than one track (default 1). */
  minTracks?: number;
  /** Optional caption rendered before the chips. */
  caption?: string;
}

/**
 * Shared track filter: one chip per track plus "取消全部" / "全部" shortcuts.
 * Used above the charts, the timeline and the per-track stats table so all of
 * them follow the same `hiddenTracks` state.
 */
export default function TrackFilter({
  trackIds,
  hiddenTracks,
  onToggleTrack,
  onShowAllTracks,
  onHideAllTracks,
  minTracks = 1,
  caption,
}: Props) {
  if (trackIds.length <= minTracks) return null;

  const anyHidden = trackIds.some((id) => hiddenTracks?.has(id) ?? false);

  return (
    <div className="chart-legend">
      {caption && <span className="muted">{caption}</span>}
      {trackIds.map((trackId) => {
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
      {onHideAllTracks && anyHidden === false && (
        <button
          type="button"
          className="legend-chip none"
          onClick={onHideAllTracks}
          title="隐藏全部轨迹"
        >
          取消全部
        </button>
      )}
      {anyHidden && onShowAllTracks && (
        <button type="button" className="legend-chip all" onClick={onShowAllTracks}>
          全部
        </button>
      )}
    </div>
  );
}
