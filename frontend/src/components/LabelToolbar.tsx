import type { SplitMember } from '../api';

/** Where the auto-save debounce currently stands. */
export type SaveState = 'idle' | 'dirty' | 'saving' | 'saved';

export type AssignTarget = 'train' | 'val' | 'none';

interface Props {
  /** 0-based index of the current frame, and the number of frames. */
  index: number;
  count: number;
  hasPrev: boolean;
  hasNext: boolean;
  onStep: (delta: -1 | 1) => void;
  /** Current frame file name (`null` when the dataset has no images). */
  frameName?: string | null;
  /** True once `data.yaml` exists (train/val folders were generated). */
  hasSplit: boolean;
  /** Membership of the current frame (`null` = in neither split). */
  member?: SplitMember | null;
  /** Boxes on the current frame; 0 disables 清空. */
  boxCount: number;
  onClear: () => void;
  onAssign: (target: AssignTarget) => void;
  assignBusy?: boolean;
  canCopyPrevious?: boolean;
  onCopyPrevious?: () => void;
  saveState: SaveState;
}

/**
 * Toolbar of the label editor: frame navigation, per-frame train/val
 * assignment, clear-all and the auto-save indicator.
 */
export default function LabelToolbar({
  index,
  count,
  hasPrev,
  hasNext,
  onStep,
  frameName,
  hasSplit,
  member = null,
  boxCount,
  onClear,
  onAssign,
  assignBusy = false,
  canCopyPrevious = false,
  onCopyPrevious,
  saveState,
}: Props) {
  const inTrain = member === 'train' || member === 'both';
  const inVal = member === 'val' || member === 'both';
  const assignDisabled = assignBusy || !frameName || !hasSplit;

  const saveText =
    saveState === 'saving'
      ? '保存中…'
      : saveState === 'dirty'
        ? '未保存…'
        : saveState === 'saved'
          ? '已保存 ✓'
          : '';

  return (
    <div className="editor-toolbar">
      <button className="ghost" onClick={() => onStep(-1)} disabled={!hasPrev}>
        ← 上一帧
      </button>
      <button className="ghost" onClick={() => onStep(1)} disabled={!hasNext}>
        下一帧 →
      </button>
      <span className="mono muted">
        {count ? index + 1 : 0} / {count}
      </span>
      {hasSplit && (
        <span className="frame-split-state">
          {inTrain && <em className="split-chip train">train</em>}
          {inVal && <em className="split-chip val">val</em>}
          {!inTrain && !inVal && <span className="muted">未分配</span>}
        </span>
      )}
      <span className="spacer" />
      <button className="ghost" onClick={onCopyPrevious} disabled={!canCopyPrevious}>
        复制上一帧
      </button>
      <button
        className="ghost"
        onClick={() => onAssign('train')}
        disabled={assignDisabled}
        title="把当前帧复制到 images/train（同时从 val 移除）"
      >
        移入训练集
      </button>
      <button
        className="ghost"
        onClick={() => onAssign('val')}
        disabled={assignDisabled}
        title="把当前帧复制到 images/val（同时从 train 移除）"
      >
        移入验证集
      </button>
      <button
        className="ghost"
        onClick={() => onAssign('none')}
        disabled={assignDisabled || (!inTrain && !inVal)}
        title="从 images/train 和 images/val 中移除（保留 raw/ 原图）"
      >
        移出划分
      </button>
      <button className="ghost" onClick={onClear} disabled={boxCount <= 0}>
        清空
      </button>
      <span className="save-chip">{saveText}</span>
    </div>
  );
}
