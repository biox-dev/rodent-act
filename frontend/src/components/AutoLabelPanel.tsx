import { useState } from 'react';
import type { AutolabelOptions } from '../api';

/** ROI rectangle, normalised to 0..1. */
export interface AutoLabelRoi {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Every knob of the blob-based pre-labeler (`POST /api/datasets/{name}/autolabel`). */
export interface AutoLabelConfig {
  /** How much darker than the frame median a pixel must be. */
  threshold: number;
  /** Smallest accepted blob, as a fraction of the frame area. */
  minArea: number;
  /** Hard cap on boxes per frame. */
  maxObjects: number;
  /** Expected number of channels (0 = unlimited). */
  channels: number;
  /** Keep frames that already have labels untouched. */
  onlyEmpty: boolean;
  useRoi: boolean;
  roi: AutoLabelRoi;
}

export const defaultAutoLabelConfig: AutoLabelConfig = {
  threshold: 45,
  minArea: 0.0008,
  maxObjects: 12,
  channels: 6,
  onlyEmpty: false,
  useRoi: true,
  roi: { x1: 0.1, y1: 0.1, x2: 0.98, y2: 0.95 },
};

/** Convert the UI config into the request body expected by the API. */
export function autoLabelOptions(c: AutoLabelConfig): AutolabelOptions {
  return {
    threshold: c.threshold,
    minArea: c.minArea,
    maxObjects: c.maxObjects,
    channels: c.channels,
    onlyEmpty: c.onlyEmpty,
    roi: c.useRoi ? [c.roi.x1, c.roi.y1, c.roi.x2, c.roi.y2] : undefined,
  };
}

const ROI_FIELDS = [
  ['x1', 'ROI 左', '左'],
  ['y1', 'ROI 上', '上'],
  ['x2', 'ROI 右', '右'],
  ['y2', 'ROI 下', '下'],
] as const;

interface FieldsProps {
  value: AutoLabelConfig;
  onChange: (next: AutoLabelConfig) => void;
  /** Narrow side panels: shorter labels and no per-field hints. */
  compact?: boolean;
}

/**
 * Controlled pre-label form. Shared by the dataset extractor (wide form) and
 * the label editor's side panel (compact).
 */
export function AutoLabelFields({ value, onChange, compact = false }: FieldsProps) {
  const set = <K extends keyof AutoLabelConfig>(key: K, v: AutoLabelConfig[K]) =>
    onChange({ ...value, [key]: v });

  return (
    <>
      <div className="row">
        <label className="field">
          暗度阈值
          <input
            type="number"
            min={1}
            max={120}
            value={value.threshold}
            onChange={(e) => set('threshold', Number(e.target.value) || 45)}
          />
          {!compact && (
            <span className="hint">比全图中位亮度暗多少算目标。调高更保守，调低更灵敏。</span>
          )}
        </label>
        <label className="field">
          {compact ? '最小面积' : '最小面积比例'}
          <input
            type="number"
            step={0.0002}
            value={value.minArea}
            onChange={(e) => set('minArea', Number(e.target.value) || 0.0008)}
          />
          {!compact && (
            <span className="hint">过滤反光/噪点，0.0008 ≈ 640×480 下约 245 像素。</span>
          )}
        </label>
        <label className="field">
          {compact ? '通道数' : '每帧最多几个通道'}
          <input
            type="number"
            min={0}
            value={value.channels}
            onChange={(e) => set('channels', Math.max(0, Number(e.target.value) || 0))}
          />
          {!compact && <span className="hint">装置是 6 通道就填 6；0 = 不限制。</span>}
        </label>
        <label className="field">
          候选上限
          <input
            type="number"
            min={1}
            value={value.maxObjects}
            onChange={(e) => set('maxObjects', Math.max(1, Number(e.target.value) || 12))}
          />
        </label>
      </div>

      <label className="field checkbox" style={{ marginTop: compact ? 8 : 10 }}>
        <input
          type="checkbox"
          checked={value.onlyEmpty}
          onChange={(e) => set('onlyEmpty', e.target.checked)}
        />
        {compact ? '只标空帧（保留手工标注）' : '只标空帧（不覆盖已有框）'}
      </label>
      <label className="field checkbox" style={{ marginTop: compact ? 4 : 10 }}>
        <input
          type="checkbox"
          checked={value.useRoi}
          onChange={(e) => set('useRoi', e.target.checked)}
        />
        {compact ? '使用 ROI' : '限制 ROI（推荐，去掉灯管/线缆/手）'}
      </label>
      {value.useRoi && (
        <div className="row" style={{ marginTop: 8 }}>
          {ROI_FIELDS.map(([key, label, compactLabel]) => (
            <label className="field" key={key}>
              {compact ? compactLabel : `${label} (${key})`}
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={value.roi[key]}
                onChange={(e) =>
                  onChange({
                    ...value,
                    roi: { ...value.roi, [key]: Number(e.target.value) || 0 },
                  })
                }
              />
            </label>
          ))}
        </div>
      )}
    </>
  );
}

interface PanelProps {
  /**
   * Run the pre-label request. The resolved string (if any) is shown as a
   * success notice; throw to show an error instead.
   */
  onRun: (opts: AutolabelOptions) => Promise<string | void>;
  disabled?: boolean;
  /** Tooltip shown while the run button is disabled. */
  disabledHint?: string;
  /** Overrides for the initial form values. */
  initial?: Partial<AutoLabelConfig>;
  subtitle?: string;
}

/**
 * "自动预标注" card: form + run button + result, self-contained so any page can
 * drop it in.
 */
export default function AutoLabelPanel({
  onRun,
  disabled,
  disabledHint,
  initial,
  subtitle = '覆盖当前 .txt',
}: PanelProps) {
  const [cfg, setCfg] = useState<AutoLabelConfig>(() => ({
    ...defaultAutoLabelConfig,
    ...initial,
  }));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    setDone(null);
    try {
      const msg = await onRun(autoLabelOptions(cfg));
      if (msg) setDone(msg);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <h3>
        自动预标注<span className="sub">{subtitle}</span>
      </h3>
      <AutoLabelFields value={cfg} onChange={setCfg} compact />
      <div className="row" style={{ marginTop: 10 }}>
        <button
          onClick={() => void run()}
          disabled={busy || disabled}
          title={disabled ? disabledHint : undefined}
        >
          {busy ? '标注中…' : '开始预标注'}
        </button>
      </div>
      {error && (
        <div className="error-box" style={{ marginTop: 8 }}>
          {error}
        </div>
      )}
      {done && (
        <div className="ok-box" style={{ marginTop: 8 }}>
          {done}
        </div>
      )}
      <div className="hint" style={{ marginTop: 8, fontSize: 11 }}>
        预标注会写回每张图的 <code>.txt</code>；已手工修正的帧请勾选「只标空帧」。
      </div>
    </div>
  );
}
