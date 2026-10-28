import { useEffect, useMemo, useState } from 'react';
import { Checkbox, Input, InputNumber, Modal, Select } from 'antd';
import { api, type DatasetSummary, type LabelBox } from '../api';

/** One frame grabbed from the player, frozen while the dialog is open. */
export interface CapturedFrame {
  /** `data:image/jpeg;base64,...` */
  dataUrl: string;
  /** Pixel size of the stored image (labels are relative, so any size works). */
  width: number;
  height: number;
  /** Source frame index, used in the file name (`<prefix>_000123.jpg`). */
  frame: number;
  /** Tracking boxes normalised to 0..1. */
  boxes: LabelBox[];
}

interface Props {
  /** Grab whatever the player shows right now (null when unavailable). */
  capture: () => CapturedFrame | null;
  /** File-name prefix, normally the video stem. */
  prefix?: string;
  disabled?: boolean;
  /** Why the button is disabled (shown as a tooltip). */
  disabledHint?: string;
}

/**
 * "加入训练集" button + dialog: freeze the current video frame, pick a dataset
 * and write the image with its tracked boxes into `<dataset>/raw/`, so hard
 * frames can be re-used for fine-tuning.
 */
export default function AddFrameToDataset({ capture, prefix, disabled, disabledHint }: Props) {
  const [open, setOpen] = useState(false);
  const [shot, setShot] = useState<CapturedFrame | null>(null);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [name, setName] = useState('');
  const [classId, setClassId] = useState(0);
  const [overwrite, setOverwrite] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);

  const start = () => {
    const grabbed = capture();
    setShot(grabbed);
    setError(grabbed ? null : '拿不到当前画面：请先播放/拖动视频到有画面的位置');
    setDone(null);
    setOverwrite(false);
    // Default to the dataset extracted from this same video, if any.
    setName((prev) => (prev.trim() ? prev : (prefix ?? '')));
    setOpen(true);
  };

  // Re-read the list on every open: frames may have been added elsewhere.
  useEffect(() => {
    if (!open) return;
    let alive = true;
    setLoading(true);
    api
      .listDatasets()
      .then((list) => {
        if (alive) setDatasets(list);
      })
      .catch((e) => {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [open]);

  const options = useMemo(
    () =>
      datasets.map((d) => ({
        value: d.name,
        label: `${d.name} · ${d.nImages} 帧 / ${d.nBoxes} 框`,
      })),
    [datasets],
  );

  const submit = async () => {
    if (!shot) return;
    const target = name.trim();
    if (!target) {
      setError('请选择或输入数据集名称');
      return;
    }
    setBusy(true);
    setError(null);
    setDone(null);
    try {
      const res = await api.captureDatasetFrame(target, {
        image: shot.dataUrl,
        boxes: shot.boxes.map((b) => ({ ...b, classId })),
        prefix,
        frame: shot.frame,
        overwrite,
      });
      setDone(
        `已写入 ${target}/raw/${res.image}（${res.boxes} 个框）· 数据集现有 ` +
          `${res.dataset.nImages} 帧 / ${res.dataset.nBoxes} 个框` +
          (res.overwritten ? ' · 已覆盖同名帧' : ''),
      );
      setDatasets((prev) =>
        [...prev.filter((d) => d.name !== res.dataset.name), res.dataset].sort((a, b) =>
          a.name.localeCompare(b.name),
        ),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ marginLeft: 'auto' }}>
      <button
        className="ghost"
        onClick={start}
        disabled={disabled}
        title={disabled ? disabledHint : '把当前帧图片 + 检测框写入数据集，用于 fine-tune'}
      >
        加入训练集
      </button>
      <Modal
        open={open}
        title="当前帧加入训练集"
        width={560}
        onCancel={() => setOpen(false)}
        footer={null}
      >
        {shot ? (
          <>
            <div className="capture-preview">
              <img src={shot.dataUrl} alt="current frame" />
              {shot.boxes.map((b, i) => (
                <span
                  key={i}
                  className="capture-box"
                  style={{
                    left: `${(b.xc - b.w / 2) * 100}%`,
                    top: `${(b.yc - b.h / 2) * 100}%`,
                    width: `${b.w * 100}%`,
                    height: `${b.h * 100}%`,
                  }}
                />
              ))}
              <span className="capture-meta">
                frame {shot.frame} · {shot.width}×{shot.height} · {shot.boxes.length} 个框
              </span>
            </div>

            <div className="row">
              <label className="field" style={{ minWidth: 240 }}>
                数据集
                <Select
                  size="small"
                  style={{ width: 240 }}
                  loading={loading}
                  options={options}
                  value={datasets.some((d) => d.name === name) ? name : undefined}
                  onChange={(v) => setName(v ?? '')}
                  placeholder="选择已有数据集"
                  showSearch
                  allowClear
                  notFoundContent={loading ? '加载中…' : '还没有数据集，直接输入新名称'}
                />
                <span className="hint">选择已有数据集，或直接在下方输入新名称创建。</span>
              </label>
              <label className="field" style={{ minWidth: 180 }}>
                数据集名称
                <Input
                  size="small"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="例如 TST_2026_01"
                />
                <span className="hint">只允许字母/数字/_/-。</span>
              </label>
              <label className="field" style={{ minWidth: 110 }}>
                类别 ID
                <InputNumber
                  size="small"
                  min={0}
                  max={99}
                  value={classId}
                  onChange={(v) => setClassId(Math.max(0, Number(v) || 0))}
                />
                <span className="hint">训练集 data.yaml 里的类别索引。</span>
              </label>
            </div>

            <div className="row" style={{ marginTop: 12 }}>
              <label className="field checkbox">
                <Checkbox checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)}>
                  覆盖同名帧
                </Checkbox>
              </label>
              <span className="hint" style={{ fontSize: 11, alignSelf: 'center' }}>
                写入 <code>raw/</code>；若数据集已划分，请到标注编辑器重新划分训练/验证集。
              </span>
            </div>

            {error && <div className="error-box" style={{ marginTop: 12 }}>{error}</div>}
            {done && <div className="ok-box" style={{ marginTop: 12 }}>{done}</div>}

            <div className="row" style={{ marginTop: 16, justifyContent: 'flex-end' }}>
              <button className="ghost" onClick={() => setOpen(false)}>
                关闭
              </button>
              <button onClick={() => void submit()} disabled={busy || !shot}>
                {busy ? '写入中…' : '写入数据集'}
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="error-box">{error ?? '当前帧不可用'}</div>
            <div className="row" style={{ marginTop: 16, justifyContent: 'flex-end' }}>
              <button className="ghost" onClick={() => setOpen(false)}>
                关闭
              </button>
            </div>
          </>
        )}
      </Modal>
    </div>
  );
}
