import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { api, type DatasetFrame, type LabelBox } from '../api';
import AutoLabelPanel from '../components/AutoLabelPanel';
import BoxEditor from '../components/BoxEditor';
import DatasetSplitPanel from '../components/DatasetSplitPanel';
import LabelToolbar, { type SaveState } from '../components/LabelToolbar';

export default function DatasetEditorPage() {
  const { name = '' } = useParams();
  // `?frame=<image>` deep-links to one frame (used by the "加入训练集" dialog).
  const [params, setParams] = useSearchParams();

  const [frames, setFrames] = useState<DatasetFrame[]>([]);
  const [dir, setDir] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [index, setIndex] = useState(0);
  const [boxes, setBoxes] = useState<LabelBox[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const [counts, setCounts] = useState<Record<string, number>>({});

  const [onlyEmpty, setOnlyEmpty] = useState(false);
  const [filterText, setFilterText] = useState('');

  // Frames can be moved into train/val one by one; requires an existing split.
  const [hasSplit, setHasSplit] = useState(false);
  const [assignBusy, setAssignBusy] = useState(false);

  // Refs mirror the state used by the debounced save so callbacks are stable.
  const boxesRef = useRef<LabelBox[]>([]);
  const dirtyRef = useRef(false);
  const indexRef = useRef(0);
  const framesRef = useRef<DatasetFrame[]>([]);
  // Scroll container of the frame list, so the active row can be revealed.
  const listRef = useRef<HTMLDivElement>(null);

  const applyCounts = useCallback((next: Record<string, number>) => {
    setCounts(next);
  }, []);

  const loadFrames = useCallback(async () => {
    const r = await api.datasetFrames(name);
    framesRef.current = r.frames;
    setFrames(r.frames);
    setDir(r.dir);
    setHasSplit(r.hasSplit);
    const c: Record<string, number> = {};
    for (const f of r.frames) c[f.label] = f.boxes;
    applyCounts(c);
    return r;
  }, [name, applyCounts]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .datasetFrames(name)
      .then((r) => {
        if (cancelled) return;
        framesRef.current = r.frames;
        setFrames(r.frames);
        setHasSplit(r.hasSplit);
        setDir(r.dir);
        const c: Record<string, number> = {};
        for (const f of r.frames) c[f.label] = f.boxes;
        applyCounts(c);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [name, applyCounts]);

  const frame = frames[index];
  const frameLabel = frame?.label ?? '';
  const member = frame?.split ?? null;

  // Deep link: jump to `?frame=<image>` once the frame list is known, then drop
  // the parameter so a later refresh does not jump back.
  const wantedFrame = params.get('frame');
  const appliedFrame = useRef<string | null>(null);
  useEffect(() => {
    if (!wantedFrame || appliedFrame.current === wantedFrame || !frames.length) return;
    appliedFrame.current = wantedFrame;
    const i = frames.findIndex((f) => f.name === wantedFrame);
    if (i >= 0) {
      indexRef.current = i;
      setIndex(i);
      setSelected(null);
      setNotice(`已定位到 ${wantedFrame}`);
    } else {
      setNotice(`没有找到帧 ${wantedFrame}（可能已被删除或覆盖）`);
    }
    const next = new URLSearchParams(params);
    next.delete('frame');
    setParams(next, { replace: true });
  }, [wantedFrame, frames, params, setParams]);

  // Keep the active frame visible in the (scrollable) list.
  useEffect(() => {
    const box = listRef.current;
    const el = box?.querySelector<HTMLElement>('button.active');
    if (!box || !el) return;
    const top = el.offsetTop - box.offsetTop - box.clientHeight / 2 + el.clientHeight / 2;
    box.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
  }, [index, frames.length]);

  // Load the labels of the current frame.
  useEffect(() => {
    if (!frameLabel) {
      boxesRef.current = [];
      setBoxes([]);
      return;
    }
    let cancelled = false;
    api
      .datasetLabels(name, frameLabel)
      .then((r) => {
        if (cancelled) return;
        boxesRef.current = r.boxes;
        setBoxes(r.boxes);
        dirtyRef.current = false;
        setDirty(false);
        setSelected(null);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [name, frameLabel]);

  /** Save the current frame if it has unsaved changes. */
  const saveNow = useCallback(async () => {
    const f = framesRef.current[indexRef.current];
    if (!f || !dirtyRef.current) return;
    dirtyRef.current = false;
    setDirty(false);
    setSaving(true);
    try {
      await api.saveDatasetLabels(name, f.label, boxesRef.current);
      setSavedAt(Date.now());
      setCounts((prev) => ({ ...prev, [f.label]: boxesRef.current.length }));
    } catch (e) {
      dirtyRef.current = true;
      setDirty(true);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }, [name]);

  // Debounced auto-save while drawing/editing.
  useEffect(() => {
    if (!dirty) return;
    const t = window.setTimeout(() => void saveNow(), 700);
    return () => window.clearTimeout(t);
  }, [dirty, boxes, saveNow]);

  const applyBoxes = (next: LabelBox[]) => {
    boxesRef.current = next;
    setBoxes(next);
    dirtyRef.current = true;
    setDirty(true);
    const f = framesRef.current[indexRef.current];
    if (f) {
      setCounts((prev) => ({ ...prev, [f.label]: next.length }));
    }
  };

  const goto = async (i: number) => {
    const list = framesRef.current;
    if (!list.length) return;
    const next = Math.min(Math.max(i, 0), list.length - 1);
    if (next === indexRef.current) return;
    await saveNow();
    indexRef.current = next;
    setIndex(next);
    setSelected(null);
  };

  const visible = useMemo(
    () =>
      frames
        .map((f, i) => ({ f, i }))
        .filter(
          ({ f }) =>
            (!onlyEmpty || (counts[f.label] ?? 0) === 0) &&
            (!filterText || f.name.toLowerCase().includes(filterText.toLowerCase())),
        ),
    [frames, counts, onlyEmpty, filterText],
  );

  const hasPrev = visible.some((v) => v.i < index);
  const hasNext = visible.some((v) => v.i > index);

  const step = (d: number) => {
    const target =
      d > 0
        ? visible.find((v) => v.i > index)?.i
        : [...visible].reverse().find((v) => v.i < index)?.i;
    if (target != null) void goto(target);
  };

  const deleteSelected = () => {
    if (selected == null) return;
    applyBoxes(boxes.filter((_, i) => i !== selected));
    setSelected(null);
  };

  /** Move the current frame into `train`/`val`, or out of both (`none`). */
  const assignCurrent = async (target: 'train' | 'val' | 'none') => {
    const f = framesRef.current[indexRef.current];
    if (!f) return;
    setAssignBusy(true);
    setError(null);
    setNotice(null);
    try {
      // The export copies raw/<image> + raw/<label>, so flush edits first.
      await saveNow();
      const res = await api.assignDatasetFrames(name, { frames: [f.name], target });
      const next = framesRef.current.map((x) =>
        x.name === f.name ? { ...x, split: target === 'none' ? null : target } : x,
      );
      framesRef.current = next;
      setFrames(next);
      setHasSplit(res.dataset.hasSplit);
      setNotice(
        target === 'none'
          ? `已将 ${f.name} 移出训练/验证集`
          : `已将 ${f.name} 移入${
              target === 'train' ? '训练集' : '验证集'
            }（train ${res.dataset.train} / val ${res.dataset.val}）`,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setAssignBusy(false);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && ['INPUT', 'SELECT', 'TEXTAREA'].includes(el.tagName)) return;
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        step(1);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        step(-1);
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        deleteSelected();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const copyPrevious = async () => {
    if (indexRef.current <= 0) return;
    const prev = framesRef.current[indexRef.current - 1];
    if (!prev) return;
    try {
      const r = await api.datasetLabels(name, prev.label);
      applyBoxes(r.boxes);
      setNotice(`已复制上一帧的 ${r.boxes.length} 个框（自动保存前请检查）`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const reload = async () => {
    await saveNow();
    const r = await loadFrames();
    const clamped = Math.min(indexRef.current, Math.max(r.frames.length - 1, 0));
    indexRef.current = clamped;
    setIndex(clamped);
    const f = r.frames[clamped];
    if (f) {
      const l = await api.datasetLabels(name, f.label);
      boxesRef.current = l.boxes;
      setBoxes(l.boxes);
      dirtyRef.current = false;
      setDirty(false);
      setSelected(null);
      setCounts((prev) => ({ ...prev, [f.label]: l.boxes.length }));
    }
  };

  const totalBoxes = useMemo(
    () => Object.values(counts).reduce((a, b) => a + b, 0),
    [counts],
  );
  const emptyCount = useMemo(
    () => frames.filter((f) => (counts[f.label] ?? 0) === 0).length,
    [frames, counts],
  );

  // Mirrors the debounced auto-save for the toolbar indicator.
  const saveState: SaveState = saving ? 'saving' : dirty ? 'dirty' : savedAt ? 'saved' : 'idle';

  if (loading) {
    return (
      <div className="page">
        <div className="card">正在加载数据集「{name}」…</div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ margin: 0 }}>
            数据集：{name}
            <span className="sub">YOLO 标注编辑</span>
          </h2>
          <Link className="link-button" to="/datasets">
            ← 返回数据集列表
          </Link>
        </div>
        <div className="row muted mono" style={{ marginTop: 8 }}>
          <span>{dir}</span>
          <span>图片 {frames.length}</span>
          <span>标注框 {totalBoxes}</span>
          <span>空标注帧 {emptyCount}</span>
        </div>
        {error && <div className="error-box" style={{ marginTop: 10 }}>{error}</div>}
        {notice && <div className="ok-box" style={{ marginTop: 10 }}>{notice}</div>}
      </div>

      <div className="dataset-layout">
        <div className="card">
          <h3>
            帧列表<span className="sub">{visible.length} / {frames.length}</span>
          </h3>
          <label className="field checkbox" style={{ marginBottom: 8 }}>
            <input
              type="checkbox"
              checked={onlyEmpty}
              onChange={(e) => setOnlyEmpty(e.target.checked)}
            />
            只看空标注
          </label>
          <input
            className="frame-filter"
            placeholder="过滤文件名"
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
          />
          <div className="frame-list" ref={listRef}>
            {visible.map(({ f, i }) => {
              const n = counts[f.label] ?? f.boxes;
              return (
                <button
                  key={f.label}
                  className={i === index ? 'active' : ''}
                  onClick={() => void goto(i)}
                  title={
                    f.split === 'both'
                      ? '同时在训练集和验证集'
                      : f.split === 'train'
                        ? '训练集'
                        : f.split === 'val'
                          ? '验证集'
                          : '未加入训练/验证集'
                  }
                >
                  <span>{f.name}</span>
                  <span className="frame-tags">
                    {(f.split === 'train' || f.split === 'both') && (
                      <em className="split-chip train">T</em>
                    )}
                    {(f.split === 'val' || f.split === 'both') && (
                      <em className="split-chip val">V</em>
                    )}
                  </span>
                  <span className={`cnt${n === 0 ? ' zero' : ''}`}>{n}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="card">
          <LabelToolbar
            index={index}
            count={frames.length}
            hasPrev={hasPrev}
            hasNext={hasNext}
            onStep={step}
            frameName={frame?.name ?? null}
            hasSplit={hasSplit}
            member={member}
            boxCount={boxes.length}
            onClear={() => applyBoxes([])}
            onAssign={(target) => void assignCurrent(target)}
            assignBusy={assignBusy}
            canCopyPrevious={index > 0}
            onCopyPrevious={() => void copyPrevious()}
            saveState={saveState}
          />

          {!hasSplit && (
            <div className="hint" style={{ marginTop: 6, fontSize: 11 }}>
              还没有训练/验证划分：先点右侧「划分训练/验证集」，之后才能把单帧移入 train/val。
            </div>
          )}

          {frame ? (
            <BoxEditor
              key={frame.name}
              src={api.datasetImageUrl(name, frame.name)}
              width={frame.width}
              height={frame.height}
              boxes={boxes}
              selected={selected}
              onSelect={setSelected}
              onChange={applyBoxes}
            />
          ) : (
            <div className="empty">这个数据集还没有图片</div>
          )}

          <div className="hint" style={{ marginTop: 8, fontSize: 11 }}>
            左键拖拽画新框 · 拖动框移动 · 拖四角缩放 · 右键删除 ·
            <b> ←/→</b> 切换帧 · <b>Delete</b> 删除选中 · 修改会自动保存。
            「移入训练集/验证集」会把当前帧复制到 <code>images/train</code> 或{' '}
            <code>images/val</code>（并移出另一侧）；重新「划分训练/验证集」会重置这些手动分配。
          </div>
          {selected != null && boxes[selected] && (
            <div className="mono muted" style={{ marginTop: 6 }}>
              选中 #{selected + 1} · class {boxes[selected].classId} · center (
              {boxes[selected].xc.toFixed(3)}, {boxes[selected].yc.toFixed(3)}) · size (
              {boxes[selected].w.toFixed(3)} × {boxes[selected].h.toFixed(3)})
            </div>
          )}
        </div>

        <div className="dataset-side">
          <AutoLabelPanel
            disabled={!frames.length}
            disabledHint="数据集里还没有图片"
            onRun={async (opts) => {
              const res = await api.autolabelDataset(name, opts);
              await reload();
              return (
                `预标注完成：${res.report.images} 帧 / ${res.report.boxes} 个框` +
                (res.report.empty ? `（${res.report.empty} 帧没有候选）` : '')
              );
            }}
          />

          <DatasetSplitPanel
            name={name}
            disabled={!frames.length}
            flush={saveNow}
            onSplit={async () => {
              // The split re-assigns every frame, so refresh the per-frame
              // train/val badges (and enable the manual assign buttons).
              await loadFrames();
            }}
          />
        </div>
      </div>
    </div>
  );
}
