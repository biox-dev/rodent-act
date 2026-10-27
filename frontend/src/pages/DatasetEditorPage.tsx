import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  api,
  type CheckReport,
  type DatasetFrame,
  type LabelBox,
  type SplitReport,
} from '../api';
import BoxEditor from '../components/BoxEditor';

export default function DatasetEditorPage() {
  const { name = '' } = useParams();

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

  const [autoThreshold, setAutoThreshold] = useState(45);
  const [autoMinArea, setAutoMinArea] = useState(0.0008);
  const [autoMaxObjects, setAutoMaxObjects] = useState(12);
  const [autoChannels, setAutoChannels] = useState(6);
  const [autoOnlyEmpty, setAutoOnlyEmpty] = useState(false);
  const [useAutoRoi, setUseAutoRoi] = useState(true);
  const [roi, setRoi] = useState({ x1: 0.1, y1: 0.1, x2: 0.98, y2: 0.95 });
  const [autoBusy, setAutoBusy] = useState(false);

  const [valFrac, setValFrac] = useState(0.2);
  const [seed, setSeed] = useState(0);
  const [className, setClassName] = useState('mouse');
  const [splitBusy, setSplitBusy] = useState(false);
  const [split, setSplit] = useState<SplitReport | null>(null);
  const [check, setCheck] = useState<CheckReport | null>(null);
  const [checkBusy, setCheckBusy] = useState(false);

  // Refs mirror the state used by the debounced save so callbacks are stable.
  const boxesRef = useRef<LabelBox[]>([]);
  const dirtyRef = useRef(false);
  const indexRef = useRef(0);
  const framesRef = useRef<DatasetFrame[]>([]);

  const applyCounts = useCallback((next: Record<string, number>) => {
    setCounts(next);
  }, []);

  const loadFrames = useCallback(async () => {
    const r = await api.datasetFrames(name);
    framesRef.current = r.frames;
    setFrames(r.frames);
    setDir(r.dir);
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

  const runAutolabel = async () => {
    setAutoBusy(true);
    setError(null);
    setNotice(null);
    try {
      const res = await api.autolabelDataset(name, {
        threshold: autoThreshold,
        minArea: autoMinArea,
        maxObjects: autoMaxObjects,
        channels: autoChannels,
        roi: useAutoRoi ? [roi.x1, roi.y1, roi.x2, roi.y2] : undefined,
        onlyEmpty: autoOnlyEmpty,
      });
      setNotice(
        `预标注完成：${res.report.images} 帧 / ${res.report.boxes} 个框` +
          (res.report.empty ? `（${res.report.empty} 帧没有候选）` : ''),
      );
      await reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setAutoBusy(false);
    }
  };

  const runSplit = async () => {
    setSplitBusy(true);
    setError(null);
    setNotice(null);
    try {
      await saveNow();
      const res = await api.splitDataset(name, { valFrac, seed, className });
      setSplit(res.split);
      setCheck(res.check);
      setNotice(`已生成训练集：train=${res.split.train} val=${res.split.val}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSplitBusy(false);
    }
  };

  const runCheck = async () => {
    setCheckBusy(true);
    setError(null);
    try {
      await saveNow();
      const r = await api.checkDataset(name);
      setCheck(r);
      setNotice(
        r.ok
          ? `校验通过：${r.images} 张图 / ${r.boxes} 个框`
          : `发现 ${r.problems.length} 个问题`,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setCheckBusy(false);
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
          <div className="frame-list">
            {visible.map(({ f, i }) => {
              const n = counts[f.label] ?? f.boxes;
              return (
                <button
                  key={f.label}
                  className={i === index ? 'active' : ''}
                  onClick={() => void goto(i)}
                >
                  <span>{f.name}</span>
                  <span className={`cnt${n === 0 ? ' zero' : ''}`}>{n}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="card">
          <div className="editor-toolbar">
            <button className="ghost" onClick={() => step(-1)} disabled={!hasPrev}>
              ← 上一帧
            </button>
            <button className="ghost" onClick={() => step(1)} disabled={!hasNext}>
              下一帧 →
            </button>
            <span className="mono muted">
              {frames.length ? index + 1 : 0} / {frames.length}
            </span>
            <span className="spacer" />
            <button className="ghost" onClick={() => void copyPrevious()} disabled={index === 0}>
              复制上一帧
            </button>
            <button className="ghost" onClick={() => applyBoxes([])} disabled={!boxes.length}>
              清空
            </button>
            <span className="save-chip">
              {saving ? '保存中…' : dirty ? '未保存…' : savedAt ? '已保存 ✓' : ''}
            </span>
          </div>

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
            <b> ←/→</b> 切换帧 · <b>Delete</b> 删除选中 · 修改会自动保存
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
          <div className="card">
            <h3>
              自动预标注<span className="sub">覆盖当前 .txt</span>
            </h3>
            <div className="row">
              <label className="field">
                暗度阈值
                <input
                  type="number"
                  min={1}
                  max={120}
                  value={autoThreshold}
                  onChange={(e) => setAutoThreshold(Number(e.target.value) || 45)}
                />
              </label>
              <label className="field">
                最小面积
                <input
                  type="number"
                  step={0.0002}
                  value={autoMinArea}
                  onChange={(e) => setAutoMinArea(Number(e.target.value) || 0.0008)}
                />
              </label>
              <label className="field">
                通道数
                <input
                  type="number"
                  min={0}
                  value={autoChannels}
                  onChange={(e) => setAutoChannels(Math.max(0, Number(e.target.value) || 0))}
                />
              </label>
              <label className="field">
                候选上限
                <input
                  type="number"
                  min={1}
                  value={autoMaxObjects}
                  onChange={(e) => setAutoMaxObjects(Math.max(1, Number(e.target.value) || 12))}
                />
              </label>
            </div>
            <label className="field checkbox" style={{ marginTop: 8 }}>
              <input
                type="checkbox"
                checked={autoOnlyEmpty}
                onChange={(e) => setAutoOnlyEmpty(e.target.checked)}
              />
              只标空帧（保留手工标注）
            </label>
            <label className="field checkbox">
              <input
                type="checkbox"
                checked={useAutoRoi}
                onChange={(e) => setUseAutoRoi(e.target.checked)}
              />
              使用 ROI
            </label>
            {useAutoRoi && (
              <div className="row">
                {(
                  [
                    ['x1', '左'],
                    ['y1', '上'],
                    ['x2', '右'],
                    ['y2', '下'],
                  ] as const
                ).map(([key, label]) => (
                  <label className="field" key={key}>
                    {label}
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      value={roi[key]}
                      onChange={(e) =>
                        setRoi((r) => ({ ...r, [key]: Number(e.target.value) || 0 }))
                      }
                    />
                  </label>
                ))}
              </div>
            )}
            <div className="row" style={{ marginTop: 10 }}>
              <button onClick={() => void runAutolabel()} disabled={autoBusy || !frames.length}>
                {autoBusy ? '标注中…' : '开始预标注'}
              </button>
            </div>
            <div className="hint" style={{ marginTop: 8, fontSize: 11 }}>
              预标注会写回每张图的 <code>.txt</code>；已手工修正的帧请勾选「只标空帧」。
            </div>
          </div>

          <div className="card">
            <h3>
              校验与划分<span className="sub">生成 data.yaml</span>
            </h3>
            <div className="row">
              <label className="field">
                验证集比例
                <input
                  type="number"
                  min={0.01}
                  max={0.9}
                  step={0.05}
                  value={valFrac}
                  onChange={(e) => setValFrac(Math.min(0.9, Math.max(0.01, Number(e.target.value) || 0.2)))}
                />
              </label>
              <label className="field">
                随机种子
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))}
                />
              </label>
              <label className="field">
                类别名
                <input value={className} onChange={(e) => setClassName(e.target.value)} />
              </label>
            </div>
            <div className="row" style={{ marginTop: 10 }}>
              <button className="ghost" onClick={() => void runCheck()} disabled={checkBusy}>
                {checkBusy ? '校验中…' : '校验数据集'}
              </button>
              <button onClick={() => void runSplit()} disabled={splitBusy || !frames.length}>
                {splitBusy ? '划分中…' : '划分训练/验证集'}
              </button>
            </div>
            {split && (
              <div className="ok-box" style={{ marginTop: 10 }}>
                train={split.train} · val={split.val}
                <div className="mono" style={{ marginTop: 4, wordBreak: 'break-all' }}>
                  {split.dataYaml}
                </div>
              </div>
            )}
            {check && (
              <div
                className={check.ok ? 'ok-box' : 'error-box'}
                style={{ marginTop: 10, fontSize: 12 }}
              >
                scope={check.scope} · images={check.images} · labels={check.labeled} · boxes=
                {check.boxes} · empty={check.emptyImages}
                {check.problems.slice(0, 8).map((p) => (
                  <div key={p} className="mono">
                    {p}
                  </div>
                ))}
                {check.problems.length > 8 && (
                  <div className="mono">… 还有 {check.problems.length - 8} 个问题</div>
                )}
                {check.warnings.map((w) => (
                  <div key={w} className="mono" style={{ color: 'var(--warn)' }}>
                    ⚠ {w}
                  </div>
                ))}
              </div>
            )}
            <div className="hint" style={{ marginTop: 8, fontSize: 11 }}>
              划分会先清空 <code>images/</code> 与 <code>labels/</code>，再按
              <code> raw/</code> 重新拷贝。生成后可直接训练：
              <div className="mono" style={{ marginTop: 4, wordBreak: 'break-all' }}>
                python analytics/model_tools.py fine-tune --data dataset/data.yaml
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
