import { useState } from 'react';
import { api, type CheckReport, type SplitReport } from '../api';

interface Props {
  /** Dataset folder name. */
  name: string;
  /** Disable "划分" while the dataset has no frames. */
  disabled?: boolean;
  /** Flush pending label edits before the server reads the dataset. */
  flush?: () => Promise<void>;
  /** Called after a successful split, once every frame was re-assigned. */
  onSplit?: (result: { split: SplitReport; check: CheckReport }) => void | Promise<void>;
}

/**
 * "校验与划分" card: validates the raw labels and generates
 * `images/{train,val}` + `data.yaml`. Keeps its own form/report state so pages
 * only have to refresh their frame list afterwards.
 */
export default function DatasetSplitPanel({ name, disabled, flush, onSplit }: Props) {
  const [valFrac, setValFrac] = useState(0.2);
  const [seed, setSeed] = useState(0);
  const [classLabel, setClassLabel] = useState('mouse');

  const [split, setSplit] = useState<SplitReport | null>(null);
  const [check, setCheck] = useState<CheckReport | null>(null);
  const [splitBusy, setSplitBusy] = useState(false);
  const [checkBusy, setCheckBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runSplit = async () => {
    setSplitBusy(true);
    setError(null);
    try {
      await flush?.();
      const res = await api.splitDataset(name, { valFrac, seed, className: classLabel });
      setSplit(res.split);
      setCheck(res.check);
      await onSplit?.(res);
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
      await flush?.();
      setCheck(await api.checkDataset(name));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setCheckBusy(false);
    }
  };

  return (
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
            onChange={(e) =>
              setValFrac(Math.min(0.9, Math.max(0.01, Number(e.target.value) || 0.2)))
            }
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
          <input value={classLabel} onChange={(e) => setClassLabel(e.target.value)} />
        </label>
      </div>
      <div className="row" style={{ marginTop: 10 }}>
        <button className="ghost" onClick={() => void runCheck()} disabled={checkBusy}>
          {checkBusy ? '校验中…' : '校验数据集'}
        </button>
        <button onClick={() => void runSplit()} disabled={splitBusy || disabled}>
          {splitBusy ? '划分中…' : '划分训练/验证集'}
        </button>
      </div>

      {error && (
        <div className="error-box" style={{ marginTop: 8 }}>
          {error}
        </div>
      )}
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
        <div style={{ marginTop: 4 }}>
          注意：重新划分会按随机种子重新分配，覆盖编辑器里手动「移入训练集/验证集」的结果。
        </div>
      </div>
    </div>
  );
}
