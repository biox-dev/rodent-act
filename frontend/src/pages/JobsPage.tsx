import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, type JobView } from '../api';

function fmtDuration(s: number | null): string {
  if (s === null) return '—';
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

function fmtTime(ms: number): string {
  return new Date(ms).toLocaleString();
}

export default function JobsPage() {
  const [jobs, setJobs] = useState<JobView[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const list = await api.listJobs();
        if (alive) {
          setJobs(list);
          setError(null);
        }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (alive) setLoading(false);
      }
    };
    void load();
    const timer = window.setInterval(load, 3000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  return (
    <div className="page">
      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ margin: 0 }}>任务列表</h2>
          <Link to="/" className="field">
            <button>+ 新建分析</button>
          </Link>
        </div>
        <div style={{ marginTop: 14 }}>
          {error && <div className="error-box">无法获取任务列表: {error}</div>}
          {!error && loading && <div className="empty">加载中…</div>}
          {!error && !loading && jobs.length === 0 && (
            <div className="empty">还没有任务，先创建一个分析。</div>
          )}
          {jobs.length > 0 && (
            <table>
              <thead>
                <tr>
                  <th>任务</th>
                  <th>视频</th>
                  <th>状态</th>
                  <th>进度</th>
                  <th>轨迹</th>
                  <th>窗口</th>
                  <th>不动比例</th>
                  <th>耗时</th>
                  <th>创建时间</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => {
                  const s = j.summary;
                  const immobileFrac =
                    s && s.n_windows > 0 ? s.immobile_windows / s.n_windows : null;
                  return (
                    <tr key={j.id}>
                      <td>
                        <Link to={`/jobs/${j.id}`} className="mono">
                          {j.id}
                        </Link>
                      </td>
                      <td>{j.input_name}</td>
                      <td>
                        <span className={`badge ${j.status}`}>{j.status}</span>
                      </td>
                      <td style={{ minWidth: 110 }}>
                        <div className="progress">
                          <div style={{ width: `${Math.round(j.progress * 100)}%` }} />
                        </div>
                        <span className="muted mono">{Math.round(j.progress * 100)}%</span>
                      </td>
                      <td>{s ? s.n_tracks : '—'}</td>
                      <td>{s ? s.n_windows : '—'}</td>
                      <td>
                        {immobileFrac !== null ? `${(immobileFrac * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td>{fmtDuration(j.duration_s)}</td>
                      <td className="muted">{fmtTime(j.created_ms)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
