import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { api, type LogRecord, type ScoredWindow } from '../api';
import SeriesChart from '../components/SeriesChart';
import Timeline from '../components/Timeline';
import { fmtClock } from '../components/ClipRange';
import { TrackingOverlay } from '../components/TrackingOverlay';
import { isTerminal, useAnalysis, useFrameStream, useJobEvents, useLiveWindows } from '../hooks';

function fmtTime(ms: number): string {
  return new Date(ms).toLocaleTimeString();
}

function pct(v: number | undefined | null): string {
  if (v === undefined || v === null || Number.isNaN(v)) return '—';
  return `${(v * 100).toFixed(1)}%`;
}

export default function JobPage() {
  const { id } = useParams<{ id: string }>();
  const { job, progress, liveLogs, refresh } = useJobEvents(id);
  const running = !!job && !isTerminal(job.status);
  const frames = useFrameStream(id, running);

  const done = job?.status === 'done';
  const live = useLiveWindows(id, running);
  // Per-frame boxes only exist after the tracking pass finishes.
  const analysis = useAnalysis(id, !!job && !running);
  const windows = live.windows;
  const hasWindows = windows.length > 0;

  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [selectedWindow, setSelectedWindow] = useState<ScoredWindow | null>(null);
  const [speedFeature, setSpeedFeature] = useState('speed_mean');
  // One track filter for both charts, so hiding a track in the probability
  // curve also hides it in the feature curve. Empty set = show everything.
  const [hiddenTracks, setHiddenTracks] = useState<ReadonlySet<number>>(() => new Set());
  const toggleTrack = useCallback((trackId: number) => {
    setHiddenTracks((prev) => {
      const next = new Set(prev);
      if (next.has(trackId)) next.delete(trackId);
      else next.add(trackId);
      return next;
    });
  }, []);
  const showAllTracks = useCallback(() => setHiddenTracks(new Set()), []);

  // Frames whose boxes we draw over the <video>: index = round(t * fps).
  // Tracks only start after `min_hits` detections, so the first frames are
  // empty; hold the newest known boxes (or the first ones) so the overlay
  // never flickers off at the very start of the video.
  //
  // Everything the pipeline reports (frame indices, window times) is relative
  // to the analysed clip, while this player shows the original file - so all
  // lookups subtract `clipStart`.
  const clipStart = job?.start_s ?? analysis.records?.start_s ?? 0;
  const clipEnd = job?.end_s ?? analysis.records?.end_s ?? null;
  const clipDuration = job?.summary?.duration_s ?? 0;
  const clipTime = currentTime - clipStart;
  const inClip =
    clipTime >= -0.05 && (clipEnd === null || currentTime <= clipEnd + 0.05);
  const fps = job?.summary?.fps ?? analysis.records?.fps ?? 25;
  const frameIndex = Math.max(0, Math.round(clipTime * fps));
  const frameRecords = useMemo(() => {
    if (!inClip) return undefined;
    const exact = analysis.byFrame.get(frameIndex);
    if (exact) return exact;
    for (let back = 1; back <= Math.round(fps); back++) {
      const prev = analysis.byFrame.get(frameIndex - back);
      if (prev) return prev;
    }
    const keys = [...analysis.byFrame.keys()].sort((a, b) => a - b);
    return keys.length > 0 ? analysis.byFrame.get(keys[0]) : undefined;
  }, [analysis.byFrame, frameIndex, fps, inClip]);

  const seek = useCallback(
    (t: number) => {
      const v = videoRef.current;
      if (!v) return;
      v.currentTime = Math.min(Math.max(clipStart + t, 0), v.duration || t);
      void v.play();
    },
    [clipStart],
  );

  useEffect(() => {
    if (!done) return;
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => window.clearInterval(timer);
  }, [done, refresh]);

  const speedIndex = useMemo(() => {
    const i = live.featureNames.indexOf(speedFeature);
    return i >= 0 ? i : 0;
  }, [live.featureNames, speedFeature]);

  const logs: LogRecord[] = useMemo(() => {
    // The SSE stream replays the buffered log at connect; on reconnect the
    // duplicates are harmless but ugly, so only collapse consecutive repeats.
    const out: LogRecord[] = [];
    for (const rec of liveLogs) {
      const prev = out[out.length - 1];
      if (prev && prev.ts_ms === rec.ts_ms && prev.message === rec.message) continue;
      out.push(rec);
    }
    return out;
  }, [liveLogs]);

  if (!id) return null;

  return (
    <div className="page">
      <div className="card">
        
        <pre style={{ whiteSpace: 'pre', fontFamily: 'monospace', overflow: 'auto' }}>
          {JSON.stringify(frameRecords, null, 2)}
        </pre>

        <div className="row" style={{ justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h2 style={{ marginBottom: 4 }}>
              {job?.input_name ?? id}
              {job && (
                <span className={`badge ${job.status}`} style={{ marginLeft: 10, verticalAlign: 2 }}>
                  {job.status}
                </span>
              )}
            </h2>
            <div className="muted mono">
              job {id}
              {job?.duration_s != null && ` · 耗时 ${job.duration_s.toFixed(1)}s`}
            </div>
          </div>
          <div className="row">
            {job?.has_video && (
              <a href={api.videoUrl(id)} download="annotated.mp4">
                <button className="ghost">下载标注视频</button>
              </a>
            )}
            {running && (
              <button className="danger" onClick={() => void api.cancelJob(id).then(refresh)}>
                取消
              </button>
            )}
            <Link to="/jobs">
              <button className="ghost">返回列表</button>
            </Link>
          </div>
        </div>

        {running && (
          <div style={{ marginTop: 14 }}>
            <div className="progress">
              <div style={{ width: `${Math.round(progress.progress * 100)}%` }} />
            </div>
            <div className="muted mono" style={{ marginTop: 6 }}>
              {progress.status} · {Math.round(progress.progress * 100)}%
              {job && job.total_frames > 0 && ` · ${job.frames_done}/${job.total_frames} 帧`}
              {progress.message && ` · ${progress.message}`}
            </div>
          </div>
        )}

        {job?.error && <div className="error-box" style={{ marginTop: 12 }}>{job.error}</div>}
      </div>

      {job?.summary && (
        <div className="stats">
          <div className="stat">
            <div className="label">轨迹数</div>
            <div className="value">{job.summary.n_tracks}</div>
          </div>
          <div className="stat">
            <div className="label">窗口数</div>
            <div className="value">{job.summary.n_windows}</div>
          </div>
          <div className="stat">
            <div className="label">不动窗口</div>
            <div className="value">
              {job.summary.immobile_windows}
              <span className="muted" style={{ fontSize: 12 }}>
                {' '}
                / {job.summary.n_windows}
              </span>
            </div>
          </div>
          <div className="stat">
            <div className="label">不动比例</div>
            <div className="value">
              {pct(
                job.summary.n_windows > 0
                  ? job.summary.immobile_windows / job.summary.n_windows
                  : 0,
              )}
            </div>
          </div>
          <div className="stat">
            <div className="label">平均 p(immobile)</div>
            <div className="value">{job.summary.mean_immobile_prob.toFixed(3)}</div>
          </div>
          <div className="stat">
            <div className="label">平均 RII</div>
            <div className="value">{job.summary.mean_rii.toFixed(3)}</div>
          </div>
          <div className="stat">
            <div className="label">视频时长</div>
            <div className="value">{job.summary.duration_s.toFixed(1)}s</div>
            {(job.summary.start_s > 0.01 || job.summary.end_s != null) && (
              <div className="muted" style={{ fontSize: 10 }}>
                原 {job.summary.source_duration_s.toFixed(1)}s
              </div>
            )}
          </div>
        </div>
      )}

      <div>
        <div className="card">
          <h3>
            视频
            <span className="sub">
              {running
                ? '实时标注流（分析进行中）'
                : clipStart > 0.01 || clipEnd !== null
                  ? `原始视频 + 检测框叠加 · 分析区间 ${fmtClock(clipStart)}–${fmtClock(
                    clipEnd ?? job?.summary?.source_duration_s ?? 0,
                  )}`
                  : '原始视频 + 检测框叠加'}
            </span>
          </h3>
          <div className="video-stage">

            {running ? (
              frames.src ? (
                <img src={frames.src} alt="live frame" />
              ) : (
                <div className="empty">
                  <span className="spinner" />
                  等待首帧…
                  {!frames.connected && <div className="muted">帧流未连接</div>}
                </div>
              )
            ) : (
              <>

                <video
                  ref={videoRef}
                  src={api.mediaUrl(job?.input_name ?? '')}
                  controls
                  preload="auto"
                  onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                  onSeeked={(e) => setCurrentTime(e.currentTarget.currentTime)}
                />
                <TrackingOverlay
                  records={frameRecords}
                  videoWidth={analysis.records?.width ?? job?.summary?.width ?? 0}
                  videoHeight={analysis.records?.height ?? job?.summary?.height ?? 0}
                  visible={running === false && !!analysis.records && inClip}
                />
              </>
            )}
            {running && frames.src && (
              <div className="frame-badge">
                frame {frames.index}/{frames.total || '?'} · t={frames.t.toFixed(2)}s
              </div>
            )}
          </div>

          {!running && (
            <div className="transport">
              <span className="mono">
                {currentTime.toFixed(2)}s · frame {frameIndex}
                {!inClip && ' · 不在分析区间'}
              </span>
              {frameRecords && frameRecords.length > 0 ? (
                <span className="mono">
                  {frameRecords
                    .map((r) => `ID${r.track_id} p=${r.motion_energy.toFixed(2)}`)
                    .join('  ')}
                </span>
              ) : (
                <span className="muted">该帧无检测</span>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="split">
        <div className="card">
          <h3>
            轨迹特征曲线
            <span className="sub">
              按窗口绘制任一特征，用于核对分类依据；悬停显示时刻，拖动/点击定位视频
              {running && !live.complete ? ' · 实时增长' : ''}
            </span>
          </h3>
          {hasWindows ? (
            <>
              <div className="row" style={{ marginBottom: 10 }}>
                <label className="field">
                  特征
                  <select value={speedFeature} onChange={(e) => setSpeedFeature(e.target.value)}>
                    {live.featureNames.map((n) => (
                      <option key={n} value={n}>
                        {n}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <SeriesChart
                mode="speed"
                featureIndex={speedIndex}
                windows={windows}
                duration={clipDuration}
                current={running ? undefined : Math.max(0, clipTime)}
                hiddenTracks={hiddenTracks}
                onToggleTrack={toggleTrack}
                onShowAllTracks={showAllTracks}
                onSeek={seek}
              />
            </>
          ) : (
            <div className="empty">
              {running ? '分析进行中，窗口完成后即时绘制' : '暂无窗口数据'}
            </div>
          )}

          <h3 style={{ marginTop: 22 }}>
            不动概率曲线
            <span className="sub">
              每窗口 p(immobile)，虚线=0.5 判定阈值；悬停显示时刻，拖动/点击定位视频
              {running && !live.complete ? ' · 实时增长' : ''}
            </span>
          </h3>
          {hasWindows ? (
            <SeriesChart
              mode="prob"
              windows={windows}
              duration={clipDuration}
              current={running ? undefined : Math.max(0, clipTime)}
              hiddenTracks={hiddenTracks}
              onToggleTrack={toggleTrack}
              onShowAllTracks={showAllTracks}
              onSeek={seek}
            />
          ) : (
            <div className="empty">
              {running ? (
                <>
                  <span className="spinner" /> 分析进行中，窗口完成后即时绘制
                </>
              ) : (
                '暂无窗口数据'
              )}
            </div>
          )}

          <h3 style={{ marginTop: 22 }}>
            窗口时间轴
            <span className="sub">
              悬停显示时刻，拖动/点击定位视频
              {running && hasWindows && ` · 实时累计 ${windows.length} 个窗口`}
            </span>
          </h3>
          {hasWindows ? (
            <>
              <Timeline
                windows={windows}
                duration={clipDuration}
                current={Math.max(0, Math.min(clipTime, clipDuration || clipTime))}
                onSeek={seek}
              />
              {selectedWindow && (
                <div className="muted mono" style={{ marginTop: 10 }}>
                  track {selectedWindow.track_id} · {selectedWindow.start_t.toFixed(2)}–
                  {selectedWindow.end_t.toFixed(2)}s · {selectedWindow.label} · p=
                  {selectedWindow.immobile_prob.toFixed(3)}
                </div>
              )}
            </>
          ) : (
            <div className="empty">
              {running ? (
                <>
                  <span className="spinner" /> 等待首个窗口 ({Math.round(progress.progress * 100)}%)
                </>
              ) : (
                '暂无窗口数据'
              )}
            </div>
          )}

          {job?.summary && job.summary.per_track.length > 0 && (
            <>
              <h3 style={{ marginTop: 20 }}>
                每条轨迹的不动比例
                <span className="sub">
                  共 {job.summary.n_tracks} 条，均值 {pct(job.summary.mean_immobile_prob)}
                </span>
              </h3>
              <table>
                <thead>
                  <tr>
                    <th>Track</th>
                    <th>窗口数</th>
                    <th>不动比例</th>
                    <th style={{ width: '45%' }} />
                  </tr>
                </thead>
                <tbody>
                  {job.summary.per_track.map((t) => (
                    <tr key={t.track_id}>
                      <td className="mono">ID {t.track_id}</td>
                      <td>{t.n_windows}</td>
                      <td>{pct(t.immobile_fraction)}</td>
                      <td>
                        <div className="progress">
                          <div style={{ width: `${Math.round(t.immobile_fraction * 100)}%` }} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>

        <div className="card">
          <h3>
            运行日志
            <span className="sub">来自 pipeline 的 tracing 事件（SSE 实时推送）</span>
          </h3>
          <div className="logs">
            {logs.length === 0 && <div className="muted">暂无日志</div>}
            {logs.map((rec, i) => (
              <div key={`${rec.ts_ms}-${i}`} className={`lvl-${rec.level}`}>
                <span className="ts">{fmtTime(rec.ts_ms)}</span>
                {rec.message}
              </div>
            ))}
          </div>

          <h3 style={{ marginTop: 20 }}>
            窗口明细
            <span className="sub">
              {hasWindows
                ? `${windows.length} 行${running ? ' · 实时追加' : ''}`
                : running
                  ? '分析进行中…'
                  : '—'}
            </span>
          </h3>
          <div className="feature-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>track</th>
                  <th>start</th>
                  <th>end</th>
                  <th>p(imm)</th>
                  <th>RII</th>
                  <th>label</th>
                </tr>
              </thead>
              <tbody>
                {/* Newest first while running, so the growing tail stays visible. */}
                {(running ? windows.slice(-200).reverse() : windows.slice(0, 500)).map((w) => (
                  <tr
                    key={`${w.track_id}-${w.start_frame}`}
                    onClick={() => {
                      setSelectedWindow(w);
                      seek((w.start_t + w.end_t) / 2);
                    }}
                    style={{ cursor: 'pointer' }}
                  >
                    <td className="mono">{w.track_id}</td>
                    <td className="num">{w.start_t.toFixed(2)}</td>
                    <td className="num">{w.end_t.toFixed(2)}</td>
                    <td className="num">{w.immobile_prob.toFixed(3)}</td>
                    <td className="num">{w.rii.toFixed(3)}</td>
                    <td>
                      <span
                        className="badge"
                        style={{
                          background:
                            w.label === 'immobile'
                              ? 'rgba(41,211,152,.15)'
                              : 'rgba(74,168,255,.15)',
                          color: w.label === 'immobile' ? 'var(--immobile)' : 'var(--mobile)',
                        }}
                      >
                        {w.label}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {analysis.error && <div className="error-box" style={{ marginTop: 10 }}>{analysis.error}</div>}
        </div>
      </div>
    </div>
  );
}
