import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api, type DatasetSummary, type MediaFile, type ServerInfo } from '../api';
import ClipRange, { fmtClock } from '../components/ClipRange';

const VIDEO_EXT = /\.(mp4|avi|mov|mkv|webm)$/i;

function stemOf(name: string): string {
  const raw = name
    .replace(/\.[^.]+$/, '')
    .replace(/[^A-Za-z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return raw || 'dataset';
}

function fmtTime(ms: number): string {
  if (!ms) return '—';
  return new Date(ms).toLocaleString();
}

export default function DatasetPage() {
  const navigate = useNavigate();
  const [info, setInfo] = useState<ServerInfo | null>(null);
  const [files, setFiles] = useState<MediaFile[]>([]);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [video, setVideo] = useState('');
  const [name, setName] = useState('');
  const [frames, setFrames] = useState(60);
  const [fps, setFps] = useState(1);
  const [maxSide, setMaxSide] = useState(960);
  const [quality, setQuality] = useState(92);
  const [overwrite, setOverwrite] = useState(false);

  // Source-time range selected in the preview. Extraction decodes only this
  // segment (`ffmpeg -ss/-t`), so no intermediate video is produced.
  const previewRef = useRef<HTMLVideoElement>(null);
  const [duration, setDuration] = useState(0);
  const [clipStart, setClipStart] = useState(0);
  const [clipEnd, setClipEnd] = useState(0);
  const [previewingClip, setPreviewingClip] = useState(false);

  const [autoLabel, setAutoLabel] = useState(true);
  const [threshold, setThreshold] = useState(45);
  const [minArea, setMinArea] = useState(0.0008);
  const [channels, setChannels] = useState(6);
  const [maxObjects, setMaxObjects] = useState(12);
  const [useRoi, setUseRoi] = useState(true);
  const [roi, setRoi] = useState({ x1: 0.1, y1: 0.1, x2: 0.98, y2: 0.95 });
  const [onlyEmpty, setOnlyEmpty] = useState(false);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const videos = useMemo(() => files.filter((f) => VIDEO_EXT.test(f.name)), [files]);

  const refreshDatasets = async () => {
    try {
      setDatasets(await api.listDatasets());
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    (async () => {
      try {
        const [i, f] = await Promise.all([api.info(), api.listMedia()]);
        setInfo(i);
        setFiles(f);
        const first = f.find((x) => VIDEO_EXT.test(x.name));
        if (first) {
          setVideo(first.name);
          setName(stemOf(first.name));
        }
        await refreshDatasets();
      } catch (e) {
        setLoadError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  const pickVideo = (v: string) => {
    setVideo(v);
    setName(stemOf(v));
  };

  // A new source file invalidates the previous range; the metadata handler
  // below resets it once the duration is known.
  useEffect(() => {
    setPreviewingClip(false);
    setDuration(0);
    setClipStart(0);
    setClipEnd(0);
  }, [video]);

  const seekPreview = (t: number) => {
    const v = previewRef.current;
    if (!v || !Number.isFinite(t)) return;
    setPreviewingClip(false);
    v.currentTime = Math.min(Math.max(t, 0), duration || t);
  };

  const playSelection = () => {
    const v = previewRef.current;
    if (!v || duration <= 0) return;
    v.currentTime = clipStart;
    setPreviewingClip(true);
    void v.play().catch(() => setPreviewingClip(false));
  };

  const selectWholeVideo = () => {
    setClipStart(0);
    setClipEnd(duration);
    seekPreview(0);
  };

  const setEdgeFromCurrent = (edge: 'start' | 'end') => {
    const v = previewRef.current;
    if (!v) return;
    const t = Math.min(Math.max(v.currentTime, 0), duration);
    if (edge === 'start') setClipStart(Math.min(t, clipEnd - 0.05));
    else setClipEnd(Math.max(t, clipStart + 0.05));
  };

  const wholeVideo = duration > 0 && clipStart <= 0.01 && clipEnd >= duration - 0.01;
  const clipOk = duration <= 0 || clipEnd - clipStart >= 0.05;

  const estimateFrames = useMemo(() => {
    const len = Math.max(0, clipEnd - clipStart);
    if (len <= 0) return 0;
    // Samples land on clip-relative frames 0, step, 2*step, ... so a half-open
    // clip of `len` seconds yields about ceil(len * sampling fps) frames.
    if (fps > 0) return Math.min(frames, Math.max(1, Math.ceil(len * fps)));
    return frames;
  }, [clipStart, clipEnd, fps, frames]);

  const submit = async () => {
    if (!video) {
      setError('请选择输入视频');
      return;
    }
    if (duration > 0 && clipEnd - clipStart < 0.05) {
      setError('抽取区间太短（至少 0.05 s）');
      return;
    }
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const res = await api.extractDataset({
        media: video,
        dataset: name || stemOf(video),
        frames,
        fps,
        maxSide,
        quality,
        overwrite,
        // Omit a full-range selection so the server keeps the container's
        // exact frame count and skips the extra ffmpeg seek options.
        startS: duration > 0 && clipStart > 0.01 ? clipStart : undefined,
        endS: duration > 0 && clipEnd < duration - 0.01 ? clipEnd : undefined,
        autolabel: autoLabel
          ? {
              threshold,
              minArea,
              maxObjects,
              channels,
              roi: useRoi ? [roi.x1, roi.y1, roi.x2, roi.y2] : undefined,
              onlyEmpty,
            }
          : undefined,
      });
      const auto = res.autolabel
        ? `，预标注 ${res.autolabel.boxes} 个框（${res.autolabel.images} 帧）`
        : '';
      setNotice(
        `已生成 ${res.extract.written} 帧${auto}。正在打开标注编辑器…`,
      );
      await refreshDatasets();
      navigate(`/datasets/${encodeURIComponent(res.dataset.name)}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  const removeDataset = async (name: string) => {
    if (!window.confirm(`删除数据集「${name}」及其所有图片/标注？此操作不可撤销。`)) return;
    try {
      await api.deleteDataset(name);
      await refreshDatasets();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="page">
      <div className="card">
        <h2>生成数据集（Rust 实现）</h2>
        <div className="row muted mono" style={{ marginBottom: 12 }}>
          {info && (
            <>
              <span>媒体目录: {info.media_dir}</span>
              <span>输出目录: {info.out_dir}</span>
              <span>ffmpeg: {info.ffmpeg ? '✓' : '✗'}</span>
            </>
          )}
        </div>
        <div className="hint" style={{ marginBottom: 12, fontSize: 12 }}>
          等价于 <code>make_dataset.sh</code> 的第 1-3 步：从视频抽帧 → 可选自动预标注。
          结果保存到 <code>{info?.out_dir ?? 'out'}/datasets/&lt;名称&gt;/raw/</code>
          （每张图一个同名 <code>.txt</code>，YOLO 格式）。抽帧后可在此页面直接拖框修正。
        </div>
        {loadError && <div className="error-box">无法连接后端: {loadError}</div>}
        {error && <div className="error-box" style={{ marginBottom: 12 }}>{error}</div>}
        {notice && <div className="ok-box" style={{ marginBottom: 12 }}>{notice}</div>}

        <h3>
          1. 源视频<span className="sub">{videos.length} 个可用文件</span>
        </h3>
        {videos.length === 0 ? (
          <div className="empty">
            媒体目录中没有视频。把 .mp4 放进 <code>{info?.media_dir ?? 'media'}</code> 后刷新。
          </div>
        ) : (
          <div className="row">
            {videos.map((f) => (
              <div
                key={f.name}
                className={`media-card${video === f.name ? ' selected' : ''}`}
                onClick={() => pickVideo(f.name)}
              >
                <div className="name">{f.name}</div>
                <div className="meta">{(f.size / 1024 ** 2).toFixed(1)} MB</div>
              </div>
            ))}
          </div>
        )}

        <h3 style={{ marginTop: 18 }}>
          2. 选择抽取区间
          <span className="sub">
            {duration > 0
              ? wholeVideo
                ? `整段视频 · ${fmtClock(duration)}`
                : `${fmtClock(clipStart)} – ${fmtClock(clipEnd)} · ${(
                    clipEnd - clipStart
                  ).toFixed(1)} s`
              : '只解码选中的片段'}
          </span>
        </h3>
        {video ? (
          <>
            <div className="clip-layout">
              <div className="clip-preview">
                <video
                  key={video}
                  ref={previewRef}
                  src={api.mediaUrl(video)}
                  muted
                  playsInline
                  controls
                  preload="metadata"
                  onLoadedMetadata={(e) => {
                    const d = Number.isFinite(e.currentTarget.duration)
                      ? e.currentTarget.duration
                      : 0;
                    setDuration(d);
                    setClipStart(0);
                    setClipEnd(d);
                  }}
                  onTimeUpdate={(e) => {
                    if (previewingClip && e.currentTarget.currentTime >= clipEnd - 0.02) {
                      e.currentTarget.pause();
                      setPreviewingClip(false);
                    }
                  }}
                />
              </div>
              <div className="clip-controls">
                <div className="row">
                  <label className="field">
                    起点 (s)
                    <input
                      type="number"
                      min={0}
                      max={clipEnd > 0.05 ? clipEnd - 0.05 : undefined}
                      step={0.1}
                      value={Number(clipStart.toFixed(2))}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (!Number.isFinite(v)) return;
                        setClipStart(Math.min(Math.max(v, 0), Math.max(clipEnd - 0.05, 0)));
                        seekPreview(v);
                      }}
                    />
                  </label>
                  <label className="field">
                    终点 (s)
                    <input
                      type="number"
                      min={duration > 0 ? Math.min(clipStart + 0.05, duration) : undefined}
                      max={duration || undefined}
                      step={0.1}
                      value={Number(clipEnd.toFixed(2))}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        if (!Number.isFinite(v)) return;
                        setClipEnd(Math.min(Math.max(v, clipStart + 0.05), duration || v));
                        seekPreview(v);
                      }}
                    />
                  </label>
                </div>
                <div className="row" style={{ marginTop: 10 }}>
                  <button className="ghost" onClick={playSelection} disabled={duration <= 0}>
                    ▶ 预览选区
                  </button>
                  <button className="ghost" onClick={selectWholeVideo} disabled={duration <= 0}>
                    整段视频
                  </button>
                  <button
                    className="ghost"
                    onClick={() => setEdgeFromCurrent('start')}
                    disabled={duration <= 0}
                  >
                    当前设为起点
                  </button>
                  <button
                    className="ghost"
                    onClick={() => setEdgeFromCurrent('end')}
                    disabled={duration <= 0}
                  >
                    当前设为终点
                  </button>
                </div>
                <div className="hint" style={{ marginTop: 10, fontSize: 11 }}>
                  抽帧只解码 <b>起点–终点</b> 这一段（ffmpeg 输入 seek + 限时），不会生成中间视频；
                  保存的文件名里仍是<b>原视频的绝对帧号</b>，方便和视频对照。
                </div>
              </div>
            </div>
            <ClipRange
              duration={duration}
              start={clipStart}
              end={clipEnd}
              onChange={(s, e) => {
                setClipStart(s);
                setClipEnd(e);
              }}
              onScrub={(t) => {
                const v = previewRef.current;
                if (v) v.currentTime = Math.min(Math.max(t, 0), duration || t);
              }}
            />
            <div className="hint" style={{ marginTop: 8, fontSize: 11 }}>
              预计抽取 <b>{estimateFrames}</b> 帧（帧数上限 {frames}，采样{' '}
              {fps > 0 ? `${fps} fps` : '自动（按帧数均匀取）'}）
            </div>
          </>
        ) : (
          <div className="empty">先在上方选择视频</div>
        )}

        <h3 style={{ marginTop: 18 }}>3. 抽帧参数</h3>
        <div className="row">
          <label className="field" style={{ minWidth: 220 }}>
            数据集名称
            <input value={name} onChange={(e) => setName(e.target.value)} />
            <span className="hint">
              目录名，只允许字母/数字/_/-。同名数据集会追加抽帧（默认不覆盖已有图片）。
            </span>
          </label>
          <label className="field">
            帧数上限
            <input
              type="number"
              min={1}
              value={frames}
              onChange={(e) => setFrames(Math.max(1, Number(e.target.value) || 1))}
            />
            <span className="hint">最多保存多少帧，脚本里是 60（正式数据建议 200-500）。</span>
          </label>
          <label className="field">
            采样帧率 (fps)
            <input
              type="number"
              min={0}
              step={0.5}
              value={fps}
              onChange={(e) => setFps(Math.max(0, Number(e.target.value) || 0))}
            />
            <span className="hint">
              每秒取几帧；<b>0</b> = 按帧数上限在整段视频上均匀取。
            </span>
          </label>
          <label className="field">
            最长边
            <input
              type="number"
              min={64}
              max={3840}
              value={maxSide}
              onChange={(e) => setMaxSide(Number(e.target.value) || 960)}
            />
            <span className="hint">保存的 JPEG 会缩放到该尺寸，训练更快。</span>
          </label>
          <label className="field">
            JPEG 质量
            <input
              type="number"
              min={1}
              max={100}
              value={quality}
              onChange={(e) => setQuality(Number(e.target.value) || 92)}
            />
          </label>
          <label className="field checkbox">
            <input
              type="checkbox"
              checked={overwrite}
              onChange={(e) => setOverwrite(e.target.checked)}
            />
            覆盖已有图片
          </label>
        </div>

        <h3 style={{ marginTop: 18 }}>
          4. 自动预标注<span className="sub">可选，生成后仍可手动修正</span>
        </h3>
        <label className="field checkbox" style={{ marginBottom: 10 }}>
          <input
            type="checkbox"
            checked={autoLabel}
            onChange={(e) => setAutoLabel(e.target.checked)}
          />
          抽帧后自动生成初始框（比全图更暗的连通域）
        </label>
        {autoLabel && (
          <>
            <div className="row">
              <label className="field">
                暗度阈值
                <input
                  type="number"
                  min={1}
                  max={120}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value) || 45)}
                />
                <span className="hint">
                  比全图中位亮度暗多少算目标。调高更保守，调低更灵敏。
                </span>
              </label>
              <label className="field">
                最小面积比例
                <input
                  type="number"
                  step={0.0002}
                  value={minArea}
                  onChange={(e) => setMinArea(Number(e.target.value) || 0.0008)}
                />
                <span className="hint">过滤反光/噪点，0.0008 ≈ 640×480 下约 245 像素。</span>
              </label>
              <label className="field">
                每帧最多几个通道
                <input
                  type="number"
                  min={0}
                  value={channels}
                  onChange={(e) => setChannels(Math.max(0, Number(e.target.value) || 0))}
                />
                <span className="hint">装置是 6 通道就填 6；0 = 不限制。</span>
              </label>
              <label className="field">
                候选上限
                <input
                  type="number"
                  min={1}
                  value={maxObjects}
                  onChange={(e) => setMaxObjects(Math.max(1, Number(e.target.value) || 12))}
                />
              </label>
              <label className="field checkbox">
                <input
                  type="checkbox"
                  checked={onlyEmpty}
                  onChange={(e) => setOnlyEmpty(e.target.checked)}
                />
                只标空帧（不覆盖已有框）
              </label>
            </div>
            <label className="field checkbox" style={{ marginTop: 10 }}>
              <input
                type="checkbox"
                checked={useRoi}
                onChange={(e) => setUseRoi(e.target.checked)}
              />
              限制 ROI（推荐，去掉灯管/线缆/手）
            </label>
            {useRoi && (
              <div className="row" style={{ marginTop: 8 }}>
                {(
                  [
                    ['x1', '左'],
                    ['y1', '上'],
                    ['x2', '右'],
                    ['y2', '下'],
                  ] as const
                ).map(([key, label]) => (
                  <label className="field" key={key}>
                    ROI {label} ({key})
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
          </>
        )}

        <div className="row" style={{ marginTop: 16 }}>
          <button onClick={submit} disabled={busy || !video || !clipOk}>
            {busy ? '抽帧中…（长视频需要解码整段）' : '开始生成并打开标注'}
          </button>
          <span className="hint" style={{ fontSize: 11 }}>
            抽帧后请务必人工检查：预标注只是起点，漏标/多标都很正常。
          </span>
        </div>
      </div>

      <div className="card">
        <h3>
          已有数据集<span className="sub">{datasets.length} 个</span>
        </h3>
        <div className="row" style={{ marginBottom: 10 }}>
          <button className="ghost" onClick={() => void refreshDatasets()}>
            刷新
          </button>
        </div>
        {datasets.length === 0 ? (
          <div className="empty">还没有数据集，先在上方生成一个。</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>名称</th>
                <th>图片</th>
                <th>标注框</th>
                <th>空标注帧</th>
                <th>划分</th>
                <th>更新时间</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {datasets.map((d) => (
                <tr key={d.name}>
                  <td>
                    <Link to={`/datasets/${encodeURIComponent(d.name)}`}>{d.name}</Link>
                  </td>
                  <td>{d.nImages}</td>
                  <td>{d.nBoxes}</td>
                  <td className={d.emptyImages > 0 ? 'muted' : ''}>{d.emptyImages}</td>
                  <td>
                    {d.hasSplit ? (
                      <span className="badge done">
                        train {d.train} / val {d.val}
                      </span>
                    ) : (
                      <span className="badge queued">未划分</span>
                    )}
                  </td>
                  <td className="muted mono">{fmtTime(d.modifiedMs)}</td>
                  <td>
                    <div className="row" style={{ gap: 8 }}>
                      <Link className="link-button" to={`/datasets/${encodeURIComponent(d.name)}`}>
                        编辑标注
                      </Link>
                      <button className="ghost danger-text" onClick={() => void removeDataset(d.name)}>
                        删除
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <div className="hint" style={{ marginTop: 10, fontSize: 11 }}>
          标注完成后在编辑器里点「划分训练/验证集」，会生成
          <code> images/labels/{'{train,val}'} </code> 和 <code>data.yaml</code>，
          可直接用于 <code>analytics/model_tools.py fine-tune</code> 或 Ultralytics 训练。
        </div>
      </div>
    </div>
  );
}
