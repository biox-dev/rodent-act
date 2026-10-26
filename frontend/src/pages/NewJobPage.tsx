import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  api,
  type CreateJobRequest,
  type MediaFile,
  type ServerInfo,
} from '../api';

const VIDEO_EXT = /\.(mp4|avi|mov|mkv|webm)$/i;
const IMAGE_EXT = /\.(png|jpe?g)$/i;

function prettySize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

export default function NewJobPage() {
  const navigate = useNavigate();
  const [info, setInfo] = useState<ServerInfo | null>(null);
  const [files, setFiles] = useState<MediaFile[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [media, setMedia] = useState('');
  const [detector, setDetector] = useState<'blob' | 'yolo'>('blob');
  const [reference, setReference] = useState('');
  const [blobThreshold, setBlobThreshold] = useState(22);
  const [blobMinArea, setBlobMinArea] = useState(0.0015);
  const [maxSide, setMaxSide] = useState(640);
  const [minHits, setMinHits] = useState(5);
  const [maxAge, setMaxAge] = useState(30);
  const [maxFrames, setMaxFrames] = useState('');
  const [className, setClassName] = useState('mouse');
  const [scoreThr, setScoreThr] = useState(0.2);
  const [window, setWindowS] = useState(0.5);
  const [clf, setClf] = useState('');
  const [writeVideo, setWriteVideo] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [i, f] = await Promise.all([api.info(), api.listMedia()]);
        setInfo(i);
        setFiles(f);
        setWindowS(i.default_window_s);
        const firstVideo = f.find((x) => VIDEO_EXT.test(x.name));
        if (firstVideo) setMedia(firstVideo.name);
        const emptyRef = f.find((x) => IMAGE_EXT.test(x.name) && /ref|empty/i.test(x.name));
        if (emptyRef) setReference(emptyRef.name);
      } catch (e) {
        setLoadError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  const videos = useMemo(() => files.filter((f) => VIDEO_EXT.test(f.name)), [files]);
  const images = useMemo(() => files.filter((f) => IMAGE_EXT.test(f.name)), [files]);

  const submit = async () => {
    if (!media) {
      setError('请选择输入视频');
      return;
    }
    setSubmitting(true);
    setError(null);
    const body: CreateJobRequest = {
      media,
      detector,
      reference: reference || undefined,
      blobThreshold,
      blobMinArea,
      maxSide,
      minHits,
      maxAge,
      maxFrames: maxFrames ? Number(maxFrames) : undefined,
      className: detector === 'yolo' ? className : undefined,
      scoreThr: detector === 'yolo' ? scoreThr : undefined,
      window,
      clf: clf || undefined,
      writeVideo,
    };
    try {
      const job = await api.createJob(body);
      navigate(`/jobs/${job.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSubmitting(false);
    }
  };

  return (
    <div className="page">
      <div className="card">
        <h2>新建分析任务</h2>
        {loadError && <div className="error-box">无法连接后端: {loadError}</div>}
        <div className="row muted mono" style={{ marginBottom: 12 }}>
          {info && (
            <>
              <span>媒体目录: {info.media_dir}</span>
              <span>输出目录: {info.out_dir}</span>
              <span>ffmpeg: {info.ffmpeg ? '✓' : '✗'}</span>
              <span>v{info.version}</span>
            </>
          )}
        </div>

        <h3>
          1. 选择视频<span className="sub">{videos.length} 个可用文件</span>
        </h3>
        {videos.length === 0 && (
          <div className="empty">
            媒体目录中没有视频。把 .mp4 文件放进 <code>{info?.media_dir ?? 'media'}</code>{' '}
            后刷新页面。
          </div>
        )}
        <div className="media-grid">
          {videos.map((f) => (
            <div
              key={f.name}
              className={`media-card${media === f.name ? ' selected' : ''}`}
              onClick={() => setMedia(f.name)}
            >
              <div className="name">{f.name}</div>
              <div className="meta">{prettySize(f.size)}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3>2. 检测与追踪参数</h3>
        <div className="row">
          <label className="field">
            检测后端
            <select
              value={detector}
              onChange={(e) => setDetector(e.target.value as 'blob' | 'yolo')}
            >
              <option value="blob">blob（背景差分，无需模型）</option>
              <option value="yolo">yolo（ONNX 模型）</option>
            </select>
          </label>

          <label className="field" style={{ minWidth: 220 }}>
            空场参考图（强烈推荐）
            <select value={reference} onChange={(e) => setReference(e.target.value)}>
              <option value="">不使用（时间背景）</option>
              <option value="frame:0">视频第 0 帧</option>
              {images.map((f) => (
                <option key={f.name} value={f.name}>
                  {f.name}
                </option>
              ))}
            </select>
          </label>

          {detector === 'blob' ? (
            <>
              <label className="field">
                灰度阈值
                <input
                  type="number"
                  value={blobThreshold}
                  min={1}
                  max={120}
                  onChange={(e) => setBlobThreshold(Number(e.target.value))}
                />
              </label>
              <label className="field">
                最小面积比例
                <input
                  type="number"
                  step={0.0005}
                  value={blobMinArea}
                  onChange={(e) => setBlobMinArea(Number(e.target.value))}
                />
              </label>
            </>
          ) : (
            <>
              <label className="field">
                类别名
                <input value={className} onChange={(e) => setClassName(e.target.value)} />
              </label>
              <label className="field">
                置信度阈值
                <input
                  type="number"
                  step={0.05}
                  value={scoreThr}
                  onChange={(e) => setScoreThr(Number(e.target.value))}
                />
              </label>
            </>
          )}

          <label className="field">
            处理最长边
            <input
              type="number"
              value={maxSide}
              min={160}
              max={1920}
              step={20}
              onChange={(e) => setMaxSide(Number(e.target.value))}
            />
          </label>
          <label className="field">
            min_hits
            <input
              type="number"
              value={minHits}
              min={1}
              onChange={(e) => setMinHits(Number(e.target.value))}
            />
          </label>
          <label className="field">
            max_age
            <input
              type="number"
              value={maxAge}
              min={1}
              onChange={(e) => setMaxAge(Number(e.target.value))}
            />
          </label>
          <label className="field">
            最大帧数（留空=全部）
            <input
              type="number"
              placeholder="全部"
              value={maxFrames}
              onChange={(e) => setMaxFrames(e.target.value)}
            />
          </label>
        </div>
      </div>

      <div className="card">
        <h3>
          3. 窗口与分类
          <span className="sub">默认 0.5 s 窗口 / 50% 重叠，无模型时使用速度阈值基线</span>
        </h3>
        <div className="row">
          <label className="field">
            窗口长度 (s)
            <input
              type="number"
              step={0.1}
              value={window}
              min={0.1}
              onChange={(e) => setWindowS(Number(e.target.value))}
            />
          </label>
          <label className="field" style={{ minWidth: 240 }}>
            XGBoost 模型路径（可选）
            <input
              placeholder="例如 models/clf.json"
              value={clf}
              onChange={(e) => setClf(e.target.value)}
            />
          </label>
          <label className="field checkbox">
            <input
              type="checkbox"
              checked={writeVideo}
              onChange={(e) => setWriteVideo(e.target.checked)}
            />
            同时导出标注 mp4（额外 ffmpeg 编码）
          </label>
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}

      <div className="row">
        <button onClick={submit} disabled={submitting || !media}>
          {submitting ? (
            <>
              <span className="spinner" />
              提交中…
            </>
          ) : (
            '开始分析'
          )}
        </button>
        <span className="muted" style={{ fontSize: 12 }}>
          分析在后台线程运行，页面可实时查看标注帧与日志，完成后显示窗口时间轴。
        </span>
      </div>
    </div>
  );
}
