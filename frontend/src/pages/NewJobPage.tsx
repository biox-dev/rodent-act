import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  api,
  type CreateJobRequest,
  type MediaFile,
  type ModelFile,
  type ServerInfo,
} from '../api';
import ClipRange, { fmtClock } from '../components/ClipRange';

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
  const [models, setModels] = useState<ModelFile[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [media, setMedia] = useState('');
  const [detector, setDetector] = useState<'blob' | 'yolo'>('yolo');
  const [model, setModel] = useState('');
  const [reference, setReference] = useState('');
  const [blobThreshold, setBlobThreshold] = useState(22);
  const [blobMinArea, setBlobMinArea] = useState(0.0015);
  const [maxSide, setMaxSide] = useState(640);
  const [minHits, setMinHits] = useState(5);
  const [maxAge, setMaxAge] = useState(30);
  const [maxFrames, setMaxFrames] = useState('');
  const [className, setClassName] = useState('any');
  const [scoreThr, setScoreThr] = useState(0.2);
  const [window, setWindowS] = useState(0.5);
  const [stride, setStride] = useState('');
  const [threshold, setThreshold] = useState(0.5);
  const [clf, setClf] = useState('');
  const [writeVideo, setWriteVideo] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Visual clip editor. Analysis decodes only [clipStart, clipEnd) of the
  // source file (no intermediate clipped video is produced).
  const previewRef = useRef<HTMLVideoElement>(null);
  const [duration, setDuration] = useState(0);
  const [clipStart, setClipStart] = useState(0);
  const [clipEnd, setClipEnd] = useState(0);
  const [previewingClip, setPreviewingClip] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const [i, f, m] = await Promise.all([
          api.info(),
          api.listMedia(),
          api.listModels(),
        ]);
        setInfo(i);
        setFiles(f);
        setModels(m);
        setWindowS(i.default_window_s);
        const firstVideo = f.find((x) => VIDEO_EXT.test(x.name));
        if (firstVideo) setMedia(firstVideo.name);
        const emptyRef = f.find((x) => IMAGE_EXT.test(x.name) && /ref|empty/i.test(x.name));
        if (emptyRef) setReference(emptyRef.name);
        // Prefer the fine-tuned mouse detector over any stock COCO model.
        const preferred =
          m.find((x) => x.name === 'mouse.onnx') ??
          m.find((x) => /mouse|rodent|tst/i.test(x.name)) ??
          m[0];
        if (preferred) setModel(preferred.path);
      } catch (e) {
        setLoadError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  const videos = useMemo(() => files.filter((f) => VIDEO_EXT.test(f.name)), [files]);
  const images = useMemo(() => files.filter((f) => IMAGE_EXT.test(f.name)), [files]);

  // A new source file invalidates the previous selection; the metadata handler
  // below resets the range once the duration is known.
  useEffect(() => {
    setPreviewingClip(false);
    setDuration(0);
    setClipStart(0);
    setClipEnd(0);
  }, [media]);

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

  const setEdgeFromCurrent = (edge: 'start' | 'end') => {
    const v = previewRef.current;
    if (!v) return;
    const t = Math.min(Math.max(v.currentTime, 0), duration);
    if (edge === 'start') setClipStart(Math.min(t, clipEnd - 0.05));
    else setClipEnd(Math.max(t, clipStart + 0.05));
  };

  const wholeVideo = duration > 0 && clipStart <= 0.01 && clipEnd >= duration - 0.01;

  const submit = async () => {
    if (!media) {
      setError('请选择输入视频');
      return;
    }
    setSubmitting(true);
    setError(null);
    const body: CreateJobRequest = {
      media,
      // Omit a full-range selection so the server keeps the container's exact
      // frame count (and skips the extra ffmpeg seek options).
      startS: duration > 0 && clipStart > 0.01 ? clipStart : undefined,
      endS: duration > 0 && clipEnd < duration - 0.01 ? clipEnd : undefined,
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
      model: detector === 'yolo' ? model || undefined : undefined,
      window,
      stride: stride ? Number(stride) : undefined,
      threshold,
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
        <h3>
          2. 选择分析区间
          <span className="sub">
            拖动下方手柄或直接输入时间，只分析选中的片段
            {wholeVideo ? ' · 当前为整段视频' : ` · ${fmtClock(clipStart)} – ${fmtClock(clipEnd)}`}
          </span>
        </h3>
        {media ? (
          <>
            <div className="clip-layout">
              <div className="clip-preview">
                <video
                  key={media}
                  ref={previewRef}
                  src={api.mediaUrl(media)}
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
                      max={Math.max(clipEnd - 0.05, 0)}
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
                      min={Math.min(clipStart + 0.05, duration)}
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
                  <button
                    className="ghost"
                    onClick={() => {
                      setClipStart(0);
                      setClipEnd(duration);
                      seekPreview(0);
                    }}
                    disabled={duration <= 0}
                  >
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
                  分析直接从该区间解码（ffmpeg 输入 seek + 限时），<b>不会</b>生成中间视频文件，
                  因此没有二次转码的画质损失和时间开销。所有窗口时间、帧号都相对区间起点从 0 开始。
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
          </>
        ) : (
          <div className="empty">先在上方选择一个视频</div>
        )}
      </div>

      <div className="card">
        <h3>3. 检测与追踪参数</h3>
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
            <span className="hint">
              决定每帧用什么方法找出动物。<b>blob</b> 用运动/背景差分，速度快但要求地面干净；
              <b>yolo</b> 用神经网络，适合杂物多或已微调的场景。
            </span>
          </label>

          <label className="field" style={{ minWidth: 220 }}>
            空场参考图（强烈推荐）
            <select value={reference} onChange={(e) => setReference(e.target.value)}>
              <option value="">不使用（时间背景）</option>
              <option value="frame:0">分析区间第 0 帧</option>
              {images.map((f) => (
                <option key={f.name} value={f.name}>
                  {f.name}
                </option>
              ))}
            </select>
            <span className="hint">
              箱体<b>没有动物时</b>的一张底图，用来做背景相减。有了它，动物从第 1 帧就在场也能被检出；
              否则只能靠运动触发，静止不动的动物会漏检。
            </span>
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
                <span className="hint">
                  与参考图的灰度差超过该值的像素算作“前景”。调<b>低</b>更灵敏（含噪点），
                  调<b>高</b>更保守（可能漏掉对比度低的小鼠）。
                </span>
              </label>
              <label className="field">
                最小面积比例
                <input
                  type="number"
                  step={0.0005}
                  value={blobMinArea}
                  onChange={(e) => setBlobMinArea(Number(e.target.value))}
                />
                <span className="hint">
                  连通域面积占整帧的比例下限，用于过滤反光/阴影等小噪点。
                  0.0015 ≈ 640×480 下约 460 像素。
                </span>
              </label>
            </>
          ) : (
            <>
              <label className="field" style={{ minWidth: 220 }}>
                检测模型
                <select value={model} onChange={(e) => setModel(e.target.value)}>
                  {models.length === 0 && (
                    <option value="">models/ 下没有 .onnx 文件</option>
                  )}
                  {models.map((m) => (
                    <option key={m.path} value={m.path}>
                      {m.name}
                      {m.name === 'mouse.onnx' ? '（微调·推荐）' : ''}
                    </option>
                  ))}
                </select>
                <span className="hint">
                  要加载的 ONNX 权重。<b>mouse.onnx</b> 已在 TST 数据上微调，效果远好于
                  通用 COCO 模型（yolo11n.onnx）。
                </span>
              </label>
              <label className="field">
                类别名（any = 全部）
                <input value={className} onChange={(e) => setClassName(e.target.value)} />
                <span className="hint">
                  微调模型只有一个类，保留检测框即可，填 <b>any</b>；留空等价于不过滤。
                  需要严格过滤时（如 COCO 模型只看动物），填类别名或 ID。
                </span>
              </label>
              <label className="field">
                置信度阈值
                <input
                  type="number"
                  step={0.05}
                  value={scoreThr}
                  onChange={(e) => setScoreThr(Number(e.target.value))}
                />
                <span className="hint">
                  低于此分数的检测框直接丢弃。调低召回更高但误检更多；
                  小鼠小、距离远时可适当降到 0.1–0.15。
                </span>
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
            <span className="hint">
              帧先等比缩放到最长边为该值再检测。越小越快，但小目标更易丢失；
              640 是速度与精度的常见折中。
            </span>
          </label>
          <label className="field">
            min_hits
            <input
              type="number"
              value={minHits}
              min={1}
              onChange={(e) => setMinHits(Number(e.target.value))}
            />
            <span className="hint">
              一条轨迹被<b>确认</b>前需连续命中该帧数。调大更抗误检，但动物刚出现的前几帧不计入。
            </span>
          </label>
          <label className="field">
            max_age
            <input
              type="number"
              value={maxAge}
              min={1}
              onChange={(e) => setMaxAge(Number(e.target.value))}
            />
            <span className="hint">
              轨迹连续丢失检测后仍保留（靠卡尔曼预测）的最大帧数。调大可跨过遮挡，
              调小则更快结束消失的个体。
            </span>
          </label>
          <label className="field">
            最大帧数（留空=全部）
            <input
              type="number"
              placeholder="全部"
              value={maxFrames}
              onChange={(e) => setMaxFrames(e.target.value)}
            />
            <span className="hint">
              只处理视频前 N 帧，便于快速试参数；留空分析整段视频。
            </span>
          </label>
        </div>
      </div>

      <div className="card">
        <h3>
          4. 窗口与分类
          <span className="sub">
            默认 0.5 s 窗口 / 50% 重叠（stride = 窗口一半），无模型时使用速度阈值基线
          </span>
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
            <span className="hint">
              把轨迹切成固定时长的小段，每段单独判定不动/活动。TST 标准通常取 0.5 s。
              窗口长度与步长都可在任务完成后调整重算，无需重新分析视频。
            </span>
          </label>
          <label className="field">
            步长 stride (s)
            <input
              type="number"
              step={0.05}
              min={0.05}
              placeholder={`默认 ${(window / 2).toFixed(2)}（窗口一半）`}
              value={stride}
              onChange={(e) => setStride(e.target.value)}
            />
            <span className="hint">
              相邻窗口起点的时间间隔。留空 = 窗口的一半（50% 重叠）；
              填成与窗口相同（如 0.5 / 0.5）即<b>不重叠</b>，窗口数减半、结果更“整段化”。
            </span>
          </label>
          <label className="field">
            判定阈值 p(immobile)
            <input
              type="number"
              step={0.05}
              min={0.05}
              max={0.95}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
            <span className="hint">
              窗口的“不动”概率高于该值即判为不动，默认 0.5。调高更严格（少判不动），
              调低更宽松。任务完成后也能在任务页改此值重算，无需重新分析视频。
            </span>
          </label>
          <label className="field" style={{ minWidth: 240 }}>
            XGBoost 模型路径（可选）
            <input
              placeholder="例如 models/clf.json"
              value={clf}
              onChange={(e) => setClf(e.target.value)}
            />
            <span className="hint">
              每个窗口提取速度、路径、停留比例等特征后交给该模型输出“不动”概率。
              留空则用<b>速度阈值基线</b>：窗口平均速度 &lt; 0.30 体长/秒即判为不动。
            </span>
          </label>
          <label className="field checkbox" style={{ alignSelf: 'center' }}>
            <input
              type="checkbox"
              checked={writeVideo}
              onChange={(e) => setWriteVideo(e.target.checked)}
            />
            同时导出标注 mp4（额外 ffmpeg 编码）
          </label>
        </div>
        <div className="hint" style={{ marginTop: 10, fontSize: 11 }}>
          说明：页面上的实时标注帧不受此开关影响；勾选后会额外把整段带框视频编码成 mp4 保存到任务输出目录。
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
