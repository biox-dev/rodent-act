/** Types mirroring `src/web/state.rs` + `src/pipeline.rs` serde payloads. */

export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled';

export interface ServerInfo {
  version: string;
  media_dir: string;
  out_dir: string;
  ffmpeg: boolean;
  features: string[];
  default_window_s: number;
}

export interface MediaFile {
  name: string;
  size: number;
  modified_ms: number;
}

/** `GET /api/models` - ONNX files the server can load. */
export interface ModelFile {
  /** Path to send back as `CreateJobRequest.model`. */
  path: string;
  name: string;
  size: number;
}

export interface TrackView {
  track_id: number;
  n_windows: number;
  immobile_fraction: number;
}

export interface SummaryView {
  width: number;
  height: number;
  fps: number;
  frames: number;
  /** Duration of the analysed clip (not of the source file). */
  duration_s: number;
  /** Source-time offset of the first analysed frame. */
  start_s: number;
  /** Requested source-time end of the analysed clip, if any. */
  end_s: number | null;
  /** Duration of the source file, ignoring the clip. */
  source_duration_s: number;
  n_tracks: number;
  n_windows: number;
  immobile_windows: number;
  mobile_windows: number;
  mean_immobile_prob: number;
  mean_rii: number;
  per_track: TrackView[];
}

export interface JobView {
  id: string;
  input: string;
  input_name: string;
  /** Source-time offset of the first analysed frame (seconds). */
  start_s: number;
  /** Requested source-time end of the analysed clip (seconds). */
  end_s: number | null;
  status: JobStatus;
  error: string | null;
  created_ms: number;
  frames_done: number;
  total_frames: number;
  progress: number;
  duration_s: number | null;
  has_video: boolean;
  /** Window length currently used for scoring (seconds). */
  window_s: number;
  /** Window stride currently used for scoring (seconds). */
  stride_s: number;
  /** XGBoost model in use (`null` = speed-threshold baseline). */
  clf: string | null;
  summary: SummaryView | null;
}

/** `pipeline::DetectionRecord` */
export interface DetectionRecord {
  frame: number;
  track_id: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  measured: boolean;
  motion_energy: number;
}

/** `pipeline::TrackSummary` */
export interface TrackSummary {
  id: number;
  class_id: number;
  start_frame: number;
  end_frame: number;
  n_points: number;
  n_measured: number;
}

export interface RecordsResponse {
  width: number;
  height: number;
  fps: number;
  frames: number;
  /** Frame indices/times below are relative to this source offset. */
  start_s: number;
  end_s: number | null;
  source_duration_s: number;
  tracks: TrackSummary[];
  records: DetectionRecord[];
}

/** `pipeline::ScoredWindow` */
export interface ScoredWindow {
  track_id: number;
  start_frame: number;
  end_frame: number;
  start_t: number;
  end_t: number;
  immobile_prob: number;
  label: 'immobile' | 'mobile';
  rii: number;
  values: number[];
}

export interface WindowsResponse {
  feature_names: string[];
  /** `null` while the job is still running (live windows have no report yet). */
  mean_immobile_prob: number | null;
  /** `null` while the job is still running. */
  per_track_immobile_fraction: [number, number][] | null;
  windows: ScoredWindow[];
  /** False until the final report exists (or the job reached a terminal state). */
  complete: boolean;
  status: JobStatus;
  progress: number;
  frames_done: number;
  total_frames: number;
}

export interface LogRecord {
  ts_ms: number;
  level: string;
  target: string;
  message: string;
}

export interface LogsResponse {
  total: number;
  next: number;
  records: LogRecord[];
}

/** `POST /api/jobs/{id}/rescore` - re-score from the stored tracking result. */
export interface RescoreRequest {
  /** Window length in seconds (omit = keep the job's current value). */
  window?: number;
  /** Window stride in seconds (omit = keep the job's current value). */
  stride?: number;
  /** XGBoost model path; `''` falls back to the speed baseline. */
  clf?: string;
}

export interface RescoreResponse {
  feature_names: string[];
  n_windows: number;
  mean_immobile_prob: number;
  per_track_immobile_fraction: [number, number][];
  windows: ScoredWindow[];
  job: JobView;
}

export interface CreateJobRequest {
  media: string;
  /** Start of the analysed segment, in source seconds. */
  startS?: number;
  /** End of the analysed segment, in source seconds (default: EOF). */
  endS?: number;
  detector?: 'blob' | 'yolo';
  reference?: string;
  blobThreshold?: number;
  blobMinArea?: number;
  maxSide?: number;
  minHits?: number;
  maxAge?: number;
  maxFrames?: number;
  scoreThr?: number;
  className?: string;
  model?: string;
  window?: number;
  stride?: number;
  clf?: string;
  writeVideo?: boolean;
}

// --------------------------------------------------------------------------- //
// dataset builder (extract / autolabel / review / split)
// --------------------------------------------------------------------------- //

/** `GET /api/datasets` - one folder under `<out>/datasets`. */
export interface DatasetSummary {
  name: string;
  path: string;
  nImages: number;
  nBoxes: number;
  emptyImages: number;
  hasSplit: boolean;
  train: number;
  val: number;
  modifiedMs: number;
}

export interface DatasetFrame {
  /** Image file name (`TST_000012.jpg`). */
  name: string;
  /** Label file name (`TST_000012.txt`). */
  label: string;
  boxes: number;
  width: number;
  height: number;
}

export interface DatasetFramesResponse {
  name: string;
  dir: string;
  frames: DatasetFrame[];
}

/** One YOLO box, normalised to 0..1 (mirrors `dataset::LabelBox`). */
export interface LabelBox {
  classId: number;
  xc: number;
  yc: number;
  w: number;
  h: number;
}

export interface ExtractReport {
  outDir: string;
  written: number;
  skipped: number;
  width: number;
  height: number;
  fps: number;
  step: number;
  scanned: number;
}

export interface AutolabelReport {
  images: number;
  boxes: number;
  skipped: number;
  empty: number;
}

export interface AutolabelOptions {
  threshold?: number;
  minArea?: number;
  maxObjects?: number;
  /** Keep at most N boxes per frame (0 = unlimited). */
  channels?: number;
  /** `[x1, y1, x2, y2]`, normalised 0..1. */
  roi?: [number, number, number, number];
  classId?: number;
  onlyEmpty?: boolean;
}

export interface ExtractDatasetRequest {
  /** File name inside the media directory. */
  media: string;
  /** Dataset folder name (default: the video file stem). */
  dataset?: string;
  frames?: number;
  /** Sampling rate in frames per second (0 = spread `frames` evenly). */
  fps?: number;
  maxSide?: number;
  startS?: number;
  endS?: number;
  quality?: number;
  overwrite?: boolean;
  autolabel?: AutolabelOptions;
}

export interface ExtractDatasetResponse {
  dataset: DatasetSummary;
  extract: ExtractReport;
  autolabel: AutolabelReport | null;
}

export interface SplitReport {
  train: number;
  val: number;
  outDir: string;
  dataYaml: string;
}

export interface CheckReport {
  root: string;
  scope: 'split' | 'raw';
  images: number;
  labeled: number;
  boxes: number;
  emptyImages: number;
  problems: string[];
  warnings: string[];
  ok: boolean;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

const BASE = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let message = res.statusText;
    try {
      const body = await res.json();
      message = body?.error ?? message;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, message);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  info: () => request<ServerInfo>('/info'),
  listMedia: () => request<MediaFile[]>('/media'),
  listModels: () => request<ModelFile[]>('/models'),
  mediaUrl: (name: string) => `${BASE}/media/${encodeURIComponent(name)}`,
  listJobs: () => request<JobView[]>('/jobs'),
  getJob: (id: string) => request<JobView>(`/jobs/${id}`),
  createJob: (body: CreateJobRequest) =>
    request<JobView>('/jobs', { method: 'POST', body: JSON.stringify(body) }),
  cancelJob: (id: string) => request<JobView>(`/jobs/${id}/cancel`, { method: 'POST' }),
  /** Recompute the windows only - the video is not analysed again. */
  rescore: (id: string, body: RescoreRequest) =>
    request<RescoreResponse>(`/jobs/${id}/rescore`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  records: (id: string) => request<RecordsResponse>(`/jobs/${id}/records`),
  windows: (id: string) => request<WindowsResponse>(`/jobs/${id}/windows`),
  logs: (id: string, since = 0) => request<LogsResponse>(`/jobs/${id}/logs?since=${since}`),
  eventsUrl: (id: string) => `${BASE}/jobs/${id}/events`,
  windowEventsUrl: (id: string) => `${BASE}/jobs/${id}/window-events`,
  framesUrl: (id: string) => `${BASE}/jobs/${id}/frames`,
  frameUrl: (id: string) => `${BASE}/jobs/${id}/frame`,
  videoUrl: (id: string) => `${BASE}/jobs/${id}/video`,

  listDatasets: () => request<DatasetSummary[]>('/datasets'),
  extractDataset: (body: ExtractDatasetRequest) =>
    request<ExtractDatasetResponse>('/datasets/extract', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  datasetFrames: (name: string) =>
    request<DatasetFramesResponse>(`/datasets/${encodeURIComponent(name)}/frames`),
  datasetImageUrl: (name: string, file: string) =>
    `${BASE}/datasets/${encodeURIComponent(name)}/image/${encodeURIComponent(file)}`,
  datasetLabels: (name: string, file: string) =>
    request<{ boxes: LabelBox[] }>(
      `/datasets/${encodeURIComponent(name)}/labels/${encodeURIComponent(file)}`,
    ),
  saveDatasetLabels: (name: string, file: string, boxes: LabelBox[]) =>
    request<{ ok: boolean; boxes: number }>(
      `/datasets/${encodeURIComponent(name)}/labels/${encodeURIComponent(file)}`,
      { method: 'PUT', body: JSON.stringify({ boxes }) },
    ),
  autolabelDataset: (name: string, opts: AutolabelOptions) =>
    request<{ report: AutolabelReport; dataset: DatasetSummary }>(
      `/datasets/${encodeURIComponent(name)}/autolabel`,
      { method: 'POST', body: JSON.stringify(opts) },
    ),
  splitDataset: (
    name: string,
    opts: { valFrac?: number; seed?: number; className?: string },
  ) =>
    request<{ split: SplitReport; check: CheckReport }>(
      `/datasets/${encodeURIComponent(name)}/split`,
      { method: 'POST', body: JSON.stringify(opts) },
    ),
  checkDataset: (name: string) =>
    request<CheckReport>(`/datasets/${encodeURIComponent(name)}/check`, {
      method: 'POST',
    }),
  deleteDataset: (name: string) =>
    request<{ ok: boolean }>(`/datasets/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),
};
