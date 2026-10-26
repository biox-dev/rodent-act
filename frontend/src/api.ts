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
  duration_s: number;
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
  status: JobStatus;
  error: string | null;
  created_ms: number;
  frames_done: number;
  total_frames: number;
  progress: number;
  duration_s: number | null;
  has_video: boolean;
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

export interface CreateJobRequest {
  media: string;
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
  records: (id: string) => request<RecordsResponse>(`/jobs/${id}/records`),
  windows: (id: string) => request<WindowsResponse>(`/jobs/${id}/windows`),
  logs: (id: string, since = 0) => request<LogsResponse>(`/jobs/${id}/logs?since=${since}`),
  eventsUrl: (id: string) => `${BASE}/jobs/${id}/events`,
  windowEventsUrl: (id: string) => `${BASE}/jobs/${id}/window-events`,
  framesUrl: (id: string) => `${BASE}/jobs/${id}/frames`,
  frameUrl: (id: string) => `${BASE}/jobs/${id}/frame`,
  videoUrl: (id: string) => `${BASE}/jobs/${id}/video`,
};
