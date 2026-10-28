import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, type DetectionRecord, type JobView, type LogRecord, type ScoredWindow } from './api';

export interface ProgressState {
  running: boolean;
  status: string;
  progress: number;
  message: string;
}

const TERMINAL = new Set(['done', 'failed', 'cancelled']);

export function isTerminal(status: string | undefined): boolean {
  return !!status && TERMINAL.has(status);
}

/**
 * Live job status + progress over SSE (`/api/jobs/{id}/events`).
 *
 * Falls back to a REST poll when `EventSource` errors (e.g. a proxy that
 * buffers streaming responses), so the page always reflects reality.
 */
export function useJobEvents(jobId: string | undefined, initial?: JobView | null) {
  const [job, setJob] = useState<JobView | null>(initial ?? null);
  const [progress, setProgress] = useState<ProgressState>({
    running: false,
    status: initial?.status ?? 'queued',
    progress: initial?.progress ?? 0,
    message: '',
  });
  const [liveLogs, setLiveLogs] = useState<LogRecord[]>([]);
  const finishedRef = useRef(false);

  const refresh = useCallback(async () => {
    if (!jobId) return null;
    const next = await api.getJob(jobId);
    setJob(next);
    setProgress((p) => ({
      ...p,
      status: next.status,
      progress: next.progress,
      running: !isTerminal(next.status),
    }));
    if (isTerminal(next.status)) finishedRef.current = true;
    return next;
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    finishedRef.current = false;
    setLiveLogs([]);
    void refresh();

    const es = new EventSource(api.eventsUrl(jobId));
    let pollTimer: number | undefined;

    es.addEventListener('status', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as { status: string };
      setProgress((p) => ({
        ...p,
        status: data.status,
        running: !isTerminal(data.status),
        progress: data.status === 'done' ? 1 : p.progress,
      }));
    });

    es.addEventListener('progress', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as {
        progress: number;
        message: string;
      };
      setProgress((p) => ({ ...p, ...data, running: true }));
    });

    es.addEventListener('log', (e) => {
      const rec = JSON.parse((e as MessageEvent).data) as LogRecord;
      setLiveLogs((prev) => (prev.length > 500 ? [...prev.slice(-400), rec] : [...prev, rec]));
    });

    const finish = () => {
      finishedRef.current = true;
      es.close();
      void refresh();
    };
    es.addEventListener('done', finish);

    es.onerror = () => {
      // EventSource retries by itself; poll as a safety net for terminal state.
      if (!pollTimer) {
        pollTimer = window.setInterval(() => {
          if (finishedRef.current) {
            window.clearInterval(pollTimer);
            es.close();
            return;
          }
          void refresh();
        }, 2000);
      }
    };

    return () => {
      es.close();
      if (pollTimer) window.clearInterval(pollTimer);
    };
  }, [jobId, refresh]);

  return { job, progress, liveLogs, refresh };
}

export interface FrameState {
  src: string | null;
  index: number;
  t: number;
  total: number;
  connected: boolean;
}

/**
 * Live annotated frames (one-way video replacement while the tracker runs).
 *
 * The stream keeps exactly one object URL alive; each new frame revokes the
 * previous one so memory stays flat over thousands of frames.
 */
export function useFrameStream(jobId: string | undefined, enabled: boolean): FrameState {
  const [frame, setFrame] = useState<FrameState>({
    src: null,
    index: 0,
    t: 0,
    total: 0,
    connected: false,
  });
  const urlRef = useRef<string | null>(null);

  useEffect(() => {
    if (!jobId || !enabled) return;
    const es = new EventSource(api.framesUrl(jobId));
    es.addEventListener('frame', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as {
        index: number;
        t: number;
        total: number;
        jpeg: string;
      };
      const bytes = Uint8Array.from(atob(data.jpeg), (c) => c.charCodeAt(0));
      const blob = new Blob([bytes], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      const previous = urlRef.current;
      urlRef.current = url;
      setFrame({
        src: url,
        index: data.index,
        t: data.t,
        total: data.total,
        connected: true,
      });
      if (previous) URL.revokeObjectURL(previous);
    });
    es.onerror = () => setFrame((f) => ({ ...f, connected: false }));
    es.onopen = () => setFrame((f) => ({ ...f, connected: true }));

    return () => {
      es.close();
      if (urlRef.current) {
        URL.revokeObjectURL(urlRef.current);
        urlRef.current = null;
      }
    };
  }, [jobId, enabled]);

  return frame;
}

export interface LiveWindows {
  featureNames: string[];
  windows: ScoredWindow[];
  /** True once the final report exists. */
  complete: boolean;
}

/**
 * Scored windows, live.
 *
 * Windows are scored while the video is still being tracked, so we fetch the
 * snapshot with `/api/jobs/{id}/windows` and then append the deltas pushed by
 * `/api/jobs/{id}/window-events` (SSE). A short REST poll covers proxies that
 * buffer streams, and it is the only mechanism needed once the job is done.
 */
export function useLiveWindows(jobId: string | undefined, active: boolean) {
  const [state, setState] = useState<LiveWindows>({
    featureNames: [],
    windows: [],
    complete: false,
  });

  // Keep a set of scored window keys so re-connects / polls never duplicate.
  const seen = useRef<Set<string>>(new Set());
  const key = (w: ScoredWindow) => `${w.track_id}:${w.start_frame}`;

  const merge = useCallback((incoming: ScoredWindow[]) => {
    if (incoming.length === 0) return;
    setState((prev) => {
      const next = prev.windows.slice();
      let added = false;
      for (const w of incoming) {
        const k = key(w);
        if (seen.current.has(k)) continue;
        seen.current.add(k);
        next.push(w);
        added = true;
      }
      return added ? { ...prev, windows: next } : prev;
    });
  }, []);

  const load = useCallback(async () => {
    if (!jobId) return;
    try {
      const res = await api.windows(jobId);
      setState((prev) => ({
        featureNames: res.feature_names.length ? res.feature_names : prev.featureNames,
        windows: res.windows,
        complete: res.complete,
      }));
      seen.current = new Set(res.windows.map(key));
    } catch {
      /* job just created; retry on the next tick */
    }
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    seen.current = new Set();
    setState({ featureNames: [], windows: [], complete: false });
    void load();

    const es = new EventSource(api.windowEventsUrl(jobId));
    es.addEventListener('windows', (e) => {
      const data = JSON.parse((e as MessageEvent).data) as {
        complete: boolean;
        windows: ScoredWindow[];
      };
      merge(data.windows);
      if (data.complete) setState((p) => ({ ...p, complete: true }));
    });
    es.addEventListener('done', () => {
      es.close();
      setState((p) => ({ ...p, complete: true }));
      void load();
    });
    es.onerror = () => {
      /* EventSource reconnects by itself */
    };

    // Safety net: also sync the snapshot while running, so a buffering proxy
    // still shows the growing table (just at poll latency).
    const timer = active
      ? window.setInterval(() => void load(), 2000)
      : undefined;

    return () => {
      es.close();
      if (timer) window.clearInterval(timer);
    };
  }, [jobId, active, load, merge]);

  // `reload` lets the caller re-fetch the snapshot after an action that changes
  // the scored windows (e.g. re-scoring with a new window/stride).
  return { ...state, reload: load };
}

/** Load tracking records (per-frame boxes) once the tracking pass is done. */
export function useAnalysis(jobId: string | undefined, ready: boolean) {
  const [records, setRecords] = useState<Awaited<ReturnType<typeof api.records>> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId || !ready) return;
    let cancelled = false;
    (async () => {
      try {
        const r = await api.records(jobId);
        if (!cancelled) {
          setRecords(r);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [jobId, ready]);

  /**
   * frame index -> records, built once per dataset so the video overlay can
   * look up boxes in O(1) during playback.
   */
  const byFrame = useMemo(() => {
    const map = new Map<number, DetectionRecord[]>();
    if (records) {
      for (const rec of records.records) {
        const list = map.get(rec.frame);
        if (list) list.push(rec);
        else map.set(rec.frame, [rec]);
      }
    }
    return map;
  }, [records]);

  return { records, byFrame, error };
}
