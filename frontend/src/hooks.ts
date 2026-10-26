import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, type DetectionRecord, type JobView, type LogRecord } from './api';

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

/** Load tracking records + scored windows once a job is done. */
export function useAnalysis(jobId: string | undefined, ready: boolean) {
  const [records, setRecords] = useState<Awaited<ReturnType<typeof api.records>> | null>(null);
  const [windows, setWindows] = useState<Awaited<ReturnType<typeof api.windows>> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId || !ready) return;
    let cancelled = false;
    (async () => {
      try {
        const [r, w] = await Promise.all([api.records(jobId), api.windows(jobId)]);
        if (!cancelled) {
          setRecords(r);
          setWindows(w);
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

  return { records, windows, byFrame, error };
}
