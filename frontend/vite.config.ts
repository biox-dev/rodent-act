import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The Rust server (cargo run -- serve) owns /api on :8080. In dev, Vite runs on
// :5173 and proxies API + SSE calls to it, so the UI hot-reloads without CORS
// surprises.
const backend = process.env.RODENT_API ?? 'http://127.0.0.1:8080';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    // The container's inotify limits are tiny (fs.inotify.max_user_watches=65536,
    // max_user_instances=128) and node_modules alone is ~2.4k files, so a real
    // watcher swarm blows ENOSPC. Polling avoids inotify watches entirely; the
    // ignores keep the poller's scan cheap and skip build artifacts.
    watch: {
      usePolling: true,
      interval: 300,
      ignored: [
        '**/node_modules/**',
        '**/.git/**',
        '**/dist/**',
        '**/target/**',
        '**/runs/**',
        '**/dataset/**',
        '**/media/**',
        '**/models/**',
        '**/*.tsbuildinfo',
        '**/*.onnx',
        '**/*.mp4',
        '**/*.avi',
        '**/*.mov',
        '**/*.mkv',
        '**/*.webm',
      ],
    },
    proxy: {
      '/api': {
        target: backend,
        changeOrigin: true,
        // SSE endpoints must not be buffered by the proxy
        ws: false,
      },
    },
  },
});
