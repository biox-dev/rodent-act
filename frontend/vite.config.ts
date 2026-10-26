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
