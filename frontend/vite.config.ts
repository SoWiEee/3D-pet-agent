import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8000", ws: true, changeOrigin: true },
      "/pet": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/perception": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/semantic": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/scene": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/command": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/exploration": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/planning": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/control": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/healthz": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
