// Demo over a Cloudflare quick tunnel: `start_demo_tunnel.sh` builds with
// VITE_AEROVIZ_BACKEND_URL=/api and serves dist/ with `vite preview` through this config.
// The tunnel exposes ONE origin, so the backend is reached same-origin: /api/* is proxied
// to the local backend with the prefix stripped (the browser never needs port 8765).
import { defineConfig, mergeConfig } from "vite";
import baseConfig from "./vite.config";

const backendPort = process.env.AEROVIZ_BACKEND_PORT || "8765";

export default mergeConfig(baseConfig, defineConfig({
  preview: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
    // quick-tunnel hostnames are random subdomains of trycloudflare.com
    allowedHosts: [".trycloudflare.com"],
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${backendPort}`,
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/api/, ""),
      },
    },
  },
}));
