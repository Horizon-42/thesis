import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import cesium from "vite-plugin-cesium";
import { cpSync, existsSync, rmSync, symlinkSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.dirname(fileURLToPath(import.meta.url));

function preserveRuntimePublicAssets() {
  return {
    name: "aeroviz-preserve-runtime-public-assets",
    apply: "build" as const,
    closeBundle() {
      const distDir = path.join(repoRoot, "dist");
      const publicModelsDir = path.join(repoRoot, "public", "models");
      const distModelsDir = path.join(distDir, "models");
      const publicDataDir = path.join(repoRoot, "public", "data");
      const distDataPath = path.join(distDir, "data");

      if (existsSync(publicModelsDir)) {
        rmSync(distModelsDir, { recursive: true, force: true });
        cpSync(publicModelsDir, distModelsDir, { recursive: true });
      }

      rmSync(distDataPath, { recursive: true, force: true });
      if (existsSync(publicDataDir)) {
        symlinkSync(path.relative(distDir, publicDataDir), distDataPath, "dir");
      }
    },
  };
}

// vite-plugin-cesium handles:
//   1. Copying Cesium's static assets (Workers, Assets, Widgets) to dist/
//   2. Setting CESIUM_BASE_URL so CesiumJS can find them at runtime
// You do NOT need to manually configure anything here for Cesium.
export default defineConfig({
  plugins: [
    react(),
    cesium(), // must come AFTER react()
    preserveRuntimePublicAssets(),
  ],

  build: {
    // public/data is several GB and Vite would otherwise duplicate it into dist/data.
    copyPublicDir: false,
  },

  server: {
    // Listen on every interface so the dev site is reachable from other
    // devices on the same network. The full-stack launcher can still override
    // this with --host when a narrower bind address is required.
    host: "0.0.0.0",
    watch: {
      // public/data is ~40k files (mostly local-terrain .f32 heightmap tiles), and
      // chokidar takes one inotify watch per file. Two dev servers at once exceeded
      // fs.inotify.max_user_watches (65536) and vite died on boot with ENOSPC.
      // Nothing under data/ is a build input — it's git-ignored generated output
      // fetched over HTTP at runtime — so watching it buys no HMR, only the crash.
      ignored: ["**/public/data/**"],
    },
  },

  // ── Test configuration (Vitest) ──────────────────────────────────────────
  // Vitest reads this block when you run `npm test`.
  test: {
    environment: "jsdom", // simulate a browser DOM for React component tests
    globals: true,        // makes describe/it/expect available without imports
    setupFiles: ["./src/test-setup.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text", "html"],
    },
  },
});
