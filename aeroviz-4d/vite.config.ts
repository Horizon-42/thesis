import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import cesium from "vite-plugin-cesium";

// vite-plugin-cesium handles:
//   1. Copying Cesium's static assets (Workers, Assets, Widgets) to dist/
//   2. Setting CESIUM_BASE_URL so CesiumJS can find them at runtime
// You do NOT need to manually configure anything here for Cesium.
export default defineConfig({
  plugins: [
    react(),
    cesium(), // must come AFTER react()
  ],

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
