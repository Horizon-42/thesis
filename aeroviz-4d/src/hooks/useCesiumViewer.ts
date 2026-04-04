/**
 * useCesiumViewer.ts
 * ------------------
 * Custom hook: initialise a CesiumJS Viewer inside a given DOM container.
 *
 * Why a hook instead of doing this in the component?
 *   Separation of concerns.  The component owns the DOM ref; this hook owns
 *   the Cesium Viewer lifecycle (create on mount, destroy on unmount).
 *   This also makes the initialisation logic unit-testable in isolation.
 *
 * 📖 Tutorial: see docs/01-cesium-viewer.md
 */

import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";

// ─────────────────────────────────────────────────────────────────────────────
// CONFIGURATION CONSTANTS
// Replace the placeholder with your own token from https://cesium.com/ion/tokens
// A free "Community" tier token is sufficient for this project.
// ─────────────────────────────────────────────────────────────────────────────
const CESIUM_ION_TOKEN = import.meta.env.VITE_CESIUM_ION_TOKEN

/** WGS84 coordinates for the initial camera target (default: Kelowna CYLW) */
export const DEFAULT_AIRPORT = {
  lon: -119.3775,
  lat: 49.9561,
  /** Initial camera altitude in metres */
  height: 15_000,
} as const;

/**
 * Initialise a Cesium Viewer mounted inside `containerRef`.
 * Stores the resulting Viewer in AppContext so other hooks can use it.
 *
 * @param containerRef - a React ref pointing to the `<div>` that Cesium owns
 */
export function useCesiumViewer(
  containerRef: React.RefObject<HTMLDivElement>
): void {
  // We store the Viewer in a local ref (not state) to avoid React re-renders
  // when the Viewer object changes internally.
  const viewerRef = useRef<Cesium.Viewer | null>(null);
  const { setViewer } = useApp();

  useEffect(() => {
    // Guard: only run once, and only after the DOM node exists.
    if (!containerRef.current || viewerRef.current) return;

    // ── Step 1: Set the Ion access token ─────────────────────────────────────
    Cesium.Ion.defaultAccessToken = CESIUM_ION_TOKEN;

    // ── Step 2: Create the Viewer ─────────────────────────────────────────────
    // ① — Create `new Cesium.Viewer(...)` with the following settings:
    //   • terrain:             Cesium.Terrain.fromWorldTerrain({ requestVertexNormals: true, requestWaterMask: true })
    //   • baseLayerPicker:     false   (hide the base layer picker button)
    //   • geocoder:            false   (hide the search bar)
    //   • homeButton:          false
    //   • sceneModePicker:     false
    //   • navigationHelpButton: false
    //   • animation:           true    (keep the animation widget — needed for 4D playback)
    //   • timeline:            true    (keep the timeline bar)
    //   • skyAtmosphere:       new Cesium.SkyAtmosphere()
    //
    // Hint: `requestVertexNormals: true` tells Cesium to download slope data
    // alongside elevation so the terrain shader can compute light/shadow.
    // Without it, mountains look flat and grey.
    //
    // Reference: docs/01-cesium-viewer.md § "Viewer options"
    const viewer = new Cesium.Viewer("cesiumContainer", {
      terrain: Cesium.Terrain.fromWorldTerrain({ requestVertexNormals: true, requestWaterMask: true }),
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      animation: true,
      timeline: true,
      skyAtmosphere: new Cesium.SkyAtmosphere()
    });
    // const viewer = null as unknown as Cesium.Viewer; // ← replace this line

    viewerRef.current = viewer;

    // ── Step 3: Enable terrain lighting ───────────────────────────────────────
    // ② — Enable the globe's built-in directional lighting:
    viewer.scene.globe.enableLighting = true;
    //
    // Hint: set it to `true`.  The sun's position is computed from the
    // simulation clock time, so you'll see day/night and mountain shadows.

    // ── Step 4: Set initial camera view ───────────────────────────────────────
    // ③ — Fly the camera to DEFAULT_AIRPORT using viewer.camera.setView().
    //
    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(DEFAULT_AIRPORT.lon, DEFAULT_AIRPORT.lat, DEFAULT_AIRPORT.height),
      orientation: {
        heading: Cesium.Math.toRadians(0),   // compass bearing (0 = north)
        pitch:   Cesium.Math.toRadians(-45),   // tilt angle (negative = look down)
        roll:    0,                          // bank angle (0 = level)
      },
    });
    //
    // Hint: a pitch of -45° gives a nice oblique view; -90° is straight down.
    // Reference: docs/01-cesium-viewer.md § "Camera orientation"

    // Share the Viewer with the rest of the app via context.
    setViewer(viewer);

    // ── Cleanup: destroy the Viewer when the component unmounts ───────────────
    // This releases WebGL resources and prevents memory leaks.
    return () => {
      viewerRef.current?.destroy();
      viewerRef.current = null;
    };
  }, [containerRef, setViewer]);
}
