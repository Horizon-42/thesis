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
const CESIUM_ION_TOKEN = import.meta.env.VITE_CESIUM_ION_TOKEN;

/** WGS84 coordinates for the initial camera target (default: Calgary CYYC) */
export const DEFAULT_AIRPORT = {
  code: "CYYC",
  lon: -114.009933,
  lat: 51.118822,
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

    if (!CESIUM_ION_TOKEN) {
      throw new Error("Missing VITE_CESIUM_ION_TOKEN in .env");
    }

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
    const viewer = new Cesium.Viewer(containerRef.current, {
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

    // ── Custom mouse mapping ───────────────────────────────────────────────────
    // Keep wheel zoom, but repurpose right-drag for camera orientation control.
    const controller = viewer.scene.screenSpaceCameraController;
    controller.zoomEventTypes = [
      Cesium.CameraEventType.WHEEL,
      Cesium.CameraEventType.PINCH,
    ];

    // Disable built-in right-drag camera actions; we provide custom behavior.
    controller.tiltEventTypes = [
      {
        eventType: Cesium.CameraEventType.MIDDLE_DRAG,
      },
      {
        eventType: Cesium.CameraEventType.PINCH,
      },
      {
        eventType: Cesium.CameraEventType.LEFT_DRAG,
        modifier: Cesium.KeyboardEventModifier.CTRL,
      },
      {
        eventType: Cesium.CameraEventType.RIGHT_DRAG,
        modifier: Cesium.KeyboardEventModifier.CTRL,
      },
    ];
    controller.lookEventTypes = [];

    const canvas = viewer.scene.canvas;
    const PITCH_SENSITIVITY = 0.005;
    const HEADING_SENSITIVITY = 0.005;
    let pointerDragging = false;
    let activePointerId: number | null = null;
    let lastX = 0;
    let lastY = 0;

    const isRightLikeButton = (event: PointerEvent): boolean => {
      return event.button === 2 || (event.button === 0 && event.ctrlKey);
    };

    const isRightLikePressed = (event: PointerEvent): boolean => {
      const rightPressed = (event.buttons & 2) !== 0;
      const ctrlLeftPressed = (event.buttons & 1) !== 0 && event.ctrlKey;
      return rightPressed || ctrlLeftPressed;
    };

    const onContextMenu = (event: MouseEvent) => {
      event.preventDefault();
    };

    const onPointerDown = (event: PointerEvent) => {
      if (!isRightLikeButton(event)) return;
      pointerDragging = true;
      activePointerId = event.pointerId;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
      event.preventDefault();
    };

    const onPointerUp = (event: PointerEvent) => {
      if (!pointerDragging) return;
      if (activePointerId !== null && event.pointerId !== activePointerId) return;
      pointerDragging = false;
      if (activePointerId !== null && canvas.hasPointerCapture(activePointerId)) {
        canvas.releasePointerCapture(activePointerId);
      }
      activePointerId = null;
    };

    const onPointerCancel = (event: PointerEvent) => {
      if (!pointerDragging) return;
      if (activePointerId !== null && event.pointerId !== activePointerId) return;
      pointerDragging = false;
      activePointerId = null;
    };

    const onPointerMove = (event: PointerEvent) => {
      if (!pointerDragging) return;
      if (activePointerId !== null && event.pointerId !== activePointerId) return;
      if (!isRightLikePressed(event)) return;

      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;

      const camera = viewer.camera;
      if (event.shiftKey) {
        const nextHeading = camera.heading - dx * HEADING_SENSITIVITY;
        camera.setView({
          orientation: {
            heading: nextHeading,
            pitch: camera.pitch,
            roll: camera.roll,
          },
        });
      } else {
        const nextPitch = Cesium.Math.clamp(
          camera.pitch + dy * PITCH_SENSITIVITY,
          Cesium.Math.toRadians(-89),
          Cesium.Math.toRadians(-5)
        );
        camera.setView({
          orientation: {
            heading: camera.heading,
            pitch: nextPitch,
            roll: camera.roll,
          },
        });
      }

      event.preventDefault();
    };

    canvas.addEventListener("contextmenu", onContextMenu);
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerCancel);

    // ── Step 3: Enable terrain lighting ───────────────────────────────────────
    // ② — Enable the globe's built-in directional lighting:
    viewer.scene.globe.enableLighting = true;
    // Ensure entities behind terrain are depth-tested to prevent view-angle
    // dependent visual drifting against runway polygons.
    viewer.scene.globe.depthTestAgainstTerrain = true;
    //
    // Hint: set it to `true`.  The sun's position is computed from the
    // simulation clock time, so you'll see day/night and mountain shadows.

    // set current clock to mid-afternoon
    // viewer.clock.currentTime = Cesium.JulianDate.fromDate(new Date("2026-01-01T21:00:00Z"));

    // ── Step 4: Set initial camera view ───────────────────────────────────────
    // ③ — Fly the camera to DEFAULT_AIRPORT using viewer.camera.setView().
    //
    // viewer.camera.setView({
    //   destination: Cesium.Cartesian3.fromDegrees(DEFAULT_AIRPORT.lon, DEFAULT_AIRPORT.lat, DEFAULT_AIRPORT.height),
    //   orientation: {
    //     heading: Cesium.Math.toRadians(0),   // compass bearing (0 = north)
    //     pitch:   Cesium.Math.toRadians(-42),   // tilt angle (negative = look down)
    //     roll:    0,                          // bank angle (0 = level)
    //   },
    // });
    viewer.camera.flyToBoundingSphere(
      new Cesium.BoundingSphere(
        Cesium.Cartesian3.fromDegrees(DEFAULT_AIRPORT.lon, DEFAULT_AIRPORT.lat, 0),
        DEFAULT_AIRPORT.height
      ),
      {
        duration: 1, // seconds; set to 0 for no animation
        offset: new Cesium.HeadingPitchRange(
          Cesium.Math.toRadians(-45),   // heading (compass bearing)
          Cesium.Math.toRadians(-42),   // pitch (tilt angle)
          DEFAULT_AIRPORT.height       // range (distance from target)
        ),
      }
    );

    // ── Step 5: Add an airport marker so the target spot is obvious ──────────
    viewer.entities.add({
      id: `airport-${DEFAULT_AIRPORT.code}`,
      position: Cesium.Cartesian3.fromDegrees(DEFAULT_AIRPORT.lon, DEFAULT_AIRPORT.lat),
      point: {
        pixelSize: 12,
        color: Cesium.Color.fromCssColorString("#ff4d4f"),
        outlineColor: Cesium.Color.WHITE,
        outlineWidth: 2,
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: DEFAULT_AIRPORT.code,
        font: "bold 14px sans-serif",
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        fillColor: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 3,
        pixelOffset: new Cesium.Cartesian2(0, -22),
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
    //
    // Hint: a pitch of -45° gives a nice oblique view; -90° is straight down.
    // Reference: docs/01-cesium-viewer.md § "Camera orientation"

    // cinematic rendering
    viewer.scene.postProcessStages.fxaa.enabled = true; // anti-aliasing
    // viewer.scene.fog.enabled = true;

    // Share the Viewer with the rest of the app via context.
    setViewer(viewer);

    // ── Cleanup: destroy the Viewer when the component unmounts ───────────────
    // This releases WebGL resources and prevents memory leaks.
    return () => {
      canvas.removeEventListener("contextmenu", onContextMenu);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointercancel", onPointerCancel);

      viewerRef.current?.destroy();
      viewerRef.current = null;
    };
  }, [containerRef, setViewer]);
}
