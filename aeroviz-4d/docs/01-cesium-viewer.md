# Tutorial 01 — CesiumJS Viewer Initialisation

**Covers:** `src/hooks/useCesiumViewer.ts`, `src/components/CesiumViewer.tsx`

---

## What you will implement

By the end of this tutorial you will have a full-screen 3D globe that:
- Displays the Earth with high-resolution satellite imagery
- Renders real terrain elevation (mountains, valleys)
- Shows atmospheric effects (day/night, sky scattering)
- Has its initial camera aimed at your target airport from a 45° oblique angle

---

## Concept 1 — The Cesium Viewer

`Cesium.Viewer` is the God object of CesiumJS.  It owns:

| Sub-object | What it controls |
|------------|-----------------|
| `viewer.scene` | Rendering pipeline, lighting, sky |
| `viewer.scene.globe` | The Earth surface, terrain, imagery |
| `viewer.camera` | 3D camera position and orientation |
| `viewer.clock` | Simulation time (drives 4D animation) |
| `viewer.timeline` | The scrubber UI bar at the bottom |
| `viewer.entities` | Scene graph of static/animated shapes |
| `viewer.dataSources` | Container for bulk GeoJSON/CZML datasets |

You create it once with `new Cesium.Viewer(domElement, options)`.  Because it
manages a WebGL context, you must call `viewer.destroy()` when the component
unmounts — otherwise the GPU resources are leaked.

---

## Concept 2 — Terrain

CesiumJS supports swappable terrain providers.  For this project we use
**Cesium World Terrain** (streamed from Cesium Ion):

```typescript
Cesium.Terrain.fromWorldTerrain({
  requestVertexNormals: true,  // ← download surface normals for lighting
  requestWaterMask: true,      // ← enable ocean / lake reflections
})
```

`requestVertexNormals: true` downloads an extra data stream that tells the
shader how steep the terrain is at each point.  Without it, mountains look
uniformly grey because the renderer can't compute which faces point toward
the sun.

---

## Concept 3 — Camera orientation

CesiumJS cameras use a spherical orientation defined by three angles:

| Angle | Unit | Meaning |
|-------|------|---------|
| `heading` | radians | Compass bearing (0 = north, π/2 = east) |
| `pitch`   | radians | Tilt from horizontal (**negative** = look down) |
| `roll`    | radians | Bank angle around the look-at axis (usually 0) |

For an oblique overview: `heading = 0` (north), `pitch = -π/4` (−45°).
For a straight-down plan view: `pitch = -π/2` (−90°).

```typescript
viewer.camera.setView({
  destination: Cesium.Cartesian3.fromDegrees(lon, lat, heightMetres),
  orientation: {
    heading: Cesium.Math.toRadians(0),
    pitch:   Cesium.Math.toRadians(-45),
    roll:    0,
  },
});
```

`Cesium.Math.toRadians(degrees)` — always use this conversion helper, never
multiply by `Math.PI / 180` manually (it's easy to forget where you last used
magic numbers).

---

## Your TODOs in useCesiumViewer.ts

Open `src/hooks/useCesiumViewer.ts` and complete these three TODOs in order:

### ① — Create the Viewer

Replace the placeholder `null as unknown as Cesium.Viewer` with a real
`new Cesium.Viewer(containerRef.current!, { ... })` call using the options
documented in the file.

**Checklist after completing ①:**
- [x] The page shows a 3D globe (may take a few seconds to load imagery tiles)
- [x] No `Missing Ion access token` console error (you filled in your token)
- [x] The timeline bar and animation wheel are visible at the bottom

### ② — Enable terrain lighting

One line: `viewer.scene.globe.enableLighting = true;`

**Checklist after completing ②:**
- [x] Mountains show darker shadows on slopes facing away from the sun
- [x] The effect is more visible when you set `viewer.clock.currentTime` to a
      mid-afternoon time

### ③ — Set the initial camera view

Use `viewer.camera.setView(...)` with the `DEFAULT_AIRPORT` constants.

**Checklist after completing ③:**
- [x] The globe opens looking at your target airport, not at 0°N 0°E
- [x] The view is tilted (oblique), not straight down

---

## Common mistakes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| White screen, no globe | Token not set or invalid | Check `CESIUM_ION_TOKEN` constant |
| Globe loads but terrain is flat | `requestVertexNormals` is false or terrain not set | Pass `fromWorldTerrain({requestVertexNormals: true})` |
| Camera starts at Africa (0°N 0°E) | `setView` not called or using wrong coordinates | Check `DEFAULT_AIRPORT.lon/lat` values |
| React StrictMode: globe renders twice | Normal in dev — React mounts twice to detect side-effects | Ignore in development; disappears in production build |

---

## Stretch goals (optional)

Once the basic TODOs work, try these:

1. **HUD overlay** — Display the current camera altitude (metres) in a
   `<div>` overlay.  Read it from `viewer.camera.positionCartographic.height`.
2. **Fly-to animation** — Instead of `setView`, use `viewer.camera.flyTo({...})`
   which animates the camera smoothly.  Add a `duration` option to control speed.
3. **Depth of field** — Try `viewer.scene.postProcessStages.ambientOcclusion.enabled = true`
   for cinematic rendering.
