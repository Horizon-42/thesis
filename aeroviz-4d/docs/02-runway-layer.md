# Tutorial 02 — Runway & Waypoint Layers (GeoJSON in CesiumJS)

**Covers:** `src/hooks/useRunwayLayer.ts`, `src/hooks/useWaypointLayer.ts`

---

## What you will implement

- Runway polygons loaded from GeoJSON, clamped to the real terrain surface
- Dual runway zoning style:
  - `runway_surface` (dark grey)
  - `landing_zone` (light green overlay)
- Approach waypoints displayed as 3D cylinder markers with text labels
- Layer visibility toggled by the ControlPanel checkboxes

---

## Concept 1 — GeoJsonDataSource

`Cesium.GeoJsonDataSource` is the easiest way to load GeoJSON into Cesium.
It converts each GeoJSON Feature into a Cesium Entity automatically:

| GeoJSON geometry | Cesium Entity property |
|-----------------|----------------------|
| `Point`         | `billboard` + `label` |
| `LineString`    | `polyline` |
| `Polygon`       | `polygon` |

```typescript
const ds = new Cesium.GeoJsonDataSource("my-layer");
ds.load("/data/runway.geojson", {
  clampToGround: true,
  fill: new Cesium.Color(0.15, 0.15, 0.15, 0.9),
  stroke: Cesium.Color.YELLOW,
  strokeWidth: 2,
}).then(ds => {
  viewer.dataSources.add(ds);

  // Style per-feature by properties.zone_type
  ds.entities.values.forEach((entity) => {
    if (!entity.polygon) return;
    const zoneType = entity.properties?.zone_type?.getValue(Cesium.JulianDate.now());
    const isLandingZone = zoneType === "landing_zone";
    entity.polygon.material = new Cesium.ColorMaterialProperty(
      isLandingZone
        ? new Cesium.Color(0.65, 0.9, 0.65, 0.35)
        : new Cesium.Color(0.15, 0.15, 0.15, 0.85)
    );
  });
});
```

Key options:

| Option | Type | What it does |
|--------|------|-------------|
| `clampToGround` | boolean | Drapes the polygon onto the terrain surface |
| `fill` | `Cesium.Color` | Interior fill colour (RGBA) |
| `stroke` | `Cesium.Color` | Outline colour |
| `strokeWidth` | number | Outline width in pixels |

---

## Concept 2 — ClassificationType

When `clampToGround: true`, the polygon is projected downward onto the scene.
`ClassificationType` controls what it drapes over:

| Value | Drapes onto |
|-------|------------|
| `TERRAIN` | Terrain mesh only — does NOT cover 3D buildings or aircraft models |
| `CESIUM_3D_TILE` | 3D tile sets only |
| `BOTH` | Both terrain and 3D tiles |

For runways, use `TERRAIN` so the polygons don't cover the aircraft models:

```typescript
entity.polygon!.classificationType =
  new Cesium.ConstantProperty(Cesium.ClassificationType.TERRAIN);
```

Why `ConstantProperty`?  Cesium uses a "Property" wrapper around most values
to support time-varying properties (a polygon that changes colour over time,
for example).  For static values you use `ConstantProperty`.

---

## Concept 3 — Two-effect pattern for layers

Notice that `useRunwayLayer.ts` has **two separate `useEffect` calls**:

```
Effect 1 (deps: [viewer])    → load the GeoJSON once, add to scene
Effect 2 (deps: [viewer, layers.runways]) → show/hide without reloading
```

This is intentional.  If you put everything in one effect with `layers.runways`
in the dependency array, toggling the layer would trigger a full network re-fetch
of the GeoJSON file every time.  Splitting the effects means:
- Effect 1 runs **once** (when viewer first becomes available)
- Effect 2 runs **cheaply** (just sets `ds.show = true/false`)

---

## Concept 4 — Why cylinders for waypoints?

CesiumJS `billboard` icons are 2D sprites that always face the camera.  They
look great at high altitudes but become very small when you zoom in close.

3D cylinders are rendered as actual geometry in the scene — they stay the same
size in metres regardless of zoom level, which makes them easy to see when you
are flying through the approach at low altitude.

---

## Your TODOs

### In `useRunwayLayer.ts`

| TODO | What to do |
|------|-----------|
| ① | Call `dataSource.load(url, options)` with the correct options |
| ② | Add the loaded DataSource to `viewer.dataSources` |
| ③ | Set `ClassificationType.TERRAIN` on each polygon entity |
| ④ | Set `ds.show = layers.runways` after adding |
| ⑤ | In the second effect, set `ds.show = layers.runways` |

### In `useWaypointLayer.ts`

| TODO | What to do |
|------|-----------|
| ① | Add each waypoint as a Cesium Entity with cylinder + label |
| ② | In the visibility effect, loop and set `entity.show` for waypoint entities |

---

## Checklist

After completing all TODOs:
- [ ] Dark grey runways are visible at the airport, clamped to terrain
- [ ] Coloured cylinders appear at waypoint locations
- [ ] Waypoint names are legible at ~5 km altitude
- [ ] Unchecking "Runways" in the panel hides them; re-checking shows them
- [ ] Unchecking "Runways" does NOT trigger a second network request
      (verify in browser DevTools → Network tab)

---

## Stretch goals

1. **Click to inspect** — Add a `viewer.screenSpaceEventHandler` that fires
   when you click a polygon, reads the entity's properties, and displays them
   in an info panel.
2. **Runway threshold arrows** — Use a `Polyline` entity to draw a small arrow
   pointing in the landing direction at each runway end.
