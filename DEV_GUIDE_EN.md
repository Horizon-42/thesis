<!--
  This file was automatically generated from the bilingual source by split_doc.py (English version).
  Source bilingual document: DEV_GUIDE_BILINGUAL.md
-->

# 机场4D轨迹与地形数字孪生可视化系统 — 开发文档
# Airport 4D Trajectory & Terrain Digital Twin Visualization — Development Guide


## 0. Document Metadata

| Field | Content |
|-------|---------|
| Project Code | AeroViz-4D |
| Document Version | v1.0.0 |
| Last Updated | 2026-04-01 |
| Target Audience | Frontend developers, aviation algorithm researchers, AI-assisted (Vibe-coding) users |
| Tech Stack | React + TypeScript + CesiumJS + Python FastAPI |
| Data Protocol | GeoJSON (static) / CZML (dynamic 4D trajectories) |

---

## 1. Project Purpose & Research Background

### 1.1 Why This System Is Needed

Modern Terminal Maneuvering Area (TMA) management faces three compounding challenges:

1. **Terrain Complexity**: Approach procedures at mountainous airports (e.g., Kelowna CYLW in Canada, Lukla VNLK in Nepal) must precisely avoid terrain obstacles. No path-optimization algorithm can be validated without real elevation data as the ground truth.
2. **4D Trajectory Temporality**: An aircraft's position (X/Y/Z) combined with a precise arrival time constraint (T) constitutes a 4D trajectory. Traditional 2D paper charts cannot intuitively convey the temporal spacing coordination among multiple aircraft.
3. **Algorithm Result Interpretability**: Sequencing and scheduling models such as Mixed-Integer Linear Programming (MILP) or genetic algorithms output abstract numeric sequences. Researchers need a visualization sandbox to validate correctness and present findings intuitively to review committees (thesis defense panels, conference reviewers).

The core value of this system: **Transform abstract algorithm output into interactive 3D spatiotemporal demonstrations**, making the abstract constraint "aircraft X should be at position Y at 13:42:30" into a visible trajectory verifiable by dragging a timeline slider.

### 1.2 Dual Academic & Engineering Positioning

- **Academic side**: Supports result visualization and validation for "terrain-aware 4D trajectory prediction" and "arrival sequencing scheduling" chapters in the thesis.
- **Engineering side**: Builds a reusable visualization component library for iterative research.

---

## 2. Required Knowledge Background

### 2.1 Aviation Domain Knowledge

#### 2.1.1 Terminal Area & Approach Procedure Basics

| Concept | Description |
|---------|-------------|
| TMA (Terminal Maneuvering Area) | Controlled airspace surrounding an airport, typically 30–100 NM radius and below FL245 |
| IAF / IF / FAF | Initial Approach Fix / Intermediate Fix / Final Approach Fix — the waypoint nodes of an approach procedure |
| RNAV/RNP | Area Navigation / Required Navigation Performance — allows aircraft to fly curved paths with precision guidance |
| MSA (Minimum Sector Altitude) | Lowest altitude within a defined radius of the airport navaid that provides 300 m obstacle clearance |
| OCA/H | Obstacle Clearance Altitude/Height — aircraft must fly above this to clear terrain obstacles |

#### 2.1.2 4D Trajectory Concepts

- **RBT (Reference Business Trajectory)**: The flight path an airline commits to with ATC; the core construct of TBO (Trajectory-Based Operations).
- **CTA (Controlled Time of Arrival)**: The precise time slot ATC assigns for an aircraft to cross a given fix, typically within a ±30-second tolerance window.
- **Separation Standards**: Radar separation minimum 3 NM; non-radar 5–10 flight minutes. These constraints form the core inequality constraints of the scheduling algorithm.

#### 2.1.3 Obstacle Clearance Surface (OCS)

PANS-OPS (ICAO Doc 8168) defines the protection area geometry for approach procedures:
- **Primary Protection Area**: Symmetric extension on both sides of the runway centerline; full obstacle clearance is guaranteed throughout.
- **Secondary Protection Area**: Outer flanks of the primary area; slopes outward and upward at a 7:1 horizontal-to-vertical ratio, with obstacle clearance gradually reducing to zero.
- **OCS Slope**: The obstacle clearance surface for the final approach segment rises toward the runway threshold at a defined gradient (e.g., 2.5% slope); all terrain and man-made obstacles must remain below this surface.

### 2.2 Frontend Technical Knowledge

#### 2.2.1 React + TypeScript Fundamentals

Key concepts to master:
- `useEffect` / `useRef`: The CesiumJS Viewer must be initialized after the DOM mounts, and its instance must be held via a ref to avoid stale closure issues.
- `useState` + Context API: Manages global UI state such as timeline playback status, selected aircraft, and layer visibility.
- TypeScript interface definitions: Strongly type CZML data structures and GeoJSON Feature properties to prevent runtime errors.

#### 2.2.2 CesiumJS Core Concepts

| API Layer | Purpose |
|-----------|---------|
| `Viewer` | Top-level container holding scene, camera, clock, timeline, and all subsystems |
| `Scene` / `Globe` | Controls Earth rendering, lighting, and atmospheric effects |
| `Entity API` | Declarative geometry drawing (Polygon, Polyline, Billboard, Model); suited for small-to-medium static datasets |
| `Primitive API` | Imperative high-performance rendering; suited for large-scale dynamic update scenarios |
| `DataSource` | Container for bulk-loading GeoJSON/CZML datasets |
| `Clock` | System clock driving animation time, synchronized with CZML timestamps |
| `SampledPositionProperty` | Stores time-stamped position sequences with interpolation support (linear/Lagrange/Hermite) |
| `Camera` | Controls the viewpoint; supports `flyTo`, `lookAt`, `setView`, and other operations |

#### 2.2.3 Coordinate Systems

- **WGS84**: Earth ellipsoid coordinate system — longitude, latitude, and ellipsoidal height; the foundational coordinate system for CesiumJS.
- **Cartesian3**: CesiumJS internal Cartesian coordinates; use `Cesium.Cartesian3.fromDegrees(lon, lat, alt)` for conversion.
- **ENU (East-North-Up) Local Frame**: Used for computing local offsets of OCS geometry before converting back to WGS84.

### 2.3 Python Backend Knowledge

#### 2.3.1 The OpenAP Library

OpenAP is an open-source aircraft performance model library providing:
- Climb/cruise/descent performance envelopes for common aircraft types (B737, A320, etc.).
- Speed/altitude profile computation satisfying performance constraints for a given waypoint sequence.
- Flyability validation for 4D trajectories.

#### 2.3.2 CZML Data Format

CZML (Cesium Language) is a JSON time-series format designed specifically for CesiumJS:
```json
[
  { "id": "document", "name": "4D Trajectories", "version": "1.0",
    "clock": { "interval": "2026-04-01T08:00:00Z/2026-04-01T09:00:00Z",
               "currentTime": "2026-04-01T08:00:00Z", "multiplier": 60 }},
  { "id": "UAL123",
    "model": { "gltf": "/models/aircraft.glb", "scale": 3.0 },
    "position": {
      "epoch": "2026-04-01T08:00:00Z",
      "cartographicDegrees": [
        0,   -119.38, 49.95, 4500,
        120, -119.42, 49.88, 3800,
        240, -119.45, 49.80, 3200
      ]
    },
    "orientation": { "velocityReference": "#UAL123.position" }
  }
]
```
Each group of 4 values in `cartographicDegrees`: `[secondOffset, longitude, latitude, altitude(m)]`.

---

## 3. System Architecture Design

### 3.1 Overall Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       User Browser                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              React + TypeScript Frontend              │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │ CesiumViewer│  │ControlPanel  │  │FlightTable │  │   │
│  │  │  Component  │  │  Component   │  │ Component  │  │   │
│  │  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  │   │
│  │         └────────────────┼────────────────┘          │   │
│  │                    AppContext                         │   │
│  │         (clock state / selected flight / layers)      │   │
│  └──────────────────────────────────────────────────────┘   │
│                             │                                │
│               HTTP / Static file serving                     │
└─────────────────────────────┼───────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    ┌────┴─────┐        ┌─────┴─────┐       ┌─────┴──────┐
    │ GeoJSON  │        │   CZML    │       │  Python    │
    │  Static  │        │ Trajectory│       │  FastAPI   │
    │ (runway/ │        │  (4D seq) │       │  (backend  │
    │  OCS)    │        │           │       │  optional) │
    └──────────┘        └───────────┘       └────────────┘
```

### 3.2 Data Flow

```
OurAirports CSV → Python preprocessing → runway.geojson
                                               ↓
ICAO PANS-OPS geometry calculation → ocs_surfaces.geojson
                                               ↓
Nav Canada/FAA CIFP → waypoints.geojson
                                               ↓
Scheduling algorithm (MILP/GA) → trajectories.czml
                                               ↓
                                    CesiumJS 3D rendering
```

### 3.3 Directory Structure

```
aeroviz-4d/
├── public/
│   ├── models/
│   │   └── aircraft.glb          # 3D aircraft model
│   └── data/
│       ├── runway.geojson         # runway polygons
│       ├── ocs_surfaces.geojson   # OCS protection surfaces
│       ├── waypoints.geojson      # approach waypoints
│       └── trajectories.czml     # 4D trajectories (Python-generated)
├── src/
│   ├── components/
│   │   ├── CesiumViewer.tsx       # main 3D view component
│   │   ├── ControlPanel.tsx       # playback control panel
│   │   ├── FlightTable.tsx        # flight sequence table
│   │   └── LayerToggle.tsx        # layer toggle component
│   ├── hooks/
│   │   ├── useCesiumViewer.ts     # Viewer initialization hook
│   │   ├── useTerrainLoader.ts    # terrain loading hook
│   │   └── useCzmlLoader.ts      # CZML loading hook
│   ├── context/
│   │   └── AppContext.tsx         # global state context
│   ├── types/
│   │   ├── czml.d.ts             # CZML type definitions
│   │   └── geojson-aviation.d.ts # aviation GeoJSON property types
│   ├── utils/
│   │   ├── ocsGeometry.ts        # OCS geometry computation utilities
│   │   └── czmlBuilder.ts        # CZML construction helpers
│   ├── App.tsx
│   └── main.tsx
├── python/
│   ├── generate_czml.py          # CZML generation script
│   ├── preprocess_airports.py    # airport data preprocessing
│   └── requirements.txt
├── vite.config.ts
├── tsconfig.json
└── package.json
```

---

## 4. Phase 1: Environment Setup & CesiumJS Initialization

### 4.1 Objective

Establish a 3D canvas with realistic Earth curvature, high-resolution satellite imagery, terrain rendering, and dynamic lighting — giving all subsequent layers a correct geographic coordinate foundation.

### 4.2 Environment Installation

```bash
# Initialize Vite + React + TypeScript project
npm create vite@latest aeroviz-4d -- --template react-ts
cd aeroviz-4d

# Install CesiumJS and the Vite plugin
npm install cesium
npm install -D vite-plugin-cesium

# Install all dependencies
npm install
```

### 4.3 Vite Configuration

```typescript
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import cesium from 'vite-plugin-cesium';

export default defineConfig({
  plugins: [react(), cesium()],
});
```

### 4.4 Main Viewer Component

```typescript
// src/components/CesiumViewer.tsx
import { useEffect, useRef } from 'react';
import * as Cesium from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';

// !! Fill in your Cesium Ion Access Token here
// Get one free at https://cesium.com/ion/tokens
const CESIUM_ION_TOKEN = 'YOUR_TOKEN_HERE';

// Target airport coordinates (Kelowna CYLW as example)
const AIRPORT_LON = -119.3775;
const AIRPORT_LAT = 49.9561;
const INITIAL_HEIGHT = 15000; // initial camera height in meters

export default function CesiumViewerComponent() {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Cesium.Viewer | null>(null);

  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return;

    Cesium.Ion.defaultAccessToken = CESIUM_ION_TOKEN;

    viewerRef.current = new Cesium.Viewer(containerRef.current, {
      // Use Cesium World Terrain high-precision elevation
      terrain: Cesium.Terrain.fromWorldTerrain({
        requestVertexNormals: true,   // enable normals for lighting/shadows
        requestWaterMask: true,        // enable water reflection mask
      }),
      // Hide default UI controls for a clean interface
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      // Keep timeline and animation controller (needed for 4D trajectories)
      animation: true,
      timeline: true,
      // Enable HDR rendering and atmospheric effects
      skyAtmosphere: new Cesium.SkyAtmosphere(),
    });

    // Enable terrain lighting
    viewerRef.current.scene.globe.enableLighting = true;

    // Set initial camera view (tilted bird's-eye view over airport)
    viewerRef.current.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(
        AIRPORT_LON, AIRPORT_LAT, INITIAL_HEIGHT
      ),
      orientation: {
        heading: Cesium.Math.toRadians(0),   // facing north
        pitch: Cesium.Math.toRadians(-45),    // tilted 45° downward
        roll: 0,
      },
    });

    return () => {
      viewerRef.current?.destroy();
      viewerRef.current = null;
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ width: '100vw', height: '100vh', position: 'absolute', top: 0, left: 0 }}
    />
  );
}
```

### 4.5 AI-Assisted Prompt

> Send the following prompt to Cursor / Windsurf / Claude or similar AI coding tools:

```
Based on the CesiumViewerComponent above, complete the following:
1. Import the component in App.tsx so it fills the full screen.
2. Add a semi-transparent HUD panel in the top-left corner using absolute
   positioning, displaying the current camera altitude (meters) and
   lat/lon coordinates read from viewer.camera.
3. All styling must use CSS Modules (*.module.css) — no inline styles.
```

---

## 5. Phase 2: High-Fidelity Terrain & Airport Geometry Modeling

### 5.1 Objective

Overlay real runway geometry (extracted from the OurAirports dataset), clamped precisely to the terrain surface, and add runway lighting visual effects.

### 5.2 Data Preparation: OurAirports → GeoJSON

```python
# python/preprocess_airports.py
import pandas as pd
import json
import math

def runway_to_polygon(lat1, lon1, lat2, lon2, width_ft=150):
    """
    Convert runway centerline endpoints to a polygon (accounting for width).
    width_ft: runway width in feet (typical: 150 ft ≈ 45.7 m)
    """
    width_m = width_ft * 0.3048
    bearing = math.atan2(lon2 - lon1, lat2 - lat1)
    perp = bearing + math.pi / 2

    # Lat/lon offsets (approximate, valid for small areas)
    dlat = (width_m / 2) / 111320
    dlon = (width_m / 2) / (111320 * math.cos(math.radians(lat1)))

    # Four corner points
    corners = [
        [lon1 + dlon * math.cos(perp), lat1 + dlat * math.sin(perp)],
        [lon1 - dlon * math.cos(perp), lat1 - dlat * math.sin(perp)],
        [lon2 - dlon * math.cos(perp), lat2 - dlat * math.sin(perp)],
        [lon2 + dlon * math.cos(perp), lat2 + dlat * math.sin(perp)],
        [lon1 + dlon * math.cos(perp), lat1 + dlat * math.sin(perp)],  # close
    ]
    return corners

# Load OurAirports runways.csv
df = pd.read_csv('runways.csv')
cylw = df[df['airport_ident'] == 'CYLW']

features = []
for _, row in cylw.iterrows():
    coords = runway_to_polygon(
        row['le_latitude_deg'], row['le_longitude_deg'],
        row['he_latitude_deg'], row['he_longitude_deg'],
        width_ft=row.get('width_ft', 150)
    )
    features.append({
        "type": "Feature",
        "properties": {
            "id": row['id'],
            "airport": row['airport_ident'],
            "le_ident": row['le_ident'],
            "he_ident": row['he_ident'],
            "surface": row.get('surface', 'ASP'),
            "length_ft": row.get('length_ft', 0),
        },
        "geometry": { "type": "Polygon", "coordinates": [coords] }
    })

with open('../public/data/runway.geojson', 'w') as f:
    json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
print(f"Generated {len(features)} runway polygons")
```

### 5.3 Frontend Runway Loading

```typescript
// src/hooks/useRunwayLayer.ts
import { useEffect } from 'react';
import * as Cesium from 'cesium';

export function useRunwayLayer(viewer: Cesium.Viewer | null) {
  useEffect(() => {
    if (!viewer) return;

    const dataSource = new Cesium.GeoJsonDataSource('runways');
    
    dataSource.load('/data/runway.geojson', {
      clampToGround: true,   // strictly clamp runway polygon to terrain
      fill: new Cesium.Color(0.15, 0.15, 0.15, 0.9),   // dark grey semi-transparent
      stroke: new Cesium.Color(1.0, 0.9, 0.0, 1.0),     // yellow border
      strokeWidth: 2,
    }).then((ds) => {
      viewer.dataSources.add(ds);
      ds.entities.values.forEach((entity) => {
        if (entity.polygon) {
          entity.polygon.classificationType =
            Cesium.ClassificationType.TERRAIN; // clamp to terrain only, don't occlude models
        }
      });
    });

    return () => {
      viewer.dataSources.removeAll();
    };
  }, [viewer]);
}
```

### 5.4 AI-Assisted Prompt

```
Building on the existing useRunwayLayer hook, add the following:
1. When the user clicks a runway polygon, show an InfoBox popup
   displaying runway identifier (e.g. "34L/16R"), length (feet),
   and surface type.
2. The clicked runway highlights in blue; all others return to dark grey.
3. Highlight state is managed through selectedRunway in AppContext.
```

---

## 6. Phase 3: Static Airspace Structures & OCS Visualization

### 6.1 Objective

Convert the abstract protection surface geometry rules defined by PANS-OPS into interactive 3D semi-transparent solids floating above real terrain, visually communicating the safety margins aircraft must maintain during approach.

### 6.2 OCS Geometry Calculation

```typescript
// src/utils/ocsGeometry.ts
import * as Cesium from 'cesium';

interface OCSParams {
  /** Final Approach Fix (FAF) coordinates */
  fafLon: number;
  fafLat: number;
  /** Runway threshold coordinates */
  thresholdLon: number;
  thresholdLat: number;
  /** FAF altitude (meters) */
  fafAlt: number;
  /** Runway threshold altitude (meters) */
  thresholdAlt: number;
  /** Primary protection area half-width (meters, typical: 75m for Cat I ILS) */
  primaryHalfWidth: number;
  /** Secondary protection area additional width (meters) */
  secondaryWidth: number;
}

/**
 * Generate an array of Cesium Entities representing the final approach OCS surfaces.
 * Returns: primary area (red semi-transparent) + 2× secondary areas (orange semi-transparent)
 */
export function buildOCSSurfaces(params: OCSParams): Cesium.Entity[] {
  const {
    fafLon, fafLat, fafAlt,
    thresholdLon, thresholdLat, thresholdAlt,
    primaryHalfWidth, secondaryWidth,
  } = params;

  // Compute runway bearing
  const dx = thresholdLon - fafLon;
  const dy = thresholdLat - fafLat;
  const bearingRad = Math.atan2(dx, dy);
  const perpRad = bearingRad + Math.PI / 2;

  // 1° latitude ≈ 111320 m; 1° longitude ≈ 111320 × cos(lat) m
  const metersPerDegLat = 111320;
  const metersPerDegLon = 111320 * Math.cos(Cesium.Math.toRadians(fafLat));

  function offsetPoint(lon: number, lat: number, bearing: number, distMeters: number) {
    return {
      lon: lon + (distMeters / metersPerDegLon) * Math.sin(bearing),
      lat: lat + (distMeters / metersPerDegLat) * Math.cos(bearing),
    };
  }

  // Four corners of the primary protection area at FAF
  const fafLeft = offsetPoint(fafLon, fafLat, perpRad, primaryHalfWidth);
  const fafRight = offsetPoint(fafLon, fafLat, perpRad, -primaryHalfWidth);
  const thrLeft = offsetPoint(thresholdLon, thresholdLat, perpRad, primaryHalfWidth);
  const thrRight = offsetPoint(thresholdLon, thresholdLat, perpRad, -primaryHalfWidth);

  const primaryEntity = new Cesium.Entity({
    name: 'OCS Primary Protection Area',
    polygon: {
      hierarchy: new Cesium.PolygonHierarchy(
        Cesium.Cartesian3.fromDegreesArrayHeights([
          fafLeft.lon, fafLeft.lat, fafAlt,
          fafRight.lon, fafRight.lat, fafAlt,
          thrRight.lon, thrRight.lat, thresholdAlt,
          thrLeft.lon, thrLeft.lat, thresholdAlt,
        ])
      ),
      perPositionHeight: true,
      material: Cesium.Color.RED.withAlpha(0.25),
      outline: true,
      outlineColor: Cesium.Color.RED,
    },
  });

  // Secondary protection area (right side, 7:1 slope)
  const secFafRight = offsetPoint(fafLon, fafLat, perpRad, -(primaryHalfWidth + secondaryWidth));
  const secAltAtFaf = fafAlt - secondaryWidth / 7; // 7:1 slope ratio
  const secThrRight = offsetPoint(thresholdLon, thresholdLat, perpRad, -(primaryHalfWidth + secondaryWidth));

  const secondaryRightEntity = new Cesium.Entity({
    name: 'OCS Secondary Protection Area (Right)',
    polygon: {
      hierarchy: new Cesium.PolygonHierarchy(
        Cesium.Cartesian3.fromDegreesArrayHeights([
          fafRight.lon, fafRight.lat, fafAlt,
          secFafRight.lon, secFafRight.lat, secAltAtFaf,
          secThrRight.lon, secThrRight.lat, thresholdAlt,
          thrRight.lon, thrRight.lat, thresholdAlt,
        ])
      ),
      perPositionHeight: true,
      material: Cesium.Color.ORANGE.withAlpha(0.2),
      outline: true,
      outlineColor: Cesium.Color.ORANGE,
    },
  });

  return [primaryEntity, secondaryRightEntity];
}
```

### 6.3 Waypoint & Approach Path Rendering

```typescript
// src/hooks/useWaypointLayer.ts
import { useEffect } from 'react';
import * as Cesium from 'cesium';

export function useWaypointLayer(viewer: Cesium.Viewer | null) {
  useEffect(() => {
    if (!viewer) return;

    fetch('/data/waypoints.geojson')
      .then(r => r.json())
      .then(geojson => {
        geojson.features.forEach((f: GeoJSON.Feature) => {
          const [lon, lat, alt] = f.geometry.coordinates as number[];
          const props = f.properties as { name: string; type: string; minAlt?: number };

          // Waypoint cylinder marker
          viewer.entities.add({
            position: Cesium.Cartesian3.fromDegrees(lon, lat, alt),
            cylinder: {
              length: 300,
              topRadius: 150,
              bottomRadius: 150,
              material: props.type === 'FAF'
                ? Cesium.Color.YELLOW.withAlpha(0.8)
                : Cesium.Color.CYAN.withAlpha(0.7),
            },
            label: {
              text: props.name,
              font: '14px monospace',
              fillColor: Cesium.Color.WHITE,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              outlineWidth: 2,
              verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
              pixelOffset: new Cesium.Cartesian2(0, -20),
            },
          });
        });
      });
  }, [viewer]);
}
```

---

## 7. Phase 4: 4D Trajectory Spatiotemporal Playback System

### 7.1 Objective

Load multi-aircraft 4D trajectories generated by the Python scheduling algorithm in CZML format into CesiumJS, enabling high-fidelity dynamic playback with a draggable timeline. Support single-aircraft tracking, speed control, and trajectory path coloring.

### 7.2 Python Backend: CZML Generator

```python
# python/generate_czml.py
import json
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

def build_czml(
    flights: List[dict],
    start_time: datetime,
    playback_multiplier: int = 60
) -> list:
    """
    Args:
        flights: list of flight dicts, each with format:
            {
              "id": "UAL123",
              "callsign": "United 123",
              "type": "B738",
              "waypoints": [
                  (offset_sec, lon, lat, alt_m),
                  ...
              ]
            }
        start_time: simulation start time in UTC
        playback_multiplier: time acceleration (60 = 1 second represents 1 minute)
    """
    end_time = start_time + timedelta(
        seconds=max(wpt[0] for f in flights for wpt in f['waypoints'])
    )

    document = {
        "id": "document",
        "name": "AeroViz-4D Trajectories",
        "version": "1.0",
        "clock": {
            "interval": f"{start_time.isoformat()}/{end_time.isoformat()}",
            "currentTime": start_time.isoformat(),
            "multiplier": playback_multiplier,
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER"
        }
    }

    entities = [document]
    colors = [
        [1.0, 0.5, 0.0],  # orange
        [0.0, 0.8, 1.0],  # cyan
        [0.8, 1.0, 0.0],  # yellow-green
        [1.0, 0.2, 0.8],  # pink
    ]

    for i, flight in enumerate(flights):
        color = colors[i % len(colors)]
        epoch_iso = start_time.isoformat()

        # Build cartographicDegrees array: [offsetSec, lon, lat, alt, ...]
        cart_degrees = []
        for (offset_sec, lon, lat, alt_m) in flight['waypoints']:
            cart_degrees.extend([offset_sec, lon, lat, alt_m])

        entity = {
            "id": flight['id'],
            "name": flight['callsign'],
            "description": f"<b>{flight['callsign']}</b><br/>Type: {flight['type']}",
            # 3D aircraft model
            "model": {
                "gltf": "/models/aircraft.glb",
                "scale": 3.0,
                "minimumPixelSize": 32,
                "maximumScale": 20000,
                "runAnimations": True
            },
            # Position time series (Lagrange interpolation)
            "position": {
                "epoch": epoch_iso,
                "cartographicDegrees": cart_degrees,
                "interpolationAlgorithm": "LAGRANGE",
                "interpolationDegree": 3,
                "forwardExtrapolationType": "HOLD"
            },
            # Auto-compute heading from velocity vector
            "orientation": {
                "velocityReference": f"#{flight['id']}.position"
            },
            # Trail path
            "path": {
                "show": True,
                "leadTime": 0,
                "trailTime": 300,  # show last 300 seconds of trail
                "width": 2,
                "material": {
                    "solidColor": {
                        "color": {
                            "rgba": [int(c*255) for c in color] + [200]
                        }
                    }
                }
            },
            # Callsign label
            "label": {
                "text": flight['callsign'],
                "font": "12px sans-serif",
                "fillColor": {"rgba": [255, 255, 255, 255]},
                "outlineColor": {"rgba": [0, 0, 0, 255]},
                "outlineWidth": 2,
                "style": "FILL_AND_OUTLINE",
                "verticalOrigin": "BOTTOM",
                "pixelOffset": {"cartesian2": [0, -30]}
            }
        }
        entities.append(entity)

    return entities

# Example: generate mock data for frontend integration testing
if __name__ == '__main__':
    start = datetime(2026, 4, 1, 8, 0, 0, tzinfo=timezone.utc)

    mock_flights = [
        {
            "id": "UAL123", "callsign": "United 123", "type": "B738",
            "waypoints": [
                (0,    -119.10, 50.20, 5500),
                (180,  -119.20, 50.10, 4800),
                (360,  -119.30, 50.00, 4000),
                (540,  -119.36, 49.97, 3200),
                (660,  -119.38, 49.96, 2500),
                (780,  -119.385, 49.957, 1800),
                (900,  -119.390, 49.955, 1200),
            ]
        },
        {
            "id": "WJA456", "callsign": "WestJet 456", "type": "B737",
            "waypoints": [
                (0,    -119.05, 50.30, 6000),
                (240,  -119.15, 50.15, 5200),
                (480,  -119.28, 50.02, 4200),
                (720,  -119.35, 49.98, 3300),
                (900,  -119.37, 49.96, 2600),
                (1020, -119.382, 49.958, 1900),
                (1140, -119.390, 49.955, 1200),
            ]
        },
    ]

    czml_data = build_czml(mock_flights, start, playback_multiplier=60)
    
    output_path = '../public/data/trajectories.czml'
    with open(output_path, 'w') as f:
        json.dump(czml_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated CZML for {len(mock_flights)} flights")
    print(f"  Output: {output_path}")
```

### 7.3 Frontend CZML Loading Hook

```typescript
// src/hooks/useCzmlLoader.ts
import { useEffect, useState } from 'react';
import * as Cesium from 'cesium';

interface CzmlState {
  isLoaded: boolean;
  flightIds: string[];
  error: string | null;
}

export function useCzmlLoader(
  viewer: Cesium.Viewer | null,
  czmlUrl: string
): CzmlState {
  const [state, setState] = useState<CzmlState>({
    isLoaded: false, flightIds: [], error: null,
  });

  useEffect(() => {
    if (!viewer) return;

    let dataSource: Cesium.CzmlDataSource;

    Cesium.CzmlDataSource.load(czmlUrl)
      .then((ds) => {
        dataSource = ds;
        viewer.dataSources.add(ds);

        // Synchronize Viewer clock with CZML time interval
        const clock = viewer.clock;
        clock.startTime = ds.clock.startTime.clone();
        clock.stopTime = ds.clock.stopTime.clone();
        clock.currentTime = ds.clock.startTime.clone();
        clock.clockRange = Cesium.ClockRange.LOOP_STOP;
        clock.multiplier = 60;
        clock.shouldAnimate = true;

        // Sync timeline display range
        viewer.timeline.zoomTo(clock.startTime, clock.stopTime);

        // Collect all flight IDs
        const ids = ds.entities.values
          .filter(e => e.id !== 'document')
          .map(e => e.id);

        // Default: track the first flight
        if (ids.length > 0) {
          viewer.trackedEntity = ds.entities.getById(ids[0]) ?? undefined;
        }

        setState({ isLoaded: true, flightIds: ids, error: null });
      })
      .catch((err) => {
        setState({ isLoaded: false, flightIds: [], error: err.message });
      });

    return () => {
      if (dataSource) viewer.dataSources.remove(dataSource, true);
    };
  }, [viewer, czmlUrl]);

  return state;
}
```

---

## 8. Phase 5: Control Panel & UI Integration

### 8.1 Global State Management

```typescript
// src/context/AppContext.tsx
import { createContext, useContext, useState, ReactNode } from 'react';
import * as Cesium from 'cesium';

interface AppState {
  viewer: Cesium.Viewer | null;
  setViewer: (v: Cesium.Viewer) => void;
  selectedFlightId: string | null;
  setSelectedFlightId: (id: string | null) => void;
  layers: {
    terrain: boolean;
    runways: boolean;
    waypoints: boolean;
    ocsSurfaces: boolean;
    trajectories: boolean;
  };
  toggleLayer: (key: keyof AppState['layers']) => void;
  playbackSpeed: number;
  setPlaybackSpeed: (speed: number) => void;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [viewer, setViewer] = useState<Cesium.Viewer | null>(null);
  const [selectedFlightId, setSelectedFlightId] = useState<string | null>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState(60);
  const [layers, setLayers] = useState({
    terrain: true,
    runways: true,
    waypoints: true,
    ocsSurfaces: true,
    trajectories: true,
  });

  const toggleLayer = (key: keyof typeof layers) => {
    setLayers(prev => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <AppContext.Provider value={{
      viewer, setViewer,
      selectedFlightId, setSelectedFlightId,
      layers, toggleLayer,
      playbackSpeed, setPlaybackSpeed,
    }}>
      {children}
    </AppContext.Provider>
  );
}

export const useApp = () => {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
};
```

### 8.2 Control Panel Component

```typescript
// src/components/ControlPanel.tsx
import { useApp } from '../context/AppContext';

const SPEED_OPTIONS = [
  { label: '1×', value: 1 },
  { label: '10×', value: 10 },
  { label: '30×', value: 30 },
  { label: '60×', value: 60 },
  { label: '120×', value: 120 },
];

export default function ControlPanel() {
  const { viewer, layers, toggleLayer, playbackSpeed, setPlaybackSpeed } = useApp();

  const handleSpeedChange = (speed: number) => {
    if (!viewer) return;
    viewer.clock.multiplier = speed;
    setPlaybackSpeed(speed);
  };

  const handlePlayPause = () => {
    if (!viewer) return;
    viewer.clock.shouldAnimate = !viewer.clock.shouldAnimate;
  };

  const handleReset = () => {
    if (!viewer) return;
    viewer.clock.currentTime = viewer.clock.startTime.clone();
    viewer.clock.shouldAnimate = false;
  };

  return (
    <div className="control-panel">
      <h3>AeroViz-4D</h3>

      {/* Playback controls */}
      <section>
        <button onClick={handlePlayPause}>Play / Pause</button>
        <button onClick={handleReset}>Reset</button>
        <div className="speed-buttons">
          {SPEED_OPTIONS.map(opt => (
            <button
              key={opt.value}
              className={playbackSpeed === opt.value ? 'active' : ''}
              onClick={() => handleSpeedChange(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </section>

      {/* Layer toggles */}
      <section>
        <h4>Layers</h4>
        {(Object.keys(layers) as Array<keyof typeof layers>).map(key => (
          <label key={key}>
            <input
              type="checkbox"
              checked={layers[key]}
              onChange={() => toggleLayer(key)}
            />
            {key}
          </label>
        ))}
      </section>
    </div>
  );
}
```

---

## 9. Data Interface Specifications

### 9.1 runway.geojson Specification

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon1,lat1],[lon2,lat2],[lon3,lat3],[lon4,lat4],[lon1,lat1]]]
      },
      "properties": {
        "airport_ident": "CYLW",
        "le_ident": "34",
        "he_ident": "16",
        "length_ft": 8000,
        "width_ft": 150,
        "surface": "ASP",
        "lighted": 1,
        "le_elevation_ft": 1421,
        "he_elevation_ft": 1398
      }
    }
  ]
}
```

### 9.2 waypoints.geojson Specification

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [lon, lat, alt_m]
      },
      "properties": {
        "name": "KEVOL",
        "type": "IAF",
        "min_alt_ft": 9000,
        "procedure": "RNAV(GNSS) Z RWY 34",
        "sequence": 1
      }
    }
  ]
}
```

### 9.3 Python–Frontend Data Contract Summary

| Data File | Producer | Consumer | Update Frequency |
|-----------|----------|----------|-----------------|
| `runway.geojson` | Python preprocessing script (one-time) | React frontend | At deployment |
| `waypoints.geojson` | Python preprocessing script (one-time) | React frontend | At deployment |
| `ocs_surfaces.geojson` | Python algorithm module | React frontend | Each research iteration |
| `trajectories.czml` | Python scheduling algorithm (core output) | React frontend | After each algorithm run |

---

## 10. Development Execution Roadmap

### 10.1 Recommended Execution Order

```
Week 1  ────────────────────────────────────────────────────────
  Day 1-2  Phase 1: Vite+React+CesiumJS setup, verify globe renders
  Day 3    Phase 2: Load World Terrain, verify mountain elevation display
  Day 4    Phase 2: Process OurAirports data, render ground-clamped runway polygons
  Day 5    Phase 3: Draw waypoint markers and approach path polylines

Week 2  ────────────────────────────────────────────────────────
  Day 6-7  Phase 3: Implement OCS geometry calculation and 3D rendering
  Day 8    Phase 4: Generate mock CZML with Python, load in frontend
  Day 9    Phase 4: Timeline sync, clock synchronization, camera tracking
  Day 10   Phase 5: Control panel UI, layer toggles, playback speed

Week 3+  ────────────────────────────────────────────────────────
  Integrate real scheduling algorithm output → replace mock CZML
  → capture thesis screenshots and live demo
```

### 10.2 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|---------|
| Viewer throws `AccessToken` error | Ion token not configured | Get a free token at cesium.com/ion/tokens |
| Runway polygons float above terrain | `clampToGround` not set | Add `clampToGround: true` to `GeoJsonDataSource.load()` options |
| Aircraft don't move after CZML loads | Clock not synced to DataSource | Manually set `viewer.clock.startTime` to CZML start time |
| 3D model not visible | `.glb` path wrong or not in `public/` | Confirm the model path is under `public/`; Vite serves static assets from there |
| Terrain loads slowly | Network latency | Disable terrain temporarily during development; re-enable for final demo |
| OCS surfaces clip through terrain | Inconsistent altitude datum | Confirm all altitudes are MSL (Mean Sea Level) in meters, not AGL |

---

## 11. Python Dependencies & Environment Setup

```
# python/requirements.txt
openap>=1.3.0          # aircraft performance model
pandas>=2.0.0          # data processing (OurAirports CSV)
numpy>=1.24.0          # numerical computation (coordinate transforms)
scipy>=1.10.0          # interpolation and optimization
fastapi>=0.100.0       # optional: RESTful API server
uvicorn>=0.23.0        # optional: FastAPI ASGI server
pyproj>=3.5.0          # geodetic coordinate projection
shapely>=2.0.0         # geometric computation (protection area polygons)
pulp>=2.7.0            # MILP solver (scheduling algorithm)
```

Installation:
```bash
cd python
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 12. References

| Resource | Link / Source |
|----------|--------------|
| CesiumJS Official Docs | https://cesium.com/learn/cesiumjs/ref-doc/ |
| CZML Format Specification | https://github.com/AnalyticalGraphicsInc/czml-writer/wiki/CZML-Guide |
| OurAirports Dataset | https://ourairports.com/data/ |
| OpenAP Performance Library | https://openap.dev/ |
| ICAO PANS-OPS Doc 8168 | ICAO official publication (purchase or institutional access required) |
| FAA CIFP Data | https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/cifp/ |
| Nav Canada AIP | https://www.navcanada.ca/en/aeronautical-information/ |
| Vite Configuration Docs | https://vitejs.dev/config/ |

---

*文档结束 / End of Document*

