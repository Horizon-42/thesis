# Terrain Loading Stability

This note explains why Cesium terrain can look unstable while changing camera
height or vertical exaggeration, and documents the tuning used by AeroViz-4D.

## Problem

Cesium terrain is streamed as a level-of-detail tile hierarchy. When the camera
zooms, pans, or changes height quickly, Cesium refines from coarse parent tiles
to more detailed child tiles. That can create visible pop-in, especially when:

- satellite imagery is hidden and the globe uses a flat gray material
- `scene.verticalExaggeration` is high
- the camera is close to the ground
- the airport region is relatively flat and small terrain deltas are magnified
- the terrain provider is recreated after every toggle

This is expected behavior for streamed terrain, but it can be made less
disruptive.

## Current Tuning

`src/hooks/useTerrainLayer.ts` applies these settings when Cesium World Terrain
is active:

| Setting | Value | Purpose |
| --- | ---: | --- |
| `globe.maximumScreenSpaceError` | `1` | Requests higher-detail terrain before tiles become visibly coarse. |
| `globe.tileCacheSize` | at least `512` | Keeps more tiles resident during zoom in/out changes. |
| `globe.loadingDescendantLimit` | `20` | Balances quick coarse feedback against delayed all-at-once refinement. |
| `globe.preloadAncestors` | `true` | Improves zoom-out continuity. |
| `globe.preloadSiblings` | `true` | Improves pan and newly exposed area continuity. |

The hook also caches the loaded World Terrain provider instead of recreating it
on every terrain toggle.

`src/components/HUD.tsx` also avoids applying every slider movement directly to
Cesium. The UI updates immediately, but the real
`scene.verticalExaggeration` value is committed after a short debounce or when
the pointer is released. This reduces repeated geometry rebuilds while dragging.

For airport-scoped local terrain packages, see
[`docs/24-airport-local-heightmap-terrain.md`](24-airport-local-heightmap-terrain.md).

## Terrain Load Status

The HUD reads `viewer.scene.globe.tileLoadProgressEvent` and displays:

- `Ready` when the terrain queue is empty
- `Refining N` while Cesium is still loading or refining `N` terrain tiles

If terrain looks wrong while `Refining N` is shown, wait for `Ready` before
judging geometry stability.

## Recommended Operating Range

For Cesium World Terrain:

- `1x` to `5x`: stable for general inspection
- `5x` to `20x`: useful for seeing low relief, but LOD pop-in is more visible
- above `20x`: not recommended for World Terrain in the main viewer

For local DSM terrain, higher exaggeration can be useful for debugging, but
source noise and no-data edges are also magnified.

## Debug Checklist

When terrain appears unstable:

1. Check the HUD terrain load status. If it says `Refining N`, wait for `Ready`.
2. Lower terrain exaggeration to `1x` and verify whether the instability remains.
3. Confirm only one terrain source is active: World Terrain or DSM terrain.
4. Verify the terrain provider is not being recreated by repeated layer toggles.
5. Increase `tileCacheSize` only if memory is available and zooming repeatedly
   revisits the same area.
6. Lower `maximumScreenSpaceError` only if network/GPU budget can support
   higher-detail terrain.

## Tradeoffs

More stable terrain costs more memory, network traffic, and GPU time. The current
settings favor visual stability for analysis over maximum rendering performance.
