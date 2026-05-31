/**
 * useTerrainLayer.test.ts
 * -----------------------
 * Verifies the terrain toggle hook:
 *   1. First mount with terrain ON → loads world terrain and tunes streaming
 *   2. Toggle OFF → sets EllipsoidTerrainProvider
 *   3. Toggle ON  → calls CesiumTerrainProvider.fromIonAssetId and applies tuned settings
 *   4. Multiple toggles → reuses the loaded world terrain provider
 *   5. Rapid toggle (ON before async completes) → cancelled, no stale write
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

// ── Hoisted mocks (available inside vi.mock factories) ───────────────────────
const { fromIonAssetId, mockWorldTerrainProvider, mockViewer, getTerrainFlag, setTerrainFlag } =
  vi.hoisted(() => {
    const mockWorldTerrainProvider = { _tag: "world-terrain" } as any;
    const fromIonAssetId = vi.fn(() => Promise.resolve(mockWorldTerrainProvider));
    const mockViewer = {
      scene: {
        terrainProvider: { _tag: "initial-world-terrain" } as any,
        globe: {
          maximumScreenSpaceError: 2,
          tileCacheSize: 100,
          loadingDescendantLimit: 10,
          preloadAncestors: false,
          preloadSiblings: false,
        },
        requestRender: vi.fn(),
      },
      isDestroyed: () => false,
    };
    let terrainFlag = true;
    return {
      fromIonAssetId,
      mockWorldTerrainProvider,
      mockViewer,
      getTerrainFlag: () => terrainFlag,
      setTerrainFlag: (v: boolean) => { terrainFlag = v; },
    };
  });

// ── Mock Cesium ──────────────────────────────────────────────────────────────
vi.mock("cesium", () => ({
  EllipsoidTerrainProvider: class EllipsoidTerrainProvider {
    _tag = "ellipsoid";
  },
  CesiumTerrainProvider: {
    fromIonAssetId,
  },
}));

// ── Mock AppContext ──────────────────────────────────────────────────────────
vi.mock("../../context/AppContext", () => ({
  useApp: () => ({
    viewer: mockViewer,
    layers: { terrain: getTerrainFlag() },
  }),
}));

// ── Import under test (after mocks are registered) ──────────────────────────
import { useTerrainLayer } from "../useTerrainLayer";

// ── Helpers ──────────────────────────────────────────────────────────────────
const flushPromises = () => act(() => new Promise((r) => setTimeout(r, 0)));

describe("useTerrainLayer", () => {
  beforeEach(() => {
    setTerrainFlag(true);
    mockViewer.scene.terrainProvider = { _tag: "initial-world-terrain" } as any;
    mockViewer.scene.globe.maximumScreenSpaceError = 2;
    mockViewer.scene.globe.tileCacheSize = 100;
    mockViewer.scene.globe.loadingDescendantLimit = 10;
    mockViewer.scene.globe.preloadAncestors = false;
    mockViewer.scene.globe.preloadSiblings = false;
    mockViewer.scene.requestRender.mockClear();
    fromIonAssetId.mockClear();
    fromIonAssetId.mockImplementation(() => Promise.resolve(mockWorldTerrainProvider));
  });

  it("loads and tunes world terrain when terrain is ON", async () => {
    renderHook(() => useTerrainLayer());
    await flushPromises();

    expect(fromIonAssetId).toHaveBeenCalledWith(1, {
      requestVertexNormals: true,
      requestWaterMask: true,
    });
    expect(mockViewer.scene.terrainProvider).toBe(mockWorldTerrainProvider);
    expect(mockViewer.scene.globe.maximumScreenSpaceError).toBe(1);
    expect(mockViewer.scene.globe.tileCacheSize).toBe(512);
    expect(mockViewer.scene.globe.loadingDescendantLimit).toBe(20);
    expect(mockViewer.scene.globe.preloadAncestors).toBe(true);
    expect(mockViewer.scene.globe.preloadSiblings).toBe(true);
  });

  it("sets EllipsoidTerrainProvider when toggled OFF", () => {
    const { rerender } = renderHook(() => useTerrainLayer());

    setTerrainFlag(false);
    rerender();

    expect(mockViewer.scene.terrainProvider._tag).toBe("ellipsoid");
    expect(mockViewer.scene.globe.maximumScreenSpaceError).toBe(2);
    expect(mockViewer.scene.globe.tileCacheSize).toBe(100);
    expect(mockViewer.scene.globe.loadingDescendantLimit).toBe(10);
    expect(mockViewer.scene.globe.preloadAncestors).toBe(false);
    expect(mockViewer.scene.globe.preloadSiblings).toBe(false);
    expect(fromIonAssetId).toHaveBeenCalledTimes(1);
  });

  it("restores world terrain via fromIonAssetId when toggled back ON", async () => {
    setTerrainFlag(false);
    const { rerender } = renderHook(() => useTerrainLayer());

    expect(mockViewer.scene.terrainProvider._tag).toBe("ellipsoid");

    // ON
    setTerrainFlag(true);
    rerender();
    await flushPromises();

    expect(fromIonAssetId).toHaveBeenCalledWith(1, {
      requestVertexNormals: true,
      requestWaterMask: true,
    });
    expect(mockViewer.scene.terrainProvider).toBe(mockWorldTerrainProvider);
    expect(mockViewer.scene.globe.maximumScreenSpaceError).toBe(1);
    expect(mockViewer.scene.globe.tileCacheSize).toBe(512);
    expect(mockViewer.scene.globe.loadingDescendantLimit).toBe(20);
    expect(mockViewer.scene.globe.preloadAncestors).toBe(true);
    expect(mockViewer.scene.globe.preloadSiblings).toBe(true);
  });

  it("reuses the loaded world terrain provider across multiple ON→OFF→ON cycles", async () => {
    setTerrainFlag(false);
    const { rerender } = renderHook(() => useTerrainLayer());

    for (let i = 0; i < 3; i++) {
      expect(mockViewer.scene.terrainProvider._tag).toBe("ellipsoid");

      // ON
      setTerrainFlag(true);
      rerender();
      await flushPromises();
      expect(mockViewer.scene.terrainProvider).toBe(mockWorldTerrainProvider);

      // OFF again for the next cycle
      setTerrainFlag(false);
      rerender();
    }

    expect(fromIonAssetId).toHaveBeenCalledTimes(1);
  });

  it("cancels pending async restore if toggled OFF before it completes", async () => {
    let resolveProvider!: (v: any) => void;
    fromIonAssetId.mockImplementationOnce(
      () => new Promise((r) => { resolveProvider = r; }),
    );

    setTerrainFlag(false);
    const { rerender } = renderHook(() => useTerrainLayer());

    // ON starts async load
    setTerrainFlag(true);
    rerender();

    // Before async resolves, toggle OFF again
    setTerrainFlag(false);
    rerender();
    expect(mockViewer.scene.terrainProvider._tag).toBe("ellipsoid");

    // Resolve the stale async — should NOT overwrite the ellipsoid
    resolveProvider({ _tag: "stale-world-terrain" });
    await flushPromises();

    expect(mockViewer.scene.terrainProvider._tag).toBe("ellipsoid");
  });
});
