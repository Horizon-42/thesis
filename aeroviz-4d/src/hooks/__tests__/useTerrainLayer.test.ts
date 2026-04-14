/**
 * useTerrainLayer.test.ts
 * -----------------------
 * Verifies the terrain toggle hook:
 *   1. First mount with terrain ON → does NOT touch the terrain provider
 *   2. Toggle OFF → sets EllipsoidTerrainProvider
 *   3. Toggle ON  → calls CesiumTerrainProvider.fromIonAssetId and applies the result
 *   4. Multiple toggles → each one applies correctly
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
    fromIonAssetId.mockClear();
    fromIonAssetId.mockImplementation(() => Promise.resolve(mockWorldTerrainProvider));
  });

  it("does NOT change terrain provider on initial mount when terrain is ON", () => {
    renderHook(() => useTerrainLayer());

    expect(mockViewer.scene.terrainProvider._tag).toBe("initial-world-terrain");
    expect(fromIonAssetId).not.toHaveBeenCalled();
  });

  it("sets EllipsoidTerrainProvider when toggled OFF", () => {
    const { rerender } = renderHook(() => useTerrainLayer());

    setTerrainFlag(false);
    rerender();

    expect(mockViewer.scene.terrainProvider._tag).toBe("ellipsoid");
    expect(fromIonAssetId).not.toHaveBeenCalled();
  });

  it("restores world terrain via fromIonAssetId when toggled back ON", async () => {
    const { rerender } = renderHook(() => useTerrainLayer());

    // OFF
    setTerrainFlag(false);
    rerender();
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
  });

  it("handles multiple ON→OFF→ON cycles", async () => {
    const { rerender } = renderHook(() => useTerrainLayer());

    for (let i = 0; i < 3; i++) {
      // OFF
      setTerrainFlag(false);
      rerender();
      expect(mockViewer.scene.terrainProvider._tag).toBe("ellipsoid");

      // ON
      setTerrainFlag(true);
      rerender();
      await flushPromises();
      expect(mockViewer.scene.terrainProvider).toBe(mockWorldTerrainProvider);
    }

    expect(fromIonAssetId).toHaveBeenCalledTimes(3);
  });

  it("cancels pending async restore if toggled OFF before it completes", async () => {
    let resolveProvider!: (v: any) => void;
    fromIonAssetId.mockImplementationOnce(
      () => new Promise((r) => { resolveProvider = r; }),
    );

    const { rerender } = renderHook(() => useTerrainLayer());

    // OFF → ON (starts async load)
    setTerrainFlag(false);
    rerender();
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
