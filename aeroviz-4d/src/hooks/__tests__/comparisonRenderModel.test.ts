import { describe, expect, it } from "vitest";
import * as Cesium from "cesium";
import { applyComparisonRenderModel } from "../useComparisonTrajectoryLayer";
import { COMPARISON_KIND_COLORS } from "../../utils/trajectoryRenderModel";

/**
 * Which paths get repainted from the legend, and which keep the colour the CZML baked in.
 *
 * The rule is not "off-target groups keep their colour" — it is "paths carrying a baked
 * VERDICT colour keep it". Those are the reference (always) and the optimizer/simulator
 * paths of an off-target group (bright yellow). Predictions never get a verdict bake, so
 * they must be repainted from the legend even though their status is "offTarget": a
 * forecast essentially always misses the 106.75 m gate, and if the skip applied to them
 * the rendered colour would silently depend on the Python builder's PREDICTION_COLOR
 * matching the TypeScript legend's.
 */

const BAKED = Cesium.Color.fromBytes(1, 2, 3, 255);   // a colour no legend entry uses

function entity(id: string, status: string): Cesium.Entity {
  return new Cesium.Entity({
    id,
    path: new Cesium.PathGraphics({ material: new Cesium.ColorMaterialProperty(BAKED) }),
    properties: new Cesium.PropertyBag({ status }),
  });
}

function renderedColor(e: Cesium.Entity): Cesium.Color {
  const material = e.path?.material as Cesium.ColorMaterialProperty;
  return material.color!.getValue(Cesium.JulianDate.now());
}

/** RGB only — the hook applies its own path alpha to the legend's opaque swatch colour. */
function expectLegendColor(e: Cesium.Entity, kind: keyof typeof COMPARISON_KIND_COLORS): void {
  const actual = renderedColor(e);
  const expected = Cesium.Color.fromCssColorString(COMPARISON_KIND_COLORS[kind]);
  expect([actual.red, actual.green, actual.blue])
    .toEqual([expected.red, expected.green, expected.blue]);
}

describe("applyComparisonRenderModel path colouring", () => {
  it("repaints a prediction from the legend even though its status is offTarget", () => {
    const e = entity("pred-AAL542_05L", "offTarget");
    applyComparisonRenderModel(e, new Set());
    expectLegendColor(e, "predicted");
  });

  it("repaints optimizer and simulator paths from the legend when on target", () => {
    for (const [id, kind] of [["opt-X_05L", "optimizer"], ["sim-X_05L", "simulator"]] as const) {
      const e = entity(id, "solved");
      applyComparisonRenderModel(e, new Set());
      expectLegendColor(e, kind);
    }
  });

  it("keeps the baked yellow on an off-target simulator path", () => {
    // The marking has to sit on the trajectory that missed, not just the reference.
    const e = entity("sim-X_05L", "offTarget");
    applyComparisonRenderModel(e, new Set());
    expect(renderedColor(e)).toEqual(BAKED);
  });

  it("never repaints the reference, whatever its status", () => {
    for (const status of ["solved", "offTarget", "failed"]) {
      const e = entity("ref-X_05L", status);
      applyComparisonRenderModel(e, new Set());
      expect(renderedColor(e)).toEqual(BAKED);
    }
  });
});
