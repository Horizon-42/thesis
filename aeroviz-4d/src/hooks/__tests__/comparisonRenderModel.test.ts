import { describe, expect, it } from "vitest";
import * as Cesium from "cesium";
import {
  applyComparisonRenderModel,
  availabilityByEntityId,
  captureObservedEntityStyle,
  isComparisonEntity,
  kindOfEntityId,
  restoreObservedEntityStyle,
} from "../useComparisonTrajectoryLayer";
import { COMPARISON_KIND_COLORS, COMPARISON_KIND_ALPHA } from "../../utils/trajectoryRenderModel";

/**
 * Which paths get repainted from the legend, and which keep the colour the CZML baked in.
 *
 * The rule is not "off-target groups keep their colour" — it is "paths carrying a baked
 * VERDICT colour keep it". Those are the optimizer/simulator paths of an off-target group
 * (bright yellow). The reference comes from the canonical observed datasource and is
 * styled separately. Predictions never get a verdict bake, so they must be repainted from
 * the legend even though their status is "offTarget": a forecast essentially always
 * misses the 106.75 m gate, and if the skip applied to them the rendered colour would
 * silently depend on the Python builder's PREDICTION_COLOR matching the TypeScript
 * legend's.
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

  it("repaints the lookback from the legend even though its status is offTarget", () => {
    // Same rule as the prediction it belongs to: the builder bakes no verdict colour onto
    // prediction-schema entities, so there is nothing to preserve.
    const e = entity("look-AAL542_05L", "offTarget");
    applyComparisonRenderModel(e, new Set());
    expectLegendColor(e, "lookback");
  });

  it("draws the lookback in the prediction's hue but faded", () => {
    // One continuous track: the input half is told apart from the forecast half by alpha, not
    // by colour — so the hue must MATCH and the alpha must not.
    const e = entity("look-X_05L", "solved");
    applyComparisonRenderModel(e, new Set());
    expectLegendColor(e, "predicted");
    expect(renderedColor(e).alpha).toBeCloseTo(COMPARISON_KIND_ALPHA.lookback);
    expect(COMPARISON_KIND_ALPHA.lookback).toBeLessThan(COMPARISON_KIND_ALPHA.predicted);
  });

  it("gives the lookback no aircraft model — the reference already flies that span", () => {
    // The lookback retraces observed samples the reference track covers exactly, so a model
    // here would draw a second aircraft on top of the reference's for the whole input window.
    const e = new Cesium.Entity({
      id: "look-X_05L",
      position: new Cesium.ConstantPositionProperty(Cesium.Cartesian3.fromDegrees(-78.4, 35.7, 900)),
      point: new Cesium.PointGraphics({ pixelSize: 9 }),
      path: new Cesium.PathGraphics({ material: new Cesium.ColorMaterialProperty(BAKED) }),
      properties: new Cesium.PropertyBag({ status: "solved" }),
    });
    applyComparisonRenderModel(e, new Set(["look-X_05L"]));   // sampled — a pred- would get one
    expect(e.model).toBeUndefined();
    expect(e.point!.show!.getValue(Cesium.JulianDate.now())).toBe(false);
  });
});

describe("comparison entity ids", () => {
  it("maps every result prefix to its kind", () => {
    expect(kindOfEntityId("opt-X_05L")).toBe("optimizer");
    expect(kindOfEntityId("sim-X_05L")).toBe("simulator");
    expect(kindOfEntityId("pred-X_05L")).toBe("predicted");
    expect(kindOfEntityId("look-X_05L")).toBe("lookback");
  });

  it("recognises every comparison prefix as pickable", () => {
    // The picker's prefix list once omitted `pred-`, so prediction tracks silently could not
    // be hovered for their callsign. Both readers now share one prefix table.
    for (const id of ["opt-X_05L", "sim-X_05L", "pred-X_05L", "look-X_05L"]) {
      expect(isComparisonEntity(new Cesium.Entity({ id }))).toBe(true);
    }
    // References are canonical observed entities, not embedded ref-* comparison packets.
    expect(isComparisonEntity(new Cesium.Entity({ id: "ref-X_05L" }))).toBe(false);
    expect(isComparisonEntity(new Cesium.Entity({ id: "AAL542_05L_a15c80_20260701T033111Z" })))
      .toBe(false);
    expect(isComparisonEntity(undefined)).toBe(false);
  });
});

describe("comparison entity availability", () => {
  it("keeps predictor input available through the matching forecast", () => {
    const epoch = "2026-04-01T08:00:00Z";
    const availability = availabilityByEntityId([
      {
        id: "look-X_05L",
        position: {
          epoch,
          cartographicDegrees: [0, -78.4, 35.7, 900, 4, -78.3, 35.6, 800],
        },
      },
      {
        id: "pred-X_05L",
        position: {
          epoch,
          cartographicDegrees: [4, -78.3, 35.6, 800, 9, -78.2, 35.5, 700],
        },
      },
      {
        id: "sim-Y_05L",
        position: {
          epoch,
          cartographicDegrees: [0, -78.5, 35.8, 1000, 3, -78.4, 35.7, 900],
        },
      },
    ]);

    const start = Cesium.JulianDate.fromIso8601(epoch);
    expect(Cesium.JulianDate.secondsDifference(availability.get("look-X_05L")!.stop, start))
      .toBe(9);
    expect(Cesium.JulianDate.secondsDifference(availability.get("pred-X_05L")!.stop, start))
      .toBe(9);
    expect(Cesium.JulianDate.secondsDifference(availability.get("sim-Y_05L")!.stop, start))
      .toBe(3);
  });
});

describe("canonical observed style lifetime", () => {
  it("restores the exact pre-comparison property objects", () => {
    const originalMaterial = new Cesium.ColorMaterialProperty(Cesium.Color.BLUE);
    const originalWidth = new Cesium.ConstantProperty(2);
    const originalLabelColor = new Cesium.ConstantProperty(Cesium.Color.WHITE);
    const originalLabelShow = new Cesium.ConstantProperty(true);
    const originalAnimation = new Cesium.ConstantProperty(true);
    const observed = new Cesium.Entity({
      id: "AFR074_05L",
      show: true,
      path: new Cesium.PathGraphics({ material: originalMaterial, width: originalWidth }),
      label: new Cesium.LabelGraphics({
        fillColor: originalLabelColor,
        show: originalLabelShow,
      }),
      model: new Cesium.ModelGraphics({ runAnimations: originalAnimation }),
    });
    const snapshot = captureObservedEntityStyle(observed);

    observed.show = false;
    observed.path!.material = new Cesium.ColorMaterialProperty(Cesium.Color.RED);
    observed.path!.width = new Cesium.ConstantProperty(9);
    observed.label!.fillColor = new Cesium.ConstantProperty(Cesium.Color.YELLOW);
    observed.label!.show = new Cesium.ConstantProperty(false);
    observed.model!.runAnimations = new Cesium.ConstantProperty(false);

    restoreObservedEntityStyle(observed, snapshot);

    expect(observed.show).toBe(true);
    expect(observed.path!.material).toBe(originalMaterial);
    expect(observed.path!.width).toBe(originalWidth);
    expect(observed.label!.fillColor).toBe(originalLabelColor);
    expect(observed.label!.show).toBe(originalLabelShow);
    expect(observed.model!.runAnimations).toBe(originalAnimation);
  });
});
