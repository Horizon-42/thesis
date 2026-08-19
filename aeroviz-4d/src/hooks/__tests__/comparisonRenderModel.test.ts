import { describe, expect, it } from "vitest";
import * as Cesium from "cesium";
import {
  applyComparisonRenderModel,
  applyComparisonReferenceRenderModel,
  availabilityByEntityId,
  isComparisonEntity,
  kindOfEntityId,
} from "../useComparisonTrajectoryLayer";
import { COMPARISON_KIND_COLORS, COMPARISON_KIND_ALPHA } from "../../utils/trajectoryRenderModel";
import { OBSERVED_VERDICT_COLORS } from "../../utils/observedVerdictColors";

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

function rgbOf(color: Cesium.Color): [number, number, number] {
  return [color.red, color.green, color.blue];
}

/** RGB only — the hook applies its own path alpha to the legend's opaque swatch colour. */
function expectLegendColor(e: Cesium.Entity, kind: keyof typeof COMPARISON_KIND_COLORS): void {
  expect(rgbOf(renderedColor(e)))
    .toEqual(rgbOf(Cesium.Color.fromCssColorString(COMPARISON_KIND_COLORS[kind])));
}

/** A verdict hue from the shared Baseline palette, whatever alpha the kind renders at. */
function expectVerdictColor(e: Cesium.Entity, css: string): void {
  expect(rgbOf(renderedColor(e))).toEqual(rgbOf(Cesium.Color.fromCssColorString(css)));
}

describe("applyComparisonRenderModel path colouring", () => {
  it("draws a failed prediction red like a failed baseline", () => {
    const e = entity("pred-AAL542_05L", "offTarget");
    applyComparisonRenderModel(e, new Set());
    const expected = Cesium.Color.fromCssColorString(OBSERVED_VERDICT_COLORS.fail);
    const actual = renderedColor(e);
    expect([actual.red, actual.green, actual.blue])
      .toEqual([expected.red, expected.green, expected.blue]);
  });

  it("draws a passing prediction green like a passing baseline", () => {
    const e = entity("pred-AAL542_05L", "solved");
    applyComparisonRenderModel(e, new Set());
    const expected = Cesium.Color.fromCssColorString(OBSERVED_VERDICT_COLORS.pass);
    const actual = renderedColor(e);
    expect([actual.red, actual.green, actual.blue])
      .toEqual([expected.red, expected.green, expected.blue]);
  });

  it("draws an indeterminate prediction gray like an undecided baseline", () => {
    const e = entity("pred-AAL542_05L", "indeterminate");
    applyComparisonRenderModel(e, new Set());
    const expected = Cesium.Color.fromCssColorString(OBSERVED_VERDICT_COLORS.undecided);
    const actual = renderedColor(e);
    expect([actual.red, actual.green, actual.blue])
      .toEqual([expected.red, expected.green, expected.blue]);
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

  it("repaints a failed lookback red, like the forecast it feeds", () => {
    // Same rule as the prediction it belongs to: the builder bakes no verdict colour onto
    // prediction-schema entities, so there is nothing to preserve — and the input window
    // must not keep the fallback purple while its forecast turns red.
    const e = entity("look-AAL542_05L", "offTarget");
    applyComparisonRenderModel(e, new Set());
    expectVerdictColor(e, OBSERVED_VERDICT_COLORS.fail);
  });

  it("draws the lookback in the prediction's hue but faded", () => {
    // One continuous track: the input half is told apart from the forecast half by alpha, not
    // by colour — so the hue must MATCH and the alpha must not. A purple input in front of a
    // green forecast reads as a third kind of path, which is what this pins down.
    const pass = entity("look-X_05L", "solved");
    applyComparisonRenderModel(pass, new Set());
    expectVerdictColor(pass, OBSERVED_VERDICT_COLORS.pass);
    expect(renderedColor(pass).alpha).toBeCloseTo(COMPARISON_KIND_ALPHA.lookback);
    expect(COMPARISON_KIND_ALPHA.lookback).toBeLessThan(COMPARISON_KIND_ALPHA.predicted);

    const forecast = entity("pred-X_05L", "solved");
    applyComparisonRenderModel(forecast, new Set());
    const [lr, lg, lb] = rgbOf(renderedColor(pass));
    const [pr, pg, pb] = rgbOf(renderedColor(forecast));
    expect([lr, lg, lb]).toEqual([pr, pg, pb]);
  });

  it("falls back to the faded shared purple when the group has no verdict", () => {
    // Only the fallback is a hue of its own, and the prediction fallback is the same one.
    const e = entity("look-X_05L", "unknown");
    applyComparisonRenderModel(e, new Set());
    expectLegendColor(e, "lookback");
    expect(renderedColor(e).alpha).toBeCloseTo(COMPARISON_KIND_ALPHA.lookback);
  });

  it("takes the indexed group status over the entity's own for the lookback", () => {
    // `look-` and `pred-` share the group key, so one index entry colours both halves.
    const e = entity("look-X_05L", "solved");
    applyComparisonRenderModel(e, new Set(), new Map([["X_05L", "offTarget" as const]]));
    expectVerdictColor(e, OBSERVED_VERDICT_COLORS.fail);
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

describe("comparison references", () => {
  it("keeps exact observed references white regardless of prediction outcome", () => {
    const observed = entity("AFR074_05L", "offTarget");
    applyComparisonReferenceRenderModel(observed, new Set());

    expectLegendColor(observed, "reference");
  });
});
