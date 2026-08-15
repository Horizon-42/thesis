import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";

const {
  appState,
  entities,
  fetchJsonMock,
  ColorMaterialProperty,
} = vi.hoisted(() => {
  class ConstantProperty {
    constructor(private readonly value: unknown) {}

    getValue() {
      return this.value;
    }
  }

  class ColorMaterialProperty {
    color: ConstantProperty;

    constructor(color: unknown) {
      this.color = new ConstantProperty(color);
    }
  }

  const entities: Array<{
    id: string;
    path: { material: InstanceType<typeof ColorMaterialProperty> };
  }> = [];

  return {
    entities,
    fetchJsonMock: vi.fn(),
    ColorMaterialProperty,
    appState: {
      viewer: {},
      activeAirportCode: "KRDU",
      trajectoryDataSource: {
        entities: { values: entities },
      },
    },
  };
});

vi.mock("cesium", () => ({
  Color: {
    fromCssColorString: (value: string) => value,
  },
  ConstantProperty: class ConstantProperty {
    constructor(private readonly value: unknown) {}

    getValue() {
      return this.value;
    }
  },
  ColorMaterialProperty,
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => appState,
}));

vi.mock("../../utils/fetchJson", () => ({
  fetchJson: fetchJsonMock,
  isMissingJsonAsset: () => false,
}));

import { OBSERVED_VERDICT_COLORS } from "../../utils/observedVerdictColors";
import { useObservedVerdictColors } from "../useObservedVerdictColors";

describe("useObservedVerdictColors", () => {
  beforeEach(() => {
    entities.splice(
      0,
      entities.length,
      {
        id: "MATCHED_05L_abc123_20260701T000000Z",
        path: { material: new ColorMaterialProperty("original") },
      },
      {
        id: "NO_TCH_32_def456_20260701T010000Z",
        path: { material: new ColorMaterialProperty("original") },
      },
    );
    fetchJsonMock.mockReset();
    fetchJsonMock.mockResolvedValue({
      schema_version: "terminal-approach-evaluation-v3",
      methodology: {
        terminal_vertical: {
          reference: "LTP elevation MSL + published FAS TCH",
          trajectory_altitude_datum: "msl",
          target_context_tolerance_m: 0.01,
          lpv: {
            scale_model: "do229_lpv_angular_min_clamped",
            one_sided_minimum_fsd_m: 15,
            normal_fsd_fraction: 0.5,
            effective_threshold_bound_m: 7.5,
            sources: [
              { document: "ICAO Doc 9613", location: "§5.3.3.1.1.1(b)", use: "half FSD" },
            ],
          },
        },
      },
      assessment_contexts: [],
      subject: "observed",
      total: 1,
      measured: 1,
      solved: 1,
      solve_rate: 1,
      successful: 1,
      failed: 0,
      indeterminate: 0,
      verdict_counts: { pass: 1, fail: 0, indeterminate: 0 },
      success_rate: 1,
      lateral_m: null,
      vertical_m: null,
      final_time_s: null,
      reference: null,
      trajectories: [
        {
          id: "MATCHED",
          file: null,
          solved: true,
          success: true,
          verdict: "pass",
          event_status: "estimated",
          lateral_result: "pass",
          vertical_result: "pass",
          violations: [],
          flight_key: "MATCHED_05L_abc123_20260701T000000Z",
          subject: "observed",
          airport: "KRDU",
          runway: "05L",
          benchmark: "lpv",
          bounds: { guidance_lateral_m: 53.375, runway_lateral_m: 22.86,
            effective_lateral_m: 22.86, vertical_lower_m: null, vertical_upper_m: null },
        },
      ],
    });
  });

  it("reads the fixed observed report and paints every track with the three-state palette", async () => {
    const { result } = renderHook(() => useObservedVerdictColors(true));

    await waitFor(() => expect(result.current.counts).not.toBeNull());

    expect(fetchJsonMock).toHaveBeenCalledTimes(1);
    expect(fetchJsonMock).toHaveBeenCalledWith(
      "/data/airports/KRDU/comparison/observed/evaluation_report.json",
    );
    expect(entities[0].path.material.color.getValue()).toBe(
      OBSERVED_VERDICT_COLORS.pass,
    );
    expect(entities[1].path.material.color.getValue()).toBe(
      OBSERVED_VERDICT_COLORS.undecided,
    );
    expect(result.current.counts).toEqual({
      pass: 1,
      fail: 0,
      undecided: 1,
    });
    expect([...result.current.verdictsByFlightId!.entries()]).toEqual([
      ["MATCHED_05L_abc123_20260701T000000Z", "pass"],
      ["NO_TCH_32_def456_20260701T010000Z", "undecided"],
    ]);
    expect(result.current.matched).toBe(1);
    expect(result.current.total).toBe(2);
  });

  it("prepares verdicts while inactive without repainting comparison-owned entities", async () => {
    const { result } = renderHook(() => useObservedVerdictColors(false));

    await waitFor(() => expect(result.current.verdictsByFlightId).not.toBeNull());

    expect(fetchJsonMock).toHaveBeenCalledTimes(1);
    expect(entities[0].path.material.color.getValue()).toBe("original");
    expect(entities[1].path.material.color.getValue()).toBe("original");
  });
});
