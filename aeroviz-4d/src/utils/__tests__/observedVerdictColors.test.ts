import { describe, expect, it } from "vitest";

import {
  OBSERVED_FITTED_TAIL_COLORS,
  OBSERVED_FITTED_TAIL_OUTLINE_COLORS,
  OBSERVED_VERDICT_COLORS,
} from "../observedVerdictColors";

describe("colours", () => {
  it("gives the three states distinct colours", () => {
    const values = Object.values(OBSERVED_VERDICT_COLORS);
    expect(new Set(values).size).toBe(values.length);
  });

  it("gives fitted tails three distinct mixed fill and outline colours", () => {
    expect(new Set(Object.values(OBSERVED_FITTED_TAIL_COLORS)).size).toBe(3);
    expect(new Set(Object.values(OBSERVED_FITTED_TAIL_OUTLINE_COLORS)).size).toBe(3);
    expect(OBSERVED_FITTED_TAIL_COLORS.fail).toBe("rgba(186, 117, 135, 0.58)");
  });
});
