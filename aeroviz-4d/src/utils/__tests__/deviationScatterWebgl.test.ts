import { describe, expect, it } from "vitest";
import { deviationScatterColor } from "../deviationScatterWebgl";

describe("deviationScatterColor", () => {
  it("keeps indeterminate distinct from pass and fail", () => {
    expect(deviationScatterColor("indeterminate")).not.toEqual(
      deviationScatterColor("fail"),
    );
    expect(deviationScatterColor("indeterminate")).not.toEqual(
      deviationScatterColor("pass"),
    );
  });
});
