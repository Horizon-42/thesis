import { describe, expect, it } from "vitest";
import { planTrajectoryModels } from "../trajectoryRenderModel";

/** Deterministic RNG so the random sampling is reproducible. */
function seededRng(seed = 0.42): () => number {
  let x = seed;
  return () => {
    x = (x * 9301 + 49297) % 233280;
    return x / 233280;
  };
}

const shown = Array.from({ length: 100 }, (_, i) => `f${i}`);

describe("planTrajectoryModels", () => {
  it("always includes the selected flight when it is shown", () => {
    const { modelIds } = planTrajectoryModels(shown, "f7", 20, seededRng());
    expect(modelIds.has("f7")).toBe(true);
  });

  it("caps the total at the budget", () => {
    const { modelIds } = planTrajectoryModels(shown, "f7", 20, seededRng());
    expect(modelIds.size).toBe(20);
  });

  it("excludes a selected flight that is not shown", () => {
    const { modelIds } = planTrajectoryModels(shown, "ghost", 20, seededRng());
    expect(modelIds.has("ghost")).toBe(false);
    expect(modelIds.size).toBe(20);
  });

  it("gives every shown entity a model when there are fewer than the budget", () => {
    const few = ["a", "b", "c"];
    const { modelIds } = planTrajectoryModels(few, null, 20, seededRng());
    expect(modelIds).toEqual(new Set(few));
  });

  it("returns only the selected flight when the budget is zero", () => {
    expect(planTrajectoryModels(shown, "f3", 0, seededRng()).modelIds).toEqual(new Set(["f3"]));
    expect(planTrajectoryModels(shown, null, 0, seededRng()).modelIds.size).toBe(0);
  });
});
