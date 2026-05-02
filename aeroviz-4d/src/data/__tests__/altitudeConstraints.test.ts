import { describe, expect, it } from "vitest";
import {
  altitudeConstraintClassName,
  altitudeConstraintLabel,
  altitudeConstraintReferenceFt,
  altitudeConstraintText,
} from "../altitudeConstraints";

describe("altitude constraint display helpers", () => {
  it("formats exact constraints", () => {
    const constraint = { kind: "AT" as const, minFtMsl: 3000, maxFtMsl: 3000 };

    expect(altitudeConstraintReferenceFt(constraint)).toBe(3000);
    expect(altitudeConstraintText(constraint)).toBe("AT 3,000 ft");
    expect(altitudeConstraintLabel("FAF", constraint)).toBe("FAF AT 3,000 ft");
    expect(altitudeConstraintClassName(constraint)).toBe("is-at");
  });

  it("uses the lower bound for at-or-above constraints", () => {
    const constraint = { kind: "AT_OR_ABOVE" as const, minFtMsl: 5200 };

    expect(altitudeConstraintReferenceFt(constraint)).toBe(5200);
    expect(altitudeConstraintText(constraint)).toBe(">= 5,200 ft");
    expect(altitudeConstraintClassName(constraint)).toBe("is-at-or-above");
  });

  it("uses the upper bound for at-or-below constraints", () => {
    const constraint = { kind: "AT_OR_BELOW" as const, maxFtMsl: 3900 };

    expect(altitudeConstraintReferenceFt(constraint)).toBe(3900);
    expect(altitudeConstraintText(constraint)).toBe("<= 3,900 ft");
    expect(altitudeConstraintClassName(constraint)).toBe("is-at-or-below");
  });

  it("formats altitude windows", () => {
    const constraint = { kind: "WINDOW" as const, minFtMsl: 2500, maxFtMsl: 4000 };

    expect(altitudeConstraintReferenceFt(constraint)).toBe(2500);
    expect(altitudeConstraintText(constraint)).toBe("2,500 ft-4,000 ft");
    expect(altitudeConstraintClassName(constraint)).toBe("is-window");
  });
});
