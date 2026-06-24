import { describe, expect, it } from "vitest";
import {
  altitudeConstraintClassName,
  altitudeConstraintFromCifp,
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

describe("altitudeConstraintFromCifp (single canonical CIFP conversion)", () => {
  it("returns null for an uncoded altitude", () => {
    expect(altitudeConstraintFromCifp(null)).toBeNull();
  });

  it("maps the at-or-above descriptor to a lower-bounded window", () => {
    expect(
      altitudeConstraintFromCifp({ qualifier: "atOrAbove", valueFt: 3400, rawText: "3400 ft" }),
    ).toEqual({ kind: "AT_OR_ABOVE", minFtMsl: 3400, sourceText: "3400 ft" });
  });

  it("maps the at-or-below descriptor to an upper-bounded window", () => {
    expect(
      altitudeConstraintFromCifp({ qualifier: "atOrBelow", valueFt: 5000, rawText: "5000 ft" }),
    ).toEqual({ kind: "AT_OR_BELOW", maxFtMsl: 5000, sourceText: "5000 ft" });
  });

  it("maps a plain crossing altitude to AT with equal bounds", () => {
    expect(
      altitudeConstraintFromCifp({ qualifier: "at", valueFt: 2200, rawText: "2200 ft" }),
    ).toEqual({ kind: "AT", minFtMsl: 2200, maxFtMsl: 2200, sourceText: "2200 ft" });
  });

  it("keeps BOTH bounds of a block altitude (the consolidation fix)", () => {
    // Previously the package adapter collapsed a block to AT and dropped the
    // upper bound; both pipelines now share this WINDOW interpretation.
    expect(
      altitudeConstraintFromCifp({ qualifier: "block", valueFt: 5000, rawText: "5000-7000" }),
    ).toEqual({ kind: "WINDOW", minFtMsl: 5000, maxFtMsl: 7000, sourceText: "5000-7000" });
  });
});
