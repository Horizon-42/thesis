import { describe, expect, it } from "vitest";
import { bareRunwayIdent, normalizeRunwayIdent, runwayMatchesSelection } from "../runwayIdent";

describe("normalizeRunwayIdent", () => {
  it("prefixes bare identifiers with RW", () => {
    expect(normalizeRunwayIdent("05L")).toBe("RW05L");
    expect(normalizeRunwayIdent("32")).toBe("RW32");
  });

  it("leaves already-prefixed identifiers unchanged (and upper-cases/trims)", () => {
    expect(normalizeRunwayIdent("RW05L")).toBe("RW05L");
    expect(normalizeRunwayIdent("  rw23r ")).toBe("RW23R");
    expect(normalizeRunwayIdent("05l")).toBe("RW05L");
  });
});

describe("bareRunwayIdent", () => {
  it("drops the RW prefix to the bare landings spelling", () => {
    expect(bareRunwayIdent("RW05L")).toBe("05L");
    expect(bareRunwayIdent("rw23r")).toBe("23R");
  });

  it("leaves an already-bare identifier unchanged", () => {
    expect(bareRunwayIdent("05L")).toBe("05L");
    expect(bareRunwayIdent(" 32 ")).toBe("32");
  });

  it("round-trips with normalizeRunwayIdent", () => {
    expect(bareRunwayIdent(normalizeRunwayIdent("23R"))).toBe("23R");
  });
});

describe("runwayMatchesSelection", () => {
  it("matches everything when nothing is selected (All runways)", () => {
    expect(runwayMatchesSelection(null, "RW05L")).toBe(true);
    expect(runwayMatchesSelection(null, "32")).toBe(true);
  });

  it("matches across the bare/RW-prefixed spelling gap", () => {
    expect(runwayMatchesSelection("05L", "RW05L")).toBe(true);
    expect(runwayMatchesSelection("RW05L", "05L")).toBe(true);
    expect(runwayMatchesSelection("32", "RW32")).toBe(true);
  });

  it("rejects a different runway", () => {
    expect(runwayMatchesSelection("23R", "RW05L")).toBe(false);
    expect(runwayMatchesSelection("05L", "RW05R")).toBe(false);
  });
});
