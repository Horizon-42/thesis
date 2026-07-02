import { describe, expect, it } from "vitest";
import { formatDuration, formatSpeed, formatMass, formatPercent } from "../flightListFormat";

describe("flightListFormat", () => {
  it("formats durations as m:ss and h:mm:ss", () => {
    expect(formatDuration(0)).toBe("0:00");
    expect(formatDuration(9)).toBe("0:09");
    expect(formatDuration(562)).toBe("9:22");
    expect(formatDuration(3661)).toBe("1:01:01");
  });

  it("returns a dash for missing / invalid durations", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(-5)).toBe("—");
    expect(formatDuration(Number.NaN)).toBe("—");
  });

  it("formats ground speed as a rounded number (unit lives in the header)", () => {
    expect(formatSpeed(141.85)).toBe("142");
    expect(formatSpeed(0)).toBe("0");
    expect(formatSpeed(null)).toBe("—");
  });

  it("formats mass in tonnes with one decimal (unit lives in the header)", () => {
    expect(formatMass(66300)).toBe("66.3");
    expect(formatMass(6804)).toBe("6.8");
    expect(formatMass(null)).toBe("—");
  });

  it("formats a fraction as a percentage", () => {
    expect(formatPercent(0.6267)).toBe("62.7%");
    expect(formatPercent(1)).toBe("100.0%");
    expect(formatPercent(null)).toBe("—");
    expect(formatPercent(undefined)).toBe("—");
  });
});
