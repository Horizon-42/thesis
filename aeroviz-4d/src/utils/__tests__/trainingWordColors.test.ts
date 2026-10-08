/**
 * The landed green (frontend D177 (1)): at least the palette's floor, OKLab ΔE 11.9, from the post-trained yellow-green
 * and from the closed loop's teal, which it is drawn beside (a round's tab swatch and outcome dot, the other rounds).
 */
import { describe, expect, it } from "vitest";
import { TRAINING_DECISION_PASS_COLOR, TRAINING_SENTENCE_COLOR, trainingOutcomeColour } from "../trainingWordColors";

/** OKLab of a #rrggbb colour (Björn Ottosson's matrices). */
function oklab(hex: string): [number, number, number] {
  const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const [r, g, b] = [1, 3, 5].map((i) => lin(parseInt(hex.slice(i, i + 2), 16) / 255));
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
const deltaE = (a: string, b: string) => 100 * Math.hypot(...oklab(a).map((v, k) => v - oklab(b)[k]));

describe("the landed green (D177 (1))", () => {
  it("is a landed outcome's colour, at least 11.9 OKLab ΔE from the post-trained yellow-green and the teal", () => {
    expect(trainingOutcomeColour("landed")).toBe(TRAINING_DECISION_PASS_COLOR);
    expect(deltaE("#4ade80", TRAINING_SENTENCE_COLOR.postTrained)).toBeCloseTo(9.5, 1);     // the former green
    expect(deltaE(TRAINING_DECISION_PASS_COLOR, TRAINING_SENTENCE_COLOR.postTrained)).toBeGreaterThanOrEqual(11.9);
    expect(deltaE(TRAINING_DECISION_PASS_COLOR, TRAINING_SENTENCE_COLOR.closedLoop)).toBeGreaterThanOrEqual(11.9);
    const [, a, b] = oklab(TRAINING_DECISION_PASS_COLOR);
    expect(a).toBeLessThan(0);                                       // green, not yellow or blue: a < 0, hue near 140°
    expect(Math.atan2(b, a) * 180 / Math.PI).toBeGreaterThan(120);
    expect(Math.atan2(b, a) * 180 / Math.PI).toBeLessThan(160);
  });
});
