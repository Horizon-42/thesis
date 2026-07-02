/**
 * observedFlightSummary.ts
 * ------------------------
 * Pure per-flight facts read straight off the observed CZML packets for the flight list.
 *
 * Only the **duration** is read here: it is unambiguous (last sample time − first) with no
 * estimation, so there is nothing to be inconsistent with. The flight's **V** and **mass** are
 * NOT re-estimated on the frontend — they come from the optimizer's scenario initial state (see
 * useFlightOptimizerData), the single source the optimizer itself started from.
 *
 * The observed packets carry position as `position.cartographicDegrees = [t, lon, lat, alt, …]`
 * (t in seconds) and have NO `availability` field, so duration must come from the sample times.
 */

export interface ObservedFlightSummary {
  /** Observed track duration in seconds (last sample time − first). */
  durationS: number;
}

/** Duration for one packet's `cartographicDegrees`, or null if there are no samples. */
export function summarizeCartographicDegrees(samples: number[]): ObservedFlightSummary | null {
  const n = Math.floor(samples.length / 4);
  if (n < 1) return null;
  return { durationS: Math.max(0, samples[(n - 1) * 4] - samples[0]) };
}

/** Map every non-document packet with a `cartographicDegrees` position to its summary. */
export function summarizeObservedCzml(czml: unknown): Record<string, ObservedFlightSummary> {
  const out: Record<string, ObservedFlightSummary> = {};
  if (!Array.isArray(czml)) return out;
  for (const packet of czml) {
    if (!packet || typeof packet !== "object") continue;
    const p = packet as { id?: unknown; position?: { cartographicDegrees?: unknown } };
    if (typeof p.id !== "string" || p.id === "document") continue;
    const samples = p.position?.cartographicDegrees;
    if (!Array.isArray(samples)) continue;
    const summary = summarizeCartographicDegrees(samples as number[]);
    if (summary) out[p.id] = summary;
  }
  return out;
}
