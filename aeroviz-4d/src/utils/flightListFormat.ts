/**
 * flightListFormat.ts
 * -------------------
 * Small pure formatters for the Observe-mode flight list columns.
 */

/** Seconds → "m:ss" (or "h:mm:ss" past an hour). Returns "—" for null/negative/non-finite. */
export function formatDuration(seconds: number | null): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "—";
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = h > 0 ? String(m).padStart(2, "0") : String(m);
  const ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

/** Ground speed m/s → "142" (unit lives in the column header). Returns "—" for null/non-finite. */
export function formatSpeed(mps: number | null): string {
  if (mps == null || !Number.isFinite(mps)) return "—";
  return `${Math.round(mps)}`;
}

/** Mass kg → tonnes "66.3" (unit lives in the column header). Returns "—" for null/non-finite. */
export function formatMass(kg: number | null): string {
  if (kg == null || !Number.isFinite(kg)) return "—";
  return `${(kg / 1000).toFixed(1)}`;
}
