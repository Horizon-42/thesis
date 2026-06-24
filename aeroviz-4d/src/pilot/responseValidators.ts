/**
 * responseValidators.ts
 * ---------------------
 * Small shared validators for parsing AeroViz backend JSON responses. Used by
 * the optimization and dynamics-comparison clients so the "is this a finite
 * number / non-empty string / record" checks live in one place.
 */

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** A finite number, or throw. */
export function asFiniteNumber(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error("AeroViz backend response has a non-finite number");
  }
  return value;
}

export function readNumber(value: Record<string, unknown>, key: string): number {
  const nested = value[key];
  if (typeof nested !== "number" || !Number.isFinite(nested)) {
    throw new Error(`AeroViz backend response has invalid ${key}`);
  }
  return nested;
}

/** The number at `key`, or null when the key is absent (present-but-invalid throws). */
export function readOptionalNumber(
  value: Record<string, unknown>,
  key: string,
): number | null {
  if (!(key in value)) return null;
  return readNumber(value, key);
}

export function readPositiveNumber(value: Record<string, unknown>, key: string): number {
  const number = readNumber(value, key);
  if (number <= 0) {
    throw new Error(`AeroViz backend response has invalid ${key}`);
  }
  return number;
}

export function readString(value: Record<string, unknown>, key: string): string {
  const nested = value[key];
  if (typeof nested !== "string" || nested.length === 0) {
    throw new Error(`AeroViz backend response has invalid ${key}`);
  }
  return nested;
}

export function readBoolean(value: Record<string, unknown>, key: string): boolean {
  const nested = value[key];
  if (typeof nested !== "boolean") {
    throw new Error(`AeroViz backend response has invalid ${key}`);
  }
  return nested;
}

export function readNumberArray(value: Record<string, unknown>, key: string): number[] {
  const nested = value[key];
  if (!Array.isArray(nested)) {
    throw new Error(`AeroViz backend response has invalid ${key}`);
  }
  return nested.map(asFiniteNumber);
}
