import type { AltitudeConstraint } from "./procedurePackage";

export type DisplayAltitudeConstraint = Pick<
  AltitudeConstraint,
  "kind" | "minFtMsl" | "maxFtMsl" | "sourceText"
>;

/**
 * A leg altitude exactly as the Python preprocessor codes it on disk
 * (`ProcedureDetailLeg.constraints.altitude`): the CIFP altitude descriptor
 * mapped to a `qualifier` plus the binding `valueFt` and its original text.
 */
export interface CifpLegAltitude {
  qualifier: string;
  valueFt: number;
  rawText: string;
}

function parseAltitudeWindowBounds(
  rawText: string,
): { minFtMsl: number; maxFtMsl: number } | null {
  const values = [...rawText.matchAll(/\d[\d,]*/g)]
    .map((match) => Number(match[0].replace(/,/g, "")))
    .filter((value) => Number.isFinite(value));
  if (values.length < 2) return null;
  const [first, second] = values;
  return { minFtMsl: Math.min(first, second), maxFtMsl: Math.max(first, second) };
}

/**
 * The single canonical conversion from a coded CIFP leg altitude to the
 * in-memory {@link AltitudeConstraint}. Both the route view-model
 * (`procedureRoutes`) and the normalized procedure package
 * (`procedurePackageAdapter`) go through this, so a block / window / unknown
 * altitude is interpreted identically everywhere. (Previously the two
 * pipelines had divergent copies: the route builder produced WINDOW/UNKNOWN
 * while the package adapter silently collapsed every non-above/below altitude
 * to AT, dropping the upper bound of a block altitude.)
 */
export function altitudeConstraintFromCifp(
  altitude: CifpLegAltitude | null | undefined,
): AltitudeConstraint | null {
  if (!altitude) return null;
  const qualifier = altitude.qualifier.toLowerCase();
  const rawText = altitude.rawText;

  const looksLikeWindow =
    qualifier.includes("block") ||
    qualifier.includes("window") ||
    qualifier.includes("between") ||
    /\bbetween\b|-/.test(rawText);
  if (looksLikeWindow) {
    const window = parseAltitudeWindowBounds(rawText);
    if (window) return { kind: "WINDOW", ...window, sourceText: rawText };
  }

  if (qualifier.includes("above")) {
    return { kind: "AT_OR_ABOVE", minFtMsl: altitude.valueFt, sourceText: rawText };
  }
  if (qualifier.includes("below")) {
    return { kind: "AT_OR_BELOW", maxFtMsl: altitude.valueFt, sourceText: rawText };
  }
  return {
    kind: qualifier.includes("unknown") ? "UNKNOWN" : "AT",
    minFtMsl: altitude.valueFt,
    maxFtMsl: altitude.valueFt,
    sourceText: rawText,
  };
}

function formatFt(value: number): string {
  return `${Math.round(value).toLocaleString()} ft`;
}

export function altitudeConstraintReferenceFt(
  constraint: DisplayAltitudeConstraint | null | undefined,
): number | null {
  if (!constraint) return null;
  if (constraint.kind === "AT_OR_BELOW") return constraint.maxFtMsl ?? null;
  if (constraint.kind === "WINDOW") return constraint.minFtMsl ?? constraint.maxFtMsl ?? null;
  return constraint.minFtMsl ?? constraint.maxFtMsl ?? null;
}

export function altitudeConstraintText(
  constraint: DisplayAltitudeConstraint | null | undefined,
): string {
  if (!constraint) return "";
  if (constraint.kind === "AT") {
    const value = constraint.minFtMsl ?? constraint.maxFtMsl;
    return typeof value === "number" ? `AT ${formatFt(value)}` : (constraint.sourceText ?? "AT");
  }
  if (constraint.kind === "AT_OR_ABOVE") {
    return typeof constraint.minFtMsl === "number"
      ? `>= ${formatFt(constraint.minFtMsl)}`
      : (constraint.sourceText ?? "AT OR ABOVE");
  }
  if (constraint.kind === "AT_OR_BELOW") {
    return typeof constraint.maxFtMsl === "number"
      ? `<= ${formatFt(constraint.maxFtMsl)}`
      : (constraint.sourceText ?? "AT OR BELOW");
  }
  if (constraint.kind === "WINDOW") {
    if (
      typeof constraint.minFtMsl === "number" &&
      typeof constraint.maxFtMsl === "number"
    ) {
      return `${formatFt(constraint.minFtMsl)}-${formatFt(constraint.maxFtMsl)}`;
    }
    return constraint.sourceText ?? "WINDOW";
  }
  return constraint.sourceText ?? "ALT";
}

export function altitudeConstraintLabel(
  fixIdent: string,
  constraint: DisplayAltitudeConstraint | null | undefined,
): string {
  const constraintText = altitudeConstraintText(constraint);
  return constraintText ? `${fixIdent} ${constraintText}` : fixIdent;
}

export function altitudeConstraintClassName(
  constraint: DisplayAltitudeConstraint | null | undefined,
): string {
  if (!constraint) return "is-unknown";
  if (constraint.kind === "AT") return "is-at";
  if (constraint.kind === "AT_OR_ABOVE") return "is-at-or-above";
  if (constraint.kind === "AT_OR_BELOW") return "is-at-or-below";
  if (constraint.kind === "WINDOW") return "is-window";
  return "is-unknown";
}
