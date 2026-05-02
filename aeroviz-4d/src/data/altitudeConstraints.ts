import type { AltitudeConstraint } from "./procedurePackage";

export type DisplayAltitudeConstraint = Pick<
  AltitudeConstraint,
  "kind" | "minFtMsl" | "maxFtMsl" | "sourceText"
>;

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
