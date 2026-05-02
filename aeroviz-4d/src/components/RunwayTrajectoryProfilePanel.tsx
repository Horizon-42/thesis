import { useMemo, useState } from "react";
import {
  useApp,
  type RunwayProfileViewMode,
} from "../context/AppContext";
import {
  useRunwayTrajectoryProfile,
  type ProfileAircraftTrack,
} from "../hooks/useRunwayTrajectoryProfile";
import type {
  HorizontalPlateAssessmentSegment,
  HorizontalPlateRoute,
  RunwayProfilePoint,
  RunwayReferenceMark,
} from "../utils/runwayProfileGeometry";
import {
  isProcedureAnnotationVisibleAtDisplayLevel,
  type ProcedureAnnotationKind,
  type ProcedureAnnotationStatus,
  type ProcedureDisplayLevel,
  type ProcedureEntityAnnotation,
} from "../data/procedureAnnotations";
import {
  altitudeConstraintClassName,
  altitudeConstraintLabel,
} from "../data/altitudeConstraints";

const METERS_PER_NM = 1852;
const METERS_PER_FOOT = 0.3048;

type DistanceUnit = "nm" | "m";

interface PlotDomain {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

interface ProfilePlotProps {
  title: string;
  subtitle: string;
  mode: "side" | "top";
  displayLevel: ProcedureDisplayLevel;
  distanceUnit: DistanceUnit;
  tracks: ProfileAircraftTrack[];
  plateRoutes: HorizontalPlateRoute[];
  referenceMarks: RunwayReferenceMark[];
}

function showProfileElement(
  kind: ProcedureAnnotationKind,
  status: ProcedureAnnotationStatus,
  displayLevel: ProcedureDisplayLevel,
): boolean {
  return isProcedureAnnotationVisibleAtDisplayLevel(
    { kind, status } as ProcedureEntityAnnotation,
    displayLevel,
  );
}

function formatIsoTime(iso: string | null): string {
  if (!iso) return "No active simulation time";
  return iso.replace("T", " ").replace(".000Z", "Z");
}

function collectViewDomain(
  mode: "side" | "top",
  tracks: ProfileAircraftTrack[],
  plateRoutes: HorizontalPlateRoute[],
  referenceMarks: RunwayReferenceMark[],
): PlotDomain {
  const xValues = [0];
  const yValues = [0];
  const pushPoint = (point: RunwayProfilePoint) => {
    xValues.push(point.xM);
    yValues.push(mode === "side" ? point.zM : point.yM);
  };
  const pushTopBand = (points: RunwayProfilePoint[], halfWidthM: number | null | undefined) => {
    if (mode !== "top" || typeof halfWidthM !== "number" || !Number.isFinite(halfWidthM)) return;
    points.forEach((point) => {
      yValues.push(point.yM + halfWidthM, point.yM - halfWidthM);
    });
  };

  plateRoutes.forEach((route) => {
    route.points.forEach((point) => {
      pushPoint(point);
      pushTopBand([point], route.halfWidthM);
    });
    (route.assessmentSegments ?? []).forEach((segment) => {
      segment.points.forEach(pushPoint);
      pushTopBand(segment.points, segment.primaryHalfWidthM);
      pushTopBand(segment.points, segment.secondaryHalfWidthM);
      if (segment.finalVerticalReference) {
        segment.finalVerticalReference.points.forEach(pushPoint);
        pushTopBand(segment.finalVerticalReference.points, segment.finalVerticalReference.halfWidthM);
      }
      if (segment.lnavVnavOcs) {
        segment.lnavVnavOcs.points.forEach(pushPoint);
        pushTopBand(segment.lnavVnavOcs.points, segment.lnavVnavOcs.primaryHalfWidthM);
        pushTopBand(segment.lnavVnavOcs.points, segment.lnavVnavOcs.secondaryHalfWidthM);
      }
      (segment.precisionSurfaces ?? []).forEach((surface) => {
        surface.points.forEach(pushPoint);
      });
    });
  });

  tracks.forEach((track) => {
    track.trail.forEach((point) => {
      xValues.push(point.xM);
      yValues.push(mode === "side" ? point.zM : point.yM);
    });
  });

  referenceMarks.forEach((mark) => {
    xValues.push(mark.xM);
  });

  const minX = Math.min(...xValues, -250);
  const maxX = Math.max(...xValues, 250);
  const xPadding = Math.max(120, (maxX - minX) * 0.08);

  if (mode === "side") {
    const minY = Math.min(...yValues, 0);
    const maxY = Math.max(...yValues, 120);
    const yPadding = Math.max(40, (maxY - minY) * 0.1);
    return {
      minX: minX - xPadding,
      maxX: maxX + xPadding,
      minY: Math.min(0, minY - yPadding),
      maxY: maxY + yPadding,
    };
  }

  const maxAbsY = Math.max(...yValues.map((value) => Math.abs(value)), 180);
  const yPadding = Math.max(50, maxAbsY * 0.14);
  return {
    minX: minX - xPadding,
    maxX: maxX + xPadding,
    minY: -(maxAbsY + yPadding),
    maxY: maxAbsY + yPadding,
  };
}

function formatNm(valueM: number): string {
  const valueNm = valueM / METERS_PER_NM;
  return `${Math.abs(valueNm) < 0.05 ? "0.0" : valueNm.toFixed(1)} NM`;
}

function formatMeters(valueM: number): string {
  return `${Math.abs(valueM) < 0.5 ? "0" : Math.round(valueM).toLocaleString()} m`;
}

function formatFeet(valueM: number): string {
  return `${Math.round(valueM / METERS_PER_FOOT).toLocaleString()} ft`;
}

function formatSignedFeet(valueM: number): string {
  const prefix = valueM > 0 ? "+" : "";
  return `${prefix}${formatFeet(valueM)}`;
}

function formatDistance(valueM: number, unit: DistanceUnit): string {
  return unit === "nm" ? formatNm(valueM) : formatMeters(valueM);
}

function formatSignedDistance(valueM: number, unit: DistanceUnit): string {
  const prefix = valueM > 0 ? "+" : "";
  return `${prefix}${formatDistance(valueM, unit)}`;
}

function distanceUnitLabel(unit: DistanceUnit): string {
  return unit === "nm" ? "NM" : "m";
}

function formatProfileAxisValue(
  mode: "side" | "top",
  valueM: number,
  distanceUnit: DistanceUnit,
): string {
  return mode === "side" ? formatFeet(valueM) : formatDistance(valueM, distanceUnit);
}

function routeLabel(route: HorizontalPlateRoute): string {
  if (route.branchType.toLowerCase() === "transition") {
    return `${route.transitionIdent ?? route.branchIdent} transition`;
  }
  return `${route.procedureName} final`;
}

function segmentDebugLabel(segment: HorizontalPlateAssessmentSegment, index: number): string {
  const segmentKey = segment.segmentId.split(":").slice(-1)[0] ?? `${index + 1}`;
  return `S${index + 1} ${segmentKey}`;
}

function niceTickStep(span: number, targetTickCount: number): number {
  const rough = Math.max(1, span / Math.max(1, targetTickCount));
  const power = 10 ** Math.floor(Math.log10(rough));
  const scaled = rough / power;
  const niceScaled = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return niceScaled * power;
}

function buildLinearUnitTicks(
  minM: number,
  maxM: number,
  targetTickCount: number,
  unitM: number,
): number[] {
  const minUnit = minM / unitM;
  const maxUnit = maxM / unitM;
  const stepUnit = niceTickStep(Math.max(1, maxUnit - minUnit), targetTickCount);
  const startUnit = Math.ceil(minUnit / stepUnit) * stepUnit;
  const ticks: number[] = [];
  for (let valueUnit = startUnit; valueUnit <= maxUnit + stepUnit * 0.25; valueUnit += stepUnit) {
    ticks.push(Number((valueUnit * unitM).toFixed(6)));
  }
  return ticks;
}

function selectVisibleReferenceMarks(
  marks: RunwayReferenceMark[],
  plotX: (value: number) => number,
): RunwayReferenceMark[] {
  const accepted: RunwayReferenceMark[] = [];

  marks.forEach((mark) => {
    const pixelX = plotX(mark.xM);
    const conflicts = accepted.some((acceptedMark) => {
      const acceptedPixelX = plotX(acceptedMark.xM);
      return Math.abs(pixelX - acceptedPixelX) < 44;
    });
    if (!conflicts) {
      accepted.push(mark);
    }
  });

  return [...accepted].sort((left, right) => right.xM - left.xM);
}

function referenceMarkKey(mark: RunwayReferenceMark): string {
  return `${mark.label}|${mark.detail}|${Math.round(mark.xM)}|${Math.round(mark.yM)}|${Math.round(mark.zM)}`;
}

function referencePriority(role: string, branchType?: string): number {
  const normalizedRole = role.toUpperCase();
  const rolePriority =
    normalizedRole === "FAF"
      ? 6
      : normalizedRole === "MAPT"
        ? 5
        : normalizedRole === "IF"
          ? 4
          : normalizedRole === "IAF"
            ? 3
            : 2;
  return rolePriority + ((branchType ?? "final").toLowerCase() === "final" ? 1 : 0);
}

function buildSideReferenceMarks(
  routes: HorizontalPlateRoute[],
  thresholdMarks: RunwayReferenceMark[],
): RunwayReferenceMark[] {
  const routeMarks = routes.flatMap((route) =>
    route.points.map((point) => ({
      xM: point.xM,
      yM: point.yM,
      zM: point.zM,
      label: point.fixIdent,
      detail: point.role,
      priority: referencePriority(point.role, route.branchType),
    })),
  );

  return [...thresholdMarks, ...routeMarks].sort((left, right) => {
    if (left.priority === right.priority) return right.xM - left.xM;
    return right.priority - left.priority;
  });
}

function routeHasEstimatedVerticalGeometry(route: HorizontalPlateRoute): boolean {
  return (route.assessmentSegments ?? []).some(
    (segment) =>
      Boolean(segment.finalVerticalReference) ||
      Boolean(segment.lnavVnavOcs) ||
      (segment.precisionSurfaces ?? []).length > 0,
  );
}

function plotPointPath(
  points: RunwayProfilePoint[],
  domain: PlotDomain,
  plotWidth: number,
  plotHeight: number,
  marginLeft: number,
  marginTop: number,
  mode: "side" | "top",
): string {
  const xSpan = Math.max(1, domain.maxX - domain.minX);
  const ySpan = Math.max(1, domain.maxY - domain.minY);
  const plotX = (value: number) =>
    marginLeft + ((domain.maxX - value) / xSpan) * plotWidth;
  const plotY = (value: number) =>
    marginTop + ((domain.maxY - value) / ySpan) * plotHeight;

  return points
    .map((point, index) => {
      const x = plotX(point.xM);
      const y = plotY(mode === "side" ? point.zM : point.yM);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

function ProfilePlot({
  title,
  subtitle,
  mode,
  displayLevel,
  distanceUnit,
  tracks,
  plateRoutes,
  referenceMarks,
}: ProfilePlotProps) {
  const width = 760;
  const height = 340;
  const marginLeft = 60;
  const marginRight = 18;
  const marginTop = 22;
  const marginBottom = 126;
  const plotWidth = width - marginLeft - marginRight;
  const plotHeight = height - marginTop - marginBottom;
  const displayedPlateRoutes = useMemo(
    () => {
      if (mode !== "side") return plateRoutes;
      const selectedTransitions = plateRoutes.filter(
        (route) => route.branchType.toLowerCase() === "transition",
      );
      const finalVerticalRoutes = plateRoutes.filter(
        (route) =>
          route.branchType.toLowerCase() === "final" && routeHasEstimatedVerticalGeometry(route),
      );
      if (selectedTransitions.length === 0) {
        return plateRoutes.filter((route) => route.branchType.toLowerCase() === "final");
      }
      const displayedByRouteId = new Map<string, HorizontalPlateRoute>();
      [...selectedTransitions, ...finalVerticalRoutes].forEach((route) => {
        displayedByRouteId.set(route.routeId, route);
      });
      return [...displayedByRouteId.values()];
    },
    [mode, plateRoutes],
  );
  const displayedReferenceMarks = useMemo(
    () => {
      if (mode !== "side") return referenceMarks;
      return buildSideReferenceMarks(
        displayedPlateRoutes,
        referenceMarks.filter((mark) => mark.detail === "Threshold"),
      );
    },
    [displayedPlateRoutes, mode, referenceMarks],
  );
  const domain = useMemo(
    () => collectViewDomain(mode, tracks, displayedPlateRoutes, displayedReferenceMarks),
    [displayedPlateRoutes, displayedReferenceMarks, mode, tracks],
  );

  const xSpan = Math.max(1, domain.maxX - domain.minX);
  const ySpan = Math.max(1, domain.maxY - domain.minY);
  const plotClipId = `runway-profile-plot-clip-${mode}`;
  const plotX = (value: number) => marginLeft + ((domain.maxX - value) / xSpan) * plotWidth;
  const plotY = (value: number) => marginTop + ((domain.maxY - value) / ySpan) * plotHeight;
  const zeroX = plotX(0);
  const zeroY = plotY(0);
  const yScale = plotHeight / ySpan;
  const horizontalTickUnitM = distanceUnit === "nm" ? METERS_PER_NM : 1;
  const xTicks = useMemo(
    () => buildLinearUnitTicks(domain.minX, domain.maxX, 7, horizontalTickUnitM),
    [domain.maxX, domain.minX, horizontalTickUnitM],
  );
  const visibleReferenceMarks = useMemo(
    () => selectVisibleReferenceMarks(displayedReferenceMarks, plotX),
    [displayedReferenceMarks, plotX],
  );
  const labeledReferenceMarkKeys = useMemo(
    () => new Set(visibleReferenceMarks.map(referenceMarkKey)),
    [visibleReferenceMarks],
  );
  const routeLabelReferenceMarks = useMemo(
    () =>
      visibleReferenceMarks.filter((mark) => mark.detail !== "Threshold"),
    [visibleReferenceMarks],
  );
  const plottedReferenceMarks = displayedReferenceMarks.filter(
    (mark) => mark.detail === "Threshold",
  );
  // Keep the embedded profile aligned with the same semantic display levels as the 3D procedure layer.
  const showProtectionGeometry = showProfileElement(
    "SEGMENT_ENVELOPE_PRIMARY",
    "SOURCE_BACKED",
    displayLevel,
  );
  const showFinalVerticalReferenceGeometry = showProfileElement(
    "FINAL_VERTICAL_REFERENCE",
    "ESTIMATED",
    displayLevel,
  );
  const showOcsGeometry = showProfileElement(
    "LNAV_VNAV_OCS",
    "ESTIMATED",
    displayLevel,
  );
  const showAltitudeConstraintGeometry = showProfileElement(
    "ALTITUDE_CONSTRAINT",
    "SOURCE_BACKED",
    displayLevel,
  );
  const showDebugFinalSurfaceGeometry = showProfileElement(
    "PRECISION_SURFACE",
    "DEBUG_ESTIMATE",
    displayLevel,
  );

  return (
    <section className="runway-profile-plot">
      <header>
        <h4>{title}</h4>
        <p>{subtitle}</p>
      </header>

      <svg
        className="runway-profile-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={title}
      >
        <defs>
          <clipPath id={plotClipId}>
            <rect x={marginLeft} y={marginTop} width={plotWidth} height={plotHeight} rx={10} />
          </clipPath>
        </defs>
        <rect
          x={marginLeft}
          y={marginTop}
          width={plotWidth}
          height={plotHeight}
          rx={10}
          className="runway-profile-plot-frame"
        />

        <line
          x1={marginLeft}
          y1={marginTop + plotHeight}
          x2={marginLeft + plotWidth}
          y2={marginTop + plotHeight}
          className="runway-profile-axis"
        />
        <line
          x1={marginLeft}
          y1={marginTop}
          x2={marginLeft}
          y2={marginTop + plotHeight}
          className="runway-profile-axis"
        />

        {xTicks.map((tick) => {
          const tickX = plotX(tick);
          return (
            <g key={`x-tick-${mode}-${tick}`}>
              <line
                x1={tickX}
                y1={marginTop}
                x2={tickX}
                y2={marginTop + plotHeight}
                className="runway-profile-grid-line"
              />
              <line
                x1={tickX}
                y1={marginTop + plotHeight}
                x2={tickX}
                y2={marginTop + plotHeight + 6}
                className="runway-profile-axis"
              />
              <text
                x={tickX}
                y={marginTop + plotHeight + 22}
                textAnchor="middle"
                className="runway-profile-axis-tick"
              >
                {formatDistance(tick, distanceUnit)}
              </text>
            </g>
          );
        })}

        <line
          x1={zeroX}
          y1={marginTop}
          x2={zeroX}
          y2={marginTop + plotHeight}
          className="runway-profile-threshold-line"
        />
        <text x={zeroX + 6} y={marginTop + 14} className="runway-profile-threshold-label">
          Threshold
        </text>

        {mode === "top" ? (
          <>
            <line
              x1={marginLeft}
              y1={zeroY}
              x2={marginLeft + plotWidth}
              y2={zeroY}
              className="runway-profile-centerline"
            />
            <text x={marginLeft + 8} y={zeroY - 8} className="runway-profile-centerline-label">
              Centerline y = 0
            </text>
          </>
        ) : (
          <>
            <line
              x1={marginLeft}
              y1={zeroY}
              x2={marginLeft + plotWidth}
              y2={zeroY}
              className="runway-profile-ground-line"
            />
            <text x={marginLeft + 8} y={zeroY - 8} className="runway-profile-centerline-label">
              Threshold elevation z = 0
            </text>
          </>
        )}

        {plottedReferenceMarks.map((mark) => {
          const markX = plotX(mark.xM);
          const markY = plotY(mode === "side" ? mark.zM : mark.yM);
          const label = `${mark.label} · ${mark.detail}`;
          const shouldLabel = labeledReferenceMarkKeys.has(referenceMarkKey(mark));
          return (
            <g key={`${mode}-mark-${referenceMarkKey(mark)}`}>
              <line
                x1={markX}
                y1={marginTop}
                x2={markX}
                y2={marginTop + plotHeight}
                className="runway-profile-reference-line"
              />
              <circle
                cx={markX}
                cy={markY}
                r={4.1}
                className="runway-profile-reference-point"
                data-fix-ident={mark.label}
              />
              <line
                x1={markX}
                y1={marginTop + plotHeight}
                x2={markX}
                y2={marginTop + plotHeight + 20}
                className="runway-profile-reference-tick"
              />
              {shouldLabel ? (
                <text
                  x={markX + 4}
                  y={marginTop + plotHeight + 56}
                  transform={`rotate(-28 ${markX + 4} ${marginTop + plotHeight + 56})`}
                  textAnchor="start"
                  className="runway-profile-reference-label"
                >
                  {label}
                </text>
              ) : null}
            </g>
          );
        })}

        {routeLabelReferenceMarks.map((mark) => {
          const markX = plotX(mark.xM);
          const label = `${mark.label} · ${mark.detail}`;
          return (
            <g key={`${mode}-route-label-${referenceMarkKey(mark)}`}>
              <line
                x1={markX}
                y1={marginTop}
                x2={markX}
                y2={marginTop + plotHeight}
                className="runway-profile-reference-line"
              />
              <line
                x1={markX}
                y1={marginTop + plotHeight}
                x2={markX}
                y2={marginTop + plotHeight + 20}
                className="runway-profile-reference-tick"
              />
              <text
                x={markX + 4}
                y={marginTop + plotHeight + 56}
                transform={`rotate(-28 ${markX + 4} ${marginTop + plotHeight + 56})`}
                textAnchor="start"
                className="runway-profile-reference-label"
              >
                {label}
              </text>
            </g>
          );
        })}

        {displayedPlateRoutes.map((route) => {
          const d = plotPointPath(
            route.points,
            domain,
            plotWidth,
            plotHeight,
            marginLeft,
            marginTop,
            mode,
          );
          const assessmentSegments = (route.assessmentSegments ?? []).filter(
            (segment) => segment.points.length >= 2,
          );
          const bandWidth = mode === "top" ? Math.max(2, route.halfWidthM * yScale * 2) : 2;
          const labelPoint = route.points[Math.max(0, Math.floor((route.points.length - 1) / 2))];
          const labelX = labelPoint ? plotX(labelPoint.xM) : 0;
          const labelY = labelPoint ? plotY(mode === "side" ? labelPoint.zM : labelPoint.yM) : 0;
          const isTransition = route.branchType.toLowerCase() === "transition";
          return (
            <g key={`${mode}-${route.routeId}`}>
              {mode === "top" && showProtectionGeometry && assessmentSegments.length === 0 ? (
                <path
                  d={d}
                  className="runway-profile-route-band"
                  clipPath={`url(#${plotClipId})`}
                  style={{ strokeWidth: bandWidth }}
                />
              ) : null}
              {mode === "top" && showProtectionGeometry
                ? assessmentSegments.map((segment, index) => {
                    const segmentPath = plotPointPath(
                      segment.points,
                      domain,
                      plotWidth,
                      plotHeight,
                      marginLeft,
                      marginTop,
                      mode,
                    );
                    return (
                      <g key={`${route.routeId}-assessment-band-${segment.segmentId}-${index}`}>
                        {segment.secondaryHalfWidthM ? (
                          <path
                            d={segmentPath}
                            className="runway-profile-assessment-secondary-band"
                            clipPath={`url(#${plotClipId})`}
                            style={{
                              strokeWidth: Math.max(2, segment.secondaryHalfWidthM * yScale * 2),
                            }}
                          />
                        ) : null}
                        <path
                          d={segmentPath}
                          className="runway-profile-assessment-primary-band"
                          clipPath={`url(#${plotClipId})`}
                          style={{
                            strokeWidth: Math.max(2, segment.primaryHalfWidthM * yScale * 2),
                          }}
                        />
                      </g>
                    );
                  })
                : null}
              {showOcsGeometry
                ? assessmentSegments
                    .filter((segment) => segment.lnavVnavOcs)
                    .map((segment, index) => {
                      const ocs = segment.lnavVnavOcs;
                      if (!ocs) return null;
                      const segmentPath = plotPointPath(
                        ocs.points,
                        domain,
                        plotWidth,
                        plotHeight,
                        marginLeft,
                        marginTop,
                        mode,
                      );
                      return (
                        <g key={`${route.routeId}-lnav-vnav-ocs-${segment.segmentId}-${index}`}>
                          {mode === "top" ? (
                            <>
                              {ocs.secondaryHalfWidthM ? (
                                <path
                                  d={segmentPath}
                                  className="runway-profile-lnav-vnav-ocs-secondary-band"
                                  clipPath={`url(#${plotClipId})`}
                                  style={{
                                    strokeWidth: Math.max(2, ocs.secondaryHalfWidthM * yScale * 2),
                                  }}
                                />
                              ) : null}
                              <path
                                d={segmentPath}
                                className="runway-profile-lnav-vnav-ocs-primary-band"
                                clipPath={`url(#${plotClipId})`}
                                style={{
                                  strokeWidth: Math.max(2, ocs.primaryHalfWidthM * yScale * 2),
                                }}
                              />
                            </>
                          ) : (
                            <path
                              d={segmentPath}
                              className="runway-profile-lnav-vnav-ocs-line"
                              clipPath={`url(#${plotClipId})`}
                              data-segment-id={segment.segmentId}
                            />
                          )}
                        </g>
                      );
                    })
                : null}
              {showFinalVerticalReferenceGeometry
                ? assessmentSegments
                    .filter((segment) => segment.finalVerticalReference)
                    .map((segment, index) => {
                      const reference = segment.finalVerticalReference;
                      if (!reference) return null;
                      const referencePath = plotPointPath(
                        reference.points,
                        domain,
                        plotWidth,
                        plotHeight,
                        marginLeft,
                        marginTop,
                        mode,
                      );
                      const labelPoint =
                        reference.points[Math.max(0, Math.floor((reference.points.length - 1) / 2))];
                      const referenceLabelX = labelPoint ? plotX(labelPoint.xM) + 7 : 0;
                      const referenceLabelY = labelPoint
                        ? plotY(mode === "side" ? labelPoint.zM : labelPoint.yM) - 9
                        : 0;
                      return (
                        <g key={`${route.routeId}-final-vertical-reference-${segment.segmentId}-${index}`}>
                          {mode === "top" ? (
                            <path
                              d={referencePath}
                              className="runway-profile-final-vertical-reference-band"
                              clipPath={`url(#${plotClipId})`}
                              style={{
                                strokeWidth: Math.max(2, reference.halfWidthM * yScale * 2),
                              }}
                              data-segment-id={segment.segmentId}
                            />
                          ) : null}
                          <path
                            d={referencePath}
                            className={
                              mode === "side"
                                ? "runway-profile-final-vertical-reference-line"
                                : "runway-profile-final-vertical-reference-plan-line"
                            }
                            clipPath={`url(#${plotClipId})`}
                            data-segment-id={segment.segmentId}
                          />
                          <text
                            x={referenceLabelX}
                            y={referenceLabelY}
                            className="runway-profile-final-vertical-reference-label"
                            data-segment-id={segment.segmentId}
                          >
                            {reference.label}
                          </text>
                        </g>
                      );
                    })
                : null}
              <path
                d={d}
                className={`runway-profile-route-line ${
                  isTransition ? "is-transition" : "is-final"
                }`}
              />
              {route.points.map((point, index) => (
                <circle
                  key={`${route.routeId}-point-${point.fixIdent}-${index}`}
                  cx={plotX(point.xM)}
                  cy={plotY(mode === "side" ? point.zM : point.yM)}
                  r={4.1}
                  className="runway-profile-reference-point"
                  data-route-id={route.routeId}
                  data-fix-ident={point.fixIdent}
                />
              ))}
              {mode === "side" && showAltitudeConstraintGeometry
                ? route.points
                    .filter((point) => point.altitudeConstraint !== null)
                    .map((point, index) => {
                      const constraint = point.altitudeConstraint;
                      if (!constraint) return null;
                      const cx = plotX(point.xM);
                      const cy = plotY(point.zM);
                      const constraintClassName = altitudeConstraintClassName(constraint);
                      const label = altitudeConstraintLabel(point.fixIdent, constraint);
                      return (
                        <g key={`${route.routeId}-altitude-constraint-${point.fixIdent}-${index}`}>
                          <line
                            x1={cx}
                            y1={marginTop}
                            x2={cx}
                            y2={marginTop + plotHeight}
                            className={`runway-profile-altitude-constraint-station-line ${constraintClassName}`}
                          />
                          <line
                            x1={cx}
                            y1={plotY(0)}
                            x2={cx}
                            y2={cy}
                            className={`runway-profile-altitude-constraint-link ${constraintClassName}`}
                          />
                          <circle
                            cx={cx}
                            cy={cy}
                            r={5.2}
                            className={`runway-profile-altitude-constraint-point ${constraintClassName}`}
                            data-constraint-route-id={route.routeId}
                            data-constraint-fix-ident={point.fixIdent}
                          />
                          <text
                            x={cx + 7}
                            y={cy - 7}
                            className={`runway-profile-altitude-constraint-label ${constraintClassName}`}
                          >
                            {label}
                          </text>
                        </g>
                      );
                    })
                : null}
              {mode === "top" && isTransition ? (
                <text
                  x={labelX + 7}
                  y={labelY - 7}
                  className="runway-profile-route-label is-transition"
                >
                  {routeLabel(route)}
                </text>
              ) : null}
              {showDebugFinalSurfaceGeometry
                ? assessmentSegments.flatMap((segment, index) => {
                    const precisionSurfaceEntries = (segment.precisionSurfaces ?? []).map(
                      (surface, surfaceIndex) => {
                        const surfacePath = plotPointPath(
                          surface.points,
                          domain,
                          plotWidth,
                          plotHeight,
                          marginLeft,
                          marginTop,
                          mode,
                        );
                        const surfacePoint =
                          surface.points[Math.floor((surface.points.length - 1) / 2)];
                        return (
                          <g
                            key={`${route.routeId}-precision-surface-${segment.segmentId}-${surface.surfaceType}-${surfaceIndex}`}
                          >
                            <path
                              d={surfacePath}
                              className="runway-profile-precision-surface-line"
                              clipPath={`url(#${plotClipId})`}
                              data-segment-id={segment.segmentId}
                            />
                            {surfacePoint ? (
                              <text
                                x={plotX(surfacePoint.xM) + 6}
                                y={plotY(mode === "side" ? surfacePoint.zM : surfacePoint.yM) + 12}
                                className="runway-profile-segment-debug-label"
                                data-segment-id={segment.segmentId}
                              >
                                {surface.label}
                              </text>
                            ) : null}
                          </g>
                        );
                      },
                    );
                    const point = segment.points[Math.floor((segment.points.length - 1) / 2)];
                    if (!point) return precisionSurfaceEntries;
                    const textX = plotX(point.xM) + 6;
                    const textY = plotY(mode === "side" ? point.zM : point.yM) - 10;
                    return [
                      ...precisionSurfaceEntries,
                      <text
                        key={`${route.routeId}-segment-debug-${segment.segmentId}-${index}`}
                        x={textX}
                        y={textY}
                        className="runway-profile-segment-debug-label"
                        data-segment-id={segment.segmentId}
                      >
                        {segmentDebugLabel(segment, index)}
                      </text>,
                    ];
                  })
                : null}
            </g>
          );
        })}

        {tracks.map((track) => {
          const d = plotPointPath(
            track.trail,
            domain,
            plotWidth,
            plotHeight,
            marginLeft,
            marginTop,
            mode,
          );
          const currentYValue = mode === "side" ? track.current.zM : track.current.yM;
          const cx = plotX(track.current.xM);
          const cy = plotY(currentYValue);
          return (
            <g key={`${mode}-${track.flightId}`}>
              <path
                d={d}
                fill="none"
                stroke={track.color}
                strokeWidth={track.isSelected ? 3.2 : 2.2}
                strokeLinecap="round"
                strokeLinejoin="round"
                opacity={track.isSelected ? 0.96 : 0.74}
              />
              <circle
                cx={cx}
                cy={cy}
                r={track.isSelected ? 5.2 : 3.8}
                fill={track.color}
                stroke="rgba(15, 23, 42, 0.95)"
                strokeWidth={track.isSelected ? 2 : 1.4}
              />
              {track.isSelected ? (
                <text x={cx + 8} y={cy - 8} className="runway-profile-flight-label">
                  {track.flightId}
                </text>
              ) : null}
            </g>
          );
        })}

        <text
          x={marginLeft + plotWidth / 2}
          y={height - 12}
          textAnchor="middle"
          className="runway-profile-axis-label"
        >
          x: approach distance from threshold ({distanceUnitLabel(distanceUnit)})
        </text>
        <text
          x={18}
          y={marginTop + plotHeight / 2}
          textAnchor="middle"
          transform={`rotate(-90 18 ${marginTop + plotHeight / 2})`}
          className="runway-profile-axis-label"
        >
          {mode === "side"
            ? "z: height above threshold (ft)"
            : `y: lateral offset from centerline (${distanceUnitLabel(distanceUnit)})`}
        </text>

        <text x={18} y={marginTop + 10} className="runway-profile-axis-tick">
          {formatProfileAxisValue(mode, domain.maxY, distanceUnit)}
        </text>
        <text x={18} y={marginTop + plotHeight} className="runway-profile-axis-tick">
          {formatProfileAxisValue(mode, domain.minY, distanceUnit)}
        </text>
      </svg>
    </section>
  );
}

export default function RunwayTrajectoryProfilePanel() {
  const {
    activeAirportCode,
    isRunwayProfileOpen,
    selectedProfileRunwayIdent,
    runwayProfileViewMode,
    setRunwayProfileOpen,
    setRunwayProfileViewMode,
    trajectoryDataSource,
    procedureDisplayLevel,
  } = useApp();
  const profile = useRunwayTrajectoryProfile();
  const [distanceUnit, setDistanceUnit] = useState<DistanceUnit>("nm");

  if (!isRunwayProfileOpen || !selectedProfileRunwayIdent) return null;

  const viewModes: Array<{ label: string; value: RunwayProfileViewMode }> = [
    { label: "Split", value: "split" },
    { label: "Vertical", value: "side-xz" },
    { label: "Plan", value: "top-xy" },
  ];

  const routeCount = profile.plateRoutes.length;
  const trackCount = profile.aircraftTracks.length;
  const assessmentTrack =
    profile.aircraftTracks.find((track) => track.isSelected) ?? profile.aircraftTracks[0];

  return (
    <aside className="runway-profile-panel" aria-label="Runway trajectory profile">
      <header className="runway-profile-panel-header">
        <div>
          <h3>
            {activeAirportCode} {selectedProfileRunwayIdent} trajectory profile
          </h3>
          <p>
            Time: {formatIsoTime(profile.currentTimeIso)}{" "}
            {profile.sourceCycle ? `· CIFP ${profile.sourceCycle}` : ""}
          </p>
        </div>

        <div className="runway-profile-panel-actions">
          <div className="runway-profile-view-modes" role="tablist" aria-label="Profile view mode">
            {viewModes.map((mode) => (
              <button
                key={mode.value}
                type="button"
                className={runwayProfileViewMode === mode.value ? "active" : ""}
                onClick={() => setRunwayProfileViewMode(mode.value)}
              >
                {mode.label}
              </button>
            ))}
          </div>
          <div className="runway-profile-unit-switch" role="group" aria-label="Distance unit">
            {(["nm", "m"] as const).map((unit) => (
              <button
                key={unit}
                type="button"
                className={distanceUnit === unit ? "active" : ""}
                onClick={() => setDistanceUnit(unit)}
              >
                {unit === "nm" ? "NM" : "m"}
              </button>
            ))}
          </div>
          <button type="button" onClick={() => setRunwayProfileOpen(false)}>
            Close
          </button>
        </div>
      </header>

      <div className="runway-profile-summary">
        <span>{routeCount} active RNAV branches in plate</span>
        <span>{trackCount} aircraft currently inside plate</span>
        <span>{trajectoryDataSource ? "CZML linked" : "CZML missing"}</span>
        {assessmentTrack ? (
          <span>
            {assessmentTrack.flightId}: {assessmentTrack.current.segmentAssessment.activeSegmentId} ·
            station {formatDistance(assessmentTrack.current.segmentAssessment.stationM, distanceUnit)} ·
            xtrack{" "}
            {formatSignedDistance(
              assessmentTrack.current.segmentAssessment.crossTrackErrorM,
              distanceUnit,
            )}
            {assessmentTrack.current.segmentAssessment.verticalErrorM !== null
              ? ` · verr ${formatSignedFeet(
                  assessmentTrack.current.segmentAssessment.verticalErrorM,
                )}`
              : ""}
          </span>
        ) : null}
      </div>

      {profile.procedureNames.length > 0 ? (
        <p className="runway-profile-procedure-list">
          {profile.procedureNames.join(" · ")}
        </p>
      ) : null}

      {profile.error ? <p className="runway-profile-error">{profile.error}</p> : null}
      {!profile.error && !profile.isLoading && routeCount === 0 ? (
        <p className="runway-profile-empty">
          No RNAV arrival branches were found for this runway.
        </p>
      ) : null}
      {!profile.error && !profile.isLoading && routeCount > 0 && trackCount === 0 ? (
        <p className="runway-profile-empty">
          No aircraft are inside the RNAV horizontal plate at the current simulation time.
        </p>
      ) : null}

      {profile.isLoading ? <p className="runway-profile-empty">Loading runway profile…</p> : null}

      {!profile.isLoading && !profile.error && routeCount > 0 ? (
        <div
          className={`runway-profile-plots runway-profile-plots-${runwayProfileViewMode}`}
        >
          {runwayProfileViewMode !== "top-xy" ? (
            <ProfilePlot
              title="Vertical profile"
              subtitle="Runway-aligned x-z profile: follows the active procedure branch selected in the 3D procedure panel"
              mode="side"
              displayLevel={procedureDisplayLevel}
              distanceUnit={distanceUnit}
              tracks={profile.aircraftTracks}
              plateRoutes={profile.plateRoutes}
              referenceMarks={profile.referenceMarks}
            />
          ) : null}
          {runwayProfileViewMode !== "side-xz" ? (
            <ProfilePlot
              title="Plan view"
              subtitle="Runway-centered x-y plate: shows active procedure branches selected in the 3D procedure panel"
              mode="top"
              displayLevel={procedureDisplayLevel}
              distanceUnit={distanceUnit}
              tracks={profile.aircraftTracks}
              plateRoutes={profile.plateRoutes}
              referenceMarks={profile.referenceMarks}
            />
          ) : null}
        </div>
      ) : null}
    </aside>
  );
}
