import { useEffect, useMemo, useRef, useState } from "react";

import { useApp } from "../context/AppContext";
import type {
  ProcedureChartManifestEntry,
  ProcedureChartsManifest,
  ProcedureDetailBranch,
  ProcedureDetailDocument,
  ProcedureDetailFix,
  ProcedureDetailLeg,
  ProcedureDetailsIndexManifest,
  ProcedureDetailsIndexRunwaySummary,
} from "../data/procedureDetails";
import { normalizeProcedurePackage } from "../data/procedurePackageAdapter";
import {
  buildProcedureRenderBundle,
  type ProcedureRenderBundle,
} from "../data/procedureRenderBundle";
import {
  procedureChartsIndexUrl,
  procedureDetailsDocumentUrl,
  procedureDetailsIndexUrl,
} from "../data/procedureDetails";
import {
  contextualTermDetails,
  contextualTerms,
  formatContextualTermMeaning,
  formatTermBrief,
  groupedTerms,
  isKnownGlossaryTerm,
  isSpecificFixRole,
  termDetails,
} from "../data/procedureTerms";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import { navigateWithinApp } from "../utils/navigation";
import {
  buildProcedureBranchPolylines,
  buildRunwayMarker,
  findFix,
  nmFromMeters,
  pointToEastNorth,
  procedureBranchForFix,
  type ProcedureBranchPolyline,
  type ProcedureChartPoint,
  type ProcedureRunwayMarker,
} from "../utils/procedureDetailsGeometry";

const SVG_WIDTH = 1120;
const PLAN_SVG_HEIGHT = 680;
const PROFILE_SVG_HEIGHT = 420;
const SVG_PADDING_X = 64;
const SVG_PADDING_Y = 44;
const PLAN_AXIS_TICK_COUNT = 10;
const PLAN_FIX_SYMBOL_SIZE = 7;
const PLAN_SELECTED_FIX_SYMBOL_SIZE = 8;
const PROFILE_AXIS_TICK_COUNT = 8;
const PROFILE_FIX_SYMBOL_SIZE = 6;
const PROFILE_SELECTED_FIX_SYMBOL_SIZE = 7;
const MISSED_OUTBOUND_ARROW_START_GAP_PX = 92;
const MISSED_OUTBOUND_PROJECTED_LENGTH_M = 6400;
const IMPORTANT_FIX_ROLES = new Set(["IAF", "IF", "FAF", "MAPT", "MAHF"]);
const METERS_PER_NM = 1852;

type DistanceUnit = "nm" | "m";

function readRouteParams(): {
  airport: string | null;
  runway: string | null;
  procedureUid: string | null;
} {
  const search = new URLSearchParams(window.location.search);
  if (window.location.hash.startsWith("#procedure-details?")) {
    const hashSearch = new URLSearchParams(window.location.hash.split("?")[1] ?? "");
    hashSearch.forEach((value, key) => {
      if (!search.has(key)) search.set(key, value);
    });
  }

  return {
    airport: search.get("airport"),
    runway: search.get("runway"),
    procedureUid: search.get("procedureUid"),
  };
}

function buildFaaChartUrl(faaCode: string): string {
  return `https://www.faa.gov/air_traffic/flight_info/aeronav/procedures/application/?event=procedure.results&nasrId=${encodeURIComponent(
    faaCode,
  )}#searchResultsTop`;
}

function formatRunway(runwayIdent: string | null | undefined): string {
  if (!runwayIdent) return "Unknown runway";
  return runwayIdent.startsWith("RW") ? `RWY ${runwayIdent.slice(2)}` : runwayIdent;
}

function formatAltitudeFt(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Not available";
  return `${Math.round(value).toLocaleString()} ft`;
}

function formatCoordinate(value: number | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Not available";
  return value.toFixed(6);
}

function formatFixRef(fixRef: string | null | undefined): string {
  if (!fixRef) return "Unknown fix";
  return fixRef.replace(/^fix:/, "");
}

function roleMeaning(role: string): string {
  return termDetails(role.toUpperCase()).definition;
}

function terminatorMeaning(pathTerminator: string): string {
  return termDetails(`${pathTerminator.toUpperCase()}_TERMINATOR`).definition;
}

function localChartForProcedure(
  charts: ProcedureChartManifestEntry[],
  procedureUid: string | null,
  runwayIdent: string | null,
): ProcedureChartManifestEntry | null {
  if (procedureUid) {
    const exact = charts.find((chart) => chart.procedureUid === procedureUid);
    if (exact) return exact;
  }
  if (runwayIdent) {
    return charts.find((chart) => chart.runwayIdent === runwayIdent) ?? null;
  }
  return null;
}

function chartScale(
  values: number[],
  minTarget: number,
  maxTarget: number,
): (value: number) => number {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1);
  return (value: number) => minTarget + ((value - min) / span) * (maxTarget - minTarget);
}

function chartDomain(values: number[], paddingRatio = 0.08): { min: number; max: number } {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1);
  return {
    min: min - span * paddingRatio,
    max: max + span * paddingRatio,
  };
}

function equalAspectChartDomains(
  xDomain: { min: number; max: number },
  yDomain: { min: number; max: number },
  plotWidth: number,
  plotHeight: number,
): { xDomain: { min: number; max: number }; yDomain: { min: number; max: number } } {
  const xSpan = Math.max(xDomain.max - xDomain.min, 1);
  const ySpan = Math.max(yDomain.max - yDomain.min, 1);
  const metersPerPixel = Math.max(xSpan / plotWidth, ySpan / plotHeight);
  const targetXSpan = metersPerPixel * plotWidth;
  const targetYSpan = metersPerPixel * plotHeight;
  const xCenter = (xDomain.min + xDomain.max) / 2;
  const yCenter = (yDomain.min + yDomain.max) / 2;

  return {
    xDomain: {
      min: xCenter - targetXSpan / 2,
      max: xCenter + targetXSpan / 2,
    },
    yDomain: {
      min: yCenter - targetYSpan / 2,
      max: yCenter + targetYSpan / 2,
    },
  };
}

function niceChartStep(span: number, tickCount: number): number {
  const roughStep = Math.max(span, 1) / Math.max(tickCount, 1);
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const residual = roughStep / magnitude;
  const niceResiduals = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10];
  const niceResidual =
    niceResiduals.find((candidate) => residual <= candidate) ??
    niceResiduals[niceResiduals.length - 1];
  return niceResidual * magnitude;
}

function chartTicksByStep(domain: { min: number; max: number }, step: number): number[] {
  const start = Math.ceil(domain.min / step) * step;
  const end = Math.floor(domain.max / step) * step;
  const ticks: number[] = [];

  for (let tick = start; tick <= end + step * 0.5; tick += step) {
    ticks.push(Math.round(tick * 1000) / 1000);
  }

  return ticks;
}

function formatSignedMeters(valueM: number): string {
  if (Math.abs(valueM) < 0.5) return "0 m";
  return `${valueM > 0 ? "+" : ""}${Math.round(valueM).toLocaleString()} m`;
}

function formatSignedDistance(valueM: number, unit: DistanceUnit): string {
  if (unit === "m") return formatSignedMeters(valueM);
  const valueNm = nmFromMeters(valueM);
  if (Math.abs(valueNm) < 0.05) return "0.0 NM";
  return `${valueNm > 0 ? "+" : ""}${valueNm.toFixed(1)} NM`;
}

function distanceUnitLabel(unit: DistanceUnit): string {
  return unit === "nm" ? "NM" : "m";
}

function horizontalTickStepM(spanM: number, tickCount: number, unit: DistanceUnit): number {
  if (unit === "nm") {
    return niceChartStep(spanM / METERS_PER_NM, tickCount) * METERS_PER_NM;
  }
  return niceChartStep(spanM, tickCount);
}

function hasNamedLegEndpoint(leg: ProcedureDetailLeg): boolean {
  const endFix = leg.path.endFixRef;
  return Boolean(endFix && formatFixRef(endFix).trim());
}

function legActionLabel(leg: ProcedureDetailLeg): string {
  const pathTerminator = leg.path.pathTerminator.toUpperCase();
  const altitudeText = formatAltitudeFt(
    leg.constraints.geometryAltitudeFt ?? leg.constraints.altitude?.valueFt,
  );

  if (pathTerminator === "CA") return `Course to ${altitudeText}`;
  if (pathTerminator === "VA") return `Heading to ${altitudeText}`;
  if (pathTerminator === "FA") return `Fix to ${altitudeText}`;
  if (pathTerminator === "DF") return "Direct to fix";
  if (pathTerminator === "CF") return "Course to fix";
  if (pathTerminator === "TF") return "Track to fix";
  if (pathTerminator === "IF") return "Initial fix";
  if (pathTerminator === "HM") return "Hold to manual termination";
  if (pathTerminator === "HF") return "Hold to fix";
  if (pathTerminator === "HA") return "Hold to altitude";
  return `${pathTerminator} leg`;
}

function legEndpointLabel(leg: ProcedureDetailLeg, endFix: ProcedureDetailFix | null): string {
  if (endFix) return endFix.ident;
  if (hasNamedLegEndpoint(leg)) {
    return formatFixRef(leg.path.endFixRef);
  }
  return legActionLabel(leg);
}

function legRoleLabel(leg: ProcedureDetailLeg): string {
  if (isSpecificFixRole(leg.roleAtEnd)) return formatTermBrief(leg.roleAtEnd);
  return legActionLabel(leg);
}

function branchRoleLabel(branchRole: string): string {
  return branchRole === "final" ? "Final segment" : "Transition segment";
}

function branchDisplayLabel(branch: ProcedureDetailBranch): string {
  if (branch.transitionIdent) return branch.transitionIdent;
  return branch.branchIdent;
}

function branchTypeDisplayLabel(branch: ProcedureDetailBranch): string {
  const typePrefix = branch.procedureType
    ? `Approach route type ${branch.procedureType}`
    : "Approach route type";
  return `${typePrefix} · ${branchRoleLabel(branch.branchRole)}`;
}

function isImportantFixRole(role: string): boolean {
  return IMPORTANT_FIX_ROLES.has(role.toUpperCase());
}

function shouldShowPointLabel(
  point: ProcedureChartPoint,
  focusedFixId: string | null,
  focusedBranchId: string | null,
): boolean {
  return (
    point.fixId === focusedFixId ||
    point.branchId === focusedBranchId ||
    isImportantFixRole(point.role)
  );
}

function pointRole(point: ProcedureChartPoint): string {
  return point.role.toUpperCase();
}

function profileAnchorIndex(points: ProcedureChartPoint[]): number {
  const maptIndex = points.findIndex((point) => pointRole(point) === "MAPT");
  if (maptIndex >= 0) return maptIndex;
  return points.reduce((bestIndex, point, index) => {
    const best = points[bestIndex];
    return Math.hypot(point.xM, point.yM) < Math.hypot(best.xM, best.yM) ? index : bestIndex;
  }, 0);
}

interface StationedProfilePoint {
  point: ProcedureChartPoint;
  stationM: number;
}

function stationedProfilePoints(points: ProcedureChartPoint[]): StationedProfilePoint[] {
  if (points.length === 0) return [];
  const anchor = points[profileAnchorIndex(points)];
  return points.map((point) => ({
    point,
    stationM: point.distanceM - anchor.distanceM,
  }));
}

function splitPlanBranchPoints(branch: ProcedureBranchPolyline): {
  approachPoints: ProcedureChartPoint[];
  missedPoints: ProcedureChartPoint[];
} {
  if (branch.branchRole.toLowerCase() === "missed") {
    return { approachPoints: [], missedPoints: branch.points };
  }

  const missedStartIndex = branch.points.findIndex((point) => pointRole(point) === "MAPT");
  if (missedStartIndex < 0 || missedStartIndex === branch.points.length - 1) {
    return { approachPoints: branch.points, missedPoints: [] };
  }

  return {
    approachPoints: branch.points.slice(0, missedStartIndex + 1),
    missedPoints: branch.points.slice(missedStartIndex),
  };
}

function planSegments(
  points: ProcedureChartPoint[],
): Array<{ from: ProcedureChartPoint; to: ProcedureChartPoint; isOutbound: boolean }> {
  return points.slice(1).map((point, index) => ({
    from: points[index],
    to: point,
    isOutbound: index === points.length - 2,
  }));
}

function clippedSegmentCoords(
  from: { xM: number; yM: number },
  to: { xM: number; yM: number },
  scaleX: (value: number) => number,
  scaleY: (value: number) => number,
  endGap = 15,
  startGap = 8,
): { x1: number; y1: number; x2: number; y2: number } {
  const rawX1 = scaleX(from.xM);
  const rawY1 = scaleY(from.yM);
  const rawX2 = scaleX(to.xM);
  const rawY2 = scaleY(to.yM);
  const dx = rawX2 - rawX1;
  const dy = rawY2 - rawY1;
  const length = Math.hypot(dx, dy);

  if (length <= 4) {
    return { x1: rawX1, y1: rawY1, x2: rawX2, y2: rawY2 };
  }

  const ux = dx / length;
  const uy = dy / length;
  const effectiveStartGap = Math.min(startGap, length * 0.35);
  const effectiveEndGap = Math.min(endGap, length * 0.35);

  return {
    x1: rawX1 + ux * effectiveStartGap,
    y1: rawY1 + uy * effectiveStartGap,
    x2: rawX2 - ux * effectiveEndGap,
    y2: rawY2 - uy * effectiveEndGap,
  };
}

function missedOutboundArrowForPoints(
  missedPoints: ProcedureChartPoint[],
): {
  anchor: ProcedureChartPoint;
  target: { xM: number; yM: number };
  hasTerminalFix: boolean;
} | null {
  if (missedPoints.length < 2) return null;

  const startsAtMapt = pointRole(missedPoints[0]) === "MAPT";
  const anchorIndex = startsAtMapt ? 1 : 0;
  const anchor = missedPoints[anchorIndex];
  const procedureTarget = missedPoints[anchorIndex + 1];

  if (procedureTarget) {
    return { anchor, target: procedureTarget, hasTerminalFix: true };
  }

  const previous = missedPoints[anchorIndex - 1];
  if (!previous) return null;

  const dx = anchor.xM - previous.xM;
  const dy = anchor.yM - previous.yM;
  const length = Math.hypot(dx, dy);
  if (length === 0) return null;

  const ux = dx / length;
  const uy = dy / length;
  return {
    anchor,
    target: {
      xM: anchor.xM + ux * MISSED_OUTBOUND_PROJECTED_LENGTH_M,
      yM: anchor.yM + uy * MISSED_OUTBOUND_PROJECTED_LENGTH_M,
    },
    hasTerminalFix: false,
  };
}

function missedInboundSegments(
  segments: Array<{ from: ProcedureChartPoint; to: ProcedureChartPoint; isOutbound: boolean }>,
  arrow: ReturnType<typeof missedOutboundArrowForPoints>,
): Array<{ from: ProcedureChartPoint; to: ProcedureChartPoint; isOutbound: boolean }> {
  if (!arrow) return segments;
  return segments.filter((segment) => segment.to.fixId === arrow.anchor.fixId);
}

function focusedLegDescription(leg: ProcedureDetailLeg): string {
  const pathTermMeaning = terminatorMeaning(leg.path.pathTerminator);
  if (isSpecificFixRole(leg.roleAtEnd)) {
    return `${pathTermMeaning} This leg ends at the ${leg.roleAtEnd} point.`;
  }
  if (!hasNamedLegEndpoint(leg)) {
    return `${pathTermMeaning} This leg does not terminate at a named fix.`;
  }
  return pathTermMeaning;
}

function legRfMetadataLabels(leg: ProcedureDetailLeg): string[] {
  if (leg.path.pathTerminator.toUpperCase() !== "RF") return [];

  const labels: string[] = [];
  if (leg.path.turnDirection) labels.push(`Turn ${leg.path.turnDirection}`);
  if (typeof leg.path.arcRadiusNm === "number" && Number.isFinite(leg.path.arcRadiusNm)) {
    labels.push(`Radius ${leg.path.arcRadiusNm.toFixed(2)} NM`);
  }
  if (leg.path.centerFixRef) {
    labels.push(`Center ${formatFixRef(leg.path.centerFixRef)}`);
  } else if (
    typeof leg.path.centerLatDeg === "number" &&
    Number.isFinite(leg.path.centerLatDeg) &&
    typeof leg.path.centerLonDeg === "number" &&
    Number.isFinite(leg.path.centerLonDeg)
  ) {
    labels.push(
      `Center ${formatCoordinate(leg.path.centerLatDeg)}, ${formatCoordinate(leg.path.centerLonDeg)}`,
    );
  }
  return labels;
}

interface RfPlanMarker {
  branchId: string;
  legId: string;
  centerX: number;
  centerY: number;
  endpointX: number;
  endpointY: number;
  radiusM: number;
  label: string;
}

interface MissedSectionMarker {
  branchId: string;
  segmentId: string;
  point: ProcedureChartPoint;
  label: string;
}

interface MissedLegMarker {
  branchId: string;
  legId: string;
  point: ProcedureChartPoint;
  label: string;
  courseTarget?: { xM: number; yM: number };
}

const MISSED_LEG_MARKER_TYPES = new Set(["CA", "DF", "HM", "HA", "HF"]);

function buildRfPlanMarkers(
  document: ProcedureDetailDocument | null,
  polylines: ProcedureBranchPolyline[],
): RfPlanMarker[] {
  if (!document) return [];
  const origin =
    document.runway.threshold ??
    document.fixes.find((fix) => fix.position)?.position ?? {
      lon: 0,
      lat: 0,
    };
  const pointByBranchAndFix = new Map(
    polylines.flatMap((branch) =>
      branch.points.map((point) => [`${branch.branchId}:${point.fixId}`, point] as const),
    ),
  );

  return document.branches.flatMap((branch) =>
    branch.legs.flatMap((leg): RfPlanMarker[] => {
      if (
        leg.path.pathTerminator.toUpperCase() !== "RF" ||
        typeof leg.path.centerLatDeg !== "number" ||
        !Number.isFinite(leg.path.centerLatDeg) ||
        typeof leg.path.centerLonDeg !== "number" ||
        !Number.isFinite(leg.path.centerLonDeg) ||
        typeof leg.path.arcRadiusNm !== "number" ||
        !Number.isFinite(leg.path.arcRadiusNm)
      ) {
        return [];
      }
      const endpoint = pointByBranchAndFix.get(`${branch.branchId}:${leg.path.endFixRef}`);
      if (!endpoint) return [];

      const center = pointToEastNorth(
        leg.path.centerLonDeg,
        leg.path.centerLatDeg,
        origin.lon,
        origin.lat,
      );
      return [
        {
          branchId: branch.branchId,
          legId: leg.legId,
          centerX: center.east,
          centerY: center.north,
          endpointX: endpoint.xM,
          endpointY: endpoint.yM,
          radiusM: leg.path.arcRadiusNm * METERS_PER_NM,
          label: leg.path.centerFixRef ? formatFixRef(leg.path.centerFixRef) : "RF center",
        },
      ];
    }),
  );
}

function buildMissedLegMarkers(
  document: ProcedureDetailDocument | null,
  renderBundle: ProcedureRenderBundle | null,
  polylines: ProcedureBranchPolyline[],
): MissedLegMarker[] {
  if (!document || !renderBundle) return [];
  const origin =
    document.runway.threshold ??
    document.fixes.find((fix) => fix.position)?.position ?? {
      lon: 0,
      lat: 0,
    };
  const branchByScopedId = new Map(
    polylines.map((branch) => [scopedBranchIdFor(document, branch), branch]),
  );
  const pointByBranchAndFix = new Map(
    polylines.flatMap((branch) =>
      branch.points.map((point) => [`${branch.branchId}:${point.fixId}`, point] as const),
    ),
  );

  return renderBundle.branchBundles.flatMap((branchBundle) => {
    const branch = branchByScopedId.get(branchBundle.branchId);
    if (!branch) return [];

    return branchBundle.segmentBundles.flatMap((segmentBundle): MissedLegMarker[] => {
      if (
        segmentBundle.segment.segmentType !== "MISSED_S1" &&
        segmentBundle.segment.segmentType !== "MISSED_S2"
      ) {
        return [];
      }

      const courseGuideByLegId = new Map(
        segmentBundle.missedCourseGuides.map((guide) => [guide.legId, guide] as const),
      );

      return segmentBundle.legs.flatMap((leg): MissedLegMarker[] => {
        if (!MISSED_LEG_MARKER_TYPES.has(leg.legType)) return [];

        const anchorPoint =
          (leg.endFixId ? pointByBranchAndFix.get(`${branch.branchId}:${leg.endFixId}`) : undefined) ??
          (leg.startFixId ? pointByBranchAndFix.get(`${branch.branchId}:${leg.startFixId}`) : undefined);
        if (!anchorPoint) return [];

        const courseGuide = courseGuideByLegId.get(leg.legId);
        const projectedTarget = courseGuide
          ? pointToEastNorth(
              courseGuide.geoPositions[1].lonDeg,
              courseGuide.geoPositions[1].latDeg,
              origin.lon,
              origin.lat,
            )
          : null;

        return [
          {
            branchId: branch.branchId,
            legId: leg.legId,
            point: anchorPoint,
            label: courseGuide ? `CA ${courseGuide.courseDeg.toFixed(0)} deg` : `${leg.legType} leg`,
            courseTarget: projectedTarget
              ? { xM: projectedTarget.east, yM: projectedTarget.north }
              : undefined,
          },
        ];
      });
    });
  });
}

function scopedBranchIdFor(document: ProcedureDetailDocument, branch: ProcedureBranchPolyline): string {
  const packageId = `${document.airport.icao.toUpperCase()}-${document.procedure.procedureIdent.toUpperCase()}-${
    document.runway.ident ?? "UNKNOWN"
  }`;
  return `${packageId}:branch:${branch.branchKey.toUpperCase()}`;
}

function buildMissedSectionMarkers(
  document: ProcedureDetailDocument | null,
  renderBundle: ProcedureRenderBundle | null,
  polylines: ProcedureBranchPolyline[],
): MissedSectionMarker[] {
  if (!document || !renderBundle) return [];
  const branchByScopedId = new Map(
    polylines.map((branch) => [scopedBranchIdFor(document, branch), branch]),
  );

  return renderBundle.branchBundles.flatMap((branchBundle) => {
    const branch = branchByScopedId.get(branchBundle.branchId);
    if (!branch) return [];

    return branchBundle.segmentBundles.flatMap((segmentBundle): MissedSectionMarker[] => {
      if (segmentBundle.segment.segmentType !== "MISSED_S2") return [];
      const splitFixId = segmentBundle.segment.startFixId;
      if (!splitFixId) return [];
      const point = branch.points.find((candidate) => candidate.fixId === splitFixId);
      if (!point) return [];

      return [
        {
          branchId: branch.branchId,
          segmentId: segmentBundle.segment.segmentId,
          point,
          label: "S1/S2",
        },
      ];
    });
  });
}

function selectedBranchDefaultId(document: ProcedureDetailDocument | null): string | null {
  if (!document) return null;
  return (
    document.branches.find((branch) => branch.branchIdent === document.procedure.baseBranchIdent)
      ?.branchId ??
    document.branches[0]?.branchId ??
    null
  );
}

interface SvgChartProps {
  polylines: ProcedureBranchPolyline[];
  runwayMarker: ProcedureRunwayMarker | null;
  rfMarkers: RfPlanMarker[];
  missedSectionMarkers: MissedSectionMarker[];
  missedLegMarkers: MissedLegMarker[];
  focusedFixId: string | null;
  focusedBranchId: string | null;
  distanceUnit: DistanceUnit;
  onPreviewFix: (fixId: string | null, branchId: string | null) => void;
  onSelectFix: (fixId: string, branchId: string) => void;
  onPreviewBranch: (branchId: string | null) => void;
  onSelectBranch: (branchId: string) => void;
}

function ProcedurePlanView({
  polylines,
  runwayMarker,
  rfMarkers,
  missedSectionMarkers,
  missedLegMarkers,
  focusedFixId,
  focusedBranchId,
  distanceUnit,
  onPreviewFix,
  onSelectFix,
  onPreviewBranch,
  onSelectBranch,
}: SvgChartProps) {
  const allPoints = polylines.flatMap((branch) => branch.points);
  if (allPoints.length === 0) {
    return (
      <div className="procedure-details-empty-chart">
        No positioned fixes available for the plan view yet.
      </div>
    );
  }

  const missedOutboundArrowTargets = polylines.flatMap((branch) => {
    const { missedPoints } = splitPlanBranchPoints(branch);
    const arrow = missedOutboundArrowForPoints(missedPoints);
    return arrow ? [arrow.target] : [];
  });
  const rfDomainPoints = rfMarkers.flatMap((marker) => [
    { xM: marker.centerX, yM: marker.centerY },
    { xM: marker.centerX - marker.radiusM, yM: marker.centerY },
    { xM: marker.centerX + marker.radiusM, yM: marker.centerY },
    { xM: marker.centerX, yM: marker.centerY - marker.radiusM },
    { xM: marker.centerX, yM: marker.centerY + marker.radiusM },
  ]);
  const domainPoints = [
    ...allPoints,
    ...missedOutboundArrowTargets,
    ...rfDomainPoints,
    ...missedSectionMarkers.map((marker) => marker.point),
    ...missedLegMarkers.map((marker) => marker.point),
    ...missedLegMarkers.flatMap((marker) => marker.courseTarget ? [marker.courseTarget] : []),
  ];
  const plotWidth = SVG_WIDTH - SVG_PADDING_X * 2;
  const plotHeight = PLAN_SVG_HEIGHT - SVG_PADDING_Y * 2;
  const rawXDomain = chartDomain(domainPoints.map((point) => point.xM), 0.18);
  const rawYDomain = chartDomain(domainPoints.map((point) => point.yM), 0.18);
  const { xDomain, yDomain } = equalAspectChartDomains(
    rawXDomain,
    rawYDomain,
    plotWidth,
    plotHeight,
  );
  const scaleX = chartScale(
    [xDomain.min, xDomain.max],
    SVG_PADDING_X,
    SVG_WIDTH - SVG_PADDING_X,
  );
  const scaleY = chartScale(
    [yDomain.min, yDomain.max],
    PLAN_SVG_HEIGHT - SVG_PADDING_Y,
    SVG_PADDING_Y,
  );
  const planTickStep = horizontalTickStepM(
    Math.max(xDomain.max - xDomain.min, yDomain.max - yDomain.min),
    PLAN_AXIS_TICK_COUNT,
    distanceUnit,
  );
  const xTicks = chartTicksByStep(xDomain, planTickStep);
  const yTicks = chartTicksByStep(yDomain, planTickStep);
  const zeroY = yDomain.min <= 0 && yDomain.max >= 0 ? scaleY(0) : null;
  const zeroX = xDomain.min <= 0 && xDomain.max >= 0 ? scaleX(0) : null;

  return (
    <svg
      className="procedure-details-chart-svg"
      viewBox={`0 0 ${SVG_WIDTH} ${PLAN_SVG_HEIGHT}`}
      role="img"
      aria-label="Procedure plan view"
      onMouseLeave={() => {
        onPreviewFix(null, null);
        onPreviewBranch(null);
      }}
    >
      <defs>
        <marker
          id="procedure-details-arrowhead"
          viewBox="0 0 10 10"
          refX="7"
          refY="5"
          markerWidth="3.4"
          markerHeight="3.4"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" className="procedure-details-arrowhead" />
        </marker>
        <marker
          id="procedure-details-missed-outbound-arrowhead"
          viewBox="0 0 12 12"
          refX="8.5"
          refY="6"
          markerWidth="3.8"
          markerHeight="3.8"
          orient="auto-start-reverse"
        >
          <path d="M 1 1 L 11 6 L 1 11 z" className="procedure-details-arrowhead" />
        </marker>
      </defs>

      <rect
        x={SVG_PADDING_X}
        y={SVG_PADDING_Y}
        width={plotWidth}
        height={plotHeight}
        className="procedure-details-plot-frame"
      />

      {xTicks.map((tick) => {
        const x = scaleX(tick);
        return (
          <g key={`plan-x-${tick}`}>
            <line
              x1={x}
              y1={SVG_PADDING_Y}
              x2={x}
              y2={PLAN_SVG_HEIGHT - SVG_PADDING_Y}
              className="procedure-details-grid-line"
            />
            <line
              x1={x}
              y1={PLAN_SVG_HEIGHT - SVG_PADDING_Y}
              x2={x}
              y2={PLAN_SVG_HEIGHT - SVG_PADDING_Y + 6}
              className="procedure-details-axis-tick"
            />
            <text
              x={x}
              y={PLAN_SVG_HEIGHT - SVG_PADDING_Y + 24}
              className="procedure-details-axis-label is-centered"
            >
              {formatSignedDistance(tick, distanceUnit)}
            </text>
          </g>
        );
      })}

      {yTicks.map((tick) => {
        const y = scaleY(tick);
        return (
          <g key={`plan-y-${tick}`}>
            <line
              x1={SVG_PADDING_X}
              y1={y}
              x2={SVG_WIDTH - SVG_PADDING_X}
              y2={y}
              className="procedure-details-grid-line"
            />
            <line
              x1={SVG_PADDING_X - 6}
              y1={y}
              x2={SVG_PADDING_X}
              y2={y}
              className="procedure-details-axis-tick"
            />
            <text x={SVG_PADDING_X - 12} y={y + 4} className="procedure-details-axis-label">
              {formatSignedDistance(tick, distanceUnit)}
            </text>
          </g>
        );
      })}

      {zeroY !== null ? (
        <line
          x1={SVG_PADDING_X}
          y1={zeroY}
          x2={SVG_WIDTH - SVG_PADDING_X}
          y2={zeroY}
          className="procedure-details-axis-line"
        />
      ) : null}
      {zeroX !== null ? (
        <line
          x1={zeroX}
          y1={SVG_PADDING_Y}
          x2={zeroX}
          y2={PLAN_SVG_HEIGHT - SVG_PADDING_Y}
          className="procedure-details-axis-line"
        />
      ) : null}

      <text
        x={SVG_WIDTH - SVG_PADDING_X}
        y={PLAN_SVG_HEIGHT - 10}
        className="procedure-details-axis-title"
      >
        East offset from origin ({distanceUnitLabel(distanceUnit)})
      </text>
      <text
        x={18}
        y={SVG_PADDING_Y}
        transform={`rotate(-90 18 ${SVG_PADDING_Y})`}
        className="procedure-details-axis-title is-vertical"
      >
        North offset from origin ({distanceUnitLabel(distanceUnit)})
      </text>

      {polylines.map((branch) => {
        const isFocused = !focusedBranchId || branch.branchId === focusedBranchId;
        const { approachPoints, missedPoints } = splitPlanBranchPoints(branch);
        const approachSegments = planSegments(approachPoints);
        const missedSegments = planSegments(missedPoints);
        const missedOutboundArrow = missedOutboundArrowForPoints(missedPoints);
        const missedLinkSegments = missedInboundSegments(missedSegments, missedOutboundArrow);

        return (
          <g key={branch.branchId}>
            {approachSegments.map((segment, index) => (
              (() => {
                const coords = clippedSegmentCoords(segment.from, segment.to, scaleX, scaleY, 28, 14);
                return (
                  <line
                    key={`${branch.branchId}-approach-${segment.from.fixId}-${segment.to.fixId}-${index}`}
                    x1={coords.x1}
                    y1={coords.y1}
                    x2={coords.x2}
                    y2={coords.y2}
                    className={`procedure-details-branch-line procedure-details-branch-${branch.branchRole} ${
                      isFocused ? "is-focused" : "is-muted"
                    }`}
                    markerEnd="url(#procedure-details-arrowhead)"
                    onMouseEnter={() => onPreviewBranch(branch.branchId)}
                    onClick={() => onSelectBranch(branch.branchId)}
                  />
                );
              })()
            ))}
            {missedLinkSegments.map((segment, index) => {
              const coords = clippedSegmentCoords(
                segment.from,
                segment.to,
                scaleX,
                scaleY,
                28,
                18,
              );
              return (
                <line
                  key={`${branch.branchId}-missed-${segment.from.fixId}-${segment.to.fixId}-${index}`}
                  x1={coords.x1}
                  y1={coords.y1}
                  x2={coords.x2}
                  y2={coords.y2}
                  className={`procedure-details-branch-line procedure-details-missed-line ${
                    isFocused ? "is-focused" : "is-muted"
                  }`}
                  onMouseEnter={() => onPreviewBranch(branch.branchId)}
                  onClick={() => onSelectBranch(branch.branchId)}
                />
              );
            })}
            {missedOutboundArrow
              ? (() => {
                  const coords = clippedSegmentCoords(
                    missedOutboundArrow.anchor,
                    missedOutboundArrow.target,
                    scaleX,
                    scaleY,
                    missedOutboundArrow.hasTerminalFix ? 28 : 0,
                    MISSED_OUTBOUND_ARROW_START_GAP_PX,
                  );
                  return (
                    <line
                      key={`${branch.branchId}-missed-outbound-arrow`}
                      x1={coords.x1}
                      y1={coords.y1}
                      x2={coords.x2}
                      y2={coords.y2}
                      className={`procedure-details-branch-line procedure-details-missed-line is-outbound ${
                        isFocused ? "is-focused" : "is-muted"
                      }`}
                      markerEnd="url(#procedure-details-missed-outbound-arrowhead)"
                      onMouseEnter={() => onPreviewBranch(branch.branchId)}
                      onClick={() => onSelectBranch(branch.branchId)}
                    />
                  );
                })()
              : null}
          </g>
        );
      })}

      {missedSectionMarkers.map((marker) => {
        const isFocused = !focusedBranchId || marker.branchId === focusedBranchId;
        const x = scaleX(marker.point.xM);
        const y = scaleY(marker.point.yM);
        return (
          <g
            key={`missed-section-${marker.segmentId}`}
            className={`procedure-details-missed-section-marker ${
              isFocused ? "is-focused" : "is-muted"
            }`}
            onMouseEnter={() => onPreviewFix(marker.point.fixId, marker.branchId)}
            onClick={() => onSelectFix(marker.point.fixId, marker.branchId)}
          >
            <circle cx={x} cy={y} r={8} />
            <text x={x + 11} y={y - 10}>{marker.label}</text>
          </g>
        );
      })}

      {missedLegMarkers.map((marker) => {
        const isFocused = !focusedBranchId || marker.branchId === focusedBranchId;
        const x = scaleX(marker.point.xM);
        const y = scaleY(marker.point.yM);
        const courseTarget = marker.courseTarget
          ? { x: scaleX(marker.courseTarget.xM), y: scaleY(marker.courseTarget.yM) }
          : null;
        return (
          <g
            key={`missed-leg-${marker.legId}`}
            className={`procedure-details-missed-leg-marker ${
              isFocused ? "is-focused" : "is-muted"
            }`}
            onMouseEnter={() => onPreviewFix(marker.point.fixId, marker.branchId)}
            onClick={() => onSelectFix(marker.point.fixId, marker.branchId)}
          >
            {courseTarget ? (
              <line
                x1={x}
                y1={y}
                x2={courseTarget.x}
                y2={courseTarget.y}
                className="procedure-details-ca-course-ray"
              />
            ) : null}
            <rect x={x - 14} y={y + 11} width={54} height={18} rx={4} />
            <text x={x - 8} y={y + 24}>{marker.label}</text>
          </g>
        );
      })}

      {runwayMarker ? (
        <g className="procedure-details-runway">
          <line
            x1={scaleX(runwayMarker.x1)}
            y1={scaleY(runwayMarker.y1)}
            x2={scaleX(runwayMarker.x2)}
            y2={scaleY(runwayMarker.y2)}
            className="procedure-details-runway-casing"
          />
          <line
            x1={scaleX(runwayMarker.x1)}
            y1={scaleY(runwayMarker.y1)}
            x2={scaleX(runwayMarker.x2)}
            y2={scaleY(runwayMarker.y2)}
            className="procedure-details-runway-bar"
          />
          <circle
            cx={scaleX(runwayMarker.centerX)}
            cy={scaleY(runwayMarker.centerY)}
            r={5}
            className="procedure-details-runway-threshold"
          />
        </g>
      ) : null}

      {rfMarkers.map((marker) => {
        const isFocused = !focusedBranchId || marker.branchId === focusedBranchId;
        const centerX = scaleX(marker.centerX);
        const centerY = scaleY(marker.centerY);
        const endpointX = scaleX(marker.endpointX);
        const endpointY = scaleY(marker.endpointY);
        const radiusPx = Math.abs(scaleX(marker.centerX + marker.radiusM) - centerX);
        return (
          <g
            key={marker.legId}
            className={`procedure-details-rf-marker ${isFocused ? "is-focused" : "is-muted"}`}
          >
            <circle
              cx={centerX}
              cy={centerY}
              r={radiusPx}
              className="procedure-details-rf-radius"
            />
            <line
              x1={centerX}
              y1={centerY}
              x2={endpointX}
              y2={endpointY}
              className="procedure-details-rf-radius-line"
            />
            <circle
              cx={centerX}
              cy={centerY}
              r={5.5}
              className="procedure-details-rf-center"
            />
            <text
              x={centerX + 9}
              y={centerY - 9}
              className="procedure-details-rf-label"
            >
              RF center {marker.label}
            </text>
          </g>
        );
      })}

      {polylines.map((branch) => {
        const isFocused = !focusedBranchId || branch.branchId === focusedBranchId;
        return (
          <g key={`${branch.branchId}-fixes`}>
            {branch.points.map((point) => {
              const selected = point.fixId === focusedFixId;
              const showLabel = shouldShowPointLabel(point, focusedFixId, focusedBranchId);
              const x = scaleX(point.xM);
              const y = scaleY(point.yM);
              return (
                <g
                  key={`${branch.branchId}-${point.fixId}`}
                  onMouseEnter={() => onPreviewFix(point.fixId, point.branchId)}
                  onClick={() => onSelectFix(point.fixId, point.branchId)}
                >
                  <rect
                    x={x - (selected ? PLAN_SELECTED_FIX_SYMBOL_SIZE : PLAN_FIX_SYMBOL_SIZE)}
                    y={y - (selected ? PLAN_SELECTED_FIX_SYMBOL_SIZE : PLAN_FIX_SYMBOL_SIZE)}
                    width={(selected ? PLAN_SELECTED_FIX_SYMBOL_SIZE : PLAN_FIX_SYMBOL_SIZE) * 2}
                    height={(selected ? PLAN_SELECTED_FIX_SYMBOL_SIZE : PLAN_FIX_SYMBOL_SIZE) * 2}
                    className={`procedure-details-fix-point ${selected ? "is-selected" : ""} ${
                      isFocused ? "is-focused" : "is-muted"
                    }`}
                  />
                  <circle
                    cx={x}
                    cy={y}
                    r={selected ? 7 : 6.1}
                    className={`procedure-details-fix-center ${selected ? "is-selected" : ""} ${
                      isFocused ? "is-focused" : "is-muted"
                    }`}
                  />
                  {showLabel ? (
                    <text
                      x={x + 10}
                      y={y - 10}
                      className={`procedure-details-fix-label ${selected ? "is-selected" : ""} ${
                        isFocused ? "is-focused" : "is-muted"
                      }`}
                    >
                      {point.ident}
                    </text>
                  ) : null}
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

function ProcedureVerticalProfile({
  polylines,
  missedSectionMarkers,
  missedLegMarkers,
  focusedFixId,
  focusedBranchId,
  distanceUnit,
  onPreviewFix,
  onSelectFix,
  onPreviewBranch,
  onSelectBranch,
}: Omit<SvgChartProps, "runwayMarker" | "rfMarkers">) {
  const allPoints = polylines.flatMap((branch) => branch.points);
  if (allPoints.length === 0) {
    return (
      <div className="procedure-details-empty-chart">
        No altitude-supported samples are available for the vertical profile yet.
      </div>
    );
  }

  const rawAltitudes = allPoints.map((point) => point.altitudeFt ?? 0);
  const altitudeDomain = chartDomain(rawAltitudes, 0.14);
  const stationedBranches = polylines.map((branch) => ({
    branch,
    stationedPoints: stationedProfilePoints(branch.points),
  }));
  const allStations = stationedBranches.flatMap((branch) =>
    branch.stationedPoints.map((entry) => entry.stationM),
  );
  const stationDomain = chartDomain(allStations.length > 0 ? allStations : [0], 0.1);
  const stationTickStep = horizontalTickStepM(
    stationDomain.max - stationDomain.min,
    PROFILE_AXIS_TICK_COUNT,
    distanceUnit,
  );
  const stationTicks = chartTicksByStep(stationDomain, stationTickStep);
  const scaleX = chartScale(
    [stationDomain.min, stationDomain.max],
    SVG_PADDING_X,
    SVG_WIDTH - SVG_PADDING_X,
  );
  const scaleY = chartScale(
    [Math.max(0, altitudeDomain.min), altitudeDomain.max],
    PROFILE_SVG_HEIGHT - SVG_PADDING_Y,
    SVG_PADDING_Y,
  );
  const axisTicks = 5;
  const minAltitudeFt = Math.max(0, altitudeDomain.min);
  const maxAltitudeFt = altitudeDomain.max;
  const zeroStationX =
    stationDomain.min <= 0 && stationDomain.max >= 0 ? scaleX(0) : null;

  return (
    <svg
      className="procedure-details-chart-svg"
      viewBox={`0 0 ${SVG_WIDTH} ${PROFILE_SVG_HEIGHT}`}
      role="img"
      aria-label="Procedure vertical profile"
      onMouseLeave={() => {
        onPreviewFix(null, null);
        onPreviewBranch(null);
      }}
    >
      {Array.from({ length: axisTicks + 1 }, (_, index) => {
        const ratio = index / axisTicks;
        const y = SVG_PADDING_Y + ratio * (PROFILE_SVG_HEIGHT - SVG_PADDING_Y * 2);
        const altitude = Math.round(maxAltitudeFt - (maxAltitudeFt - minAltitudeFt) * ratio);
        return (
          <g key={`y-${index}`}>
            <line
              x1={SVG_PADDING_X}
              y1={y}
              x2={SVG_WIDTH - SVG_PADDING_X}
              y2={y}
              className="procedure-details-grid-line"
            />
            <text x={SVG_PADDING_X - 12} y={y + 4} className="procedure-details-axis-label">
              {altitude.toLocaleString()} ft
            </text>
          </g>
        );
      })}

      {stationTicks.map((tick) => {
        const x = scaleX(tick);
        return (
          <g key={`profile-x-${tick}`}>
            <line
              x1={x}
              y1={SVG_PADDING_Y}
              x2={x}
              y2={PROFILE_SVG_HEIGHT - SVG_PADDING_Y}
              className="procedure-details-grid-line"
            />
            <text
              x={x}
              y={PROFILE_SVG_HEIGHT - SVG_PADDING_Y + 22}
              className="procedure-details-axis-label is-centered"
            >
              {formatSignedDistance(tick, distanceUnit)}
            </text>
          </g>
        );
      })}

      {zeroStationX !== null ? (
        <g>
          <line
            x1={zeroStationX}
            y1={SVG_PADDING_Y}
            x2={zeroStationX}
            y2={PROFILE_SVG_HEIGHT - SVG_PADDING_Y}
            className="procedure-details-axis-line procedure-details-profile-threshold-line"
          />
          <text
            x={zeroStationX + 8}
            y={SVG_PADDING_Y + 18}
            className="procedure-details-axis-title"
          >
            MAPT / RWY
          </text>
        </g>
      ) : null}

      <text
        x={SVG_WIDTH - SVG_PADDING_X}
        y={PROFILE_SVG_HEIGHT - 10}
        className="procedure-details-axis-title"
      >
        Along-track distance from MAPT / runway ({distanceUnitLabel(distanceUnit)})
      </text>
      <text
        x={18}
        y={SVG_PADDING_Y}
        transform={`rotate(-90 18 ${SVG_PADDING_Y})`}
        className="procedure-details-axis-title is-vertical"
      >
        Procedure altitude (ft)
      </text>

      {stationedBranches.map(({ branch, stationedPoints }) => {
        const isFocused = !focusedBranchId || branch.branchId === focusedBranchId;
        const { approachPoints, missedPoints } = splitPlanBranchPoints(branch);
        const stationByPointKey = new Map(
          stationedPoints.map((entry) => [
            `${entry.point.branchId}-${entry.point.fixId}-${entry.point.distanceM}`,
            entry.stationM,
          ]),
        );
        const stationForPoint = (point: ProcedureChartPoint) =>
          stationByPointKey.get(`${point.branchId}-${point.fixId}-${point.distanceM}`) ??
          point.distanceM;
        const profilePath = (points: ProcedureChartPoint[]) =>
          points
            .map((point, index) => {
              const x = scaleX(stationForPoint(point));
              const y = scaleY(point.altitudeFt ?? 0);
              return `${index === 0 ? "M" : "L"} ${x} ${y}`;
            })
            .join(" ");
        const approachPathD = profilePath(approachPoints);
        const missedPathD = profilePath(missedPoints);

        return (
          <g key={branch.branchId}>
            {approachPoints.length >= 2 ? (
              <path
                d={approachPathD}
                className={`procedure-details-branch-line procedure-details-branch-${branch.branchRole} ${
                  isFocused ? "is-focused" : "is-muted"
                }`}
                onMouseEnter={() => onPreviewBranch(branch.branchId)}
                onClick={() => onSelectBranch(branch.branchId)}
              />
            ) : null}
            {missedPoints.length >= 2 ? (
              <path
                d={missedPathD}
                className={`procedure-details-branch-line procedure-details-missed-line ${
                  isFocused ? "is-focused" : "is-muted"
                }`}
                onMouseEnter={() => onPreviewBranch(branch.branchId)}
                onClick={() => onSelectBranch(branch.branchId)}
              />
            ) : null}
            {approachPoints.length < 2 && missedPoints.length < 2 ? (
              <path
                d={profilePath(branch.points)}
                className={`procedure-details-branch-line procedure-details-branch-${branch.branchRole} ${
                  isFocused ? "is-focused" : "is-muted"
                }`}
                onMouseEnter={() => onPreviewBranch(branch.branchId)}
                onClick={() => onSelectBranch(branch.branchId)}
              />
            ) : null}
            {branch.points.map((point) => {
              const selected = point.fixId === focusedFixId;
              const showLabel = shouldShowPointLabel(point, focusedFixId, focusedBranchId);
              const symbolSize = selected
                ? PROFILE_SELECTED_FIX_SYMBOL_SIZE
                : PROFILE_FIX_SYMBOL_SIZE;
              const x = scaleX(stationForPoint(point));
              const y = scaleY(point.altitudeFt ?? 0);
              return (
                <g
                  key={`${branch.branchId}-${point.fixId}`}
                  onMouseEnter={() => onPreviewFix(point.fixId, point.branchId)}
                  onClick={() => onSelectFix(point.fixId, point.branchId)}
                >
                  <rect
                    x={x - symbolSize}
                    y={y - symbolSize}
                    width={symbolSize * 2}
                    height={symbolSize * 2}
                    className={`procedure-details-fix-point ${selected ? "is-selected" : ""} ${
                      isFocused ? "is-focused" : "is-muted"
                    }`}
                  />
                  <circle
                    cx={x}
                    cy={y}
                    r={selected ? 6.2 : 5.4}
                    className={`procedure-details-fix-center ${selected ? "is-selected" : ""} ${
                      isFocused ? "is-focused" : "is-muted"
                    }`}
                  />
                  {showLabel ? (
                    <text
                      x={x}
                      y={y - 12}
                      className={`procedure-details-fix-label is-centered ${
                        selected ? "is-selected" : ""
                      } ${isFocused ? "is-focused" : "is-muted"}`}
                    >
                      {point.ident}
                    </text>
                  ) : null}
                </g>
              );
            })}
            {missedSectionMarkers
              .filter((marker) => marker.branchId === branch.branchId)
              .map((marker) => {
                const isFocusedMarker = !focusedBranchId || marker.branchId === focusedBranchId;
                const x = scaleX(stationForPoint(marker.point));
                const y = scaleY(marker.point.altitudeFt ?? 0);
                return (
                  <g
                    key={`profile-missed-section-${marker.segmentId}`}
                    className={`procedure-details-missed-section-marker ${
                      isFocusedMarker ? "is-focused" : "is-muted"
                    }`}
                    onMouseEnter={() => onPreviewFix(marker.point.fixId, marker.branchId)}
                    onClick={() => onSelectFix(marker.point.fixId, marker.branchId)}
                  >
                    <circle cx={x} cy={y} r={7} />
                    <text x={x + 10} y={y - 10}>{marker.label}</text>
                  </g>
                );
              })}
            {missedLegMarkers
              .filter((marker) => marker.branchId === branch.branchId)
              .map((marker) => {
                const isFocusedMarker = !focusedBranchId || marker.branchId === focusedBranchId;
                const x = scaleX(stationForPoint(marker.point));
                const y = scaleY(marker.point.altitudeFt ?? 0);
                return (
                  <g
                    key={`profile-missed-leg-${marker.legId}`}
                    className={`procedure-details-missed-leg-marker ${
                      isFocusedMarker ? "is-focused" : "is-muted"
                    }`}
                    onMouseEnter={() => onPreviewFix(marker.point.fixId, marker.branchId)}
                    onClick={() => onSelectFix(marker.point.fixId, marker.branchId)}
                  >
                    <rect x={x - 14} y={y + 10} width={54} height={18} rx={4} />
                    <text x={x - 8} y={y + 23}>{marker.label}</text>
                  </g>
                );
              })}
          </g>
        );
      })}
    </svg>
  );
}

export default function ProcedureDetailsPage() {
  const initialParams = useRef(readRouteParams()).current;
  const { airports, activeAirportCode, setActiveAirportCode } = useApp();
  const [selectedAirportCode, setSelectedAirportCode] = useState(
    initialParams.airport ?? activeAirportCode,
  );
  const [selectedRunwayIdent, setSelectedRunwayIdent] = useState<string | null>(
    initialParams.runway,
  );
  const [selectedProcedureUid, setSelectedProcedureUid] = useState<string | null>(
    initialParams.procedureUid,
  );
  const [selectedFixId, setSelectedFixId] = useState<string | null>(null);
  const [selectedBranchId, setSelectedBranchId] = useState<string | null>(null);
  const [previewFixId, setPreviewFixId] = useState<string | null>(null);
  const [previewBranchId, setPreviewBranchId] = useState<string | null>(null);
  const [selectedGlossaryTerm, setSelectedGlossaryTerm] = useState<string | null>(null);
  const [distanceUnit, setDistanceUnit] = useState<DistanceUnit>("nm");
  const [indexManifest, setIndexManifest] = useState<ProcedureDetailsIndexManifest | null>(null);
  const [chartsManifest, setChartsManifest] = useState<ProcedureChartsManifest | null>(null);
  const [procedureDocument, setProcedureDocument] = useState<ProcedureDetailDocument | null>(null);
  const [isLoadingIndex, setLoadingIndex] = useState(false);
  const [indexError, setIndexError] = useState<string | null>(null);
  const [isMissingIndex, setMissingIndex] = useState(false);
  const [procedureError, setProcedureError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedAirportCode && activeAirportCode) {
      setSelectedAirportCode(activeAirportCode);
    }
  }, [activeAirportCode, selectedAirportCode]);

  useEffect(() => {
    if (!selectedAirportCode || selectedAirportCode === activeAirportCode) return;
    setActiveAirportCode(selectedAirportCode);
  }, [activeAirportCode, selectedAirportCode, setActiveAirportCode]);

  useEffect(() => {
    if (!selectedAirportCode) return;
    setLoadingIndex(true);
    setIndexError(null);
    setMissingIndex(false);
    setIndexManifest(null);
    setChartsManifest({ airport: selectedAirportCode, researchUseOnly: true, charts: [] });
    setProcedureDocument(null);
    setSelectedBranchId(null);
    setSelectedFixId(null);
    setPreviewFixId(null);
    setPreviewBranchId(null);
    setSelectedGlossaryTerm(null);

    fetchJson<ProcedureDetailsIndexManifest>(procedureDetailsIndexUrl(selectedAirportCode))
      .then((manifest) => {
        setIndexManifest(manifest);
      })
      .catch((error) => {
        if (isMissingJsonAsset(error)) {
          setMissingIndex(true);
          return;
        }
        setIndexError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => setLoadingIndex(false));

    fetchJson<ProcedureChartsManifest>(procedureChartsIndexUrl(selectedAirportCode))
      .then((manifest) => setChartsManifest(manifest))
      .catch((error) => {
        if (isMissingJsonAsset(error)) {
          setChartsManifest({ airport: selectedAirportCode, researchUseOnly: true, charts: [] });
          return;
        }
        console.error("[ProcedureDetailsPage] Failed to load chart manifest:", error);
      });
  }, [selectedAirportCode]);

  useEffect(() => {
    if (!indexManifest) return;
    const runwayStillExists = indexManifest.runways.some(
      (runway) => runway.runwayIdent === selectedRunwayIdent,
    );
    if (!runwayStillExists) {
      setSelectedRunwayIdent(indexManifest.runways[0]?.runwayIdent ?? null);
    }
  }, [indexManifest, selectedRunwayIdent]);

  const selectedRunway = useMemo<ProcedureDetailsIndexRunwaySummary | null>(() => {
    return (
      indexManifest?.runways.find((runway) => runway.runwayIdent === selectedRunwayIdent) ?? null
    );
  }, [indexManifest, selectedRunwayIdent]);

  useEffect(() => {
    if (!selectedRunway) return;
    const procedureStillExists = selectedRunway.procedures.some(
      (procedure) => procedure.procedureUid === selectedProcedureUid,
    );
    if (!procedureStillExists) {
      setSelectedProcedureUid(selectedRunway.procedures[0]?.procedureUid ?? null);
    }
  }, [selectedProcedureUid, selectedRunway]);

  useEffect(() => {
    if (!selectedAirportCode || !selectedProcedureUid) return;
    setProcedureDocument(null);
    setProcedureError(null);
    setSelectedBranchId(null);
    setSelectedFixId(null);
    setPreviewFixId(null);
    setPreviewBranchId(null);
    fetchJson<ProcedureDetailDocument>(
      procedureDetailsDocumentUrl(selectedAirportCode, selectedProcedureUid),
    )
      .then((document) => setProcedureDocument(document))
      .catch((error) => {
        setProcedureError(error instanceof Error ? error.message : String(error));
      });
  }, [selectedAirportCode, selectedProcedureUid]);

  useEffect(() => {
    if (!procedureDocument) {
      setSelectedFixId(null);
      setSelectedBranchId(null);
      return;
    }
    const defaultFix =
      procedureDocument.runway.landingThresholdFixRef ?? procedureDocument.fixes[0]?.fixId ?? null;
    setSelectedFixId((current) => {
      if (current && procedureDocument.fixes.some((fix) => fix.fixId === current)) return current;
      return defaultFix;
    });
    setSelectedBranchId((current) => {
      if (current && procedureDocument.branches.some((branch) => branch.branchId === current)) {
        return current;
      }
      return selectedBranchDefaultId(procedureDocument);
    });
  }, [procedureDocument]);

  useEffect(() => {
    if (!selectedAirportCode) return;
    const params = new URLSearchParams();
    params.set("airport", selectedAirportCode);
    if (selectedRunwayIdent) params.set("runway", selectedRunwayIdent);
    if (selectedProcedureUid) params.set("procedureUid", selectedProcedureUid);
    window.history.replaceState({}, "", `/procedure-details?${params.toString()}`);
  }, [selectedAirportCode, selectedProcedureUid, selectedRunwayIdent]);

  const polylines = useMemo(
    () => (procedureDocument ? buildProcedureBranchPolylines(procedureDocument) : []),
    [procedureDocument],
  );
  const runwayMarker = useMemo(
    () => (procedureDocument ? buildRunwayMarker(procedureDocument, polylines) : null),
    [procedureDocument, polylines],
  );
  const rfMarkers = useMemo(
    () => buildRfPlanMarkers(procedureDocument, polylines),
    [procedureDocument, polylines],
  );
  const procedureRenderBundle = useMemo(
    () =>
      procedureDocument
        ? buildProcedureRenderBundle(normalizeProcedurePackage(procedureDocument))
        : null,
    [procedureDocument],
  );
  const missedSectionMarkers = useMemo(
    () => buildMissedSectionMarkers(procedureDocument, procedureRenderBundle, polylines),
    [polylines, procedureDocument, procedureRenderBundle],
  );
  const missedLegMarkers = useMemo(
    () => buildMissedLegMarkers(procedureDocument, procedureRenderBundle, polylines),
    [polylines, procedureDocument, procedureRenderBundle],
  );
  const selectedFixBranches = useMemo(
    () => procedureBranchForFix(procedureDocument ?? null, selectedFixId),
    [procedureDocument, selectedFixId],
  );
  const localChart = useMemo(
    () =>
      localChartForProcedure(
        chartsManifest?.charts ?? [],
        procedureDocument?.procedureUid ?? selectedProcedureUid,
        procedureDocument?.runway.ident ?? selectedRunwayIdent,
      ),
    [chartsManifest?.charts, procedureDocument, selectedProcedureUid, selectedRunwayIdent],
  );

  const fallbackAirportCode = selectedAirportCode.replace(/^[KCAP]/, "") || selectedAirportCode;
  const researchAirportCode = procedureDocument?.airport.faa ?? fallbackAirportCode;
  const focusedFixId = previewFixId ?? selectedFixId;
  const focusedFix = useMemo(
    () => findFix(procedureDocument ?? null, focusedFixId),
    [procedureDocument, focusedFixId],
  );
  const focusedFixBranches = useMemo(
    () => procedureBranchForFix(procedureDocument ?? null, focusedFixId),
    [procedureDocument, focusedFixId],
  );
  const focusedBranchId =
    previewBranchId ??
    selectedBranchId ??
    focusedFixBranches[0]?.branchId ??
    selectedBranchDefaultId(procedureDocument);
  const focusedBranch = useMemo(
    () =>
      procedureDocument?.branches.find((branch) => branch.branchId === focusedBranchId) ?? null,
    [focusedBranchId, procedureDocument],
  );
  const focusedPolyline = useMemo(
    () => polylines.find((branch) => branch.branchId === focusedBranchId) ?? null,
    [focusedBranchId, polylines],
  );
  const focusedBranchWarnings = useMemo(
    () =>
      Array.from(
        new Set([
          ...(focusedBranch?.warnings ?? []),
          ...(focusedPolyline?.warnings ?? []),
        ]),
      ),
    [focusedBranch?.warnings, focusedPolyline?.warnings],
  );
  const focusedLegs = focusedBranch?.legs ?? [];
  const focusedScopedBranchId =
    procedureDocument && focusedBranch
      ? `${procedureDocument.procedureUid}:branch:${(
          focusedBranch.branchKey ?? focusedBranch.branchIdent
        ).toUpperCase()}`
      : null;
  const focusedRenderDiagnostics = useMemo(() => {
    if (!procedureRenderBundle) return [];
    if (!focusedScopedBranchId) return procedureRenderBundle.diagnostics;
    const focusedLegIds = new Set(focusedLegs.map((leg) => leg.legId));
    return procedureRenderBundle.diagnostics.filter((diagnostic) => {
      if (diagnostic.segmentId?.startsWith(focusedScopedBranchId)) return true;
      if (diagnostic.legId && focusedLegIds.has(diagnostic.legId)) return true;
      return !diagnostic.segmentId && !diagnostic.legId;
    });
  }, [focusedLegs, focusedScopedBranchId, procedureRenderBundle]);
  const focusedFixTerminalLeg = useMemo(
    () =>
      focusedLegs.find(
        (leg) =>
          leg.path.endFixRef === focusedFixId || leg.termination.fixRef === focusedFixId,
      ) ?? null,
    [focusedFixId, focusedLegs],
  );
  const focusedLeg = useMemo(
    () =>
      focusedFixTerminalLeg ??
      focusedLegs.find(
        (leg) =>
          leg.path.endFixRef === focusedFixId ||
          leg.path.startFixRef === focusedFixId ||
          leg.termination.fixRef === focusedFixId,
      ) ?? null,
    [focusedFixId, focusedFixTerminalLeg, focusedLegs],
  );
  const focusedFixProcedureAltitudeFt =
    focusedFixTerminalLeg?.constraints.altitude?.valueFt ?? null;
  const focusedFixProfileAltitudeFt =
    focusedFixTerminalLeg?.constraints.geometryAltitudeFt ?? focusedFixProcedureAltitudeFt;
  const focusedFixHasProfileRepair =
    typeof focusedFixProfileAltitudeFt === "number" &&
    focusedFixProfileAltitudeFt !== focusedFixProcedureAltitudeFt;
  const glossaryTerms = useMemo(
    () => contextualTerms(procedureDocument, focusedFix, focusedBranch),
    [procedureDocument, focusedFix, focusedBranch],
  );
  const glossaryTermGroups = useMemo(() => groupedTerms(glossaryTerms), [glossaryTerms]);

  useEffect(() => {
    if (glossaryTerms.length === 0) {
      setSelectedGlossaryTerm(null);
      return;
    }
    setSelectedGlossaryTerm((current) => {
      if (current && glossaryTerms.includes(current)) return current;
      return focusedFix?.roleHints[0] ?? glossaryTerms[0];
    });
  }, [focusedFix, glossaryTerms]);

  const activeGlossaryTerm = selectedGlossaryTerm ?? glossaryTerms[0] ?? null;
  const isPreviewMode = previewFixId !== null || previewBranchId !== null;

  function handleSelectFix(fixId: string, branchId: string) {
    setSelectedFixId(fixId);
    setSelectedBranchId(branchId);
    setPreviewFixId(null);
    setPreviewBranchId(null);
  }

  function handlePreviewFix(fixId: string | null, branchId: string | null) {
    setPreviewFixId(fixId);
    setPreviewBranchId(branchId);
  }

  function clearPreview() {
    setPreviewFixId(null);
    setPreviewBranchId(null);
  }

  function handleSelectBranch(branchId: string) {
    setSelectedBranchId(branchId);
    setPreviewBranchId(null);

    const currentFixIsOnBranch = selectedFixBranches.some((branch) => branch.branchId === branchId);
    if (!currentFixIsOnBranch) {
      const firstPoint = polylines.find((branch) => branch.branchId === branchId)?.points[0];
      if (firstPoint) {
        setSelectedFixId(firstPoint.fixId);
      }
    }
  }

  return (
    <div className="procedure-details-page">
      <header className="procedure-details-header">
        <div className="procedure-details-header-title">
          <p className="procedure-details-eyebrow">AeroViz-4D Research Companion</p>
          <h1>Procedure Details</h1>
        </div>

        <label className="procedure-details-airport-select">
          <span>Airport</span>
          <select
            value={selectedAirportCode}
            onChange={(event) => {
              const nextAirport = event.target.value;
              setSelectedAirportCode(nextAirport);
              setSelectedRunwayIdent(null);
              setSelectedProcedureUid(null);
              setSelectedFixId(null);
              setSelectedBranchId(null);
              setSelectedGlossaryTerm(null);
              clearPreview();
            }}
          >
            {airports.map((airport) => (
              <option key={airport.code} value={airport.code}>
                {airport.code} - {airport.name}
              </option>
            ))}
          </select>
        </label>

        <div className="procedure-details-header-actions">
          <button
            type="button"
            className="procedure-details-nav-button"
            onClick={() => navigateWithinApp("/")}
          >
            Back To 3D Scene
          </button>
        </div>
      </header>

      {isLoadingIndex ? <div className="procedure-details-loading">Loading procedure details…</div> : null}
      {indexError ? <div className="procedure-details-error">{indexError}</div> : null}
      {isMissingIndex ? (
        <div className="procedure-details-empty-state">
          <h2>No procedure-details dataset yet for {selectedAirportCode}</h2>
          <p>
            This airport already exists in the airport catalog, but the richer per-procedure export has
            not been generated yet. The route, profile, and details views all use this canonical
            data layer so the user experience stays consistent.
          </p>
        </div>
      ) : null}

      {!isLoadingIndex && !isMissingIndex && indexManifest ? (
        <div className="procedure-details-workspace">
          <aside className="procedure-details-sidebar">
            <section className="procedure-details-card procedure-details-selector-card">
              <p className="procedure-details-overview-label">Airport Context</p>
              <h2>{indexManifest.airportName}</h2>
              <p className="procedure-details-sidebar-meta">
                {indexManifest.airport} · cycle {indexManifest.sourceCycle ?? "Unknown"}
              </p>
            </section>

            <section className="procedure-details-card procedure-details-selector-card">
              <div className="procedure-details-card-head">
                <h3>Runways</h3>
                <span className="procedure-details-meta-pill">
                  {indexManifest.runways.length}
                </span>
              </div>
              <div className="procedure-details-runway-list">
                {indexManifest.runways.map((runway) => (
                  <button
                    type="button"
                    key={runway.runwayIdent}
                    className={`procedure-details-runway-button ${
                      runway.runwayIdent === selectedRunwayIdent ? "is-active" : ""
                    }`}
                    onClick={() => {
                      setSelectedRunwayIdent(runway.runwayIdent);
                      setSelectedProcedureUid(runway.procedures[0]?.procedureUid ?? null);
                      setSelectedBranchId(null);
                      setSelectedFixId(null);
                      clearPreview();
                    }}
                  >
                    <span>{formatRunway(runway.runwayIdent)}</span>
                    <small>
                      {runway.procedures.length}{" "}
                      {runway.procedures.length === 1 ? "procedure" : "procedures"}
                    </small>
                  </button>
                ))}
              </div>
            </section>

            <section className="procedure-details-card procedure-details-selector-card">
              <div className="procedure-details-card-head">
                <h3>Procedures</h3>
                {selectedRunway ? (
                  <span className="procedure-details-meta-pill">
                    {formatRunway(selectedRunway.runwayIdent)}
                  </span>
                ) : null}
              </div>
              <div className="procedure-details-procedure-list">
                {selectedRunway?.procedures.map((procedure) => (
                  <button
                    type="button"
                    key={procedure.procedureUid}
                    className={`procedure-details-procedure-button ${
                      procedure.procedureUid === selectedProcedureUid ? "is-active" : ""
                    }`}
                    onClick={() => {
                      setSelectedProcedureUid(procedure.procedureUid);
                      setSelectedBranchId(null);
                      setSelectedFixId(null);
                      clearPreview();
                    }}
                  >
                    <span>{procedure.chartName}</span>
                    <small>
                      {procedure.approachModes.join(" / ") || "Mode data unavailable"}
                    </small>
                  </button>
                )) ?? <p>Select a runway to see its procedures.</p>}
              </div>
            </section>

            {procedureDocument ? (
              <section className="procedure-details-card procedure-details-reference-card">
                <h3>Reference Links</h3>
                <div className="procedure-details-reference-list">
                  <a
                    href={buildFaaChartUrl(researchAirportCode)}
                    target="_blank"
                    rel="noreferrer"
                    className="procedure-details-link-button"
                  >
                    Open FAA Procedure Search
                  </a>
                  {localChart ? (
                    <a
                      href={localChart.url}
                      target="_blank"
                      rel="noreferrer"
                      className="procedure-details-link-button is-secondary"
                    >
                      Open Local Chart PDF
                    </a>
                  ) : (
                    <p className="procedure-details-muted">
                      No local chart PDF published yet.
                    </p>
                  )}
                </div>
              </section>
            ) : null}

            {procedureDocument &&
            (procedureDocument.validation.knownSimplifications.length > 0 ||
              procedureDocument.provenance.warnings.length > 0 ||
              focusedBranchWarnings.length > 0 ||
              focusedRenderDiagnostics.length > 0) ? (
              <section className="procedure-details-card procedure-details-reference-card">
                <p className="procedure-details-overview-label">Data Notes</p>
                <ul className="procedure-details-note-list">
                  {focusedBranchWarnings.map((warning) => (
                    <li key={`branch-${warning}`}>{warning}</li>
                  ))}
                  {procedureDocument.provenance.warnings.map((warning) => (
                    <li key={`provenance-${warning}`}>{warning}</li>
                  ))}
                  {procedureDocument.validation.knownSimplifications.map((warning) => (
                    <li key={`simplification-${warning}`}>{warning}</li>
                  ))}
                  {focusedRenderDiagnostics.map((diagnostic, index) => (
                    <li key={`diagnostic-${diagnostic.code}-${diagnostic.segmentId ?? ""}-${diagnostic.legId ?? ""}-${index}`}>
                      <strong>
                        {diagnostic.severity} {diagnostic.code}
                      </strong>
                      : {diagnostic.message}
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}
          </aside>

          <main className="procedure-details-main">
            {procedureError ? <div className="procedure-details-error">{procedureError}</div> : null}
            {!procedureDocument ? (
              <div className="procedure-details-loading">Loading selected procedure…</div>
            ) : (
              <>
                <section className="procedure-details-procedure-bar">
                  <div className="procedure-details-procedure-bar-title">
                    <p className="procedure-details-overview-label">Selected Procedure</p>
                    <h2>{procedureDocument.procedure.chartName}</h2>
                  </div>
                  <div className="procedure-details-procedure-bar-facts">
                    <span>
                      <em>Runway</em>
                      {formatRunway(procedureDocument.runway.ident)}
                    </span>
                    <span>
                      <em>Modes</em>
                      {procedureDocument.procedure.approachModes.join(" / ") || "Unavailable"}
                    </span>
                    <span>
                      <em>Family</em>
                      {procedureDocument.procedure.procedureFamily}
                    </span>
                    <span>
                      <em>Threshold</em>
                      {formatAltitudeFt(procedureDocument.runway.threshold?.elevationFt)}
                    </span>
                    <span>
                      <em>Base branch</em>
                      {procedureDocument.procedure.baseBranchIdent}
                    </span>
                    <span>
                      <em>Focus</em>
                      {focusedFix?.ident ?? "—"}
                    </span>
                  </div>
                  <div className="procedure-details-unit-switch" role="group" aria-label="Distance unit">
                    {(["nm", "m"] as const).map((unit) => (
                      <button
                        key={unit}
                        type="button"
                        className={distanceUnit === unit ? "is-active" : ""}
                        onClick={() => setDistanceUnit(unit)}
                      >
                        {unit === "nm" ? "NM" : "m"}
                      </button>
                    ))}
                  </div>
                </section>

                <section className="procedure-details-chart-stack">
                  <div className="procedure-details-card procedure-details-chart-card">
                    <div className="procedure-details-chart-frame-head">
                      <div>
                        <p className="procedure-details-overview-label">Plan View</p>
                        <h3>Horizontal layout of the published approach</h3>
                        <p className="procedure-details-section-intro">
                          Transition branches stay visible, but the focused branch and fix are
                          emphasized so the geometry is easier to read. Grid labels show
                          east/north offsets in {distanceUnit === "nm" ? "nautical miles" : "meters"} from the local origin.
                        </p>
                      </div>

                      <div className="procedure-details-chart-legend">
                        <span className="procedure-details-legend-item">
                          <i className="is-final" />
                          Final
                        </span>
                        <span className="procedure-details-legend-item">
                          <i className="is-transition" />
                          Transition
                        </span>
                        <span className="procedure-details-legend-item">
                          <i className="is-missed" />
                          Missed
                        </span>
                        <span className="procedure-details-legend-item">
                          <i className="is-runway" />
                          Runway
                        </span>
                      </div>
                    </div>
                    <ProcedurePlanView
                      polylines={polylines}
                      runwayMarker={runwayMarker}
                      rfMarkers={rfMarkers}
                      missedSectionMarkers={missedSectionMarkers}
                      missedLegMarkers={missedLegMarkers}
                      focusedFixId={focusedFixId}
                      focusedBranchId={focusedBranchId}
                      distanceUnit={distanceUnit}
                      onPreviewFix={handlePreviewFix}
                      onSelectFix={handleSelectFix}
                      onPreviewBranch={setPreviewBranchId}
                      onSelectBranch={handleSelectBranch}
                    />
                  </div>

                  <div className="procedure-details-card procedure-details-chart-card">
                    <div className="procedure-details-chart-frame-head">
                      <div>
                        <p className="procedure-details-overview-label">Vertical Profile</p>
                        <h3>Vertical profile aligned to the runway/MAPT</h3>
                        <p className="procedure-details-section-intro">
                          The horizontal axis is rebased at the runway/MAPT: approach fixes are
                          shown inbound on the negative side in {distanceUnit === "nm" ? "nautical miles" : "meters"}, and
                          missed-approach fixes continue outbound on the positive side. Altitudes
                          are shown in feet and repaired for display when source altitude
                          constraints are missing.
                        </p>
                      </div>
                    </div>
                    <ProcedureVerticalProfile
                      polylines={polylines}
                      missedSectionMarkers={missedSectionMarkers}
                      missedLegMarkers={missedLegMarkers}
                      focusedFixId={focusedFixId}
                      focusedBranchId={focusedBranchId}
                      distanceUnit={distanceUnit}
                      onPreviewFix={handlePreviewFix}
                      onSelectFix={handleSelectFix}
                      onPreviewBranch={setPreviewBranchId}
                      onSelectBranch={handleSelectBranch}
                    />
                  </div>
                </section>

                <section className="procedure-details-card procedure-details-sequence-card">
                    <div className="procedure-details-card-head">
                      <div>
                        <p className="procedure-details-overview-label">Focused Sequence</p>
                        <h3>
                          {focusedBranch
                            ? `${branchDisplayLabel(focusedBranch)} · ${branchTypeDisplayLabel(
                                focusedBranch,
                              )}`
                            : "Choose a branch"}
                        </h3>
                      </div>
                      {focusedLeg ? (
                        <span className="procedure-details-meta-pill">
                          Leg {focusedLeg.sequence}
                        </span>
                      ) : null}
                    </div>

                    <div className="procedure-details-branch-pills">
                      {procedureDocument.branches.map((branch) => (
                        <button
                          type="button"
                          key={branch.branchId}
                          className={`procedure-details-branch-pill ${
                            branch.branchId === focusedBranchId ? "is-active" : ""
                          }`}
                          onMouseEnter={() => setPreviewBranchId(branch.branchId)}
                          onMouseLeave={clearPreview}
                          onClick={() => handleSelectBranch(branch.branchId)}
                        >
                          <strong>{branchDisplayLabel(branch)}</strong>
                          <span>{branchTypeDisplayLabel(branch)}</span>
                        </button>
                      ))}
                    </div>

                    {focusedBranch ? (
                      <div className="procedure-details-leg-stack">
                        {focusedLegs.map((leg) => {
                          const endFix = findFix(procedureDocument, leg.path.endFixRef);
                          const isActive = leg.path.endFixRef === focusedFixId;
                          const roleLabel = legRoleLabel(leg);
                          const rfMetadataLabels = legRfMetadataLabels(leg);
                          return (
                            <button
                              type="button"
                              key={leg.legId}
                              className={`procedure-details-leg-card ${
                                isActive ? "is-active" : ""
                              }`}
                              onMouseEnter={() =>
                                handlePreviewFix(leg.path.endFixRef, focusedBranch.branchId)
                              }
                              onMouseLeave={clearPreview}
                              onClick={() =>
                                handleSelectFix(leg.path.endFixRef, focusedBranch.branchId)
                              }
                            >
                              <span className="procedure-details-leg-seq">{leg.sequence}</span>
                              <div className="procedure-details-leg-main">
                                <div className="procedure-details-leg-title-row">
                                  <strong>{legEndpointLabel(leg, endFix)}</strong>
                                  <span className="procedure-details-leg-role">
                                    {roleLabel}
                                  </span>
                                </div>
                                <p>{focusedLegDescription(leg)}</p>
                              </div>
                              <div className="procedure-details-leg-meta">
                                <span>{leg.path.pathTerminator}</span>
                                <span>
                                  {formatAltitudeFt(
                                    leg.constraints.geometryAltitudeFt ??
                                      leg.constraints.altitude?.valueFt,
                                  )}
                                </span>
                                {rfMetadataLabels.map((label) => (
                                  <span key={label}>{label}</span>
                                ))}
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    ) : (
                      <p>Select a branch to see its coded flow.</p>
                    )}
                </section>
              </>
            )}
          </main>

          {procedureDocument ? (
            <aside className="procedure-details-inspector">
              <section className="procedure-details-card procedure-details-focus-card">
                <div className="procedure-details-card-head">
                  <div>
                    <p className="procedure-details-overview-label">Focused Fix</p>
                    <h3>{focusedFix?.ident ?? "Choose a fix"}</h3>
                  </div>
                  {focusedFix ? (
                    <span className="procedure-details-meta-pill">
                      {isPreviewMode ? "Preview" : "Locked"}
                    </span>
                  ) : null}
                </div>

                {focusedFix ? (
                  <div className="procedure-details-fix-inspector">
                    <p>{roleMeaning(focusedFix.roleHints[0] ?? focusedFix.kind)}</p>
                    <div className="procedure-details-term-chip-row">
                      {focusedFix.roleHints.map((role) => (
                        <button
                          key={role}
                          type="button"
                          className={`procedure-details-term-chip ${
                            activeGlossaryTerm === role ? "is-active" : ""
                          }`}
                          onClick={() => setSelectedGlossaryTerm(role)}
                        >
                          {formatTermBrief(role)}
                        </button>
                      ))}
                    </div>

                    <dl className="procedure-details-definition-list">
                      <div>
                        <dt>Kind</dt>
                        <dd>{focusedFix.kind}</dd>
                      </div>
                      <div>
                        <dt>Latitude</dt>
                        <dd>{formatCoordinate(focusedFix.position?.lat)}</dd>
                      </div>
                      <div>
                        <dt>Longitude</dt>
                        <dd>{formatCoordinate(focusedFix.position?.lon)}</dd>
                      </div>
                      <div>
                        <dt>Surveyed elevation</dt>
                        <dd>{formatAltitudeFt(focusedFix.elevationFt)}</dd>
                      </div>
                      <div>
                        <dt>Procedure altitude</dt>
                        <dd>{formatAltitudeFt(focusedFixProcedureAltitudeFt)}</dd>
                      </div>
                      {focusedFixHasProfileRepair ? (
                        <div>
                          <dt>Profile altitude</dt>
                          <dd>{formatAltitudeFt(focusedFixProfileAltitudeFt)}</dd>
                        </div>
                      ) : null}
                      <div>
                        <dt>Role hints</dt>
                        <dd>{focusedFix.roleHints.join(", ") || "Not available"}</dd>
                      </div>
                      <div>
                        <dt>Used by branches</dt>
                        <dd>
                          {focusedFixBranches.map((branch) => branchDisplayLabel(branch)).join(", ") ||
                            "No branch mapping"}
                        </dd>
                      </div>
                    </dl>
                    {focusedFix.elevationFt === null &&
                    typeof focusedFixProcedureAltitudeFt === "number" ? (
                      <p className="procedure-details-fix-data-note">
                        This fix has no surveyed elevation in the fix coordinate record. The
                        vertical profile uses the published procedure altitude constraint instead.
                      </p>
                    ) : null}
                  </div>
                ) : (
                  <p>Hover or click a fix to inspect it here.</p>
                )}
              </section>

              <section className="procedure-details-card procedure-details-terms-card">
                <h3>Key Terms</h3>
                <p className="procedure-details-section-intro">
                  Terms are grouped by chart role, procedure capability, and coded path behavior.
                </p>
                <div className="procedure-details-term-groups">
                  {glossaryTermGroups.map((group) => (
                    <div key={group.id} className="procedure-details-term-group">
                      <h4>{group.title}</h4>
                      <div className="procedure-details-term-chip-row">
                        {group.terms.map((term) => {
                          const isActiveTerm = activeGlossaryTerm === term;
                          const hasDefinition = isKnownGlossaryTerm(term);
                          const details = contextualTermDetails(term, procedureDocument);
                          return (
                            <article
                              key={`${group.id}-${term}`}
                              className={`procedure-details-term-card ${
                                isActiveTerm ? "is-active" : ""
                              } ${hasDefinition ? "" : "is-identifier"}`}
                            >
                              <button
                                type="button"
                                className="procedure-details-term-chip"
                                onClick={() => setSelectedGlossaryTerm(term)}
                              >
                                <span>{formatTermBrief(term)}</span>
                                <small>{formatContextualTermMeaning(term, procedureDocument)}</small>
                              </button>
                              {isActiveTerm ? (
                                <div className="procedure-details-term-inline-definition">
                                  <p>{details.definition}</p>
                                  {details.references.length > 0 ? (
                                    <div className="procedure-details-term-references">
                                      <span>References</span>
                                      {details.references.map((reference) => (
                                        <a
                                          key={`${term}-${reference.url}`}
                                          href={reference.url}
                                          target="_blank"
                                          rel="noreferrer"
                                        >
                                          {reference.label}
                                        </a>
                                      ))}
                                    </div>
                                  ) : null}
                                </div>
                              ) : null}
                            </article>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            </aside>
          ) : null}
        </div>
      ) : null}

      <footer className="procedure-details-footer" role="contentinfo">
        Research use only. Always use the official FAA or local published chart for operational decisions.
      </footer>
    </div>
  );
}
