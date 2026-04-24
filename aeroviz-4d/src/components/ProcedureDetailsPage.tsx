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
import {
  procedureChartsIndexUrl,
  procedureDetailsDocumentUrl,
  procedureDetailsIndexUrl,
} from "../data/procedureDetails";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import { navigateWithinApp } from "../utils/navigation";
import {
  buildProcedureBranchPolylines,
  buildRunwayMarker,
  findFix,
  nmFromMeters,
  procedureBranchForFix,
  type ProcedureBranchPolyline,
  type ProcedureChartPoint,
  type ProcedureRunwayMarker,
} from "../utils/procedureDetailsGeometry";

const SVG_WIDTH = 1120;
const PLAN_SVG_HEIGHT = 520;
const PROFILE_SVG_HEIGHT = 420;
const SVG_PADDING_X = 64;
const SVG_PADDING_Y = 44;
const IMPORTANT_FIX_ROLES = new Set(["IAF", "IF", "FAF", "MAPT", "MAHF"]);

const TERM_EXPLANATIONS: Record<string, string> = {
  IAF: "Initial Approach Fix: where an aircraft can join the published approach from the wider route network.",
  IF: "Intermediate Fix: a point that lines the aircraft up and settles it before final approach.",
  FAF: "Final Approach Fix: the point where the final descent toward the runway is established.",
  MAPT: "Missed Approach Point: if the runway is not safely in view here, the published missed approach begins.",
  MAHF: "Missed Approach Holding Fix: the protected holding point used after a missed approach.",
  LPV: "Localizer Performance with Vertical guidance: a GPS-based approach mode with precise lateral and vertical guidance.",
  "LNAV/VNAV":
    "Lateral Navigation / Vertical Navigation: GPS-guided lateral path with approved vertical guidance.",
  LNAV: "Lateral Navigation only: the aircraft follows the lateral path and pilots manage the descent profile.",
  IF_TERMINATOR: "IF path terminator: start the published segment at that named fix.",
  TF_TERMINATOR: "TF path terminator: fly a straight published track to the next fix.",
};

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
  return (
    TERM_EXPLANATIONS[role.toUpperCase()] ??
    "Published role in the approach sequence. It helps explain where this fix sits in the procedure."
  );
}

function terminatorMeaning(pathTerminator: string): string {
  return (
    TERM_EXPLANATIONS[`${pathTerminator.toUpperCase()}_TERMINATOR`] ??
    "Published path instruction from the procedure coding."
  );
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

function displayTerm(term: string): string {
  return term.replace("_TERMINATOR", "");
}

function termMeaning(term: string): string {
  return (
    TERM_EXPLANATIONS[term] ??
    TERM_EXPLANATIONS[term.toUpperCase()] ??
    "Procedure coding term used by the chart and the intermediate dataset."
  );
}

function branchRoleLabel(branchRole: string): string {
  return branchRole === "final" ? "Final segment" : "Transition segment";
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

function summaryTerms(document: ProcedureDetailDocument | null): string[] {
  if (!document) return ["IAF", "IF", "FAF", "MAPt", "LPV", "LNAV/VNAV", "LNAV"];
  const terms = new Set(["IAF", "IF", "FAF", "MAPt", "LPV", "LNAV/VNAV", "LNAV"]);
  document.procedure.approachModes.forEach((mode) => terms.add(mode));
  document.branches.forEach((branch) =>
    branch.legs.forEach((leg) => {
      if (leg.roleAtEnd) terms.add(leg.roleAtEnd);
      if (leg.path.pathTerminator) terms.add(`${leg.path.pathTerminator}_TERMINATOR`);
    }),
  );
  return [...terms];
}

function contextualTerms(
  document: ProcedureDetailDocument | null,
  focusedFix: ProcedureDetailFix | null,
  focusedBranch: ProcedureDetailBranch | null,
): string[] {
  const terms = new Set(summaryTerms(document));
  focusedFix?.roleHints.forEach((role) => terms.add(role));
  focusedBranch?.legs.forEach((leg) => {
    if (leg.roleAtEnd) terms.add(leg.roleAtEnd);
    if (leg.path.pathTerminator) terms.add(`${leg.path.pathTerminator}_TERMINATOR`);
  });
  return [...terms];
}

function focusedLegDescription(leg: ProcedureDetailLeg): string {
  const pathTermMeaning = terminatorMeaning(leg.path.pathTerminator);
  if (leg.roleAtEnd) {
    return `${pathTermMeaning} This leg ends at the ${leg.roleAtEnd} point.`;
  }
  return pathTermMeaning;
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
  focusedFixId: string | null;
  focusedBranchId: string | null;
  onPreviewFix: (fixId: string | null, branchId: string | null) => void;
  onSelectFix: (fixId: string, branchId: string) => void;
  onPreviewBranch: (branchId: string | null) => void;
  onSelectBranch: (branchId: string) => void;
}

function ProcedurePlanView({
  polylines,
  runwayMarker,
  focusedFixId,
  focusedBranchId,
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

  const xDomain = chartDomain(allPoints.map((point) => point.xM), 0.12);
  const yDomain = chartDomain(allPoints.map((point) => point.yM), 0.12);
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
      {runwayMarker ? (
        <line
          x1={scaleX(runwayMarker.x1)}
          y1={scaleY(runwayMarker.y1)}
          x2={scaleX(runwayMarker.x2)}
          y2={scaleY(runwayMarker.y2)}
          className="procedure-details-runway-bar"
        />
      ) : null}

      {polylines.map((branch) => {
        const isFocused = !focusedBranchId || branch.branchId === focusedBranchId;
        const pathD = branch.points
          .map(
            (point, index) =>
              `${index === 0 ? "M" : "L"} ${scaleX(point.xM)} ${scaleY(point.yM)}`,
          )
          .join(" ");

        return (
          <g key={branch.branchId}>
            <path
              d={pathD}
              className={`procedure-details-branch-line procedure-details-branch-${branch.branchRole} ${
                isFocused ? "is-focused" : "is-muted"
              }`}
              onMouseEnter={() => onPreviewBranch(branch.branchId)}
              onClick={() => onSelectBranch(branch.branchId)}
            />
            {branch.points.map((point) => {
              const selected = point.fixId === focusedFixId;
              const showLabel = shouldShowPointLabel(point, focusedFixId, focusedBranchId);
              return (
                <g
                  key={`${branch.branchId}-${point.fixId}`}
                  onMouseEnter={() => onPreviewFix(point.fixId, point.branchId)}
                  onClick={() => onSelectFix(point.fixId, point.branchId)}
                >
                  <circle
                    cx={scaleX(point.xM)}
                    cy={scaleY(point.yM)}
                    r={selected ? 6 : 4}
                    className={`procedure-details-fix-point ${selected ? "is-selected" : ""} ${
                      isFocused ? "is-focused" : "is-muted"
                    }`}
                  />
                  {showLabel ? (
                    <text
                      x={scaleX(point.xM) + 8}
                      y={scaleY(point.yM) - 8}
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
  focusedFixId,
  focusedBranchId,
  onPreviewFix,
  onSelectFix,
  onPreviewBranch,
  onSelectBranch,
}: Omit<SvgChartProps, "runwayMarker">) {
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
  const maxDistanceM = Math.max(...allPoints.map((point) => point.distanceM), 1);
  const scaleX = chartScale(
    [0, maxDistanceM],
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

      {Array.from({ length: axisTicks + 1 }, (_, index) => {
        const ratio = index / axisTicks;
        const x = SVG_PADDING_X + ratio * (SVG_WIDTH - SVG_PADDING_X * 2);
        const distanceNm = nmFromMeters(maxDistanceM * ratio);
        return (
          <g key={`x-${index}`}>
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
              {distanceNm.toFixed(1)} NM
            </text>
          </g>
        );
      })}

      {polylines.map((branch) => {
        const isFocused = !focusedBranchId || branch.branchId === focusedBranchId;
        const pathD = branch.points
          .map((point, index) => {
            const x = scaleX(point.distanceM);
            const y = scaleY(point.altitudeFt ?? 0);
            return `${index === 0 ? "M" : "L"} ${x} ${y}`;
          })
          .join(" ");

        return (
          <g key={branch.branchId}>
            <path
              d={pathD}
              className={`procedure-details-branch-line procedure-details-branch-${branch.branchRole} ${
                isFocused ? "is-focused" : "is-muted"
              }`}
              onMouseEnter={() => onPreviewBranch(branch.branchId)}
              onClick={() => onSelectBranch(branch.branchId)}
            />
            {branch.points.map((point) => {
              const selected = point.fixId === focusedFixId;
              const showLabel = shouldShowPointLabel(point, focusedFixId, focusedBranchId);
              return (
                <g
                  key={`${branch.branchId}-${point.fixId}`}
                  onMouseEnter={() => onPreviewFix(point.fixId, point.branchId)}
                  onClick={() => onSelectFix(point.fixId, point.branchId)}
                >
                  <circle
                    cx={scaleX(point.distanceM)}
                    cy={scaleY(point.altitudeFt ?? 0)}
                    r={selected ? 6 : 4}
                    className={`procedure-details-fix-point ${selected ? "is-selected" : ""} ${
                      isFocused ? "is-focused" : "is-muted"
                    }`}
                  />
                  {showLabel ? (
                    <text
                      x={scaleX(point.distanceM)}
                      y={scaleY(point.altitudeFt ?? 0) - 10}
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
  const focusedLegs = focusedBranch?.legs ?? [];
  const focusedLeg = useMemo(
    () =>
      focusedLegs.find(
        (leg) =>
          leg.path.endFixRef === focusedFixId ||
          leg.path.startFixRef === focusedFixId ||
          leg.termination.fixRef === focusedFixId,
      ) ?? null,
    [focusedFixId, focusedLegs],
  );
  const glossaryTerms = useMemo(
    () => contextualTerms(procedureDocument, focusedFix, focusedBranch),
    [procedureDocument, focusedFix, focusedBranch],
  );

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
            not been generated yet. The page is intentionally not falling back to the flatter
            <code>procedures.geojson</code> file so the user experience stays consistent.
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
              (focusedBranch?.warnings.length ?? 0) > 0) ? (
              <section className="procedure-details-card procedure-details-reference-card">
                <p className="procedure-details-overview-label">Data Notes</p>
                <ul className="procedure-details-note-list">
                  {focusedBranch?.warnings.map((warning) => (
                    <li key={`branch-${warning}`}>{warning}</li>
                  ))}
                  {procedureDocument.provenance.warnings.map((warning) => (
                    <li key={`provenance-${warning}`}>{warning}</li>
                  ))}
                  {procedureDocument.validation.knownSimplifications.map((warning) => (
                    <li key={`simplification-${warning}`}>{warning}</li>
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
                </section>

                <section className="procedure-details-chart-stack">
                  <div className="procedure-details-card procedure-details-chart-card">
                    <div className="procedure-details-chart-frame-head">
                      <div>
                        <p className="procedure-details-overview-label">Plan View</p>
                        <h3>Horizontal layout of the published approach</h3>
                        <p className="procedure-details-section-intro">
                          Transition branches stay visible, but the focused branch and fix are
                          emphasized so the geometry is easier to read.
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
                          <i className="is-runway" />
                          Runway
                        </span>
                      </div>
                    </div>
                    <ProcedurePlanView
                      polylines={polylines}
                      runwayMarker={runwayMarker}
                      focusedFixId={focusedFixId}
                      focusedBranchId={focusedBranchId}
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
                        <h3>Altitude versus along-track distance</h3>
                        <p className="procedure-details-section-intro">
                          Heights are repaired for display when the intermediate data leaves a fix
                          altitude unknown, so the descent picture stays readable.
                        </p>
                      </div>
                    </div>
                    <ProcedureVerticalProfile
                      polylines={polylines}
                      focusedFixId={focusedFixId}
                      focusedBranchId={focusedBranchId}
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
                            ? `${focusedBranch.branchIdent} · ${branchRoleLabel(
                                focusedBranch.branchRole,
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

                    {focusedBranch ? (
                      <div className="procedure-details-leg-stack">
                        {focusedLegs.map((leg) => {
                          const endFix = findFix(procedureDocument, leg.path.endFixRef);
                          const isActive = leg.path.endFixRef === focusedFixId;
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
                                  <strong>{endFix?.ident ?? formatFixRef(leg.path.endFixRef)}</strong>
                                  <span className="procedure-details-leg-role">
                                    {displayTerm(leg.roleAtEnd)}
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
                          {displayTerm(role)}
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
                        <dt>Elevation</dt>
                        <dd>{formatAltitudeFt(focusedFix.elevationFt)}</dd>
                      </div>
                      <div>
                        <dt>Role hints</dt>
                        <dd>{focusedFix.roleHints.join(", ") || "Not available"}</dd>
                      </div>
                      <div>
                        <dt>Used by branches</dt>
                        <dd>
                          {focusedFixBranches.map((branch) => branch.branchIdent).join(", ") ||
                            "No branch mapping"}
                        </dd>
                      </div>
                    </dl>
                  </div>
                ) : (
                  <p>Hover or click a fix to inspect it here.</p>
                )}
              </section>

              <section className="procedure-details-card procedure-details-explorer-card">
                <div className="procedure-details-explorer-head">
                  <div>
                    <p className="procedure-details-overview-label">Interactive Explorer</p>
                    <h3>Focus a branch or fix</h3>
                  </div>
                  <span className="procedure-details-focus-pill">
                    {isPreviewMode ? "Preview" : "Focus"}{" "}
                    {focusedFix?.ident ?? focusedBranch?.branchIdent ?? "—"}
                  </span>
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
                      <strong>{branch.branchIdent}</strong>
                      <span>{branchRoleLabel(branch.branchRole)}</span>
                    </button>
                  ))}
                </div>

                {focusedPolyline ? (
                  <div className="procedure-details-fix-strip">
                    {focusedPolyline.points.map((point) => (
                      <button
                        type="button"
                        key={`${point.branchId}-${point.fixId}`}
                        className={`procedure-details-fix-chip ${
                          point.fixId === focusedFixId ? "is-active" : ""
                        }`}
                        onMouseEnter={() => handlePreviewFix(point.fixId, point.branchId)}
                        onMouseLeave={clearPreview}
                        onClick={() => handleSelectFix(point.fixId, point.branchId)}
                      >
                        <strong>{point.ident}</strong>
                        <span>{displayTerm(point.role)}</span>
                      </button>
                    ))}
                  </div>
                ) : null}
              </section>

              <section className="procedure-details-card procedure-details-terms-card">
                <h3>Key Terms</h3>
                <p className="procedure-details-section-intro">
                  Pick a term only when you need the full explanation.
                </p>
                <div className="procedure-details-term-chip-row">
                  {glossaryTerms.map((term) => (
                    <button
                      key={term}
                      type="button"
                      className={`procedure-details-term-chip ${
                        activeGlossaryTerm === term ? "is-active" : ""
                      }`}
                      onClick={() => setSelectedGlossaryTerm(term)}
                    >
                      {displayTerm(term)}
                    </button>
                  ))}
                </div>
                {activeGlossaryTerm ? (
                  <div className="procedure-details-term-definition">
                    <strong>{displayTerm(activeGlossaryTerm)}</strong>
                    <p>{termMeaning(activeGlossaryTerm)}</p>
                  </div>
                ) : null}
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
