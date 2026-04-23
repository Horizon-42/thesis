import { useEffect, useMemo, useRef, useState } from "react";

import { useApp } from "../context/AppContext";
import type {
  ProcedureChartManifestEntry,
  ProcedureChartsManifest,
  ProcedureDetailDocument,
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
  type ProcedureRunwayMarker,
} from "../utils/procedureDetailsGeometry";

const SVG_WIDTH = 960;
const SVG_HEIGHT = 320;
const SVG_PADDING = 56;

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

interface SvgChartProps {
  polylines: ProcedureBranchPolyline[];
  runwayMarker: ProcedureRunwayMarker | null;
  selectedFixId: string | null;
  onSelectFix: (fixId: string) => void;
}

function ProcedurePlanView({
  polylines,
  runwayMarker,
  selectedFixId,
  onSelectFix,
}: SvgChartProps) {
  const allPoints = polylines.flatMap((branch) => branch.points);
  if (allPoints.length === 0) {
    return <div className="procedure-details-empty-chart">No positioned fixes available for the plan view yet.</div>;
  }

  const scaleX = chartScale(
    allPoints.map((point) => point.xM),
    SVG_PADDING,
    SVG_WIDTH - SVG_PADDING,
  );
  const scaleY = chartScale(
    allPoints.map((point) => point.yM),
    SVG_HEIGHT - SVG_PADDING,
    SVG_PADDING,
  );

  return (
    <svg
      className="procedure-details-chart-svg"
      viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
      role="img"
      aria-label="Procedure plan view"
    >
      <text className="procedure-details-chart-title" x={SVG_PADDING} y={28}>
        Plan View
      </text>
      <text className="procedure-details-chart-subtitle" x={SVG_PADDING} y={46}>
        North-up layout of the published branches and key fixes.
      </text>

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
        const pathD = branch.points
          .map((point, index) => `${index === 0 ? "M" : "L"} ${scaleX(point.xM)} ${scaleY(point.yM)}`)
          .join(" ");

        return (
          <g key={branch.branchId}>
            <path
              d={pathD}
              className={`procedure-details-branch-line procedure-details-branch-${branch.branchRole}`}
            />
            {branch.points.map((point) => {
              const selected = point.fixId === selectedFixId;
              return (
                <g key={`${branch.branchId}-${point.fixId}`}>
                  <circle
                    cx={scaleX(point.xM)}
                    cy={scaleY(point.yM)}
                    r={selected ? 6 : 4}
                    className={`procedure-details-fix-point ${selected ? "is-selected" : ""}`}
                    onClick={() => onSelectFix(point.fixId)}
                  />
                  <text
                    x={scaleX(point.xM) + 8}
                    y={scaleY(point.yM) - 8}
                    className={`procedure-details-fix-label ${selected ? "is-selected" : ""}`}
                    onClick={() => onSelectFix(point.fixId)}
                  >
                    {point.ident}
                  </text>
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
  selectedFixId,
  onSelectFix,
}: Omit<SvgChartProps, "runwayMarker">) {
  const allPoints = polylines.flatMap((branch) => branch.points);
  if (allPoints.length === 0) {
    return (
      <div className="procedure-details-empty-chart">
        No altitude-supported samples are available for the vertical profile yet.
      </div>
    );
  }

  const scaleX = chartScale(
    allPoints.map((point) => point.distanceM),
    SVG_PADDING,
    SVG_WIDTH - SVG_PADDING,
  );
  const scaleY = chartScale(
    allPoints.map((point) => point.altitudeFt ?? 0),
    SVG_HEIGHT - SVG_PADDING,
    SVG_PADDING,
  );
  const axisTicks = 5;
  const maxDistanceM = Math.max(...allPoints.map((point) => point.distanceM), 1);
  const maxAltitudeFt = Math.max(...allPoints.map((point) => point.altitudeFt ?? 0), 1);

  return (
    <svg
      className="procedure-details-chart-svg"
      viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
      role="img"
      aria-label="Procedure vertical profile"
    >
      <text className="procedure-details-chart-title" x={SVG_PADDING} y={28}>
        Vertical Profile
      </text>
      <text className="procedure-details-chart-subtitle" x={SVG_PADDING} y={46}>
        Along-path distance versus altitude, with missing heights repaired by interpolation.
      </text>

      {Array.from({ length: axisTicks + 1 }, (_, index) => {
        const ratio = index / axisTicks;
        const y = SVG_PADDING + ratio * (SVG_HEIGHT - SVG_PADDING * 2);
        const altitude = Math.round(maxAltitudeFt * (1 - ratio));
        return (
          <g key={`y-${index}`}>
            <line
              x1={SVG_PADDING}
              y1={y}
              x2={SVG_WIDTH - SVG_PADDING}
              y2={y}
              className="procedure-details-grid-line"
            />
            <text x={SVG_PADDING - 12} y={y + 4} className="procedure-details-axis-label">
              {altitude.toLocaleString()} ft
            </text>
          </g>
        );
      })}

      {Array.from({ length: axisTicks + 1 }, (_, index) => {
        const ratio = index / axisTicks;
        const x = SVG_PADDING + ratio * (SVG_WIDTH - SVG_PADDING * 2);
        const distanceNm = nmFromMeters(maxDistanceM * ratio);
        return (
          <g key={`x-${index}`}>
            <line
              x1={x}
              y1={SVG_PADDING}
              x2={x}
              y2={SVG_HEIGHT - SVG_PADDING}
              className="procedure-details-grid-line"
            />
            <text x={x} y={SVG_HEIGHT - SVG_PADDING + 22} className="procedure-details-axis-label is-centered">
              {distanceNm.toFixed(1)} NM
            </text>
          </g>
        );
      })}

      {polylines.map((branch) => {
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
              className={`procedure-details-branch-line procedure-details-branch-${branch.branchRole}`}
            />
            {branch.points.map((point) => {
              const selected = point.fixId === selectedFixId;
              return (
                <g key={`${branch.branchId}-${point.fixId}`}>
                  <circle
                    cx={scaleX(point.distanceM)}
                    cy={scaleY(point.altitudeFt ?? 0)}
                    r={selected ? 6 : 4}
                    className={`procedure-details-fix-point ${selected ? "is-selected" : ""}`}
                    onClick={() => onSelectFix(point.fixId)}
                  />
                  <text
                    x={scaleX(point.distanceM)}
                    y={scaleY(point.altitudeFt ?? 0) - 10}
                    className={`procedure-details-fix-label is-centered ${selected ? "is-selected" : ""}`}
                    onClick={() => onSelectFix(point.fixId)}
                  >
                    {point.ident}
                  </text>
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
      return;
    }
    const defaultFix =
      procedureDocument.runway.landingThresholdFixRef ?? procedureDocument.fixes[0]?.fixId ?? null;
    setSelectedFixId((current) => {
      if (current && procedureDocument.fixes.some((fix) => fix.fixId === current)) return current;
      return defaultFix;
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
  const selectedFix = useMemo(
    () => findFix(procedureDocument ?? null, selectedFixId),
    [procedureDocument, selectedFixId],
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
  const glossaryTerms = summaryTerms(procedureDocument);

  return (
    <div className="procedure-details-page">
      <header className="procedure-details-header">
        <div>
          <p className="procedure-details-eyebrow">AeroViz-4D Research Companion</p>
          <h1>Procedure Details</h1>
          <p className="procedure-details-intro">
            This page turns the intermediate RNAV procedure data into a user-friendly briefing view.
            It is designed to explain what the approach is asking an aircraft to do, not to replace
            official flight documents.
          </p>
        </div>

        <div className="procedure-details-header-actions">
          <button
            type="button"
            className="procedure-details-nav-button"
            onClick={() => navigateWithinApp("/")}
          >
            Back To 3D Scene
          </button>

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
              }}
            >
              {airports.map((airport) => (
                <option key={airport.code} value={airport.code}>
                  {airport.code} - {airport.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      <section className="procedure-details-note" role="note">
        Research use only. Always use the official FAA or local published chart for operational decisions.
      </section>

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
        <div className="procedure-details-layout">
          <aside className="procedure-details-sidebar">
            <div className="procedure-details-sidebar-card">
              <h2>{indexManifest.airportName}</h2>
              <p className="procedure-details-sidebar-meta">
                {indexManifest.airport} procedure cycle {indexManifest.sourceCycle ?? "Unknown"}
              </p>
            </div>

            <div className="procedure-details-sidebar-card">
              <h3>Runways</h3>
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
                    }}
                  >
                    <span>{formatRunway(runway.runwayIdent)}</span>
                    <small>{runway.procedures.length} procedure</small>
                  </button>
                ))}
              </div>
            </div>

            <div className="procedure-details-sidebar-card">
              <h3>Procedures</h3>
              <div className="procedure-details-procedure-list">
                {selectedRunway?.procedures.map((procedure) => (
                  <button
                    type="button"
                    key={procedure.procedureUid}
                    className={`procedure-details-procedure-button ${
                      procedure.procedureUid === selectedProcedureUid ? "is-active" : ""
                    }`}
                    onClick={() => setSelectedProcedureUid(procedure.procedureUid)}
                  >
                    <span>{procedure.chartName}</span>
                    <small>{procedure.approachModes.join(" / ") || "Mode data unavailable"}</small>
                  </button>
                )) ?? <p>Select a runway to see its procedures.</p>}
              </div>
            </div>
          </aside>

          <main className="procedure-details-main">
            {procedureError ? <div className="procedure-details-error">{procedureError}</div> : null}
            {!procedureDocument ? (
              <div className="procedure-details-loading">Loading selected procedure…</div>
            ) : (
              <>
                <section className="procedure-details-card procedure-details-overview">
                  <div>
                    <p className="procedure-details-overview-label">Selected Procedure</p>
                    <h2>{procedureDocument.procedure.chartName}</h2>
                    <p className="procedure-details-overview-meta">
                      {formatRunway(procedureDocument.runway.ident)} ·{" "}
                      {procedureDocument.procedure.approachModes.join(" / ") || "Modes unavailable"}
                    </p>
                  </div>

                  <div className="procedure-details-overview-grid">
                    <div>
                      <span className="procedure-details-key">Procedure family</span>
                      <strong>{procedureDocument.procedure.procedureFamily}</strong>
                    </div>
                    <div>
                      <span className="procedure-details-key">Threshold elevation</span>
                      <strong>{formatAltitudeFt(procedureDocument.runway.threshold?.elevationFt)}</strong>
                    </div>
                    <div>
                      <span className="procedure-details-key">Base branch</span>
                      <strong>{procedureDocument.procedure.baseBranchIdent}</strong>
                    </div>
                    <div>
                      <span className="procedure-details-key">Known simplifications</span>
                      <strong>{procedureDocument.validation.knownSimplifications.length}</strong>
                    </div>
                  </div>
                </section>

                <section className="procedure-details-chart-grid">
                  <div className="procedure-details-card">
                    <ProcedurePlanView
                      polylines={polylines}
                      runwayMarker={runwayMarker}
                      selectedFixId={selectedFixId}
                      onSelectFix={setSelectedFixId}
                    />
                  </div>

                  <div className="procedure-details-card">
                    <ProcedureVerticalProfile
                      polylines={polylines}
                      selectedFixId={selectedFixId}
                      onSelectFix={setSelectedFixId}
                    />
                  </div>
                </section>

                <section className="procedure-details-dual-grid">
                  <section className="procedure-details-card">
                    <h3>Leg Ladder</h3>
                    <p className="procedure-details-section-intro">
                      Each row is one coded leg from the intermediate dataset, translated into plain
                      language.
                    </p>
                    <div className="procedure-details-table-wrap">
                      <table className="procedure-details-table">
                        <thead>
                          <tr>
                            <th>Branch</th>
                            <th>Seq</th>
                            <th>Fix</th>
                            <th>Role</th>
                            <th>Path</th>
                            <th>Altitude</th>
                            <th>Meaning</th>
                          </tr>
                        </thead>
                        <tbody>
                          {procedureDocument.branches.flatMap((branch) =>
                            branch.legs.map((leg) => (
                              <tr
                                key={leg.legId}
                                className={selectedFixId === leg.path.endFixRef ? "is-selected" : ""}
                                onClick={() => setSelectedFixId(leg.path.endFixRef)}
                              >
                                <td>{branch.branchIdent}</td>
                                <td>{leg.sequence}</td>
                                <td>{leg.path.endFixRef.replace("fix:", "")}</td>
                                <td title={roleMeaning(leg.roleAtEnd)}>{leg.roleAtEnd}</td>
                                <td title={terminatorMeaning(leg.path.pathTerminator)}>
                                  {leg.path.pathTerminator}
                                </td>
                                <td>{formatAltitudeFt(leg.constraints.geometryAltitudeFt)}</td>
                                <td>{terminatorMeaning(leg.path.pathTerminator)}</td>
                              </tr>
                            )),
                          )}
                        </tbody>
                      </table>
                    </div>
                  </section>

                  <section className="procedure-details-card">
                    <h3>Fix Inspector</h3>
                    {selectedFix ? (
                      <div className="procedure-details-fix-inspector">
                        <h4>{selectedFix.ident}</h4>
                        <p>{roleMeaning(selectedFix.roleHints[0] ?? selectedFix.kind)}</p>
                        <dl className="procedure-details-definition-list">
                          <div>
                            <dt>Kind</dt>
                            <dd>{selectedFix.kind}</dd>
                          </div>
                          <div>
                            <dt>Latitude</dt>
                            <dd>{formatCoordinate(selectedFix.position?.lat)}</dd>
                          </div>
                          <div>
                            <dt>Longitude</dt>
                            <dd>{formatCoordinate(selectedFix.position?.lon)}</dd>
                          </div>
                          <div>
                            <dt>Elevation</dt>
                            <dd>{formatAltitudeFt(selectedFix.elevationFt)}</dd>
                          </div>
                          <div>
                            <dt>Role hints</dt>
                            <dd>{selectedFix.roleHints.join(", ") || "Not available"}</dd>
                          </div>
                          <div>
                            <dt>Used by branches</dt>
                            <dd>
                              {selectedFixBranches.map((branch) => branch.branchIdent).join(", ") ||
                                "No branch mapping"}
                            </dd>
                          </div>
                        </dl>
                      </div>
                    ) : (
                      <p>Select a fix from a chart or the table to inspect it here.</p>
                    )}
                  </section>
                </section>

                <section className="procedure-details-dual-grid">
                  <section className="procedure-details-card">
                    <h3>Glossary</h3>
                    <ul className="procedure-details-glossary-list">
                      {glossaryTerms.map((term) => (
                        <li key={term}>
                          <strong>{term.replace("_TERMINATOR", "")}</strong>
                          <span>
                            {TERM_EXPLANATIONS[term] ??
                              "Procedure coding term used by the chart and the intermediate dataset."}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </section>

                  <section className="procedure-details-card">
                    <h3>Reference Links</h3>
                    <p className="procedure-details-section-intro">
                      Use the official chart source for operational work. The local PDF is a convenience
                      copy when one has been published into the browser-ready dataset.
                    </p>
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
                          No local chart PDF has been published for this procedure yet.
                        </p>
                      )}
                    </div>
                  </section>
                </section>
              </>
            )}
          </main>
        </div>
      ) : null}
    </div>
  );
}
