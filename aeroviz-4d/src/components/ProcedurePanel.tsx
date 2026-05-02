import { useEffect, useMemo, useState } from "react";
import { useApp } from "../context/AppContext";
import {
  loadProcedureRenderBundleData,
  type ProcedureRenderBundleData,
} from "../data/procedureRenderBundle";
import { isMissingJsonAsset } from "../utils/fetchJson";
import { navigateWithinApp } from "../utils/navigation";

const RUNWAY_ORDER = ["RW05L", "RW05R", "RW23L", "RW23R", "RW32"];

interface ProcedureBranchItem {
  branchId: string;
  runwayIdent: string;
  procedureIdent: string;
  procedureName: string;
  procedureFamily: string;
  branchIdent: string;
  branchType: string;
  defaultVisible: boolean;
  warnings: string[];
}

interface ProcedureGroup {
  procedureIdent: string;
  procedureName: string;
  branches: ProcedureBranchItem[];
}

interface RunwayGroup {
  runwayIdent: string;
  procedures: ProcedureGroup[];
}

interface ProtectedGeometryStatus {
  caEndpoints: number;
  caCenterlines: number;
  estimatedCaSurfaces: number;
  lnavVnavOcs: number;
  precisionSurfaces: number;
  turningMissedPrimitives: number;
  missingFinalSurfaces: number;
  sourceIncompleteDiagnostics: number;
}

function branchSortKey(branch: ProcedureBranchItem): string {
  const branchRank = branch.branchType === "STRAIGHT_IN" ? "0" : "1";
  return `${branchRank}-${branch.branchIdent}`;
}

function branchIsVisible(
  branch: ProcedureBranchItem,
  procedureVisibility: Record<string, boolean>,
): boolean {
  return procedureVisibility[branch.branchId] ?? branch.defaultVisible;
}

function diagnosticMessages(
  data: ProcedureRenderBundleData,
  branchId: string,
  sourceBranchId: string,
): string[] {
  const renderBundle = data.renderBundles.find((bundle) =>
    bundle.branchBundles.some((branch) => branch.branchId === branchId),
  );
  const branchBundle = renderBundle?.branchBundles.find((branch) => branch.branchId === branchId);
  const sourceWarnings =
    data.documents
      .flatMap((document) => document.branches)
      .find((branch) => branch.branchId === sourceBranchId)
      ?.warnings ?? [];
  const geometryWarnings =
    branchBundle?.segmentBundles.flatMap((segment) =>
      segment.diagnostics.map((diagnostic) => diagnostic.message),
    ) ?? [];
  return [...sourceWarnings, ...geometryWarnings];
}

function branchItemsFromRenderData(data: ProcedureRenderBundleData): ProcedureBranchItem[] {
  return data.packages.flatMap((pkg) =>
    pkg.branches.map((branch) => ({
      branchId: branch.branchId,
      runwayIdent: branch.runwayId ?? "Unassigned",
      procedureIdent: pkg.procedureId,
      procedureName: pkg.procedureName,
      procedureFamily: pkg.procedureFamily,
      branchIdent: branch.legacy.branchIdent,
      branchType: branch.branchRole,
      defaultVisible: branch.legacy.defaultVisible,
      warnings: diagnosticMessages(data, branch.branchId, branch.legacy.sourceBranchId),
    })),
  );
}

function protectedGeometryStatus(data: ProcedureRenderBundleData): ProtectedGeometryStatus {
  const segmentBundles = data.renderBundles.flatMap((bundle) =>
    bundle.branchBundles.flatMap((branch) => branch.segmentBundles),
  );

  return {
    caEndpoints: segmentBundles.reduce((sum, segment) => sum + segment.missedCaEndpoints.length, 0),
    caCenterlines: segmentBundles.reduce((sum, segment) => sum + segment.missedCaCenterlines.length, 0),
    estimatedCaSurfaces: segmentBundles.filter(
      (segment) => segment.missedSectionSurface?.constructionStatus === "ESTIMATED_CA",
    ).length,
    lnavVnavOcs: segmentBundles.filter((segment) => segment.lnavVnavOcs !== null).length,
    precisionSurfaces: segmentBundles.reduce(
      (sum, segment) => sum + segment.precisionFinalSurfaces.length,
      0,
    ),
    turningMissedPrimitives: segmentBundles.reduce(
      (sum, segment) => sum + segment.missedTurnDebugPrimitives.length,
      0,
    ),
    missingFinalSurfaces: segmentBundles.reduce(
      (sum, segment) => sum + (segment.finalSurfaceStatus?.missingSurfaceTypes.length ?? 0),
      0,
    ),
    sourceIncompleteDiagnostics: data.renderBundles.reduce(
      (sum, bundle) =>
        sum + bundle.diagnostics.filter((diagnostic) => diagnostic.code === "SOURCE_INCOMPLETE").length,
      0,
    ),
  };
}

function buildGroups(branches: ProcedureBranchItem[]): RunwayGroup[] {
  const byRunway = new Map<string, ProcedureBranchItem[]>();
  branches.forEach((branch) => {
    const existing = byRunway.get(branch.runwayIdent) ?? [];
    byRunway.set(branch.runwayIdent, [...existing, branch]);
  });

  return [...byRunway.entries()]
    .sort(([left], [right]) => {
      const leftIndex = RUNWAY_ORDER.indexOf(left);
      const rightIndex = RUNWAY_ORDER.indexOf(right);
      return (leftIndex < 0 ? 999 : leftIndex) - (rightIndex < 0 ? 999 : rightIndex);
    })
    .map(([runwayIdent, runwayBranches]) => {
      const byProcedure = new Map<string, ProcedureBranchItem[]>();
      runwayBranches.forEach((branch) => {
        const existing = byProcedure.get(branch.procedureIdent) ?? [];
        byProcedure.set(branch.procedureIdent, [...existing, branch]);
      });

      const procedures = [...byProcedure.entries()]
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([procedureIdent, procedureBranches]) => {
          const sortedBranches = [...procedureBranches].sort((a, b) =>
            branchSortKey(a).localeCompare(branchSortKey(b)),
          );
          return {
            procedureIdent,
            procedureName: sortedBranches[0]?.procedureName ?? procedureIdent,
            branches: sortedBranches,
          };
        });

      return { runwayIdent, procedures };
    });
}

export default function ProcedurePanel() {
  const {
    layers,
    toggleLayer,
    procedureVisibility,
    setProcedureBranchVisible,
    setProcedureBranchesVisible,
    activeAirportCode,
    selectedProfileRunwayIdent,
    setSelectedProfileRunwayIdent,
    isRunwayProfileOpen,
    setRunwayProfileOpen,
  } = useApp();
  const [branches, setBranches] = useState<ProcedureBranchItem[]>([]);
  const [sourceCycle, setSourceCycle] = useState<string | null>(null);
  const [sourceAirport, setSourceAirport] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [expandedRunways, setExpandedRunways] = useState<Set<string>>(new Set());
  const [geometryStatus, setGeometryStatus] = useState<ProtectedGeometryStatus | null>(null);

  useEffect(() => {
    if (!activeAirportCode) {
      setBranches([]);
      setSourceCycle(null);
      setSourceAirport(null);
      setLoadError(null);
      setExpandedRunways(new Set());
      setGeometryStatus(null);
      return;
    }

    let cancelled = false;
    setLoadError(null);
    loadProcedureRenderBundleData(activeAirportCode)
      .then((data) => {
        if (cancelled) return;
        const branchItems = branchItemsFromRenderData(data);

        setBranches(branchItems);
        setGeometryStatus(protectedGeometryStatus(data));
        setSourceCycle(data.index.sourceCycle ?? null);
        setSourceAirport(data.index.airport || activeAirportCode);
        const firstRunway = buildGroups(branchItems)[0]?.runwayIdent;
        setExpandedRunways(firstRunway ? new Set([firstRunway]) : new Set());
      })
      .catch((error) => {
        if (cancelled) return;

        setBranches([]);
        setGeometryStatus(null);
        setSourceCycle(null);
        setSourceAirport(activeAirportCode);
        setExpandedRunways(new Set());
        if (isMissingJsonAsset(error)) {
          setLoadError(`No procedure-details data for ${activeAirportCode}`);
        } else {
          setLoadError(error instanceof Error ? error.message : String(error));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeAirportCode]);

  const groups = useMemo(() => buildGroups(branches), [branches]);
  const totalWarnings = branches.reduce((sum, branch) => sum + branch.warnings.length, 0);

  const setBranchesVisible = (targetBranches: ProcedureBranchItem[], visible: boolean) => {
    setProcedureBranchesVisible(
      targetBranches.map((branch) => branch.branchId),
      visible,
    );
  };

  const toggleRunwayExpanded = (runwayIdent: string) => {
    setExpandedRunways((prev) => {
      const next = new Set(prev);
      if (next.has(runwayIdent)) next.delete(runwayIdent);
      else next.add(runwayIdent);
      return next;
    });
  };

  const toggleRunwayProfile = (runwayIdent: string) => {
    if (isRunwayProfileOpen && selectedProfileRunwayIdent === runwayIdent) {
      setRunwayProfileOpen(false);
      return;
    }
    setSelectedProfileRunwayIdent(runwayIdent);
    setRunwayProfileOpen(true);
  };

  return (
    <aside className="procedure-panel" aria-label="RNAV procedure controls">
      <header className="procedure-panel-header">
        <div>
          <h3>Procedures</h3>
          <p>{(sourceAirport ?? activeAirportCode) || "Unknown"} CIFP {sourceCycle ?? "unknown"}</p>
        </div>
        <label className="procedure-master-toggle">
          <input
            type="checkbox"
            checked={layers.procedures}
            onChange={() => toggleLayer("procedures")}
          />
          On
        </label>
      </header>

      {loadError ? <p className="procedure-panel-error">{loadError}</p> : null}

      <div className="procedure-panel-summary">
        <span>{branches.length} branches</span>
        <span>{groups.length} runways</span>
        <span>{totalWarnings} warnings</span>
      </div>

      {geometryStatus ? (
        <div className="procedure-panel-geometry-status" aria-label="Protected geometry status">
          <strong>3D status</strong>
          <span>CA endpoints {geometryStatus.caEndpoints}</span>
          <span>CA paths {geometryStatus.caCenterlines}</span>
          <span>CA surfaces {geometryStatus.estimatedCaSurfaces}</span>
          <span>LNAV/VNAV OCS {geometryStatus.lnavVnavOcs}</span>
          <span>W/X/Y {geometryStatus.precisionSurfaces}</span>
          <span>Turning debug {geometryStatus.turningMissedPrimitives}</span>
          <span>Missing final {geometryStatus.missingFinalSurfaces}</span>
          <span>Source gaps {geometryStatus.sourceIncompleteDiagnostics}</span>
        </div>
      ) : null}

      <div className="procedure-runway-list">
        {groups.map((group) => {
          const runwayBranches = group.procedures.flatMap((procedure) => procedure.branches);
          const runwayVisible = runwayBranches.every((branch) =>
            branchIsVisible(branch, procedureVisibility),
          );
          const isExpanded = expandedRunways.has(group.runwayIdent);

          return (
            <section className="procedure-runway-group" key={group.runwayIdent}>
              <div className="procedure-runway-row">
                <button
                  type="button"
                  onClick={() => toggleRunwayExpanded(group.runwayIdent)}
                  aria-expanded={isExpanded}
                >
                  {isExpanded ? "Hide" : "Show"}
                </button>
                <button
                  type="button"
                  className={
                    isRunwayProfileOpen && selectedProfileRunwayIdent === group.runwayIdent
                      ? "procedure-runway-profile-button active"
                      : "procedure-runway-profile-button"
                  }
                  onClick={() => toggleRunwayProfile(group.runwayIdent)}
                >
                  {isRunwayProfileOpen && selectedProfileRunwayIdent === group.runwayIdent
                    ? "Profile On"
                    : "Profile"}
                </button>
                <label>
                  <input
                    type="checkbox"
                    checked={runwayVisible}
                    onChange={() => setBranchesVisible(runwayBranches, !runwayVisible)}
                  />
                  {group.runwayIdent}
                </label>
              </div>

              {isExpanded ? (
                <div className="procedure-list">
                  {group.procedures.map((procedure) => {
                    const procedureVisible = procedure.branches.every((branch) =>
                      branchIsVisible(branch, procedureVisibility),
                    );
                    const warningCount = procedure.branches.reduce(
                      (sum, branch) => sum + branch.warnings.length,
                      0,
                    );

                    return (
                      <div className="procedure-item" key={procedure.procedureIdent}>
                        <label className="procedure-item-title">
                          <input
                            type="checkbox"
                            checked={procedureVisible}
                            onChange={() => setBranchesVisible(procedure.branches, !procedureVisible)}
                          />
                          <span>{procedure.procedureName}</span>
                          {warningCount > 0 ? (
                            <strong title="Simplified or skipped legs">{warningCount}</strong>
                          ) : null}
                        </label>

                        <div className="procedure-branch-list">
                          {procedure.branches.map((branch) => (
                            <label key={branch.branchId}>
                              <input
                                type="checkbox"
                                checked={branchIsVisible(branch, procedureVisibility)}
                                onChange={(event) =>
                                  setProcedureBranchVisible(branch.branchId, event.target.checked)
                                }
                              />
                              <span>{branch.branchType}</span>
                              <code>{branch.branchIdent}</code>
                            </label>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : null}
            </section>
          );
        })}
      </div>

      <div className="procedure-panel-footer">
        <button
          type="button"
          className="procedure-panel-details-button"
          onClick={() =>
            navigateWithinApp(`/procedure-details?airport=${encodeURIComponent(activeAirportCode)}`)
          }
        >
          Procedure Details
        </button>
      </div>
    </aside>
  );
}
