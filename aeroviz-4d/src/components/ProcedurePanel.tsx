import { useEffect, useMemo, useState } from "react";
import { useApp } from "../context/AppContext";
import { airportDataUrl } from "../data/airportData";
import type {
  ProcedureFeatureCollection,
  ProcedureRouteProperties,
} from "../types/geojson-aviation";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";

const RUNWAY_ORDER = ["RW05L", "RW05R", "RW23L", "RW23R", "RW32"];

interface ProcedureRouteItem {
  routeId: string;
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
  routes: ProcedureRouteItem[];
}

interface RunwayGroup {
  runwayIdent: string;
  procedures: ProcedureGroup[];
}

function routeSortKey(route: ProcedureRouteItem): string {
  const branchRank = route.branchType === "final" ? "0" : "1";
  return `${branchRank}-${route.branchIdent}`;
}

function routeIsVisible(
  route: ProcedureRouteItem,
  procedureVisibility: Record<string, boolean>,
): boolean {
  return procedureVisibility[route.routeId] ?? route.defaultVisible;
}

function asRouteItem(props: ProcedureRouteProperties): ProcedureRouteItem {
  return {
    routeId: props.routeId,
    runwayIdent: props.runwayIdent ?? props.runway ?? "Unassigned",
    procedureIdent: props.procedureIdent,
    procedureName: props.procedureName,
    procedureFamily: props.procedureFamily ?? "UNKNOWN",
    branchIdent: props.branchIdent ?? props.branch,
    branchType: props.branchType ?? "final",
    defaultVisible: props.defaultVisible ?? true,
    warnings: props.warnings ?? [],
  };
}

function buildGroups(routes: ProcedureRouteItem[]): RunwayGroup[] {
  const byRunway = new Map<string, ProcedureRouteItem[]>();
  routes.forEach((route) => {
    const existing = byRunway.get(route.runwayIdent) ?? [];
    byRunway.set(route.runwayIdent, [...existing, route]);
  });

  return [...byRunway.entries()]
    .sort(([left], [right]) => {
      const leftIndex = RUNWAY_ORDER.indexOf(left);
      const rightIndex = RUNWAY_ORDER.indexOf(right);
      return (leftIndex < 0 ? 999 : leftIndex) - (rightIndex < 0 ? 999 : rightIndex);
    })
    .map(([runwayIdent, runwayRoutes]) => {
      const byProcedure = new Map<string, ProcedureRouteItem[]>();
      runwayRoutes.forEach((route) => {
        const existing = byProcedure.get(route.procedureIdent) ?? [];
        byProcedure.set(route.procedureIdent, [...existing, route]);
      });

      const procedures = [...byProcedure.entries()]
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([procedureIdent, procedureRoutes]) => {
          const sortedRoutes = [...procedureRoutes].sort((a, b) =>
            routeSortKey(a).localeCompare(routeSortKey(b)),
          );
          return {
            procedureIdent,
            procedureName: sortedRoutes[0]?.procedureName ?? procedureIdent,
            routes: sortedRoutes,
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
    setProcedureRouteVisible,
    setProcedureRoutesVisible,
    activeAirportCode,
  } = useApp();
  const [routes, setRoutes] = useState<ProcedureRouteItem[]>([]);
  const [sourceCycle, setSourceCycle] = useState<string | null>(null);
  const [sourceAirport, setSourceAirport] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [expandedRunways, setExpandedRunways] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!activeAirportCode) {
      setRoutes([]);
      setSourceCycle(null);
      setSourceAirport(null);
      setLoadError(null);
      setExpandedRunways(new Set());
      return;
    }

    let cancelled = false;
    const proceduresUrl = airportDataUrl(activeAirportCode, "procedures.geojson");
    setLoadError(null);
    fetchJson<ProcedureFeatureCollection>(proceduresUrl)
      .then((geojson) => {
        if (cancelled) return;
        const routeItems = geojson.features
          .filter((feature) => feature.properties.featureType === "procedure-route")
          .map((feature) => asRouteItem(feature.properties as ProcedureRouteProperties));

        setRoutes(routeItems);
        const cycle = geojson.metadata?.sourceCycle;
        setSourceCycle(typeof cycle === "string" ? cycle : null);
        const airport = geojson.metadata?.airport;
        setSourceAirport(typeof airport === "string" ? airport : activeAirportCode);
        const firstRunway = buildGroups(routeItems)[0]?.runwayIdent;
        setExpandedRunways(firstRunway ? new Set([firstRunway]) : new Set());
      })
      .catch((error) => {
        if (cancelled) return;

        setRoutes([]);
        setSourceCycle(null);
        setSourceAirport(activeAirportCode);
        setExpandedRunways(new Set());
        if (isMissingJsonAsset(error)) {
          setLoadError(`No procedures.geojson for ${activeAirportCode}`);
        } else {
          setLoadError(error instanceof Error ? error.message : String(error));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeAirportCode]);

  const groups = useMemo(() => buildGroups(routes), [routes]);
  const totalWarnings = routes.reduce((sum, route) => sum + route.warnings.length, 0);

  const setRoutesVisible = (targetRoutes: ProcedureRouteItem[], visible: boolean) => {
    setProcedureRoutesVisible(
      targetRoutes.map((route) => route.routeId),
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
        <span>{routes.length} branches</span>
        <span>{groups.length} runways</span>
        <span>{totalWarnings} warnings</span>
      </div>

      <div className="procedure-runway-list">
        {groups.map((group) => {
          const runwayRoutes = group.procedures.flatMap((procedure) => procedure.routes);
          const runwayVisible = runwayRoutes.every((route) =>
            routeIsVisible(route, procedureVisibility),
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
                <label>
                  <input
                    type="checkbox"
                    checked={runwayVisible}
                    onChange={() => setRoutesVisible(runwayRoutes, !runwayVisible)}
                  />
                  {group.runwayIdent}
                </label>
              </div>

              {isExpanded ? (
                <div className="procedure-list">
                  {group.procedures.map((procedure) => {
                    const procedureVisible = procedure.routes.every((route) =>
                      routeIsVisible(route, procedureVisibility),
                    );
                    const warningCount = procedure.routes.reduce(
                      (sum, route) => sum + route.warnings.length,
                      0,
                    );

                    return (
                      <div className="procedure-item" key={procedure.procedureIdent}>
                        <label className="procedure-item-title">
                          <input
                            type="checkbox"
                            checked={procedureVisible}
                            onChange={() => setRoutesVisible(procedure.routes, !procedureVisible)}
                          />
                          <span>{procedure.procedureName}</span>
                          {warningCount > 0 ? (
                            <strong title="Simplified or skipped legs">{warningCount}</strong>
                          ) : null}
                        </label>

                        <div className="procedure-branch-list">
                          {procedure.routes.map((route) => (
                            <label key={route.routeId}>
                              <input
                                type="checkbox"
                                checked={routeIsVisible(route, procedureVisibility)}
                                onChange={(event) =>
                                  setProcedureRouteVisible(route.routeId, event.target.checked)
                                }
                              />
                              <span>{route.branchType}</span>
                              <code>{route.branchIdent}</code>
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
    </aside>
  );
}
