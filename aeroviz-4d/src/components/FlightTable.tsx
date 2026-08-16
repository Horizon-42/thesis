/**
 * FlightTable.tsx
 * ---------------
 * Collapsible list of the loaded observed flights (collapsed by default). Each row shows the
 * flight's callsign (the entity NAME — the entity id is the full flight_key and stays the
 * row key / selection / lookup identity) plus facts read off its track (initial ground
 * speed V, total flight time) and its scenario aircraft mass. Comparison mode adds the
 * selected result's final time and paints Prediction outcomes with the same green/red/gray
 * pass/fail/undecided language as Baseline.
 * Clicking a row tracks that flight in the Cesium viewer.
 */

import { useState } from "react";
import { useApp } from "../context/AppContext";
import type * as Cesium from "cesium";
import type { ObservedFlightSummary } from "../utils/observedFlightSummary";
import {
  useFlightComparisonData,
  type ComparisonResultKind,
  type FlightComparisonDatum,
} from "../hooks/useFlightComparisonData";
import { formatDuration, formatSpeed, formatMass } from "../utils/flightListFormat";
import { COMPARISON_KIND_COLORS } from "../utils/trajectoryRenderModel";

interface FlightTableProps {
  /** Flight IDs from the active Baseline or Comparison trajectory layer. */
  flightIds: string[];
  /** Per-flight duration and callsign from the active reference CZML. */
  flightSummaries: Record<string, ObservedFlightSummary>;
}

/** The optimized final time reuses the "Optimize results" (simulator) path colour. */
const OPTIMIZED_TIME_COLOR = COMPARISON_KIND_COLORS.simulator;

export default function FlightTable({ flightIds, flightSummaries }: FlightTableProps) {
  const { viewer, selectedFlightId, setSelectedFlightId } = useApp();
  const { byFlightKey, comparisonActive, resultKind } = useFlightComparisonData();
  const [collapsed, setCollapsed] = useState(true);

  if (flightIds.length === 0) return null; // hide if no data loaded

  function handleRowClick(id: string) {
    setSelectedFlightId(id);
    if (!viewer) return;

    let found: Cesium.Entity | undefined;
    for (let i = 0; i < viewer.dataSources.length; i += 1) {
      const entity = viewer.dataSources.get(i).entities.getById(id);
      if (entity) {
        found = entity;
        break;
      }
    }
    viewer.trackedEntity = found;
  }

  return (
    <div className="flight-table">
      <button
        type="button"
        className="flight-table-toggle"
        aria-expanded={!collapsed}
        onClick={() => setCollapsed((prev) => !prev)}
      >
        <span className="flight-table-caret">{collapsed ? "▸" : "▾"}</span>
        Flights ({flightIds.length})
      </button>

      {collapsed ? null : (
        <div className="flight-table-scroll">
          <table>
            <thead>
              <tr>
                <th>Flight</th>
                <th title="Initial ground speed (m/s)">
                  V<span className="flight-table-unit">m/s</span>
                </th>
                <th title="Scenario aircraft mass (tonnes)">
                  Mass<span className="flight-table-unit">t</span>
                </th>
                <th title="Observed track duration (m:ss)">Time</th>
                {comparisonActive ? (
                  <th
                    style={resultKind === "optimization" ? { color: OPTIMIZED_TIME_COLOR } : undefined}
                    title={`${resultKind === "prediction" ? "Predicted" : "Optimized"} final time for the selected category (m:ss)`}
                  >
                    {resultKind === "prediction" ? "Pred" : "Opt"}
                  </th>
                ) : null}
              </tr>
            </thead>
            <tbody>
              {flightIds.map((id) => {
              const summary = flightSummaries[id];
              // Observed entity id === comparison group key (both are the flight_key),
              // so this is an exact per-flight join — never a callsign match.
              const comparison = byFlightKey.get(id);
              const outcome = comparisonActive
                ? comparisonOutcome(comparison, resultKind)
                : null;
              const callsign = summary?.callsign ?? id;
              return (
                <tr
                  key={id}
                  className={id === selectedFlightId ? "selected" : ""}
                  onClick={() => handleRowClick(id)}
                  style={{ cursor: "pointer" }}
                >
                  <td
                    className={`flight-table-id${outcome ? ` flight-table-${outcome.style}` : ""}`}
                    title={outcome ? `${id} — ${outcome.label}` : id}
                  >
                    {callsign}
                  </td>
                  <td>{formatSpeed(comparison?.initialVMps ?? null)}</td>
                  <td>{formatMass(comparison?.massKg ?? null)}</td>
                  <td>{formatDuration(summary?.durationS ?? null)}</td>
                  {comparisonActive ? (
                    <td
                      className={outcome ? `flight-table-${outcome.style}` : undefined}
                      style={resultKind === "optimization" && !outcome
                        ? { color: OPTIMIZED_TIME_COLOR }
                        : undefined}
                    >
                      {formatDuration(comparison?.resultTimeS ?? null)}
                    </td>
                  ) : null}
                </tr>
              );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

interface ComparisonOutcome {
  style: "pass" | "failed" | "offtarget" | "indeterminate";
  label: string;
}

function comparisonOutcome(
  datum: FlightComparisonDatum | undefined,
  kind: ComparisonResultKind | null,
): ComparisonOutcome | null {
  if (!datum) return null;
  if (kind === "prediction") {
    if (datum.status === "solved") return { style: "pass", label: "prediction passed" };
    if (datum.status === "indeterminate") {
      return { style: "indeterminate", label: "prediction verdict indeterminate" };
    }
    return { style: "failed", label: "prediction failed" };
  }
  if (datum.status === "failed") return { style: "failed", label: "optimization failed" };
  if (datum.status === "offTarget") {
    return { style: "offtarget", label: "optimized but missed the target (off target)" };
  }
  if (datum.status === "indeterminate") {
    return { style: "indeterminate", label: "terminal verdict indeterminate" };
  }
  return null;
}
