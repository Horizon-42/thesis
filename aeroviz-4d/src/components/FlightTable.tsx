/**
 * FlightTable.tsx
 * ---------------
 * Collapsible list of the loaded observed flights (collapsed by default). Each row shows the
 * flight's callsign (the entity NAME — the entity id is the full flight_key and stays the
 * row key / selection / lookup identity) plus facts read off its track (initial ground
 * speed V, total flight time) and, from the optimizer, its aircraft mass. When the
 * Prediction comparison is on, a fourth column adds the optimized final time for the
 * selected category, in the optimizer "results" colour (blue). Namesake flights show the
 * same callsign on two rows — the row tooltip carries the full identity.
 * Clicking a row tracks that flight in the Cesium viewer.
 */

import { useState } from "react";
import { useApp } from "../context/AppContext";
import type * as Cesium from "cesium";
import type { ObservedFlightSummary } from "../hooks/useCzmlLoader";
import { useFlightOptimizerData } from "../hooks/useFlightOptimizerData";
import { formatDuration, formatSpeed, formatMass } from "../utils/flightListFormat";
import { COMPARISON_KIND_COLORS } from "../utils/trajectoryRenderModel";

interface FlightTableProps {
  /** Flight IDs from useCzmlLoader. */
  flightIds: string[];
  /** Per-flight duration + initial ground speed from useCzmlLoader. */
  flightSummaries: Record<string, ObservedFlightSummary>;
}

/** The optimized final time reuses the "Optimize results" (simulator) path colour. */
const OPTIMIZED_TIME_COLOR = COMPARISON_KIND_COLORS.simulator;

export default function FlightTable({ flightIds, flightSummaries }: FlightTableProps) {
  const { viewer, selectedFlightId, setSelectedFlightId } = useApp();
  const { byFlightKey, comparisonActive } = useFlightOptimizerData();
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
              <th title="Optimizer aircraft mass (tonnes)">
                Mass<span className="flight-table-unit">t</span>
              </th>
              <th title="Observed track duration (m:ss)">Time</th>
              {comparisonActive ? (
                <th style={{ color: OPTIMIZED_TIME_COLOR }} title="Optimized final time for the selected category (m:ss)">
                  Opt
                </th>
              ) : null}
            </tr>
          </thead>
          <tbody>
            {flightIds.map((id) => {
              const summary = flightSummaries[id];
              // Observed entity id === comparison group key (both are the flight_key),
              // so this is an exact per-flight join — never a callsign match.
              const optimizer = byFlightKey.get(id);
              const callsign = summary?.callsign ?? id;
              return (
                <tr
                  key={id}
                  className={id === selectedFlightId ? "selected" : ""}
                  onClick={() => handleRowClick(id)}
                  style={{ cursor: "pointer" }}
                >
                  <td
                    className={`flight-table-id${
                      optimizer?.failed
                        ? " flight-table-failed"
                        : optimizer?.offTarget
                          ? " flight-table-offtarget"
                          : ""
                    }`}
                    title={
                      optimizer?.failed
                        ? `${id} — optimization failed`
                        : optimizer?.offTarget
                          ? `${id} — optimized but missed the target (off target)`
                          : id
                    }
                  >
                    {callsign}
                  </td>
                  <td>{formatSpeed(optimizer?.initialVMps ?? null)}</td>
                  <td>{formatMass(optimizer?.massKg ?? null)}</td>
                  <td>{formatDuration(summary?.durationS ?? null)}</td>
                  {comparisonActive ? (
                    <td style={{ color: OPTIMIZED_TIME_COLOR }}>
                      {formatDuration(optimizer?.optimizedTimeS ?? null)}
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
