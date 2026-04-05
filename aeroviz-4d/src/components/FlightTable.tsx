/**
 * FlightTable.tsx
 * ---------------
 * Floating table that lists all loaded flights and highlights the selected one.
 * Clicking a row sets it as the tracked entity in the Cesium Viewer.
 *
 * Data flow:
 *   useCzmlLoader → sets flightIds in state
 *   FlightTable reads flightIds from props (passed down from App or via context)
 *   Clicking a row → setSelectedFlightId → viewer.trackedEntity
 */

import { useApp } from "../context/AppContext";
import type * as Cesium from "cesium";

interface FlightTableProps {
  /** Flight IDs from useCzmlLoader */
  flightIds: string[];
}

export default function FlightTable({ flightIds }: FlightTableProps) {
  const { viewer, selectedFlightId, setSelectedFlightId } = useApp();

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
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Flight ID</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {flightIds.map((id, index) => (
            <tr
              key={id}
              className={id === selectedFlightId ? "selected" : ""}
              onClick={() => handleRowClick(id)}
              style={{ cursor: "pointer" }}
            >
              <td>{index + 1}</td>
              <td>{id}</td>
              <td>{id === selectedFlightId ? "Tracking" : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
