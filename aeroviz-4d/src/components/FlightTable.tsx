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

interface FlightTableProps {
  /** Flight IDs from useCzmlLoader */
  flightIds: string[];
}

export default function FlightTable({ flightIds }: FlightTableProps) {
  const { viewer, selectedFlightId, setSelectedFlightId } = useApp();

  if (flightIds.length === 0) return null; // hide if no data loaded

  function handleRowClick(id: string) {
    // TODO ① — When a row is clicked:
    //   1. Call setSelectedFlightId(id)
    //   2. If viewer is available, find the entity and set viewer.trackedEntity:
    //        viewer.trackedEntity = viewer.dataSources
    //          .getByName("...")[0]?.entities.getById(id) ?? undefined;
    //
    // Hint: CZML entities live inside a CzmlDataSource, not in viewer.entities.
    // You need to search dataSources.  The CzmlDataSource doesn't have a fixed name
    // unless you set one — you may need to loop over viewer.dataSources to find it.
    //
    // Alternative approach: search ALL dataSources:
    //   for (let i = 0; i < viewer.dataSources.length; i++) {
    //     const entity = viewer.dataSources.get(i).entities.getById(id);
    //     if (entity) { viewer.trackedEntity = entity; break; }
    //   }
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
              {/* TODO ② — Add a "Tracking" badge when id === selectedFlightId */}
              <td>{id === selectedFlightId ? "📡 tracking" : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
