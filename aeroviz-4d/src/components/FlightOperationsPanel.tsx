import { useState } from "react";
import FlightTable from "./FlightTable";
import PilotPanel from "./PilotPanel";

interface FlightOperationsPanelProps {
  flightIds: string[];
}

export default function FlightOperationsPanel({ flightIds }: FlightOperationsPanelProps) {
  const [isFlightListOpen, setFlightListOpen] = useState(false);
  const hasFlights = flightIds.length > 0;

  return (
    <div className="flight-ops-panel">
      <header className="flight-ops-header">
        <div>
          <h3>Flight Ops</h3>
          <span>{hasFlights ? `${flightIds.length} flights loaded` : "No CZML flights"}</span>
        </div>
        <button
          type="button"
          onClick={() => setFlightListOpen((current) => !current)}
          disabled={!hasFlights}
        >
          {isFlightListOpen ? "Hide List" : "Show List"}
        </button>
      </header>

      <PilotPanel />

      {isFlightListOpen && hasFlights ? (
        <section className="flight-list-section">
          <FlightTable flightIds={flightIds} />
        </section>
      ) : null}
    </div>
  );
}
