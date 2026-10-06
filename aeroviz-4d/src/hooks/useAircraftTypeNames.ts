/**
 * useAircraftTypeNames.ts
 * -----------------------
 * The plain names of the ICAO type codes the app knows (`utils/aircraftTypeNames.ts`), asked of the backend once when the panel
 * that shows type codes mounts. Until they arrive — or when the backend cannot say — every type keeps its code and no title:
 * the names only decorate the codes, so a failed request is logged and never stops the panel.
 */

import { useEffect, useState } from "react";
import { fetchPilotAircraftConfigs } from "../pilot/pilotClient";
import { NO_AIRCRAFT_TYPE_NAMES, aircraftTypeNames, type AircraftTypeNames } from "../utils/aircraftTypeNames";

export function useAircraftTypeNames(): AircraftTypeNames {
  const [names, setNames] = useState<AircraftTypeNames>(NO_AIRCRAFT_TYPE_NAMES);
  useEffect(() => {
    let cancelled = false;
    fetchPilotAircraftConfigs().then(
      (configs) => {
        if (!cancelled) setNames(aircraftTypeNames(configs));
      },
      (error: unknown) => console.warn("[aircraft types] the aircraft catalog could not be read; type codes show without names", error),
    );
    return () => {
      cancelled = true;
    };
  }, []);
  return names;
}
