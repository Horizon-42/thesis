/**
 * aircraftTypeNames.ts
 * --------------------
 * The plain names of ICAO type codes, where the app has them: the backend's aircraft catalog (`GET /simulation/aircraft`, the
 * handful of types the dynamics model flies — `A320` is "Airbus A320-200"). A type the catalog does not list keeps its code and
 * no title: a name made up for it would be a guess.
 */

import type { PilotAircraftConfig } from "../pilot/pilotClient";

/** ICAO type code → plain name. */
export type AircraftTypeNames = ReadonlyMap<string, string>;

export const NO_AIRCRAFT_TYPE_NAMES: AircraftTypeNames = new Map();

export function aircraftTypeNames(configs: readonly Pick<PilotAircraftConfig, "code" | "name">[]): AircraftTypeNames {
  return new Map(configs.map((config) => [config.code, config.name]));
}

/** The plain name of `code` (an arrival's type, null when untyped), or undefined when the app has none. */
export function typeNameOf(names: AircraftTypeNames, code: string | null): string | undefined {
  return code === null ? undefined : names.get(code);
}
