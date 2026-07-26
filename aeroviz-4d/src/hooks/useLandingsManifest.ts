/**
 * useLandingsManifest.ts
 * ----------------------
 * Fetches the per-airport landings manifest (public/data/airports/<ICAO>/
 * landings/index.json), which lists runway counts and the canonical observed CZML
 * selector. Multiple runway entries may intentionally point to the same file; filtering
 * happens in-memory. Airports without landings simply 404, reported as status "empty".
 */

import { useEffect, useState } from "react";
import {
  airportLandingsIndexUrl,
  isLandingsManifest,
  type LandingsManifest,
} from "../data/airportData";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";

export type LandingsManifestStatus = "idle" | "loading" | "ready" | "empty" | "error";

export interface LandingsManifestState {
  manifest: LandingsManifest | null;
  status: LandingsManifestStatus;
  error: string | null;
}

export function useLandingsManifest(airportCode: string): LandingsManifestState {
  const [state, setState] = useState<LandingsManifestState>({
    manifest: null,
    status: "idle",
    error: null,
  });

  useEffect(() => {
    if (!airportCode) {
      setState({ manifest: null, status: "idle", error: null });
      return;
    }

    let cancelled = false;
    setState({ manifest: null, status: "loading", error: null });

    fetchJson<unknown>(airportLandingsIndexUrl(airportCode))
      .then((data) => {
        if (cancelled) return;
        if (!isLandingsManifest(data)) {
          throw new Error(
            `Observed data for ${airportCode} does not use ` +
              `"observed-landings-v2-canonical". Rebuild it with ` +
              `prepare_scenario_inputs.py --airport ${airportCode}.`,
          );
        }
        setState({ manifest: data, status: "ready", error: null });
      })
      .catch((error) => {
        if (cancelled) return;
        if (isMissingJsonAsset(error)) {
          setState({ manifest: null, status: "empty", error: null });
          return;
        }
        console.error("[useLandingsManifest] Failed to load landings manifest:", error);
        setState({
          manifest: null,
          status: "error",
          error: error instanceof Error ? error.message : String(error),
        });
      });

    return () => {
      cancelled = true;
    };
  }, [airportCode]);

  return state;
}
