/**
 * useLandingsManifest.ts
 * ----------------------
 * Fetches the per-airport landings manifest (public/data/airports/<ICAO>/
 * landings/index.json), which lists the per-runway landing CZML files. Airports
 * without landings simply 404, reported as status "empty" (no error).
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
}

export function useLandingsManifest(airportCode: string): LandingsManifestState {
  const [state, setState] = useState<LandingsManifestState>({ manifest: null, status: "idle" });

  useEffect(() => {
    if (!airportCode) {
      setState({ manifest: null, status: "idle" });
      return;
    }

    let cancelled = false;
    setState({ manifest: null, status: "loading" });

    fetchJson<unknown>(airportLandingsIndexUrl(airportCode))
      .then((data) => {
        if (cancelled) return;
        if (!isLandingsManifest(data)) {
          throw new Error(`landings index for ${airportCode} is not a valid manifest`);
        }
        setState({ manifest: data, status: "ready" });
      })
      .catch((error) => {
        if (cancelled) return;
        if (isMissingJsonAsset(error)) {
          setState({ manifest: null, status: "empty" });
          return;
        }
        console.error("[useLandingsManifest] Failed to load landings manifest:", error);
        setState({ manifest: null, status: "error" });
      });

    return () => {
      cancelled = true;
    };
  }, [airportCode]);

  return state;
}
