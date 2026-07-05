/**
 * useFlightOptimizerData.ts
 * -------------------------
 * Per-flight facts the flight list borrows from the optimizer's comparison index:
 *   • the flight's **V** (initial ground speed, m/s) and aircraft **mass** (kg) — both from the
 *     scenario initial state, so they are the SAME value the optimizer started from (no
 *     independent frontend re-estimate) and exist for EVERY flight, solved OR failed. They are
 *     category-independent, so they show even when the comparison overlay is off (read from
 *     whichever category is active, defaulting to the first);
 *   • the optimized **final time** (s) for the currently-selected category — shown only while
 *     the comparison overlay is on (`comparisonActive`), and only for solved flights;
 *   • whether the flight's optimization **failed** — so the list can flag it.
 */

import { useEffect, useMemo, useState } from "react";
import { useApp } from "../context/AppContext";
import { useComparisonCategories } from "./useComparisonCategories";
import {
  airportComparisonIndexUrl,
  isComparisonIndex,
  type ComparisonGroup,
  type OptimizationStats,
} from "../data/airportData";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";

export interface FlightOptimizerDatum {
  /** Initial ground speed in m/s (scenario initial state), or null if unavailable. */
  initialVMps: number | null;
  /** Aircraft mass in kg (scenario initial state), or null if unavailable. */
  massKg: number | null;
  /** Optimized final time in seconds for the loaded category, or null if not solved. */
  optimizedTimeS: number | null;
  /** True when this flight's optimization failed (no optimized trajectory). */
  failed: boolean;
  /** True when it solved but the final state FAILED the evaluation gates (off target). */
  offTarget: boolean;
}

export interface FlightOptimizerData {
  byFlightId: Map<string, FlightOptimizerDatum>;
  /** True when the comparison overlay is on and a category is selected → show the optimized column. */
  comparisonActive: boolean;
  /** The loaded category's optimization stats (solve rate + placeholders), or null. */
  stats: OptimizationStats | null;
}

const EMPTY: Map<string, FlightOptimizerDatum> = new Map();

export function useFlightOptimizerData(): FlightOptimizerData {
  const { activeAirportCode, trajectoryComparison, trajectoryComparisonCategory } = useApp();
  const { categories } = useComparisonCategories(activeAirportCode);

  // Mass is category-independent, so when the overlay is off we still read the first category's
  // index for it; when the overlay is on we read the selected category (same mass + its finalTimeS).
  const categoryDir =
    (trajectoryComparison && trajectoryComparisonCategory) || categories[0]?.dir || null;
  const comparisonActive = trajectoryComparison && !!trajectoryComparisonCategory;

  const [byFlightId, setByFlightId] = useState<Map<string, FlightOptimizerDatum>>(EMPTY);
  const [stats, setStats] = useState<OptimizationStats | null>(null);

  useEffect(() => {
    if (!activeAirportCode || !categoryDir) {
      setByFlightId(EMPTY);
      setStats(null);
      return;
    }

    let cancelled = false;
    fetchJson<unknown>(airportComparisonIndexUrl(activeAirportCode, categoryDir))
      .then((data) => {
        if (cancelled) return;
        if (!isComparisonIndex(data)) {
          throw new Error(`comparison index for ${activeAirportCode}/${categoryDir} is malformed`);
        }
        // Every group (solved AND failed) carries V + mass; only solved carry an optimized time.
        const map = new Map<string, FlightOptimizerDatum>();
        for (const group of data.groups as ComparisonGroup[]) {
          map.set(group.flightId, {
            initialVMps: group.initialVMps ?? group.initialState?.V ?? null,
            massKg: group.massKg ?? group.initialState?.m ?? null,
            // offTarget groups ARE solved (they carry an optimized trajectory + time);
            // they just missed the evaluation gates at the end.
            optimizedTimeS: group.status !== "failed" ? group.finalTimeS : null,
            failed: group.status === "failed",
            offTarget: group.status === "offTarget",
          });
        }
        setByFlightId(map);
        setStats(data.optimization ?? null);
      })
      .catch((error) => {
        if (cancelled) return;
        if (!isMissingJsonAsset(error)) {
          console.warn("[useFlightOptimizerData] failed to load comparison index", error);
        }
        setByFlightId(EMPTY);
        setStats(null);
      });

    return () => {
      cancelled = true;
    };
  }, [activeAirportCode, categoryDir]);

  return useMemo(() => ({ byFlightId, comparisonActive, stats }), [byFlightId, comparisonActive, stats]);
}
