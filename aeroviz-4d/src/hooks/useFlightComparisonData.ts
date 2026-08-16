/** Per-flight facts and outcomes from the selected comparison index. */

import { useEffect, useMemo, useState } from "react";
import { useApp } from "../context/AppContext";
import { useComparisonCategories } from "./useComparisonCategories";
import {
  airportComparisonIndexUrl,
  isDrawableComparisonCategory,
  isComparisonIndex,
  type ComparisonGroup,
} from "../data/airportData";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";

export type ComparisonResultKind = "prediction" | "optimization";

export interface FlightComparisonDatum {
  initialVMps: number | null;
  massKg: number | null;
  resultTimeS: number | null;
  status: ComparisonGroup["status"];
}

export interface FlightComparisonData {
  byFlightKey: Map<string, FlightComparisonDatum>;
  comparisonActive: boolean;
  resultKind: ComparisonResultKind | null;
}

const EMPTY: Map<string, FlightComparisonDatum> = new Map();

function resultKind(groups: ComparisonGroup[]): ComparisonResultKind | null {
  if (groups.some((group) => group.entities.some((id) => id.startsWith("pred-")))) {
    return "prediction";
  }
  if (groups.some((group) => group.entities.some((id) => id.startsWith("sim-")))) {
    return "optimization";
  }
  return null;
}

export function useFlightComparisonData(): FlightComparisonData {
  const { activeAirportCode, trajectoryComparison, trajectoryComparisonCategory } = useApp();
  const { categories } = useComparisonCategories(activeAirportCode);
  const drawableCategories = categories.filter(isDrawableComparisonCategory);
  const selectedCategory = drawableCategories.find(
    (category) => category.dir === trajectoryComparisonCategory,
  );
  // V and mass are useful in Baseline too, so keep reading the first published index when
  // comparison is off. Outcome styling remains gated by comparisonActive in FlightTable.
  const categoryDir = trajectoryComparison
    ? selectedCategory?.dir ?? null
    : drawableCategories[0]?.dir ?? null;
  const comparisonActive = trajectoryComparison && !!selectedCategory;

  const [byFlightKey, setByFlightKey] = useState<Map<string, FlightComparisonDatum>>(EMPTY);
  const [kind, setKind] = useState<ComparisonResultKind | null>(null);

  useEffect(() => {
    if (!activeAirportCode || !categoryDir) {
      setByFlightKey(EMPTY);
      setKind(null);
      return;
    }

    let cancelled = false;
    fetchJson<unknown>(airportComparisonIndexUrl(activeAirportCode, categoryDir))
      .then((data) => {
        if (cancelled) return;
        if (!isComparisonIndex(data)) {
          throw new Error(`${activeAirportCode}/${categoryDir} comparison index is invalid`);
        }
        const map = new Map<string, FlightComparisonDatum>();
        for (const group of data.groups) {
          map.set(group.group, {
            initialVMps: group.initialVMps ?? group.initialState?.V ?? null,
            massKg: group.massKg ?? group.initialState?.m ?? null,
            resultTimeS: group.status !== "failed" ? group.finalTimeS : null,
            status: group.status,
          });
        }
        setByFlightKey(map);
        setKind(resultKind(data.groups));
      })
      .catch((error) => {
        if (cancelled) return;
        if (!isMissingJsonAsset(error)) {
          console.warn("[useFlightComparisonData] failed to load comparison index", error);
        }
        setByFlightKey(EMPTY);
        setKind(null);
      });

    return () => {
      cancelled = true;
    };
  }, [activeAirportCode, categoryDir]);

  return useMemo(
    () => ({ byFlightKey, comparisonActive, resultKind: kind }),
    [byFlightKey, comparisonActive, kind],
  );
}
