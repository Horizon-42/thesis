/**
 * useComparisonCategories.ts
 * --------------------------
 * Fetches the per-airport evaluation-category manifest
 * (public/data/airports/<ICAO>/comparison/categories.json) listing which observed,
 * optimization and data-driven comparison categories exist.
 * Airports without comparison data simply 404, reported as status "empty" (no error).
 */

import { useEffect, useState } from "react";
import {
  airportComparisonCategoriesUrl,
  isComparisonCategoriesManifest,
  type ComparisonCategory,
} from "../data/airportData";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";

export type ComparisonCategoriesStatus = "idle" | "loading" | "ready" | "empty" | "error";

export interface ComparisonCategoriesState {
  categories: ComparisonCategory[];
  status: ComparisonCategoriesStatus;
}

export function useComparisonCategories(airportCode: string): ComparisonCategoriesState {
  const [state, setState] = useState<ComparisonCategoriesState>({ categories: [], status: "idle" });

  useEffect(() => {
    if (!airportCode) {
      setState({ categories: [], status: "idle" });
      return;
    }

    let cancelled = false;
    setState({ categories: [], status: "loading" });

    fetchJson<unknown>(airportComparisonCategoriesUrl(airportCode))
      .then((data) => {
        if (cancelled) return;
        if (!isComparisonCategoriesManifest(data)) {
          throw new Error(`comparison categories for ${airportCode} is not a valid manifest`);
        }
        setState({ categories: data.categories, status: "ready" });
      })
      .catch((error) => {
        if (cancelled) return;
        if (isMissingJsonAsset(error)) {
          setState({ categories: [], status: "empty" });
          return;
        }
        console.error("[useComparisonCategories] Failed to load categories manifest:", error);
        setState({ categories: [], status: "error" });
      });

    return () => {
      cancelled = true;
    };
  }, [airportCode]);

  return state;
}
