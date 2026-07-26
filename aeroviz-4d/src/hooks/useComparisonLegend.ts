/**
 * Loads the selected evaluation category's index and derives its trajectory legend.
 */

import { useEffect, useMemo, useState } from "react";
import {
  airportComparisonIndexUrl,
  isComparisonIndex,
  type ComparisonIndex,
} from "../data/airportData";
import { buildComparisonLegend, type ComparisonLegendModel } from "../utils/comparisonLegend";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";

export type ComparisonLegendStatus = "idle" | "loading" | "ready" | "empty" | "error";

export interface ComparisonLegendState extends ComparisonLegendModel {
  status: ComparisonLegendStatus;
}

export function useComparisonLegend(
  airportCode: string,
  categoryDir: string | null,
  selectedRunway: string | null,
  active: boolean,
): ComparisonLegendState {
  const [index, setIndex] = useState<ComparisonIndex | null>(null);
  const [status, setStatus] = useState<ComparisonLegendStatus>("idle");

  useEffect(() => {
    if (!active || !airportCode || !categoryDir) {
      setIndex(null);
      setStatus("idle");
      return;
    }

    let cancelled = false;
    setIndex(null);
    setStatus("loading");

    fetchJson<unknown>(airportComparisonIndexUrl(airportCode, categoryDir))
      .then((data) => {
        if (cancelled) return;
        if (!isComparisonIndex(data)) {
          throw new Error(`${airportCode}/${categoryDir} comparison index is invalid`);
        }
        setIndex(data);
        setStatus(data.groups.length > 0 ? "ready" : "empty");
      })
      .catch((error) => {
        if (cancelled) return;
        if (!isMissingJsonAsset(error)) {
          console.error("[useComparisonLegend] Failed to load comparison index:", error);
          setStatus("error");
        } else {
          setStatus("empty");
        }
        setIndex(null);
      });

    return () => {
      cancelled = true;
    };
  }, [active, airportCode, categoryDir]);

  const model = useMemo(
    () => index ? buildComparisonLegend(index, selectedRunway) : { kinds: [], statuses: [] },
    [index, selectedRunway],
  );

  return { ...model, status };
}
