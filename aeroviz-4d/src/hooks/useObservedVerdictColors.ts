/**
 * useObservedVerdictColors.ts
 * ---------------------------
 * Repaints the plain observed ADS-B tracks by their FAA 8260.58D gate verdict:
 * green inside both gates, red outside one, grey when the data cannot decide.
 *
 * WHY REPAINT INSTEAD OF BAKING THE COLOUR INTO THE CZML
 * A verdict is not a property of the recorded flight — it is the output of an
 * evaluation that depends on the gates, the fit window and the established criteria,
 * all of which are tuned. Baking it would make every CZML stale on a threshold change
 * and would couple track generation to the judging package it deliberately does not
 * import. The report is a separate, cheap fetch, so the colour is applied here.
 *
 * THE JOIN
 * Entity ids in the observed CZML are `flight_scenarios.identity.flight_key`, and the
 * report rows carry the same key. Nothing here falls back to the callsign: it is not
 * unique, and joining on it swaps verdicts between namesakes. A row that fails to
 * match simply keeps its CZML colour, and `matched` / `total` are returned so a
 * wholesale mismatch (e.g. tracks regenerated from a different harvest than the
 * report) is visible in the UI instead of silently painting everything grey.
 */

import { useEffect, useState } from "react";
import * as Cesium from "cesium";

import { useApp } from "../context/AppContext";
import { airportEvaluationReportUrl } from "../data/airportData";
import { isEvaluationReport } from "../data/evaluationReport";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import {
  OBSERVED_VERDICT_COLORS,
  countVerdicts,
  verdictsByFlightKey,
  type ObservedVerdict,
} from "../utils/observedVerdictColors";

export interface ObservedVerdictState {
  /** Verdict counts over the tracks actually painted, or null when inactive. */
  counts: Record<ObservedVerdict, number> | null;
  /** How many painted entities found a verdict, and how many exist. */
  matched: number;
  total: number;
  /** True while the report is being fetched. */
  loading: boolean;
  /** Set when the report is absent — the tracks keep their CZML colours. */
  missing: boolean;
}

const EMPTY: ObservedVerdictState = {
  counts: null,
  matched: 0,
  total: 0,
  loading: false,
  missing: false,
};

export function useObservedVerdictColors(
  categoryDir: string | null,
  active: boolean,
): ObservedVerdictState {
  const { viewer, trajectoryDataSource, activeAirportCode } = useApp();
  const [state, setState] = useState<ObservedVerdictState>(EMPTY);

  useEffect(() => {
    if (!active || !activeAirportCode || !categoryDir || !trajectoryDataSource) {
      setState(EMPTY);
      return;
    }
    if (!isCesiumViewerUsable(viewer)) return;

    let cancelled = false;
    setState({ ...EMPTY, loading: true });

    void (async () => {
      let report: unknown;
      try {
        report = await fetchJson(airportEvaluationReportUrl(activeAirportCode, categoryDir));
      } catch (error) {
        if (cancelled) return;
        setState({ ...EMPTY, missing: isMissingJsonAsset(error) });
        return;
      }
      if (cancelled || !isEvaluationReport(report)) {
        if (!cancelled) setState({ ...EMPTY, missing: true });
        return;
      }

      const byKey = verdictsByFlightKey(report.trajectories);
      const painted: ObservedVerdict[] = [];
      let total = 0;
      for (const entity of trajectoryDataSource.entities.values) {
        if (entity.id === "document") continue;
        total += 1;
        const verdict = byKey.get(entity.id);
        if (!verdict) continue;
        painted.push(verdict);
        const color = Cesium.Color.fromCssColorString(OBSERVED_VERDICT_COLORS[verdict]);
        if (entity.path?.material instanceof Cesium.ColorMaterialProperty) {
          entity.path.material.color = new Cesium.ConstantProperty(color);
        } else if (entity.path) {
          entity.path.material = new Cesium.ColorMaterialProperty(color);
        }
        if (entity.polyline) {
          entity.polyline.material = new Cesium.ColorMaterialProperty(color);
        }
      }
      if (cancelled) return;
      setState({
        counts: countVerdicts(painted),
        matched: painted.length,
        total,
        loading: false,
        missing: false,
      });
    })();

    return () => {
      cancelled = true;
    };
  }, [viewer, trajectoryDataSource, activeAirportCode, categoryDir, active]);

  return state;
}
