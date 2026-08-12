/**
 * useObservedVerdictColors.ts
 * ---------------------------
 * Repaints the plain observed ADS-B tracks by their terminal-event verdict:
 * green for pass, red for fail, grey when the available evidence cannot decide.
 *
 * WHY REPAINT INSTEAD OF BAKING THE COLOUR INTO THE CZML
 * A verdict is not a property of the recorded flight — it is the output of an
 * evaluation that depends on the serialized threshold event and assessment context,
 * all of which are versioned. Baking it would make every CZML stale on a policy change
 * and would couple track generation to the judging package it deliberately does not
 * import. The report is a separate, cheap fetch, so the colour is applied here.
 *
 * THE JOIN
 * Entity ids in the observed CZML are `flight_scenarios.identity.flight_key`, and the
 * report rows carry the same key. Nothing here falls back to the callsign: it is not
 * unique, and joining on it swaps verdicts between namesakes. A row that fails to
 * match is painted neutral grey: it has no defensible pass/fail verdict, but must not
 * leak the CZML's five-colour identity palette into the three-state evaluation view.
 * `matched` / `total` are still returned so missing evaluation coverage remains visible.
 */

import { useEffect, useState } from "react";
import * as Cesium from "cesium";

import { useApp } from "../context/AppContext";
import {
  airportEvaluationReportUrl,
  OBSERVED_CATEGORY_KEY,
  OBSERVED_EVALUATION_REPORT_FILE,
} from "../data/airportData";
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
  /** Verdict counts over every painted track, or null when inactive. */
  counts: Record<ObservedVerdict, number> | null;
  /** How many entities found a published verdict, and how many exist. */
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
  active: boolean,
): ObservedVerdictState {
  const { viewer, trajectoryDataSource, activeAirportCode } = useApp();
  const [state, setState] = useState<ObservedVerdictState>(EMPTY);

  useEffect(() => {
    if (!active || !activeAirportCode || !trajectoryDataSource) {
      setState(EMPTY);
      return;
    }
    if (!isCesiumViewerUsable(viewer)) return;

    let cancelled = false;
    setState({ ...EMPTY, loading: true });

    void (async () => {
      let report: unknown;
      try {
        report = await fetchJson(
          airportEvaluationReportUrl(
            activeAirportCode,
            OBSERVED_CATEGORY_KEY,
            OBSERVED_EVALUATION_REPORT_FILE,
          ),
        );
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
      let matched = 0;
      let total = 0;
      for (const entity of trajectoryDataSource.entities.values) {
        if (entity.id === "document") continue;
        total += 1;
        const publishedVerdict = byKey.get(entity.id);
        if (publishedVerdict) matched += 1;
        const verdict = publishedVerdict ?? "undecided";
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
        matched,
        total,
        loading: false,
        missing: false,
      });
    })();

    return () => {
      cancelled = true;
    };
  }, [viewer, trajectoryDataSource, activeAirportCode, active]);

  return state;
}
