/**
 * useObservedTrajectoryLayer.ts
 * -----------------------------
 * Observed-layer hook: load one bounded backend response, paint terminal
 * verdicts, and synchronise the Cesium clock.
 *
 * What this hook does:
 *   1. Loads the bounded observed response into a CzmlDataSource.
 *   2. Reads the clock interval embedded in the CZML "document" packet.
 *   3. Writes those times into viewer.clock so the timeline bar shows the
 *      correct start/end, and animation begins from the start.
 *   4. Applies verdict colours and the aircraft-model rendering budget.
 *   5. Returns the selected roster and compact evaluation metadata.
 *
 * Key Cesium concepts:
 *   • CzmlDataSource  — loads and drives animated entities from CZML.
 *   • viewer.clock    — the master simulation clock (a Cesium.Clock instance).
 *   • JulianDate      — Cesium's internal time representation (Julian Day Number).
 *                       Always use Cesium.JulianDate methods; never raw Date math.
 *   • viewer.timeline — the UI bar at the bottom; call `.zoomTo()` to set range.
 *
 * 📖 Tutorial: see docs/04-czml-loader.md
 */

import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { planTrajectoryModels, TRAJECTORY_PATH_WIDTH } from "../utils/trajectoryRenderModel";
import { addDataSourceHidden } from "../utils/cesiumDataSource";
import { summarizeObservedCzml, type ObservedFlightSummary } from "../utils/observedFlightSummary";
import {
  OBSERVED_FITTED_TAIL_COLORS,
  OBSERVED_FITTED_TAIL_OUTLINE_COLORS,
  OBSERVED_VERDICT_COLORS,
} from "../utils/observedVerdictColors";
import {
  NO_OBSERVED_VERDICTS,
  isObservedTrajectoryResponse,
  decodeObservedVerdicts,
  type ObservedEvaluationSummary,
  type ObservedVerdict,
  type ObservedVerdicts,
} from "../data/observedTracks";

const LAYER_NAME = "observed-trajectories";

// ── Return type ───────────────────────────────────────────────────────────────
export interface ObservedTrajectoryLayerState {
  isLoaded: boolean;
  /** IDs of all aircraft entities found in the CZML (excludes "document") */
  flightIds: string[];
  /** Per-flight duration + initial ground speed, keyed by flight id (for the flight list). */
  flightSummaries: Record<string, ObservedFlightSummary>;
  /** Non-fatal data issue that should be shown to the user. */
  warning: string | null;
  error: string | null;
  /** Terminal verdicts and counts delivered with the bounded trajectory response. */
  observedVerdicts: ObservedVerdicts;
  /** Compact observed aggregate; the full report is loaded only when Details opens. */
  observedEvaluation: ObservedEvaluationSummary | null;
}

/**
 * Load a bounded observed-trajectory response and drive the Cesium clock from it.
 *
 * Loading is keyed on `responseUrl` only. The URL already encodes airport, runway,
 * verdict, and sample limit; `visible` can hide/reveal the loaded source without a reload.
 *
 * @param responseUrl - backend URL for one runway/verdict/sample selection ("" = none)
 * @param visible - whether the layer should be shown (still gated by the Trajectories toggle)
 */
export function useObservedTrajectoryLayer(
  responseUrl: string,
  visible: boolean = true,
): ObservedTrajectoryLayerState {
  const {
    viewer,
    layers,
    autoReplay,
    selectedFlightId,
    setSelectedFlightId,
    setTrajectoryDataSource,
  } = useApp();
  // Hold a direct reference — the CZML document packet can overwrite the
  // datasource name, making getByName() unreliable for visibility sync.
  const dsRef = useRef<Cesium.CzmlDataSource | null>(null);
  // The backend already returned the final sampled roster. Only aircraft-model selection
  // remains client-side because it is a rendering budget, not a data-selection rule.
  const modelIdsRef = useRef<Set<string>>(new Set());
  const [state, setState] = useState<ObservedTrajectoryLayerState>({
    isLoaded: false,
    flightIds: [],
    flightSummaries: {},
    warning: null,
    error: null,
    observedVerdicts: NO_OBSERVED_VERDICTS,
    observedEvaluation: null,
  });

  useEffect(() => {
    if (!viewer || !responseUrl) {
      setTrajectoryDataSource(null);
      setState(emptyState());
      return;
    }

    // We need to hold a reference to the DataSource so we can clean it up.
    let dataSource: Cesium.CzmlDataSource | undefined;
    let cancelled = false;

    setState(emptyState());
    setTrajectoryDataSource(null);
    modelIdsRef.current = new Set();
    viewer.trackedEntity = undefined;
    setSelectedFlightId(null);

    // ── Step 1: Fetch once, then hand the parsed packet array to Cesium.
    // This preserves the missing-asset guard without downloading the CZML twice, and lets us read
    // the flight-list facts (duration + initial ground speed) straight off the raw packets — the
    // observed CZML carries no `availability`, so Cesium entities have none to derive duration from.
    const ds = new Cesium.CzmlDataSource(LAYER_NAME);
    let flightSummaries: Record<string, ObservedFlightSummary> = {};
    let observedVerdicts = NO_OBSERVED_VERDICTS;
    let verdictsByFlightId: ReadonlyMap<string, ObservedVerdict> | null = null;
    let observedEvaluation: ObservedEvaluationSummary | null = null;
    fetchJson<unknown>(responseUrl)
      .then((response) => {
        if (!isObservedTrajectoryResponse(response)) {
          throw new Error(`${responseUrl} is not an observed-trajectories-v1 response`);
        }
        flightSummaries = summarizeObservedCzml(response.czml);
        const decodedVerdicts = decodeObservedVerdicts(response.verdicts);
        observedVerdicts = decodedVerdicts.summary;
        verdictsByFlightId = decodedVerdicts.byFlightId;
        observedEvaluation = response.evaluation;
        return ds.load(response.czml);
      })
      .then((loadedDs) => {
        if (cancelled) return;

        // ── Inside .then(ds => { ... }): ─────────────────────────────────────────

        const ids = loadedDs.entities.values
          .filter((e) => e.id !== "document")
          .map((e) => e.id);

        if (ids.length === 0) {
          const warning =
            `No trajectory entities were found in ${responseUrl}. ` +
            "The globe will stay open, but playback is disabled until CZML data is generated.";

          console.warn(`[useObservedTrajectoryLayer] ${warning}`);
          viewer.trackedEntity = undefined;
          setSelectedFlightId(null);
          setTrajectoryDataSource(null);
          setState({
            isLoaded: true,
            flightIds: [],
            flightSummaries: {},
            warning,
            error: null,
            observedVerdicts,
            observedEvaluation,
          });
          return;
        }

        modelIdsRef.current = planTrajectoryModels(ids, null).modelIds;
        if (verdictsByFlightId) {
          for (const entity of loadedDs.entities.values) {
            if (entity.id === "document") continue;
            paintObservedEntity(
              entity,
              verdictsByFlightId.get(entity.id) ?? "undecided",
            );
          }
        }

        dataSource = loadedDs;
        dsRef.current = loadedDs;
        // Add hidden: the render-model pass below configures every selected entity before the
        // visibility-sync effect reveals the source, preventing a one-frame unstyled flash.
        addDataSourceHidden(viewer, loadedDs);
        setTrajectoryDataSource(loadedDs);

        let warning: string | null = null;
        if (loadedDs.clock) {
          const startTime = loadedDs.clock.startTime.clone();
          const stopTime = loadedDs.clock.stopTime.clone();

          if (Cesium.JulianDate.lessThan(startTime, stopTime)) {
            viewer.clock.startTime = startTime;
            viewer.clock.stopTime = stopTime;
            viewer.clock.currentTime = startTime.clone();
            viewer.clock.clockRange = autoReplay
              ? Cesium.ClockRange.LOOP_STOP
              : Cesium.ClockRange.CLAMPED;
            viewer.clock.multiplier = 60;
            viewer.clock.shouldAnimate = true;
            viewer.timeline?.zoomTo(viewer.clock.startTime, viewer.clock.stopTime);
          } else {
            warning =
              `The CZML clock interval in ${responseUrl} has no duration. ` +
              "Trajectory entities were loaded, but playback timing was not changed.";
            console.warn(`[useObservedTrajectoryLayer] ${warning}`);
          }
        }

        // Keep camera fixed at the airport by default; tracking starts only
        // when the user clicks a flight row in FlightTable.
        viewer.trackedEntity = undefined;
        setSelectedFlightId(null);

        setState({
          isLoaded: true,
          flightIds: ids,
          flightSummaries,
          warning,
          error: null,
          observedVerdicts,
          observedEvaluation,
        });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (isMissingJsonAsset(err)) {
          const warning =
            `${responseUrl} was not found. ` +
            "The globe will stay open, but playback is disabled until CZML data is generated.";
          console.warn(`[useObservedTrajectoryLayer] ${warning}`);
          viewer.trackedEntity = undefined;
          setSelectedFlightId(null);
          setTrajectoryDataSource(null);
          setState({
            ...emptyState(),
            isLoaded: true,
            warning,
          });
          return;
        }

        const message = err instanceof Error ? err.message : String(err);
        setTrajectoryDataSource(null);
        setState({ ...emptyState(), error: message });
      });

    // ── Cleanup ───────────────────────────────────────────────────────────────
    return () => {
      cancelled = true;
      dsRef.current = null;
      setTrajectoryDataSource(null);
      if (dataSource && isCesiumViewerUsable(viewer)) {
        viewer.dataSources.remove(dataSource, true);
        viewer.trackedEntity = undefined;
      }
    };
  }, [viewer, responseUrl, setSelectedFlightId, setTrajectoryDataSource]);

  // ── Sync visibility (toggles the loaded layer's show WITHOUT reloading) ──────
  useEffect(() => {
    if (dsRef.current) dsRef.current.show = visible && layers.trajectories;
  }, [visible, layers.trajectories, state.isLoaded]);

  // ── Apply the render model to entities ───────────────────────────────────────
  // Every shown flight draws as a uniform-width path; only the subset (plus the
  // selected flight) carries an aircraft model; glTF animation is off; the label
  // shows for the selected flight alone (1000 labels is the dominant Cesium cost).
  // Re-runs on selection change so the tracked aircraft always shows its model+label.
  useEffect(() => {
    const ds = dsRef.current;
    if (!ds || !state.isLoaded || !visible) return;
    const selectedIsLoaded =
      selectedFlightId !== null && state.flightIds.includes(selectedFlightId);
    const selectedDisplaces =
      selectedIsLoaded && !modelIdsRef.current.has(selectedFlightId);
    const displacedModelId = selectedDisplaces
      ? modelIdsRef.current.values().next().value as string | undefined
      : undefined;
    for (const entity of ds.entities.values) {
      if (entity.id === "document") continue;
      entity.show = true;
      if (entity.path) entity.path.width = new Cesium.ConstantProperty(TRAJECTORY_PATH_WIDTH);
      if (entity.model) {
        const hasModel =
          entity.id === selectedFlightId ||
          (modelIdsRef.current.has(entity.id) && entity.id !== displacedModelId);
        entity.model.show = new Cesium.ConstantProperty(hasModel);
        entity.model.runAnimations = new Cesium.ConstantProperty(false);
      }
      if (entity.label) {
        entity.label.show = new Cesium.ConstantProperty(entity.id === selectedFlightId);
      }
    }
  }, [
    visible,
    state.isLoaded,
    state.flightIds,
    selectedFlightId,
  ]);
  return state;
}

function emptyState(): ObservedTrajectoryLayerState {
  return {
    isLoaded: false,
    flightIds: [],
    flightSummaries: {},
    warning: null,
    error: null,
    observedVerdicts: NO_OBSERVED_VERDICTS,
    observedEvaluation: null,
  };
}

function paintObservedEntity(entity: Cesium.Entity, verdict: ObservedVerdict): void {
  const color = Cesium.Color.fromCssColorString(OBSERVED_VERDICT_COLORS[verdict]);
  if (entity.path?.material instanceof Cesium.ColorMaterialProperty) {
    entity.path.material.color = new Cesium.ConstantProperty(color);
  } else if (entity.path) {
    entity.path.material = new Cesium.ColorMaterialProperty(color);
  }
  if (entity.polyline) {
    entity.polyline.material = new Cesium.ColorMaterialProperty(color);
  }
  if (entity.polylineVolume) {
    const fittedColor = Cesium.Color.fromCssColorString(OBSERVED_FITTED_TAIL_COLORS[verdict]);
    const fittedOutlineColor = Cesium.Color.fromCssColorString(
      OBSERVED_FITTED_TAIL_OUTLINE_COLORS[verdict],
    );
    entity.polylineVolume.material = new Cesium.ColorMaterialProperty(fittedColor);
    entity.polylineVolume.outlineColor = new Cesium.ConstantProperty(fittedOutlineColor);
  }
}
