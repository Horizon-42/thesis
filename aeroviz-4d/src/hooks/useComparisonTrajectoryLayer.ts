/**
 * useComparisonTrajectoryLayer.ts
 * -------------------------------
 * Loads the optimizer-comparison trajectories (three coloured paths per flight:
 * reference / optimizer / simulator) when the Trajectories layer is in "comparison"
 * mode. It is index-driven so it never loads every (large) per-runway CZML:
 *
 *   1. fetch the selected category's `comparison_index.json` (one record per flight group);
 *   2. filter to the selected runway (null = all) and randomly sample N groups;
 *   3. load ONLY the CZML files those sampled groups live in;
 *   4. reveal the sampled groups' entities, gated per kind (the three checkboxes).
 *
 * Loading happens in one effect (keyed on airport/category/runway/sample); a separate
 * visibility effect re-applies per-entity `show` when the per-kind toggles change, so
 * flipping a kind checkbox is instant (no reload/refetch).
 */

import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import { useApp, type ComparisonKind } from "../context/AppContext";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import {
  airportComparisonCzmlUrl,
  airportComparisonIndexUrl,
  isComparisonIndex,
} from "../data/airportData";
import { selectComparisonGroups } from "../utils/sampleTrajectories";
import { TRAJECTORY_PATH_WIDTH } from "../utils/trajectoryRenderModel";
import { makeStableVelocityOrientation } from "../utils/velocityOrientation";

/** The entity id prefix encodes its kind: ref-/opt-/sim-. */
function kindOfEntityId(id: string): ComparisonKind {
  if (id.startsWith("ref-")) return "reference";
  if (id.startsWith("opt-")) return "optimizer";
  return "simulator";
}

/**
 * Make a comparison entity render like the observed tracks: uniform path width, and —
 * for the entities that are actually shown — an aircraft model (glTF animation off)
 * pointing down its path instead of a point marker. A CZML file packs many flight
 * groups but only the sampled ones are revealed, so models are attached ONLY to
 * `shownEntityIds`; the rest keep their (hidden) point marker and never allocate a
 * glТF model. The reference (ref-) entity copied from trajectories.czml already has a
 * model; opt-/sim- are state-sequence paths. Per-kind colour comes from the CZML path.
 */
function applyComparisonRenderModel(entity: Cesium.Entity, shownEntityIds: Set<string>): void {
  if (entity.id === "document") return;
  if (entity.path) entity.path.width = new Cesium.ConstantProperty(TRAJECTORY_PATH_WIDTH);
  if (!shownEntityIds.has(entity.id)) return;
  if (entity.model) {
    entity.model.runAnimations = new Cesium.ConstantProperty(false);
    return;
  }
  if (!entity.position) return;
  entity.model = new Cesium.ModelGraphics({
    uri: "/models/aircraft.glb",
    scale: 3,
    minimumPixelSize: 32,
    runAnimations: false,
  });
  entity.orientation = makeStableVelocityOrientation(entity.position);
  if (entity.point) entity.point.show = new Cesium.ConstantProperty(false);
}

export function useComparisonTrajectoryLayer(): void {
  const {
    viewer,
    layers,
    trajectoryComparison,
    trajectoryComparisonCategory,
    trajectoryComparisonKinds,
    activeAirportCode,
    selectedRunway,
    trajectorySampleCount,
  } = useApp();

  // Active when the Trajectories layer is on, comparison mode is on, and a category is chosen.
  const active =
    !!viewer &&
    trajectoryComparison &&
    layers.trajectories &&
    !!activeAirportCode &&
    !!trajectoryComparisonCategory;

  const sourcesRef = useRef<Cesium.CzmlDataSource[]>([]);
  const shownRef = useRef<Set<string>>(new Set());
  const [loadVersion, setLoadVersion] = useState(0);

  // ── Load: fetch index, sample groups, load only the needed CZML files ────────
  useEffect(() => {
    if (!viewer || !active || !trajectoryComparisonCategory) return;
    const categoryDir = trajectoryComparisonCategory;

    let cancelled = false;
    const added: Cesium.CzmlDataSource[] = [];

    (async () => {
      // 1. Index (one record per flight group) for the selected category.
      let index;
      try {
        const raw = await fetchJson<unknown>(airportComparisonIndexUrl(activeAirportCode, categoryDir));
        if (!isComparisonIndex(raw)) {
          console.warn(`[comparison] ${activeAirportCode}/${categoryDir} index is malformed.`);
          return;
        }
        index = raw;
      } catch (err) {
        if (!isMissingJsonAsset(err)) console.warn("[comparison] failed to load index", err);
        return; // missing data is expected until the comparison CZMLs are generated
      }
      if (cancelled) return;

      // 2. Filter to the selected runway, sample N groups, collect the files to load.
      const selection = selectComparisonGroups(index, selectedRunway, trajectorySampleCount);
      if (selection.files.length === 0) return;

      // 3. Load only the sampled groups' CZML files.
      let start: Cesium.JulianDate | null = null;
      let stop: Cesium.JulianDate | null = null;
      for (const file of selection.files) {
        let loaded: Cesium.CzmlDataSource;
        try {
          const czml = await fetchJson<unknown>(
            airportComparisonCzmlUrl(activeAirportCode, categoryDir, file),
          );
          loaded = await new Cesium.CzmlDataSource(`comparison-${categoryDir}-${file}`).load(czml);
        } catch (err) {
          if (!isMissingJsonAsset(err)) console.warn(`[comparison] failed to load ${file}`, err);
          continue;
        }
        if (cancelled || !isCesiumViewerUsable(viewer)) return;

        viewer.dataSources.add(loaded);
        added.push(loaded);
        for (const entity of loaded.entities.values) {
          applyComparisonRenderModel(entity, selection.shownEntityIds);
        }
        if (loaded.clock) {
          if (!start || Cesium.JulianDate.lessThan(loaded.clock.startTime, start)) {
            start = loaded.clock.startTime.clone();
          }
          if (!stop || Cesium.JulianDate.greaterThan(loaded.clock.stopTime, stop)) {
            stop = loaded.clock.stopTime.clone();
          }
        }
      }
      if (cancelled) return;

      // Publish to the refs and trigger the visibility pass below.
      sourcesRef.current = added;
      shownRef.current = selection.shownEntityIds;
      setLoadVersion((version) => version + 1);

      // Span the clock over every loaded comparison trajectory.
      if (start && stop && Cesium.JulianDate.lessThan(start, stop)) {
        viewer.clock.startTime = start;
        viewer.clock.stopTime = stop;
        viewer.clock.currentTime = start.clone();
        viewer.clock.multiplier = 60;
        viewer.clock.shouldAnimate = true;
        viewer.timeline?.zoomTo(start, stop);
      }
    })();

    return () => {
      cancelled = true;
      if (isCesiumViewerUsable(viewer)) {
        for (const ds of added) viewer.dataSources.remove(ds, true);
      }
      sourcesRef.current = [];
      shownRef.current = new Set();
    };
  }, [viewer, active, activeAirportCode, trajectoryComparisonCategory, selectedRunway,
    trajectorySampleCount, layers.trajectories]);

  // ── Visibility: reveal sampled entities, gated per kind (instant, no reload) ──
  useEffect(() => {
    for (const ds of sourcesRef.current) {
      ds.show = true;
      for (const entity of ds.entities.values) {
        if (entity.id === "document") continue;
        entity.show =
          shownRef.current.has(entity.id) && trajectoryComparisonKinds[kindOfEntityId(entity.id)];
      }
    }
  }, [loadVersion, trajectoryComparisonKinds]);
}
