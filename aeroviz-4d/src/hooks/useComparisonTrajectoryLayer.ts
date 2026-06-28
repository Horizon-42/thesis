/**
 * useComparisonTrajectoryLayer.ts
 * -------------------------------
 * Loads the optimizer-comparison trajectories (three coloured paths per flight:
 * reference / optimizer / simulator) when the Trajectories layer is in "comparison"
 * mode. It is index-driven so it never loads every (large) per-runway CZML:
 *
 *   1. fetch `comparison/comparison_index.json` (one record per flight group);
 *   2. filter to the selected runway (null = all) and randomly sample N groups;
 *   3. load ONLY the CZML files those sampled groups live in;
 *   4. reveal only the sampled groups' entities (they ship `show:false`).
 *
 * Follows `useCzmlLoader`'s pattern: each load happens inside the effect, and the
 * effect's cleanup removes exactly the datasources that run added — so changing the
 * runway / sample count / airport tears down and reloads cleanly.
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import {
  airportComparisonCzmlUrl,
  airportComparisonIndexUrl,
  isComparisonIndex,
} from "../data/airportData";
import { selectComparisonGroups } from "../utils/sampleTrajectories";

export function useComparisonTrajectoryLayer(): void {
  const {
    viewer,
    layers,
    trajectoryComparison,
    trajectoryComparisonCategory,
    activeAirportCode,
    selectedRunway,
    trajectorySampleCount,
  } = useApp();

  // Only active when the Trajectories layer is on, comparison mode is on, AND a category
  // (which optimization to show) is selected.
  const active =
    !!viewer &&
    trajectoryComparison &&
    layers.trajectories &&
    !!activeAirportCode &&
    !!trajectoryComparisonCategory;

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
          console.warn(`[comparison] ${activeAirportCode} comparison_index.json is malformed.`);
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

      // 3. Load only the sampled groups' CZML files; reveal only their entities.
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
        loaded.show = layers.trajectories;
        for (const entity of loaded.entities.values) {
          if (entity.id !== "document") entity.show = selection.shownEntityIds.has(entity.id);
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

      // 4. Span the clock over every loaded comparison trajectory.
      if (!cancelled && start && stop && Cesium.JulianDate.lessThan(start, stop)) {
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
    };
  }, [viewer, active, activeAirportCode, trajectoryComparisonCategory, selectedRunway,
    trajectorySampleCount, layers.trajectories]);
}
