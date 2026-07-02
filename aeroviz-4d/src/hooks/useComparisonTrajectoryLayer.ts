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
import { TRAJECTORY_PATH_WIDTH, COMPARISON_KIND_COLORS } from "../utils/trajectoryRenderModel";
import { makeStableVelocityOrientation } from "../utils/velocityOrientation";
import { addDataSourceHidden } from "../utils/cesiumDataSource";

/** Path/label alpha for the recoloured comparison tracks (matches the CZML's ~220/255). */
const COMPARISON_PATH_ALPHA = 0.86;

/** The legend colour for a kind, as a Cesium.Color (single source of truth, see render-model). */
function comparisonKindColor(kind: ComparisonKind): Cesium.Color {
  return Cesium.Color.fromCssColorString(COMPARISON_KIND_COLORS[kind]).withAlpha(COMPARISON_PATH_ALPHA);
}

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
  // The overlay aggregates up to N flights, so its per-entity name labels are hidden by default
  // (hundreds of them just clutter the globe); the hover/pick effect reveals only the one under
  // the cursor or clicked. The observed layer likewise labels only the selected flight.
  const kind = kindOfEntityId(entity.id);
  if (entity.path) {
    entity.path.width = new Cesium.ConstantProperty(TRAJECTORY_PATH_WIDTH);
    // Colour the path from the legend (single source of truth) so the rendered track always
    // matches its checkbox swatch — regardless of which colour this category's CZML baked in.
    // The reference keeps the CZML colour (white, or dark-red for failed optimizations).
    if (kind !== "reference") {
      const color = comparisonKindColor(kind);
      entity.path.material = new Cesium.ColorMaterialProperty(color);
      if (entity.label) entity.label.fillColor = new Cesium.ConstantProperty(color);
    }
  }
  if (entity.label) entity.label.show = new Cesium.ConstantProperty(false);
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

/** Prefixes that mark a comparison-overlay entity (vs. anything else the user might pick). */
function isComparisonEntity(entity: Cesium.Entity | undefined): entity is Cesium.Entity {
  const id = entity?.id;
  return typeof id === "string" && (id.startsWith("ref-") || id.startsWith("opt-") || id.startsWith("sim-"));
}

/**
 * An availability interval per entity, matching its own position samples, keyed by entity id.
 *
 * The comparison packets bake in NO `availability`, and the position uses HOLD extrapolation, so a
 * short optimizer/simulator trajectory would otherwise FREEZE at its end and hang in the scene
 * while the clock runs on to the far-longer reference tracks. Giving each entity an availability =
 * [firstSample, lastSample] makes it simply DISAPPEAR when its trajectory ends. Derived from the
 * CZML `position.epoch` + first/last time offset (seconds).
 */
export function availabilityByEntityId(czml: unknown): Map<string, Cesium.TimeIntervalCollection> {
  const out = new Map<string, Cesium.TimeIntervalCollection>();
  if (!Array.isArray(czml)) return out;
  for (const raw of czml as unknown[]) {
    const packet = raw as { id?: unknown; position?: { epoch?: unknown; cartographicDegrees?: unknown } };
    const id = packet.id;
    const cd = packet.position?.cartographicDegrees;
    const epochIso = packet.position?.epoch;
    if (typeof id !== "string" || id === "document" || typeof epochIso !== "string") continue;
    if (!Array.isArray(cd) || cd.length < 4) continue;
    const epoch = Cesium.JulianDate.fromIso8601(epochIso);
    const nums = cd as number[];
    out.set(id, new Cesium.TimeIntervalCollection([
      new Cesium.TimeInterval({
        start: Cesium.JulianDate.addSeconds(epoch, nums[0], new Cesium.JulianDate()),
        stop: Cesium.JulianDate.addSeconds(epoch, nums[nums.length - 4], new Cesium.JulianDate()),
      }),
    ]));
  }
  return out;
}

export function useComparisonTrajectoryLayer(): void {
  const {
    viewer,
    layers,
    mode,
    trajectoryComparison,
    trajectoryComparisonCategory,
    trajectoryComparisonKinds,
    activeAirportCode,
    selectedRunway,
    trajectorySampleCount,
  } = useApp();

  // The 3-colour comparison is an Observe-mode overlay (its toggle lives in the Observe dock),
  // so it is painted only in Observe — switching to Fly/Optimize/Compare tears it down, keeping
  // the tasks visually exclusive. It also needs the Trajectories layer on, comparison mode on,
  // and a category chosen.
  const active =
    !!viewer &&
    mode === "observe" &&
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
        let availability: Map<string, Cesium.TimeIntervalCollection>;
        try {
          const czml = await fetchJson<unknown>(
            airportComparisonCzmlUrl(activeAirportCode, categoryDir, file),
          );
          availability = availabilityByEntityId(czml);
          loaded = await new Cesium.CzmlDataSource(`comparison-${categoryDir}-${file}`).load(czml);
        } catch (err) {
          if (!isMissingJsonAsset(err)) console.warn(`[comparison] failed to load ${file}`, err);
          continue;
        }
        if (cancelled || !isCesiumViewerUsable(viewer)) return;

        // Add hidden — the visibility pass below reveals only the sampled subset. Adding it
        // visible flashes every entity of each file on load (see addDataSourceHidden).
        addDataSourceHidden(viewer, loaded);
        added.push(loaded);
        for (const entity of loaded.entities.values) {
          const iv = availability.get(entity.id);
          if (iv) entity.availability = iv;
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

  // ── Hover / click reveals a single label ─────────────────────────────────────
  // Labels are hidden by default (applyComparisonRenderModel). Show only the label of the
  // comparison entity under the cursor (hover) plus the last one clicked (pinned) — so a flight's
  // identity is reachable without hundreds of labels cluttering the aggregate view.
  useEffect(() => {
    if (!viewer || !active) return;
    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    const labelled = new Set<Cesium.Entity>();
    let hovered: Cesium.Entity | null = null;
    let pinned: Cesium.Entity | null = null;

    const setLabelShown = (entity: Cesium.Entity, show: boolean) => {
      if (entity.label) entity.label.show = new Cesium.ConstantProperty(show);
    };
    const refresh = () => {
      const desired = new Set<Cesium.Entity>();
      if (hovered) desired.add(hovered);
      if (pinned) desired.add(pinned);
      for (const e of labelled) if (!desired.has(e)) setLabelShown(e, false);
      for (const e of desired) if (!labelled.has(e)) setLabelShown(e, true);
      labelled.clear();
      desired.forEach((e) => labelled.add(e));
    };
    const pickComparison = (position: Cesium.Cartesian2): Cesium.Entity | null => {
      const picked = viewer.scene.pick(position);
      const entity = picked && picked.id;
      return isComparisonEntity(entity) ? entity : null;
    };

    handler.setInputAction((m: { endPosition: Cesium.Cartesian2 }) => {
      hovered = pickComparison(m.endPosition);
      refresh();
    }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
    handler.setInputAction((m: { position: Cesium.Cartesian2 }) => {
      const clicked = pickComparison(m.position);
      pinned = clicked === pinned ? null : clicked; // click again (or empty space) to unpin
      refresh();
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

    return () => {
      handler.destroy();
      for (const e of labelled) setLabelShown(e, false);
    };
  }, [viewer, active, loadVersion]);
}
