/**
 * useComparisonTrajectoryLayer.ts
 * -------------------------------
 * Loads prediction-comparison trajectories when the Trajectories layer is in
 * "comparison" mode. Indexes resolve references from the already-loaded canonical
 * observed datasource; category files contain only optimizer/simulator/prediction paths.
 * It is index-driven so it never loads every result CZML:
 *
 *   1. fetch the selected category's `comparison_index.json` (one record per flight group);
 *   2. filter to the selected runway (null = all) and randomly sample N groups;
 *   3. load ONLY the CZML files those sampled groups live in;
 *   4. reveal sampled result entities and matching canonical observed flights.
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
import {
  TRAJECTORY_PATH_WIDTH,
  COMPARISON_KIND_COLORS,
  COMPARISON_KIND_ALPHA,
} from "../utils/trajectoryRenderModel";
import { makeStableVelocityOrientation } from "../utils/velocityOrientation";
import { addDataSourceHidden } from "../utils/cesiumDataSource";

/** The legend colour for a kind, as a Cesium.Color (single source of truth, see render-model). */
function comparisonKindColor(kind: ComparisonKind): Cesium.Color {
  return Cesium.Color.fromCssColorString(COMPARISON_KIND_COLORS[kind])
    .withAlpha(COMPARISON_KIND_ALPHA[kind]);
}

/**
 * The entity-id prefix a comparison kind is written with, by the CZML builder. This is the one
 * place the mapping lives: `kindOfEntityId` reads it to colour and gate an entity, and
 * `isComparisonEntity` reads it to decide what the hover/click handler may label — those two
 * drifted apart once already (the prefix list in the picker was missing `pred-`, so prediction
 * tracks silently could not be hovered for their callsign).
 *
 * `look-` is checked before `pred-` only for readability; the prefixes are disjoint.
 */
const COMPARISON_KIND_PREFIXES: ReadonlyArray<readonly [string, ComparisonKind]> = [
  ["opt-", "optimizer"],
  ["sim-", "simulator"],
  ["look-", "lookback"],
  ["pred-", "predicted"],
];

/** The entity id prefix encodes its kind: opt-/sim-/pred-/look-. */
export function kindOfEntityId(id: string): ComparisonKind {
  for (const [prefix, kind] of COMPARISON_KIND_PREFIXES) {
    if (id.startsWith(prefix)) return kind;
  }
  return "simulator";
}

/**
 * The kinds a ts_transformer (prediction-schema) states file produces. The builder never bakes
 * the off-target yellow onto these — a forecast essentially always misses the 106.75 m lateral
 * gate, so marking it would repaint the whole batch — which is why they are excluded from the
 * "keep the CZML's verdict colour" rule below and simply take their legend colour.
 */
const PREDICTION_SCHEMA_KINDS: ReadonlySet<ComparisonKind> = new Set(["predicted", "lookback"]);

/**
 * Make a comparison entity render like the observed tracks: uniform path width, and —
 * for the entities that are actually shown — an aircraft model (glTF animation off)
 * pointing down its path instead of a point marker. A CZML file packs many flight
 * groups but only the sampled ones are revealed, so models are attached ONLY to
 * `shownEntityIds`; the rest keep their (hidden) point marker and never allocate a
 * glТF model. Canonical observed references are styled by the visibility pass below;
 * opt-/sim- are state-sequence paths.
 */
export function applyComparisonRenderModel(
  entity: Cesium.Entity, shownEntityIds: Set<string>
): void {
  if (entity.id === "document") return;
  // The overlay aggregates up to N flights, so its per-entity name labels are hidden by default
  // (hundreds of them just clutter the globe); the hover/pick effect reveals only the one under
  // the cursor or clicked. The observed layer likewise labels only the selected flight.
  const kind = kindOfEntityId(entity.id);
  if (entity.path) {
    entity.path.width = new Cesium.ConstantProperty(TRAJECTORY_PATH_WIDTH);
    // Colour the path from the legend (single source of truth) so the rendered track always
    // matches its checkbox swatch — regardless of which colour this category's CZML baked in.
    // Every path of an off-target group keeps its CZML-baked bright yellow — the marking
    // must sit on the trajectory that missed the target, not just the canonical reference.
    //
    // Predictions are excluded from that second exception because the builder never bakes
    // the yellow for them: a forecast almost always misses the 106.75 m gate, so marking it
    // would repaint every prediction and say nothing. `status` stays "offTarget" (it is
    // true, and comparison_index.json reports it), but there is no verdict colour to
    // preserve — so predictions take the legend colour like any other kind, instead of
    // depending on the builder's PREDICTION_COLOR happening to equal the legend's.
    const status = entity.properties?.status?.getValue(Cesium.JulianDate.now());
    const keepsBakedVerdictColor =
      status === "offTarget" && !PREDICTION_SCHEMA_KINDS.has(kind);
    if (!keepsBakedVerdictColor) {
      const color = comparisonKindColor(kind);
      entity.path.material = new Cesium.ColorMaterialProperty(color);
      if (entity.label) entity.label.fillColor = new Cesium.ConstantProperty(color);
    }
  }
  if (entity.label) entity.label.show = new Cesium.ConstantProperty(false);
  // The lookback is a path-only annotation: it retraces the span the reference track ALREADY
  // covers, sample for sample, so giving it a model or a point marker draws a second aircraft
  // exactly on top of the reference's for the whole input window.
  if (kind === "lookback") {
    if (entity.point) entity.point.show = new Cesium.ConstantProperty(false);
    return;
  }
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
export function isComparisonEntity(entity: Cesium.Entity | undefined): entity is Cesium.Entity {
  const id = entity?.id;
  return typeof id === "string" && COMPARISON_KIND_PREFIXES.some(([prefix]) => id.startsWith(prefix));
}

export interface ObservedEntityStyleSnapshot {
  show: boolean;
  pathMaterial: Cesium.MaterialProperty | undefined;
  pathWidth: Cesium.Property | undefined;
  labelFillColor: Cesium.Property | undefined;
  labelShow: Cesium.Property | undefined;
  modelRunAnimations: Cesium.Property | undefined;
}

/** Capture the canonical entity's exact property objects before comparison owns its style. */
export function captureObservedEntityStyle(
  entity: Cesium.Entity,
): ObservedEntityStyleSnapshot {
  return {
    show: entity.show,
    pathMaterial: entity.path?.material,
    pathWidth: entity.path?.width,
    labelFillColor: entity.label?.fillColor,
    labelShow: entity.label?.show,
    modelRunAnimations: entity.model?.runAnimations,
  };
}

/** Return ownership to the observed layer without reconstructing any Cesium property. */
export function restoreObservedEntityStyle(
  entity: Cesium.Entity,
  snapshot: ObservedEntityStyleSnapshot,
): void {
  entity.show = snapshot.show;
  if (entity.path) {
    // Cesium's declaration marks the material setter non-optional even though its own
    // constructor and getter allow no material. Object.assign preserves that valid
    // pre-comparison `undefined` state without inventing a fallback style.
    Object.assign(entity.path, { material: snapshot.pathMaterial });
    entity.path.width = snapshot.pathWidth;
  }
  if (entity.label) {
    entity.label.fillColor = snapshot.labelFillColor;
    entity.label.show = snapshot.labelShow;
  }
  if (entity.model) entity.model.runAnimations = snapshot.modelRunAnimations;
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
    trajectoryDataSource,
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
    !!trajectoryComparisonCategory &&
    !!trajectoryDataSource;

  const sourcesRef = useRef<Cesium.CzmlDataSource[]>([]);
  const shownRef = useRef<Set<string>>(new Set());
  const observedGroupsRef = useRef<Map<string, "solved" | "offTarget" | "failed">>(
    new Map(),
  );
  const canonicalStylesRef = useRef<Map<Cesium.Entity, ObservedEntityStyleSnapshot>>(
    new Map(),
  );
  const canonicalDataSourceShowRef = useRef<boolean | null>(null);
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
          console.warn(
            `[comparison] ${activeAirportCode}/${categoryDir} does not use ` +
              "comparison-v2-generation; rerun run_scenario_optimization.py " +
              `--airport ${activeAirportCode}.`,
          );
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
      observedGroupsRef.current = new Map(
        selection.groups.map((group) => [group.group, group.status]),
      );

      // 3. Load only the sampled groups' CZML files.
      let start: Cesium.JulianDate | null = null;
      let stop: Cesium.JulianDate | null = null;
      if (trajectoryDataSource?.clock) {
        start = trajectoryDataSource.clock.startTime.clone();
        stop = trajectoryDataSource.clock.stopTime.clone();
      }
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
      observedGroupsRef.current = new Map();
      for (const [entity, snapshot] of canonicalStylesRef.current) {
        restoreObservedEntityStyle(entity, snapshot);
      }
      canonicalStylesRef.current.clear();
      if (trajectoryDataSource && canonicalDataSourceShowRef.current !== null) {
        trajectoryDataSource.show = canonicalDataSourceShowRef.current;
      }
      canonicalDataSourceShowRef.current = null;
    };
  }, [viewer, active, activeAirportCode, trajectoryComparisonCategory, selectedRunway,
    trajectorySampleCount, layers.trajectories, trajectoryDataSource]);

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
    if (trajectoryDataSource) {
      if (canonicalDataSourceShowRef.current === null) {
        canonicalDataSourceShowRef.current = trajectoryDataSource.show;
      }
      trajectoryDataSource.show = true;
      for (const entity of trajectoryDataSource.entities.values) {
        if (entity.id === "document") continue;
        if (!canonicalStylesRef.current.has(entity)) {
          canonicalStylesRef.current.set(entity, captureObservedEntityStyle(entity));
        }
        const status = observedGroupsRef.current.get(entity.id);
        entity.show = !!status && trajectoryComparisonKinds.reference;
        if (!status) continue;
        const rgba =
          status === "failed"
            ? [200, 60, 60, 200]
            : status === "offTarget"
              ? [150, 118, 25, 200]
              : [235, 235, 235, 200];
        const color = new Cesium.Color(
          rgba[0] / 255,
          rgba[1] / 255,
          rgba[2] / 255,
          rgba[3] / 255,
        );
        if (entity.path) {
          entity.path.width = new Cesium.ConstantProperty(TRAJECTORY_PATH_WIDTH);
          entity.path.material = new Cesium.ColorMaterialProperty(color);
        }
        if (entity.label) {
          entity.label.fillColor = new Cesium.ConstantProperty(color);
          entity.label.show = new Cesium.ConstantProperty(false);
        }
        if (entity.model) entity.model.runAnimations = new Cesium.ConstantProperty(false);
      }
    }
  }, [loadVersion, trajectoryComparisonKinds, trajectoryDataSource]);

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
      return isComparisonEntity(entity) ||
        (entity instanceof Cesium.Entity && observedGroupsRef.current.has(entity.id))
        ? entity
        : null;
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
