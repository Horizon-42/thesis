/**
 * Comparison trajectory layer.
 *
 * The comparison index is the sole sampling authority. Once it selects a roster,
 * this hook asks the backend for those exact observed flight keys and loads only the
 * result CZML files needed by the same groups. References and results therefore cannot
 * drift apart through independent sampling.
 */

import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import { useApp, type ComparisonKind } from "../context/AppContext";
import {
  airportComparisonCzmlUrl,
  airportComparisonIndexUrl,
  isComparisonIndex,
  type ComparisonGroup,
} from "../data/airportData";
import {
  isObservedTrajectoryResponse,
  observedReferenceTracksUrl,
} from "../data/observedTracks";
import { AEROVIZ_BACKEND_URL } from "../pilot/pilotClient";
import { addDataSourceHidden } from "../utils/cesiumDataSource";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import {
  summarizeObservedCzml,
  type ObservedFlightSummary,
} from "../utils/observedFlightSummary";
import { OBSERVED_VERDICT_COLORS } from "../utils/observedVerdictColors";
import { selectComparisonGroups } from "../utils/sampleTrajectories";
import {
  COMPARISON_KIND_ALPHA,
  COMPARISON_KIND_COLORS,
  DEFAULT_MODEL_BUDGET,
  TRAJECTORY_PATH_WIDTH,
  planTrajectoryModels,
} from "../utils/trajectoryRenderModel";
import { makeStableVelocityOrientation } from "../utils/velocityOrientation";

type ComparisonStatus = ComparisonGroup["status"];

const COMPARISON_KIND_PREFIXES: ReadonlyArray<readonly [string, ComparisonKind]> = [
  ["opt-", "optimizer"],
  ["sim-", "simulator"],
  ["look-", "lookback"],
  ["pred-", "predicted"],
];

export interface ComparisonTrajectoryLayerState {
  isLoaded: boolean;
  flightIds: string[];
  flightSummaries: Record<string, ObservedFlightSummary>;
  warning: string | null;
  error: string | null;
}

function emptyState(): ComparisonTrajectoryLayerState {
  return {
    isLoaded: false,
    flightIds: [],
    flightSummaries: {},
    warning: null,
    error: null,
  };
}

/** The entity id prefix encodes its result kind. */
export function kindOfEntityId(id: string): ComparisonKind {
  for (const [prefix, kind] of COMPARISON_KIND_PREFIXES) {
    if (id.startsWith(prefix)) return kind;
  }
  return "simulator";
}

function groupOfEntityId(id: string): string | null {
  for (const [prefix] of COMPARISON_KIND_PREFIXES) {
    if (id.startsWith(prefix)) return id.slice(prefix.length);
  }
  return null;
}

function cssColor(css: string, alpha: number): Cesium.Color {
  return Cesium.Color.fromCssColorString(css).withAlpha(alpha);
}

function comparisonKindColor(kind: ComparisonKind): Cesium.Color {
  return cssColor(COMPARISON_KIND_COLORS[kind], COMPARISON_KIND_ALPHA[kind]);
}

function predictionOutcomeColor(status: ComparisonStatus | undefined): Cesium.Color | null {
  if (status === "solved") {
    return cssColor(OBSERVED_VERDICT_COLORS.pass, COMPARISON_KIND_ALPHA.predicted);
  }
  if (status === "offTarget" || status === "failed") {
    return cssColor(OBSERVED_VERDICT_COLORS.fail, COMPARISON_KIND_ALPHA.predicted);
  }
  if (status === "indeterminate") {
    return cssColor(OBSERVED_VERDICT_COLORS.undecided, COMPARISON_KIND_ALPHA.predicted);
  }
  return null;
}

function entityStatus(
  entity: Cesium.Entity,
  statusByGroup?: ReadonlyMap<string, ComparisonStatus>,
): ComparisonStatus | undefined {
  const group = groupOfEntityId(entity.id);
  const indexed = group ? statusByGroup?.get(group) : undefined;
  if (indexed) return indexed;
  const raw = entity.properties?.status?.getValue(Cesium.JulianDate.now());
  return raw === "solved" || raw === "offTarget" || raw === "indeterminate" || raw === "failed"
    ? raw
    : undefined;
}

/**
 * Apply the result render policy. Prediction paths carry their terminal outcome:
 * pass is green, fail is red, and indeterminate is gray. Predictor input stays faded
 * purple because it is model input rather than an evaluated output.
 */
export function applyComparisonRenderModel(
  entity: Cesium.Entity,
  shownEntityIds: Set<string>,
  statusByGroup?: ReadonlyMap<string, ComparisonStatus>,
): void {
  if (entity.id === "document") return;
  const kind = kindOfEntityId(entity.id);
  const status = entityStatus(entity, statusByGroup);

  if (entity.path) {
    entity.path.width = new Cesium.ConstantProperty(TRAJECTORY_PATH_WIDTH);
    const predictionColor = kind === "predicted" ? predictionOutcomeColor(status) : null;
    // Optimizer replay CZML deliberately bakes yellow into an off-target result.
    // Every other path is painted from the frontend's single colour contract.
    const keepBakedOptimizerFailure =
      status === "offTarget" && (kind === "optimizer" || kind === "simulator");
    if (!keepBakedOptimizerFailure) {
      const color = predictionColor ?? comparisonKindColor(kind);
      entity.path.material = new Cesium.ColorMaterialProperty(color);
      if (entity.label) entity.label.fillColor = new Cesium.ConstantProperty(color);
    }
  }
  if (entity.label) entity.label.show = new Cesium.ConstantProperty(false);

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

/** Apply the one reference contract: exact observed trajectory, always white. */
export function applyComparisonReferenceRenderModel(
  entity: Cesium.Entity,
  modelIds: ReadonlySet<string>,
): void {
  if (entity.id === "document") return;
  const color = comparisonKindColor("reference");
  if (entity.path) {
    entity.path.width = new Cesium.ConstantProperty(TRAJECTORY_PATH_WIDTH);
    entity.path.material = new Cesium.ColorMaterialProperty(color);
  }
  if (entity.label) {
    entity.label.fillColor = new Cesium.ConstantProperty(color);
    entity.label.show = new Cesium.ConstantProperty(false);
  }
  if (entity.model) {
    entity.model.show = new Cesium.ConstantProperty(modelIds.has(entity.id));
    entity.model.runAnimations = new Cesium.ConstantProperty(false);
  }
}

/** Prefixes that mark a result entity (exact references use their bare flight key). */
export function isComparisonEntity(entity: Cesium.Entity | undefined): entity is Cesium.Entity {
  const id = entity?.id;
  return typeof id === "string" && COMPARISON_KIND_PREFIXES.some(([prefix]) => id.startsWith(prefix));
}

/**
 * Derive each result entity's availability from its first and last CZML sample.
 * Predictor input remains available through its matching forecast so its trail ages
 * out naturally after the shared anchor.
 */
export function availabilityByEntityId(czml: unknown): Map<string, Cesium.TimeIntervalCollection> {
  const out = new Map<string, Cesium.TimeIntervalCollection>();
  if (!Array.isArray(czml)) return out;
  const intervals = new Map<string, { start: Cesium.JulianDate; stop: Cesium.JulianDate }>();
  for (const raw of czml as unknown[]) {
    const packet = raw as {
      id?: unknown;
      position?: { epoch?: unknown; cartographicDegrees?: unknown };
    };
    const id = packet.id;
    const samples = packet.position?.cartographicDegrees;
    const epochIso = packet.position?.epoch;
    if (typeof id !== "string" || id === "document" || typeof epochIso !== "string") continue;
    if (!Array.isArray(samples) || samples.length < 4) continue;
    const epoch = Cesium.JulianDate.fromIso8601(epochIso);
    const values = samples as number[];
    intervals.set(id, {
      start: Cesium.JulianDate.addSeconds(epoch, values[0], new Cesium.JulianDate()),
      stop: Cesium.JulianDate.addSeconds(
        epoch,
        values[values.length - 4],
        new Cesium.JulianDate(),
      ),
    });
  }

  for (const [id, interval] of intervals) {
    let stop = interval.stop;
    if (id.startsWith("look-")) {
      const prediction = intervals.get(`pred-${id.slice("look-".length)}`);
      if (prediction && Cesium.JulianDate.lessThan(stop, prediction.stop)) {
        stop = prediction.stop;
      }
    }
    out.set(id, new Cesium.TimeIntervalCollection([
      new Cesium.TimeInterval({ start: interval.start, stop }),
    ]));
  }
  return out;
}

function includeClock(
  clock: Cesium.DataSourceClock | undefined,
  bounds: { start: Cesium.JulianDate | null; stop: Cesium.JulianDate | null },
): void {
  if (!clock) return;
  if (!bounds.start || Cesium.JulianDate.lessThan(clock.startTime, bounds.start)) {
    bounds.start = clock.startTime.clone();
  }
  if (!bounds.stop || Cesium.JulianDate.greaterThan(clock.stopTime, bounds.stop)) {
    bounds.stop = clock.stopTime.clone();
  }
}

export function useComparisonTrajectoryLayer(): ComparisonTrajectoryLayerState {
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
    setSelectedFlightId,
    setTrajectoryDataSource,
  } = useApp();
  const enabled =
    !!viewer &&
    mode === "observe" &&
    trajectoryComparison &&
    !!activeAirportCode &&
    !!trajectoryComparisonCategory;
  const visible = enabled && layers.trajectories;

  const resultSourcesRef = useRef<Cesium.CzmlDataSource[]>([]);
  const referenceSourceRef = useRef<Cesium.CzmlDataSource | null>(null);
  const shownResultIdsRef = useRef<Set<string>>(new Set());
  const referenceIdsRef = useRef<Set<string>>(new Set());
  const [loadVersion, setLoadVersion] = useState(0);
  const [state, setState] = useState<ComparisonTrajectoryLayerState>(emptyState);

  useEffect(() => {
    if (!viewer || !enabled || !trajectoryComparisonCategory) {
      setState(emptyState());
      return;
    }
    const categoryDir = trajectoryComparisonCategory;
    let cancelled = false;
    const added: Cesium.CzmlDataSource[] = [];

    setState(emptyState());
    setTrajectoryDataSource(null);
    setSelectedFlightId(null);
    viewer.trackedEntity = undefined;

    (async () => {
      try {
        const rawIndex = await fetchJson<unknown>(
          airportComparisonIndexUrl(activeAirportCode, categoryDir),
        );
        if (!isComparisonIndex(rawIndex)) {
          throw new Error(`${activeAirportCode}/${categoryDir} comparison index is invalid`);
        }
        if (cancelled) return;

        const selection = selectComparisonGroups(
          rawIndex,
          selectedRunway,
          trajectorySampleCount,
        );
        if (selection.groups.length === 0) {
          setState({
            ...emptyState(),
            isLoaded: true,
            warning: "No comparison trajectories match the current runway.",
          });
          return;
        }

        const flightKeys = selection.groups.map((group) => group.group);
        const referenceUrl = observedReferenceTracksUrl({
          backendUrl: AEROVIZ_BACKEND_URL,
          airport: activeAirportCode,
          flightKeys,
        });
        const referenceResponse = await fetchJson<unknown>(referenceUrl);
        if (!isObservedTrajectoryResponse(referenceResponse)) {
          throw new Error(`${referenceUrl} is not an observed-trajectories-v1 response`);
        }
        const referenceSource = await new Cesium.CzmlDataSource(
          `comparison-reference-${categoryDir}`,
        ).load(referenceResponse.czml);
        if (cancelled || !isCesiumViewerUsable(viewer)) return;

        const referenceIds = referenceSource.entities.values
          .filter((entity) => entity.id !== "document")
          .map((entity) => entity.id);
        const modelIds = planTrajectoryModels(
          referenceIds,
          null,
          DEFAULT_MODEL_BUDGET,
        ).modelIds;
        for (const entity of referenceSource.entities.values) {
          applyComparisonReferenceRenderModel(entity, modelIds);
        }
        addDataSourceHidden(viewer, referenceSource);
        added.push(referenceSource);

        const bounds = { start: null, stop: null } as {
          start: Cesium.JulianDate | null;
          stop: Cesium.JulianDate | null;
        };
        includeClock(referenceSource.clock, bounds);
        const statusByGroup = new Map(
          selection.groups.map((group) => [group.group, group.status]),
        );
        const resultSources: Cesium.CzmlDataSource[] = [];
        const failedFiles: string[] = [];

        for (const file of selection.files) {
          try {
            const czml = await fetchJson<unknown>(
              airportComparisonCzmlUrl(activeAirportCode, categoryDir, file),
            );
            const availability = availabilityByEntityId(czml);
            const loaded = await new Cesium.CzmlDataSource(
              `comparison-${categoryDir}-${file}`,
            ).load(czml);
            if (cancelled || !isCesiumViewerUsable(viewer)) return;
            for (const entity of loaded.entities.values) {
              const interval = availability.get(entity.id);
              if (interval) entity.availability = interval;
              applyComparisonRenderModel(
                entity,
                selection.shownEntityIds,
                statusByGroup,
              );
            }
            addDataSourceHidden(viewer, loaded);
            added.push(loaded);
            resultSources.push(loaded);
            includeClock(loaded.clock, bounds);
          } catch (error) {
            failedFiles.push(file);
            if (!isMissingJsonAsset(error)) {
              console.warn(`[comparison] failed to load ${file}`, error);
            }
          }
        }
        if (cancelled) return;

        referenceSourceRef.current = referenceSource;
        resultSourcesRef.current = resultSources;
        shownResultIdsRef.current = selection.shownEntityIds;
        referenceIdsRef.current = new Set(referenceIds);
        setTrajectoryDataSource(referenceSource);
        setState({
          isLoaded: true,
          flightIds: referenceIds,
          flightSummaries: summarizeObservedCzml(referenceResponse.czml),
          warning: failedFiles.length > 0
            ? `${failedFiles.length} comparison trajectory file(s) could not be loaded.`
            : null,
          error: null,
        });
        setLoadVersion((version) => version + 1);

        if (
          bounds.start &&
          bounds.stop &&
          Cesium.JulianDate.lessThan(bounds.start, bounds.stop)
        ) {
          viewer.clock.startTime = bounds.start;
          viewer.clock.stopTime = bounds.stop;
          viewer.clock.currentTime = bounds.start.clone();
          viewer.clock.multiplier = 60;
          viewer.clock.shouldAnimate = true;
          viewer.timeline?.zoomTo(bounds.start, bounds.stop);
        }
      } catch (error) {
        if (cancelled) return;
        const message = error instanceof Error ? error.message : String(error);
        setState({ ...emptyState(), isLoaded: true, error: message });
        setTrajectoryDataSource(null);
      }
    })();

    return () => {
      cancelled = true;
      if (isCesiumViewerUsable(viewer)) {
        for (const source of added) viewer.dataSources.remove(source, true);
        viewer.trackedEntity = undefined;
      }
      resultSourcesRef.current = [];
      referenceSourceRef.current = null;
      shownResultIdsRef.current = new Set();
      referenceIdsRef.current = new Set();
      setTrajectoryDataSource(null);
    };
  }, [
    viewer,
    enabled,
    activeAirportCode,
    trajectoryComparisonCategory,
    selectedRunway,
    trajectorySampleCount,
    setSelectedFlightId,
    setTrajectoryDataSource,
  ]);

  useEffect(() => {
    const referenceSource = referenceSourceRef.current;
    if (referenceSource) referenceSource.show = visible && trajectoryComparisonKinds.reference;
    for (const source of resultSourcesRef.current) {
      source.show = visible;
      for (const entity of source.entities.values) {
        if (entity.id === "document") continue;
        entity.show =
          shownResultIdsRef.current.has(entity.id) &&
          trajectoryComparisonKinds[kindOfEntityId(entity.id)];
      }
    }
  }, [visible, loadVersion, trajectoryComparisonKinds]);

  useEffect(() => {
    if (!viewer || !visible) return;
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
      for (const entity of labelled) if (!desired.has(entity)) setLabelShown(entity, false);
      for (const entity of desired) if (!labelled.has(entity)) setLabelShown(entity, true);
      labelled.clear();
      desired.forEach((entity) => labelled.add(entity));
    };
    const pickComparison = (position: Cesium.Cartesian2): Cesium.Entity | null => {
      const picked = viewer.scene.pick(position);
      const entity = picked && picked.id;
      return isComparisonEntity(entity) ||
        (entity instanceof Cesium.Entity && referenceIdsRef.current.has(entity.id))
        ? entity
        : null;
    };

    handler.setInputAction((movement: { endPosition: Cesium.Cartesian2 }) => {
      hovered = pickComparison(movement.endPosition);
      refresh();
    }, Cesium.ScreenSpaceEventType.MOUSE_MOVE);
    handler.setInputAction((movement: { position: Cesium.Cartesian2 }) => {
      const clicked = pickComparison(movement.position);
      pinned = clicked === pinned ? null : clicked;
      refresh();
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

    return () => {
      handler.destroy();
      for (const entity of labelled) setLabelShown(entity, false);
    };
  }, [viewer, visible, loadVersion]);

  return state;
}
