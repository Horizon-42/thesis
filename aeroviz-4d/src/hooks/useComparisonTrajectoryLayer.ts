/**
 * Comparison trajectory layer.
 *
 * The comparison index is the sole sampling authority. Once it selects a roster,
 * this hook asks the backend for those exact observed flight keys and loads only the
 * result CZML files needed by the same groups. References and results therefore cannot
 * drift apart through independent sampling.
 *
 * Its source is a published category (Evaluate) or a traffic job's files (Optimize, `comparisonSource`). An index of
 * traffic windows (M1, AV46) is drawn ONE window at a time — the selected flight's group, the first by default — because
 * every window has its own clock: drawn together, one window's neighbours fly beside another window's controlled
 * aircraft. The other windows stay in the flight list.
 */

import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import { useApp, type ComparisonKind } from "../context/AppContext";
import { isComparisonIndex, type ComparisonGroup } from "../data/airportData";
import {
  MAX_FLIGHT_KEYS_PER_REQUEST,
  isObservedTrajectoryResponse,
  observedReferenceTracksUrl,
  type ObservedTrajectoryResponse,
} from "../data/observedTracks";
import { AEROVIZ_BACKEND_URL } from "../pilot/pilotClient";
import { addDataSourceHidden } from "../utils/cesiumDataSource";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import {
  comparisonCzmlUrl,
  comparisonIndexUrl,
  comparisonSourceKey,
  comparisonSourceOf,
  isWindowIndex,
  requestedWindow,
} from "../utils/comparisonSource";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import {
  summarizeObservedCzml,
  type ObservedFlightSummary,
} from "../utils/observedFlightSummary";
import { OBSERVED_VERDICT_COLORS } from "../utils/observedVerdictColors";
import { PREDICTION_OTHER_RUNWAY_COLOR } from "../utils/trajectoryRenderModel";
import { selectComparisonGroups } from "../utils/sampleTrajectories";
import {
  TRAFFIC_DOCUMENT_PACKET,
  sceneBackgroundKeys,
  sceneReferencePackets,
  scenePackets,
  trafficFlightKeys,
  trafficPackets,
  type TimedCzmlPacket,
} from "../utils/comparisonTraffic";
import {
  COMPARISON_KIND_ALPHA,
  COMPARISON_KIND_COLORS,
  COMPARISON_TRAFFIC_COLOR,
  DEFAULT_MODEL_BUDGET,
  TRAJECTORY_PATH_WIDTH,
  planTrajectoryModels,
} from "../utils/trajectoryRenderModel";
import { czmlPathSamples } from "../utils/czmlPathSamples";
import { frameTrajectoryCamera } from "../utils/frameTrajectoryCamera";
import { entryUtcOf } from "../utils/sceneTime";
import { makeStableVelocityOrientation } from "../utils/velocityOrientation";

type ComparisonStatus = ComparisonGroup["status"];

/** How much wider than the controlled aircraft's paths the camera frames a job's scene (the left dock covers part of it). */
const JOB_FRAME_MARGIN = 1.4;

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

/**
 * The terminal-verdict colour for a prediction group, at the caller's alpha.
 *
 * Both halves of a forecast — the predictor input window (`look-`) and the forecast itself
 * (`pred-`) — take this colour, so the pair reads as one track whose hue is its verdict. The
 * alpha is what separates them, which is why it is a parameter rather than a constant.
 */
function predictionOutcomeColor(
  status: ComparisonStatus | undefined,
  alpha: number,
): Cesium.Color | null {
  if (status === "solved") {
    return cssColor(OBSERVED_VERDICT_COLORS.pass, alpha);
  }
  if (status === "otherRunway") {
    return cssColor(PREDICTION_OTHER_RUNWAY_COLOR, alpha);
  }
  if (status === "offTarget" || status === "failed") {
    return cssColor(OBSERVED_VERDICT_COLORS.fail, alpha);
  }
  if (status === "indeterminate") {
    return cssColor(OBSERVED_VERDICT_COLORS.undecided, alpha);
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
  return raw === "solved" || raw === "offTarget" || raw === "indeterminate" || raw === "failed" ||
    raw === "otherRunway"
    ? raw
    : undefined;
}

/**
 * Apply the result render policy. Prediction paths carry their terminal outcome:
 * pass is green, fail is red, and indeterminate is gray. Predictor input takes the SAME
 * outcome colour as the forecast it feeds, only faded (`COMPARISON_KIND_ALPHA.lookback`):
 * an input window in an unrelated hue reads as a third kind of result rather than as the
 * first half of the track it belongs to. Groups with no terminal verdict fall back to the
 * shared purple, still faded on the input half.
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
    const predictionColor =
      kind === "predicted" || kind === "lookback"
        ? predictionOutcomeColor(status, COMPARISON_KIND_ALPHA[kind])
        : null;
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

/**
 * Apply the recorded-traffic contract: the recorded aircraft around a commanded flight, in ONE colour
 * (`COMPARISON_TRAFFIC_COLOR`) on the track and, mixed in, on the aircraft model. Only the `modelIds` carry
 * a model (the references' budget, `planTrajectoryModels`): the rest stay tracks.
 */
export function applyComparisonTrafficRenderModel(
  entity: Cesium.Entity,
  modelIds: ReadonlySet<string>,
): void {
  const color = cssColor(COMPARISON_TRAFFIC_COLOR, COMPARISON_KIND_ALPHA.reference);
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
    entity.model.color = new Cesium.ConstantProperty(color.withAlpha(1));
    entity.model.colorBlendMode = new Cesium.ConstantProperty(Cesium.ColorBlendMode.MIX);
    entity.model.colorBlendAmount = new Cesium.ConstantProperty(0.5);
    entity.model.runAnimations = new Cesium.ConstantProperty(false);
  }
}

/**
 * The backend's arrival window for these flight keys. A comparison group is drawn from records anchored
 * at terminal-ring entry, and a full-track answer to the same request has a different time origin that
 * nothing downstream can tell — it just renders the group kilometres ahead of its own reference — so
 * anything but the arrival window is refused here rather than drawn as a silently wrong overlay.
 */
async function fetchArrivalWindow(
  airport: string,
  flightKeys: string[],
): Promise<ObservedTrajectoryResponse> {
  const url = observedReferenceTracksUrl({ backendUrl: AEROVIZ_BACKEND_URL, airport, flightKeys });
  const response = await fetchJson<unknown>(url);
  if (!isObservedTrajectoryResponse(response)) {
    throw new Error(`${url} is not an observed-trajectories-v2 response`);
  }
  if (response.trackWindow !== "arrival") {
    throw new Error(
      `${url} returned the "${response.trackWindow}" track window; ` +
        "the comparison reference requires the arrival window",
    );
  }
  return response;
}

/** The arrival-window packet of every recorded aircraft the shown groups list, by flight key. */
async function fetchTrafficPackets(
  airport: string,
  flightKeys: string[],
): Promise<Map<string, TimedCzmlPacket>> {
  const packets = new Map<string, TimedCzmlPacket>();
  for (let start = 0; start < flightKeys.length; start += MAX_FLIGHT_KEYS_PER_REQUEST) {
    const response = await fetchArrivalWindow(
      airport,
      flightKeys.slice(start, start + MAX_FLIGHT_KEYS_PER_REQUEST),
    );
    for (const packet of response.czml as TimedCzmlPacket[]) {
      if (packet.id !== "document") packets.set(packet.id, packet);
    }
  }
  return packets;
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

type ClockBounds = { start: Cesium.JulianDate | null; stop: Cesium.JulianDate | null };

function includeSpan(start: Cesium.JulianDate, stop: Cesium.JulianDate, bounds: ClockBounds): void {
  if (!bounds.start || Cesium.JulianDate.lessThan(start, bounds.start)) {
    bounds.start = start.clone();
  }
  if (!bounds.stop || Cesium.JulianDate.greaterThan(stop, bounds.stop)) {
    bounds.stop = stop.clone();
  }
}

function includeClock(
  clock: Cesium.DataSourceClock | undefined,
  bounds: ClockBounds,
): void {
  if (clock) includeSpan(clock.startTime, clock.stopTime, bounds);
}

/**
 * A recorded neighbour's availability cut to the viewer's clock, on purpose: the clock is the groups' own span
 * (the neighbours never move it), so a neighbour airborne at the clock start shows from there, mid-flight, and
 * one that enters after the clock stop — or left before its start — is not shown (an empty collection).
 */
export function clipAvailabilityToClock(
  availability: Cesium.TimeIntervalCollection,
  clockStart: Cesium.JulianDate,
  clockStop: Cesium.JulianDate,
): Cesium.TimeIntervalCollection {
  return availability.intersect(
    new Cesium.TimeIntervalCollection([new Cesium.TimeInterval({ start: clockStart, stop: clockStop })]),
  );
}

/**
 * The recorded neighbours as one data source, cut to the clock `start .. stop` (not yet on the viewer): a neighbour
 * with nothing left inside it is not loaded at all, and the model budget is planned over those that are. Null when
 * none is left.
 */
async function buildNeighbourSource(
  name: string,
  neighbours: TimedCzmlPacket[],
  clockStart: Cesium.JulianDate,
  clockStop: Cesium.JulianDate,
): Promise<Cesium.CzmlDataSource | null> {
  const availability = availabilityByEntityId(neighbours);
  const clipped = new Map<string, Cesium.TimeIntervalCollection>();
  for (const neighbour of neighbours) {
    const interval = clipAvailabilityToClock(availability.get(neighbour.id)!, clockStart, clockStop);
    if (!interval.isEmpty) clipped.set(neighbour.id, interval);
  }
  const shown = neighbours.filter((neighbour) => clipped.has(neighbour.id));
  if (shown.length === 0) return null;
  const source = await new Cesium.CzmlDataSource(name).load([TRAFFIC_DOCUMENT_PACKET, ...shown]);
  const modelIds = planTrajectoryModels(
    shown.map((neighbour) => neighbour.id),
    null,
    DEFAULT_MODEL_BUDGET,
  ).modelIds;
  for (const entity of source.entities.values) {
    entity.availability = clipped.get(entity.id)!;
    applyComparisonTrafficRenderModel(entity, modelIds);
  }
  return source;
}

/** The span of one window on its own clock: its reference and its result paths (the neighbours never move it). */
export function windowSpan(
  key: string,
  referenceAvailability: ReadonlyMap<string, Cesium.TimeIntervalCollection>,
  resultAvailability: ReadonlyMap<string, Cesium.TimeIntervalCollection>,
): { start: Cesium.JulianDate; stop: Cesium.JulianDate } | null {
  const bounds: ClockBounds = { start: null, stop: null };
  for (const interval of [
    referenceAvailability.get(key),
    resultAvailability.get(`opt-${key}`),
    resultAvailability.get(`sim-${key}`),
  ]) {
    if (interval) includeSpan(interval.start, interval.stop, bounds);
  }
  return bounds.start && bounds.stop ? { start: bounds.start, stop: bounds.stop } : null;
}

/** Put the viewer's clock on `start .. stop`, at `start`, playing. */
function setViewerClock(viewer: Cesium.Viewer, start: Cesium.JulianDate, stop: Cesium.JulianDate): void {
  if (!Cesium.JulianDate.lessThan(start, stop)) return;
  viewer.clock.startTime = start;
  viewer.clock.stopTime = stop;
  viewer.clock.currentTime = start.clone();
  viewer.clock.multiplier = 60;
  viewer.clock.shouldAnimate = true;
  viewer.timeline?.zoomTo(start, stop);
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
    selectedFlightId,
    trafficScene,
    trajectorySampleCount,
    setSelectedFlightId,
    setTrajectoryDataSource,
    setSceneTime,
  } = useApp();
  const source = comparisonSourceOf({
    mode,
    trajectoryComparison,
    trajectoryComparisonCategory,
    activeAirportCode,
    trafficScene,
  });
  const sourceKey = source === null ? null : comparisonSourceKey(source);
  const enabled = !!viewer && source !== null;
  // A traffic job's scene is what the Optimize task is for: it needs no Trajectories switch (an Evaluate option).
  const visible = enabled && (source?.kind === "job" || layers.trajectories);

  const resultSourcesRef = useRef<Cesium.CzmlDataSource[]>([]);
  const referenceSourceRef = useRef<Cesium.CzmlDataSource | null>(null);
  const trafficSourceRef = useRef<Cesium.CzmlDataSource | null>(null);
  const trafficIdsRef = useRef<Set<string>>(new Set());
  const shownResultIdsRef = useRef<Set<string>>(new Set());
  const referenceIdsRef = useRef<Set<string>>(new Set());
  // The traffic window drawn now (an index of windows, else null), and how to draw another one.
  const shownWindowRef = useRef<string | null>(null);
  const showWindowRef = useRef<((key: string) => Promise<string | null>) | null>(null);
  const [windowGroupKeys, setWindowGroupKeys] = useState<string[]>([]);
  const [loadVersion, setLoadVersion] = useState(0);
  const [state, setState] = useState<ComparisonTrajectoryLayerState>(emptyState);
  // Only a selected group changes the window drawn: a cleared selection (Reset view) leaves it where it is.
  const windowKey = requestedWindow(windowGroupKeys, selectedFlightId ?? null);

  // The runway selector and the sample count choose WHICH groups of a published index are drawn, so a change reloads —
  // except for a scene (an M2 run) and a traffic job's files, which draw every group whatever they say (the top bar's
  // runway and the sample count are Evaluate's): there they change nothing, and a reload would refetch everything and reset
  // the clock. The load effect therefore does not list them; this one reloads it.
  const selectionIgnoredRef = useRef(false);
  const selectionRef = useRef({ selectedRunway, trajectorySampleCount });
  const [selectionVersion, setSelectionVersion] = useState(0);
  useEffect(() => {
    const previous = selectionRef.current;
    if (previous.selectedRunway === selectedRunway && previous.trajectorySampleCount === trajectorySampleCount) return;
    selectionRef.current = { selectedRunway, trajectorySampleCount };
    if (!selectionIgnoredRef.current) setSelectionVersion((version) => version + 1);
  }, [selectedRunway, trajectorySampleCount]);

  useEffect(() => {
    selectionIgnoredRef.current = false;
    showWindowRef.current = null;
    shownWindowRef.current = null;
    if (!viewer || !enabled || source === null || sourceKey === null) {
      setState(emptyState());
      setWindowGroupKeys([]);
      setSceneTime(null);
      return;
    }
    const loadSource = source;
    const isJob = loadSource.kind === "job";
    selectionIgnoredRef.current = isJob;
    const airportCode = loadSource.airportCode;
    let cancelled = false;
    const added: Cesium.CzmlDataSource[] = [];

    setState(emptyState());
    setWindowGroupKeys([]);
    setSceneTime(null);
    setTrajectoryDataSource(null);
    setSelectedFlightId(null);
    viewer.trackedEntity = undefined;

    (async () => {
      try {
        const rawIndex = await fetchJson<unknown>(comparisonIndexUrl(loadSource));
        if (!isComparisonIndex(rawIndex)) {
          throw new Error(`${airportCode}/${sourceKey} comparison index is invalid`);
        }
        if (cancelled) return;
        selectionIgnoredRef.current = isJob || rawIndex.scene !== undefined;

        // A job's files are one run, drawn whole: the top bar's runway and the sample count belong to Evaluate.
        const selection = selectComparisonGroups(
          rawIndex,
          isJob ? null : selectedRunway,
          isJob ? 0 : trajectorySampleCount,
        );
        if (selection.groups.length === 0) {
          setState({
            ...emptyState(),
            isLoaded: true,
            warning: "No comparison trajectories match the current runway.",
          });
          return;
        }

        const scene = rawIndex.scene;
        const oneWindow = isWindowIndex(rawIndex);
        const flightKeys = selection.groups.map((group) => group.group);
        const referenceResponse = await fetchArrivalWindow(airportCode, flightKeys);
        // A scene's references are served on their own entry clocks and drawn on the scene's.
        const referenceCzml = scene
          ? sceneReferencePackets(referenceResponse.czml, selection.groups)
          : referenceResponse.czml;
        const referenceSource = await new Cesium.CzmlDataSource(
          `comparison-reference-${sourceKey}`,
        ).load(referenceCzml);
        if (cancelled || !isCesiumViewerUsable(viewer)) return;

        const referenceIds = referenceSource.entities.values
          .filter((entity) => entity.id !== "document")
          .map((entity) => entity.id);
        // One window is drawn at a time, so every reference may carry a model: only the shown window's is instantiated.
        const modelIds = oneWindow
          ? new Set(referenceIds)
          : planTrajectoryModels(referenceIds, null, DEFAULT_MODEL_BUDGET).modelIds;
        const bounds: ClockBounds = { start: null, stop: null };
        const referenceAvailability = scene || oneWindow ? availabilityByEntityId(referenceCzml) : null;
        for (const entity of referenceSource.entities.values) {
          applyComparisonReferenceRenderModel(entity, modelIds);
          const span = scene ? referenceAvailability?.get(entity.id) : undefined;
          if (span) {
            // On a shared clock a reference that has landed must not hold at its runway for the rest of the scene,
            // and its span is part of the scene's clock.
            entity.availability = span;
            includeSpan(span.start, span.stop, bounds);
          }
        }
        addDataSourceHidden(viewer, referenceSource);
        added.push(referenceSource);

        // The backend's document clock is the unshifted references' (0 .. the longest); a scene's clock is the
        // groups' spans on the scene clock, taken above; a window's clock is its own span (`showWindow`).
        if (!scene && !oneWindow) includeClock(referenceSource.clock, bounds);
        const statusByGroup = new Map(
          selection.groups.map((group) => [group.group, group.status]),
        );
        const resultSources: Cesium.CzmlDataSource[] = [];
        const resultAvailability = new Map<string, Cesium.TimeIntervalCollection>();
        const resultCzml: unknown[] = [];
        const failedFiles: string[] = [];

        for (const file of selection.files) {
          try {
            const czml = await fetchJson<unknown>(comparisonCzmlUrl(loadSource, file));
            const availability = availabilityByEntityId(czml);
            const loaded = await new Cesium.CzmlDataSource(
              `comparison-${sourceKey}-${file}`,
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
            availability.forEach((interval, id) => resultAvailability.set(id, interval));
            resultCzml.push(...(czml as unknown[]));
            if (!oneWindow) includeClock(loaded.clock, bounds);
          } catch (error) {
            failedFiles.push(file);
            if (!isMissingJsonAsset(error)) {
              console.warn(`[comparison] failed to load ${file}`, error);
            }
          }
        }
        if (cancelled) return;

        // The recorded aircraft around the shown groups: the scene's background, loaded once, on the scene clock (M2) —
        // or, for an index of windows, those of the window drawn (`showWindow`, below). Anything else has none.
        let trafficSource: Cesium.CzmlDataSource | null = null;
        if (!oneWindow) {
          const trafficByKey = await fetchTrafficPackets(
            airportCode,
            scene ? sceneBackgroundKeys(scene) : trafficFlightKeys(selection.groups),
          );
          if (cancelled) return;
          const neighbours = scene
            ? scenePackets(scene, trafficByKey)
            : trafficPackets(selection.groups, trafficByKey);
          if (neighbours.length > 0) {
            if (!bounds.start || !bounds.stop) {
              throw new Error(
                "the comparison carries no clock to cut the recorded aircraft's availability to",
              );
            }
            const built = await buildNeighbourSource(
              `comparison-traffic-${sourceKey}`, neighbours, bounds.start, bounds.stop);
            if (cancelled || !isCesiumViewerUsable(viewer)) return;
            if (built) {
              addDataSourceHidden(viewer, built);
              added.push(built);
              trafficSource = built;
            }
          }
        }

        referenceSourceRef.current = referenceSource;
        trafficSourceRef.current = trafficSource;
        trafficIdsRef.current = new Set(trafficSource?.entities.values.map((entity) => entity.id));
        resultSourcesRef.current = resultSources;
        shownResultIdsRef.current = selection.shownEntityIds;
        referenceIdsRef.current = new Set(referenceIds);

        // what the files that did not load say; a window's own warnings come after it
        const loadWarning = failedFiles.length > 0
          ? `${failedFiles.length} comparison trajectory file(s) could not be loaded.`
          : null;
        let windowWarning: string | null = null;
        if (oneWindow) {
          // ONE window at a time: its group's paths, its reference, its own neighbours, on its own clock.
          const packetCache = new Map<string, TimedCzmlPacket>();
          let windowTraffic: Cesium.CzmlDataSource | null = null;
          let windowToken = 0;
          /**
           * Draw the window of `key`; resolves with why its real time is unknown (null: it is known, or superseded) — a
           * warning, not an error: the window is drawn all the same.
           */
          const showWindow = async (key: string): Promise<string | null> => {
            const token = ++windowToken;
            const group = selection.groups.find((candidate) => candidate.group === key)!;
            // a swap starts clean: the last window's error, and its time warning, are not this one's
            setState((current) => (current.error === null && current.warning === loadWarning
              ? current
              : { ...current, error: null, warning: loadWarning }));
            shownWindowRef.current = key;
            shownResultIdsRef.current = new Set(group.entities);
            if (windowTraffic) {
              viewer.dataSources.remove(windowTraffic, true);
              added.splice(added.indexOf(windowTraffic), 1);
              windowTraffic = null;
            }
            trafficSourceRef.current = null;
            trafficIdsRef.current = new Set();
            setSceneTime(null);
            const span = windowSpan(key, referenceAvailability!, resultAvailability);
            if (span) setViewerClock(viewer, span.start, span.stop);
            setLoadVersion((version) => version + 1);

            // the window's real start (the display epoch's UTC): the commanded flight's entry, from the backend's roster
            const timeKnown = entryUtcOf(airportCode, key).then(
              (entryUtc) => {
                if (!cancelled && token === windowToken) setSceneTime({ startUtc: entryUtc, epoch: rawIndex.epoch });
                return null;
              },
              (error: unknown) => (token === windowToken
                ? `Scene time unavailable: ${error instanceof Error ? error.message : String(error)}`
                : null),
            );

            const missing = trafficFlightKeys([group]).filter((id) => !packetCache.has(id));
            if (missing.length > 0) {
              const fetched = await fetchTrafficPackets(airportCode, missing);
              fetched.forEach((packet, id) => packetCache.set(id, packet));
            }
            if (cancelled || token !== windowToken) return null;
            const neighbours = trafficPackets([group], packetCache);
            if (neighbours.length > 0) {
              if (!span) {
                throw new Error(
                  "the comparison carries no clock to cut the recorded aircraft's availability to",
                );
              }
              const neighbourSource = await buildNeighbourSource(
                `comparison-traffic-${sourceKey}`, neighbours, span.start, span.stop);
              if (cancelled || token !== windowToken || !isCesiumViewerUsable(viewer)) return null;
              if (neighbourSource) {
                addDataSourceHidden(viewer, neighbourSource);
                windowTraffic = neighbourSource;
                added.push(neighbourSource);
                trafficSourceRef.current = neighbourSource;
                trafficIdsRef.current = new Set(neighbourSource.entities.values.map((entity) => entity.id));
                setLoadVersion((version) => version + 1);
              }
            }
            return timeKnown;
          };
          showWindowRef.current = showWindow;
          const first = flightKeys[0];
          windowWarning = await showWindow(first);
          if (cancelled) return;
          setWindowGroupKeys(flightKeys);
          setSelectedFlightId(first);
        } else if (scene) {
          setSceneTime({ startUtc: scene.startUtc, epoch: rawIndex.epoch });
        }
        setTrajectoryDataSource(referenceSource);
        setState({
          isLoaded: true,
          flightIds: referenceIds,
          flightSummaries: summarizeObservedCzml(referenceResponse.czml),
          warning: [loadWarning, windowWarning].filter((text) => text !== null).join(" ") || null,
          error: null,
        });
        setLoadVersion((version) => version + 1);

        if (!oneWindow && bounds.start && bounds.stop) setViewerClock(viewer, bounds.start, bounds.stop);
        // A job's scene was asked for: fly to it, the controlled aircraft in frame (the left dock covers part of the view).
        if (isJob) {
          frameTrajectoryCamera(
            viewer, czmlPathSamples(resultCzml, selection.groups.map((group) => `sim-${group.group}`)),
            { margin: JOB_FRAME_MARGIN });
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
      selectionIgnoredRef.current = false;
      showWindowRef.current = null;
      shownWindowRef.current = null;
      setSceneTime(null);
      if (isCesiumViewerUsable(viewer)) {
        for (const source of added) viewer.dataSources.remove(source, true);
        viewer.trackedEntity = undefined;
      }
      resultSourcesRef.current = [];
      referenceSourceRef.current = null;
      trafficSourceRef.current = null;
      trafficIdsRef.current = new Set();
      shownResultIdsRef.current = new Set();
      referenceIdsRef.current = new Set();
      setTrajectoryDataSource(null);
    };
  }, [
    viewer,
    enabled,
    activeAirportCode,
    sourceKey,
    selectionVersion,
    setSelectedFlightId,
    setTrajectoryDataSource,
  ]);

  // Another flight selected (the Flights table, a result row): the window of its group is drawn instead.
  useEffect(() => {
    if (windowKey === null || windowKey === shownWindowRef.current) return;
    const show = showWindowRef.current;
    if (show === null) return;
    show(windowKey).then(
      (timeWarning) => {
        if (timeWarning !== null) {
          setState((current) => ({
            ...current, warning: [current.warning, timeWarning].filter((text) => text !== null).join(" ") }));
        }
      },
      (error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        setState((current) => ({ ...current, error: message }));
      },
    );
  }, [windowKey]);

  useEffect(() => {
    const referenceSource = referenceSourceRef.current;
    if (referenceSource) {
      referenceSource.show = visible && trajectoryComparisonKinds.reference;
      // an index of windows draws the reference of ITS window only
      const window = shownWindowRef.current;
      if (window !== null) {
        for (const entity of referenceSource.entities.values) {
          if (entity.id !== "document") entity.show = entity.id === window;
        }
      }
    }
    // The recorded neighbours are recorded tracks too: they follow the Reference switch.
    const trafficSource = trafficSourceRef.current;
    if (trafficSource) trafficSource.show = visible && trajectoryComparisonKinds.reference;
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
        (entity instanceof Cesium.Entity &&
          (referenceIdsRef.current.has(entity.id) || trafficIdsRef.current.has(entity.id)))
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
