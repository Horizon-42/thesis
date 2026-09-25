/**
 * useTrainingExecutorLayers.ts
 * ----------------------------
 * What the executor flew over the selected Training flight, in the 3D scene (called from `useTrainingTrackLayer`):
 *
 *  • THE REPLAY (`trainingExecutor`, when the panel publishes it): its flown track at its ellipsoid height in teal, its
 *    ground trace dashed and where it ended, named with its outcome, and the flown rows outside the heading word it was
 *    told, red on its ground trace (its judge's own row verdicts).
 *  • THE EXECUTOR, LIVE (`trainingAutopilot`, ready): the picked word's segment as the backend flew it — an aircraft flies
 *    it out from where the word was said (`autopilotPlaybackSpeedup` × real time — at least 8×, faster for a segment
 *    that would take more than 20 s — from the moment the answer arrived or "Replay in 3D" was pressed), in blue, or a
 *    loud red when the word flew outside its envelope (`autopilotColour`), its line growing behind it and labelled with
 *    the speed-up, its ground speed, height and bank. Once flown out the line, the aircraft and its label are made
 *    static (a CallbackProperty is re-evaluated, and a dynamic polyline rebuilt, every frame; and only a static line
 *    draws its dashed depth-fail material), and its ground trace is draped dashed with — for a heading word — its judged
 *    rows outside the word red.
 *
 * The rows outside a heading word, the replay's and the live executor's, show with the heading bands' switch; neither
 * layer is rebuilt by a switch, and neither ever moves the camera. The aircraft is read on each frame (the viewer
 * renders continuously), not time-sampled: `viewer.clock` belongs to Observe.
 */

import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { TRAINING_EXECUTOR_COLOR, TRAINING_OUTSIDE_COLOR } from "../utils/trainingWordColors";
import { overlayOnScreen, type TrainingExecutorFlown } from "../data/trainingOverlays";
import {
  autopilotAircraftLabel,
  autopilotColour,
  autopilotFlownAt,
  autopilotJudgedPoints,
  autopilotOnScreen,
  autopilotPlaybackSpeedup,
  type TrainingAutopilotView,
} from "../data/trainingAutopilot";
import { TRAINING_OUTCOME_TAG } from "../data/trainingText";
import {
  airLine,
  colour,
  entityGroup,
  GROUND_ROWS_WIDTH,
  groundLine,
  lonLatHeights,
  marker,
  planDegrees,
  TRAINING_ENTITY,
  trainingBandOutsideGround,
} from "../scene/trainingEntities";

type Ready = Extract<TrainingAutopilotView, { status: "ready" }>;

/** Show or hide the entities of ``ids`` that are there. */
function showEntities(viewer: Cesium.Viewer, ids: string[], shown: boolean): void {
  for (const id of ids) {
    const entity = viewer.entities.getById(id);
    if (entity) entity.show = shown;
  }
}

/** The replay's flown track, its ground trace, where it ended and its rows outside a heading word (their ids returned,
 *  for the bands' switch). */
function buildReplay(viewer: Cesium.Viewer, flight: TrainingExecutorFlown) {
  const { track } = flight;
  const group = entityGroup(viewer);
  group.add(groundLine(TRAINING_ENTITY.executorGround, "The executor's ground trace", planDegrees(track), 2,
    new Cesium.PolylineDashMaterialProperty({ color: colour(TRAINING_EXECUTOR_COLOR, 0.6) })));
  group.add(airLine(TRAINING_ENTITY.executorTrack, "The executor's flown track",
    Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(track)), TRAINING_EXECUTOR_COLOR, 3));
  // its flown rows outside the heading word it was told: the judge's own row verdicts (the judged steps are the track's
  // first points)
  const judged = flight.judgedTrackDeg?.length ?? 0;
  const lon = track.lon.slice(0, judged);
  const lat = track.lat.slice(0, judged);
  const outside = flight.words
    .flatMap((word) => (word.heading === null ? [] : trainingBandOutsideGround(lon, lat, word.heading)))
    .map((degrees, run) => group.add(groundLine(TRAINING_ENTITY.executorOutside(run), "The executor off the heading word it was told",
      degrees, GROUND_ROWS_WIDTH, colour(TRAINING_OUTSIDE_COLOR))).id);
  const end = track.lon.length - 1;
  group.add(marker(TRAINING_ENTITY.executorEnd, "Where the executor's flight ended",
    Cesium.Cartesian3.fromDegrees(track.lon[end], track.lat[end], track.altitudeHaeM[end]), TRAINING_EXECUTOR_COLOR, 10,
    `executor: ${TRAINING_OUTCOME_TAG[flight.outcome]}`));
  return { outside, remove: group.remove };
}

/** The live segment flown out from ``playedAt``; when it is done — now, if it already is — made static, and its ground
 *  trace and rows outside the picked heading word draped (`bandsShown()` at that moment; their ids to ``onOutside``). */
function flyOut(viewer: Cesium.Viewer, view: Ready, bandsShown: () => boolean, onOutside: (ids: string[]) => void) {
  const { segment, playedAt } = view;
  const { track } = segment;
  const group = entityGroup(viewer);
  // blue, or red when the picked word flew outside its envelope: the line, its ground trace, the aircraft and its label
  const hue = autopilotColour(segment);
  const positions = Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(track));
  const last = positions.length - 1;
  const flownS = track.tS[last] - track.tS[0];
  const speedup = autopilotPlaybackSpeedup(flownS);
  const now = () => autopilotFlownAt(track, ((Date.now() - playedAt) / 1000) * speedup);
  const aircraftAt = ({ index, fraction }: { index: number; fraction: number }) => (index === last
    ? positions[last] : Cesium.Cartesian3.lerp(positions[index], positions[index + 1], fraction, new Cesium.Cartesian3()));
  const line = group.add(airLine(TRAINING_ENTITY.autopilotTrack, "The autopilot's flown segment",
    new Cesium.CallbackProperty(() => {
      const at = now();
      return at.index === last ? positions : [...positions.slice(0, at.index + 1), aircraftAt(at)];
    }, false), hue, 4));
  group.add({
    id: TRAINING_ENTITY.autopilotStart, name: "Where the autopilot took over: the observed state as the word was said",
    position: positions[0],
    point: { pixelSize: 8, color: colour(hue), outlineColor: Cesium.Color.WHITE, outlineWidth: 1.5,
      disableDepthTestDistance: Number.POSITIVE_INFINITY },
  });
  const aircraft = group.add({
    id: TRAINING_ENTITY.autopilotAircraft, name: "The autopilot's aircraft",
    position: new Cesium.CallbackPositionProperty(() => aircraftAt(now()), false),
    point: { pixelSize: 12, color: colour(hue), outlineColor: Cesium.Color.WHITE, outlineWidth: 2,
      disableDepthTestDistance: Number.POSITIVE_INFINITY },
    label: {
      text: new Cesium.CallbackProperty(() => autopilotAircraftLabel(track, now().index, speedup), false),
      font: "600 12px sans-serif",
      // white on the segment's colour: the simulated clock must read over any terrain
      fillColor: Cesium.Color.WHITE, showBackground: true, backgroundColor: colour(hue, 0.85), style: Cesium.LabelStyle.FILL,
      pixelOffset: new Cesium.Cartesian2(0, -20), disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
  });
  const land = () => {
    if (!isCesiumViewerUsable(viewer)) return;
    line.polyline!.positions = new Cesium.ConstantProperty(positions);
    aircraft.position = new Cesium.ConstantPositionProperty(positions[last]);
    aircraft.label!.text = new Cesium.ConstantProperty(autopilotAircraftLabel(track, last, speedup));
    group.add(groundLine(TRAINING_ENTITY.autopilotGround, "The autopilot's ground trace", planDegrees(track), 2,
      new Cesium.PolylineDashMaterialProperty({ color: colour(hue, 0.6) })));
    const band = segment.word.heading;
    if (band === null) return;
    // the band's rows are the flight's; the judged points begin at the segment's first step
    const { lon, lat } = autopilotJudgedPoints(segment);
    const shifted = { ...band, firstRow: band.firstRow - segment.segment.row, stopRow: band.stopRow - segment.segment.row };
    onOutside(trainingBandOutsideGround(lon, lat, shifted).map((degrees, run) => {
      const entity = group.add(groundLine(TRAINING_ENTITY.autopilotOutside(run), "The autopilot off the picked heading word",
        degrees, GROUND_ROWS_WIDTH, colour(TRAINING_OUTSIDE_COLOR)));
      entity.show = bandsShown();
      return entity.id;
    }));
  };
  const remaining = playedAt + (flownS / speedup) * 1000 - Date.now();
  const timer = remaining > 0 ? window.setTimeout(land, remaining) : null;
  if (timer === null) land();
  return () => {
    if (timer !== null) window.clearTimeout(timer);
    group.remove();
  };
}

export default function useTrainingExecutorLayers(): void {
  const { viewer, mode, trainingSelection, trainingLayers, trainingExecutor, trainingAutopilot } = useApp();
  const selection = mode === "training" ? trainingSelection : null;
  const replay = overlayOnScreen(trainingExecutor, selection)?.flight ?? null;
  const flown = replay?.flown ? replay : null;
  const live = autopilotOnScreen(trainingAutopilot, selection);
  // a segment of one state (a dynamics failure in its first cycle) has no line to fly out: the card and the status say so
  const ready = live?.status === "ready" && live.segment.track.tS.length >= 2 ? live : null;
  const { headingBands } = trainingLayers;
  // the rows outside a heading word, the replay's and the live executor's, shown with the bands' switch
  const bandsShown = useRef(headingBands);
  bandsShown.current = headingBands;
  const replayOutside = useRef<string[]>([]);
  const liveOutside = useRef<string[]>([]);

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || flown === null) return;
    const built = buildReplay(viewer, flown);
    replayOutside.current = built.outside;
    showEntities(viewer, built.outside, bandsShown.current);
    return () => {
      replayOutside.current = [];
      built.remove();
    };
  }, [viewer, flown]);

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || ready === null) return;
    const remove = flyOut(viewer, ready, () => bandsShown.current, (ids) => { liveOutside.current = ids; });
    return () => {
      liveOutside.current = [];
      remove();
    };
  }, [viewer, ready]);

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer)) return;
    showEntities(viewer, [...replayOutside.current, ...liveOutside.current], headingBands);
  }, [viewer, headingBands]);
}
