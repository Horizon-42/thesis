/**
 * useTrainingLiveLayer.ts
 * -----------------------
 * THE EXECUTOR, LIVE (`trainingAutopilot`, ready) in the 3D scene (called from `useTrainingTrackLayer`): the clicked word's
 * segment as the backend flew it — an aircraft flies it out from where the word was said (`autopilotPlaybackSpeedup` ×
 * real time — at least 8×, faster for a segment that would take more than 20 s — from the moment the answer arrived), in
 * blue, or a loud red when the flight was flown on to an outcome other than a landing (`autopilotColour`), its line
 * growing behind it and labelled with the speed-up, its ground speed, height and bank. Once flown out the line, the
 * aircraft and its label are made static (a CallbackProperty is re-evaluated, and a dynamic polyline rebuilt, every
 * frame; and only a static line draws its dashed depth-fail material), and its ground trace is draped dashed.
 *
 * It never moves the camera. The aircraft is read on each frame (the viewer renders continuously), not time-sampled:
 * `viewer.clock` belongs to Evaluation.
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import {
  autopilotAircraftLabel,
  autopilotColour,
  autopilotFlownAt,
  autopilotHasLine,
  autopilotOnScreen,
  autopilotPlaybackS,
  autopilotPlaybackSpeedup,
  type TrainingAutopilotView,
} from "../data/trainingAutopilot";
import { poseAt } from "../data/trainingAttitude";
import {
  aircraftGraphics,
  airLine,
  colour,
  entityGroup,
  groundLine,
  lonLatHeights,
  planDegrees,
  poseOrientation,
  TRAINING_ENTITY,
} from "../scene/trainingEntities";

type Ready = Extract<TrainingAutopilotView, { status: "ready" }>;

/** The live aircraft model's least size on screen (px). */
const AUTOPILOT_AIRCRAFT_PX = 56;

/** The live segment flown out from ``playedAt``; when it is done — now, if it already is — made static and its ground trace
 *  draped. */
function flyOut(viewer: Cesium.Viewer, view: Ready) {
  const { segment, playedAt } = view;
  const { track } = segment;
  const group = entityGroup(viewer);
  // blue, or red when the flight was flown on to an outcome that is not a landing: the line, its ground trace, the aircraft
  // and its label
  const hue = autopilotColour(segment);
  const positions = Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(track));
  const last = positions.length - 1;
  const flownS = track.tS[last] - track.tS[0];
  const speedup = autopilotPlaybackSpeedup(flownS);
  const now = () => autopilotFlownAt(track, autopilotPlaybackS(track, playedAt, Date.now()));
  const aircraftAt = ({ index, fraction }: { index: number; fraction: number }) => (index === last
    ? positions[last] : Cesium.Cartesian3.lerp(positions[index], positions[index + 1], fraction, new Cesium.Cartesian3()));
  const grown = new Cesium.CallbackProperty(() => {
    const at = now();
    if (at.index >= last) return positions;
    return [...positions.slice(0, at.index + 1), aircraftAt(at)];
  }, false);
  const line = group.add(airLine(TRAINING_ENTITY.autopilotTrack, "The autopilot's flown segment", grown, hue, 4));
  group.add({
    id: TRAINING_ENTITY.autopilotStart,
    name: "Where the autopilot took over: the closed-loop flight, flown again from the first predicted step, at the word",
    position: positions[0],
    point: { pixelSize: 8, color: colour(hue), outlineColor: Cesium.Color.WHITE, outlineWidth: 1.5,
      disableDepthTestDistance: Number.POSITIVE_INFINITY },
  });
  // the aircraft model in the executor's attitude where it is (`poseAt` at the same instant)
  const orientationAt = (at: { index: number; fraction: number }) => poseOrientation(aircraftAt(at),
    poseAt(track, at.index === last ? track.tS[last] : track.tS[at.index] + at.fraction * (track.tS[at.index + 1] - track.tS[at.index]))!);
  const aircraft = group.add({
    id: TRAINING_ENTITY.autopilotAircraft, name: "The autopilot's aircraft",
    position: new Cesium.CallbackPositionProperty(() => aircraftAt(now()), false),
    orientation: new Cesium.CallbackProperty(() => orientationAt(now()), false),
    model: aircraftGraphics({ css: hue, blend: 0.5, alpha: 1, minimumPixelSize: AUTOPILOT_AIRCRAFT_PX, ringCss: "#ffffff", ringPx: 1.5 }),
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
    aircraft.orientation = new Cesium.ConstantProperty(orientationAt({ index: last, fraction: 0 }));
    aircraft.label!.text = new Cesium.ConstantProperty(autopilotAircraftLabel(track, last, speedup));
    group.add(groundLine(TRAINING_ENTITY.autopilotGround, "The autopilot's ground trace", planDegrees(track), 2,
      new Cesium.PolylineDashMaterialProperty({ color: colour(hue, 0.6) })));
  };
  const remaining = playedAt + (flownS / speedup) * 1000 - Date.now();
  const timer = remaining > 0 ? window.setTimeout(land, remaining) : null;
  if (timer === null) land();
  return () => {
    if (timer !== null) window.clearTimeout(timer);
    group.remove();
  };
}

export default function useTrainingLiveLayer(): void {
  const { viewer, mode, trainingSelection, trainingAutopilot, trainingIntervalS } = useApp();
  const selection = mode === "training" ? trainingSelection : null;
  // the live answer of the reading on screen
  const live = autopilotOnScreen(trainingAutopilot, selection, trainingIntervalS);
  // a segment of one state (a dynamics failure in its first cycle) has no line to fly out: the bar's status says so
  const ready = live?.status === "ready" && autopilotHasLine(live.segment) ? live : null;

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || ready === null) return;
    return flyOut(viewer, ready);
  }, [viewer, ready]);
}
