/**
 * useTrainingWindowLayer.ts
 * -------------------------
 * The 3D scene of a window set beyond stage A's layers (`useTrainingTrackLayer` draws the commanded aircraft's flown path,
 * DA point and words as it does for stage A, since a round's sentence is read as a closed-loop one):
 *
 *  • THE OTHER AIRCRAFT (`trafficTracks`, `traffic` switches): each one's track over the window, thin, in its role's colour
 *    (on its record; window A's inserted one; window D's moved one), and where each is at the cursor's time — a point with
 *    its key, placed between the record's 2 s rows the cursor falls between (the only number computed here: a straight
 *    line between two of the exporter's points). An aircraft not in the air at the cursor is not drawn. The traffic is on
 *    the window's clock; the cursor is on the commanded aircraft's (`onAircraftClock`).
 *  • THE LOSSES OF SEPARATION (`loss` switch) of the round on screen that the commanded aircraft on screen is in: it and
 *    the other aircraft at the loss's time, joined by a red line, with the minimum and the distance the judge found (the
 *    exporter's numbers).
 *  • THE AIRCRAFT'S OTHER ROUNDS (`otherRounds` switch): their flown paths, thin and faded, in their end's colour.
 *  • WINDOW B'S MOVED START: its observed rows to the first predicted step as the start moved them, beside the recorded
 *    flight's observed track (stage A's layer draws that one; the flown path starts from the moved start).
 *
 * Static entities, never time-sampled (`viewer.clock` belongs to Evaluation); the points follow the Training cursor. Called
 * from a leaf (`TrainingWindowScene`).
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp, useTrainingCursor } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { useTrainingWindowLayers } from "../data/trainingWindowLayers";
import {
  lossesOf,
  onAircraftClock,
  onWindowClock,
  otherOf,
  trainingWindowOriginOf,
  windowShiftText,
  type TrainingWindowOrigin,
  type TrainingWindowRole,
} from "../data/trainingWindowSample";
import { TRAINING_FAILURE_COLOR, trainingOutcomeColour } from "../utils/trainingWordColors";
import { TRAINING_OUTCOME_TAG } from "../data/trainingText";
import { airLine, entityGroup, lonLatHeights, marker } from "../scene/trainingEntities";

/** Each role's colour: an aircraft on its record, window A's inserted one, window D's moved one. */
const TRAINING_WINDOW_ROLE_COLOR_MOVED = "#fbbf24";
export const TRAINING_WINDOW_ROLE_COLOR: Record<TrainingWindowRole, string> = {
  recorded: "#94a3b8",
  inserted: "#f472b6",
  moved: TRAINING_WINDOW_ROLE_COLOR_MOVED,
};

const ENTITY = {
  track: (key: string) => `training-window-track-${key}`,
  at: (key: string) => `training-window-at-${key}`,
  loss: (k: number) => `training-window-loss-${k}`,
  lossOwn: (k: number) => `training-window-loss-own-${k}`,
  lossOther: (k: number) => `training-window-loss-other-${k}`,
  round: (round: string) => `training-window-round-${round}`,
  movedStart: "training-window-moved-start",
};
/** Window B's moved start: the colour of the moved aircraft of window D (a start moved, not the record). */
const MOVED_START_COLOR = TRAINING_WINDOW_ROLE_COLOR_MOVED;

/** The keys of the other aircraft of the losses the commanded aircraft on screen is in, in its round. */
function lostWith(origin: TrainingWindowOrigin): Set<string> {
  const id = origin.aircraft.datasetId;
  return new Set(lossesOf(origin.end, id).map((loss) => otherOf(loss, id)));
}

/** Where a track is at time ``t`` of its own clock (on the straight line between the two rows around it); null outside it. */
export function positionAt(track: { tS: number[]; lon: number[]; lat: number[]; altitudeHaeM: number[] }, t: number): [number, number, number] | null {
  const times = track.tS;
  if (times.length === 0 || t < times[0] || t > times[times.length - 1]) return null;
  const hi = times.findIndex((value) => value >= t);
  if (times[hi] === t) return [track.lon[hi], track.lat[hi], track.altitudeHaeM[hi]];
  const lo = hi - 1;
  const f = (t - times[lo]) / (times[hi] - times[lo]);
  const mix = (values: number[]) => values[lo] + f * (values[hi] - values[lo]);
  return [mix(track.lon), mix(track.lat), mix(track.altitudeHaeM)];
}

function drawTrafficTracks(viewer: Cesium.Viewer, origin: TrainingWindowOrigin) {
  const group = entityGroup(viewer);
  const lost = lostWith(origin);
  for (const aircraft of origin.window.traffic) {
    const hue = lost.has(aircraft.key) ? TRAINING_FAILURE_COLOR : TRAINING_WINDOW_ROLE_COLOR[aircraft.role];
    group.add(airLine(ENTITY.track(aircraft.key), `${aircraft.key} (${aircraft.role}${aircraft.shiftS === null ? "" : `, shifted ${windowShiftText(aircraft.shiftS)}`}) · runway ${aircraft.runway}`,
      Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(aircraft)), hue, 1.5, 0.6));
  }
  return group;
}

/** ``timeS``: the cursor, on the commanded aircraft's flight clock. */
function drawTrafficAt(viewer: Cesium.Viewer, origin: TrainingWindowOrigin, timeS: number) {
  const group = entityGroup(viewer);
  const lost = lostWith(origin);
  const windowS = onWindowClock(origin.aircraft, timeS);
  for (const aircraft of origin.window.traffic) {
    const at = positionAt(aircraft, windowS);
    if (at === null) continue;
    const hue = lost.has(aircraft.key) ? TRAINING_FAILURE_COLOR : TRAINING_WINDOW_ROLE_COLOR[aircraft.role];
    group.add(marker(ENTITY.at(aircraft.key), `${aircraft.key} at ${timeS.toFixed(0)} s`, Cesium.Cartesian3.fromDegrees(...at), hue, 9,
      aircraft.key.split(":").pop()));
  }
  return group;
}

function drawLosses(viewer: Cesium.Viewer, origin: TrainingWindowOrigin) {
  const group = entityGroup(viewer);
  const id = origin.aircraft.datasetId;
  lossesOf(origin.end, id).forEach((loss, k) => {
    const key = otherOf(loss, id);
    const atS = onAircraftClock(origin.aircraft, loss.timeS);
    const own = positionAt(origin.sentence.flown, atS);
    const other = origin.window.traffic.find((aircraft) => aircraft.key === key);
    const there = other === undefined ? null : positionAt(other, loss.timeS);
    if (own === null || there === null) return;
    const ownAt = Cesium.Cartesian3.fromDegrees(...own);
    const otherAt = Cesium.Cartesian3.fromDegrees(...there);
    const text = `lost separation (${loss.kind}) · ${loss.distanceM.toFixed(0)} m of ${loss.requiredM.toFixed(0)} m required`;
    group.add(airLine(ENTITY.loss(k), text, [ownAt, otherAt], TRAINING_FAILURE_COLOR, 3));
    group.add(marker(ENTITY.lossOwn(k), `the commanded aircraft at the loss (${atS.toFixed(0)} s)`, ownAt, TRAINING_FAILURE_COLOR, 10, text));
    group.add(marker(ENTITY.lossOther(k), `${key} at the loss`, otherAt, TRAINING_FAILURE_COLOR, 10));
  });
  return group;
}

/** Window B's observed rows to its first predicted step as its start moved them, beside the recorded flight's observed
 *  track (which stage A's layer draws). */
function drawMovedStart(viewer: Cesium.Viewer, origin: TrainingWindowOrigin) {
  const group = entityGroup(viewer);
  const start = origin.aircraft.movedStart;
  if (start === null) return group;
  const move = origin.aircraft.startMove;
  group.add(airLine(ENTITY.movedStart, `The start moved (window B): turned ${move.turnDeg.toFixed(1)}°, ${move.heightM.toFixed(0)} m, speed × ${move.speedScale.toFixed(3)}`,
    Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(start)), MOVED_START_COLOR, 3));
  return group;
}

function drawOtherRounds(viewer: Cesium.Viewer, origin: TrainingWindowOrigin) {
  const group = entityGroup(viewer);
  for (const sentence of origin.aircraft.rounds) {
    if (sentence.round === origin.round) continue;
    group.add(airLine(ENTITY.round(String(sentence.round)), `Round ${sentence.round}: ${TRAINING_OUTCOME_TAG[sentence.outcome]}`,
      Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(sentence.flown)), trainingOutcomeColour(sentence.outcome), 1.5, 0.5));
  }
  return group;
}

export default function useTrainingWindowLayer(): void {
  const { viewer, mode, trainingSelection } = useApp();
  const layers = useTrainingWindowLayers();
  const flight = mode === "training" && trainingSelection !== null ? trainingSelection.flight : null;
  const origin = flight === null ? undefined : trainingWindowOriginOf(flight);
  const { trainingCursorS } = useTrainingCursor();

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || origin === undefined || !layers.trafficTracks) return;
    const group = drawTrafficTracks(viewer, origin);
    return () => group.remove();
  }, [viewer, origin, layers.trafficTracks]);

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || origin === undefined || !layers.traffic) return;
    const group = drawTrafficAt(viewer, origin, trainingCursorS);
    return () => group.remove();
  }, [viewer, origin, trainingCursorS, layers.traffic]);

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || origin === undefined || !layers.loss) return;
    const group = drawLosses(viewer, origin);
    return () => group.remove();
  }, [viewer, origin, layers.loss]);

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || origin === undefined) return;
    const group = drawMovedStart(viewer, origin);
    return () => group.remove();
  }, [viewer, origin]);

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || origin === undefined || !layers.otherRounds) return;
    const group = drawOtherRounds(viewer, origin);
    return () => group.remove();
  }, [viewer, origin, layers.otherRounds]);
}
