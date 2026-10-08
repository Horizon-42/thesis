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
 *  • THE LOSSES OF SEPARATION (`loss` switch) of the round on screen, between any two of the window's aircraft (stage C's:
 *    the one its commanded aircraft is in): the two at the loss's time, joined by a red line, with the minimum and the
 *    distance the judge found (the exporter's numbers); dashed when no commanded aircraft answers for it (D145).
 *  • STAGE D'S OTHER COMMANDED AIRCRAFT (frontend §5.4): each one's flown track of the round (`otherCommanded` switch),
 *    thinner and at half opacity, and its point at the cursor (always; a click selects it); a commanded aircraft's track
 *    from the row it is silent on is slate and dashed (D144), the one on screen's too.
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
import { pickTrainingWindowAircraft, useTrainingWindowLayers } from "../data/trainingWindowLayers";
import {
  lossesOf,
  onAircraftClock,
  onWindowClock,
  otherOf,
  roundLabel,
  silentFromS,
  trackAt,
  trainingWindowOriginOf,
  windowAircraftAt,
  windowShiftText,
  type TrainingWindowOrigin,
  type TrainingWindowRole,
} from "../data/trainingWindowSample";
import { roundKind } from "../data/trainingSentenceKind";
import {
  TRAINING_FAILURE_COLOR, TRAINING_SENTENCE_COLOR, TRAINING_TRAFFIC_COLOR, trainingOutcomeColour,
} from "../utils/trainingWordColors";
import { TRAINING_OUTCOME_TAG } from "../data/trainingText";
import { airLine, entityGroup, lonLatHeights, marker } from "../scene/trainingEntities";

/** The id prefix of another commanded aircraft's point at the cursor: a click on one selects that aircraft. */
const COMMANDED_AT = "training-window-commanded-at-";

/** Each role's colour: an aircraft on its record, window A's inserted one, window D's moved one. */
const TRAINING_WINDOW_ROLE_COLOR_MOVED = "#fbbf24";
export const TRAINING_WINDOW_ROLE_COLOR: Record<TrainingWindowRole, string> = {
  recorded: TRAINING_TRAFFIC_COLOR,
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
  commandedTrack: (key: string) => `training-window-commanded-track-${key}`,
  commandedSilent: (key: string) => `training-window-commanded-silent-${key}`,
  commandedAt: (key: string) => `${COMMANDED_AT}${key}`,
  ownSilent: "training-window-own-silent",
  movedStart: "training-window-moved-start",
};
/** Window B's moved start: the colour of the moved aircraft of window D (a start moved, not the record). */
const MOVED_START_COLOR = TRAINING_WINDOW_ROLE_COLOR_MOVED;

/** The keys of the other aircraft of the losses the commanded aircraft on screen is in, in its round. */
function lostWith(origin: TrainingWindowOrigin): Set<string> {
  const id = origin.aircraft.datasetId;
  return new Set(lossesOf(origin.end, id).map((loss) => otherOf(loss, id)));
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
    const at = trackAt(aircraft, windowS);
    if (at === null) continue;
    const hue = lost.has(aircraft.key) ? TRAINING_FAILURE_COLOR : TRAINING_WINDOW_ROLE_COLOR[aircraft.role];
    group.add(marker(ENTITY.at(aircraft.key), `${aircraft.key} at ${timeS.toFixed(0)} s`, Cesium.Cartesian3.fromDegrees(...at), hue, 9,
      aircraft.key.split(":").pop()));
  }
  return group;
}

/** Every loss of separation of the round on screen, between any two of the window's aircraft (stage C's: the one its
 *  commanded aircraft is in): the two at the loss's time joined by a red line, with the minimum and the distance; dashed
 *  when no commanded aircraft answers for it (a loss the records also have, multi-aircraft control D145). */
function drawLosses(viewer: Cesium.Viewer, origin: TrainingWindowOrigin) {
  const group = entityGroup(viewer);
  const place = origin.window.rounds.indexOf(origin.end);
  origin.end.losses.forEach((loss, k) => {
    const [first, second] = loss.aircraft;
    const one = windowAircraftAt(origin.window, place, first, loss.timeS);
    const two = windowAircraftAt(origin.window, place, second, loss.timeS);
    if (one === null || two === null) return;
    const oneAt = Cesium.Cartesian3.fromDegrees(...one);
    const twoAt = Cesium.Cartesian3.fromDegrees(...two);
    const answered = loss.answering.length === 0 ? "answered by no commanded aircraft (D145)"
      : `${loss.answering.join(", ")} answer${loss.answering.length === 1 ? "s" : ""} for it`;
    const text = `lost separation (${loss.kind}) · ${loss.distanceM.toFixed(0)} m of ${loss.requiredM.toFixed(0)} m required · ${answered}`;
    group.add(airLine(ENTITY.loss(k), text, [oneAt, twoAt], TRAINING_FAILURE_COLOR, 3, 1, loss.answering.length === 0));
    group.add(marker(ENTITY.lossOwn(k), `${first} at the loss (${loss.timeS.toFixed(0)} s of the window)`, oneAt, TRAINING_FAILURE_COLOR, 10, text));
    group.add(marker(ENTITY.lossOther(k), `${second} at the loss`, twoAt, TRAINING_FAILURE_COLOR, 10));
  });
  return group;
}

/** The window's other commanded aircraft in the round on screen (frontend §5.4): each one's flown track in the colour
 *  of its round's sentences, thinner and at half opacity, slate and dashed from the row it is silent on (D144). */
function drawOtherCommandedTracks(viewer: Cesium.Viewer, origin: TrainingWindowOrigin) {
  const group = entityGroup(viewer);
  const place = origin.window.rounds.indexOf(origin.end);
  const colour = TRAINING_SENTENCE_COLOR[roundKind(origin.round, origin.sample.model.start, origin.sample.stage)];
  for (const aircraft of origin.window.commanded) {
    if (aircraft === origin.aircraft) continue;
    const sentence = aircraft.rounds[place];
    const silentS = silentFromS(aircraft, sentence, origin.sample.model.rowIntervalS);
    const title = `${aircraft.head.callsign} (${aircraft.datasetId}), commanded: ${TRAINING_OUTCOME_TAG[sentence.outcome]} · r ${sentence.reward.toFixed(2)}`;
    const [spoken, silent] = splitTrack(sentence.flown, silentS);
    group.add(airLine(ENTITY.commandedTrack(aircraft.datasetId), title, Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(spoken)),
      colour, 1.5, 0.5));
    if (silent !== null) {
      group.add(airLine(ENTITY.commandedSilent(aircraft.datasetId), `${title} · silent from ${silentS!.toFixed(0)} s`,
        Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(silent)), TRAINING_TRAFFIC_COLOR, 1.5, 0.6, true));
    }
  }
  return group;
}

/** The commanded aircraft on screen from the row it is silent on: its flown track there, slate and dashed, over the
 *  flown path stage A's layer draws (frontend §5.4, D144). */
function drawOwnSilence(viewer: Cesium.Viewer, origin: TrainingWindowOrigin) {
  const group = entityGroup(viewer);
  const silentS = silentFromS(origin.aircraft, origin.sentence, origin.sample.model.rowIntervalS);
  const [, silent] = splitTrack(origin.sentence.flown, silentS);
  if (silent !== null) {
    group.add(airLine(ENTITY.ownSilent, `silent from ${silentS!.toFixed(0)} s: it answered for a loss of separation and flies on its words in force`,
      Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(silent)), TRAINING_TRAFFIC_COLOR, 3, 0.9, true));
  }
  return group;
}

/** The window's other commanded aircraft at the cursor (``timeS``, on the flight clock of the aircraft on screen): each
 *  one's point from the row it joins to its end, always drawn (Draw "Other commanded tracks" hides only the tracks); a
 *  click on one selects it (`pickTrainingWindowAircraft`). */
function drawOtherCommandedAt(viewer: Cesium.Viewer, origin: TrainingWindowOrigin, timeS: number) {
  const group = entityGroup(viewer);
  const place = origin.window.rounds.indexOf(origin.end);
  const windowS = onWindowClock(origin.aircraft, timeS);
  const colour = TRAINING_SENTENCE_COLOR[roundKind(origin.round, origin.sample.model.start, origin.sample.stage)];
  for (const aircraft of origin.window.commanded) {
    if (aircraft === origin.aircraft) continue;
    const sentence = aircraft.rounds[place];
    const at = windowAircraftAt(origin.window, place, aircraft.datasetId, windowS);
    if (at === null) continue;
    const silentS = silentFromS(aircraft, sentence, origin.sample.model.rowIntervalS);
    const silent = silentS !== null && onAircraftClock(aircraft, windowS) >= silentS;
    group.add(marker(ENTITY.commandedAt(aircraft.datasetId), `${aircraft.head.callsign}: click to select it`,
      Cesium.Cartesian3.fromDegrees(...at), silent ? TRAINING_TRAFFIC_COLOR : colour, 11, aircraft.head.callsign));
  }
  return group;
}

/** A flown track cut at flight time ``fromS``: the part before (to the row at it) and the part from it (null: no cut). */
function splitTrack<T extends { tS: number[]; lon: number[]; lat: number[]; altitudeHaeM: number[] }>(track: T, fromS: number | null
): [{ tS: number[]; lon: number[]; lat: number[]; altitudeHaeM: number[] }, { tS: number[]; lon: number[]; lat: number[]; altitudeHaeM: number[] } | null] {
  if (fromS === null) return [track, null];
  const cut = track.tS.findIndex((t) => t >= fromS);
  if (cut < 0) return [track, null];
  const part = (from: number, to: number) => ({ tS: track.tS.slice(from, to), lon: track.lon.slice(from, to),
    lat: track.lat.slice(from, to), altitudeHaeM: track.altitudeHaeM.slice(from, to) });
  return [part(0, cut + 1), part(cut, track.tS.length)];
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
    group.add(airLine(ENTITY.round(String(sentence.round)), `${roundLabel(sentence.round, origin.sample.model.start)}: ${TRAINING_OUTCOME_TAG[sentence.outcome]}`,
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

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || origin === undefined || !layers.otherCommanded) return;
    const group = drawOtherCommandedTracks(viewer, origin);
    return () => group.remove();
  }, [viewer, origin, layers.otherCommanded]);

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || origin === undefined) return;
    const group = drawOtherCommandedAt(viewer, origin, trainingCursorS);
    return () => group.remove();
  }, [viewer, origin, trainingCursorS]);

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || origin === undefined) return;
    const group = drawOwnSilence(viewer, origin);
    return () => group.remove();
  }, [viewer, origin]);

  // a click on another commanded aircraft's point selects it (frontend §5.2); only while a window with several is shown
  const several = origin !== undefined && origin.window.commanded.length > 1;
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || !several) return;
    const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction((click: Cesium.ScreenSpaceEventHandler.PositionedEvent) => {
      const picked: unknown = viewer.scene.pick(click.position);
      const id = (picked as { id?: { id?: unknown } } | undefined)?.id?.id;
      if (typeof id === "string" && id.startsWith(COMMANDED_AT)) pickTrainingWindowAircraft(id.slice(COMMANDED_AT.length));
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
    return () => handler.destroy();
  }, [viewer, several]);
}
