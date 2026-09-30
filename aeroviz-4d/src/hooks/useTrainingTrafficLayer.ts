/**
 * useTrainingTrafficLayer.ts
 * --------------------------
 * A multi-aircraft window in the 3D scene (`trainingWindow`, the Training module §4.11): every aircraft of the window
 * besides the one on screen — whose track, sentence and envelopes the single-flight layers draw (`useTrainingTrackLayer`)
 * — as the sentence bar's source reads the window (`windowReading`: the record, or one sample of a model).
 *
 *  • EACH AIRCRAFT BY ITS ROLE (`TrainingAircraftRole`, one table: `ROLE_DRAW`), so the aircraft the view is about is
 *    told from the rest at a glance (the user, 2026-09-30): the one ON SCREEN — its track drawn by the single-flight
 *    layers — the aircraft model large and white, ringed in the selection's yellow, its callsign "▶ …" and its attitude
 *    on a yellow chip; every COMMANDED aircraft its track in the model's colour (the record's: the observed track's
 *    near-white), its model tinted and its callsign in it; a REPLAYED aircraft thin and faded in slate, a BACKGROUND arrival fainter and darker
 *    (`TRAINING_OTHER_AIRCRAFT_COLOR`). The commanded are a set: several drawn as commanded at once is the ordinary case.
 *  • WHERE EACH IS AT THE CURSOR (the window's clock, `trainingSceneS`): the aircraft model in its exported attitude
 *    (`trainingAttitude.poseAt`: heading, path angle as the pitch, right bank), hidden outside its span — drawn, never
 *    flown here.
 *  • LOSSES OF SEPARATION: a pair under its minimum at the cursor joined by a red line with the closest it came against
 *    its minimum (the judge's numbers); VISUAL solid — the reading that ends an aircraft — IFR dashed where only IFR has
 *    it; and a red cross, fixed, where the judge ended an aircraft.
 *
 * Built once per window and reading; the cursor moves only the aircraft and the loss lines. STATIC ENTITIES, as every
 * Training layer: the shared `viewer.clock` belongs to Observe. Call it from the leaf (`TrainingScene`): it reads the
 * cursor.
 */

import { useEffect, useMemo, useRef } from "react";
import * as Cesium from "cesium";
import { useApp, useTrainingCursor } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import {
  aircraftModel, airLine, colour, entityGroup, lonLatHeights, placeAircraft, type EntityOptions,
} from "../scene/trainingEntities";
import { poseAt, poseText, type TrainingAircraftPose } from "../data/trainingAttitude";
import {
  episodesAt,
  windowReading,
  type TrainingAircraftRole,
  type TrainingSceneTrack,
  type TrainingWindowReading,
  type TrainingWindowView,
} from "../data/trainingTraffic";
import {
  TRAINING_LOSS_COLOR,
  TRAINING_OTHER_AIRCRAFT_ALPHA,
  TRAINING_OTHER_AIRCRAFT_COLOR,
  TRAINING_ON_SCREEN_COLOR,
  TRAINING_SURFACE_COLOR,
  trainingWindowReadingColour,
} from "../utils/trainingWordColors";

const ID = "training-traffic";

/** How each role is drawn: its track (null: the single-flight layers draw it), its model — least size on screen (px),
 *  how much of its tint shows, how opaque, its ring (px) — and its callsign. */
const ROLE_DRAW: Record<TrainingAircraftRole, {
  track: { width: number; alpha: number } | null;
  modelPx: number;
  blend: number;
  alpha: number;
  ringPx: number;
  labelFont: string;
  labelAlpha: number;
}> = {
  onScreen: { track: null, modelPx: 64, blend: 0.2, alpha: 1, ringPx: 3, labelFont: "700 13px sans-serif", labelAlpha: 1 },
  commanded: { track: { width: 2.5, alpha: 1 }, modelPx: 46, blend: 0.6, alpha: 1, ringPx: 2, labelFont: "600 12px sans-serif",
    labelAlpha: 1 },
  replayed: { track: { width: 1.2, alpha: TRAINING_OTHER_AIRCRAFT_ALPHA.replayed }, modelPx: 34, blend: 0.7, alpha: 0.85, ringPx: 0,
    labelFont: "500 11px sans-serif", labelAlpha: 0.8 },
  background: { track: { width: 1, alpha: TRAINING_OTHER_AIRCRAFT_ALPHA.background }, modelPx: 28, blend: 0.7, alpha: 0.7,
    ringPx: 0, labelFont: "500 10px sans-serif", labelAlpha: 0.65 },
};

/** Every aircraft of the window as the reading draws it: its id, name, role, colour, track on the window's clock. */
interface SceneAircraft {
  datasetId: string;
  callsign: string;
  role: TrainingAircraftRole;
  css: string;
  track: TrainingSceneTrack;
}

function sceneAircraft(view: TrainingWindowView, reading: TrainingWindowReading, onScreen: string | null): SceneAircraft[] {
  const commandedCss = trainingWindowReadingColour(reading);
  return [
    ...view.window.commanded.map((one, at) => ({ datasetId: one.flight.datasetId, callsign: one.flight.callsign,
      role: one.flight.datasetId === onScreen ? "onScreen" as const : "commanded" as const, css: commandedCss, track: reading.tracks[at] })),
    ...view.window.others.map((other) => ({ datasetId: other.datasetId, callsign: other.callsign, role: other.role,
      css: TRAINING_OTHER_AIRCRAFT_COLOR[other.role], track: other.track })),
  ];
}

/** An aircraft's label: its callsign — the one on screen's "▶ …" and its attitude there. */
function labelText(one: SceneAircraft, pose: TrainingAircraftPose): string {
  return one.role === "onScreen" ? `▶ ${one.callsign}\n${poseText(pose)}` : one.callsign;
}

/** An aircraft where ``pose`` has it: its model and callsign, drawn by its role; the label visible through terrain. */
function aircraftMark(one: SceneAircraft, pose: TrainingAircraftPose): EntityOptions {
  const draw = ROLE_DRAW[one.role];
  const onScreen = one.role === "onScreen";
  return {
    ...aircraftModel(`${ID}-at-${one.datasetId}`, onScreen ? `${one.callsign} (on screen)` : `${one.callsign} (${one.role})`, pose,
      { css: onScreen ? "#ffffff" : one.css, blend: draw.blend, alpha: draw.alpha, minimumPixelSize: draw.modelPx,
        ringCss: onScreen ? TRAINING_ON_SCREEN_COLOR : "#000000", ringPx: draw.ringPx }),
    label: {
      text: labelText(one, pose), font: draw.labelFont,
      fillColor: onScreen ? colour(TRAINING_SURFACE_COLOR) : colour(one.css, draw.labelAlpha),
      ...(onScreen
        ? { showBackground: true, backgroundColor: colour(TRAINING_ON_SCREEN_COLOR, 0.9), style: Cesium.LabelStyle.FILL }
        : { outlineColor: Cesium.Color.BLACK, outlineWidth: 3, style: Cesium.LabelStyle.FILL_AND_OUTLINE }),
      pixelOffset: new Cesium.Cartesian2(0, 30), verticalOrigin: Cesium.VerticalOrigin.TOP,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
  };
}

function cartesian(at: { lon: number; lat: number; heightHaeM: number }): Cesium.Cartesian3 {
  return Cesium.Cartesian3.fromDegrees(at.lon, at.lat, at.heightHaeM);
}

export default function useTrainingTrafficLayer(): void {
  const { viewer, mode, trainingWindow, trainingSelection, trainingSource } = useApp();
  const { trainingSceneS } = useTrainingCursor();
  const view = mode === "training" ? trainingWindow : null;
  const onScreen = trainingSelection?.flight.datasetId ?? null;
  const reading = useMemo(() => (view === null ? null : windowReading(view, trainingSource)), [view, trainingSource]);
  const aircraft = useMemo(() => (view === null || reading === null ? [] : sceneAircraft(view, reading, onScreen)),
    [view, reading, onScreen]);
  const models = useRef<Map<string, Cesium.Entity>>(new Map());

  // THE TRACKS, the aircraft and the judge's ends: built once per window, reading and aircraft on screen
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || view === null || reading === null) return;
    const group = entityGroup(viewer);
    const made = new Map<string, Cesium.Entity>();
    for (const one of aircraft) {
      const track = ROLE_DRAW[one.role].track;
      if (track !== null) {
        group.add(airLine(`${ID}-track-${one.datasetId}`, `${one.callsign} (${one.role})`,
          Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(one.track)), one.css, track.width, track.alpha));
      }
      made.set(one.datasetId, group.add(aircraftMark(one, poseAt(one.track, one.track.tS[0])!)));
    }
    for (const end of reading.losses.visual.ended) {
      const one = aircraft.find((item) => item.datasetId === end.datasetId)!;
      const at = poseAt(one.track, end.atS);
      if (at === null) continue;                            // ended before its track starts: nothing to mark there
      group.add({ id: `${ID}-ended-${end.datasetId}`, name: `${one.callsign}: lost separation (${end.kind})`, position: cartesian(at),
        label: { text: "✕", font: "700 26px sans-serif", fillColor: colour(TRAINING_LOSS_COLOR), outlineColor: Cesium.Color.BLACK,
          outlineWidth: 3, style: Cesium.LabelStyle.FILL_AND_OUTLINE, disableDepthTestDistance: Number.POSITIVE_INFINITY } });
    }
    models.current = made;
    return () => {
      models.current = new Map();
      group.remove();
    };
  }, [viewer, view, reading, aircraft]);

  // AT THE CURSOR: each aircraft where it is and how it sits, the pairs under their minimum joined
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || view === null || reading === null) return;
    const where = new Map<string, Cesium.Cartesian3>();
    for (const one of aircraft) {
      const pose = poseAt(one.track, trainingSceneS);
      const entity = models.current.get(one.datasetId)!;
      entity.show = pose !== null;
      if (pose !== null) {
        placeAircraft(entity, pose);
        entity.label!.text = new Cesium.ConstantProperty(labelText(one, pose));
        where.set(one.datasetId, cartesian(pose));
      }
    }
    const group = entityGroup(viewer);
    const stepS = view.set.vocabulary.stepS;
    const visual = episodesAt(reading.losses.visual, trainingSceneS, stepS);
    const pairKey = (pair: [string, string]) => pair.join("|");
    const drawn = new Set(visual.map((episode) => pairKey(episode.pair)));
    const ifrOnly = episodesAt(reading.losses.ifr, trainingSceneS, stepS).filter((episode) => !drawn.has(pairKey(episode.pair)));
    for (const [episode, solid] of [...visual.map((item) => [item, true] as const), ...ifrOnly.map((item) => [item, false] as const)]) {
      const [a, b] = episode.pair.map((id) => where.get(id));
      if (a === undefined || b === undefined) continue;
      group.add({
        id: `${ID}-loss-${solid ? "visual" : "ifr"}-${pairKey(episode.pair)}`,
        name: `${solid ? "VISUAL" : "IFR"} loss: closest ${Math.round(episode.closestM)} m of ${Math.round(episode.requiredM)} m`,
        position: Cesium.Cartesian3.midpoint(a, b, new Cesium.Cartesian3()),
        polyline: { positions: [a, b], width: 3,
          material: solid ? colour(TRAINING_LOSS_COLOR) : new Cesium.PolylineDashMaterialProperty({ color: colour(TRAINING_LOSS_COLOR) }) },
        label: { text: `${Math.round(episode.closestM)} / ${Math.round(episode.requiredM)} m`, font: "600 12px sans-serif",
          fillColor: colour(TRAINING_LOSS_COLOR), outlineColor: Cesium.Color.BLACK, outlineWidth: 3,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE, disableDepthTestDistance: Number.POSITIVE_INFINITY },
      });
    }
    viewer.scene.requestRender();
    return () => {
      group.remove();
      if (isCesiumViewerUsable(viewer)) viewer.scene.requestRender();
    };
  }, [viewer, view, reading, aircraft, trainingSceneS]);
}
