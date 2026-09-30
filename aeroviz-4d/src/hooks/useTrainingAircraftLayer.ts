/**
 * useTrainingAircraftLayer.ts
 * ---------------------------
 * THE AIRCRAFT AT THE CURSOR on a flight of its own (Training module design, doc 36 §4.12): the aircraft model on the
 * sentence read — the observed track for the truth, the sample's flown track for a model's (from its first predicted
 * step: the steps before are observed only) — where it is at the cursor and in the attitude exported there
 * (`trainingAttitude.poseAt`: heading, path angle as the pitch, right bank), its attitude written under it; past the
 * track's end it stays where the flight ended. A window's aircraft are the traffic layer's (`useTrainingTrafficLayer`),
 * the one on screen too, and the live executor's aircraft is its own (`useTrainingExecutorLayers`).
 *
 * Built once per flight and sentence read; the cursor moves only the aircraft. Call it from the leaf
 * (`useTrainingTrackLayer`): it reads the cursor.
 */

import { useEffect, useMemo, useRef } from "react";
import * as Cesium from "cesium";
import { useApp, useTrainingCursor } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { aircraftModel, colour, entityGroup, placeAircraft } from "../scene/trainingEntities";
import { poseAt, poseText, type TrainingAttitudeTrack } from "../data/trainingAttitude";
import { cursorOnFlight } from "../data/trainingSample";
import { generationOnScreen, sentenceAxisEndS } from "../data/trainingOverlays";
import { windowOnScreen } from "../data/trainingTraffic";
import { TRAINING_TRACE_COLOR, trainingModelColour } from "../utils/trainingWordColors";

const ID = "training-aircraft";
/** The model's least size on screen (px). */
const AIRCRAFT_PX = 44;

export default function useTrainingAircraftLayer(): void {
  const { viewer, mode, trainingSelection, trainingGenerations, trainingSource, trainingWindow } = useApp();
  const { trainingCursorS } = useTrainingCursor();
  // a flight of its own: a window's aircraft are the traffic layer's
  const selection = mode === "training" && windowOnScreen(trainingWindow, trainingSelection) === null ? trainingSelection : null;
  const shown = generationOnScreen(trainingGenerations, trainingSource, selection);
  const view = shown?.view ?? null;
  const read = shown?.sentence ?? null;
  // the sentence read and whose it is: its track, its colour (keyed on what it is built from: a new object every render)
  const drawn = useMemo((): { track: TrainingAttitudeTrack; css: string; who: string } | null => {
    if (selection === null) return null;
    if (view === null) return { track: selection.flight.signals, css: TRAINING_TRACE_COLOR, who: `${selection.flight.callsign} (observed)` };
    return read === null ? null
      : { track: read.track, css: trainingModelColour(view.overlay.model), who: `${selection.flight.callsign} (sample ${read.sample + 1})` };
  }, [selection, view, read]);
  const entity = useRef<Cesium.Entity | null>(null);

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || drawn === null) return;
    const group = entityGroup(viewer);
    const first = poseAt(drawn.track, drawn.track.tS[0])!;
    entity.current = group.add({
      ...aircraftModel(ID, drawn.who, first, { css: drawn.css, blend: 0.4, alpha: 1, minimumPixelSize: AIRCRAFT_PX,
        ringCss: "#000000", ringPx: 1 }),
      label: { text: poseText(first), font: "600 12px sans-serif", fillColor: colour(drawn.css), outlineColor: Cesium.Color.BLACK,
        outlineWidth: 3, style: Cesium.LabelStyle.FILL_AND_OUTLINE, pixelOffset: new Cesium.Cartesian2(0, 30),
        verticalOrigin: Cesium.VerticalOrigin.TOP, disableDepthTestDistance: Number.POSITIVE_INFINITY },
    });
    return () => {
      entity.current = null;
      group.remove();
    };
  }, [viewer, drawn]);

  // AT THE CURSOR: hidden before the track starts (and off the flight's clock), held at its end after it
  const endS = selection === null ? 0 : sentenceAxisEndS(selection.flight, selection.vocabulary.stepS, read);
  const onFlight = selection !== null && cursorOnFlight(selection, trainingCursorS, endS);
  useEffect(() => {
    const aircraft = entity.current;
    if (!isCesiumViewerUsable(viewer) || drawn === null || aircraft === null) return;
    const { tS } = drawn.track;
    const pose = onFlight && trainingCursorS >= tS[0] ? poseAt(drawn.track, Math.min(trainingCursorS, tS[tS.length - 1])) : null;
    aircraft.show = pose !== null;
    if (pose !== null) {
      placeAircraft(aircraft, pose);
      aircraft.label!.text = new Cesium.ConstantProperty(poseText(pose));
    }
    viewer.scene.requestRender();
  }, [viewer, drawn, onFlight, trainingCursorS]);
}
