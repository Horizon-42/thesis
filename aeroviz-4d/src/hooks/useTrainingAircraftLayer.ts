/**
 * useTrainingAircraftLayer.ts
 * ---------------------------
 * THE AIRCRAFT AT THE CURSOR (Training module design, doc 36 §4.12): the aircraft model on the observed track where it is
 * at the cursor, in the attitude exported there (`trainingAttitude.poseAt`: heading, path angle as the pitch, right bank),
 * its attitude written under it — and, when a closed-loop sentence is read, a second one in the flown path's colour on the
 * path the executor flew, so the two are seen apart in 3D at the same moment. Each is hidden before its track starts and
 * held where its track ends. The live executor's aircraft is its own (`useTrainingLiveLayer`).
 *
 * Built once per flight and reading; the cursor moves only the aircraft. Call it from the leaf
 * (`useTrainingTrackLayer`): it reads the cursor.
 */

import { useEffect, useMemo, useRef } from "react";
import * as Cesium from "cesium";
import { useApp, useTrainingCursor } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { aircraftModel, colour, entityGroup, placeAircraft, TRAINING_ENTITY } from "../scene/trainingEntities";
import { poseAt, poseText, type TrainingAttitudeTrack } from "../data/trainingAttitude";
import { trainingReadingOf } from "../data/trainingSample";
import { TRAINING_TRACE_COLOR } from "../utils/trainingWordColors";
import { flownSentenceColour } from "../data/trainingSentenceKind";

/** The model's least size on screen (px). */
const AIRCRAFT_PX = 64;
/** How far under its aircraft each label sits (px): the flown one lower, so the two never overlap. */
const LABEL_DROP_PX: Record<string, number> = { [TRAINING_ENTITY.aircraftObserved]: 30, [TRAINING_ENTITY.aircraftFlown]: 62 };

interface Drawn {
  id: string;
  track: TrainingAttitudeTrack;
  css: string;
  who: string;
}

export default function useTrainingAircraftLayer(): void {
  const { viewer, mode, trainingSelection, trainingIntervalS, trainingLayers } = useApp();
  const observedShown = trainingLayers.observed;
  const { trainingCursorS } = useTrainingCursor();
  const selection = mode === "training" ? trainingSelection : null;
  // the aircraft drawn for the reading on screen: the observed one (unless its switch is off), and the flown one of a
  // closed-loop reading
  const drawn = useMemo((): Drawn[] => {
    if (selection === null) return [];
    const { flight } = selection;
    const reading = trainingReadingOf(flight, selection.vocabulary.stepS, trainingIntervalS);
    return [
      ...(observedShown
        ? [{ id: TRAINING_ENTITY.aircraftObserved, track: reading.observed, css: TRAINING_TRACE_COLOR, who: `${flight.callsign} (observed)` }]
        : []),
      ...(reading.closed === null ? []
        : [{ id: TRAINING_ENTITY.aircraftFlown, track: reading.closed.flown, css: flownSentenceColour(flight),
          who: `${flight.callsign} (flown, Δ ${reading.closed.rowIntervalS} s)` }]),
    ];
  }, [selection, trainingIntervalS, observedShown]);
  const entities = useRef<Map<string, Cesium.Entity>>(new Map());

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || drawn.length === 0) return;
    const group = entityGroup(viewer);
    for (const item of drawn) {
      const first = poseAt(item.track, item.track.tS[0])!;
      entities.current.set(item.id, group.add({
        ...aircraftModel(item.id, item.who, first, { css: item.css, blend: 0.4, alpha: 1, minimumPixelSize: AIRCRAFT_PX,
          ringCss: "#000000", ringPx: 2 }),
        label: { text: poseText(first), font: "600 12px sans-serif", fillColor: colour(item.css), outlineColor: Cesium.Color.BLACK,
          outlineWidth: 3, style: Cesium.LabelStyle.FILL_AND_OUTLINE, pixelOffset: new Cesium.Cartesian2(0, LABEL_DROP_PX[item.id]),
          verticalOrigin: Cesium.VerticalOrigin.TOP, disableDepthTestDistance: Number.POSITIVE_INFINITY },
      }));
    }
    return () => {
      entities.current.clear();
      group.remove();
    };
  }, [viewer, drawn]);

  // AT THE CURSOR: hidden before a track starts, held at its end after it
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer)) return;
    for (const item of drawn) {
      const aircraft = entities.current.get(item.id);
      if (aircraft === undefined) continue;
      const { tS } = item.track;
      const pose = trainingCursorS >= tS[0] ? poseAt(item.track, Math.min(trainingCursorS, tS[tS.length - 1])) : null;
      aircraft.show = pose !== null;
      if (pose !== null) {
        placeAircraft(aircraft, pose);
        aircraft.label!.text = new Cesium.ConstantProperty(poseText(pose));
      }
    }
    viewer.scene.requestRender();
  }, [viewer, drawn, trainingCursorS]);
}
