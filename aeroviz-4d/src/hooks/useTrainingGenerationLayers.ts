/**
 * useTrainingGenerationLayers.ts
 * ------------------------------
 * The MODEL SENTENCE read, in the 3D scene (called from `useTrainingTrackLayer`, in the leaf `TrainingScene`): when the
 * sentence bar reads one of a model's own sentences (`trainingSource`, `generationOnScreen`), every sample that model said
 * over the flight on screen, as the executor flew it —
 *
 *  • THE SAMPLE READ solid in the model's colour at its ellipsoid height (dashed where terrain hides it), its ground trace
 *    dashed under it, where its flight ended (named with the model, the sample and the outcome), and where it said each
 *    heading word and the clearance — every mark of the model's in its own colour, the truth's in the column colours;
 *  • ITS OTHER SAMPLES thin and faint, so the spread of what the model says over one flight is seen at once.
 *
 * The truth — the observed track and the envelopes of its sentence — is drawn whatever is read (`useTrainingTrackLayer`):
 * a model's flight is always read against it. When the truth is read, nothing of a model is drawn.
 *
 * THE SELECTED WORD, when a model's sentence is read: the stretch of the sample's track where the word is in force,
 * yellow, and its issue named — the truth's envelopes keep their colours (they are the truth's words, not this one).
 * Keyed on the word's issue row: a cursor moving inside one word repaints nothing.
 *
 * STATIC ENTITIES, built once per flight, model and sample; nothing here moves the camera or the shared clock.
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp, useTrainingCursor } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { TRAINING_OTHER_SAMPLE_ALPHA, TRAINING_WORD_COLOR, trainingModelColour } from "../utils/trainingWordColors";
import {
  generatedRowAt,
  generatedTrackRows,
  generationOnScreen,
  type TrainingGeneratedSentence,
  type TrainingGenerationView,
} from "../data/trainingOverlays";
import {
  sentenceWordAt,
  trainingClearedValue,
  trainingWordLabel,
  TRAINING_COLUMN_INDEX,
  type TrainingColumn,
  type TrainingSelection,
} from "../data/trainingSample";
import { TRAINING_OUTCOME_TAG } from "../data/trainingText";
import {
  airLine,
  colour,
  entityGroup,
  groundLine,
  lonLatHeights,
  marker,
  planDegrees,
  TRAINING_ENTITY,
} from "../scene/trainingEntities";

/** Every sample of the model over the flight, the one read solid, its ground trace, end and word issues. */
function buildModel(viewer: Cesium.Viewer, view: TrainingGenerationView, read: TrainingGeneratedSentence,
  selection: TrainingSelection): () => void {
  const group = entityGroup(viewer);
  const css = trainingModelColour(view.overlay.model);
  const label = view.overlay.model.label;
  for (const item of view.flight.samples) {
    if (item.sample === read.sample) continue;
    group.add({
      id: TRAINING_ENTITY.modelTrack(item.sample), name: `${label}, sample ${item.sample + 1}: ${TRAINING_OUTCOME_TAG[item.outcome]}`,
      polyline: { positions: Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(item.track)), width: 1.5,
        material: colour(css, TRAINING_OTHER_SAMPLE_ALPHA) },
    });
  }
  const { track } = read;
  group.add(groundLine(TRAINING_ENTITY.modelGround, `${label}'s ground trace`, planDegrees(track), 2,
    new Cesium.PolylineDashMaterialProperty({ color: colour(css, 0.6) })));
  group.add(airLine(TRAINING_ENTITY.modelTrack(read.sample), `${label}, sample ${read.sample + 1}, as the executor flew it`,
    Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(track)), css, 3));
  const at = (point: number) => Cesium.Cartesian3.fromDegrees(track.lon[point], track.lat[point], track.altitudeHaeM[point]);
  const last = track.tS.length - 1;
  // where it said its heading words and cleared the flight (a word said after the flight ended has no point)
  const cleared = trainingClearedValue(selection.vocabulary);
  read.events.forEach((event, index) => {
    const point = event.row - read.firstRow;
    if (point > last) return;
    if (event.column === TRAINING_COLUMN_INDEX.heading && event.row > read.firstRow) {
      group.add(marker(TRAINING_ENTITY.modelIssue(index), `${label}: heading word said`, at(point), css, 6));
    } else if (event.column === TRAINING_COLUMN_INDEX.approach && event.value === cleared && event.row > read.firstRow) {
      group.add(marker(TRAINING_ENTITY.modelIssue(index), `${label}: cleared to join the final`, at(point), css, 10));
    }
  });
  group.add(marker(TRAINING_ENTITY.modelEnd, `Where ${label}'s flight ended`, at(last), css, 10,
    `${label} #${read.sample + 1}: ${TRAINING_OUTCOME_TAG[read.outcome]}`));
  return group.remove;
}

/** THE SELECTED WORD of the model's sentence: the stretch of its track where it is in force, and its issue named. */
function paintModelFocus(viewer: Cesium.Viewer, read: TrainingGeneratedSentence, selection: TrainingSelection,
  column: TrainingColumn, row: number, endRow: number, value: number): () => void {
  const group = entityGroup(viewer);
  const rows = generatedTrackRows(read, read.firstRow, row, endRow);
  if (rows === null) return group.remove;
  const { track } = read;
  if (rows.last > rows.first) {
    group.add({
      id: TRAINING_ENTITY.modelFocusStretch, name: "Where the selected word is in force",
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(track).slice(rows.first * 3, (rows.last + 1) * 3)),
        width: 7, material: colour(TRAINING_WORD_COLOR, 0.85),
        depthFailMaterial: new Cesium.PolylineDashMaterialProperty({ color: colour(TRAINING_WORD_COLOR, 0.5) }),
      },
    });
  }
  group.add(marker(TRAINING_ENTITY.modelFocusIssue, "The selected word, where it was said",
    Cesium.Cartesian3.fromDegrees(track.lon[rows.first], track.lat[rows.first], track.altitudeHaeM[rows.first]), TRAINING_WORD_COLOR, 13,
    `${column} ${trainingWordLabel(selection.vocabulary, selection.candidates, column, value)} · step ${row}`));
  return group.remove;
}

export default function useTrainingGenerationLayers(): void {
  const { viewer, mode, trainingSelection, trainingGenerations, trainingSource, trainingColumn } = useApp();
  const { trainingCursorS } = useTrainingCursor();
  const selection = mode === "training" ? trainingSelection : null;
  const shown = generationOnScreen(trainingGenerations, trainingSource, selection);
  const view = shown?.view ?? null;
  const read = shown?.sentence ?? null;

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || view === null || read === null || selection === null) return;
    const remove = buildModel(viewer, view, read, selection);
    return () => {
      if (isCesiumViewerUsable(viewer)) remove();
    };
  }, [viewer, view, read, selection]);

  const word = read !== null && selection !== null && trainingColumn !== null
    ? sentenceWordAt(read, trainingColumn, generatedRowAt(selection.vocabulary.stepS, trainingCursorS)) : null;
  const wordRow = word?.row ?? null;
  const wordEnd = word?.endRow ?? null;
  const wordValue = word?.value ?? null;
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || read === null || selection === null || trainingColumn === null || wordRow === null) return;
    const remove = paintModelFocus(viewer, read, selection, trainingColumn, wordRow, wordEnd!, wordValue!);
    return () => {
      if (isCesiumViewerUsable(viewer)) remove();
    };
  }, [viewer, read, selection, trainingColumn, wordRow, wordEnd, wordValue]);
}
