/**
 * useTrainingProcedureLayer.ts
 * ----------------------------
 * The 3D scene of a prior set beyond stage A's layers (`useTrainingTrackLayer` draws the sentence on screen's flown path, DA
 * point and words as it does for stage A, since the prior's sentence is read as a closed-loop one):
 *
 *  • THE PROCEDURE'S LIMITS (`procedure` switch), per candidate runway: the outline of the region the masks rule (inside the FAF
 *    and the LPV cone) on the ground, the glidepath lower edge along the course (a line at its height), the DA point where the
 *    glidepath reaches the decision height and the entry point at the FAF — every coordinate the exporter's, none computed
 *    here. The runway in force at the cursor's row in the sentence on screen (its own runway words, not the observed flight's) is drawn in the designated colour, the others in the candidates' grey.
 *  • THE FLIGHT'S OTHER SENTENCES (`otherSentences` switch): the flown path of every sentence of the flight but the one on
 *    screen — the prior's other samples in their outcome's colour, the closed-loop sentence in the executor's teal — thin and
 *    faded, so the sentences can be read side by side.
 *
 * Static entities, never time-sampled (`viewer.clock` belongs to Evaluation). Called from a leaf (`TrainingPriorScene`).
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp, useTrainingCursor } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { useTrainingPriorLayers } from "../data/trainingPriorLayers";
import { trainingPriorOriginOf, type TrainingPriorOrigin } from "../data/trainingPriorSample";
import { runwayInForce } from "../data/trainingProcedure";
import { drawProcedureLimits } from "../scene/trainingProcedure";
import { TRAINING_SENTENCE_COLOR, trainingOutcomeColour } from "../utils/trainingWordColors";
import { TRAINING_OUTCOME_TAG } from "../data/trainingText";
import { airLine, entityGroup, lonLatHeights } from "../scene/trainingEntities";

const ENTITY = {
  sentence: (key: string) => `training-prior-sentence-${key}`,
};

function drawOtherSentences(viewer: Cesium.Viewer, origin: TrainingPriorOrigin) {
  const group = entityGroup(viewer);
  const closed = origin.flight.head.closedLoop[String(origin.sample.model.rowIntervalS)];
  const sentences = [
    ...(origin.which === "closedLoop" ? [] : [{ key: "closed", name: "Closed-loop sentence", track: closed.flown, hue: TRAINING_SENTENCE_COLOR.closedLoop, tag: TRAINING_OUTCOME_TAG[closed.replay.outcome] }]),
    ...origin.flight.sentences.filter((item) => item.sample !== origin.which).map((item) => ({
      key: String(item.sample), name: `Prior sentence ${item.sample}`, track: item.flown, hue: trainingOutcomeColour(item.outcome),
      tag: TRAINING_OUTCOME_TAG[item.outcome],
    })),
  ];
  for (const item of sentences) {
    group.add(airLine(ENTITY.sentence(item.key), `${item.name}: ${item.tag}`,
      Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(item.track)), item.hue, 1.5, 0.55));
  }
  return group;
}

export default function useTrainingProcedureLayer(): void {
  const { viewer, mode, trainingSelection } = useApp();
  const layers = useTrainingPriorLayers();
  const flight = mode === "training" && trainingSelection !== null ? trainingSelection.flight : null;
  const origin = flight === null ? undefined : trainingPriorOriginOf(flight);
  const { trainingCursorS } = useTrainingCursor();
  // the runway the masks act on: the one in force at the cursor's row in the sentence on screen
  const closed = flight === null || origin === undefined ? null : flight.closedLoop[String(origin.sample.model.rowIntervalS)];
  const designated = closed === null ? -1
    : runwayInForce(closed.events, Math.max(Math.floor((trainingCursorS - closed.startS) / closed.rowIntervalS + 1e-9), 0));

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || origin === undefined || !layers.procedure) return;
    const group = drawProcedureLimits(viewer, origin.sample.procedure, designated);
    return () => group.remove();
  }, [viewer, origin, designated, layers.procedure]);

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || origin === undefined || !layers.otherSentences) return;
    const group = drawOtherSentences(viewer, origin);
    return () => group.remove();
  }, [viewer, origin, layers.otherSentences]);
}
