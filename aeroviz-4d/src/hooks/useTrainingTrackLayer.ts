/**
 * useTrainingTrackLayer.ts
 * ------------------------
 * The selected Training flight in the 3D scene: its observed track, the flown path of the sentence read beside it, and
 * the envelopes the sentence allows. The pieces are `scene/trainingEntities.ts`; the live executor's segment is
 * `useTrainingLiveLayer`, the aircraft at the cursor `useTrainingAircraftLayer`, both called from here.
 *
 *  • THE OBSERVED TRACK, in 3D, at its ellipsoid height (the exporter's MSL plus the flight's runway's HAE − MSL, added
 *    once in the reader), and its GROUND TRACE draped under it: the lateral envelopes lie on the ground, and from any
 *    oblique view the airborne line is displaced from them — the trace is what they are read against.
 *  • THE FLOWN PATH (a closed-loop reading): the states the executor flew from the first predicted step
 *    when it was told the sentence at the chosen Δ, in its kind's colour beside the observed track (frontend §3 item 11:
 *    the closed loop teal, the base's sample magenta, a post-trained round yellow-green), its ground trace dashed, where
 *    it ended — named with the judge's outcome — and the DA POINT of the threshold crossing, green when the check
 *    passed, red when it did not (its values in the label).
 *  • THE CORRECTION WORDS: an orange point on the flown path where each word the closed-loop reading added is said.
 *  • THE HEADING WORDS (`headingBands`): a heading word bounds no position — it says where the track is a lead after it
 *    is said, and is judged on the rows from then on — so what is honest in plan is the stretch of ground trace it is
 *    judged on, draped in the band colour, one entity per word, and the rows of that stretch outside the band drawn over
 *    it in red, never faded. The envelopes are the judged track's: the observed track's for the labelled sentence, the
 *    flown path's for a closed-loop one.
 *  • THE ALTITUDE TUBES (`vertical`): one Cesium wall per altitude word, over the judged track's ground position, between
 *    the tube's lower and upper edge (exported MSL plus the flight's HAE − MSL), and the two edges as lines.
 *  • THE CANDIDATE RUNWAYS (`candidates`): every threshold the runway word can point at, as a point with its name; the one
 *    the flight lands on is drawn whatever the switch says. (The runway itself is the airport's own layer.)
 *
 * BUILT ONCE PER FLIGHT AND READING: a Draw switch shows or hides its entities and rebuilds nothing — the draped layers
 * would otherwise be re-draped on every switch.
 *
 * THE SELECTED WORD (`trainingColumn`, its word in force at the cursor) is the only thing highlighted: its own envelope
 * turns yellow (a line) or keeps its hue deepened with a yellow edge (a fill); the rows it is in force are drawn yellow
 * over the track, and its issue is marked with its name. Every other word's envelope recedes (its colours faded). Moving
 * the cursor within one word repaints nothing. Selecting a flight frames it once; the cursor never moves the camera.
 * It reads the cursor, which moves on every hover over a chart: call it from a leaf (`TrainingScene`), never from the
 * app shell, or every hover re-renders the whole workbench.
 *
 * STATIC ENTITIES, NOT TIME-SAMPLED ONES: a time-dynamic entity would drive the shared `viewer.clock`, which belongs to
 * Evaluation's playback.
 */

import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import * as Cesium from "cesium";
import { useApp, useTrainingCursor, type TrainingLayers } from "../context/AppContext";
import useTrainingAircraftLayer from "./useTrainingAircraftLayer";
import useTrainingLiveLayer from "./useTrainingLiveLayer";
import { flownSentenceColour, flownSentenceKind, TRAINING_SENTENCE_KIND_TEXT } from "../data/trainingSentenceKind";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { frameTrajectoryCamera } from "../utils/frameTrajectoryCamera";
import {
  TRAINING_CANDIDATE_COLOR,
  TRAINING_CORRECTION_COLOR,
  TRAINING_DECISION_FAIL_COLOR,
  TRAINING_DECISION_PASS_COLOR,
  TRAINING_DESIGNATED_COLOR,
  TRAINING_ENVELOPE_ALPHA,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import {
  readingRowAt,
  readingRowTimeS,
  rowAtTime,
  sentenceWordAt,
  trainingBandLabel,
  trainingReadingOf,
  trainingWordLabel,
  type TrainingColumn,
  type TrainingReading,
  type TrainingSelection,
} from "../data/trainingSample";
import { decisionText, TRAINING_OUTCOME_TAG } from "../data/trainingText";
import {
  airLine,
  colour,
  entityGroup,
  groundLine,
  lonLatHeights,
  marker,
  planDegrees,
  TRAINING_ENTITY,
  trainingBandGround,
  trainingBandOutsideGround,
  trainingEnvelopeEntities,
  trainingFocusEntity,
  trainingFocusStretch,
  trainingTubeWall,
  type EntityOptions,
} from "../scene/trainingEntities";

const ALPHA = TRAINING_ENVELOPE_ALPHA;
/** An envelope's edge, at rest and when it is the selected word's (px). */
const EDGE_WIDTH = 1.5;
const EDGE_SELECTED_WIDTH = 3;
/** How much of its colour's opacity another word's envelope keeps while a word is selected. */
const FADED = 0.3;
/** How much wider than the track the framed view is: the sentence bar and the dock cover a third of the canvas. */
const FRAME_MARGIN = 1.5;

type LayerIds = Record<keyof TrainingLayers, string[]>;
const noLayerIds = (): LayerIds => ({ headingBands: [], vertical: [], candidates: [] });

/** Every entity of the flight's scene at one reading, added once; the ids each Draw switch shows and hides. */
function buildScene(viewer: Cesium.Viewer, selection: TrainingSelection, reading: TrainingReading): { layerIds: LayerIds; remove: () => void } {
  const { flight, candidates } = selection;
  const { observed, judged, envelopes, closed } = reading;
  const group = entityGroup(viewer);
  const layerIds = noLayerIds();
  const add = (layer: keyof TrainingLayers | null, options: EntityOptions) => {
    group.add(options);
    if (layer !== null) layerIds[layer].push(options.id);
  };
  const verdict = (ok: boolean, css: string) => (ok ? css : TRAINING_OUTSIDE_COLOR);

  // THE RUNWAYS' THRESHOLDS first; the one the flight lands on whatever the switch says.
  for (const candidate of candidates) {
    const designated = candidate.index === flight.runwayIndex;
    add(designated ? null : "candidates", marker(TRAINING_ENTITY.candidate(candidate.ident),
      `${designated ? "The runway the flight lands on" : "Candidate runway"} ${candidate.ident}: course ${candidate.courseDeg.toFixed(1)}°`,
      Cesium.Cartesian3.fromDegrees(candidate.lonDeg, candidate.latDeg, candidate.elevationM + candidate.haeMinusMslM),
      designated ? TRAINING_DESIGNATED_COLOR : TRAINING_CANDIDATE_COLOR, designated ? 10 : 7, candidate.ident));
  }

  if (envelopes !== null) {
    envelopes.heading.forEach((band, index) => {
      const rows = trainingBandGround(judged, band);
      if (!rows.length) return;
      add("headingBands", groundLine(TRAINING_ENTITY.heading(index), `Heading word ${index + 1}: the rows it is judged on`,
        rows, TRAINING_HEADING_BAND_COLOR));
      trainingBandOutsideGround(judged, band).forEach((degrees, run) =>
        add("headingBands", groundLine(TRAINING_ENTITY.headingOutside(index, run), `Heading word ${index + 1}: rows outside its band`,
          degrees, TRAINING_OUTSIDE_COLOR)));
    });

    envelopes.altitude.forEach((tube, index) => {
      const id = TRAINING_ENTITY.tube(index);
      const wall = trainingTubeWall(judged, tube, flight.haeMinusMslM);
      const edgeCss = verdict(tube.contained, TRAINING_TUBE_COLOR);
      if (wall.minimumHeights.length < 2) {
        // A tube of ONE row has no length to be a wall along: its extent is a vertical segment.
        if (wall.minimumHeights.length === 1) {
          add("vertical", {
            id, name: `Altitude tube ${index + 1}`,
            polyline: {
              positions: Cesium.Cartesian3.fromDegreesArrayHeights([
                wall.positions[0], wall.positions[1], wall.minimumHeights[0],
                wall.positions[0], wall.positions[1], wall.maximumHeights[0],
              ]),
              width: 3, material: colour(edgeCss, 0.8),
            },
          });
        }
        return;
      }
      add("vertical", {
        id, name: `Altitude tube ${index + 1}`,
        wall: {
          positions: Cesium.Cartesian3.fromDegreesArray(wall.positions), minimumHeights: wall.minimumHeights,
          maximumHeights: wall.maximumHeights, material: colour(TRAINING_TUBE_COLOR, ALPHA.tube), outline: false,
        },
      });
      for (const side of ["upper", "lower"] as const) {
        const heights = side === "upper" ? wall.maximumHeights : wall.minimumHeights;
        add("vertical", {
          id: TRAINING_ENTITY.edge(id, side),
          polyline: {
            positions: Cesium.Cartesian3.fromDegreesArrayHeights(
              heights.flatMap((height, offset) => [wall.positions[offset * 2], wall.positions[offset * 2 + 1], height])),
            width: EDGE_WIDTH, material: colour(edgeCss, 0.9),
          },
        });
      }
    });
  }

  // The observed track's plan position, on the ground with the envelopes that bound it; the track in the air.
  add(null, groundLine(TRAINING_ENTITY.observedGround, "The observed track's ground trace", planDegrees(observed),
    TRAINING_TRACE_COLOR, true));
  add(null, airLine(TRAINING_ENTITY.observedTrack, "The observed track",
    Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(observed)), TRAINING_TRACE_COLOR, 3));

  if (closed !== null) {
    const { flown, replay } = closed;
    // its kind's colour (frontend §3 item 11): the closed loop, the base's sample, a post-trained round
    const hue = flownSentenceColour(flight);
    add(null, groundLine(TRAINING_ENTITY.flownGround, "The flown path's ground trace", planDegrees(flown), hue, true));
    add(null, airLine(TRAINING_ENTITY.flownTrack, `The flown path: ${TRAINING_SENTENCE_KIND_TEXT[flownSentenceKind(flight)]}`,
      Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(flown)), hue, 3));
    const end = flown.lon.length - 1;
    add(null, marker(TRAINING_ENTITY.flownEnd, "Where the flown path ends",
      Cesium.Cartesian3.fromDegrees(flown.lon[end], flown.lat[end], flown.altitudeHaeM[end]), hue, 9,
      `flown: ${TRAINING_OUTCOME_TAG[replay.outcome]}`,
      // above and to the LEFT of the end, right-aligned: the DA label sits above-right, the runway designators below
      new Cesium.Cartesian2(-12, -30), Cesium.HorizontalOrigin.RIGHT));
    const decision = replay.crossing?.decision ?? null;
    if (decision !== null) {
      add(null, marker(TRAINING_ENTITY.decision, `The decision-altitude point: ${decisionText(decision)}`,
        Cesium.Cartesian3.fromDegrees(decision.lonDeg, decision.latDeg, decision.heightMslM + flight.haeMinusMslM),
        decision.passed ? TRAINING_DECISION_PASS_COLOR : TRAINING_DECISION_FAIL_COLOR, 12,
        `DA ${decision.passed ? "✓" : "✗"} · ${Math.abs(decision.aboveGlidepathM).toFixed(0)} m ` +
        `${decision.aboveGlidepathM >= 0 ? "above" : "below"} · ${Math.abs(decision.rightM).toFixed(0)} m ` +
        `${decision.rightM >= 0 ? "right" : "left"}`,
        // above and to the right of the point: the runway designators are labelled below theirs
        new Cesium.Cartesian2(14, -34)));
    }
    // the words the reading added, where they were said on the flown path
    reading.events.filter((event) => event.correction).forEach((event, index) => {
      const point = rowAtTime(flown.tS, readingRowTimeS(reading, event.row));
      add(null, marker(TRAINING_ENTITY.correction(index),
        `Correction: ${event.says.column} ${trainingBandLabel(event.says)} added by the closed-loop reading at ${readingRowTimeS(reading, event.row)} s`,
        Cesium.Cartesian3.fromDegrees(flown.lon[point], flown.lat[point], flown.altitudeHaeM[point]), TRAINING_CORRECTION_COLOR, 8));
    });
  }
  return { layerIds, remove: group.remove };
}

/** THE SELECTED WORD: its own envelope yellow (or its fill deepened, its edge yellow), every other word's faded, the
 *  rows it is in force yellow over the track and its issue named. Returns the undo. */
function paintFocus(
  viewer: Cesium.Viewer, selection: TrainingSelection, reading: TrainingReading, column: TrainingColumn, wordRow: number,
): () => void {
  const run = sentenceWordAt(reading, column, wordRow)!;
  const time = Cesium.JulianDate.now();
  const selectedEdge = colour(TRAINING_WORD_COLOR);
  type Graphics = Cesium.PolygonGraphics | Cesium.WallGraphics | Cesium.PolylineGraphics;
  const restore: Array<() => void> = [];
  const repaint = (graphics: Graphics, material: Cesium.MaterialProperty, width?: number) => {
    const saved = graphics.material;
    const savedWidth = graphics instanceof Cesium.PolylineGraphics ? graphics.width : undefined;
    graphics.material = material;
    if (width !== undefined && graphics instanceof Cesium.PolylineGraphics) graphics.width = new Cesium.ConstantProperty(width);
    restore.push(() => {
      graphics.material = saved;
      if (graphics instanceof Cesium.PolylineGraphics) graphics.width = savedWidth;
    });
  };
  const recolour = (graphics: Cesium.PolylineGraphics, colourOf: (own: Cesium.Color) => Cesium.Color, width?: number) => {
    const own = (graphics.material.getValue(time) as { color: Cesium.Color }).color;
    const material = graphics.material;
    repaint(graphics, material instanceof Cesium.PolylineDashMaterialProperty
      ? new Cesium.PolylineDashMaterialProperty({ color: colourOf(own), dashLength: material.dashLength, dashPattern: material.dashPattern })
      : new Cesium.ColorMaterialProperty(colourOf(own)), width);
  };
  const edges = (id: string) => [TRAINING_ENTITY.edge(id), TRAINING_ENTITY.edge(id, "upper"), TRAINING_ENTITY.edge(id, "lower")];

  const focusId = trainingFocusEntity(reading, selection.vocabulary.stepS, column, run);
  const mine = new Set(focusId === null ? [] : [focusId]);
  // Every other word's envelope recedes: its fill and its edges keep their hue.
  for (const id of trainingEnvelopeEntities(reading)) {
    if (mine.has(id)) continue;
    for (const part of [id, ...edges(id)]) {
      const entity = viewer.entities.getById(part);
      if (!entity) continue;
      const fill = entity.polygon ?? entity.wall;
      if (fill) {
        const colourNow = (fill.material.getValue(time) as { color: Cesium.Color }).color;
        repaint(fill, new Cesium.ColorMaterialProperty(colourNow.withAlpha(colourNow.alpha * FADED)));
      } else if (entity.polyline) {
        recolour(entity.polyline, (colourNow) => colourNow.withAlpha(colourNow.alpha * FADED));
      }
    }
  }
  for (const id of mine) {
    const entity = viewer.entities.getById(id);
    if (!entity) continue;
    const fill = entity.polygon ?? entity.wall;
    if (fill) {
      // Its own hue, deepened: the tube stays violet, so the word still reads.
      const own = (fill.material.getValue(time) as { color: Cesium.Color }).color;
      repaint(fill, new Cesium.ColorMaterialProperty(own.withAlpha(ALPHA.selected)));
    } else if (entity.polyline) {
      recolour(entity.polyline, () => selectedEdge);
    }
    for (const edgeId of edges(id)) {
      const edge = viewer.entities.getById(edgeId)?.polyline;
      if (edge) recolour(edge, () => selectedEdge, EDGE_SELECTED_WIDTH);
    }
  }

  const group = entityGroup(viewer);
  const { judged } = reading;
  const fromS = readingRowTimeS(reading, run.row);
  const stretch = trainingFocusStretch(judged, fromS, readingRowTimeS(reading, run.endRow));
  if (stretch.length) {
    group.add({
      id: TRAINING_ENTITY.focusStretch, name: "Where the selected word is in force",
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArrayHeights(stretch), width: 7, material: colour(TRAINING_WORD_COLOR, 0.85),
        depthFailMaterial: new Cesium.PolylineDashMaterialProperty({ color: colour(TRAINING_WORD_COLOR, 0.5) }),
      },
    });
  }
  const point = rowAtTime(judged.tS, fromS);
  group.add({
    id: TRAINING_ENTITY.focusIssue, name: "The selected word, where it was said",
    position: Cesium.Cartesian3.fromDegrees(judged.lon[point], judged.lat[point], judged.altitudeHaeM[point]),
    point: {
      pixelSize: 13, color: selectedEdge, outlineColor: Cesium.Color.BLACK.withAlpha(0.7), outlineWidth: 2,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
    label: {
      text: `${column} ${trainingWordLabel(run.event.says)}${run.event.correction ? " (correction)" : ""}`,
      font: "600 13px sans-serif", fillColor: selectedEdge, outlineColor: Cesium.Color.BLACK, outlineWidth: 3,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE, pixelOffset: new Cesium.Cartesian2(0, -20),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
  });
  return () => {
    for (const undo of restore) undo();
    group.remove();
  };
}

export default function useTrainingTrackLayer(): void {
  const { viewer, mode, trainingSelection, trainingLayers, trainingColumn, trainingIntervalS } = useApp();
  const { trainingCursorS } = useTrainingCursor();
  const selection = mode === "training" ? trainingSelection : null;
  const reading = useMemo(
    () => (selection === null ? null : trainingReadingOf(selection.flight, selection.vocabulary.stepS, trainingIntervalS)),
    [selection, trainingIntervalS]);

  // The flight's scene, built once per flight and reading; the ids each switch shows, for the effect below.
  const layerIds = useRef<LayerIds>(noLayerIds());
  const switches = useRef(trainingLayers);
  switches.current = trainingLayers;
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || selection === null || reading === null) return;
    const scene = buildScene(viewer, selection, reading);
    layerIds.current = scene.layerIds;
    // the switches as they stand now (this effect runs again only for a new flight or reading)
    for (const layer of Object.keys(scene.layerIds) as Array<keyof TrainingLayers>) {
      for (const id of scene.layerIds[layer]) {
        const entity = viewer.entities.getById(id);
        if (entity) entity.show = switches.current[layer];
      }
    }
    return () => {
      layerIds.current = noLayerIds();
      scene.remove();
    };
  }, [viewer, selection, reading]);

  // THE DRAW SWITCHES show and hide what they name.
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer)) return;
    for (const layer of Object.keys(layerIds.current) as Array<keyof TrainingLayers>) {
      for (const id of layerIds.current[layer]) {
        const entity = viewer.entities.getById(id);
        if (entity) entity.show = trainingLayers[layer];
      }
    }
  }, [viewer, selection, reading, trainingLayers]);

  // A new FLIGHT is framed ONCE — a flight can lie tens of kilometres from the airport view. Coming back to Training frames
  // it again (another task moved the camera).
  const frameKey = selection === null ? null : `${selection.airport}/${selection.setId}/${selection.flight.flightKey}`;
  const framing = useRef(selection);
  useLayoutEffect(() => {
    framing.current = selection;
  });
  useEffect(() => {
    const shown = framing.current;
    if (!isCesiumViewerUsable(viewer) || shown === null) return;
    const { observed } = shown.flight;
    frameTrajectoryCamera(viewer, observed.lon.map((value, row) => ({ lon: value, lat: observed.lat[row], altM: observed.altitudeHaeM[row] })),
      { margin: FRAME_MARGIN });
  }, [viewer, frameKey]);

  // THE SELECTED WORD: the column's word in force at the cursor. Keyed on the word's issue row, so a cursor moving inside
  // one word repaints nothing.
  const focusRow = useMemo(() => {
    if (reading === null || trainingColumn === null) return null;
    return sentenceWordAt(reading, trainingColumn, readingRowAt(reading, trainingCursorS))?.row ?? null;
  }, [reading, trainingColumn, trainingCursorS]);
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || selection === null || reading === null || trainingColumn === null || focusRow === null) return;
    const undo = paintFocus(viewer, selection, reading, trainingColumn, focusRow);
    return () => {
      if (isCesiumViewerUsable(viewer)) undo();
    };
  }, [viewer, selection, reading, trainingColumn, focusRow]);

  useTrainingLiveLayer();
  useTrainingAircraftLayer();
}
