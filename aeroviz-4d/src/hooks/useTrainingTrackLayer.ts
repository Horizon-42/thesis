/**
 * useTrainingTrackLayer.ts
 * ------------------------
 * The selected Training flight in the 3D scene: its track, and the envelopes its sentence allows. Design:
 * `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`. The pieces are `scene/trainingEntities.ts`; the executor's
 * replay and the live executor are `useTrainingExecutorLayers`, a model's own sentence `useTrainingGenerationLayers`,
 * both called from here. The truth — this track and its sentence's envelopes — is drawn whichever sentence is read.
 *
 *  • THE TRACK, in 3D, at its ellipsoid height (the exporter converted MSL once: h = H + N), and its GROUND TRACE draped
 *    under it: the lateral envelopes lie on the ground, and from any oblique view the airborne line is displaced from
 *    them — the trace is what they are read against.
 *  • THE HEADING WORDS (`headingBands`, instruction-v3): a heading word bounds no position — it says where the track is a
 *    lead after it is said, and is judged on the rows from then to the next heading word's — so what is honest in plan
 *    is the stretch of ground trace it is judged on, draped in the band colour, one entity per word, and the rows of that
 *    stretch outside the band drawn over it in red, never faded. The band itself — target ± tolerance against time — is
 *    the read-back window's heading chart.
 *  • THE CAPTURE (`corridor`): the capture turn's rows on the ground, dashed, in the turn's verdict colour; the capture
 *    corridor and its centreline, a draped polygon with a draped edge.
 *  • THE ALTITUDE TUBES (`vertical`): one Cesium wall per altitude word, over the aircraft's own ground track, between
 *    the tube's lower and upper edge (both HAE, exported), and the two edges as lines — ±25 m is a sliver under the track
 *    from any distance; the lines are not.
 *  • THE CANDIDATE RUNWAYS (`candidates`): every threshold the runway pointer can point at, its runway and extended
 *    centreline; the designated one is drawn whatever the switch says, because the corridor and the landing are measured
 *    from it.
 *  • WHERE WORDS WERE ISSUED: a point on the track at every heading word, the clearance, the capture and the end of the
 *    sentence.
 *
 * BUILT ONCE PER FLIGHT: a Draw switch shows or hides its entities and rebuilds nothing — the draped layers would
 * otherwise be re-draped on every switch, and the runways drawn first would land on top of the envelopes they sit under.
 *
 * THE SELECTED WORD (`trainingColumn`, its word in force at the cursor — of the truth; a model's sentence read paints its
 * own, `useTrainingGenerationLayers`) is the only thing highlighted: its own envelope
 * turns yellow (a line) or keeps its hue deepened with a yellow edge (a fill); the rows it is in force are drawn yellow
 * over the track, and its issue is marked with its name. Every other word's envelope recedes (its colours faded). Moving
 * the cursor within one word repaints nothing. Selecting a flight frames it once; the cursor never moves the camera.
 * It reads the cursor, which moves on every hover over a chart: call it from a leaf (`TrainingScene`), never from the
 * app shell, or every hover re-renders the whole workbench.
 *
 * STATIC ENTITIES, NOT TIME-SAMPLED ONES: a time-dynamic entity would drive the shared `viewer.clock`, which belongs to
 * Observe's playback.
 */

import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp, useTrainingCursor, type TrainingLayers } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { frameTrajectoryCamera } from "../utils/frameTrajectoryCamera";
import {
  TRAINING_CANDIDATE_COLOR,
  TRAINING_CAPTURE_TURN_COLOR,
  TRAINING_COLUMN_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_DESIGNATED_COLOR,
  TRAINING_ENVELOPE_ALPHA,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import {
  rowAtTime,
  trainingWordAt,
  trainingWordLabel,
  type TrainingColumn,
  type TrainingPlanLine,
  type TrainingSelection,
} from "../data/trainingSample";
import {
  airLine,
  colour,
  dash,
  entityGroup,
  GROUND_ROWS_WIDTH,
  groundLine,
  groundRows,
  lonLatHeights,
  marker,
  planDegrees,
  planRingDegrees,
  TRAINING_ENTITY,
  trainingBandGround,
  trainingBandOutsideGround,
  trainingEnvelopeEntities,
  trainingFocusEntities,
  trainingFocusStretch,
  trainingTubeWall,
  type EntityOptions,
} from "../scene/trainingEntities";
import useTrainingExecutorLayers from "./useTrainingExecutorLayers";
import useTrainingGenerationLayers from "./useTrainingGenerationLayers";
import { generationOnScreen } from "../data/trainingOverlays";

const ALPHA = TRAINING_ENVELOPE_ALPHA;
/** An envelope's edge, at rest and when it is the selected word's (px). */
const EDGE_WIDTH = 1.5;
const EDGE_SELECTED_WIDTH = 3;
/** How much of its colour's opacity another word's envelope keeps while a word is selected. */
const FADED = 0.3;
/** How much wider than the track the framed view is: the sentence bar and the dock cover a third of the canvas. */
const FRAME_MARGIN = 1.5;

type LayerIds = Record<keyof TrainingLayers, string[]>;
const noLayerIds = (): LayerIds => ({ headingBands: [], corridor: [], vertical: [], candidates: [] });

/** Every entity of the flight's scene, added once; the ids each Draw switch shows and hides. */
function buildScene(viewer: Cesium.Viewer, selection: TrainingSelection): { layerIds: LayerIds; remove: () => void } {
  const { flight, candidates } = selection;
  const { envelopes, signals } = flight;
  const group = entityGroup(viewer);
  const layerIds = noLayerIds();
  const add = (layer: keyof TrainingLayers | null, options: EntityOptions) => {
    group.add(options);
    if (layer !== null) layerIds[layer].push(options.id);
  };
  /** A draped region and its edge. */
  const region = (layer: keyof TrainingLayers, id: string, line: TrainingPlanLine, css: string, alpha: number, name: string) => {
    add(layer, {
      id, name,
      polygon: {
        hierarchy: new Cesium.PolygonHierarchy(Cesium.Cartesian3.fromDegreesArray(planDegrees(line))),
        material: colour(css, alpha), classificationType: Cesium.ClassificationType.BOTH,
      },
    });
    add(layer, groundLine(TRAINING_ENTITY.edge(id), undefined, planRingDegrees(line), EDGE_WIDTH, colour(css)));
  };
  const verdict = (ok: boolean, css: string) => (ok ? css : TRAINING_OUTSIDE_COLOR);

  // THE RUNWAYS first, so every envelope reads over them; the designated one whatever the switch says.
  for (const candidate of candidates) {
    const pointed = candidate.index === flight.runwayIndex;
    const layer = pointed ? null : "candidates";
    const css = pointed ? TRAINING_DESIGNATED_COLOR : TRAINING_CANDIDATE_COLOR;
    add(layer, groundLine(TRAINING_ENTITY.centreline(candidate.ident), undefined, planDegrees(candidate.centreline),
      pointed ? 2 : 1.5, dash(css)));
    add(layer, groundLine(TRAINING_ENTITY.runway(candidate.ident), undefined, planDegrees(candidate.runway), pointed ? 7 : 5,
      colour(css)));
    add(layer, {
      id: TRAINING_ENTITY.runwayLabel(candidate.ident),
      position: Cesium.Cartesian3.fromDegrees(candidate.runway.lon[0], candidate.runway.lat[0]),
      label: {
        text: candidate.ident, font: "13px sans-serif", fillColor: colour(css),
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND, pixelOffset: new Cesium.Cartesian2(0, -14),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
  }

  region("corridor", TRAINING_ENTITY.corridor, envelopes.approach.corridor.outline, TRAINING_CORRIDOR_COLOR, ALPHA.corridor,
    "The capture corridor");
  add("corridor", groundLine(TRAINING_ENTITY.corridorAxis, undefined, planDegrees(envelopes.approach.corridor.axis), 2,
    colour(TRAINING_CORRIDOR_COLOR)));
  const capture = envelopes.approach.captureTurn;
  const turnRows = capture === null ? [] : groundRows(signals.lon, signals.lat, capture.startRow, capture.endRow);
  if (capture !== null && turnRows.length) {
    add("corridor", groundLine(TRAINING_ENTITY.captureTurn, "The capture turn", turnRows, GROUND_ROWS_WIDTH,
      dash(verdict(capture.check.progressOk && capture.check.rateOk, TRAINING_CAPTURE_TURN_COLOR), ALPHA.captureTurn)));
  }

  envelopes.heading.forEach((item, index) => {
    const rows = trainingBandGround(signals.lon, signals.lat, item);
    if (!rows.length) return;
    add("headingBands", groundLine(TRAINING_ENTITY.heading(index), `Heading word ${index + 1}: the rows it is judged on`,
      rows, GROUND_ROWS_WIDTH, colour(TRAINING_HEADING_BAND_COLOR, ALPHA.headingBand)));
    trainingBandOutsideGround(signals.lon, signals.lat, item).forEach((degrees, run) =>
      add("headingBands", groundLine(TRAINING_ENTITY.headingOutside(index, run), `Heading word ${index + 1}: rows outside its band`,
        degrees, GROUND_ROWS_WIDTH, colour(TRAINING_OUTSIDE_COLOR))));
  });

  envelopes.altitude.forEach((tube, index) => {
    const id = TRAINING_ENTITY.tube(index);
    const wall = trainingTubeWall(flight, tube);
    const edgeCss = verdict(tube.check.contained, TRAINING_TUBE_COLOR);
    if (wall.minimumHeights.length < 2) {
      // A tube of ONE row has no length to be a wall along: its extent is a vertical segment.
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

  // The track's plan position, on the ground with the envelopes that bound it; the track in the air.
  add(null, groundLine(TRAINING_ENTITY.groundTrace, "The track's ground trace", planDegrees(signals), 2,
    colour(TRAINING_TRACE_COLOR, 0.55)));
  add(null, airLine(TRAINING_ENTITY.track, "The observed track",
    Cesium.Cartesian3.fromDegreesArrayHeights(lonLatHeights(signals)), TRAINING_TRACE_COLOR, 3));
  const at = (row: number) => Cesium.Cartesian3.fromDegrees(signals.lon[row], signals.lat[row], signals.altitudeHaeM[row]);
  envelopes.heading.forEach((item, index) => add(null,
    marker(TRAINING_ENTITY.issue(index), `Heading word ${index + 1} issued`, at(item.row), TRAINING_COLUMN_COLOR.heading, 7)));
  add(null, marker(TRAINING_ENTITY.clearance, "Cleared to join the final", at(flight.joinRow), TRAINING_COLUMN_COLOR.approach, 10));
  add(null, marker(TRAINING_ENTITY.capture, "The final captured", at(flight.captureRow), TRAINING_CORRIDOR_COLOR, 10));
  add(null, marker(TRAINING_ENTITY.end, "The end of the sentence", at(flight.rows - 1), TRAINING_TRACE_COLOR, 8));
  return { layerIds, remove: group.remove };
}

/** THE SELECTED WORD: its own envelope yellow (or its fill deepened, its edge yellow), every other word's faded, the
 *  rows it is in force yellow over the track and its issue named. Returns the undo. */
function paintFocus(viewer: Cesium.Viewer, selection: TrainingSelection, column: TrainingColumn, focusRow: number): () => void {
  const { flight, vocabulary, candidates } = selection;
  const word = trainingWordAt(flight, column, focusRow);
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

  const mine = new Set(trainingFocusEntities(selection, column, word));
  // Every other word's envelope recedes: its fill and its edges keep their hue.
  for (const id of trainingEnvelopeEntities(flight)) {
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
      // Its own hue, deepened: the corridor stays green and the tube violet, so the word still reads.
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
  const stretch = trainingFocusStretch(flight, word);
  if (stretch.length) {
    group.add({
      id: TRAINING_ENTITY.focusStretch, name: "Where the selected word is in force",
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArrayHeights(stretch), width: 7, material: colour(TRAINING_WORD_COLOR, 0.85),
        depthFailMaterial: new Cesium.PolylineDashMaterialProperty({ color: colour(TRAINING_WORD_COLOR, 0.5) }),
      },
    });
  }
  const { lon, lat, altitudeHaeM } = flight.signals;
  group.add({
    id: TRAINING_ENTITY.focusIssue, name: "The selected word, where it was issued",
    position: Cesium.Cartesian3.fromDegrees(lon[word.row], lat[word.row], altitudeHaeM[word.row]),
    point: {
      pixelSize: 13, color: selectedEdge, outlineColor: Cesium.Color.BLACK.withAlpha(0.7), outlineWidth: 2,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
    label: {
      text: `${column} ${trainingWordLabel(vocabulary, candidates, column, word.value)} · step ${word.row}`,
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
  const { viewer, mode, trainingSelection, trainingLayers, trainingColumn, trainingGenerations, trainingSource } = useApp();
  const { trainingCursorS } = useTrainingCursor();
  const selection = mode === "training" ? trainingSelection : null;

  // The flight's scene, built once per flight; the ids each switch shows, for the effect below.
  const layerIds = useRef<LayerIds>(noLayerIds());
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || selection === null) return;
    const scene = buildScene(viewer, selection);
    layerIds.current = scene.layerIds;
    return () => {
      layerIds.current = noLayerIds();
      scene.remove();
    };
  }, [viewer, selection]);

  // THE DRAW SWITCHES show and hide what they name.
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer)) return;
    for (const layer of Object.keys(layerIds.current) as Array<keyof TrainingLayers>) {
      for (const id of layerIds.current[layer]) {
        const entity = viewer.entities.getById(id);
        if (entity) entity.show = trainingLayers[layer];
      }
    }
  }, [viewer, selection, trainingLayers]);

  // A newly selected flight is framed ONCE: it can lie tens of kilometres from the airport view. The panel publishes a
  // new selection only for a new flight or a reloaded set, never for a switch.
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || selection === null) return;
    const { lon, lat, altitudeHaeM } = selection.flight.signals;
    frameTrajectoryCamera(viewer, lon.map((value, row) => ({ lon: value, lat: lat[row], altM: altitudeHaeM[row] })),
      { margin: FRAME_MARGIN });
  }, [viewer, selection]);

  // THE SELECTED WORD: the column's word in force at the cursor. Keyed on the word's issue row, so a cursor moving inside
  // one word repaints nothing. Only the truth's: a model's sentence read paints its own word on its own track.
  const modelRead = generationOnScreen(trainingGenerations, trainingSource, selection)?.sentence ?? null;
  const focusRow = selection !== null && trainingColumn !== null && modelRead === null
    ? trainingWordAt(selection.flight, trainingColumn, rowAtTime(selection.flight.signals.tS, trainingCursorS)).row
    : null;
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || selection === null || trainingColumn === null || focusRow === null) return;
    const undo = paintFocus(viewer, selection, trainingColumn, focusRow);
    return () => {
      if (isCesiumViewerUsable(viewer)) undo();
    };
  }, [viewer, selection, trainingColumn, focusRow]);

  useTrainingExecutorLayers();
  useTrainingGenerationLayers();
}
