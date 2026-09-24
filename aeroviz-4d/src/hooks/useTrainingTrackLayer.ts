/**
 * useTrainingTrackLayer.ts
 * ------------------------
 * The selected Training flight in the 3D scene: its track, and the envelopes its sentence allows.
 * Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`.
 *
 *  • THE TRACK, in 3D, at its ellipsoid height (the exporter converted MSL once: h = H + N), and its
 *    GROUND TRACE draped under it: the lateral envelopes lie on the ground, and from any oblique
 *    view the airborne line is displaced from them — the trace is what they are read against.
 *  • THE HEADING WORDS (`headingBands`, instruction-v3): a heading word bounds no position — it says
 *    where the track is a lead after it is said, and is judged on the rows from then to the next
 *    heading word's — so what is honest in plan is the stretch of ground trace it is judged on, draped
 *    in the band colour, one entity per word (a selected word lights up, the rest recede), and the
 *    rows of that stretch outside the band drawn over it in red, never faded. The band itself —
 *    target ± tolerance against time — is the read-back window's heading chart.
 *  • THE CAPTURE (`corridor`): the capture turn's rows on the ground, dashed, in the turn's verdict
 *    colour; the capture corridor and its centreline, a draped polygon with a draped edge.
 *  • THE ALTITUDE TUBES (`trainingLayers.vertical`): one Cesium wall per altitude word, over the
 *    aircraft's own ground track, between the tube's lower and upper edge (both HAE, exported), and
 *    the two edges as lines — ±25 m is a sliver under the track from any distance; the lines are not.
 *  • THE CANDIDATE RUNWAYS (`trainingLayers.candidates`): every threshold the runway pointer can
 *    point at, its runway and extended centreline; the designated one is drawn whatever the
 *    switch says, because the corridor and the landing are measured from it.
 *  • WHERE WORDS WERE ISSUED: a point on the track at every heading word, the clearance, the
 *    capture and the end of the sentence.
 *
 * Every coordinate and every row verdict is the exporter's (lon / lat, heights and bands computed in
 * Python); this hook draws them and computes no geometry.
 *
 * THE SELECTED WORD (`trainingColumn`, its word in force at the cursor) is the only thing
 * highlighted: its own envelope turns yellow (a line) or keeps its hue deepened with a yellow edge
 * (a fill); the rows it is in force are drawn yellow over the track, and its issue is marked with its
 * name. Every other word's envelope recedes (its colours faded). Moving the cursor within one word
 * repaints nothing. Selecting a flight frames it once; the cursor never moves the camera.
 *
 * THE EXECUTOR'S REPLAY (`trainingExecutor`, when the panel publishes it): its flown track at its ellipsoid height in
 * teal, its ground trace dashed and where it ended, named with its outcome, and — with the heading bands on — the
 * flown rows outside the heading word it was told, red on its ground trace (its judge's own row verdicts): entities
 * of their own, so switching it redraws nothing else and never moves the camera.
 *
 * STATIC ENTITIES, NOT TIME-SAMPLED ONES: a time-dynamic entity would drive the shared
 * `viewer.clock`, which belongs to Observe's playback.
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { frameTrajectoryCamera } from "../utils/frameTrajectoryCamera";
import {
  TRAINING_CANDIDATE_COLOR,
  TRAINING_CAPTURE_TURN_COLOR,
  TRAINING_COLUMN_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_DESIGNATED_COLOR,
  TRAINING_ENVELOPE_ALPHA,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import {
  rowAtTime,
  trainingBandOutsideSpans,
  trainingWordAt,
  trainingWordLabel,
  type TrainingAltitudeTube,
  type TrainingColumn,
  type TrainingFlight,
  type TrainingHeadingBand,
  type TrainingPlanLine,
  type TrainingSelection,
  type TrainingWordRun,
} from "../data/trainingSample";
import type { TrainingExecutorFlight, TrainingExecutorTrack } from "../data/trainingOverlays";

export const TRAINING_ENTITY = {
  track: "training-track",
  groundTrace: "training-ground-trace",
  /** A heading word's judged rows on the ground, and its rows outside the band (`run` counts them). */
  heading: (index: number) => `training-heading-${index}`,
  headingOutside: (index: number, run: number) => `training-heading-${index}-outside-${run}`,
  captureTurn: "training-capture-turn",
  corridor: "training-corridor",
  corridorAxis: "training-corridor-axis",
  tube: (index: number) => `training-tube-${index}`,
  /** An envelope's edge: the outline of a draped polygon, or a tube's upper / lower line. */
  edge: (id: string, side?: "upper" | "lower") => (side ? `${id}-edge-${side}` : `${id}-edge`),
  centreline: (ident: string) => `training-centreline-${ident}`,
  runway: (ident: string) => `training-runway-${ident}`,
  runwayLabel: (ident: string) => `training-runway-label-${ident}`,
  issue: (index: number) => `training-issue-${index}`,
  clearance: "training-clearance",
  capture: "training-capture",
  end: "training-end",
  /** The selected word: the rows it is in force, and its issue with its name. */
  focusStretch: "training-focus-stretch",
  focusIssue: "training-focus-issue",
  /** The executor's replay: its flown track, its ground trace, where it ended, and its rows outside a heading word. */
  executorTrack: "training-executor-track",
  executorGround: "training-executor-ground",
  executorEnd: "training-executor-end",
  executorOutside: (run: number) => `training-executor-outside-${run}`,
} as const;

const ALPHA = TRAINING_ENVELOPE_ALPHA;
/** An envelope's edge, at rest and when it is the selected word's (px). */
const EDGE_WIDTH = 1.5;
const EDGE_SELECTED_WIDTH = 3;
/** How much of its colour's opacity another word's envelope keeps while a word is selected. */
const FADED = 0.3;
/** A heading word's judged rows and the capture turn on the ground (px): wider than the ground trace they lie on. */
const GROUND_ROWS_WIDTH = 5;
/** How much wider than the track the framed view is. */
const FRAME_MARGIN = 1.5;

/** Cesium wants [lon, lat, height, …]; the track carries the three as columns. */
export function trainingTrackPositions(flight: TrainingFlight): number[] {
  const { lon, lat, altitudeHaeM } = flight.signals;
  return lon.flatMap((value, row) => [value, lat[row], altitudeHaeM[row]]);
}

/** The executor's flown track as Cesium's [lon, lat, height, …] — at its ellipsoid height, as the exporter wrote it. */
export function executorTrackPositions(track: TrainingExecutorTrack): number[] {
  return track.lon.flatMap((value, index) => [value, track.lat[index], track.altitudeHaeM[index]]);
}

/** A plan line as Cesium's flat [lon, lat, …]. */
export function planDegrees(line: TrainingPlanLine): number[] {
  return line.lon.flatMap((value, point) => [value, line.lat[point]]);
}

/** A region's outline as a closed line: the exporter's rings are open (a polygon closes itself). */
export function planRingDegrees(line: TrainingPlanLine): number[] {
  return [...planDegrees(line), line.lon[0], line.lat[0]];
}

/** One altitude tube as a wall over the aircraft's own ground track: a position per row it
 *  covers, with that row's lower and upper edge (HAE, as exported). */
export function trainingTubeWall(flight: TrainingFlight, tube: TrainingAltitudeTube) {
  const positions: number[] = [];
  for (let row = tube.row; row < tube.endRow; row += 1) positions.push(flight.signals.lon[row], flight.signals.lat[row]);
  return { positions, minimumHeights: tube.lowerHaeM, maximumHeights: tube.upperHaeM };
}

/**
 * Rows ``first..last`` (inclusive) of a line of points as Cesium's flat [lon, lat, …], or nothing when that is fewer
 * than two points (a line needs two).
 */
function groundRows(lon: number[], lat: number[], first: number, last: number): number[] {
  if (last <= first) return [];
  return Array.from({ length: last - first + 1 }, (_, offset) => [lon[first + offset], lat[first + offset]]).flat();
}

/** A heading word's judged rows on the ground: its first judged row on to its stop row, where the next word's
 *  begin, so the words meet; nothing for a word with no row of its own. */
export function trainingBandGround(lon: number[], lat: number[], band: TrainingHeadingBand): number[] {
  if (band.stopRow <= band.firstRow) return [];
  return groundRows(lon, lat, band.firstRow, Math.min(band.stopRow, lon.length - 1));
}

/** The rows a band judged outside, each as a line on the ground (`trainingBandOutsideSpans`: one row outside on to the
 *  next, so it is a segment). */
export function trainingBandOutsideGround(lon: number[], lat: number[], band: TrainingHeadingBand): number[][] {
  return trainingBandOutsideSpans(band, lon.length - 1).map(([first, last]) => groundRows(lon, lat, first, last));
}

/**
 * The entities that ARE a word's envelope — its own column's, and nothing of another column's:
 * a heading word's judged rows; the clearance's capture turn and corridor; an altitude word's tube;
 * the runway pointed at. An angle or a speed word bounds no position, so it owns none: the stretch
 * of track it is in force is its picture. An id the scene does not hold (a word the lead carries to
 * the clearance has no rows; a switch is off) is simply not there to paint.
 */
export function trainingFocusEntities(
  selection: TrainingSelection, column: TrainingColumn, word: TrainingWordRun & { index: number },
): string[] {
  switch (column) {
    case "heading":
      return [TRAINING_ENTITY.heading(word.index)];
    case "approach":
      return word.event.kind === "clear"
        ? [TRAINING_ENTITY.captureTurn, TRAINING_ENTITY.corridor, TRAINING_ENTITY.corridorAxis]
        : [];
    case "altitude":
      return [TRAINING_ENTITY.tube(word.index)];
    case "runway": {
      const ident = selection.candidates[word.value].ident;
      return [TRAINING_ENTITY.runway(ident), TRAINING_ENTITY.centreline(ident)];
    }
    case "angle":
    case "speed":
      return [];
  }
}

/** Every envelope of a flight, by the id of its main entity: the one that `trainingFocusEntities`
 *  names, and whose edges (`TRAINING_ENTITY.edge` of it) go with it. */
export function trainingEnvelopeEntities(flight: TrainingFlight): string[] {
  return [
    ...flight.envelopes.heading.map((_, index) => TRAINING_ENTITY.heading(index)),
    TRAINING_ENTITY.captureTurn, TRAINING_ENTITY.corridor, TRAINING_ENTITY.corridorAxis,
    ...flight.envelopes.altitude.map((_, index) => TRAINING_ENTITY.tube(index)),
  ];
}

/** The track rows a word is in force, as positions: on to the next word's issue, so that it meets
 *  it. Only a word issued on the last row has no stretch (its issue marker is its picture). */
export function trainingFocusStretch(flight: TrainingFlight, word: TrainingWordRun): number[] {
  const last = Math.min(word.endRow, flight.rows - 1);
  if (last <= word.row) return [];
  return trainingTrackPositions(flight).slice(word.row * 3, (last + 1) * 3);
}

const colour = (css: string, alpha = 1) => Cesium.Color.fromCssColorString(css).withAlpha(alpha);

export default function useTrainingTrackLayer(): void {
  const { viewer, mode, trainingSelection, trainingLayers, trainingCursorS, trainingColumn, trainingExecutor } = useApp();

  // THE EXECUTOR'S REPLAY of the selected flight, drawn and removed on its own.
  const executorFlight: TrainingExecutorFlight | null =
    mode === "training" && trainingExecutor?.flight.flightKey === trainingSelection?.flight.flightKey
      ? trainingExecutor?.flight ?? null : null;
  const headingBands = trainingLayers.headingBands;
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || executorFlight === null || executorFlight.track === null) return;
    const track = executorFlight.track;
    const added: string[] = [];
    const add = (options: Cesium.Entity.ConstructorOptions & { id: string }) => {
      added.push(options.id);
      viewer.entities.add(options);
    };
    if (track.lon.length >= 2) {
      add({
        id: TRAINING_ENTITY.executorGround,
        name: "The executor's ground trace",
        polyline: {
          positions: Cesium.Cartesian3.fromDegreesArray(track.lon.flatMap((lon, index) => [lon, track.lat[index]])),
          clampToGround: true,
          width: 2,
          material: new Cesium.PolylineDashMaterialProperty({ color: colour(TRAINING_EXECUTOR_COLOR, 0.6) }),
        },
      });
      add({
        id: TRAINING_ENTITY.executorTrack,
        name: "The executor's flown track",
        polyline: {
          positions: Cesium.Cartesian3.fromDegreesArrayHeights(executorTrackPositions(track)),
          width: 3,
          material: colour(TRAINING_EXECUTOR_COLOR),
          depthFailMaterial: new Cesium.PolylineDashMaterialProperty({ color: colour(TRAINING_EXECUTOR_COLOR, 0.55) }),
        },
      });
    }
    // its flown rows outside the heading word it was told: the judge's own row verdicts, red on its ground trace (the
    // judged steps are the track's first points)
    const judged = executorFlight.judgedTrackDeg?.length ?? 0;
    if (headingBands && judged > 0) {
      const lon = track.lon.slice(0, judged);
      const lat = track.lat.slice(0, judged);
      executorFlight.words
        .flatMap((word) => (word.heading === null ? [] : trainingBandOutsideGround(lon, lat, word.heading)))
        .forEach((degrees, run) => add({
          id: TRAINING_ENTITY.executorOutside(run),
          name: "The executor off the heading word it was told",
          polyline: {
            positions: Cesium.Cartesian3.fromDegreesArray(degrees), clampToGround: true, width: GROUND_ROWS_WIDTH,
            material: colour(TRAINING_OUTSIDE_COLOR),
          },
        }));
    }
    const end = track.lon.length - 1;
    add({
      id: TRAINING_ENTITY.executorEnd,
      name: "Where the executor's flight ended",
      position: Cesium.Cartesian3.fromDegrees(track.lon[end], track.lat[end], track.altitudeHaeM[end]),
      point: {
        pixelSize: 10,
        color: colour(TRAINING_EXECUTOR_COLOR),
        outlineColor: Cesium.Color.BLACK.withAlpha(0.6),
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: `executor: ${(executorFlight.outcome ?? "").replace(/_/g, " ")}`,
        font: "600 12px sans-serif",
        fillColor: colour(TRAINING_EXECUTOR_COLOR),
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 3,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(0, 18),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
    viewer.scene.requestRender();
    return () => {
      if (!isCesiumViewerUsable(viewer)) return;
      for (const id of added) viewer.entities.removeById(id);
      viewer.scene.requestRender();
    };
  }, [viewer, executorFlight, headingBands]);

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer)) return;
    const selection = mode === "training" ? trainingSelection : null;
    if (!selection) return;
    const { flight, candidates } = selection;
    const { envelopes, signals } = flight;
    const added: string[] = [];
    const add = (options: Cesium.Entity.ConstructorOptions & { id: string }) => {
      added.push(options.id);
      viewer.entities.add(options);
    };
    const dash = (css: string, alpha = 0.85) => new Cesium.PolylineDashMaterialProperty({ color: colour(css, alpha) });
    const groundLine = (id: string, degrees: number[], width: number, material: Cesium.Color | Cesium.MaterialProperty,
      name?: string) =>
      add({
        id,
        name,
        polyline: { positions: Cesium.Cartesian3.fromDegreesArray(degrees), clampToGround: true, width, material },
      });
    /** A draped region and its edge. */
    const ground = (id: string, line: TrainingPlanLine, css: string, alpha: number, name: string) => {
      add({
        id,
        name,
        polygon: {
          hierarchy: new Cesium.PolygonHierarchy(Cesium.Cartesian3.fromDegreesArray(planDegrees(line))),
          material: colour(css, alpha),
          classificationType: Cesium.ClassificationType.BOTH,
        },
      });
      groundLine(TRAINING_ENTITY.edge(id), planRingDegrees(line), EDGE_WIDTH, colour(css));
    };
    const verdict = (ok: boolean, css: string) => (ok ? css : TRAINING_OUTSIDE_COLOR);

    // THE RUNWAYS first, so every envelope reads over them.
    for (const candidate of candidates) {
      const pointed = candidate.index === flight.runwayIndex;
      if (!pointed && !trainingLayers.candidates) continue;
      const css = pointed ? TRAINING_DESIGNATED_COLOR : TRAINING_CANDIDATE_COLOR;
      groundLine(TRAINING_ENTITY.centreline(candidate.ident), planDegrees(candidate.centreline), pointed ? 2 : 1.5, dash(css));
      groundLine(TRAINING_ENTITY.runway(candidate.ident), planDegrees(candidate.runway), pointed ? 7 : 5, colour(css));
      add({
        id: TRAINING_ENTITY.runwayLabel(candidate.ident),
        position: Cesium.Cartesian3.fromDegrees(candidate.runway.lon[0], candidate.runway.lat[0]),
        label: {
          text: candidate.ident,
          font: "13px sans-serif",
          fillColor: colour(css),
          heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
          pixelOffset: new Cesium.Cartesian2(0, -14),
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
    }

    const capture = envelopes.approach.captureTurn;
    if (trainingLayers.corridor) {
      ground(TRAINING_ENTITY.corridor, envelopes.approach.corridor.outline, TRAINING_CORRIDOR_COLOR, ALPHA.corridor,
        "The capture corridor");
      groundLine(TRAINING_ENTITY.corridorAxis, planDegrees(envelopes.approach.corridor.axis), 2, colour(TRAINING_CORRIDOR_COLOR));
      const rows = capture === null ? [] : groundRows(signals.lon, signals.lat, capture.startRow, capture.endRow);
      if (capture !== null && rows.length) {
        groundLine(TRAINING_ENTITY.captureTurn, rows, GROUND_ROWS_WIDTH,
          dash(verdict(capture.check.progressOk && capture.check.rateOk, TRAINING_CAPTURE_TURN_COLOR), ALPHA.captureTurn),
          "The capture turn");
      }
    }
    if (trainingLayers.headingBands) {
      envelopes.heading.forEach((item, index) => {
        const rows = trainingBandGround(signals.lon, signals.lat, item);
        if (!rows.length) return;
        groundLine(TRAINING_ENTITY.heading(index), rows, GROUND_ROWS_WIDTH, colour(TRAINING_HEADING_BAND_COLOR, ALPHA.headingBand),
          `Heading word ${index + 1}: the rows it is judged on`);
        trainingBandOutsideGround(signals.lon, signals.lat, item).forEach((degrees, run) =>
          groundLine(TRAINING_ENTITY.headingOutside(index, run), degrees, GROUND_ROWS_WIDTH, colour(TRAINING_OUTSIDE_COLOR),
            `Heading word ${index + 1}: rows outside its band`));
      });
    }

    if (trainingLayers.vertical) {
      envelopes.altitude.forEach((tube, index) => {
        const id = TRAINING_ENTITY.tube(index);
        const wall = trainingTubeWall(flight, tube);
        const edgeCss = verdict(tube.check.contained, TRAINING_TUBE_COLOR);
        if (wall.minimumHeights.length >= 2) {
          add({
            id,
            name: `Altitude tube ${index + 1}`,
            wall: {
              positions: Cesium.Cartesian3.fromDegreesArray(wall.positions),
              minimumHeights: wall.minimumHeights,
              maximumHeights: wall.maximumHeights,
              material: colour(TRAINING_TUBE_COLOR, ALPHA.tube),
              outline: false,
            },
          });
          for (const side of ["upper", "lower"] as const) {
            const heights = side === "upper" ? wall.maximumHeights : wall.minimumHeights;
            add({
              id: TRAINING_ENTITY.edge(id, side),
              polyline: {
                positions: Cesium.Cartesian3.fromDegreesArrayHeights(
                  heights.flatMap((height, offset) => [wall.positions[offset * 2], wall.positions[offset * 2 + 1], height]),
                ),
                width: EDGE_WIDTH,
                material: colour(edgeCss, 0.9),
              },
            });
          }
        } else {
          // A tube of ONE row has no length to be a wall along: its extent is a vertical segment.
          add({
            id,
            name: `Altitude tube ${index + 1}`,
            polyline: {
              positions: Cesium.Cartesian3.fromDegreesArrayHeights([
                wall.positions[0], wall.positions[1], wall.minimumHeights[0],
                wall.positions[0], wall.positions[1], wall.maximumHeights[0],
              ]),
              width: 3,
              material: colour(edgeCss, 0.8),
            },
          });
        }
      });
    }

    // The track's plan position, on the ground with the envelopes that bound it.
    groundLine(TRAINING_ENTITY.groundTrace, signals.lon.flatMap((lon, row) => [lon, signals.lat[row]]), 2,
      colour(TRAINING_TRACE_COLOR, 0.55), "The track's ground trace");
    add({
      id: TRAINING_ENTITY.track,
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArrayHeights(trainingTrackPositions(flight)),
        width: 3,
        material: colour(TRAINING_TRACE_COLOR),
        // A track that runs into a hill is a reading, not a rendering accident: show it dashed.
        depthFailMaterial: new Cesium.PolylineDashMaterialProperty({ color: colour(TRAINING_TRACE_COLOR, 0.55) }),
      },
    });

    const point = (id: string, row: number, css: string, size: number, name: string) =>
      add({
        id,
        name,
        position: Cesium.Cartesian3.fromDegrees(signals.lon[row], signals.lat[row], signals.altitudeHaeM[row]),
        point: {
          pixelSize: size,
          color: colour(css),
          outlineColor: Cesium.Color.BLACK.withAlpha(0.6),
          outlineWidth: 2,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
    envelopes.heading.forEach((item, index) =>
      point(TRAINING_ENTITY.issue(index), item.row, TRAINING_COLUMN_COLOR.heading, 7, `Heading word ${index + 1} issued`));
    point(TRAINING_ENTITY.clearance, flight.joinRow, TRAINING_COLUMN_COLOR.approach, 10, "Cleared to join the final");
    point(TRAINING_ENTITY.capture, flight.captureRow, TRAINING_CORRIDOR_COLOR, 10, "The final captured");
    point(TRAINING_ENTITY.end, flight.rows - 1, TRAINING_TRACE_COLOR, 8, "The end of the sentence");

    return () => {
      if (!isCesiumViewerUsable(viewer)) return;
      for (const id of added) viewer.entities.removeById(id);
    };
  }, [viewer, mode, trainingSelection, trainingLayers]);

  // A newly selected flight is framed ONCE: it can lie tens of kilometres from the airport view. The
  // panel publishes a new selection only for a new flight or a reloaded set, never for a switch.
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || mode !== "training" || !trainingSelection) return;
    const { lon, lat, altitudeHaeM } = trainingSelection.flight.signals;
    frameTrajectoryCamera(viewer, lon.map((value, row) => ({ lon: value, lat: lat[row], altM: altitudeHaeM[row] })),
      { margin: FRAME_MARGIN });  // the sentence bar and the dock cover a third of the canvas
  }, [viewer, mode, trainingSelection]);

  // THE SELECTED WORD: the column's word in force at the cursor. Keyed on the word's issue row, so a
  // cursor moving inside one word repaints nothing.
  const flight = trainingSelection?.flight ?? null;
  const focusRow = flight && trainingColumn !== null
    ? trainingWordAt(flight, trainingColumn, rowAtTime(flight.signals.tS, trainingCursorS)).row
    : null;
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || mode !== "training" || !trainingSelection) return;
    if (trainingColumn === null || focusRow === null) return;
    const { flight: focusFlight, vocabulary, candidates } = trainingSelection;
    const word = trainingWordAt(focusFlight, trainingColumn, focusRow);
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
        ? new Cesium.PolylineDashMaterialProperty({
          color: colourOf(own), dashLength: material.dashLength, dashPattern: material.dashPattern,
        })
        : new Cesium.ColorMaterialProperty(colourOf(own)), width);
    };
    const yellowLine = (graphics: Cesium.PolylineGraphics, width?: number) => recolour(graphics, () => selectedEdge, width);

    const mine = trainingFocusEntities(trainingSelection, trainingColumn, word);
    // Every other word's envelope recedes: its fill and its edges keep their hue.
    const own = new Set(mine);
    for (const id of trainingEnvelopeEntities(focusFlight)) {
      if (own.has(id)) continue;
      for (const part of [id, TRAINING_ENTITY.edge(id), TRAINING_ENTITY.edge(id, "upper"), TRAINING_ENTITY.edge(id, "lower")]) {
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
        yellowLine(entity.polyline);
      }
      for (const edgeId of [TRAINING_ENTITY.edge(id), TRAINING_ENTITY.edge(id, "upper"), TRAINING_ENTITY.edge(id, "lower")]) {
        const edge = viewer.entities.getById(edgeId)?.polyline;
        if (edge) yellowLine(edge, EDGE_SELECTED_WIDTH);
      }
    }

    const added: string[] = [];
    const stretch = trainingFocusStretch(focusFlight, word);
    if (stretch.length) {
      added.push(TRAINING_ENTITY.focusStretch);
      viewer.entities.add({
        id: TRAINING_ENTITY.focusStretch,
        name: "Where the selected word is in force",
        polyline: {
          positions: Cesium.Cartesian3.fromDegreesArrayHeights(stretch),
          width: 7,
          material: colour(TRAINING_WORD_COLOR, 0.85),
          depthFailMaterial: new Cesium.PolylineDashMaterialProperty({ color: colour(TRAINING_WORD_COLOR, 0.5) }),
        },
      });
    }
    const { lon, lat, altitudeHaeM } = focusFlight.signals;
    added.push(TRAINING_ENTITY.focusIssue);
    viewer.entities.add({
      id: TRAINING_ENTITY.focusIssue,
      name: "The selected word, where it was issued",
      position: Cesium.Cartesian3.fromDegrees(lon[word.row], lat[word.row], altitudeHaeM[word.row]),
      point: {
        pixelSize: 13,
        color: selectedEdge,
        outlineColor: Cesium.Color.BLACK.withAlpha(0.7),
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: `${trainingColumn} ${trainingWordLabel(vocabulary, candidates, trainingColumn, word.value)} · step ${word.row}`,
        font: "600 13px sans-serif",
        fillColor: selectedEdge,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 3,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(0, -20),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
    viewer.scene.requestRender();

    return () => {
      if (!isCesiumViewerUsable(viewer)) return;
      for (const undo of restore) undo();
      for (const id of added) viewer.entities.removeById(id);
      viewer.scene.requestRender();
    };
  }, [viewer, mode, trainingSelection, trainingLayers, trainingColumn, focusRow]);
}
