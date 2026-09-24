/**
 * useTrainingTrackLayer.ts
 * ------------------------
 * The selected Training flight in the 3D scene: its track, and the envelopes its sentence allows.
 * Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`.
 *
 *  • THE TRACK, in 3D, at its ellipsoid height (the exporter converted MSL once: h = H + N), and its
 *    GROUND TRACE draped under it: the lateral envelopes lie on the ground, and from any oblique
 *    view the airborne line is displaced from them — the trace is what they are read against.
 *  • THE LATERAL ENVELOPES on the ground (`trainingLayers.lateral`): each heading word's turn
 *    region, where its turn may end (a parallelogram, drawn darker) and its hold funnel, the
 *    capture turn, and the capture corridor with its centreline. They
 *    bound positions in plan only — the vertical is the tubes' business — so they are draped on
 *    the terrain rather than floated at some height the words do not give them. A ground polygon
 *    carries no outline, so each has a draped EDGE styled as the plan view strokes it: a turn region
 *    and a funnel red where the labeller's check failed, a funnel dashed where its hold is not
 *    judged, the capture turn dashed (red if its check failed).
 *  • THE TURN PATHS (`trainingLayers.turnPaths`): the fastest and the slowest turn of every turn
 *    region, draped, in the region's verdict colour (the fastest runs along its edge); the slowest
 *    dashed where it does not finish before the flight ends — as the plan view draws them.
 *  • THE ALTITUDE TUBES (`trainingLayers.vertical`): one Cesium wall per altitude word, over the
 *    aircraft's own ground track, between the tube's lower and upper edge (both HAE, exported), and
 *    the two edges as lines — ±25 m is a sliver under the track from any distance; the lines are not.
 *  • THE CANDIDATE RUNWAYS (`trainingLayers.candidates`): every threshold the runway pointer can
 *    point at, its runway and extended centreline; the designated one is drawn whatever the
 *    switch says, because the corridor and the landing are measured from it.
 *  • WHERE WORDS WERE ISSUED: a point on the track at every heading word, the clearance, the
 *    capture and the end of the sentence.
 *
 * Every coordinate is the exporter's (lon / lat and heights computed in Python); this hook draws
 * them and computes no geometry.
 *
 * THE SELECTED WORD (`trainingColumn`, its word in force at the cursor) is the only thing
 * highlighted: its own envelope keeps its hue, deepened, with a yellow edge; the rows it is in force
 * are drawn yellow over the track, and its issue is marked with its name. Every other word's
 * envelope recedes (its colours faded), so one big turn is not read through another. A selected
 * turn's two paths are named where they end, and a heading word's turn as actually flown (the
 * labeller's departure and arrival rows) is marked on the track. Moving the cursor within one word
 * repaints nothing. Selecting a flight frames it once; the cursor never moves the camera.
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
  TRAINING_COLUMN_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_DESIGNATED_COLOR,
  TRAINING_FUNNEL_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_TURN_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import {
  rowAtTime,
  trainingWordAt,
  trainingWordLabel,
  type TrainingAltitudeTube,
  type TrainingColumn,
  type TrainingFlight,
  type TrainingPlanLine,
  type TrainingSelection,
  type TrainingTurnRegion,
  type TrainingWordRun,
} from "../data/trainingSample";

export const TRAINING_ENTITY = {
  track: "training-track",
  groundTrace: "training-ground-trace",
  turn: (index: number) => `training-turn-${index}`,
  turnEnd: (index: number) => `training-turn-end-${index}`,
  funnel: (index: number) => `training-funnel-${index}`,
  captureTurn: "training-capture-turn",
  /** A turn region's fastest or slowest turn (`id` is the region's). */
  path: (id: string, which: "fast" | "slow") => `${id}-${which}`,
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
  /** The selected turn's paths, named at their ends; its turn as flown, at its two rows. */
  focusPathLabel: (which: "fast" | "slow") => `training-focus-path-${which}`,
  focusFlown: (which: "starts" | "ends") => `training-focus-flown-${which}`,
} as const;

/** How opaque each envelope's fill is at rest, and when it is the selected word's. */
const ALPHA = { turn: 0.16, turnEnd: 0.4, funnel: 0.18, corridor: 0.3, tube: 0.28, selected: 0.45 } as const;
/** An envelope's edge, at rest and when it is the selected word's (px). */
const EDGE_WIDTH = 1.5;
const EDGE_SELECTED_WIDTH = 3;
/** How much of its colour's opacity another word's envelope keeps while a word is selected. */
const FADED = 0.3;
/** A turn path's width (px): above the region's edge, which its fastest turn runs along. */
const PATH_WIDTH = 2;
/** How much wider than the track the framed view is. */
const FRAME_MARGIN = 1.5;

/** Cesium wants [lon, lat, height, …]; the track carries the three as columns. */
export function trainingTrackPositions(flight: TrainingFlight): number[] {
  const { lon, lat, altitudeHaeM } = flight.signals;
  return lon.flatMap((value, row) => [value, lat[row], altitudeHaeM[row]]);
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
 * The entities that ARE a word's envelope — its own column's, and nothing of another column's:
 * a heading word's turn region, turn end and funnel; the clearance's capture turn and corridor;
 * an altitude word's tube; the runway pointed at. An angle or a speed word bounds no position, so
 * it owns none: the stretch of track it is in force is its picture. An id the scene does not hold
 * (a word held at entry has no turn; a switch is off) is simply not there to paint.
 */
export function trainingFocusEntities(
  selection: TrainingSelection, column: TrainingColumn, word: TrainingWordRun & { index: number },
): string[] {
  switch (column) {
    case "heading": {
      const turn = TRAINING_ENTITY.turn(word.index);
      return [turn, TRAINING_ENTITY.path(turn, "fast"), TRAINING_ENTITY.path(turn, "slow"),
        TRAINING_ENTITY.turnEnd(word.index), TRAINING_ENTITY.funnel(word.index)];
    }
    case "approach": {
      const turn = TRAINING_ENTITY.captureTurn;
      return word.event.kind === "clear"
        ? [turn, TRAINING_ENTITY.path(turn, "fast"), TRAINING_ENTITY.path(turn, "slow"),
          TRAINING_ENTITY.corridor, TRAINING_ENTITY.corridorAxis]
        : [];
    }
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
 *  names, and whose edges and turn paths (`TRAINING_ENTITY.edge` / `.path` of it) go with it. */
export function trainingEnvelopeEntities(flight: TrainingFlight): string[] {
  return [
    ...flight.envelopes.heading.flatMap((_, index) =>
      [TRAINING_ENTITY.turn(index), TRAINING_ENTITY.turnEnd(index), TRAINING_ENTITY.funnel(index)]),
    TRAINING_ENTITY.captureTurn, TRAINING_ENTITY.corridor, TRAINING_ENTITY.corridorAxis,
    ...flight.envelopes.altitude.map((_, index) => TRAINING_ENTITY.tube(index)),
  ];
}

/** The selected word's own turn, if it has one: a heading word's, or the clearance's capture turn. */
export function trainingFocusTurn(
  flight: TrainingFlight, column: TrainingColumn, word: TrainingWordRun & { index: number },
): TrainingTurnRegion | null {
  if (column === "heading") return flight.envelopes.heading[word.index].turn;
  if (column === "approach" && word.event.kind === "clear") return flight.envelopes.approach.captureTurn?.turn ?? null;
  return null;
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
  const { viewer, mode, trainingSelection, trainingLayers, trainingCursorS, trainingColumn } = useApp();

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
    const dash = (css: string) => new Cesium.PolylineDashMaterialProperty({ color: colour(css, 0.85) });
    const groundLine = (id: string, degrees: number[], width: number, material: Cesium.Color | Cesium.MaterialProperty,
      name?: string) =>
      add({
        id,
        name,
        polyline: { positions: Cesium.Cartesian3.fromDegreesArray(degrees), clampToGround: true, width, material },
      });
    /** A draped region and its edge; the edge carries the verdict (`edge` colour, dashed or not). */
    const ground = (
      id: string, line: TrainingPlanLine, css: string, alpha: number, name: string,
      edge: { css: string; dashed: boolean } = { css, dashed: false },
    ) => {
      add({
        id,
        name,
        polygon: {
          hierarchy: new Cesium.PolygonHierarchy(Cesium.Cartesian3.fromDegreesArray(planDegrees(line))),
          material: colour(css, alpha),
          classificationType: Cesium.ClassificationType.BOTH,
        },
      });
      groundLine(TRAINING_ENTITY.edge(id), planRingDegrees(line), EDGE_WIDTH, edge.dashed ? dash(edge.css) : colour(edge.css));
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

    if (trainingLayers.lateral) {
      ground(TRAINING_ENTITY.corridor, envelopes.approach.corridor.outline, TRAINING_CORRIDOR_COLOR, ALPHA.corridor,
        "The capture corridor");
      groundLine(TRAINING_ENTITY.corridorAxis, planDegrees(envelopes.approach.corridor.axis), 2, colour(TRAINING_CORRIDOR_COLOR));
      envelopes.heading.forEach((item, index) => {
        if (item.turn) {
          const turnOk = item.check === null || (item.check.progressOk && item.check.rateOk);
          ground(TRAINING_ENTITY.turn(index), item.turn.region, TRAINING_TURN_COLOR, ALPHA.turn,
            `Turn region, heading word ${index + 1}`, { css: verdict(turnOk, TRAINING_TURN_COLOR), dashed: false });
          ground(TRAINING_ENTITY.turnEnd(index), item.turn.end, TRAINING_TURN_COLOR, ALPHA.turnEnd,
            `Where the turn of heading word ${index + 1} may end`);
        }
        if (item.funnel) {
          const hold = item.holdCheck;
          ground(TRAINING_ENTITY.funnel(index), item.funnel.outline, TRAINING_FUNNEL_COLOR, ALPHA.funnel,
            `Hold funnel, heading word ${index + 1}`,
            { css: verdict(hold === null || hold.inside === hold.rows, TRAINING_FUNNEL_COLOR), dashed: hold === null });
        }
      });
      const capture = envelopes.approach.captureTurn;
      if (capture) {
        ground(TRAINING_ENTITY.captureTurn, capture.turn.region, TRAINING_TURN_COLOR, ALPHA.turn / 2, "The capture turn",
          { css: verdict(capture.check.progressOk && capture.check.rateOk, TRAINING_TURN_COLOR), dashed: true });
      }
    }

    if (trainingLayers.turnPaths) {
      const paths = (id: string, turn: TrainingTurnRegion, ok: boolean, name: string) => {
        const css = verdict(ok, TRAINING_TURN_COLOR);
        const fastest = planDegrees(turn.fastPath);
        const slowest = planDegrees(turn.slowPath);
        // A turn already inside its target band within one step has a path of one point: no line.
        if (fastest.length >= 4) groundLine(TRAINING_ENTITY.path(id, "fast"), fastest, PATH_WIDTH, colour(css, 0.9), `The fastest turn, ${name}`);
        if (slowest.length >= 4) {
          groundLine(TRAINING_ENTITY.path(id, "slow"), slowest, PATH_WIDTH, turn.slowFinished ? colour(css, 0.9) : dash(css),
            `The slowest turn, ${name}`);
        }
      };
      envelopes.heading.forEach((item, index) => {
        if (item.turn) {
          paths(TRAINING_ENTITY.turn(index), item.turn, item.check === null || (item.check.progressOk && item.check.rateOk),
            `heading word ${index + 1}`);
        }
      });
      const capture = envelopes.approach.captureTurn;
      if (capture) paths(TRAINING_ENTITY.captureTurn, capture.turn, capture.check.progressOk && capture.check.rateOk, "the capture turn");
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
      point(TRAINING_ENTITY.issue(index), item.row, TRAINING_COLUMN_COLOR.heading, 9, `Heading word ${index + 1} issued`));
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
    const { flight } = trainingSelection;
    const { lon, lat, altitudeHaeM } = flight.signals;
    // The track and every turn region: a big turn's slowest bound reaches kilometres past the track,
    // and selecting the word later should not find its envelope off screen. The regions lie on the
    // ground; the track's lowest point stands in for it.
    const ground = Math.min(...altitudeHaeM);
    const regions = [
      ...flight.envelopes.heading.flatMap((item) => (item.turn === null ? [] : [item.turn.region])),
      ...(flight.envelopes.approach.captureTurn === null ? [] : [flight.envelopes.approach.captureTurn.turn.region]),
    ];
    frameTrajectoryCamera(viewer, [
      ...lon.map((value, row) => ({ lon: value, lat: lat[row], altM: altitudeHaeM[row] })),
      ...regions.flatMap((line) => line.lon.map((value, point) => ({ lon: value, lat: line.lat[point], altM: ground }))),
    ], { margin: FRAME_MARGIN });  // the sentence bar and the dock cover a third of the canvas
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
      repaint(graphics, graphics.material instanceof Cesium.PolylineDashMaterialProperty
        ? new Cesium.PolylineDashMaterialProperty({ color: colourOf(own) })
        : new Cesium.ColorMaterialProperty(colourOf(own)), width);
    };
    const yellowLine = (graphics: Cesium.PolylineGraphics, width?: number) => recolour(graphics, () => selectedEdge, width);

    const mine = trainingFocusEntities(trainingSelection, trainingColumn, word);
    // Every other word's envelope recedes: its fill, its edges and its turn paths keep their hue.
    const own = new Set(mine);
    for (const id of trainingEnvelopeEntities(focusFlight)) {
      if (own.has(id)) continue;
      const parts = [id, TRAINING_ENTITY.edge(id), TRAINING_ENTITY.edge(id, "upper"), TRAINING_ENTITY.edge(id, "lower"),
        TRAINING_ENTITY.path(id, "fast"), TRAINING_ENTITY.path(id, "slow")];
      for (const part of parts) {
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
        // Its own hue, deepened: the turn stays orange and the funnel blue, so the word still reads.
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
    const tag = (text: string) => ({
      text,
      font: "600 12px sans-serif",
      fillColor: selectedEdge,
      outlineColor: Cesium.Color.BLACK,
      outlineWidth: 3,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE,
      horizontalOrigin: Cesium.HorizontalOrigin.LEFT,
      pixelOffset: new Cesium.Cartesian2(10, 0),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    });
    // The selected turn's two paths, named where they end (on the ground, with them).
    const turn = trainingFocusTurn(focusFlight, trainingColumn, word);
    if (turn !== null && trainingLayers.turnPaths) {
      const names = {
        fast: `fastest: ≤ ${turn.rateMaxDegS}°/s, ≤ ${turn.bankMaxDeg}° bank`,
        slow: `slowest: ${turn.rateMinDegS}°/s, begun ${turn.startDelayMaxS} s late`,
      };
      for (const which of ["fast", "slow"] as const) {
        const line = which === "fast" ? turn.fastPath : turn.slowPath;
        const end = line.lon.length - 1;
        if (end < 1) continue;   // a path of one point is not drawn, so it is not named
        added.push(TRAINING_ENTITY.focusPathLabel(which));
        viewer.entities.add({
          id: TRAINING_ENTITY.focusPathLabel(which),
          position: Cesium.Cartesian3.fromDegrees(line.lon[end], line.lat[end]),
          label: { ...tag(names[which]), heightReference: Cesium.HeightReference.CLAMP_TO_GROUND },
        });
      }
    }
    // A heading word's turn as the labeller read it flown: its departure and arrival rows.
    const check = trainingColumn === "heading" ? focusFlight.envelopes.heading[word.index].check : null;
    if (check !== null) {
      for (const [which, row] of [["starts", check.departureRow], ["ends", check.arrivalRow]] as const) {
        added.push(TRAINING_ENTITY.focusFlown(which));
        viewer.entities.add({
          id: TRAINING_ENTITY.focusFlown(which),
          name: `The turn flown ${which}`,
          position: Cesium.Cartesian3.fromDegrees(lon[row], lat[row], altitudeHaeM[row]),
          point: {
            pixelSize: 9,
            color: selectedEdge,
            outlineColor: Cesium.Color.BLACK.withAlpha(0.7),
            outlineWidth: 2,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
          label: tag(`turn flown ${which} · step ${row}`),
        });
      }
    }
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
