/**
 * trainingEntities.ts
 * -------------------
 * The pieces the Training 3D scene is built from (`hooks/useTrainingTrackLayer.ts`, `hooks/useTrainingExecutorLayers.ts`):
 * the entity ids, the exporter's and the backend's coordinates as Cesium's flat arrays, and the few entity shapes every
 * layer draws — a line in the air (dashed where terrain hides it), a line on the ground, a marker. Nothing here computes
 * geometry: every coordinate is the exporter's (lon / lat and heights computed in Python) or the backend's.
 *
 * Every layer adds its entities through one `entityGroup` and removes them together when its inputs change.
 */

import * as Cesium from "cesium";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import {
  outsideSpans,
  type TrainingAltitudeTube,
  type TrainingColumn,
  type TrainingFlight,
  type TrainingHeadingBand,
  type TrainingPlanLine,
  type TrainingSelection,
  type TrainingWordRun,
} from "../data/trainingSample";

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
  /** The live executor's segment: its flown line, the aircraft flying it out, where it began, its ground trace (once
   *  flown) and its rows outside the selected heading word. */
  autopilotTrack: "training-autopilot-track",
  autopilotAircraft: "training-autopilot-aircraft",
  autopilotStart: "training-autopilot-start",
  autopilotGround: "training-autopilot-ground",
  autopilotOutside: (run: number) => `training-autopilot-outside-${run}`,
  /** The model sentence read: each of its samples' flown tracks, the one read's ground trace, where it ended, where it
   *  said its heading words and the clearance (by the word's place in its events), and its selected word. */
  modelTrack: (sample: number) => `training-model-track-${sample}`,
  modelGround: "training-model-ground",
  modelEnd: "training-model-end",
  modelIssue: (index: number) => `training-model-issue-${index}`,
  modelFocusStretch: "training-model-focus-stretch",
  modelFocusIssue: "training-model-focus-issue",
} as const;

/** A heading word's judged rows and the capture turn on the ground (px): wider than the ground trace they lie on. */
export const GROUND_ROWS_WIDTH = 5;

export const colour = (css: string, alpha = 1) => Cesium.Color.fromCssColorString(css).withAlpha(alpha);

// ── one layer's entities ─────────────────────────────────────────────────────

export type EntityOptions = Cesium.Entity.ConstructorOptions & { id: string };

/** Entities added together and removed together: a layer's effect adds through one and returns its `remove`. */
export function entityGroup(viewer: Cesium.Viewer) {
  const ids: string[] = [];
  return {
    add(options: EntityOptions): Cesium.Entity {
      ids.push(options.id);
      return viewer.entities.add(options);
    },
    remove(): void {
      if (!isCesiumViewerUsable(viewer)) return;
      for (const id of ids) viewer.entities.removeById(id);
    },
  };
}

// ── coordinates ──────────────────────────────────────────────────────────────

/** Points carrying longitude, latitude and ellipsoid height as columns — the observed track, the executor's flown
 *  track, the live executor's segment — as Cesium's flat [lon, lat, height, …]. */
export function lonLatHeights(points: { lon: number[]; lat: number[]; altitudeHaeM: number[] }): number[] {
  return points.lon.flatMap((lon, index) => [lon, points.lat[index], points.altitudeHaeM[index]]);
}

/** A plan line — or any points with longitude and latitude columns — as Cesium's flat [lon, lat, …]. */
export function planDegrees(line: { lon: number[]; lat: number[] }): number[] {
  return line.lon.flatMap((lon, point) => [lon, line.lat[point]]);
}

/** A region's outline as a closed line: the exporter's rings are open (a polygon closes itself). */
export function planRingDegrees(line: TrainingPlanLine): number[] {
  return [...planDegrees(line), line.lon[0], line.lat[0]];
}

/** One altitude tube as a wall over the aircraft's own ground track: a position per row it covers, with that row's
 *  lower and upper edge (HAE, as exported). */
export function trainingTubeWall(flight: TrainingFlight, tube: TrainingAltitudeTube) {
  const positions: number[] = [];
  for (let row = tube.row; row < tube.endRow; row += 1) positions.push(flight.signals.lon[row], flight.signals.lat[row]);
  return { positions, minimumHeights: tube.lowerHaeM, maximumHeights: tube.upperHaeM };
}

/** Rows ``first..last`` (inclusive) of a line of points as Cesium's flat [lon, lat, …], or nothing when that is fewer
 *  than two points (a line needs two). */
export function groundRows(lon: number[], lat: number[], first: number, last: number): number[] {
  if (last <= first) return [];
  return Array.from({ length: last - first + 1 }, (_, offset) => [lon[first + offset], lat[first + offset]]).flat();
}

/** A heading word's judged rows on the ground: its first judged row on to its stop row, where the next word's begin,
 *  so the words meet; nothing for a word with no row of its own. */
export function trainingBandGround(lon: number[], lat: number[], band: TrainingHeadingBand): number[] {
  if (band.stopRow <= band.firstRow) return [];
  return groundRows(lon, lat, band.firstRow, Math.min(band.stopRow, lon.length - 1));
}

/** The rows a band judged outside, each as a line on the ground (`outsideSpans`: one row outside on to the next, so it
 *  is a segment). */
export function trainingBandOutsideGround(lon: number[], lat: number[], band: TrainingHeadingBand): number[][] {
  return outsideSpans(band.inside, band.firstRow, lon.length - 1).map(([first, last]) => groundRows(lon, lat, first, last));
}

// ── the selected word ────────────────────────────────────────────────────────

/**
 * The entities that ARE a word's envelope — its own column's, and nothing of another column's: a heading word's
 * judged rows; the clearance's capture turn and corridor; an altitude word's tube; the runway pointed at. An angle or a
 * speed word bounds no position, so it owns none: the stretch of track it is in force is its picture. An id the scene
 * does not hold (a word the lead carries to the clearance has no rows; a switch is off) is simply not there to paint.
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

/** Every envelope of a flight, by the id of its main entity: the one that `trainingFocusEntities` names, and whose
 *  edges (`TRAINING_ENTITY.edge` of it) go with it. */
export function trainingEnvelopeEntities(flight: TrainingFlight): string[] {
  return [
    ...flight.envelopes.heading.map((_, index) => TRAINING_ENTITY.heading(index)),
    TRAINING_ENTITY.captureTurn, TRAINING_ENTITY.corridor, TRAINING_ENTITY.corridorAxis,
    ...flight.envelopes.altitude.map((_, index) => TRAINING_ENTITY.tube(index)),
  ];
}

/** The track rows a word is in force, as positions: on to the next word's issue, so that it meets it. Only a word
 *  issued on the last row has no stretch (its issue marker is its picture). */
export function trainingFocusStretch(flight: TrainingFlight, word: TrainingWordRun): number[] {
  const last = Math.min(word.endRow, flight.rows - 1);
  if (last <= word.row) return [];
  return lonLatHeights(flight.signals).slice(word.row * 3, (last + 1) * 3);
}

// ── the shapes every layer draws ─────────────────────────────────────────────

/** A line in the air: solid, and dashed where terrain hides it (a track that runs into a hill is a reading, not a
 *  rendering accident). */
export function airLine(
  id: string, name: string, positions: Cesium.Cartesian3[] | Cesium.Property, css: string, width: number,
): EntityOptions {
  return {
    id, name,
    polyline: {
      positions, width, material: colour(css),
      depthFailMaterial: new Cesium.PolylineDashMaterialProperty({ color: colour(css, 0.55) }),
    },
  };
}

/** A line draped on the ground. */
export function groundLine(
  id: string, name: string | undefined, degrees: number[], width: number, material: Cesium.Color | Cesium.MaterialProperty,
): EntityOptions {
  return { id, name, polyline: { positions: Cesium.Cartesian3.fromDegreesArray(degrees), clampToGround: true, width, material } };
}

/** A dashed line's material. */
export const dash = (css: string, alpha = 0.85) => new Cesium.PolylineDashMaterialProperty({ color: colour(css, alpha) });

/** A point that stays visible through terrain, with an optional label under it. */
export function marker(
  id: string, name: string, position: Cesium.Cartesian3, css: string, size: number, label?: string,
): EntityOptions {
  return {
    id, name, position,
    point: {
      pixelSize: size, color: colour(css), outlineColor: Cesium.Color.BLACK.withAlpha(0.6), outlineWidth: 2,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
    label: label === undefined ? undefined : {
      text: label, font: "600 12px sans-serif", fillColor: colour(css), outlineColor: Cesium.Color.BLACK, outlineWidth: 3,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE, pixelOffset: new Cesium.Cartesian2(0, 18),
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
  };
}
