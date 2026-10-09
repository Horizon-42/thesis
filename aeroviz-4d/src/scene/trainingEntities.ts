/**
 * trainingEntities.ts
 * -------------------
 * The pieces the Training 3D scene is built from (`hooks/useTrainingTrackLayer.ts`, `hooks/useTrainingLiveLayer.ts`): the
 * entity ids, the exporter's and the backend's coordinates as Cesium's flat arrays, and the few entity shapes every
 * layer draws — a line in the air (solid; dashed where terrain hides it), a line on the ground (always dashed, thinner
 * and fainter, so that it never reads as a track in the air: frontend §3 item 6, D159), a marker. Every coordinate is the
 * exporter's (lon / lat and heights computed in Python) or the backend's, but one picture: a heading word's FAN
 * (`trainingBandFan`), drawn from the exported target and tolerance — the band in plan; the verdict stays the
 * exporter's, row by row.
 *
 * Every layer adds its entities through one `entityGroup` and removes them together when its inputs change.
 */

import * as Cesium from "cesium";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { haversineDistanceM, offsetPoint, toRadians } from "../utils/procedureGeoMath";
import type { TrainingLayers } from "../context/AppContext";
import { AIRCRAFT_MODEL_URI, aircraftOrientation } from "../utils/aircraftOrientation";
import type { TrainingAircraftPose } from "../data/trainingAttitude";
import {
  outsideSpans,
  trainingEnvelopeIndex,
  type TrainingAltitudeTube,
  type TrainingColumn,
  type TrainingHeadingBand,
  type TrainingReading,
  type TrainingTrack,
  type TrainingWordRun,
} from "../data/trainingSample";

export const TRAINING_ENTITY = {
  observedTrack: "training-observed-track",
  observedGround: "training-observed-ground",
  /** A heading word's judged rows on the ground, and its rows outside the band (`run` counts them). */
  heading: (index: number) => `training-heading-${index}`,
  headingOutside: (index: number, run: number) => `training-heading-${index}-outside-${run}`,
  /** A heading word's fan on the ground: its target ± its tolerance from its first judged row (`trainingBandFan`). */
  fan: (id: string) => `${id}-fan`,
  tube: (index: number) => `training-tube-${index}`,
  /** An envelope's edge: a tube's upper / lower line. */
  edge: (id: string, side?: "upper" | "lower") => (side ? `${id}-edge-${side}` : `${id}-edge`),
  candidate: (ident: string) => `training-candidate-${ident}`,
  /** The words the closed-loop reading added, where they were said on the flown path. */
  correction: (index: number) => `training-correction-${index}`,
  /** The flown path of the closed-loop sentence: its line, ground trace, where it ended and the DA point. */
  flownTrack: "training-flown-track",
  flownGround: "training-flown-ground",
  flownEnd: "training-flown-end",
  decision: "training-decision",
  /** The selected word: the rows it is in force, and its issue with its name. */
  focusStretch: "training-focus-stretch",
  focusIssue: "training-focus-issue",
  /** The live executor's segment: its flown line, the aircraft flying it out, where it began and its ground trace. */
  autopilotTrack: "training-autopilot-track",
  autopilotAircraft: "training-autopilot-aircraft",
  autopilotStart: "training-autopilot-start",
  autopilotGround: "training-autopilot-ground",
  /** The aircraft at the cursor: the observed one, and the flown one. */
  aircraftObserved: "training-aircraft-observed",
  aircraftFlown: "training-aircraft-flown",
} as const;

/** Every line on the ground (frontend §3 item 6, D159): dashed, at most this wide (px) and at this opacity. A heading
 *  word's judged rows are drawn this wide over the ground trace (2 px) they lie on. */
export const GROUND_LINE_WIDTH = 3;
export const GROUND_LINE_ALPHA = 0.6;

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

/** Points carrying longitude, latitude and ellipsoid height as columns — a track, the live executor's segment — as
 *  Cesium's flat [lon, lat, height, …]. */
export function lonLatHeights(points: { lon: number[]; lat: number[]; altitudeHaeM: number[] }): number[] {
  return points.lon.flatMap((lon, index) => [lon, points.lat[index], points.altitudeHaeM[index]]);
}

/** Points with longitude and latitude columns as Cesium's flat [lon, lat, …]. */
export function planDegrees(line: { lon: number[]; lat: number[] }): number[] {
  return line.lon.flatMap((lon, point) => [lon, line.lat[point]]);
}

/** One altitude tube as a wall over the judged track's ground position: a position per row it covers, with that row's lower
 *  and upper edge (MSL as exported, plus the flight's HAE − MSL). Envelope row r is the judged track's point r (both
 *  count from the same origin); a tube longer than the track is cut where the track ends. */
export function trainingTubeWall(track: TrainingTrack, tube: TrainingAltitudeTube, haeMinusMslM: number) {
  const last = Math.min(tube.endRow, track.tS.length);
  const positions: number[] = [];
  for (let row = tube.row; row < last; row += 1) positions.push(track.lon[row], track.lat[row]);
  return {
    positions,
    minimumHeights: tube.lowMslM.slice(0, last - tube.row).map((value) => value + haeMinusMslM),
    maximumHeights: tube.highMslM.slice(0, last - tube.row).map((value) => value + haeMinusMslM),
  };
}

/** Rows ``first..last`` (inclusive) of a line of points as Cesium's flat [lon, lat, …], or nothing when that is fewer
 *  than two points (a line needs two). */
export function groundRows(lon: number[], lat: number[], first: number, last: number): number[] {
  if (last <= first) return [];
  return Array.from({ length: last - first + 1 }, (_, offset) => [lon[first + offset], lat[first + offset]]).flat();
}

/** A heading word's judged rows on the ground: its first judged row on to its stop row, where the next word's begin, so
 *  the words meet; nothing for a word with no row of its own. */
export function trainingBandGround(track: TrainingTrack, band: TrainingHeadingBand): number[] {
  if (band.stopRow <= band.firstRow) return [];
  return groundRows(track.lon, track.lat, band.firstRow, Math.min(band.stopRow, track.lon.length - 1));
}

/** The Draw switches an envelope of ``reading`` needs on: its own, and — for the labelled sentence, whose envelopes judge
 *  the observed track — the observed track's. A closed-loop reading's envelopes judge its flown path. */
export function trainingEnvelopeSwitches(reading: TrainingReading, envelope: "headingBands" | "vertical"): Array<keyof TrainingLayers> {
  return reading.closed === null ? [envelope, "observed"] : [envelope];
}

/** The arc of a fan, one point a degree (at least two). */
const FAN_STEP_DEG = 1;

/** A heading word's band in plan, as a closed outline on the ground [lon, lat, …]: a fan from the judged track at the word's
 *  first judged row, centred on the word's target track, opening its tolerance to each side, as long as the farthest row
 *  it judges is from there — so a track held on the target stays inside it. A picture of the band: the band judges the
 *  track's DIRECTION row by row (its rows outside are drawn red), not its position, so a track that left the band and came
 *  back on to the target heading can run parallel outside the fan. Nothing for a word with no row of its own. */
export function trainingBandFan(track: TrainingTrack, band: TrainingHeadingBand): number[] {
  const last = Math.min(band.stopRow, track.lon.length - 1);
  if (last <= band.firstRow) return [];
  const apex = { lonDeg: track.lon[band.firstRow], latDeg: track.lat[band.firstRow], altM: 0 };
  let radiusM = 0;
  for (let row = band.firstRow + 1; row <= last; row += 1) {
    radiusM = Math.max(radiusM, haversineDistanceM(apex, { lonDeg: track.lon[row], latDeg: track.lat[row], altM: 0 }));
  }
  if (!(radiusM > 0)) return [];
  const steps = Math.max(Math.ceil((2 * band.toleranceDeg) / FAN_STEP_DEG), 1);
  const arc = Array.from({ length: steps + 1 }, (_, step) => {
    const bearing = band.targetDeg - band.toleranceDeg + (2 * band.toleranceDeg * step) / steps;
    const point = offsetPoint(apex, toRadians(bearing), radiusM);
    return [point.lonDeg, point.latDeg];
  });
  return [apex.lonDeg, apex.latDeg, ...arc.flat(), apex.lonDeg, apex.latDeg];
}

/** A fill on the ground, in ``css`` at ``alpha``. */
export function groundFill(id: string, name: string | undefined, degrees: number[], css: string, alpha: number): EntityOptions {
  return {
    id, name,
    polygon: {
      hierarchy: new Cesium.PolygonHierarchy(Cesium.Cartesian3.fromDegreesArray(degrees)),
      material: colour(css, alpha), classificationType: Cesium.ClassificationType.BOTH,
    },
  };
}

/** The rows a band judged outside, each as a line on the ground (`outsideSpans`: one row outside on to the next, so it is
 *  a segment). */
export function trainingBandOutsideGround(track: TrainingTrack, band: TrainingHeadingBand): number[][] {
  return outsideSpans(band.inside, band.firstRow, track.lon.length - 1)
    .map(([first, last]) => groundRows(track.lon, track.lat, first, last));
}

// ── the selected word ────────────────────────────────────────────────────────

/** The envelope entity a word owns, or null: a heading word's judged rows, an altitude word's tube. A runway, angle or
 *  speed word bounds no position, so it owns none: the stretch of track it is in force is its picture. */
export function trainingFocusEntity(reading: TrainingReading, stepS: number, column: TrainingColumn, word: TrainingWordRun): string | null {
  const index = trainingEnvelopeIndex(reading, stepS, column, word);
  if (index === null) return null;
  if (column === "heading") return TRAINING_ENTITY.heading(index);
  return column === "altitude" ? TRAINING_ENTITY.tube(index) : null;
}

/** Every envelope of a reading, by the id of its main entity: the one that `trainingFocusEntity` names, and whose edges
 *  (`TRAINING_ENTITY.edge` of it) go with it. */
export function trainingEnvelopeEntities(reading: TrainingReading): string[] {
  if (reading.envelopes === null) return [];
  return [
    ...reading.envelopes.heading.flatMap((_, index) => [TRAINING_ENTITY.heading(index), TRAINING_ENTITY.fan(TRAINING_ENTITY.heading(index))]),
    ...reading.envelopes.altitude.map((_, index) => TRAINING_ENTITY.tube(index)),
  ];
}

/** The flight-time stretch of a track a word is in force, as positions (on to the next word's issue, so that it meets
 *  it), or nothing when the word is said outside the track. */
export function trainingFocusStretch(track: TrainingTrack, fromS: number, toS: number): number[] {
  const first = track.tS.findIndex((t) => t >= fromS - 1e-9);
  if (first < 0) return [];
  let last = track.tS.length - 1;
  while (last > first && track.tS[last] > toS + 1e-9) last -= 1;
  if (last <= first) return [];
  return lonLatHeights(track).slice(first * 3, (last + 1) * 3);
}

// ── the shapes every layer draws ─────────────────────────────────────────────

/** A line in the air: solid (``dashed``: dashed, a silent aircraft's or a loss no commanded aircraft answers for), and
 *  dashed where terrain hides it (a track that runs into a hill is a reading, not a rendering accident); `alpha` fades
 *  both. */
export function airLine(
  id: string, name: string, positions: Cesium.Cartesian3[] | Cesium.Property, css: string, width: number, alpha = 1, dashed = false,
): EntityOptions {
  return {
    id, name,
    polyline: {
      positions, width, material: dashed ? new Cesium.PolylineDashMaterialProperty({ color: colour(css, alpha) }) : colour(css, alpha),
      depthFailMaterial: new Cesium.PolylineDashMaterialProperty({ color: colour(css, 0.55 * alpha) }),
    },
  };
}

/** A line draped on the ground, in ``css``: dashed, at `GROUND_LINE_ALPHA`; ``thin`` draws it 2 px instead of
 *  `GROUND_LINE_WIDTH` (a ground trace, under the envelopes that lie on it). */
export function groundLine(id: string, name: string | undefined, degrees: number[], css: string, thin = false): EntityOptions {
  return {
    id, name,
    polyline: {
      positions: Cesium.Cartesian3.fromDegreesArray(degrees), clampToGround: true, width: thin ? 2 : GROUND_LINE_WIDTH,
      material: dash(css, GROUND_LINE_ALPHA),
    },
  };
}

/** A dashed line's material. */
export const dash = (css: string, alpha = 0.85) => new Cesium.PolylineDashMaterialProperty({ color: colour(css, alpha) });

/** How an aircraft model is drawn: its tint (blended into the model's own colours by ``blend``), its least size on screen,
 *  a silhouette ``ring`` (px, in ``ringCss``) and how opaque it is. */
export interface AircraftLook {
  css: string;
  blend: number;
  minimumPixelSize: number;
  ringCss: string;
  ringPx: number;
  alpha: number;
}

/** The aircraft model's orientation at ``pose``: its heading, its PATH ANGLE as the pitch — the angle of attack is a
 *  reading written in the label, never drawn (a clean-wing lift curve, ~15° high on a flapped final: the user,
 *  2026-09-30) — and its right bank, wings level for a flight without an airframe (its label says so). */
export function poseOrientation(position: Cesium.Cartesian3, pose: TrainingAircraftPose): Cesium.Quaternion {
  return aircraftOrientation(position, pose.headingDeg, pose.pathAngleDeg, pose.bankRightDeg ?? 0);
}

/** The aircraft model drawn in ``look``. */
export function aircraftGraphics(look: AircraftLook): Cesium.ModelGraphics.ConstructorOptions {
  return {
    uri: AIRCRAFT_MODEL_URI, scale: 3.0, minimumPixelSize: look.minimumPixelSize, maximumScale: 20_000,
    color: colour(look.css, look.alpha), colorBlendMode: Cesium.ColorBlendMode.MIX, colorBlendAmount: look.blend,
    silhouetteColor: colour(look.ringCss), silhouetteSize: look.ringPx,
  };
}

/** An aircraft where ``pose`` has it, drawn as the aircraft model in ``look``; its position and orientation are
 *  replaced as it moves (`placeAircraft`). */
export function aircraftModel(id: string, name: string, pose: TrainingAircraftPose, look: AircraftLook): EntityOptions {
  const position = Cesium.Cartesian3.fromDegrees(pose.lon, pose.lat, pose.heightHaeM);
  return {
    id, name, position: new Cesium.ConstantPositionProperty(position),
    orientation: new Cesium.ConstantProperty(poseOrientation(position, pose)), model: aircraftGraphics(look),
  };
}

/** Moves an aircraft drawn by `aircraftModel` to ``pose``. */
export function placeAircraft(entity: Cesium.Entity, pose: TrainingAircraftPose): void {
  const position = Cesium.Cartesian3.fromDegrees(pose.lon, pose.lat, pose.heightHaeM);
  entity.position = new Cesium.ConstantPositionProperty(position);
  entity.orientation = new Cesium.ConstantProperty(poseOrientation(position, pose));
}

/** A point that stays visible through terrain, with an optional label under it. */
export function marker(
  id: string, name: string, position: Cesium.Cartesian3, css: string, size: number, label?: string,
  labelOffset: Cesium.Cartesian2 = new Cesium.Cartesian2(0, 18),
  labelOrigin: Cesium.HorizontalOrigin = Cesium.HorizontalOrigin.CENTER,
): EntityOptions {
  return {
    id, name, position,
    point: {
      pixelSize: size, color: colour(css), outlineColor: Cesium.Color.BLACK.withAlpha(0.6), outlineWidth: 2,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
    label: label === undefined ? undefined : {
      text: label, font: "600 12px sans-serif", fillColor: colour(css), outlineColor: Cesium.Color.BLACK, outlineWidth: 3,
      style: Cesium.LabelStyle.FILL_AND_OUTLINE, pixelOffset: labelOffset, horizontalOrigin: labelOrigin,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    },
  };
}
