/**
 * useTrainingTrackLayer.ts
 * ------------------------
 * The selected Training flight in the 3D scene: its track, and the envelopes its sentence allows.
 * Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`.
 *
 *  • THE TRACK, in 3D, at its ellipsoid height (the exporter converted MSL once: h = H + N).
 *  • THE LATERAL ENVELOPES on the ground (`trainingLayers.lateral`): each heading word's turn
 *    region and hold funnel, the capture turn, and the capture corridor with its centreline. They
 *    bound positions in plan only — the vertical is the tubes' business — so they are draped on
 *    the terrain rather than floated at some height the words do not give them.
 *  • THE ALTITUDE TUBES (`trainingLayers.vertical`): one Cesium wall per altitude word, over the
 *    aircraft's own ground track, between the tube's lower and upper edge (both HAE, exported).
 *  • THE CANDIDATE RUNWAYS (`trainingLayers.candidates`): every threshold the runway pointer can
 *    point at, its runway and extended centreline; the designated one is drawn whatever the
 *    switch says, because the corridor and the landing are measured from it.
 *  • WHERE WORDS WERE ISSUED: a point on the track at every heading word, the clearance, the
 *    capture and the end of the sentence.
 *
 * Every coordinate is the exporter's (lon / lat and heights computed in Python); this hook draws
 * them and computes no geometry. The envelope in force at the shared cursor is repainted yellow
 * without rebuilding anything.
 *
 * STATIC ENTITIES, NOT TIME-SAMPLED ONES: a time-dynamic entity would drive the shared
 * `viewer.clock`, which belongs to Observe's playback.
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import {
  TRAINING_CANDIDATE_COLOR,
  TRAINING_COLUMN_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_DESIGNATED_COLOR,
  TRAINING_FUNNEL_COLOR,
  TRAINING_TRACE_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_TURN_COLOR,
  TRAINING_WORD_COLOR,
} from "../utils/trainingWordColors";
import {
  altitudeTubeAt,
  headingEnvelopeAt,
  rowAtTime,
  type TrainingAltitudeTube,
  type TrainingFlight,
  type TrainingPlanLine,
} from "../data/trainingSample";

export const TRAINING_ENTITY = {
  track: "training-track",
  turn: (index: number) => `training-turn-${index}`,
  funnel: (index: number) => `training-funnel-${index}`,
  captureTurn: "training-capture-turn",
  corridor: "training-corridor",
  corridorAxis: "training-corridor-axis",
  tube: (index: number) => `training-tube-${index}`,
  centreline: (ident: string) => `training-centreline-${ident}`,
  runway: (ident: string) => `training-runway-${ident}`,
  runwayLabel: (ident: string) => `training-runway-label-${ident}`,
  issue: (index: number) => `training-issue-${index}`,
  clearance: "training-clearance",
  capture: "training-capture",
  end: "training-end",
} as const;

/** How opaque each envelope is at rest, and when it is the one in force. */
const ALPHA = { turn: 0.16, funnel: 0.18, corridor: 0.3, tube: 0.22, selected: 0.45 } as const;

/** Cesium wants [lon, lat, height, …]; the track carries the three as columns. */
export function trainingTrackPositions(flight: TrainingFlight): number[] {
  const { lon, lat, altitudeHaeM } = flight.signals;
  return lon.flatMap((value, row) => [value, lat[row], altitudeHaeM[row]]);
}

/** A plan line as Cesium's flat [lon, lat, …]. */
export function planDegrees(line: TrainingPlanLine): number[] {
  return line.lon.flatMap((value, point) => [value, line.lat[point]]);
}

/** One altitude tube as a wall over the aircraft's own ground track: a position per row it
 *  covers, with that row's lower and upper edge (HAE, as exported). */
export function trainingTubeWall(flight: TrainingFlight, tube: TrainingAltitudeTube) {
  const positions: number[] = [];
  for (let row = tube.row; row < tube.endRow; row += 1) positions.push(flight.signals.lon[row], flight.signals.lat[row]);
  return { positions, minimumHeights: tube.lowerHaeM, maximumHeights: tube.upperHaeM };
}

/** The envelopes in force at a time: the heading word's (-1 once captured) and the tube. */
export function trainingEnvelopeInForce(flight: TrainingFlight, seconds: number): { heading: number; altitude: number } {
  const row = rowAtTime(flight.signals.tS, seconds);
  return { heading: headingEnvelopeAt(flight, row), altitude: altitudeTubeAt(flight, row) };
}

const colour = (css: string, alpha = 1) => Cesium.Color.fromCssColorString(css).withAlpha(alpha);

export default function useTrainingTrackLayer(): void {
  const { viewer, mode, trainingSelection, trainingLayers, trainingCursorS } = useApp();

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
    const ground = (id: string, line: TrainingPlanLine, css: string, alpha: number, name: string) =>
      add({
        id,
        name,
        polygon: {
          hierarchy: new Cesium.PolygonHierarchy(Cesium.Cartesian3.fromDegreesArray(planDegrees(line))),
          material: colour(css, alpha),
          classificationType: Cesium.ClassificationType.BOTH,
        },
      });
    const groundLine = (id: string, line: TrainingPlanLine, css: string, width: number, dashed: boolean) =>
      add({
        id,
        polyline: {
          positions: Cesium.Cartesian3.fromDegreesArray(planDegrees(line)),
          clampToGround: true,
          width,
          material: dashed ? new Cesium.PolylineDashMaterialProperty({ color: colour(css, 0.8) }) : colour(css),
        },
      });

    // THE RUNWAYS first, so every envelope reads over them.
    for (const candidate of candidates) {
      const pointed = candidate.index === flight.runwayIndex;
      if (!pointed && !trainingLayers.candidates) continue;
      const css = pointed ? TRAINING_DESIGNATED_COLOR : TRAINING_CANDIDATE_COLOR;
      groundLine(TRAINING_ENTITY.centreline(candidate.ident), candidate.centreline, css, pointed ? 2 : 1.5, true);
      groundLine(TRAINING_ENTITY.runway(candidate.ident), candidate.runway, css, pointed ? 7 : 5, false);
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
      groundLine(TRAINING_ENTITY.corridorAxis, envelopes.approach.corridor.axis, TRAINING_CORRIDOR_COLOR, 2, false);
      envelopes.heading.forEach((item, index) => {
        if (item.turn) ground(TRAINING_ENTITY.turn(index), item.turn.region, TRAINING_TURN_COLOR, ALPHA.turn, `Turn region, heading word ${index + 1}`);
        if (item.funnel) ground(TRAINING_ENTITY.funnel(index), item.funnel.outline, TRAINING_FUNNEL_COLOR, ALPHA.funnel, `Hold funnel, heading word ${index + 1}`);
      });
      const capture = envelopes.approach.captureTurn;
      if (capture) ground(TRAINING_ENTITY.captureTurn, capture.turn.region, TRAINING_TURN_COLOR, ALPHA.turn / 2, "The capture turn");
    }

    if (trainingLayers.vertical) {
      envelopes.altitude.forEach((tube, index) => {
        const wall = trainingTubeWall(flight, tube);
        if (wall.minimumHeights.length >= 2) {
          add({
            id: TRAINING_ENTITY.tube(index),
            name: `Altitude tube ${index + 1}`,
            wall: {
              positions: Cesium.Cartesian3.fromDegreesArray(wall.positions),
              minimumHeights: wall.minimumHeights,
              maximumHeights: wall.maximumHeights,
              material: colour(TRAINING_TUBE_COLOR, ALPHA.tube),
              outline: false,
            },
          });
        } else {
          // A tube of ONE row has no length to be a wall along: its extent is a vertical segment.
          add({
            id: TRAINING_ENTITY.tube(index),
            name: `Altitude tube ${index + 1}`,
            polyline: {
              positions: Cesium.Cartesian3.fromDegreesArrayHeights([
                wall.positions[0], wall.positions[1], wall.minimumHeights[0],
                wall.positions[0], wall.positions[1], wall.maximumHeights[0],
              ]),
              width: 3,
              material: colour(TRAINING_TUBE_COLOR, 0.8),
            },
          });
        }
      });
    }

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

  // The envelopes in force at the cursor, repainted — styles only, no geometry rebuilt.
  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || mode !== "training" || !trainingSelection) return;
    const { heading, altitude } = trainingEnvelopeInForce(trainingSelection.flight, trainingCursorS);
    const selected = colour(TRAINING_WORD_COLOR, ALPHA.selected);
    const painted: Array<[Cesium.Entity, Cesium.MaterialProperty]> = [];
    const paint = (id: string) => {
      const entity = viewer.entities.getById(id);
      const graphics = entity?.polygon ?? entity?.wall ?? entity?.polyline;
      if (!entity || !graphics?.material) return;
      painted.push([entity, graphics.material]);
      graphics.material = new Cesium.ColorMaterialProperty(selected);
    };
    if (heading >= 0) {
      paint(TRAINING_ENTITY.turn(heading));
      paint(TRAINING_ENTITY.funnel(heading));
    } else {
      paint(TRAINING_ENTITY.corridor);
    }
    paint(TRAINING_ENTITY.tube(altitude));
    const issue = heading >= 0 ? viewer.entities.getById(TRAINING_ENTITY.issue(heading))?.point : undefined;
    if (issue) issue.pixelSize = new Cesium.ConstantProperty(14);
    viewer.scene.requestRender();

    return () => {
      if (!isCesiumViewerUsable(viewer)) return;
      for (const [entity, material] of painted) {
        const graphics = entity.polygon ?? entity.wall ?? entity.polyline;
        if (graphics) graphics.material = material;
      }
      if (issue) issue.pixelSize = new Cesium.ConstantProperty(9);
      viewer.scene.requestRender();
    };
  }, [viewer, mode, trainingSelection, trainingLayers, trainingCursorS]);
}
