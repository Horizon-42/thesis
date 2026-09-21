/**
 * useTrainingTrackLayer.ts
 * ------------------------
 * The selected Training flight in the 3D scene: the aircraft's own track, and the
 * ENVELOPE its sentence allows around it — the wedge wall the altitude words
 * make, and the chain of boxes the sentence is. Design:
 * `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §3.1 / §5.6, T6 / T15.
 *
 * THE SCENE DRAWS A REGION, NOT A SECOND CURVE. Under `box-v2-wedge` a word is an
 * interval and a sentence is a chain of bounding boxes, so what the words say
 * about this flight is a volume; there is no sentence flown by rule to draw
 * beside the track (that needs a height-tracking executor — the replay gate,
 * which is not built). Drawing one line here would be drawing the one thing this
 * vocabulary does not say.
 *
 * TWO SHAPES, AND THEY ANSWER DIFFERENT QUESTIONS.
 *  • The WALL is the altitude words' wedge, over the aircraft's own ground
 *    track: an ABSOLUTE bound on height, wide where a segment begins and closing
 *    onto ±5 % of its target where it ends. It is the one bound in this
 *    vocabulary that pins a position rather than a rate. It is drawn as ONE WALL
 *    PER ALTITUDE SEGMENT, not one for the flight: the bound STEPS at every
 *    altitude word (a segment closes onto its target, the next opens wide again —
 *    measured on one vectored arrival, nine steps of up to 59 m), and a single
 *    ribbon forced through those steps renders them as twisted facets over the
 *    turning ground track. Split, each ribbon tapers cleanly and the step between
 *    two of them reads as what it is.
 *  • The SOLIDS are one per word, and each one is a FRUSTUM, not a prism: a pie
 *    slice in plan, a trapezoid in every radial section. The footprint FANS OUT
 *    FROM THE AIRCRAFT — as deep as the hold times the speed box's upper edge, as
 *    wide as the heading box — and is deliberately NOT the rectangle around it,
 *    which would show flyable-looking ground beside the apex that no heading in
 *    the box can reach. The HEIGHT tapers with it: an outline point `d` metres
 *    out has `d` metres less path left to its altitude segment's end, so its
 *    slice of the wedge is that much tighter. Extruding one height over the whole
 *    footprint over-states the ceiling at the far end by up to 26 % of the box's
 *    own height on a long hold — a flat lid is the one thing the words never say.
 *    The solids are thin horizontally (a 2° box over a 4 s hold is 488 m deep and
 *    17 m wide at its far edge), and that thinness is the finding: horizontally
 *    one word says almost nothing, and it is the accumulation over a whole
 *    sentence that opens the funnel (design §5.1).
 *
 * A word is a box in STATE space — heading × speed × altitude — which is what the
 * vocabulary means by a bounding box. In POSITION space it is this sector, and it
 * is DERIVED: a word constrains the state at every instant, and where that lets
 * the aircraft go over its hold is this. Nothing bounds how fast the heading may
 * swing inside its box, because the word does not. The outline is computed at the
 * exporter, in the course frame, because the frame's transform lives there.
 *
 * STATIC GEOMETRY, NOT TIME-SAMPLED ENTITIES. Training loads no CZML on purpose
 * (design V2): a time-dynamic entity would drive the shared `viewer.clock`, and
 * the clock belongs to Observe's playback — switching tasks would move it.
 *
 * The altitudes are already HAE (`altHaeM`, `altHaeLoM`, `altHaeHiM`, converted
 * at the exporter), which is what Cesium wants. Feeding it MSL would float
 * everything ~33.5 m ABOVE where it belongs (h = H + N, and N is negative here)
 * — together, so the error would be invisible within the scene and visible only
 * against the terrain.
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import {
  TRAINING_FLOWN_COLOR,
  TRAINING_MODEL_COLOR,
  TRAINING_TRACE_COLOR,
} from "../utils/trainingWordColors";
import {
  TRAINING_KIND_COLUMN,
  altitudeWedgeM,
  eventInForce,
  type TrainingEnvelope,
  type TrainingEventBox,
  type TrainingFlight,
  type TrainingWordSpec,
} from "../data/trainingSample";

const OBSERVED_ID = "training-observed-track";
const WALL_ID = "training-envelope-wall";
const MODEL_WALL_ID = "training-model-envelope-wall";
const BOX_ID = "training-envelope-box";
const NODE_ID = "training-segment-node";

/** Cesium wants [lon, lat, height, …]; the track carries the three as columns. */
function degreesArrayHeights(track: {
  lon: number[];
  lat: number[];
  altHaeM: number[];
}): number[] {
  const flat: number[] = [];
  for (let row = 0; row < track.lon.length; row += 1) {
    flat.push(track.lon[row], track.lat[row], track.altHaeM[row]);
  }
  return flat;
}

export function trainingTrackPositions(flight: TrainingFlight): { observed: number[] } {
  return { observed: degreesArrayHeights(flight.observed) };
}

/**
 * The altitude wedge as a Cesium wall: one ground track, a maximum height per
 * position and a minimum.
 *
 * BOTH ARE HAE. `altHaeHiM` is the MAXIMUM and `altHaeLoM` the minimum, which is
 * the plain reading of the names here — unlike the retired corridor, where "lo"
 * named the shallower descent and therefore sat higher. The wedge is a height
 * interval and nothing else.
 */
export function trainingBandWall(
  flight: TrainingFlight,
  envelope: TrainingEnvelope,
  vocabulary: TrainingWordSpec,
  words: number[][],
): Array<{ positions: number[]; maximumHeights: number[]; minimumHeights: number[] }> {
  const { lon, lat, haeOffsetM } = flight.observed;
  const forced = eventInForce(flight.sentence.eventTimesS, flight.observed.tS);
  const wordAt = (row: number) => words[forced[row]][TRAINING_KIND_COLUMN.altitude];
  const walls = [];
  let row = 0;
  while (row < lon.length) {
    let last = row;
    while (last + 1 < lon.length && wordAt(last + 1) === wordAt(row)) last += 1;
    const positions: number[] = [];
    const maximumHeights: number[] = [];
    const minimumHeights: number[] = [];
    for (let index = row; index <= last; index += 1) {
      positions.push(lon[index], lat[index]);
      maximumHeights.push(envelope.altHaeHiM[index]);
      minimumHeights.push(envelope.altHaeLoM[index]);
    }
    // ONE MORE POSITION, at the next segment's first row but with THIS segment's
    // own closing box — the target's ±5 %, which is where its wedge ends. Without
    // it the ribbons stop a row apart and a hairline of unbounded height opens
    // between two segments that in fact meet; with the NEXT segment's heights
    // instead, the ribbon's last facet spans the step and renders as the twist
    // this split exists to remove.
    if (last + 1 < lon.length) {
      const [low, high] = altitudeWedgeM(vocabulary, wordAt(row), 0);
      positions.push(lon[last + 1], lat[last + 1]);
      maximumHeights.push(haeOffsetM[last + 1] + high);
      minimumHeights.push(haeOffsetM[last + 1] + low);
    }
    walls.push({ positions, maximumHeights, minimumHeights });
    row = last + 1;
  }
  return walls;
}

/**
 * ONE WORD'S SOLID, as Cesium draws it: the outline closed back onto its apex,
 * with the wedge's own two heights at every point of it.
 *
 * A `wall` takes a height pair per position, which is exactly what a frustum
 * needs and what a `polygon` with one `extrudedHeight` cannot express. The
 * outline is closed by repeating the apex, so the two radial faces — the
 * trapezoids — are drawn as well as the arc face.
 */
export function trainingBoxWall(box: TrainingEventBox): {
  positions: number[];
  maximumHeights: number[];
  minimumHeights: number[];
} {
  const positions: number[] = [];
  const maximumHeights: number[] = [];
  const minimumHeights: number[] = [];
  for (let point = 0; point <= box.lon.length; point += 1) {
    const index = point % box.lon.length;
    positions.push(box.lon[index], box.lat[index]);
    maximumHeights.push(box.altHaeHiM[index]);
    minimumHeights.push(box.altHaeLoM[index]);
  }
  return { positions, maximumHeights, minimumHeights };
}

/**
 * WHERE THE WORDS CUT THE TRACK, as positions in the scene: one per event, on the
 * track's own rows.
 *
 * Without these the shape reads as a curve rather than as a sentence — and the
 * question "which word is in force here" has no answer on screen.
 */
export function trainingSegmentNodes(
  track: { tS: number[]; lon: number[]; lat: number[]; altHaeM: number[] },
  eventTimesS: number[],
): Array<{ eventS: number; lon: number; lat: number; altHaeM: number }> {
  const last = track.tS[track.tS.length - 1];
  const nodes = [];
  for (const eventS of eventTimesS) {
    if (eventS > last) continue;
    let row = 0;
    while (row + 1 < track.tS.length && track.tS[row + 1] <= eventS) row += 1;
    nodes.push({ eventS, lon: track.lon[row], lat: track.lat[row], altHaeM: track.altHaeM[row] });
  }
  return nodes;
}

export default function useTrainingTrackLayer(): void {
  const { viewer, mode, trainingSelection, trainingLayers } = useApp();

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer)) return;
    const flight = mode === "training" ? trainingSelection?.flight : undefined;
    if (!flight) return;

    const added: string[] = [];
    const vocabulary = trainingSelection!.vocabulary;
    const drawWall = (
      id: string, envelope: TrainingEnvelope, words: number[][], colour: string, alpha: number,
    ) => {
      trainingBandWall(flight, envelope, vocabulary, words).forEach((wall, segment) => {
        const entityId = `${id}-${segment}`;
        added.push(entityId);
        viewer.entities.add({
          id: entityId,
          wall: {
            positions: Cesium.Cartesian3.fromDegreesArray(wall.positions),
            maximumHeights: wall.maximumHeights,
            minimumHeights: wall.minimumHeights,
            material: Cesium.Color.fromCssColorString(colour).withAlpha(alpha),
            outline: false,
          },
        });
      });
    };

    // The envelope goes in FIRST, so the track reads over it rather than under
    // it: it is the region the words allow, and the track is what is being
    // checked against it.
    if (trainingLayers.flown) {
      drawWall(WALL_ID, flight.envelope, flight.sentence.words, TRAINING_FLOWN_COLOR, 0.16);
      // ONE FRUSTUM PER WORD. It is a `wall` around the closed outline rather than
      // an extruded `polygon`, because a polygon takes ONE `extrudedHeight` and
      // this solid does not have one: its lid slopes. The wall carries a height
      // pair per position, which is exactly the shape, and its two radial faces
      // ARE the trapezoids. The lid is drawn on top of it, per position.
      flight.envelope.events.forEach((box, index) => {
        const wall = trainingBoxWall(box);
        const sides = `${BOX_ID}-${index}`;
        added.push(sides);
        viewer.entities.add({
          id: sides,
          wall: {
            positions: Cesium.Cartesian3.fromDegreesArray(wall.positions),
            maximumHeights: wall.maximumHeights,
            minimumHeights: wall.minimumHeights,
            material: Cesium.Color.fromCssColorString(TRAINING_FLOWN_COLOR).withAlpha(0.1),
            outline: true,
            outlineColor: Cesium.Color.fromCssColorString(TRAINING_FLOWN_COLOR).withAlpha(0.45),
          },
        });
        const lid = `${BOX_ID}-lid-${index}`;
        added.push(lid);
        const top: number[] = [];
        for (let point = 0; point < box.lon.length; point += 1) {
          top.push(box.lon[point], box.lat[point], box.altHaeHiM[point]);
        }
        viewer.entities.add({
          id: lid,
          polygon: {
            hierarchy: new Cesium.PolygonHierarchy(Cesium.Cartesian3.fromDegreesArrayHeights(top)),
            perPositionHeight: true,
            material: Cesium.Color.fromCssColorString(TRAINING_FLOWN_COLOR).withAlpha(0.07),
          },
        });
      });
    }

    added.push(OBSERVED_ID);
    viewer.entities.add({
      id: OBSERVED_ID,
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArrayHeights(degreesArrayHeights(flight.observed)),
        width: 3,
        material: Cesium.Color.fromCssColorString(TRAINING_TRACE_COLOR),
        // The track must not be hidden by terrain: a track that runs into a hill
        // is a reading, not a rendering accident. `depthFailMaterial` is what
        // says so — an unclamped polyline still loses the depth test.
        depthFailMaterial: new Cesium.PolylineDashMaterialProperty({
          color: Cesium.Color.fromCssColorString(TRAINING_TRACE_COLOR).withAlpha(0.55),
        }),
      },
    });

    // WHAT THE MODEL SAID, when the set carries it: the wedge ITS altitude words
    // would have allowed, over the same ground track — purple, never the
    // envelope's orange, because one is the reading and the other is the thing
    // being judged. Its boxes are not drawn as prisms: two chains of thin prisms
    // in one scene is one fuzzy chain, and the comparison that answers anything
    // is between the two walls.
    if (flight.prior && trainingLayers.model) {
      drawWall(MODEL_WALL_ID, flight.prior.envelope, flight.prior.words, TRAINING_MODEL_COLOR, 0.14);
    }

    // The nodes go LAST, over everything: they are POINTS rather than a second
    // polyline, because a polyline through the same positions would be the track
    // again, and what is wanted is where it was cut.
    trainingSegmentNodes(flight.observed, flight.sentence.eventTimesS).forEach((node, index) => {
      const id = `${NODE_ID}-${index}`;
      added.push(id);
      viewer.entities.add({
        id,
        position: Cesium.Cartesian3.fromDegrees(node.lon, node.lat, node.altHaeM),
        point: {
          pixelSize: 9,
          color: Cesium.Color.fromCssColorString(TRAINING_FLOWN_COLOR),
          outlineColor: Cesium.Color.BLACK.withAlpha(0.6),
          outlineWidth: 2,
          // Same reason the track carries `depthFailMaterial`: a node hidden by
          // terrain is a cut the reader cannot see.
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
      });
    });

    return () => {
      if (!isCesiumViewerUsable(viewer)) return;
      for (const id of added) viewer.entities.removeById(id);
    };
  }, [viewer, mode, trainingSelection, trainingLayers]);
}
