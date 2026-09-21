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
 *    vocabulary that pins a position rather than a rate.
 *  • The BOXES are one prism per word: the ground its heading and speed words
 *    allow while it stands, extruded over the wedge's own range there. They are
 *    thin horizontally — a 2° heading box over a 4 s hold is about ten metres of
 *    cross-track — and that thinness is the finding: horizontally one word says
 *    almost nothing, and it is the accumulation over a whole sentence that opens
 *    the funnel (design §5.1, still open).
 *
 * The box footprint is a DERIVED reachable set, not the word: a word constrains
 * the state at every instant, and where that lets the aircraft go over its hold
 * is this. The corners are computed at the exporter, in the course frame, because
 * the frame's transform lives on that side of the wire.
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
import type { TrainingEnvelope, TrainingFlight } from "../data/trainingSample";

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
export function trainingBandWall(flight: TrainingFlight, envelope: TrainingEnvelope): {
  positions: number[];
  maximumHeights: number[];
  minimumHeights: number[];
} {
  const { lon, lat } = flight.observed;
  const positions: number[] = [];
  for (let row = 0; row < lon.length; row += 1) positions.push(lon[row], lat[row]);
  return {
    positions,
    maximumHeights: envelope.altHaeHiM,
    minimumHeights: envelope.altHaeLoM,
  };
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
    const drawWall = (id: string, envelope: TrainingEnvelope, colour: string, alpha: number) => {
      const wall = trainingBandWall(flight, envelope);
      added.push(id);
      viewer.entities.add({
        id,
        wall: {
          positions: Cesium.Cartesian3.fromDegreesArray(wall.positions),
          maximumHeights: wall.maximumHeights,
          minimumHeights: wall.minimumHeights,
          material: Cesium.Color.fromCssColorString(colour).withAlpha(alpha),
          outline: false,
        },
      });
    };

    // The envelope goes in FIRST, so the track reads over it rather than under
    // it: it is the region the words allow, and the track is what is being
    // checked against it.
    if (trainingLayers.flown) {
      drawWall(WALL_ID, flight.envelope, TRAINING_FLOWN_COLOR, 0.16);
      // ONE PRISM PER WORD. `perPositionHeight` is deliberately not used: the
      // footprint is flat and the two heights are the wedge's range over that
      // word's own rows, so `height` / `extrudedHeight` say exactly that.
      flight.envelope.events.forEach((box, index) => {
        const id = `${BOX_ID}-${index}`;
        added.push(id);
        const corners: number[] = [];
        for (let corner = 0; corner < box.lon.length; corner += 1) {
          corners.push(box.lon[corner], box.lat[corner]);
        }
        viewer.entities.add({
          id,
          polygon: {
            hierarchy: new Cesium.PolygonHierarchy(Cesium.Cartesian3.fromDegreesArray(corners)),
            height: box.altHaeLoM,
            extrudedHeight: box.altHaeHiM,
            material: Cesium.Color.fromCssColorString(TRAINING_FLOWN_COLOR).withAlpha(0.08),
            outline: true,
            outlineColor: Cesium.Color.fromCssColorString(TRAINING_FLOWN_COLOR).withAlpha(0.5),
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
      drawWall(MODEL_WALL_ID, flight.prior.envelope, TRAINING_MODEL_COLOR, 0.14);
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
