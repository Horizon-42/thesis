/**
 * useTrainingTrackLayer.ts
 * ------------------------
 * The selected Training flight in the 3D scene: the aircraft's own track (white),
 * the sentence flown by rule (orange), and the CORRIDOR that sentence allows —
 * a wall between the two heights the vertical word's tolerance permits. Design:
 * `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §3.1 / §5.6, T6 / T13.
 *
 * THE WALL IS THE ONE PLACE THE VERTICAL TOLERANCE IS VISIBLE IN SPACE, and it
 * is the only band drawn here. The vertical word's slack accumulates into HEIGHT
 * with distance flown, so it is a shape; the speed word's accumulates into
 * ARRIVAL TIME, which a static scene cannot show at all — its two edge tracks
 * run within ~100 m of the flown line and would read as a thicker line, so they
 * stay on the plan view and in the sentence bar's readout (design §5.6).
 *
 * The wall's two heights share the flown track's OWN lon/lat: the commanded
 * angle never touches the horizontal step, so the edges are the same ground
 * track at different heights — which is why the artefact carries two height
 * columns rather than two tracks.
 *
 * STATIC POLYLINES, NOT TIME-SAMPLED ENTITIES. Training loads no CZML on purpose
 * (design V2): a time-dynamic entity would drive the shared `viewer.clock`, and
 * the clock belongs to Observe's playback — switching tasks would move it. Two
 * polylines draw the same information without touching it, and the moment-by-
 * moment reading is the read-back window's job.
 *
 * The altitudes are already HAE (`altHaeM`, converted at the exporter), which is
 * what `Cartesian3.fromDegreesArrayHeights` wants. Feeding it MSL would float both
 * lines ~33.5 m ABOVE where they belong (h = H + N, and N is negative here) —
 * together, so the error would be invisible in the comparison and visible only
 * against the terrain.
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { TRAINING_FLOWN_COLOR, TRAINING_TRACE_COLOR } from "../utils/trainingWordColors";
import type { TrainingFlight } from "../data/trainingSample";

const OBSERVED_ID = "training-observed-track";
const FLOWN_ID = "training-flown-track";
const BAND_ID = "training-vertical-band";

/** Cesium wants [lon, lat, height, …]; both tracks carry the three as columns. */
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

export function trainingTrackPositions(flight: TrainingFlight): {
  observed: number[];
  flown: number[];
} {
  return {
    observed: degreesArrayHeights(flight.observed),
    flown: degreesArrayHeights(flight.geometric),
  };
}

/**
 * The vertical corridor as a Cesium wall: one ground track, a maximum height per
 * position and a minimum.
 *
 * BOTH ARE HAE, like every other altitude here (`altHaeLoM` / `altHaeHiM`). The
 * shallower edge is the maximum, and it is `lo` — "lo" names the smaller descent
 * angle, which loses less height and therefore stays HIGHER. Reading the names
 * as heights would turn the wall inside out while leaving it exactly as thick.
 */
export function trainingBandWall(flight: TrainingFlight): {
  positions: number[];
  maximumHeights: number[];
  minimumHeights: number[];
} {
  const { lon, lat, verticalBand } = flight.geometric;
  const positions: number[] = [];
  for (let row = 0; row < lon.length; row += 1) positions.push(lon[row], lat[row]);
  return {
    positions,
    maximumHeights: verticalBand.altHaeLoM,
    minimumHeights: verticalBand.altHaeHiM,
  };
}

export default function useTrainingTrackLayer(): void {
  const { viewer, mode, trainingSelection } = useApp();

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer)) return;
    const flight = mode === "training" ? trainingSelection?.flight : undefined;
    if (!flight) return;

    const { observed, flown } = trainingTrackPositions(flight);
    const draw = (id: string, positions: number[], colour: string, width: number) =>
      viewer.entities.add({
        id,
        polyline: {
          positions: Cesium.Cartesian3.fromDegreesArrayHeights(positions),
          width,
          material: Cesium.Color.fromCssColorString(colour),
          // The comparison is between the two lines, so neither may be hidden by
          // terrain: a rule-flown sentence that descends into a hill is a
          // reading, not a rendering accident. `depthFailMaterial` is what says
          // so — an unclamped polyline still loses the depth test, and the
          // defaults alone would silently swallow exactly that case.
          depthFailMaterial: new Cesium.PolylineDashMaterialProperty({
            color: Cesium.Color.fromCssColorString(colour).withAlpha(0.55),
          }),
        },
      });

    // The corridor goes in FIRST, so the two lines read over it rather than
    // under it: it is the context for the orange line, not a third track.
    const wall = trainingBandWall(flight);
    viewer.entities.add({
      id: BAND_ID,
      wall: {
        positions: Cesium.Cartesian3.fromDegreesArray(wall.positions),
        maximumHeights: wall.maximumHeights,
        minimumHeights: wall.minimumHeights,
        material: Cesium.Color.fromCssColorString(TRAINING_FLOWN_COLOR).withAlpha(0.16),
        outline: false,
      },
    });
    draw(OBSERVED_ID, observed, TRAINING_TRACE_COLOR, 3);
    draw(FLOWN_ID, flown, TRAINING_FLOWN_COLOR, 2);

    return () => {
      if (!isCesiumViewerUsable(viewer)) return;
      viewer.entities.removeById(OBSERVED_ID);
      viewer.entities.removeById(FLOWN_ID);
      viewer.entities.removeById(BAND_ID);
    };
  }, [viewer, mode, trainingSelection]);
}
