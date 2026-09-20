/**
 * useTrainingTrackLayer.ts
 * ------------------------
 * The two tracks of the selected Training flight, in the 3D scene: the aircraft's
 * own (white) and the sentence flown by rule (orange). Design:
 * `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §3.1 / T6.
 *
 * STATIC POLYLINES, NOT TIME-SAMPLED ENTITIES. Training loads no CZML on purpose
 * (design V2): a time-dynamic entity would drive the shared `viewer.clock`, and
 * the clock belongs to Observe's playback — switching tasks would move it. Two
 * polylines draw the same information without touching it, and the moment-by-
 * moment reading is the read-back window's job.
 *
 * The altitudes are already HAE (`altHaeM`, converted at the exporter), which is
 * what `Cartesian3.fromDegreesArrayHeights` wants. Feeding it MSL would sink both
 * lines ~33.5 m into KRDU's terrain — together, so the error would be invisible in
 * the comparison and visible only against the ground.
 */

import { useEffect } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { isCesiumViewerUsable } from "../utils/isCesiumViewerUsable";
import { TRAINING_FLOWN_COLOR, TRAINING_TRACE_COLOR } from "../utils/trainingWordColors";
import type { TrainingFlight } from "../data/trainingSample";

const OBSERVED_ID = "training-observed-track";
const FLOWN_ID = "training-flown-track";

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

export default function useTrainingTrackLayer(): void {
  const { viewer, mode, trainingSelection } = useApp();

  useEffect(() => {
    if (!isCesiumViewerUsable(viewer) || !viewer) return;
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
          // terrain: a rule-flown track that ends up inside a hill is a reading,
          // not a rendering accident.
          arcType: Cesium.ArcType.GEODESIC,
          clampToGround: false,
        },
      });

    draw(OBSERVED_ID, observed, TRAINING_TRACE_COLOR, 3);
    draw(FLOWN_ID, flown, TRAINING_FLOWN_COLOR, 2);

    return () => {
      if (!isCesiumViewerUsable(viewer)) return;
      viewer.entities.removeById(OBSERVED_ID);
      viewer.entities.removeById(FLOWN_ID);
    };
  }, [viewer, mode, trainingSelection]);
}
