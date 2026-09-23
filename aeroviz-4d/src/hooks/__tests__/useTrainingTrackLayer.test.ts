/**
 * The 3D layer's plumbing: the exporter's coordinates handed to Cesium unchanged — the track at
 * its ellipsoid height, a plan line as [lon, lat, …], a tube as a wall over the track's own rows —
 * and which envelopes are in force at a time.
 */
import { describe, expect, it } from "vitest";

import {
  planDegrees,
  trainingEnvelopeInForce,
  trainingTrackPositions,
  trainingTubeWall,
} from "../useTrainingTrackLayer";
import { parseTrainingSample } from "../../data/trainingSample";
import { mockSample } from "../../data/__tests__/trainingSample.fixture";

function flight(position = 0) {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value.flights[position];
}

describe("useTrainingTrackLayer helpers", () => {
  it("flattens the track to lon, lat and its ellipsoid height, never the MSL altitude", () => {
    const item = flight();
    const positions = trainingTrackPositions(item);
    expect(positions).toHaveLength(item.rows * 3);
    expect(positions.slice(0, 3)).toEqual([item.signals.lon[0], item.signals.lat[0], item.signals.altitudeHaeM[0]]);
    expect(positions[2]).not.toBe(item.signals.smoothed.altitudeM[0]);
  });

  it("hands a plan line over as [lon, lat, …]", () => {
    const item = flight();
    const outline = item.envelopes.approach.corridor.outline;
    expect(planDegrees(outline)).toEqual(outline.lon.flatMap((lon, point) => [lon, outline.lat[point]]));
  });

  it("walls a tube over exactly the rows it covers, with its own edges", () => {
    const item = flight();
    const tube = item.envelopes.altitude[1];
    const wall = trainingTubeWall(item, tube);
    expect(wall.positions).toHaveLength((tube.endRow - tube.row) * 2);
    expect(wall.positions.slice(0, 2)).toEqual([item.signals.lon[tube.row], item.signals.lat[tube.row]]);
    expect(wall.minimumHeights).toBe(tube.lowerHaeM);
    expect(wall.maximumHeights).toBe(tube.upperHaeM);
  });

  it("finds the envelopes in force: a heading word's before the capture, the corridor (-1) after", () => {
    const item = flight();
    expect(trainingEnvelopeInForce(item, 0)).toEqual({ heading: 0, altitude: 0 });
    expect(trainingEnvelopeInForce(item, 20)).toEqual({ heading: 1, altitude: 0 });
    expect(trainingEnvelopeInForce(item, 40)).toEqual({ heading: 1, altitude: 1 });
    expect(trainingEnvelopeInForce(item, 50)).toEqual({ heading: -1, altitude: 1 });
    expect(trainingEnvelopeInForce(flight(1), 0)).toEqual({ heading: -1, altitude: 0 });
  });
});
