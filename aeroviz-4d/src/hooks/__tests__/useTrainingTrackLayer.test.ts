/**
 * The 3D layer's plumbing: the exporter's coordinates handed to Cesium unchanged — the track at
 * its ellipsoid height, a plan line as [lon, lat, …], a tube as a wall over the track's own rows —
 * and what ONE selected word lights up: its own column's envelope and the rows it is in force.
 */
import { describe, expect, it } from "vitest";

import {
  planDegrees,
  planRingDegrees,
  trainingFocusEntities,
  trainingFocusStretch,
  trainingTrackPositions,
  trainingTubeWall,
  TRAINING_ENTITY,
} from "../useTrainingTrackLayer";
import { parseTrainingSample, trainingWordAt, type TrainingSelection } from "../../data/trainingSample";
import { mockSample } from "../../data/__tests__/trainingSample.fixture";

function selection(position = 0): TrainingSelection {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  const { vocabulary, candidates, flights } = parsed.value;
  return { vocabulary, candidates, flight: flights[position] };
}

function flight(position = 0) {
  return selection(position).flight;
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

  it("closes an exported ring, which is open, for its outline", () => {
    const outline = flight().envelopes.approach.corridor.outline;
    const ring = planRingDegrees(outline);
    expect(ring.slice(0, -2)).toEqual(planDegrees(outline));
    expect(ring.slice(-2)).toEqual([outline.lon[0], outline.lat[0]]);
  });

  it("lights up the selected word's OWN envelope, never another column's", () => {
    const scene = selection();
    const at = (column: Parameters<typeof trainingWordAt>[1], row: number) =>
      trainingFocusEntities(scene, column, trainingWordAt(scene.flight, column, row));
    const turn = TRAINING_ENTITY.turn(1);
    expect(at("heading", 15)).toEqual([turn, TRAINING_ENTITY.path(turn, "fast"), TRAINING_ENTITY.path(turn, "slow"),
      TRAINING_ENTITY.turnEnd(1), TRAINING_ENTITY.funnel(1)]);
    // after the capture a heading word is still its own turn and funnel: the corridor is the clearance's
    expect(at("heading", 50)).toEqual(at("heading", 15));
    expect(at("altitude", 15)).toEqual([TRAINING_ENTITY.tube(0)]);
    const capture = TRAINING_ENTITY.captureTurn;
    expect(at("approach", 15)).toEqual([capture, TRAINING_ENTITY.path(capture, "fast"), TRAINING_ENTITY.path(capture, "slow"),
      TRAINING_ENTITY.captureTurnEnd, TRAINING_ENTITY.corridor, TRAINING_ENTITY.corridorAxis]);
    expect(at("approach", 5)).toEqual([]);   // "not cleared" bounds nothing
    expect(at("runway", 5)).toEqual([TRAINING_ENTITY.runway("09"), TRAINING_ENTITY.centreline("09")]);
    expect(at("angle", 30)).toEqual([]);
    expect(at("speed", 30)).toEqual([]);
  });

  it("draws the rows a word is in force, on to the next word's issue", () => {
    const item = flight();
    const altitude = trainingWordAt(item, "altitude", 5);
    expect(trainingFocusStretch(item, altitude)).toEqual(trainingTrackPositions(item).slice(0, 21 * 3));
    // the last word runs to the last row
    const last = trainingWordAt(item, "altitude", 30);
    expect(trainingFocusStretch(item, last)).toEqual(trainingTrackPositions(item).slice(20 * 3));
  });
});
