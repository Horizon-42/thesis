/**
 * The 3D layer's plumbing: the exporter's coordinates handed to Cesium unchanged — the track at
 * its ellipsoid height, a plan line as [lon, lat, …], a tube as a wall over the track's own rows, a
 * heading word's judged rows and its rows outside on the ground — and what ONE selected word lights
 * up: its own column's envelope and the rows it is in force.
 */
import { describe, expect, it } from "vitest";

import {
  executorTrackPositions,
  planDegrees,
  planRingDegrees,
  trainingBandGround,
  trainingBandOutsideGround,
  trainingEnvelopeEntities,
  trainingFocusEntities,
  trainingFocusStretch,
  trainingTrackPositions,
  trainingTubeWall,
  TRAINING_ENTITY,
} from "../useTrainingTrackLayer";
import { parseTrainingSample, trainingWordAt, type TrainingSelection } from "../../data/trainingSample";
import { mockSample } from "../../data/__tests__/trainingSample.fixture";
import { mockExecutorOverlay } from "../../data/__tests__/trainingOverlays.fixture";
import { parseTrainingExecutorOverlay } from "../../data/trainingOverlays";

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

  it("flattens the executor's flown track at its ellipsoid height", () => {
    const parsed = parseTrainingSample(mockSample());
    if (!parsed.ok) throw new Error(parsed.problem);
    const overlay = parseTrainingExecutorOverlay(mockExecutorOverlay(), parsed.value);
    if (!overlay.ok) throw new Error(overlay.problem);
    const track = overlay.value.flights[0].track!;
    const positions = executorTrackPositions(track);
    expect(positions).toHaveLength(track.lon.length * 3);
    expect(positions.slice(0, 3)).toEqual([track.lon[0], track.lat[0], track.altitudeHaeM[0]]);
    expect(positions[2]).not.toBe(track.altitudeM[0]);
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

  it("drapes a heading word's judged rows on the ground, on to where the next word's begin", () => {
    const item = flight();
    const { lon, lat } = item.signals;
    const [first, second, third] = item.envelopes.heading;
    const rows = (from: number, to: number) => Array.from({ length: to - from + 1 }, (_, k) => [lon[from + k], lat[from + k]]).flat();
    expect(trainingBandGround(lon, lat, first)).toEqual(rows(2, 10));
    expect(trainingBandGround(lon, lat, second)).toEqual(rows(10, 12));
    expect(trainingBandGround(lon, lat, third)).toEqual(rows(12, 20));
    // a word with no row of its own has nothing on the ground
    expect(trainingBandGround(lon, lat, flight(1).envelopes.heading[0])).toEqual([]);
  });

  it("draws a band's rows outside as segments on the ground, one row outside on to the next", () => {
    const item = flight();
    const { lon, lat } = item.signals;
    expect(trainingBandOutsideGround(lon, lat, item.envelopes.heading[2])).toEqual([[lon[12], lat[12], lon[13], lat[13]]]);
    expect(trainingBandOutsideGround(lon, lat, item.envelopes.heading[0])).toEqual([]);
    // a run to the band's end, and a run of one row at the line's very end (back to the row before it)
    const band = { firstRow: 57, stopRow: 60, targetOnTrackDeg: 90, bandDeg: [85.5, 94.5] as [number, number], inside: [true, false, false] };
    expect(trainingBandOutsideGround(lon, lat, band)).toEqual([[lon[58], lat[58], lon[59], lat[59]]]);
    const last = { ...band, firstRow: 59, stopRow: 60, inside: [false] };
    expect(trainingBandOutsideGround(lon, lat, last)).toEqual([[lon[58], lat[58], lon[59], lat[59]]]);
  });

  it("lights up the selected word's OWN envelope, never another column's", () => {
    const scene = selection();
    const at = (column: Parameters<typeof trainingWordAt>[1], row: number) =>
      trainingFocusEntities(scene, column, trainingWordAt(scene.flight, column, row));
    expect(at("heading", 15)).toEqual([TRAINING_ENTITY.heading(2)]);
    expect(at("heading", 9)).toEqual([TRAINING_ENTITY.heading(1)]);
    // after the capture a heading word is still its own: the corridor is the clearance's
    expect(at("heading", 50)).toEqual(at("heading", 15));
    expect(at("altitude", 15)).toEqual([TRAINING_ENTITY.tube(0)]);
    expect(at("approach", 25)).toEqual([TRAINING_ENTITY.captureTurn, TRAINING_ENTITY.corridor, TRAINING_ENTITY.corridorAxis]);
    expect(at("approach", 5)).toEqual([]);   // "not cleared" bounds nothing
    expect(at("runway", 5)).toEqual([TRAINING_ENTITY.runway("09"), TRAINING_ENTITY.centreline("09")]);
    expect(at("angle", 30)).toEqual([]);
    expect(at("speed", 30)).toEqual([]);
    expect(trainingEnvelopeEntities(scene.flight)).toEqual([
      TRAINING_ENTITY.heading(0), TRAINING_ENTITY.heading(1), TRAINING_ENTITY.heading(2),
      TRAINING_ENTITY.captureTurn, TRAINING_ENTITY.corridor, TRAINING_ENTITY.corridorAxis,
      TRAINING_ENTITY.tube(0), TRAINING_ENTITY.tube(1),
    ]);
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
