/**
 * The 3D layer's arithmetic (T6 / T15): the track flattened into Cesium's
 * [lon, lat, height, …], and the ENVELOPE the sentence allows around it — the
 * wedge as a wall, and one prism per word.
 */
import { describe, expect, it } from "vitest";

import {
  trainingBandWall,
  trainingSegmentNodes,
  trainingTrackPositions,
} from "../useTrainingTrackLayer";
import { parseTrainingSample } from "../../data/trainingSample";
import { mockPriorSample, mockSample } from "../../data/__tests__/trainingSample.fixture";

function flight(raw: unknown = mockSample(), kind: "vocabulary-readback" | "prior-generated" = "vocabulary-readback") {
  const parsed = parseTrainingSample(raw, kind);
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value.flights[0];
}

describe("trainingTrackPositions", () => {
  it("flattens the track to lon, lat, height per row", () => {
    const item = flight();
    const { observed } = trainingTrackPositions(item);
    expect(observed).toHaveLength(item.observed.tS.length * 3);
    expect(observed.slice(0, 3)).toEqual([
      item.observed.lon[0],
      item.observed.lat[0],
      item.observed.altHaeM[0],
    ]);
  });

  // The altitude handed to Cesium is HAE, because that is what
  // `cartographicDegrees` means. Passing the MSL a record carries would float
  // everything ~33.5 m above where it belongs — together, so the mistake would be
  // invisible within the scene and visible only against the terrain.
  it("takes the HAE column, not a height above the threshold", () => {
    const item = flight();
    const { observed } = trainingTrackPositions(item);
    expect(observed[2]).toBe(item.observed.altHaeM[0]);
    expect(observed[2]).not.toBe(item.observed.heightM[0]);
  });
});

describe("trainingBandWall", () => {
  // The wall is the altitude words' wedge made visible in space. Both heights are
  // HAE, and `altHaeHiM` is the MAXIMUM — the plain reading of the names, unlike
  // the retired corridor where "lo" named the shallower descent and sat higher.
  it("walls the wedge between the two heights the words allow", () => {
    const item = flight();
    const wall = trainingBandWall(item, item.envelope);
    expect(wall.positions).toHaveLength(item.observed.tS.length * 2);
    expect(wall.maximumHeights).toEqual(item.envelope.altHaeHiM);
    expect(wall.minimumHeights).toEqual(item.envelope.altHaeLoM);
    wall.maximumHeights.forEach((top, row) => {
      expect(top).toBeGreaterThanOrEqual(wall.minimumHeights[row]);
    });
  });

  it("rides the aircraft's OWN ground track — the wedge is a height bound, nothing else", () => {
    const item = flight();
    const wall = trainingBandWall(item, item.envelope);
    expect(wall.positions.slice(0, 2)).toEqual([item.observed.lon[0], item.observed.lat[0]]);
  });

  it("narrows towards the end of each segment, which is where the target is", () => {
    const item = flight();
    const wall = trainingBandWall(item, item.envelope);
    const width = (row: number) => wall.maximumHeights[row] - wall.minimumHeights[row];
    // the last row of the track is the last segment's own end
    expect(width(wall.maximumHeights.length - 1)).toBeLessThan(width(0));
  });

  it("takes the MODEL's envelope when it is handed one — a different sentence, a different wall", () => {
    const item = flight(mockPriorSample(), "prior-generated");
    const truth = trainingBandWall(item, item.envelope);
    const said = trainingBandWall(item, item.prior!.envelope);
    expect(said.positions).toEqual(truth.positions);
    expect(said.maximumHeights).not.toEqual(truth.maximumHeights);
  });
});

describe("the box chain", () => {
  it("carries one prism per word, with a four-corner footprint and two heights", () => {
    const item = flight();
    expect(item.envelope.events).toHaveLength(item.sentence.eventTimesS.length);
    for (const box of item.envelope.events) {
      expect(box.lon).toHaveLength(4);
      expect(box.lat).toHaveLength(4);
      expect(box.altHaeHiM).toBeGreaterThanOrEqual(box.altHaeLoM);
    }
  });

  it("the prism's heights are the wedge's OWN range over that word's rows", () => {
    const item = flight();
    // not the target's ±5 %: the wedge is wide where a segment begins, and the
    // box has to be as tall as the wedge is over the rows it stands for.
    const first = item.envelope.events[0];
    expect(first.altHiM - first.altLoM).toBeGreaterThan(
      2 * 0.05 * (first.altitudeTargetM + 50),
    );
  });
});

describe("trainingSegmentNodes", () => {
  // One node per EVENT, on the track's own rows — this is what makes the track
  // readable as a sentence rather than as a curve.
  it("puts a node on the track at every event", () => {
    const item = flight();
    const nodes = trainingSegmentNodes(item.observed, item.sentence.eventTimesS);
    expect(nodes).toHaveLength(item.sentence.eventTimesS.length);
    expect(nodes[0].eventS).toBe(0);
    expect(nodes[0].lon).toBe(item.observed.lon[0]);
    expect(nodes[0].lat).toBe(item.observed.lat[0]);
  });

  // The row is the last one at or before the event, so a node sits ON the line
  // rather than between two of its points.
  it("takes the row in force at the event, never the next one", () => {
    const item = flight();
    const nodes = trainingSegmentNodes(item.observed, [0, 27]);
    expect(nodes[1].lon).toBe(item.observed.lon[13]);   // rows are 2 s apart
  });

  it("drops an event past the end of the track", () => {
    const item = flight();
    const short = { ...item.observed, tS: item.observed.tS.slice(0, 10) };
    const nodes = trainingSegmentNodes(short, item.sentence.eventTimesS);
    expect(nodes.every((node) => node.eventS <= 18)).toBe(true);
    expect(nodes.length).toBeLessThan(item.sentence.eventTimesS.length);
  });
});
