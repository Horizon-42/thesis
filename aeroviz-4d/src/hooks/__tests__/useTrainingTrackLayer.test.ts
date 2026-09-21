/**
 * The 3D layer's arithmetic (T6 / T15): the track flattened into Cesium's
 * [lon, lat, height, …], and the ENVELOPE the sentence allows around it — the
 * wedge as a wall, and one prism per word.
 */
import { describe, expect, it } from "vitest";

import {
  trainingBandWall,
  trainingBoxWall,
  trainingSegmentNodes,
  trainingTrackPositions,
} from "../useTrainingTrackLayer";
import { eventInForce, parseTrainingSample } from "../../data/trainingSample";
import { MOCK_VOCABULARY, mockPriorSample, mockSample } from "../../data/__tests__/trainingSample.fixture";

function flight(raw: unknown = mockSample(), kind: "vocabulary-readback" | "prior-generated" = "vocabulary-readback") {
  const parsed = parseTrainingSample(raw, kind);
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value.flights[0];
}

/** The wall needs the vocabulary: a ribbon closes on its own segment's TARGET, and
 *  only the vocabulary knows where that is. */
function vocabulary(_flight: unknown) {
  void _flight;
  return MOCK_VOCABULARY;
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
  //
  // It comes back as ONE RIBBON PER ALTITUDE SEGMENT. The bound STEPS at every
  // altitude word — a segment closes onto its target, the next opens wide again —
  // and a single wall forced through those steps renders them as twisted facets
  // over the turning ground track.
  it("splits into one ribbon per altitude segment, covering every row", () => {
    const item = flight();
    const walls = trainingBandWall(item, item.envelope, vocabulary(item), item.sentence.words);
    // as many ribbons as the altitude column has runs over the rows
    const forced = eventInForce(item.sentence.eventTimesS, item.observed.tS);
    const perRow = forced.map((event) => item.sentence.words[event][1]);
    const runs = perRow.filter((word, row) => row === 0 || word !== perRow[row - 1]).length;
    expect(walls).toHaveLength(runs);
    expect(walls.length).toBeGreaterThan(1);
    walls.forEach((wall, index) => {
      expect(wall.positions).toHaveLength(wall.maximumHeights.length * 2);
      expect(wall.maximumHeights).toHaveLength(wall.minimumHeights.length);
      wall.maximumHeights.forEach((top, row) => {
        expect(top).toBeGreaterThanOrEqual(wall.minimumHeights[row]);
      });
      // consecutive ribbons SHARE their boundary row, or a hairline of unbounded
      // height opens between two segments that in fact meet
      if (index + 1 < walls.length) {
        const endLon = wall.positions[wall.positions.length - 2];
        expect(walls[index + 1].positions[0]).toBeCloseTo(endLon, 9);
      }
    });
  });

  it("rides the aircraft's OWN ground track — the wedge is a height bound, nothing else", () => {
    const item = flight();
    const walls = trainingBandWall(item, item.envelope, vocabulary(item), item.sentence.words);
    expect(walls[0].positions.slice(0, 2)).toEqual([item.observed.lon[0], item.observed.lat[0]]);
  });

  it("each ribbon narrows toward its own segment's end, which is where the target is", () => {
    const item = flight();
    for (const wall of trainingBandWall(item, item.envelope, vocabulary(item), item.sentence.words)) {
      const width = (row: number) => wall.maximumHeights[row] - wall.minimumHeights[row];
      expect(width(wall.maximumHeights.length - 1)).toBeLessThan(width(0));
    }
  });

  it("takes the MODEL's envelope when it is handed one — a different sentence, a different wall", () => {
    const item = flight(mockPriorSample(), "prior-generated");
    const truth = trainingBandWall(item, item.envelope, vocabulary(item), item.sentence.words);
    const said = trainingBandWall(item, item.prior!.envelope, vocabulary(item), item.prior!.words);
    expect(said.map((wall) => wall.maximumHeights.join())).not.toEqual(
      truth.map((wall) => wall.maximumHeights.join()));
  });
});

describe("the chain of solids", () => {
  it("carries one per word, with a height pair at every point of its outline", () => {
    const item = flight();
    expect(item.envelope.events).toHaveLength(item.sentence.eventTimesS.length);
    for (const box of item.envelope.events) {
      const points = box.lon.length;
      expect(points).toBeGreaterThanOrEqual(3);
      for (const array of [box.lat, box.altLoM, box.altHiM, box.altHaeLoM, box.altHaeHiM]) {
        expect(array).toHaveLength(points);
      }
      box.altHaeHiM.forEach((top, point) => expect(top).toBeGreaterThanOrEqual(box.altHaeLoM[point]));
    }
  });

  it("each solid TAPERS: its apex is the tall end", () => {
    const item = flight();
    // an outline point `d` metres out has `d` metres less path left to its segment's
    // end, so its slice of the wedge is tighter. A flat lid is the one thing the
    // words never say, and it over-states the ceiling at the far end.
    for (const box of item.envelope.events) {
      const tall = box.altHiM[0] - box.altLoM[0];
      for (let point = 1; point < box.altHiM.length; point += 1) {
        expect(box.altHiM[point] - box.altLoM[point]).toBeLessThanOrEqual(tall + 1e-6);
      }
    }
    // and on this fixture at least one of them visibly does
    const closes = item.envelope.events.map(
      (box) => (box.altHiM[0] - box.altLoM[0]) - (box.altHiM[1] - box.altLoM[1]));
    expect(Math.max(...closes)).toBeGreaterThan(1);
  });

  it("the wall Cesium draws closes the outline back onto the apex", () => {
    const item = flight();
    const box = item.envelope.events[0];
    const wall = trainingBoxWall(box);
    // one more position than the outline has points: the apex is repeated, so the
    // two radial faces — the trapezoids — are drawn as well as the arc face
    expect(wall.maximumHeights).toHaveLength(box.lon.length + 1);
    expect(wall.positions.slice(0, 2)).toEqual(wall.positions.slice(-2));
    expect(wall.maximumHeights[0]).toBe(wall.maximumHeights[wall.maximumHeights.length - 1]);
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
