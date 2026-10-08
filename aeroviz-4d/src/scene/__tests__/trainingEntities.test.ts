/**
 * The 3D scene's coordinates and the word each envelope belongs to (`scene/trainingEntities.ts`), on the stage-A fixture:
 * what is drawn is the exporter's, only put in Cesium's flat arrays.
 */
import { describe, expect, it } from "vitest";
import * as Cesium from "cesium";
import {
  airLine,
  GROUND_LINE_ALPHA,
  GROUND_LINE_WIDTH,
  groundLine,
  groundRows,
  lonLatHeights,
  planDegrees,
  TRAINING_ENTITY,
  trainingBandGround,
  trainingBandOutsideGround,
  trainingEnvelopeEntities,
  trainingFocusEntity,
  trainingFocusStretch,
  trainingTubeWall,
} from "../trainingEntities";
import { sentenceColumnRuns, trainingReadingOf } from "../../data/trainingSample";
import { stageASample } from "../../data/__tests__/stageA";
import { stageBSample } from "../../data/__tests__/stageB";
import { stageCSample, stageCSampleFromRound } from "../../data/__tests__/stageC";
import { trainingPriorFlightView } from "../../data/trainingPriorSample";
import { trainingWindowFlightView } from "../../data/trainingWindowSample";
import { flownSentenceColour, flownSentenceKind, roundKind } from "../../data/trainingSentenceKind";
import {
  TRAINING_DECISION_PASS_COLOR,
  TRAINING_FAILURE_COLOR,
  TRAINING_SENTENCE_COLOR,
  trainingOutcomeColour,
} from "../../utils/trainingWordColors";

const sample = stageASample();
const [flight] = sample.flights;
const stepS = sample.vocabulary.stepS;

describe("the coordinates", () => {
  it("lays longitude, latitude and ellipsoid height out as Cesium's flat arrays", () => {
    const track = { lon: [1, 2], lat: [3, 4], altitudeHaeM: [5, 6] };
    expect(lonLatHeights(track)).toEqual([1, 3, 5, 2, 4, 6]);
    expect(planDegrees(track)).toEqual([1, 3, 2, 4]);
    expect(groundRows([1, 2, 3], [4, 5, 6], 1, 2)).toEqual([2, 5, 3, 6]);
    expect(groundRows([1, 2, 3], [4, 5, 6], 1, 1)).toEqual([]);
  });

  it("draws a band's judged rows on the ground and its rows outside as segments of them", () => {
    const reading = trainingReadingOf(flight, stepS, 2);
    const band = reading.envelopes!.heading[0];
    expect(trainingBandGround(reading.judged, band)).toHaveLength((band.stopRow - band.firstRow + 1) * 2);
    const outside = trainingBandOutsideGround(reading.judged, { ...band, inside: band.inside.map((_, i) => i !== 1) });
    expect(outside).toHaveLength(1);
    expect(outside[0]).toHaveLength(4);
    expect(trainingBandGround(reading.judged, { ...band, stopRow: band.firstRow })).toEqual([]);
  });

  it("puts a tube on the judged track's ground position, at the exported edges plus the flight's HAE − MSL", () => {
    const reading = trainingReadingOf(flight, stepS, null);
    const tube = reading.envelopes!.altitude[0];
    const wall = trainingTubeWall(reading.judged, tube, flight.haeMinusMslM);
    expect(wall.minimumHeights).toHaveLength(tube.endRow - tube.row);
    expect(wall.minimumHeights[0]).toBeCloseTo(tube.lowMslM[0] + flight.haeMinusMslM, 6);
    expect(wall.positions.slice(0, 2)).toEqual([reading.judged.lon[tube.row], reading.judged.lat[tube.row]]);
  });
});

describe("the selected word", () => {
  it("owns its heading band or its tube, and nothing for a column without an envelope", () => {
    const reading = trainingReadingOf(flight, stepS, 4);
    const heading = sentenceColumnRuns(reading, "heading").flatMap((run) => {
      const id = trainingFocusEntity(reading, stepS, "heading", run);
      return id === null ? [] : [id];
    });
    expect(heading).toEqual(reading.envelopes!.heading.map((_, index) => TRAINING_ENTITY.heading(index)));
    const altitude = sentenceColumnRuns(reading, "altitude").map((run) => trainingFocusEntity(reading, stepS, "altitude", run));
    expect(altitude.filter((id) => id !== null)).toEqual(reading.envelopes!.altitude.map((_, index) => TRAINING_ENTITY.tube(index)));
    expect(trainingFocusEntity(reading, stepS, "angle", sentenceColumnRuns(reading, "angle")[0])).toBeNull();
    expect(trainingEnvelopeEntities(reading)).toHaveLength(reading.envelopes!.heading.length + reading.envelopes!.altitude.length);
  });

  it("gives the stretch of the judged track a word is in force", () => {
    const reading = trainingReadingOf(flight, stepS, 2);
    const stretch = trainingFocusStretch(reading.judged, reading.originS + 10, reading.originS + 20);
    expect(stretch).toHaveLength(6 * 3);
    expect(trainingFocusStretch(reading.judged, reading.originS + 10, reading.originS + 10)).toEqual([]);
    expect(trainingFocusStretch(reading.judged, 1e6, 2e6)).toEqual([]);
  });
});

describe("the envelopes of every stage's flown sentences (D135)", () => {
  it("a sentence of stage B and a round of stage C get their heading bands and tubes in 3D, on their own flown track", () => {
    const b = stageBSample();
    const c = stageCSample();
    const readings = [
      trainingReadingOf(trainingPriorFlightView(b, b.flights[0], 0), b.vocabulary.stepS, b.model.rowIntervalS),
      trainingReadingOf(trainingWindowFlightView(c, c.windows[0], c.windows[0].commanded[0], c.windows[0].rounds[c.windows[0].rounds.length - 1].round),
        c.vocabulary.stepS, c.model.rowIntervalS),
    ];
    for (const reading of readings) {
      expect(reading.loop).toBe("closed");
      expect(reading.envelopes!.heading.length).toBeGreaterThan(0);
      expect(trainingEnvelopeEntities(reading)).toHaveLength(reading.envelopes!.heading.length + reading.envelopes!.altitude.length);
      const band = reading.envelopes!.heading.find((one) => one.stopRow > one.firstRow)!;
      expect(trainingBandGround(reading.judged, band).length).toBeGreaterThan(0);              // its judged rows on the ground
    }
  });
});

describe("a line on the ground never reads as a track in the air (frontend §3 item 6, D159)", () => {
  it("is dashed, at most 3 px wide and at 0.6 opacity; a line in the air stays solid and opaque", () => {
    const time = Cesium.JulianDate.now();
    for (const [line, width] of [[groundLine("g", "ground", [0, 0, 1, 1], "#7dd3fc"), GROUND_LINE_WIDTH],
      [groundLine("t", "trace", [0, 0, 1, 1], "#e2e8f0", true), 2]] as const) {
      const material = line.polyline!.material as Cesium.PolylineDashMaterialProperty;
      expect(material).toBeInstanceOf(Cesium.PolylineDashMaterialProperty);
      expect((material.color!.getValue(time) as Cesium.Color).alpha).toBeCloseTo(GROUND_LINE_ALPHA);
      expect(line.polyline!.width).toBe(width);
      expect(width).toBeLessThanOrEqual(3);
      expect(line.polyline!.clampToGround).toBe(true);
    }
    expect(GROUND_LINE_ALPHA).toBe(0.6);
    const air = airLine("a", "air", [], "#14b8a6", 3);
    expect((air.polyline!.material as Cesium.Color).alpha).toBe(1);
  });
});

describe("each kind of sentence has its colour (frontend §3 item 11, D159)", () => {
  it("the closed loop teal, the base's sample and a campaign's start from the base magenta, a post-trained round (or a start from one) yellow-green", () => {
    const b = stageBSample();
    const c = stageCSample();
    expect(flownSentenceKind(flight)).toBe("closedLoop");
    expect(flownSentenceKind(trainingPriorFlightView(b, b.flights[0], 0))).toBe("base");
    expect(flownSentenceKind(trainingPriorFlightView(b, b.flights[0], "closedLoop"))).toBe("closedLoop");
    expect(flownSentenceKind(trainingWindowFlightView(c, c.windows[0], c.windows[0].commanded[0], "start"))).toBe("base");
    expect(roundKind(3, null, "C")).toBe("postTrained");
    // a campaign that starts from another campaign's round (D162): its start is a post-trained model
    const d = stageCSampleFromRound();
    expect(roundKind("start", d.model.start, "C")).toBe("postTrained");
    expect(flownSentenceKind(trainingWindowFlightView(d, d.windows[0], d.windows[0].commanded[0], "start"))).toBe("postTrained");
    expect(TRAINING_SENTENCE_COLOR).toEqual({
      observed: "#e2e8f0", closedLoop: "#14b8a6", base: "#d946ef", postTrained: "#a3e635", multi: "#b82e7a" });
    expect(flownSentenceColour(trainingPriorFlightView(b, b.flights[0], 0))).toBe("#d946ef");
  });

  it("a landed outcome is the pass green, any other the failure red: teal means only the closed loop (D160 (12))", () => {
    expect(trainingOutcomeColour("landed")).toBe(TRAINING_DECISION_PASS_COLOR);
    expect(trainingOutcomeColour("landed")).not.toBe(TRAINING_SENTENCE_COLOR.closedLoop);
    for (const outcome of ["timeout", "ground_contact", "lost_separation"] as const) {
      expect(trainingOutcomeColour(outcome)).toBe(TRAINING_FAILURE_COLOR);
    }
  });
});
