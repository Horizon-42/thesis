/**
 * The 3D scene's coordinates and the word each envelope belongs to (`scene/trainingEntities.ts`), on the stage-A fixture:
 * what is drawn is the exporter's, only put in Cesium's flat arrays.
 */
import { describe, expect, it } from "vitest";
import {
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
