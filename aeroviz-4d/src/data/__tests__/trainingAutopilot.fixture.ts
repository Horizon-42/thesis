/**
 * The backend's answer to `POST /autopilot/segment` for a flight of the mock sample (`trainingSample.fixture.ts`), built
 * by hand in the shape `aeroviz_backend/autopilot_segment.py` writes. The contract literals (schema, the segment end) are
 * IMPORTED from the reader, never restated; the words told are the sentence's own (`segmentWords`).
 *
 * The flown track is a straight line at 1 s cycles from the segment's first step to its stop (`segmentStopRow`: a
 * heading word's a lead past the next heading word); a heading word carries its band from its step plus the lead to the
 * next heading word's plus the lead, every row inside.
 */

import {
  segmentStopRow,
  segmentWords,
  TRAINING_AUTOPILOT_SCHEMA,
  TRAINING_AUTOPILOT_SEGMENT_END,
  type TrainingAutopilotRequest,
} from "../trainingAutopilot";
import { trainingColumnRuns, type TrainingSample } from "../trainingSample";

export function mockAutopilotRequest(sample: TrainingSample, flightKey: string, column: TrainingAutopilotRequest["column"],
  row: number): TrainingAutopilotRequest {
  return { airport: sample.airport, setId: sample.setId, flightKey, column, row };
}

/** A consistent answer for ``request`` over ``sample``; ``cycles`` flown (to the segment's end unless it is the last word). */
export function mockAutopilotAnswer(sample: TrainingSample, request: TrainingAutopilotRequest): Record<string, any> {
  const flight = sample.flights.find((item) => item.flightKey === request.flightKey)!;
  const run = trainingColumnRuns(flight, request.column).find((item) => item.row === request.row)!;
  const stepS = sample.vocabulary.stepS;
  const lead = sample.vocabulary.headingLeadRows;
  const stopRow = segmentStopRow(flight, request.column, run.endRow, lead);
  const toLanding = stopRow === flight.rows;
  const steps = (toLanding ? flight.rows - 1 : stopRow) - run.row;
  const cycles = steps * stepS;                     // one-second cycles
  const n = cycles + 1;
  const along = Array.from({ length: n }, (_, index) => index);
  const tS = along.map((index) => run.row * stepS + index);
  const judged = request.column === "heading";
  const judgedSteps = steps + 1;
  const firstRow = run.row + lead;
  const bandStop = Math.max(firstRow, Math.min(run.endRow + lead, run.row + judgedSteps));
  const envelope = flight.envelopes.heading.find((item) => item.row === run.row);
  const tolerance = sample.vocabulary.headingToleranceDeg;
  const inside = Array.from({ length: bandStop - firstRow }, () => 1);
  return {
    ok: true,
    schema: TRAINING_AUTOPILOT_SCHEMA,
    airport: request.airport, setId: request.setId, flightKey: request.flightKey, datasetId: flight.datasetId,
    computedUtc: "2026-09-25T12:00:00Z", computeS: 0.42, waitS: 0,
    executor: { spec: "4dTrajectory/outputs/POOLED/executor/test", specSha256: "9".repeat(64), sourceSha256: "8".repeat(64),
      wordClock: "track", cycleS: 1, timeoutFactor: 1.5 },
    artefact: "4dTrajectory/outputs/POOLED/instruction_language/test",
    vocabularySpecSha256: sample.vocabulary.specSha256,
    group: "own dynamics",
    segment: {
      column: request.column, row: run.row, endRow: run.endRow, stopRow, toLanding, observedS: steps * stepS,
      told: segmentWords(flight, run.row, stopRow),
    },
    end: toLanding
      ? { reason: "landed", reachedSegmentEnd: null, offsetFromObserved: null, flownS: cycles,
          crossing: { crossM: -1.5, heightM: 15.2, atS: tS[n - 1] }, refused: null }
      : { reason: TRAINING_AUTOPILOT_SEGMENT_END, reachedSegmentEnd: true,
          offsetFromObserved: { horizontalM: 120, aboveM: -8, groundSpeedMps: 1.5 }, flownS: cycles, crossing: null, refused: null },
    word: judged
      ? { status: inside.every(Boolean) ? "inside" : "outside",
          checks: [{ name: "track within the tolerance of the word", ok: inside.every(Boolean),
                     inside: inside.filter(Boolean).length, rows: inside.length }],
          reason: null,
          heading: { firstRow, stopRow: bandStop, targetOnTrackDeg: envelope!.targetDeg,
                     bandDeg: [envelope!.targetDeg - tolerance, envelope!.targetDeg + tolerance], inside } }
      : { status: "no check", checks: [], reason: "the runway pointer: judged by the landing", heading: null },
    limits: { cycles, bound: { bank_cap: 0, bank_rate: 2 } },
    track: {
      tS, eM: along.map((index) => -17600 + index * 110), nM: along.map(() => 2200),
      lon: along.map((index) => -78 + (-17600 + index * 110) / 90000), lat: along.map(() => 35 + 2200 / 111000),
      altitudeM: along.map(() => 1110), altitudeHaeM: along.map(() => 1077), groundSpeedMps: along.map(() => 110),
      verticalRateMps: along.map(() => 0), trackDeg: along.map(() => 225), distanceM: along.map((index) => 1760 + index * 110),
      thrustFraction: along.slice(1).map(() => 0.3), bankRightDeg: along.slice(1).map(() => -12), loadFactor: along.slice(1).map(() => 1.02),
    },
    judgedTrackDeg: judged ? Array.from({ length: judgedSteps }, () => 225) : null,
  };
}
