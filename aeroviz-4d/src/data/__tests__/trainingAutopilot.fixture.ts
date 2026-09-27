/**
 * The backend's answer to `POST /autopilot/segment` for a flight of the mock sample (`trainingSample.fixture.ts`), built
 * by hand in the shape `aeroviz_backend/autopilot_segment/payload.py` writes. The contract literals (schema, the segment end) are
 * IMPORTED from the reader, never restated; the words told are the sentence's own (`segmentWords`).
 *
 * The flown track is a straight line at 1 s cycles from the segment's first step to its stop (`segmentStopRow`: a
 * heading word's a lead past the next heading word, heard on the time clock at its own step — the tail); a heading word
 * carries its band from its step plus the lead to the next heading word's plus the lead, every row inside. A MODEL's word (`mockModelAutopilotRequest`, a sample of the
 * generation fixture) is answered the same way over the model's sentence, with no observed time and no offset from the
 * observed aircraft.
 */

import {
  requestSentence,
  segmentStopRow,
  segmentWords,
  TRAINING_AUTOPILOT_SCHEMA,
  TRAINING_AUTOPILOT_SEGMENT_END,
  type TrainingAutopilotRequest,
} from "../trainingAutopilot";
import { sentenceColumnRuns, trainingSelectionOf, type TrainingSample, type TrainingSelection } from "../trainingSample";
import type { TrainingGeneratedSentence, TrainingProcedureMask } from "../trainingOverlays";
import { mockGeneratedPoint } from "./trainingOverlays.fixture";

export function mockAutopilotRequest(sample: TrainingSample, flightKey: string, column: TrainingAutopilotRequest["column"],
  row: number): TrainingAutopilotRequest {
  return { airport: sample.airport, setId: sample.setId, flightKey, column, row, sentence: null };
}

/** A request for a MODEL's word: ``sentence`` (a sample the generation fixture's reader gave) of ``overlayId``, spoken
 *  under ``procedureMasks`` (none by default: base's). */
export function mockModelAutopilotRequest(sample: TrainingSample, flightKey: string, column: TrainingAutopilotRequest["column"],
  row: number, overlayId: string, sentence: TrainingGeneratedSentence,
  procedureMasks: TrainingProcedureMask[] = [],
): TrainingAutopilotRequest {
  return { airport: sample.airport, setId: sample.setId, flightKey, column, row,
    sentence: { overlayId, sample: sentence.sample, firstRow: sentence.firstRow, rows: sentence.rows,
      events: sentence.events.map(({ row: step, column: index, value }) => ({ row: step, column: index, value })), procedureMasks } };
}

/** The flight on screen that ``request`` asks about: its selection. */
export function mockSelection(sample: TrainingSample, request: TrainingAutopilotRequest): TrainingSelection {
  return trainingSelectionOf(sample, sample.flights.find((item) => item.flightKey === request.flightKey)!);
}

/** A consistent answer for ``request`` over ``sample``; ``cycles`` flown (to the segment's end unless it is the last word). */
export function mockAutopilotAnswer(sample: TrainingSample, request: TrainingAutopilotRequest): Record<string, any> {
  const flight = sample.flights.find((item) => item.flightKey === request.flightKey)!;
  const sentence = requestSentence(request, flight);
  const model = request.sentence;
  const run = sentenceColumnRuns(sentence, request.column).find((item) => item.row === request.row)!;
  const stepS = sample.vocabulary.stepS;
  const lead = sample.vocabulary.headingLeadRows;
  const stopRow = segmentStopRow(sentence.rows, request.column, run.endRow, lead);
  const toLanding = stopRow === sentence.rows;
  const steps = (toLanding ? sentence.rows - 1 : stopRow) - run.row;
  const cycles = steps * stepS;                     // one-second cycles
  const n = cycles + 1;
  const along = Array.from({ length: n }, (_, index) => index);
  const tS = along.map((index) => run.row * stepS + index);
  const judged = request.column === "heading";
  const judgedSteps = steps + 1;
  const firstRow = run.row + lead;
  const bandStop = Math.max(firstRow, Math.min(run.endRow + lead, run.row + judgedSteps));
  // the truth's heading word's target as its envelope reads it; a model's, the vocabulary's
  const targetDeg = model === null ? flight.envelopes.heading.find((item) => item.row === run.row)?.targetDeg
    : sample.vocabulary.headingTargetsDeg[run.value];
  const tolerance = sample.vocabulary.headingToleranceDeg;
  const inside = Array.from({ length: bandStop - firstRow }, () => 1);
  return {
    ok: true,
    schema: TRAINING_AUTOPILOT_SCHEMA,
    airport: request.airport, setId: request.setId, flightKey: request.flightKey, datasetId: flight.datasetId,
    computedUtc: "2026-09-25T12:00:00Z",
    timing: { waitS: 0, setupS: 0.02, openS: 0.91, flightKept: false, prepareS: 0.01, flyS: 0.21, cycles, judgeS: 0.06,
              answerS: 0.03, computeS: 1.24 },
    executor: { spec: "4dTrajectory/outputs/POOLED/executor/test", specSha256: "9".repeat(64), sourceSha256: "8".repeat(64),
      wordClock: model === null ? "track" : "time", cycleS: 1, timeoutFactor: 1.5 },
    artefact: "4dTrajectory/outputs/POOLED/instruction_language/test",
    vocabularySpecSha256: sample.vocabulary.specSha256,
    group: "own dynamics",
    source: model === null ? { kind: "truth" }
      : { kind: "model", overlayId: model.overlayId, sample: model.sample, firstRow: model.firstRow },
    segment: {
      column: request.column, row: run.row, endRow: run.endRow, stopRow, toLanding, observedS: model === null ? steps * stepS : null,
      told: segmentWords(request, flight, stopRow),
      // a heading word flown on past the next heading word hears it at its step (one-second cycles from the word's step);
      // heard on the track's last point, nothing is flown past it
      nextWordHeardS: stopRow > run.endRow && run.endRow * stepS < tS[n - 1] ? run.endRow * stepS : null,
    },
    end: toLanding
      ? { reason: "landed", reachedSegmentEnd: null, offsetFromObserved: null, flownS: cycles,
          crossing: { crossM: -1.5, heightM: 15.2, atS: tS[n - 1], runway: 0 }, refused: null }
      : { reason: TRAINING_AUTOPILOT_SEGMENT_END, reachedSegmentEnd: true,
          offsetFromObserved: model === null ? { horizontalM: 120, aboveM: -8, groundSpeedMps: 1.5 } : null, flownS: cycles,
          crossing: null, refused: null },
    word: judged
      ? { status: inside.every(Boolean) ? "inside" : "outside",
          checks: [{ name: "track within the tolerance of the word", ok: inside.every(Boolean),
                     inside: inside.filter(Boolean).length, rows: inside.length }],
          reason: null,
          heading: { firstRow, stopRow: bandStop, targetOnTrackDeg: targetDeg!,
                     bandDeg: [targetDeg! - tolerance, targetDeg! + tolerance], inside } }
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

/** ``answer`` (a model word's) flown on its sample's own line (`mockGeneratedPoint`), as the deterministic executor
 *  flies a model's sentence again. */
export function onSampleLine(answer: Record<string, any>): Record<string, any> {
  const points = answer.track.tS.map(mockGeneratedPoint);
  answer.track.lon = points.map((point: { lon: number }) => point.lon);
  answer.track.lat = points.map((point: { lat: number }) => point.lat);
  answer.track.altitudeM = points.map((point: { altitudeM: number }) => point.altitudeM);
  return answer;
}

/** ``answer`` (a heading word's, `mockAutopilotAnswer`) as a dynamics failure after ``states`` − 1 cycles: the failed
 *  state left out of the track, as the backend leaves it; its band on the rows its judge read before the failure — with
 *  one state, none: the gate refuses a track that short, and the word is not judged. */
export function failedAnswer(answer: Record<string, any>, states: number): Record<string, any> {
  const track = answer.track;
  const cycles = track.tS.length - 1;
  // the judged steps are every `stepCycles`-th point of the track, from its first
  const stepCycles = answer.judgedTrackDeg === null ? null : cycles / (answer.judgedTrackDeg.length - 1);
  for (const key of Object.keys(track)) {
    // a state per point, or a command per cycle between them
    track[key] = track[key].slice(0, track[key].length === cycles ? states - 1 : states);
  }
  answer.timing.cycles = states;                     // the failed cycle is counted
  answer.limits.cycles = states;
  // a next word heard at or after the failure has nothing flown past it
  if (answer.segment.nextWordHeardS !== null && answer.segment.nextWordHeardS >= track.tS[track.tS.length - 1]) {
    answer.segment.nextWordHeardS = null;
  }
  answer.end = { reason: "dynamics_failure", reachedSegmentEnd: false, offsetFromObserved: null, flownS: states - 1,
    crossing: null, refused: states === 1 ? "the flown track is too short to judge" : null };
  if (states === 1) {
    answer.word = { status: "not judged", checks: [], reason: "the gate refused the flown track", heading: null };
    answer.judgedTrackDeg = null;
    return answer;
  }
  const steps = Math.floor((states - 1) / stepCycles!) + 1;   // the judged steps among the states kept
  answer.judgedTrackDeg = answer.judgedTrackDeg.slice(0, steps);
  const band = answer.word.heading;
  band.stopRow = Math.max(band.firstRow, Math.min(band.stopRow, answer.segment.row + steps));
  band.inside = band.inside.slice(0, band.stopRow - band.firstRow);
  answer.word.checks[0] = { ...answer.word.checks[0], inside: band.inside.filter(Boolean).length, rows: band.inside.length };
  return answer;
}
