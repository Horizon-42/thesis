/**
 * A window set over the Training fixture's flights (`trainingSample.fixture.ts`) and a model's sentences in it, in the
 * shapes `window_training_export.py` writes, built by hand. Every contract literal is IMPORTED from the reader.
 *
 * One window opening at 2026-01-01T00:00Z: the VECTORED flight commanded from 100 s (its row 0 on the window's clock) and
 * the STRAIGHT-IN one from 160 s; a replayed arrival ABC1 and a background one BKG2. As recorded both land (VECTORED at
 * 220 s, STRAIGHT-IN at 270 s) and only IFR finds a pair under its minimum (VECTORED – ABC1). The base model's two samples:
 * #1 — VECTORED ended at 20 s of its own (120 s on the window's clock) for a loss behind STRAIGHT-IN, flying on silent to
 * its time limit (60 s), STRAIGHT-IN landing at 50 s (210 s); #2 — both time out, no loss.
 */

import { TRAINING_COLUMNS, TRAINING_SPEC_SHA256 } from "../trainingSample";
import { TRAINING_OVERLAYS_SCHEMA } from "../trainingOverlays";
import { TRAINING_TRAFFIC_SCHEMA, TRAINING_WINDOW_GENERATION_SCHEMA } from "../trainingTraffic";
import { MOCK_DATUM_M, STRAIGHT_KEY, VECTORED_KEY, WORD, mockSample } from "./trainingSample.fixture";

export const TRAFFIC_SET_ID = "traffic_windows_select";
export const WINDOW_MODEL_ID = "windows_base";
export const VECTORED_ID = `KXXX:${VECTORED_KEY}`;
export const STRAIGHT_ID = `KXXX:${STRAIGHT_KEY}`;
export const REPLAYED_ID = "KXXX:ABC1_09_fff001_20260101T000500Z";
export const BACKGROUND_ID = "KXXX:BKG2_09_fff002_20260101T000600Z";
/** The model's first predicted row; each sample's track starts there. */
export const WINDOW_FIRST_ROW = 4;
export const ROW_ZERO_S: Record<string, number> = { [VECTORED_ID]: 100, [STRAIGHT_ID]: 160 };

type SetFlight = { flightKey: string; signals: { tS: number[]; lon: number[]; lat: number[]; altitudeHaeM: number[]; raw: { altitudeM: number[] } } };

function setFlight(key: string): SetFlight {
  return (mockSample() as unknown as { flights: SetFlight[] }).flights.find((flight) => flight.flightKey === key)!;
}

/** A flight's recorded rows on the window's clock. */
function recordedTrack(key: string, rowZeroS: number) {
  const { signals } = setFlight(key);
  return { tS: signals.tS.map((at) => at + rowZeroS), lon: [...signals.lon], lat: [...signals.lat],
    altitudeM: [...signals.raw.altitudeM], altitudeHaeM: [...signals.altitudeHaeM] };
}

/** Another aircraft of the window, a straight line on the window's clock. */
function otherTrack(fromS: number, toS: number, lat: number) {
  const tS = Array.from({ length: (toS - fromS) / 2 + 1 }, (_, k) => fromS + 2 * k);
  return { tS, lon: tS.map((at) => -78.9 + at * 1e-4), lat: tS.map(() => lat), altitudeM: tS.map(() => 1500),
    altitudeHaeM: tS.map(() => 1500 + MOCK_DATUM_M) };
}

/** A sample's flown track, own clock: from the flight's observed row at the first predicted row, a step apart, to ``toS``. */
function flownTrack(key: string, toS: number) {
  const { signals } = setFlight(key);
  const row = WINDOW_FIRST_ROW;
  const tS = Array.from({ length: (toS - row * 2) / 2 + 1 }, (_, k) => row * 2 + 2 * k);
  const since = (at: number) => at - row * 2;
  return { tS, lon: tS.map((at) => signals.lon[row] + since(at) * 1e-4), lat: tS.map(() => signals.lat[row]),
    altitudeM: tS.map((at) => signals.raw.altitudeM[row] - since(at) * 5),
    altitudeHaeM: tS.map((at) => signals.raw.altitudeM[row] - since(at) * 5 + MOCK_DATUM_M), groundSpeedMps: tS.map(() => 70) };
}

const opening = [
  { row: WINDOW_FIRST_ROW, column: 0, value: WORD.runway09 },
  { row: WINDOW_FIRST_ROW, column: 1, value: WORD.notCleared },
  { row: WINDOW_FIRST_ROW, column: 2, value: WORD.heading270 },
  { row: WINDOW_FIRST_ROW, column: 3, value: WORD.altitude1110 },
  { row: WINDOW_FIRST_ROW, column: 4, value: WORD.level },
  { row: WINDOW_FIRST_ROW, column: 5, value: WORD.speed110 },
];

/** One sentence: the opening words, the rest silent; ended by the judge when ``end`` is given. */
function sentence(sample: number, datasetId: string, own: "landed" | "timeout", ownEndS: number,
  end: { atS: number; with: string } | null, cleared = false) {
  const key = datasetId.split(":")[1];
  const events = cleared ? [...opening, { row: 10, column: 1, value: WORD.cleared }] : opening;
  return {
    datasetId, sample, outcome: end === null ? own : "lost_separation", own,
    end: end === null ? null : { kind: "in_trail", relation: "same", with: end.with },
    endS: end === null ? ownEndS : end.atS, ownEndS,
    crossing: own === "landed" ? { crossM: -1, heightM: 15, atS: ownEndS - 1, runway: 0 } : null,
    firstRunway: 0, lastRunway: 0, runwayChanges: 0, goArounds: 0, clearedAtEnd: cleared, forbiddenMass: {},
    rows: Math.floor(ownEndS / 2) + 1, events, track: flownTrack(key, ownEndS),
  };
}

const noLosses = () => ({ episodes: [], atThreshold: [], ended: [] });

export function mockTrafficSet(): Record<string, unknown> {
  const { cohort: _cohort, ...head } = mockSample() as Record<string, unknown>;
  return {
    ...head,
    schema: TRAINING_TRAFFIC_SCHEMA,
    setId: TRAFFIC_SET_ID,
    cohort: { split: "select", windows: 1, seed: 1337, drawnFrom: "a test draw", drawn: 1 },
    windows: [{
      opensUtc: "2026-01-01T00:00:00Z",
      commanded: [VECTORED_ID, STRAIGHT_ID].map((id) => ({ datasetId: id, rowZeroS: ROW_ZERO_S[id], limitS: 60,
        recorded: recordedTrack(id.split(":")[1], ROW_ZERO_S[id]) })),
      others: [
        { datasetId: REPLAYED_ID, callsign: "ABC1", category: "D", role: "replayed", track: otherTrack(60, 300, 35.9) },
        { datasetId: BACKGROUND_ID, callsign: "BKG2", category: null, role: "background", track: otherTrack(200, 400, 35.95) },
      ],
      recorded: {
        visual: noLosses(),
        ifr: { episodes: [{ pair: [REPLAYED_ID, VECTORED_ID].sort(), relation: "same", fromS: 150, toS: 156, steps: 4,
          kinds: ["in_trail"], closestM: 4000, requiredM: 5556, wakeKnown: true, responsible: [VECTORED_ID], ended: [VECTORED_ID] }],
          atThreshold: [], ended: [{ datasetId: VECTORED_ID, atS: 150, kind: "in_trail", relation: "same", with: REPLAYED_ID }] },
        landings: [{ datasetId: VECTORED_ID, atS: 220 }, { datasetId: STRAIGHT_ID, atS: 270 }],
      },
    }],
  };
}

/** The manifests: the window set in the index beside the read-back one, the window model in the overlays. */
export function mockTrafficEntry() {
  return { id: TRAFFIC_SET_ID, kind: "traffic-windows", title: "windows", file: `${TRAFFIC_SET_ID}/traffic.json`,
    vocabularySha256: TRAINING_SPEC_SHA256, runwaySha256: "c".repeat(64), readingRule: "instruction-v3", flights: 2,
    cohort: { split: "select", windows: 1, seed: 1337, drawnFrom: "a test draw" } };
}

export function mockWindowOverlays(): Record<string, unknown> {
  return { schema: TRAINING_OVERLAYS_SCHEMA, writtenUtc: "2026-09-30T00:00:00+00:00", airport: "KXXX", overlays: [
    { id: WINDOW_MODEL_ID, kind: "window-generation", base: TRAFFIC_SET_ID, title: "base in the windows",
      file: `${WINDOW_MODEL_ID}/window_generation.json`, flights: 2, source: { runner: "test" } },
  ] };
}

export function mockWindowOverlay(): Record<string, any> {
  const sample = mockSample() as { candidatesSha256: string; airportFrame: Record<string, unknown> };
  return {
    schema: TRAINING_WINDOW_GENERATION_SCHEMA, overlayId: WINDOW_MODEL_ID, airport: "KXXX",
    writtenUtc: "2026-09-30T00:00:00+00:00", producedBy: { runner: "test" },
    base: { setId: TRAFFIC_SET_ID, specSha256: TRAINING_SPEC_SHA256, candidatesSha256: sample.candidatesSha256,
      airportFrame: { ...sample.airportFrame } },
    model: { name: "base", round: null, run: "4dTrajectory/outputs/POOLED/prior/v3_step1/full_s1", fineTuning: null,
      checkpointSha256: "9".repeat(64), variant: "full", trainedAt: { head: "test", dirty: false } },
    generation: { samples: 2, temperature: 1, seed: 1337, firstPredictedRow: WINDOW_FIRST_ROW, stepS: 2, procedureMasks: [],
      executor: { specSha256: "e".repeat(64), wordClock: "time", cycleS: 1, timeoutFactor: 1.5 }, trafficAttention: "zero" },
    readout: null,
    columns: [...TRAINING_COLUMNS],
    windows: [{
      opensUtc: "2026-01-01T00:00:00Z", commanded: [VECTORED_ID, STRAIGHT_ID],
      samples: [
        {
          sample: 0,
          aircraft: [sentence(0, VECTORED_ID, "timeout", 60, { atS: 20, with: STRAIGHT_ID }),
            sentence(0, STRAIGHT_ID, "landed", 50, null, true)],
          visual: { episodes: [{ pair: [VECTORED_ID, STRAIGHT_ID].sort(), relation: "same", fromS: 120, toS: 124, steps: 3,
            kinds: ["in_trail"], closestM: 3000, requiredM: 5556, wakeKnown: true, responsible: [VECTORED_ID], ended: [VECTORED_ID] }],
            atThreshold: [], ended: [{ datasetId: VECTORED_ID, atS: 120, kind: "in_trail", relation: "same", with: STRAIGHT_ID }] },
          ifr: noLosses(),
          landings: [{ datasetId: STRAIGHT_ID, atS: 210 }],
        },
        {
          sample: 1,
          aircraft: [sentence(1, VECTORED_ID, "timeout", 60, null), sentence(1, STRAIGHT_ID, "timeout", 60, null)],
          visual: noLosses(), ifr: noLosses(), landings: [],
        },
      ],
    }],
  };
}
