/**
 * trainingStatistics.ts
 * ---------------------
 * THE MODELS' STATISTICS (frontend §3 item 3, D159): the details page's one table — a row for each sentence kind the set
 * has, in the order of the bar's tabs, and two groups of columns: THIS SET, counted from the set's own sentences (the
 * outcomes its sample holds), and THE FORMAL READOUT, from the counts the results route already answers
 * (`trainingSetResults.ts`): A, the executor's replay of the closed-loop sentences of the set's splits at the row's Δ, at
 * the set's airport; B, the free generation the set was made from, at the set's airport, over every side of the prior's
 * selection (the set's flights are drawn from all of them, `prior_training_export.chosen_flights`);
 * C, each round's selection readout at the set's airport. A count the readout does not hold is null (the page writes
 * "—"). Only sums here; the page computes the shares.
 *
 * The columns: the sentences; landed; go-arounds (the go-arounds said, summed, over the sentences — the readouts count
 * them so); timed out (the judge's outcome `timeout`); in a window, the losses of separation (the outcome
 * `lost_separation`) and the reward summed (the page gives its mean).
 */

import type { TrainingSentenceKind } from "../utils/trainingWordColors";
import type { TrainingSample } from "./trainingSample";
import type { TrainingPriorSample } from "./trainingPriorSample";
import {
  startName, TRAINING_WINDOW_START, type TrainingWindowRound, type TrainingWindowSample, type TrainingWindowStart,
} from "./trainingWindowSample";
import type { Counts, TrainingSetResults } from "./trainingSetResults";
import { roundKind } from "./trainingSentenceKind";

/** One group of a row's cells. */
export interface TrainingStatsCells {
  sentences: number;
  landed: number;
  goArounds: number | null;
  timedOut: number;
  lost: number | null;
  rewardSum: number | null;
}

export interface TrainingStatsRow {
  key: string;
  label: string;
  kind: TrainingSentenceKind;
  set: TrainingStatsCells;
  /** Null: the readout holds no count of this row's sentences. */
  readout: TrainingStatsCells | null;
}

export interface TrainingStatsTable {
  /** A window stage: the columns of the losses of separation and the mean reward. */
  windows: boolean;
  rows: TrainingStatsRow[];
  /** Why the route's answer holds no readout section for the table (its file elsewhere or missing), or null. */
  readoutProblem: string | null;
  /** What a row's readout is when it is not the set's model's own (a start from another campaign's round, D162). */
  notes: string[];
}

/** The problem of a section of the route's answer, or null (no answer: the page says why itself). */
function problemOf(section: { ok: true } | { ok: false; problem: string } | null): string | null {
  return section === null || section.ok ? null : section.problem;
}

/** The outcomes of some sentences, counted by name. */
function countOutcomes(outcomes: string[]): Counts {
  const out: Counts = {};
  for (const outcome of outcomes) out[outcome] = (out[outcome] ?? 0) + 1;
  return out;
}

/** Two counts added. */
function addCounts(a: Counts, b: Counts): Counts {
  const out = { ...a };
  for (const [name, n] of Object.entries(b)) out[name] = (out[name] ?? 0) + n;
  return out;
}

/** The cells of outcomes counted by name; the go-arounds and reward given or not counted (null). */
function cellsOf(outcomes: Counts, goArounds: number | null, windows: boolean, rewardSum: number | null): TrainingStatsCells {
  return {
    sentences: Object.values(outcomes).reduce((sum, n) => sum + n, 0), landed: outcomes.landed ?? 0, goArounds,
    timedOut: outcomes.timeout ?? 0, lost: windows ? outcomes.lost_separation ?? 0 : null, rewardSum: windows ? rewardSum : null,
  };
}

const sum = (values: number[]) => values.reduce((total, value) => total + value, 0);

/** Stage A: a row for each Δ of the set (its closed-loop sentences). */
export function stageAStatistics(sample: TrainingSample, results: TrainingSetResults | null): TrainingStatsTable {
  const splits = new Set<string>(sample.flights.map((flight) => flight.split));
  const section = results !== null && results.stage === "A" ? results.closedLoop : null;
  const replays = section !== null && section.ok ? section.value : null;
  return {
    windows: false, readoutProblem: problemOf(section), notes: [],
    rows: sample.vocabulary.rowIntervalsS.map((intervalS) => {
      const own = countOutcomes(sample.flights.map((flight) => flight.closedLoop[String(intervalS)].replay.outcome));
      const cells = (replays ?? []).filter((replay) => replay.intervalS === intervalS && splits.has(replay.split))
        .flatMap((replay) => Object.values(replay.groups).map((at) => at[sample.airport]).filter((cell) => cell !== undefined));
      return {
        key: `closed-${intervalS}`, label: `closed loop · Δ ${intervalS} s`, kind: "closedLoop" as const,
        set: cellsOf(own, null, false, null),
        readout: cells.length === 0 ? null : cellsOf(cells.reduce((all, cell) => addCounts(all, cell.outcomes), {} as Counts), null, false, null),
      };
    }),
  };
}

/** Stage B: the closed loop at the prior's Δ, then the prior's samples. */
export function stageBStatistics(sample: TrainingPriorSample, results: TrainingSetResults | null): TrainingStatsTable {
  const intervalS = String(sample.model.rowIntervalS);
  const said = sample.flights.flatMap((flight) => flight.sentences);
  const section = results !== null && results.stage === "B" ? results.freeGeneration : null;
  const generation = section !== null && section.ok ? section.value : null;
  const strata = generation === null ? []
    : Object.values(generation.sides).flatMap((airports) => Object.values(airports[sample.airport] ?? {}));
  return {
    windows: false, readoutProblem: problemOf(section), notes: [],
    rows: [
      { key: "closed-loop", label: `closed loop · Δ ${intervalS} s`, kind: "closedLoop",
        set: cellsOf(countOutcomes(sample.flights.map((flight) => flight.head.closedLoop[intervalS].replay.outcome)), null, false, null),
        readout: null },
      { key: "samples", label: "the prior's samples", kind: "base",
        set: cellsOf(countOutcomes(said.map((sentence) => sentence.outcome)), sum(said.map((sentence) => sentence.goArounds)), false, null),
        readout: strata.length === 0 ? null : cellsOf(strata.reduce((all, cell) => addCounts(all, cell.outcomes), {} as Counts),
          sum(strata.map((cell) => cell.goArounds)), false, null) },
    ],
  };
}

/** A round's row label, as the bar's tabs name it; the start by what it is (`startName`). */
function roundName(round: TrainingWindowRound, start: TrainingWindowStart | null): string {
  return round === TRAINING_WINDOW_START ? `start (${startName(start)})` : `r${round}`;
}

/** Stages C and D: a row for each round of the set (the start is the base's, or another campaign's round, D162). */
export function windowStatistics(sample: TrainingWindowSample, results: TrainingSetResults | null): TrainingStatsTable {
  const section = results !== null && results.stage === "C" ? results.rounds : null;
  const rounds = section !== null && section.ok ? section.value.rounds : null;
  // the start's readout: the round it starts from, read on the same select windows (frontend §4.3); none from the base
  const start = section !== null && section.ok ? section.value.start : null;
  const notes = start === null || !sample.model.rounds.includes(TRAINING_WINDOW_START) ? []
    : [start.selection === null ? `The start's readout is not shown: ${start.why}.`
      : `The start's readout is ${startName(sample.model.start)}'s own, read on the same select windows.`];
  return {
    windows: true, readoutProblem: problemOf(section), notes,
    rows: sample.model.rounds.map((round, place) => {
      const said = sample.windows.flatMap((window) => window.commanded.map((aircraft) => aircraft.rounds[place]));
      const read = round === TRAINING_WINDOW_START ? start?.selection?.[sample.airport]
        : rounds?.find((item) => item.round === round)?.selection[sample.airport];
      return {
        key: `round-${round}`, label: roundName(round, sample.model.start), kind: roundKind(round, sample.model.start),
        set: cellsOf(countOutcomes(said.map((sentence) => sentence.outcome)), sum(said.map((sentence) => sentence.goArounds)), true,
          sum(said.map((sentence) => sentence.reward))),
        readout: read === undefined ? null : cellsOf(read.outcomes, null, true, read.rewardMean * read.windows),
      };
    }),
  };
}
