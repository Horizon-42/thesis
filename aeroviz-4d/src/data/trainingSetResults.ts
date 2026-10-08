/**
 * trainingSetResults.ts
 * ---------------------
 * The results of the experiment that made a Training set (outline §6.2 items 3, 4; D134): the backend's
 * `GET /training/results?stage=&airport=&set=` reads the files the set names under `4dTrajectory/outputs/` and answers
 * named fields of each section; a section whose file lies elsewhere or is missing answers why. This reader checks the
 * answer's shape and types each section; the details page shows them. Nothing is computed here.
 */

import { useEffect, useState } from "react";
import { AEROVIZ_BACKEND_URL } from "../pilot/pilotClient";
import { asNumber, attempt, recordOf, Reader, Refusal, type Parsed } from "./trainingReader";

/** MIRROR of the backend's route (`aeroviz_backend/http_server.py`, `training_results.TrainingResults`). */
export const TRAINING_RESULTS_PATH = "/training/results";

export type TrainingStage = "A" | "B" | "C";
export type Counts = Record<string, number>;
/** A section: its fields, or why it has none (a file elsewhere or missing; a reading the design keeps unshown). */
export type TrainingResultSection<T> = { ok: true; value: T } | { ok: false; problem: string };

export interface TrainingLabellingSplit {
  labelled: number;
  refused: number;
  refusalReasons: Counts;
  refusedByAirport: Record<string, Counts>;
  labelledByAirport: Counts;
}
export interface TrainingReplayCell {
  flights: number;
  outcomes: Counts;
  landed: number;
}
export interface TrainingReplayResult {
  split: string;
  intervalS: number;
  flights: number;
  /** By group of dynamics ("own dynamics", "stand-in dynamics"), then by place (the set's airport, "all"). */
  groups: Record<string, Record<string, TrainingReplayCell>>;
}
export interface TrainingFreeGenerationCell {
  sentences: number;
  outcomes: Counts;
  timedOut: number;
  goArounds: number;
  atTheBound: number;
  wordsPerSentence: Counts;
  labelledWordsPerSentence: Counts;
}
export interface TrainingChoiceStep {
  arms: Record<string, { score: number; folds: Counts }>;
  seedScale: number;
  chosen: string;
}
/** The configuration's choice: the best score and the configurations within the seed scale of it. */
export interface TrainingChoiceConfiguration extends TrainingChoiceStep {
  bestScore: number;
  within: string[];
}
/** The variant's choice: each variant's score. */
export interface TrainingChoiceVariant extends TrainingChoiceStep {
  scores: Counts;
}
export interface TrainingSpeedSetting {
  device: string;
  batch: number;
  loops: number;
  loopSizes: number[];
  rowsTimed: number;
  sentences: number;
  priorStepMs: Counts;
  executorStepsMs: Counts;
  rowMs: Counts;
  sentenceS: Counts;
  flightRowsPerS: number;
  shareOfInterval: { intervalS: number; prior: Counts; executor: Counts; row: Counts };
  host: Record<string, string | number>;
}
export interface TrainingRoundResult {
  round: number;
  speaking: { windows: number; rewardSum: number; outcomes: Counts };
  selection: Record<string, { windows: number; rewardMean: number; outcomes: Counts }>;
}

export interface TrainingStageAResults {
  stage: "A";
  labelling: TrainingResultSection<Record<string, TrainingLabellingSplit>>;
  closedLoop: TrainingResultSection<TrainingReplayResult[]>;
}
export interface TrainingStageBResults {
  stage: "B";
  freeGeneration: TrainingResultSection<{ split: string; selection: string;
    sides: Record<string, Record<string, Record<string, TrainingFreeGenerationCell>>> }>;
  training: TrainingResultSection<{ bestEpoch: number; epochs: Array<{ epoch: number; trainLossPerStep: number; selectLossPerStep: number }> }>;
  validation: TrainingResultSection<{ lossPerStep: number; perColumn: Counts; masksOnLabelledWords: Record<string, Record<string, Counts>> }>;
  choice: TrainingResultSection<{ configuration: TrainingChoiceConfiguration; variant: TrainingChoiceVariant }>;
  speed: TrainingResultSection<TrainingSpeed>;
}
/** The selection readout of the round a campaign starts from (post-training D162, frontend §4.3): ``selection`` null
 *  with ``why`` when that campaign read other select windows. The round is the set's `model.start` (the same record). */
export interface TrainingStartReadout {
  selection: TrainingRoundResult["selection"] | null;
  why: string | null;
}
export interface TrainingStageCResults {
  stage: "C";
  /** ``start``: null for a campaign from the base (the base has no selection readout). */
  rounds: TrainingResultSection<{ started: string; rounds: TrainingRoundResult[]; start: TrainingStartReadout | null }>;
  checks: TrainingResultSection<{ checks: Record<string, unknown> }>;
  speed: TrainingResultSection<TrainingSpeed>;
}
/** The model's speed readout (D136): the model it timed (B: the prior; C: the campaign's round) and each setting. */
export interface TrainingSpeed {
  model: string;
  split: string;
  warmupRows: number;
  smoke: boolean;
  settings: TrainingSpeedSetting[];
}
export type TrainingSetResults = (TrainingStageAResults | TrainingStageBResults | TrainingStageCResults) & {
  /** The path of the readout or campaign the set was made from: the one line of provenance. */
  provenance: string;
};

const counts = (value: unknown, where: string): Counts => recordOf(value, where, asNumber);

function section<T>(sections: Reader, key: string, read: (reader: Reader) => T): TrainingResultSection<T> {
  const reader = sections.child(key);
  if (reader.boolean("ok")) return { ok: true, value: read(reader) };
  return { ok: false, problem: reader.string("problem") };
}

function choiceStep(reader: Reader): TrainingChoiceStep {
  return {
    arms: reader.record("arms", (value, where) => {
      const arm = Reader.of(value, where);
      return { score: arm.number("score"), folds: arm.record("folds", asNumber) };
    }),
    seedScale: reader.number("seedScale"), chosen: reader.string("chosen"),
  };
}

function speedSetting(reader: Reader): TrainingSpeedSetting {
  const share = reader.child("shareOfInterval");
  return {
    device: reader.string("device"), batch: reader.count("batch", 1), loops: reader.count("loops", 1),
    loopSizes: reader.numbers("loopSizes"), rowsTimed: reader.count("rowsTimed"), sentences: reader.count("sentences"),
    priorStepMs: reader.record("priorStepMs", asNumber), executorStepsMs: reader.record("executorStepsMs", asNumber),
    rowMs: reader.record("rowMs", asNumber), sentenceS: reader.record("sentenceS", asNumber),
    flightRowsPerS: reader.number("flightRowsPerS"),
    shareOfInterval: { intervalS: share.number("intervalS"), prior: share.record("prior", asNumber),
      executor: share.record("executor", asNumber), row: share.record("row", asNumber) },
    host: reader.record("host", (value, where) => {
      if (typeof value !== "string" && typeof value !== "number") throw new Refusal(`${where} is not a string or a number`);
      return value;
    }),
  };
}

function speed(reader: Reader): TrainingSpeed {
  return {
    model: reader.string("model"), split: reader.string("split"), warmupRows: reader.count("warmupRows"), smoke: reader.boolean("smoke"),
    settings: reader.children("settings").map(speedSetting),
  };
}

/** The backend's answer, or the problem by name (the page shows it). */
export function parseTrainingSetResults(ok: boolean, raw: unknown, setId: string): Parsed<TrainingSetResults> {
  const answer = (raw ?? {}) as Record<string, unknown>;
  if (!ok || answer.ok !== true) {
    return { ok: false, problem: `No results for ${setId}: ${typeof answer.error === "string" ? answer.error : "the backend gave none"}` };
  }
  return attempt(() => {
    const reader = Reader.of(raw, "results");
    const stage = reader.oneOf("stage", ["A", "B", "C"] as const);
    const provenance = reader.string("provenance");
    const sections = reader.child("sections");
    if (stage === "A") {
      return {
        stage, provenance,
        labelling: section(sections, "labelling", (s) => s.record("splits", (value, where) => {
          const split = Reader.of(value, where);
          return {
            labelled: split.count("labelled"), refused: split.count("refused"), refusalReasons: split.record("refusalReasons", asNumber),
            refusedByAirport: split.record("refusedByAirport", counts), labelledByAirport: split.record("labelledByAirport", asNumber),
          };
        })),
        closedLoop: section(sections, "closedLoop", (s) => Object.values(s.record("replays", (value, where) => {
          const replay = Reader.of(value, where);
          return {
            split: replay.string("split"), intervalS: replay.number("intervalS"), flights: replay.count("flights"),
            groups: replay.record("groups", (group, at) => recordOf(group, at, (cell, place) => {
              const c = Reader.of(cell, place);
              return { flights: c.count("flights"), outcomes: c.record("outcomes", asNumber), landed: c.number("landed") };
            })),
          };
        }))),
      };
    }
    if (stage === "B") {
      return {
        stage, provenance,
        freeGeneration: section(sections, "freeGeneration", (s) => ({
          split: s.string("split"), selection: s.string("selection"),
          sides: s.record("sides", (side, at) => recordOf(side, at, (strata, where) => recordOf(strata, where, (cell, place) => {
            const c = Reader.of(cell, place);
            return {
              sentences: c.count("sentences"), outcomes: c.record("outcomes", asNumber), timedOut: c.count("timedOut"),
              goArounds: c.count("goArounds"), atTheBound: c.count("atTheBound"),
              wordsPerSentence: c.record("wordsPerSentence", asNumber), labelledWordsPerSentence: c.record("labelledWordsPerSentence", asNumber),
            };
          }))),
        })),
        training: section(sections, "training", (s) => ({
          bestEpoch: s.count("bestEpoch"),
          epochs: s.children("epochs").map((e) => ({ epoch: e.count("epoch"), trainLossPerStep: e.number("trainLossPerStep"),
            selectLossPerStep: e.number("selectLossPerStep") })),
        })),
        validation: section(sections, "validation", (s) => {
          const forced = s.child("teacherForced");
          return { lossPerStep: forced.number("lossPerStep"), perColumn: forced.record("perColumn", asNumber),
            masksOnLabelledWords: s.record("masksOnLabelledWords", (side, at) => recordOf(side, at, counts)) };
        }),
        choice: section(sections, "choice", (s) => {
          const configuration = s.child("configuration");
          const variant = s.child("variant");
          return {
            configuration: { ...choiceStep(configuration), bestScore: configuration.number("bestScore"), within: configuration.strings("within") },
            variant: { ...choiceStep(variant), scores: variant.record("scores", asNumber) },
          };
        }),
        speed: section(sections, "speed", speed),
      };
    }
    return {
      stage, provenance,
      rounds: section(sections, "rounds", (s) => ({
        started: s.string("started"),
        rounds: s.children("rounds").map((r) => {
          const speaking = r.child("speaking");
          return {
            round: r.count("round"),
            speaking: { windows: speaking.count("windows"), rewardSum: speaking.number("rewardSum"), outcomes: speaking.record("outcomes", asNumber) },
            selection: selectionOf(r),
          };
        }),
        start: startOf(s.nullableChild("start")),
      })),
      checks: section(sections, "checks", (s) => ({ checks: s.record("checks", (value) => value) })),
      speed: section(sections, "speed", speed),
    };
  });
}

/** A round's selection readout by airport, as the route answers it. */
function selectionOf(reader: Reader): TrainingRoundResult["selection"] {
  return reader.record("selection", (value, where) => {
    const c = Reader.of(value, where);
    return { windows: c.count("windows"), rewardMean: c.number("rewardMean"), outcomes: c.record("outcomes", asNumber) };
  });
}

/** The start's readout: its selection, or why there is none (exactly one of the two). */
function startOf(reader: Reader | null): TrainingStartReadout | null {
  if (reader === null) return null;
  const why = reader.nullableString("why");
  const selection = reader.raw("selection") === null ? null : selectionOf(reader);
  if ((selection === null) === (why === null)) reader.fail("a start readout holds its selection or why it has none, one of the two");
  return { selection, why };
}

export async function fetchTrainingSetResults(stage: TrainingStage, airport: string, setId: string,
  backendUrl: string = AEROVIZ_BACKEND_URL): Promise<Parsed<TrainingSetResults>> {
  const query = new URLSearchParams({ stage, airport, set: setId });
  let response: Response;
  try {
    response = await fetch(`${backendUrl.replace(/\/+$/, "")}${TRAINING_RESULTS_PATH}?${query}`);
  } catch (error) {
    return { ok: false, problem: `No results for ${setId}: the backend at ${backendUrl} did not answer (${
      error instanceof Error ? error.message : String(error)})` };
  }
  let raw: unknown;
  try {
    raw = JSON.parse(await response.text());
  } catch {
    return { ok: false, problem: `No results for ${setId}: the backend answered HTTP ${response.status} with something that is not JSON` };
  }
  return parseTrainingSetResults(response.ok, raw, setId);
}

export type TrainingSetResultsState = { status: "loading" } | { status: "ready"; results: TrainingSetResults }
  | { status: "absent"; problem: string } | { status: "none" };

/** The results of the set on screen (null: none asked), asked again for each set (the backend reads at each request). */
export function useTrainingSetResults(stage: TrainingStage, airport: string, setId: string | null): TrainingSetResultsState {
  const key = setId === null ? null : `${stage}/${airport}/${setId}`;
  const [state, setState] = useState<{ key: string | null; value: TrainingSetResultsState }>({ key: null, value: { status: "loading" } });
  useEffect(() => {
    if (setId === null) return;
    let live = true;
    setState({ key, value: { status: "loading" } });
    fetchTrainingSetResults(stage, airport, setId).then((parsed) => {
      if (live) setState({ key, value: parsed.ok ? { status: "ready", results: parsed.value } : { status: "absent", problem: parsed.problem } });
    });
    return () => {
      live = false;
    };
  }, [stage, airport, setId, key]);
  if (setId === null) return { status: "none" };
  return state.key === key ? state.value : { status: "loading" };
}
