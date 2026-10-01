/**
 * trainingOverlays.ts
 * -------------------
 * What another model makes of a Training set's own flights, drawn over the set: the EXECUTOR's replay (the truth
 * sentence flown from row 0 — its track, its outcome and every word's verdict), the PRIOR's predictions (at every
 * step of the truth sentence, what it gives each column) and the prior's OWN SENTENCES (free generation: the words it
 * said from its first predicted step, the executor flying them, how each flight ended — several samples a flight).
 * Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md` §2.6, §4.6 and §4.7; the writers:
 * `experiments/executor_training_export.py`, `experiments/prior_training_export.py`,
 * `experiments/prior_generation_training_export.py`.
 *
 * AN OVERLAY IS A FILE BESIDE ITS SET, NEVER INSIDE IT. `training/overlays.json` lists them, each naming the set it
 * is drawn over (`base`); the payload records what it shares with that set — the set id, the spec, the candidate
 * runways and the airport frame — and holds one entry per flight of the set, in the set's order. The reader binds the
 * two by those fields and by the flights themselves — the same keys, for the executor the same words, for the prior
 * the same number of steps, every height drawn on the set flight's own HAE − MSL, and a track flown from the set's
 * observed state starting at that observed row — and refuses the payload whole if
 * any differs: a verdict drawn on the wrong word is worse than none. It never binds by the set file's bytes or its
 * time of writing: a set exported again with the same flights keeps its overlays.
 *
 * NOTHING HERE IS COMPUTED. Every verdict is the executor's judge's, re-flown and checked against its formal replay
 * in Python — a heading word's with its band and a verdict per row on the flown track the judge read — and every
 * probability is the prior's own. The reader checks bookkeeping — lengths, ranges, the binding — and hands numbers to
 * the views.
 *
 * NO COMPATIBILITY: the five schemas are pinned below and anything else is refused by name. A payload is bound to the
 * manifest entry that listed it (its id and set) and to the sample on screen (its set, spec, candidates, frame, airport
 * and flights), or refused whole.
 *
 * SI units only: metres, m/s, degrees, seconds.
 */

import { readAttitude, type TrainingAttitude } from "./trainingAttitude";
import { fetchJson } from "../utils/fetchJson";
import { asNumber, attempt, parseManifest, Reader, recordOf, Refusal, type Parsed } from "./trainingReader";
import {
  readHeadingBand,
  TRAINING_COLUMN_INDEX,
  TRAINING_COLUMNS,
  TRAINING_STRATA,
  TRAINING_UNCHANGED,
  trainingClassCount,
  trainingClearedValue,
  trainingDirectory,
  trainingGoAroundValue,
  trainingFilePath,
  type TrainingColumn,
  type TrainingFlight,
  type TrainingHeadingBand,
  type TrainingStratum,
  type TrainingSample,
  type TrainingSelection,
  type TrainingSetHead,
  type TrainingSentence,
  type TrainingSentenceEvent,
  type TrainingVocabulary,
} from "./trainingSample";

/** MIRROR of the exporter's `OVERLAYS_SCHEMA` (`ts_transformer/instructions/training_files.py`): the manifest of overlays;
 *  v2 (2026-09-28) no longer names the set's sample by its file's sha256. */
export const TRAINING_OVERLAYS_SCHEMA = "aeroviz-training-overlays-v2";
/** MIRROR of `OVERLAY_KINDS`: what an overlay can be. */
export const TRAINING_OVERLAY_KINDS = [
  "executor-replay", "prior-prediction", "prior-generation", "prior-generation-augmented", "window-generation",
] as const;
export type TrainingOverlayKind = (typeof TRAINING_OVERLAY_KINDS)[number];
/** MIRROR of `executor_training_export.SCHEMA`: v2 (instruction-v3) gives each heading word judged its band and a
 *  verdict per row, and each flight judged its flown track as the judge read it; v1 (turns and holds) is refused. v4
 *  (2026-09-28): `base` is what the overlay shares with its set (`TrainingOverlayBase`), as in every overlay schema. */
export const TRAINING_EXECUTOR_SCHEMA = "aeroviz-training-executor-v5";
/** MIRROR of `executor_training_export.STATUSES`: one per word. */
export const TRAINING_EXECUTOR_STATUSES = [
  "inside", "outside", "not judged", "not reached", "superseded", "no check",
] as const;
export type TrainingExecutorStatus = (typeof TRAINING_EXECUTOR_STATUSES)[number];
/** MIRROR of `ts_transformer.autopilot.judge.OUTCOMES`: how the executor's flight ended — the replay's, and the live
 *  executor's when it flew on to the landing (pinned by the backend's `test_autopilot_segment.MirrorTest`). */
export const TRAINING_EXECUTOR_OUTCOMES = [
  "landed", "crossed_too_high", "crossed_off_runway", "crossed_other_runway", "crossed_without_capture", "ground_contact",
  "timeout", "dynamics_failure",
] as const;
export type TrainingExecutorOutcome = (typeof TRAINING_EXECUTOR_OUTCOMES)[number];
/** MIRROR of `prior_free_generation.BELOW_GLIDEPATH`: a free sentence under the procedure's altitudes stops at the first
 *  flown step that sank more than the track tolerance below the glidepath lower edge — whatever the executor made of the
 *  rest, its outcome is this. */
export const TRAINING_BELOW_GLIDEPATH = "below_glidepath";
/** MIRROR of `prior_free_generation.FREE_OUTCOMES`: how a model's own sentence ended — the judge's outcomes, or stopped
 *  below the glidepath. */
export const TRAINING_FREE_OUTCOMES = [...TRAINING_EXECUTOR_OUTCOMES, TRAINING_BELOW_GLIDEPATH] as const;
export type TrainingFreeOutcome = (typeof TRAINING_FREE_OUTCOMES)[number];
/** MIRROR of `traffic_loop.LOST_SEPARATION`: a sentence in a multi-aircraft window the judge ended for a loss of
 *  separation it answers for (`trainingTraffic.ts`) — it says nothing more and flies on in the scene. */
export const TRAINING_LOST_SEPARATION = "lost_separation";
/** How any model sentence ended: a free sentence's outcomes, and in a window the judge's end besides. Each reader
 *  admits only its own (a single flight's sentence is never `lost_separation`). */
export type TrainingSentenceOutcome = TrainingFreeOutcome | typeof TRAINING_LOST_SEPARATION;
/** MIRROR of `prior.masks.PROCEDURE_ALTITUDES`: the procedure's masks whose sentences the glidepath lower edge stops. */
export const TRAINING_PROCEDURE_ALTITUDES = "procedure-altitudes-v2";
/** MIRROR of `prior.masks.SETS`: every set of the procedure's masks a model can have spoken under; another is refused. */
export const TRAINING_PROCEDURE_MASK_SETS = [TRAINING_PROCEDURE_ALTITUDES] as const;
/** MIRROR of `prior.readout.RULES`: the causal rules the first-step runway readout sets the prior beside (B0, B1, B3;
 *  pinned by the backend's `test_autopilot_segment.MirrorTest`). */
export const TRAINING_PRIOR_RULES = ["B0_majority", "B1_active_config", "B3_same_sector_last"] as const;
/** MIRROR of `prior_training_export.SCHEMA`; v4 (2026-09-28): `base` as the executor's v4. */
export const TRAINING_PRIOR_SCHEMA = "aeroviz-training-prior-v4";
/** MIRROR of `prior_generation_training_export.SCHEMA`: the prior's own sentences, flown; v2 names the model (its name,
 *  round, run and start model) where v1 carried a free label; v4 (2026-09-28): `base` as the executor's v4. */
export const TRAINING_GENERATION_SCHEMA = "aeroviz-training-generation-v5";
/** MIRROR of `prior_generation_training_export.AUGMENTED_SCHEMA`: the same sentences flown from AUGMENTED starts (kind
 *  `prior-generation-augmented`, the Training module §2.8) — each flight with its move and moved observed rows, no readout;
 *  v2 (2026-09-28): `base` as the executor's v4. */
export const TRAINING_AUGMENTED_GENERATION_SCHEMA = "aeroviz-training-augmented-generation-v3";
/** MIRROR of `training_files.DATUM_TOLERANCE_M`: how far apart two readings of one flight's HAE − MSL may be, each read
 *  off a pair of heights written to 0.01 m. */
export const TRAINING_DATUM_TOLERANCE_M = 0.02;
/** How far a written position and a written height may be from another writing of the same one: the exporters write
 *  latitude and longitude to 7 decimals and heights to 2 (`training_files.rounded`). */
const WRITTEN_DEG = 1e-7;
const WRITTEN_M = 0.01;
/** A difference of two written decimals is off by up to this in their binary form. */
const BINARY_SLACK = 1e-9;
/** The two kinds a model's own sentences come in: from the set's own starts, and from augmented ones. */
export const TRAINING_GENERATION_KINDS = ["prior-generation", "prior-generation-augmented"] as const satisfies readonly TrainingOverlayKind[];
/** MIRROR of `prior_generation_training_export.MODEL_NAMES`: the prior's models, in the order they are trained — base
 *  (data alone), landing (post-trained on the landing reward), augmented (landing post-trained again on augmented
 *  starts), traffic (augmented post-trained in multi-aircraft scenes, M4; the user 2026-09-30), window (a traffic round
 *  post-trained again in windows whose every aircraft it commands; the user 2026-10-01). The views order them so. */
export const TRAINING_MODEL_NAMES = ["base", "landing", "augmented", "traffic", "window"] as const;
export type TrainingModelName = (typeof TRAINING_MODEL_NAMES)[number];
/** MIRROR of `ts_transformer.autopilot.judge.CROSSINGS`: the outcomes read at a crossing of a threshold (the pointed
 *  runway's, or another's for `crossed_other_runway`), which carry where it was crossed; no other outcome does. */
export const TRAINING_CROSSING_OUTCOMES: readonly TrainingFreeOutcome[] = [
  "landed", "crossed_too_high", "crossed_off_runway", "crossed_other_runway", "crossed_without_capture",
];

// ── shapes ───────────────────────────────────────────────────────────────────

export interface TrainingOverlayEntry {
  id: string;
  kind: TrainingOverlayKind;
  /** The set it is drawn over. */
  base: string;
  title: string;
  /** Relative to the airport's `training/` directory. */
  file: string;
  flights: number;
}

export interface TrainingOverlays {
  airport: string;
  overlays: TrainingOverlayEntry[];
  rejected: Array<{ id: string; problem: string }>;
}

/** What an overlay shares with the set it is drawn over, besides its flights — each matched against the loaded sample. */
export interface TrainingOverlayBase {
  setId: string;
  specSha256: string;
  /** The candidate runways a runway index points into, and the runway ends the landing rule reads. */
  candidatesSha256: string;
  /** The frame its metres are in. */
  airportFrame: TrainingSample["airportFrame"];
}

export interface TrainingExecutorCheck {
  name: string;
  ok: boolean;
  /** For a check over rows: how many of them were inside. */
  inside: number | null;
  rows: number | null;
}

/** The executor's verdict on ONE word of the sentence (its row, column and value are the word's). */
export interface TrainingExecutorWord {
  row: number;
  column: number;
  value: number;
  status: TrainingExecutorStatus;
  /** The flown track's step the executor was told the word at; null when it never was. */
  flownRow: number | null;
  /** A heading word the judge judged: its band over the flown rows it was judged on (from `flownRow` plus the lead),
   *  a verdict per row, on the chart branch of `TrainingExecutorFlight.judgedTrackDeg`; null for every other word. */
  heading: TrainingHeadingBand | null;
  checks: TrainingExecutorCheck[];
  /** Why a word is not judged, not reached, superseded or has no check of its own. */
  reason: string | null;
}

/** The flown track every sentence step, from row 0 to its outcome's row, on the executor's own clock. */
export interface TrainingExecutorTrack {
  tS: number[];
  eM: number[];
  nM: number[];
  lon: number[];
  lat: number[];
  altitudeM: number[];
  altitudeHaeM: number[];
  groundSpeedMps: number[];
  /** Unwrapped, on the observed smoothed track's branch at row 0. */
  trackDeg: number[];
  /** The attitude its aircraft is drawn in at each point (`trainingAttitude.ts`). */
  attitude: TrainingAttitude;
  /** The horizontal distance the executor flew. */
  distanceM: number[];
}

/** At the threshold: metres right of the centreline and above the threshold, and when. */
export interface TrainingCrossing {
  crossM: number;
  heightM: number;
  atS: number;
  /** The candidate runway crossed (the flight's candidates' order, as `firstRunway` / `lastRunway`): the pointed one,
   *  or another for `crossed_other_runway`. `crossM` and `heightM` are against it. */
  runway: number;
}

/** A flight of the set the replay does not fly: `group` says why. */
export interface TrainingExecutorUnflown {
  flightKey: string;
  datasetId: string;
  group: string;
  flown: false;
}

export interface TrainingExecutorFlown {
  flightKey: string;
  datasetId: string;
  /** "own dynamics" or "stand-in dynamics". */
  group: string;
  flown: true;
  outcome: TrainingExecutorOutcome;
  flewTheSentence: boolean;
  endS: number;
  crossing: TrainingCrossing | null;
  /** Why the labeller's gate refused the flown track (no word is then judged). */
  refused: string | null;
  /** The formal replay's evaluation verdicts: the flown track's and the observed one's. */
  evaluation: { replay: string; observed: string };
  alignment: { meanHorizontalDistanceM: number; meanVerticalDistanceM: number; landingTimeMinusObservedS: number | null };
  limits: { cycles: number; bound: Record<string, number> };
  /** The judge's own counts: the words it judged, those inside (checked against the words' own verdicts). */
  counts: { wordsJudged: number; wordsInside: number; headingWordsNotJudged: number };
  track: TrainingExecutorTrack;
  /** The flown track as the judge read it — the labeller's gate: smoothed, cut at the landing — every sentence step
   *  from row 0 (its step k is `track.tS[k]`, checked), on the same branch as `track.trackDeg`; null when the gate
   *  refused it, and for a dynamics failure (the exporter draws no band there: its words keep their statuses and
   *  checks). */
  judgedTrackDeg: number[] | null;
  /** One per word of the base flight's sentence, in its event order. */
  words: TrainingExecutorWord[];
}

export type TrainingExecutorFlight = TrainingExecutorUnflown | TrainingExecutorFlown;

export interface TrainingGateCell {
  flights: number;
  landed: number | null;
  wordsJudged: number;
  wordsInside: number | null;
  observedPasses: number;
  evaluationPaired: number | null;
  /** Each gate's verdict — for the gated group only. */
  clears: { landed: boolean; words: boolean; evaluation: boolean } | null;
  /** Why a group is reported, not gated. */
  notGated: string | null;
  outcomes: Record<string, number>;
}

/** One group's cells per stratum ("straight-in", "vectored" — each when the group has such flights — and "all"). */
export type TrainingGateStrata = Partial<Record<(typeof TRAINING_STRATA)[number], TrainingGateCell>> & { all: TrainingGateCell };

/** group → place → its strata: at the overlay's airport (`here`) and at all airports (`all`). */
export type TrainingGateTable = Record<string, { here: TrainingGateStrata; all: TrainingGateStrata }>;

export interface TrainingExecutorOverlay {
  overlayId: string;
  airport: string;
  base: TrainingOverlayBase;
  executor: { specSha256: string; wordClock: string; cycleS: number; params: Array<{ name: string; value: number | string }> };
  /** `drawn`: how many flights the formal replay drew (every flyable one of its split). */
  replay: { split: string; writtenUtc: string; gateShare: number; drawn: { flights: number } };
  gate: TrainingGateTable;
  /** The base set's flights, in its order. */
  flights: TrainingExecutorFlight[];
}

/** The prior at one column of one flight, predicted step by predicted step (from `firstPredictedRow`). */
export interface TrainingPriorColumn {
  /** How many words are ranked at each step (fewer than asked when the column has fewer values). */
  k: number;
  /** The probability that a word is said at the step (1 at the first predicted step, which says every column). */
  changeP: number[];
  /** Row-major, `k` per step: the most likely words GIVEN that one is said, and their probabilities. */
  words: number[];
  wordsP: number[];
  /** The probability of what the truth sentence says at the step (a word, or unchanged; at the first predicted step
   *  the word in force there). */
  truthP: number[];
}

export interface TrainingPriorFlight {
  flightKey: string;
  datasetId: string;
  rows: number;
  /** The first step the prior speaks at; the rows before it are only observed and carry no prediction. */
  firstPredictedRow: number;
  /** Per predicted step. */
  nllPerStep: number;
  columnNllPerStep: number[];
  /** In `TRAINING_COLUMNS` order. */
  columns: TrainingPriorColumn[];
}

export interface TrainingPriorColumnReadout {
  nllPerStep: number;
  changeSteps: number;
  /** The first predicted step's word: how often the prior's most likely one is the truth's. */
  firstStepTop1: number;
  /** After the first predicted step, where the truth says a word; null exactly for a column that never changes there
   *  (`changeSteps` 0). */
  changeProbabilityWhereChanged: number | null;
  top1GivenChange: number | null;
  /** MIRROR of `prior.readout.TOP_K` (5): the truth among the prior's five most likely words. */
  top5GivenChange: number | null;
  falseChangeShareWhereKept: number;
}

/** The first predicted step's runway: right, in the right landing direction, and right within it. */
export interface TrainingPriorRunwayReadout {
  top1: number;
  direction: number;
  sideGivenDirection: number | null;
}

export interface TrainingPriorReadout {
  split: string;
  steps: number;
  bestEpoch: number;
  model: { nllPerStep: number; perplexityPerStep: number; perColumn: Record<TrainingColumn, TrainingPriorColumnReadout> };
  /** Negative log-likelihood per step per column, and `all`. */
  baselines: { repeat: Record<TrainingColumn | "all", number>; previousWord: Record<TrainingColumn | "all", number> };
  /** The prior's first-step runway beside each airport's own runway frequency and the causal rules
   *  (`TRAINING_PRIOR_RULES`). */
  firstStepRunway: {
    model: TrainingPriorRunwayReadout;
    airportFrequency: TrainingPriorRunwayReadout;
    rules: Record<string, TrainingPriorRunwayReadout>;
  };
}

export interface TrainingPriorOverlay {
  overlayId: string;
  airport: string;
  base: TrainingOverlayBase;
  prior: { checkpointSha256: string; parameters: number; model: { dModel: number; layers: number; heads: number }; method: string };
  readout: TrainingPriorReadout;
  flights: TrainingPriorFlight[];
}

/** The executor's flight of one of the prior's own sentences, every sentence step on the flight's own clock (``tS``
 *  from the first predicted row's time) to its outcome's row. */
export interface TrainingGeneratedTrack {
  tS: number[];
  lon: number[];
  lat: number[];
  altitudeM: number[];
  altitudeHaeM: number[];
  groundSpeedMps: number[];
  /** The attitude its aircraft is drawn in at each point (`trainingAttitude.ts`). */
  attitude: TrainingAttitude;
}

/** ONE sentence the prior said over a flight (one sample), flown by the executor: its words from the first predicted row
 *  on (the rows before are observed only; the first predicted row says every column), and how the flight ended. Its
 *  words run to where the EXECUTOR stopped, which is after ``endS`` for three outcomes the judge reads earlier (crossing
 *  without the capture, another runway's threshold, the stall cut-off): the rows after the end are the model still speaking to a flight that is
 *  over — counted as the formal readout counts them, shaded in the bar. */
export interface TrainingGeneratedSentence extends TrainingSentence {
  sample: number;
  /** ``below_glidepath`` only under the procedure's altitudes: stopped at its step's end state, words and track;
   *  ``lost_separation`` only in a window (`trainingTraffic.ts`). */
  outcome: TrainingSentenceOutcome;
  /** When the flight ended, on the flight's own clock. */
  endS: number;
  crossing: TrainingCrossing | null;
  /** The runway pointed at the first predicted step and at the end (candidate indices). */
  firstRunway: number;
  lastRunway: number;
  runwayChanges: number;
  goArounds: number;
  clearedAtEnd: boolean;
  /** Per column the speaker masks (`prior.generate.Speaker.forbidden`, named by column): the mean probability a step the
   *  prior put on what the masks removed. */
  forbiddenMass: Record<string, number>;
  track: TrainingGeneratedTrack;
}

/** How an augmented start was moved (`prior.augment`): rotated clockwise about the airport, raised, sped up. */
export interface TrainingAugmentation {
  rotationDeg: number;
  altitudeM: number;
  speedScale: number;
}

/** The observed rows the prior read before it spoke — rows 0 … firstPredictedRow − 1, a step apart from 0 — moved like its
 *  start; its samples' tracks start at the first predicted row. */
export interface TrainingObservedRows {
  tS: number[];
  lon: number[];
  lat: number[];
  altitudeM: number[];
  altitudeHaeM: number[];
}

/** A flight's augmented start: how many draws it took (null: none drawn — not on its own dynamics), the move (null: none of
 *  the draws was plausible, so it is not flown) and the moved observed rows. */
export interface TrainingAugmentedStart {
  draws: number | null;
  augmentation: TrainingAugmentation | null;
  observed: TrainingObservedRows | null;
}

/** A flight of the set the val readout does not fly (not on its own dynamics): `group` says why. */
export interface TrainingGenerationFlight {
  flightKey: string;
  datasetId: string;
  group: string;
  flown: boolean;
  /** One per sample, in sample order; none when not flown. */
  samples: TrainingGeneratedSentence[];
  /** From augmented starts: this flight's; null from its own start. */
  start: TrainingAugmentedStart | null;
}

/** The landed share of the prior's sentences (and of the labelled words flown from the same row), in all and per
 *  approach kind, as the formal val free generation counted them. */
/** "all", and each approach kind (null where there was no flight of it). */
export type TrainingGenerationReadoutCells = { all: TrainingGenerationCell } & Record<TrainingStratum, TrainingGenerationCell | null>;

/** The same counted over a set's own flights — where "all" too is null when the model flies none of them. */
export type TrainingGenerationSetCells = Record<"all" | TrainingStratum, TrainingGenerationCell | null>;

export interface TrainingGenerationCell {
  flights: number;
  landed: number;
}

/** A set of the procedure's masks a model spoke under (`prior.masks`): its name and the digest of the data it read. */
export interface TrainingProcedureMask {
  name: string;
  dataSha256: string;
}

/** Which model spoke: its name and round (`trainingModelLabel`), the run holding its rounds, its checkpoint and — a
 *  post-trained round — the method and the model it started from. */
export interface TrainingGenerationModel {
  name: TrainingModelName;
  /** A post-trained round's number (from 1); null for base. */
  round: number | null;
  /** From `4dTrajectory/outputs/` on: the directory holding a post-training run's rounds; base's own directory. */
  run: string;
  checkpointSha256: string;
  variant: string;
  /** The method's checkpoint schema and the model the round started from, named the same way; null for base. */
  fineTuning: { schema: string; from: string; fromName: TrainingModelName; fromRound: number | null } | null;
}

/** What every overlay of a model's own sentences says of itself — over a read-back set's flights
 *  (`TrainingGenerationOverlay`) or in a window set's windows (`trainingTraffic.TrainingWindowGenerationOverlay`): who the
 *  model is and how its sentences were drawn. A view of one flight's sentences (`TrainingGenerationView`) reads only this. */
export interface TrainingGenerationHead {
  overlayId: string;
  airport: string;
  base: TrainingOverlayBase;
  model: TrainingGenerationModel;
  generation: {
    samples: number;
    temperature: number;
    seed: number;
    /** The first row the prior speaks at; every sentence opens there. */
    firstPredictedRow: number;
    /** The procedure's masks it spoke under — its own (`prior.masks`), each with the digest of the data it read; empty:
     *  the vocabulary's rules alone. The live executor sends them back with a sample it flies again. */
    procedureMasks: TrainingProcedureMask[];
    executor: { specSha256: string; wordClock: string; cycleS: number; timeoutFactor: number };
    /** From augmented starts: the seed every flight's move was drawn with, the draws allowed, the half-widths of the
     *  draws and the time limit's factor (of the source's observed remaining time); null from the set's own starts. */
    augment: {
      seed: number; tries: number; limits: { rotationDeg: number; altitudeM: number; speedFraction: number }; timeoutFactor: number;
    } | null;
  };
}

export interface TrainingGenerationOverlay extends TrainingGenerationHead {
  /** The prior's formal val free generation, when the exporter was given it. */
  readout: {
    split: string;
    writtenUtc: string;
    /** ``perAirport`` 0: every flight of the split. */
    drawn: { flights: number; perAirport: number };
    /** At the overlay's airport, and over every airport the readout drew. */
    prior: { here: TrainingGenerationReadoutCells; all: TrainingGenerationReadoutCells };
    labelled: { here: TrainingGenerationReadoutCells; all: TrainingGenerationReadoutCells };
  } | null;
  flights: TrainingGenerationFlight[];
}

/** What the views draw for the selected flight: its overlay's own data, and the flight's entry. */
export interface TrainingExecutorView {
  overlay: TrainingExecutorOverlay;
  flight: TrainingExecutorFlight;
}

export interface TrainingPriorView {
  overlay: TrainingPriorOverlay;
  flight: TrainingPriorFlight;
}

export interface TrainingGenerationView {
  overlay: TrainingGenerationHead;
  flight: TrainingGenerationFlight;
}

/** How the views call a model: ``base``, ``landing r1``, ``augmented r3``. */
export function trainingModelLabel(model: { name: TrainingModelName; round: number | null }): string {
  return model.round === null ? model.name : `${model.name} r${model.round}`;
}

/** A run as the views name it: its path under `…/prior/` (``v3_stage2_clip_20260926/aug_s1337``; the reader refuses a
 *  run elsewhere). */
export function trainingRunName(run: string): string {
  return run.slice(run.indexOf(PRIOR_RUNS) + PRIOR_RUNS.length);
}

/** A run named as briefly as tells it from ``runs`` (its stage's others): its campaign directory
 *  (``v3_stage2_clip_20260926``), or its whole name when two runs share that. */
function shortRunName(run: string, runs: string[]): string {
  const campaign = (item: string) => trainingRunName(item).split("/")[0];
  return runs.filter((other) => campaign(other) === campaign(run)).length > 1 ? trainingRunName(run) : campaign(run);
}

/** Where every prior run lives, as the exporter writes ``run`` (from `4dTrajectory/outputs/` on). */
const PRIOR_RUNS = "/prior/";

/** One model of a set's published sentences — a name and the run its rounds come from — with every round published,
 *  each round once (base: its one member); ``title`` names it (its tab), ``memberLabel`` each member (its round, its
 *  readout, its table row, its legend). The run is said — by its campaign directory, whole when that is shared — when another group
 *  has the same name (two runs of a stage). An overlay that
 *  repeats a round already published for its run (the same checkpoint exported twice) is a group of its own, named
 *  with its overlay id — two "r3" would say nothing. */
export interface TrainingModelGroup<T> {
  key: string;
  name: TrainingModelName;
  title: string;
  members: T[];
  memberLabel: (member: T) => string;
}

/** The published models of a set grouped for the views — by name in training order, then run, each group's rounds in
 *  order. ``overlayOf`` reads an item's overlay (items are overlays, or views of them). */
export function trainingModelGroups<T>(items: T[], overlayOf: (item: T) => TrainingGenerationHead): TrainingModelGroup<T>[] {
  const modelOf = (item: T) => overlayOf(item).model;
  const roundKey = (item: T) => `${modelOf(item).name} ${modelOf(item).run} ${modelOf(item).round}`;
  const repeated = new Set(items.map(roundKey).filter((key, index, keys) => keys.indexOf(key) !== index));
  const buckets = new Map<string, T[]>();
  for (const item of items) {
    const { name, run } = modelOf(item);
    const key = repeated.has(roundKey(item)) ? overlayOf(item).overlayId : `${name} ${run}`;
    buckets.set(key, [...(buckets.get(key) ?? []), item]);
  }
  const order = (item: T) => [TRAINING_MODEL_NAMES.indexOf(modelOf(item).name), modelOf(item).run, overlayOf(item).overlayId] as const;
  const ordered = [...buckets.entries()].map(([key, members]) => ({
    key, members: members.sort((a, b) => (modelOf(a).round ?? 0) - (modelOf(b).round ?? 0)),
  })).sort((a, b) => {
    const [x, y] = [order(a.members[0]), order(b.members[0])];
    return x[0] - y[0] || x[1].localeCompare(y[1]) || x[2].localeCompare(y[2]);
  });
  return ordered.map(({ key, members }) => {
    const { name, run } = modelOf(members[0]);
    const twin = repeated.has(roundKey(members[0]));
    const runs = [...new Set(items.filter((item) => modelOf(item).name === name).map((item) => modelOf(item).run))];
    const qualifier = twin ? ` · ${overlayOf(members[0]).overlayId}` : runs.length > 1 ? ` · ${shortRunName(run, runs)}` : "";
    return {
      key, name, members, title: `${name}${qualifier}`,
      memberLabel: (member: T) => `${trainingModelLabel(modelOf(member))}${qualifier}`,
    };
  });
}

/** WHICH SENTENCE the views read: the truth's (null) or one sample of a model's own (`TrainingGenerationView`). */
export interface TrainingSource {
  overlayId: string;
  sample: number;
}

// ── reading one step ─────────────────────────────────────────────────────────

/** An overlay's view if it is of the flight on screen — drawn over its set, at its airport — else null: in the render
 *  after a switch, the view still published is the last flight's (a flight key alone repeats across sets). */
export function overlayOnScreen<V extends TrainingExecutorView | TrainingPriorView | TrainingGenerationView>(
  view: V | null, selection: TrainingSelection | null,
): V | null {
  if (view === null || selection === null) return null;
  return view.overlay.airport === selection.airport && view.overlay.base.setId === selection.setId
    && view.flight.flightKey === selection.flight.flightKey ? view : null;
}

/** The executor's verdict on the word at (row, column) of the sentence, if the flight was flown. */
export function executorWordAt(flight: TrainingExecutorFlight, row: number, column: TrainingColumn): TrainingExecutorWord | null {
  if (!flight.flown) return null;
  const index = TRAINING_COLUMN_INDEX[column];
  return flight.words.find((word) => word.row === row && word.column === index) ?? null;
}

/** The value the truth sentence gives a column at a step: its word, or `TRAINING_UNCHANGED`. */
export function truthAt(flight: TrainingFlight, column: TrainingColumn, row: number): number {
  const index = TRAINING_COLUMN_INDEX[column];
  return flight.words.events.find((event) => event.row === row && event.column === index)?.value ?? TRAINING_UNCHANGED;
}

/** What the prior's `truthP` is the probability of at a predicted step: at the FIRST predicted step, which says every
 *  column, the word in force there; after it, what the truth sentence says (a word, or `TRAINING_UNCHANGED`). */
export function priorTruthAt(flight: TrainingFlight, predicted: TrainingPriorFlight, column: TrainingColumn, row: number): number {
  return row === predicted.firstPredictedRow ? flight.words.inForce[TRAINING_COLUMN_INDEX[column]][row] : truthAt(flight, column, row);
}

/** What the prior gives one column at one step: a word's probability, the ranked words, the truth (`priorTruthAt`) and
 *  its probability; null at a step before the first predicted one (observed only). */
export function priorStep(flight: TrainingFlight, predicted: TrainingPriorFlight, column: TrainingColumn, row: number) {
  if (row < predicted.firstPredictedRow) return null;
  const at = row - predicted.firstPredictedRow;
  const values = predicted.columns[TRAINING_COLUMN_INDEX[column]];
  const ranked = Array.from({ length: values.k }, (_, rank) => ({
    value: values.words[at * values.k + rank],
    p: values.wordsP[at * values.k + rank],
  }));
  return { changeP: values.changeP[at], truth: priorTruthAt(flight, predicted, column, row), truthP: values.truthP[at], ranked };
}

/** The model sentence the views read: the chosen overlay's view of the flight on screen and its chosen sample — ``sentence``
 *  null for a flight its readout does not fly (every flown flight holds every sample: the reader checks it) — or null for
 *  the truth (no source, or an overlay not published for the set on screen). */
export function generationOnScreen(
  views: TrainingGenerationView[], source: TrainingSource | null, selection: TrainingSelection | null,
): { view: TrainingGenerationView; sentence: TrainingGeneratedSentence | null } | null {
  if (source === null) return null;
  const view = views.map((item) => overlayOnScreen(item, selection)).find((item) => item?.overlay.overlayId === source.overlayId);
  if (!view) return null;
  // a sample number chosen over an overlay of the same id with more samples (another airport's) reads this one's last
  const { samples } = view.flight;
  return { view, sentence: samples.length === 0 ? null : samples[Math.min(source.sample, samples.length - 1)] };
}

/** Whether an overlay's sentences were flown from augmented starts (kind `prior-generation-augmented`), not the set's own. */
export function isAugmentedStart(overlay: TrainingGenerationHead): boolean {
  return overlay.generation.augment !== null;
}

/** Why a flight of a generation overlay holds no sample: not on its own dynamics (its group), or — from augmented starts —
 *  none of its draws was plausible. */
export function generationUnflownReason(flight: TrainingGenerationFlight): string {
  return flight.start !== null && flight.start.draws !== null
    ? `no plausible augmented start in ${flight.start.draws} draws` : flight.group;
}

/** The move of an augmented start in a few words: "+7.2°, +84 m, ×1.03". */
export function augmentationText(move: TrainingAugmentation): string {
  const signed = (value: number, digits: number) => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
  return `${signed(move.rotationDeg, 1)}°, ${signed(move.altitudeM, 0)} m, ×${move.speedScale.toFixed(3)}`;
}

/** A model sentence the views read (`generationOnScreen`) as the source it is — its overlay and the sample shown — or
 *  null for the truth (no model chosen, or one its readout does not fly this flight). What the live executor flies and
 *  what its answer must be of (`trainingAutopilot.autopilotOnScreen`). */
export function sourceOf(read: ReturnType<typeof generationOnScreen>): TrainingSource | null {
  return read === null || read.sentence === null ? null : { overlayId: read.view.overlay.overlayId, sample: read.sentence.sample };
}

/** The sentence read for the flight on screen, as a source (`sourceOf`), with that sample. */
export function sourceOnScreen(
  views: TrainingGenerationView[], source: TrainingSource | null, selection: TrainingSelection | null,
): { source: TrainingSource | null; sentence: TrainingGeneratedSentence | null } {
  const read = generationOnScreen(views, source, selection);
  const shown = sourceOf(read);
  return { source: shown, sentence: shown === null ? null : read!.sentence };
}

/** Where the sentence bar's axis ends for a flight: its last step's end — or the sentence read's, when that runs longer
 *  (its rows, or its end). */
export function sentenceAxisEndS(flight: TrainingFlight, stepS: number, read: TrainingGeneratedSentence | null): number {
  return Math.max(flight.rows * stepS, read === null ? 0 : Math.max(read.rows * stepS, read.endS));
}

/** The row of a model's sentence at a flight time — past its last row when the cursor is (the observed flight may last
 *  longer): no word is in force there. */
export function generatedRowAt(stepS: number, seconds: number): number {
  return Math.max(Math.floor(seconds / stepS), 0);
}

/** The samples of a model that landed over the set's flights it flies: in all and per approach kind (``flights``: the
 *  set's, in its order — the overlay's are the same) — a count of the samples on screen, nothing recomputed. */
export function generationLanded(overlay: TrainingGenerationOverlay, flights: TrainingFlight[]): TrainingGenerationSetCells {
  const count = (keep: (index: number) => boolean): TrainingGenerationCell | null => {
    const said = overlay.flights.flatMap((flight, index) => (keep(index) ? flight.samples : []));
    return said.length === 0 ? null
      : { flights: said.length, landed: said.filter((item) => item.outcome === "landed").length / said.length };
  };
  const strata = Object.fromEntries(TRAINING_STRATA.map((stratum) => [stratum, count((index) => flights[index].stratum === stratum)]));
  return { all: count(() => true), ...strata } as TrainingGenerationSetCells;
}

/** Where a sentence's track shows a word in force: its points from the word's row to the next word's (point k is row
 *  ``firstRow + k``; the last point may end part-way through a step), or null when the flight ended before the word. */
export function generatedTrackRows(sentence: TrainingGeneratedSentence, firstRow: number, row: number, endRow: number) {
  const last = sentence.track.tS.length - 1;
  const first = row - firstRow;
  if (first > last) return null;
  return { first, last: Math.min(endRow - firstRow, last) };
}

/** The heading word the judge left to intercept the final on its own, when its band also had rows judged: the one word
 *  with two checks, its band's and the intercept's — the replay gate counts it twice, and inside once when its band's
 *  check passed. */
export function countedTwice(word: TrainingExecutorWord): boolean {
  return word.column === TRAINING_COLUMN_INDEX.heading && word.checks.length === 2;
}

/** How the executor's replay of a flight went, in one of three — one rule for every view that says it, in a word or a
 *  colour: it landed and every judged word was inside its envelope; it landed, but words were outside, or the labeller's
 *  gate refused its track (then no word is judged); it did not land. */
export type TrainingReplayKind = "clean" | "flawed" | "not landed";

/** The replay's verdict on a flight: its kind, the words out as the gate counts them (judged − inside: the list's
 *  "43/45" says 2), and whether the gate refused its track. */
export function replayVerdict(flight: TrainingExecutorFlown): { kind: TrainingReplayKind; wordsOut: number; refused: boolean } {
  const wordsOut = flight.counts.wordsJudged - flight.counts.wordsInside;
  const refused = flight.refused !== null;
  const kind = flight.outcome !== "landed" ? "not landed" : wordsOut > 0 || refused ? "flawed" : "clean";
  return { kind, wordsOut, refused };
}

/** The flight's words the executor judged, and how many of them it flew inside their envelopes. */
export function executorWordCounts(flight: TrainingExecutorFlown) {
  const statuses = flight.words.map((word) => word.status);
  return {
    inside: statuses.filter((status) => status === "inside").length,
    outside: statuses.filter((status) => status === "outside").length,
    notJudged: statuses.filter((status) => status === "not judged").length,
    notReached: statuses.filter((status) => status === "not reached").length,
    superseded: statuses.filter((status) => status === "superseded").length,
  };
}

// ── the manifest ─────────────────────────────────────────────────────────────

function parseEntry(entry: Reader): TrainingOverlayEntry {
  return {
    id: entry.string("id"),
    kind: entry.oneOf("kind", TRAINING_OVERLAY_KINDS),
    base: entry.string("base"),
    title: entry.string("title"),
    file: entry.string("file"),
    flights: entry.count("flights"),
  };
}

/** Parse the overlays manifest. A bad entry is rejected on its own; only a manifest that is not one fails the call. */
export function parseTrainingOverlays(raw: unknown): Parsed<TrainingOverlays> {
  const manifest = parseManifest(raw,
    { name: "overlays manifest", schema: TRAINING_OVERLAYS_SCHEMA, listKey: "overlays", entryName: "overlay" }, parseEntry);
  if (!manifest.ok) return manifest;
  return { ok: true, value: { airport: manifest.value.airport, overlays: manifest.value.entries, rejected: manifest.value.rejected } };
}

/** The overlays of one kind drawn over a set, in the manifest's order (the latest listed last). */
export function trainingOverlaysOf(overlays: TrainingOverlays, setId: string, kind: TrainingOverlayKind): TrainingOverlayEntry[] {
  return overlays.overlays.filter((entry) => entry.base === setId && entry.kind === kind);
}

// ── the binding ──────────────────────────────────────────────────────────────

/** The payload's own account of itself, against the manifest entry that listed it and the sample the panel loaded. */
export function parseBase(reader: Reader, entry: TrainingOverlayEntry, sample: TrainingSetHead): TrainingOverlayBase {
  const overlayId = reader.string("overlayId");
  if (overlayId !== entry.id) reader.fail(`overlayId is ${overlayId}, but the manifest lists it as ${entry.id}`);
  const airport = reader.string("airport");
  if (airport !== sample.airport) reader.fail(`is drawn at ${airport}, but the loaded sample is ${sample.airport}'s`);
  const base = reader.child("base");
  const frame = base.child("airportFrame");
  const found = {
    setId: base.string("setId"),
    specSha256: base.string("specSha256"),
    candidatesSha256: base.string("candidatesSha256"),
    airportFrame: { code: frame.string("code"), lat: frame.number("lat"), lon: frame.number("lon"), elevationM: frame.number("elevationM") },
  };
  if (found.setId !== entry.base) base.fail(`is set ${found.setId}, but the manifest lists it over set ${entry.base}`);
  if (found.setId !== sample.setId) base.fail(`is drawn over set ${found.setId}, but the loaded sample is ${sample.setId}`);
  if (found.specSha256 !== sample.vocabulary.specSha256) {
    base.fail(`is of spec ${found.specSha256.slice(0, 12)}, the loaded sample of ${sample.vocabulary.specSha256.slice(0, 12)}`);
  }
  if (found.candidatesSha256 !== sample.candidatesSha256) {
    base.fail(`has candidates ${found.candidatesSha256.slice(0, 12)}, the loaded sample ${sample.candidatesSha256.slice(0, 12)}`);
  }
  const own = sample.airportFrame;
  if ((Object.keys(own) as Array<keyof typeof own>).some((key) => found.airportFrame[key] !== own[key])) {
    frame.fail(`is ${JSON.stringify(found.airportFrame)}, the loaded sample's ${JSON.stringify(own)}`);
  }
  return found;
}

/** A set flight's HAE − MSL: its first row's ellipsoid height less its reported height (MIRROR of
 *  `training_files.set_flight_datum_m`). */
function setFlightDatumM(flight: TrainingFlight): number {
  return flight.signals.altitudeHaeM[0] - flight.signals.raw.altitudeM[0];
}

/** Heights an overlay draws beside a set flight stand on that flight's own datum: every row's HAE − MSL is the set
 *  flight's, to two readings' rounding. */
export function requireSetDatum(reader: Reader, heights: { altitudeM: number[]; altitudeHaeM: number[] }, flight: TrainingFlight) {
  const datum = setFlightDatumM(flight);
  const off = heights.altitudeM.findIndex(
    (msl, row) => Math.abs(heights.altitudeHaeM[row] - msl - datum) > TRAINING_DATUM_TOLERANCE_M + BINARY_SLACK);
  if (off >= 0) {
    reader.fail(`draws row ${off} ${(heights.altitudeHaeM[off] - heights.altitudeM[off]).toFixed(2)} m HAE − MSL, the set's ` +
      `flight ${datum.toFixed(2)} m`);
  }
}

/** A track an overlay flew from a set flight's observed state starts where the set's observed track is at ``row``: its
 *  first point is that row's position and reported height, to the written rounding — the flights' keys alone would keep
 *  an overlay over a set exported again from other tracks. (A start moved by an augmentation is not the set's.) */
export function requireSetStart(reader: Reader, track: { lon: number[]; lat: number[]; altitudeM: number[] }, flight: TrainingFlight,
  row: number) {
  const { lon, lat, raw } = flight.signals;
  if (Math.abs(track.lon[0] - lon[row]) > WRITTEN_DEG + BINARY_SLACK || Math.abs(track.lat[0] - lat[row]) > WRITTEN_DEG + BINARY_SLACK
    || Math.abs(track.altitudeM[0] - raw.altitudeM[row]) > WRITTEN_M + BINARY_SLACK) {
    reader.fail(`starts at (${track.lat[0]}, ${track.lon[0]}, ${track.altitudeM[0]} m), but the set's flight is at ` +
      `(${lat[row]}, ${lon[row]}, ${raw.altitudeM[row]} m) at row ${row}, where it was flown from`);
  }
}

/** The payload's flights, one per flight of the sample, in its order. */
function eachFlight<T>(reader: Reader, sample: TrainingSetHead, read: (item: Reader, flight: TrainingFlight) => T): T[] {
  const list = reader.list("flights");
  if (list.length !== sample.flights.length) reader.fail(`holds ${list.length} flights, the set ${sample.flights.length}`);
  return list.map((raw, index) => {
    const flight = sample.flights[index];
    const item = Reader.of(raw, `flight ${flight.flightKey}`);
    if (item.string("flightKey") !== flight.flightKey || item.string("datasetId") !== flight.datasetId) {
      item.fail(`is ${item.raw("flightKey")}, but the set's flight ${index} is ${flight.flightKey}`);
    }
    return read(item, flight);
  });
}

// ── what the executor's judge says of a word (the replay's and the live executor's) ──

export function readCrossing(reader: Reader): TrainingCrossing {
  return { crossM: reader.number("crossM"), heightM: reader.number("heightM"), atS: reader.number("atS"),
    runway: reader.count("runway") };
}

/**
 * A word's verdict as the executor's judge gives it — one rule for the replay's words and the live executor's: a word
 * inside or outside carries its checks and they decide it (inside exactly when every one passed); any other status
 * says why instead; and when the labeller's gate refused the flown track (``refused``), no word is judged at all.
 */
export function readWordVerdict<S extends string>(word: Reader, statuses: readonly S[], refused: string | null) {
  const status = word.oneOf("status", statuses);
  const checks: TrainingExecutorCheck[] = word.children("checks").map((check) => ({
    name: check.string("name"), ok: check.boolean("ok"), inside: check.nullableCount("inside"), rows: check.nullableCount("rows"),
  }));
  const judged = status === "inside" || status === "outside";
  if (judged && (checks.length === 0 || (status === "inside") !== checks.every((check) => check.ok))) {
    word.fail(`is ${status}, but its checks say ${checks.map((check) => `${check.name} ${check.ok}`).join(", ") || "nothing"}`);
  }
  const reason = word.nullableString("reason");
  if (!judged && reason === null) word.fail(`is ${status} and says no reason`);
  if (judged && refused !== null) word.fail(`is ${status}, but the gate refused the flown track (${refused}): nothing is judged`);
  return { status, checks, reason, judged };
}

/**
 * A heading word the judge judged on flown rows: its band (`readHeadingBand`, its rows ending BY ``stopBy`` — the
 * judge's own clearance, capture and track end are not the reader's to know) and what the band says of its verdict:
 * a check counts exactly its flags; a word with no row judged is not judged — or outside, when the executor left it
 * to intercept the final on its own, which fails it whatever its rows; a word with rows judged is inside or outside.
 */
export function readJudgedBand(
  word: Reader, verdict: { status: string; checks: TrainingExecutorCheck[]; judged: boolean }, flownRow: number,
  targetDeg: number, vocabulary: TrainingVocabulary, stopBy: number,
): TrainingHeadingBand {
  const band = readHeadingBand(word.child("heading"), flownRow, targetDeg, vocabulary, { by: stopBy });
  const rows = band.inside.length;
  if (rows === 0 && verdict.status !== "not judged" && verdict.status !== "outside") word.fail(`is ${verdict.status} with no row judged`);
  if (rows > 0 && !verdict.judged) word.fail(`is ${verdict.status}, but ${rows} of its rows were judged`);
  const counted = band.inside.filter(Boolean).length;
  if (rows > 0 && !verdict.checks.some((check) => check.rows === rows && check.inside === counted)) {
    word.fail(`its band counts ${counted} of ${rows} rows inside, and no check says so`);
  }
  return band;
}

// ── the executor ─────────────────────────────────────────────────────────────

function parseTrack(reader: Reader): TrainingExecutorTrack {
  const tS = reader.numbers("tS");
  if (tS.length < 1 || tS[0] !== 0) reader.fail("does not start at 0 s");
  if (tS.some((value, index) => index > 0 && value <= tS[index - 1])) reader.fail("tS does not run forward");
  const n = tS.length;
  return {
    tS, eM: reader.numbers("eM", n), nM: reader.numbers("nM", n), lon: reader.numbers("lon", n), lat: reader.numbers("lat", n),
    altitudeM: reader.numbers("altitudeM", n), altitudeHaeM: reader.numbers("altitudeHaeM", n),
    groundSpeedMps: reader.numbers("groundSpeedMps", n), trackDeg: reader.numbers("trackDeg", n),
    distanceM: reader.numbers("distanceM", n), attitude: readAttitude(reader.child("attitude"), n),
  };
}

function parseExecutorWords(
  reader: Reader, flight: TrainingFlight, vocabulary: TrainingVocabulary, judgedRows: number | null, refused: string | null,
): TrainingExecutorWord[] {
  const list = reader.children("words");
  const events = flight.words.events;
  if (list.length !== events.length) reader.fail(`judges ${list.length} words, the sentence says ${events.length}`);
  return list.map((word, index) => {
    const event = events[index];
    if (word.number("row") !== event.row || word.number("column") !== event.column || word.number("value") !== event.value) {
      word.fail(`is (${word.raw("row")}, ${word.raw("column")}, ${word.raw("value")}), but the sentence's word ${index} is ` +
        `(${event.row}, ${event.column}, ${event.value})`);
    }
    const verdict = readWordVerdict(word, TRAINING_EXECUTOR_STATUSES, refused);
    const flownRow = word.nullableCount("flownRow");
    // A HEADING WORD THE JUDGE JUDGED — told on a step of the flown track it read — carries its band on the flown rows,
    // and no other word does.
    const banded = event.column === TRAINING_COLUMN_INDEX.heading && flownRow !== null && judgedRows !== null
      && flownRow < judgedRows;
    if ((word.raw("heading") !== null) !== banded) {
      word.fail(banded ? "is a heading word the judge judged on the flown track, and carries no band"
        : "carries a heading band, but it is not a heading word the judge judged on a flown track");
    }
    const heading = banded
      ? readJudgedBand(word, verdict, flownRow!, vocabulary.headingTargetsDeg[event.value], vocabulary, judgedRows!)
      : null;
    return {
      row: event.row, column: event.column, value: event.value, status: verdict.status, flownRow, heading,
      checks: verdict.checks, reason: verdict.reason,
    };
  });
}

function parseExecutorFlight(item: Reader, flight: TrainingFlight, vocabulary: TrainingVocabulary): TrainingExecutorFlight {
  const base = { flightKey: flight.flightKey, datasetId: flight.datasetId, group: item.string("group") };
  if (!item.boolean("flown")) {
    const carried = ["outcome", "flewTheSentence", "endS", "crossing", "refused", "evaluation", "alignment", "limits", "counts",
      "track", "judgedTrackDeg"].filter((key) => item.raw(key) !== null);
    if (carried.length > 0 || item.list("words").length !== 0) item.fail(`is not flown, yet carries ${carried.join(", ") || "words"}`);
    return { ...base, flown: false };
  }
  const crossing = item.nullableChild("crossing");
  const evaluation = item.child("evaluation");
  const alignment = item.child("alignment");
  const limits = item.child("limits");
  const counts = item.child("counts");
  const track = parseTrack(item.child("track"));
  requireSetDatum(item.child("track"), track, flight);
  requireSetStart(item.child("track"), track, flight, 0);
  const refused = item.nullableString("refused");
  const outcome = item.oneOf("outcome", TRAINING_EXECUTOR_OUTCOMES);
  // the judge's reading of the flown track is drawn exactly when the labeller's gate let it through and the dynamics
  // did not fail inside it; its step k is the exported track's point k
  const judgedTrackDeg = item.nullableNumbers("judgedTrackDeg");
  const drawn = refused === null && outcome !== "dynamics_failure";
  if ((judgedTrackDeg !== null) !== drawn) {
    item.fail(`judgedTrackDeg is ${judgedTrackDeg === null ? "absent" : "given"} for a flight ${refused !== null
      ? "whose flown track the gate refused" : outcome === "dynamics_failure" ? "whose dynamics failed" : "judged on its flown track"}`);
  }
  if (judgedTrackDeg !== null) {
    const misplaced = judgedTrackDeg.findIndex((_, step) => !(Math.abs(track.tS[step] - step * vocabulary.stepS) <= 1e-3));
    if (misplaced >= 0) {
      item.fail(`judgedTrackDeg's step ${misplaced} is not the flown track's point ${misplaced} (${track.tS[misplaced]} s)`);
    }
  }
  const words = parseExecutorWords(item, flight, vocabulary, judgedTrackDeg === null ? null : judgedTrackDeg.length, refused);
  // The judge's counts, from the words' own verdicts — one word counted twice (`countedTwice`).
  const twice = words.filter(countedTwice);
  const judged = words.filter((word) => word.status === "inside" || word.status === "outside").length + twice.length;
  const inside = words.filter((word) => word.status === "inside").length + twice.filter((word) => word.checks[0].ok).length;
  const wordsJudged = counts.count("wordsJudged");
  const wordsInside = counts.integer("wordsInside", 0, wordsJudged);
  if (wordsJudged !== judged || wordsInside !== inside) {
    counts.fail(`says ${wordsInside} of ${wordsJudged} words inside, but the words' own verdicts give ${inside} of ${judged}` +
      `${twice.length ? " (one counted twice)" : ""}`);
  }
  return {
    ...base,
    flown: true,
    outcome,
    flewTheSentence: item.boolean("flewTheSentence"),
    endS: item.number("endS"),
    crossing: crossing === null ? null : readCrossing(crossing),
    refused,
    evaluation: { replay: evaluation.string("replay"), observed: evaluation.string("observed") },
    alignment: {
      meanHorizontalDistanceM: alignment.number("meanHorizontalDistanceM"),
      meanVerticalDistanceM: alignment.number("meanVerticalDistanceM"),
      landingTimeMinusObservedS: alignment.nullableNumber("landingTimeMinusObservedS"),
    },
    limits: { cycles: limits.count("cycles"), bound: limits.record("bound", asNumber) },
    counts: { wordsJudged, wordsInside, headingWordsNotJudged: counts.count("headingWordsNotJudged") },
    track,
    judgedTrackDeg,
    words,
  };
}

function parseGateCell(value: unknown, where: string): TrainingGateCell {
  const cell = Reader.of(value, where);
  const clears = cell.nullableChild("clears");
  const notGated = cell.nullableString("notGated");
  if ((clears === null) === (notGated === null)) cell.fail("a cell is gated (clears) or says why not (notGated), one of the two");
  return {
    flights: cell.count("flights"),
    landed: cell.nullableShare("landed"),
    wordsJudged: cell.count("wordsJudged"),
    wordsInside: cell.nullableShare("wordsInside"),
    observedPasses: cell.count("observedPasses"),
    evaluationPaired: cell.nullableShare("evaluationPaired"),
    clears: clears === null ? null : { landed: clears.boolean("landed"), words: clears.boolean("words"), evaluation: clears.boolean("evaluation") },
    notGated,
    outcomes: cell.record("outcomes", asNumber),
  };
}

/** The formal replay's gate table: each group at the overlay's airport and at all airports, each with its "all" cell and
 *  a cell for each stratum it has flights of. */
function parseGate(overlay: Reader, airport: string): TrainingGateTable {
  const strataOf = (value: unknown, where: string): TrainingGateStrata => {
    const cells = recordOf(value, where, parseGateCell);
    const unknown = Object.keys(cells).filter((name) => name !== "all" && !(TRAINING_STRATA as readonly string[]).includes(name));
    if (unknown.length > 0 || cells.all === undefined) {
      Reader.of(value, where).fail(`holds ${Object.keys(cells).join(", ")}: expected "all" and any of ${TRAINING_STRATA.join(", ")}`);
    }
    return cells as TrainingGateStrata;
  };
  return overlay.record("gate", (group, where) => {
    const places = Reader.of(group, where);
    const names = Object.keys(recordOf(group, where, (value) => value));
    if (names.length !== 2 || !names.includes(airport) || !names.includes("all")) {
      places.fail(`holds ${names.join(", ")}: expected ${airport} and all`);
    }
    return { here: strataOf(places.raw(airport), places.at(airport)), all: strataOf(places.raw("all"), places.at("all")) };
  });
}

/** Parse an executor overlay against the manifest entry that listed it and the sample it is drawn over — all or
 *  nothing. */
export function parseTrainingExecutorOverlay(
  raw: unknown, entry: TrainingOverlayEntry, sample: TrainingSample,
): Parsed<TrainingExecutorOverlay> {
  return attempt(() => {
    const overlay = Reader.of(raw, "executor overlay");
    if (overlay.raw("schema") !== TRAINING_EXECUTOR_SCHEMA) {
      overlay.fail(`schema is ${JSON.stringify(overlay.raw("schema"))}, expected ${JSON.stringify(TRAINING_EXECUTOR_SCHEMA)}`);
    }
    const base = parseBase(overlay, entry, sample);
    const executor = overlay.child("executor");
    const replay = overlay.child("replay");
    return {
      overlayId: entry.id,
      airport: sample.airport,
      base,
      executor: {
        specSha256: executor.string("specSha256"), wordClock: executor.string("wordClock"), cycleS: executor.number("cycleS"),
        params: executor.children("params").map((param) => {
          const value = param.raw("value");
          if (typeof value !== "number" && typeof value !== "string") param.fail(`value is ${JSON.stringify(value)}`);
          return { name: param.string("name"), value: value as number | string };
        }),
      },
      replay: {
        split: replay.string("split"), writtenUtc: replay.string("writtenUtc"), gateShare: replay.share("gateShare"),
        drawn: { flights: replay.child("drawn").count("flights") },
      },
      gate: parseGate(overlay, sample.airport),
      flights: eachFlight(overlay, sample, (item, flight) => parseExecutorFlight(item, flight, sample.vocabulary)),
    };
  });
}

// ── the prior ────────────────────────────────────────────────────────────────

/** Probabilities rounded by the exporter (`prior_training_export.PROBABILITY_DIGITS`, 4): two of them multiplied and
 *  compared with a third agree to within this. */
const PROBABILITY_SLACK = 2e-4;

function parsePriorFlight(item: Reader, flight: TrainingFlight, sample: TrainingSample): TrainingPriorFlight {
  const rows = item.integer("rows", flight.rows, flight.rows);
  const firstPredictedRow = item.integer("firstPredictedRow", 0, rows - 1);
  const steps = rows - firstPredictedRow;
  const list = item.list("columns");
  if (list.length !== TRAINING_COLUMNS.length) item.fail(`holds ${list.length} columns, expected ${TRAINING_COLUMNS.length}`);
  const predicted: TrainingPriorFlight = {
    flightKey: flight.flightKey, datasetId: flight.datasetId, rows, firstPredictedRow,
    nllPerStep: item.number("nllPerStep"), columnNllPerStep: item.numbers("columnNllPerStep", TRAINING_COLUMNS.length),
    columns: TRAINING_COLUMNS.map((name, index) => {
      const column = Reader.of(list[index], item.at(`columns[${index}] (${name})`));
      const values = trainingClassCount(sample.vocabulary, sample.candidates, name);
      const k = column.integer("k", 1, values);
      const words = column.numbers("words", steps * k);
      const wrong = words.findIndex((value) => !Number.isInteger(value) || value < 0 || value >= values);
      if (wrong >= 0) column.fail(`words[${wrong}] is ${words[wrong]}, not one of the column's ${values} values`);
      const changeP = column.probabilities("changeP", steps);
      // the first predicted step says every column: "unchanged" is not a word there
      if (Math.abs(changeP[0] - 1) > PROBABILITY_SLACK) column.fail(`changeP at the first predicted step is ${changeP[0]}, not 1`);
      return { k, words, wordsP: column.probabilities("wordsP", steps * k), changeP, truthP: column.probabilities("truthP", steps) };
    }),
  };
  // The truth's probability is the prior's word probability times its change probability — the one rule that ties the
  // truth the exporter scored to the one this reader shows (`priorTruthAt`): checked wherever the truth is ranked.
  TRAINING_COLUMNS.forEach((name, index) => {
    const values = predicted.columns[index];
    for (let row = firstPredictedRow; row < rows; row += 1) {
      const at = row - firstPredictedRow;
      const truth = priorTruthAt(flight, predicted, name, row);
      const rank = values.words.slice(at * values.k, (at + 1) * values.k).indexOf(truth);
      const expected = truth === TRAINING_UNCHANGED ? 1 - values.changeP[at]
        : rank >= 0 ? values.changeP[at] * values.wordsP[at * values.k + rank] : null;
      if (expected !== null && Math.abs(values.truthP[at] - expected) > PROBABILITY_SLACK) {
        item.fail(`columns[${index}] (${name}).truthP at step ${row} is ${values.truthP[at]}, but the prior gives the truth ` +
          `there (${truth === TRAINING_UNCHANGED ? "unchanged" : `word ${truth}`}) ${expected.toFixed(4)}`);
      }
    }
  });
  return predicted;
}

function parseColumnScores<T>(reader: Reader, key: string, read: (value: unknown, where: string) => T, withAll: boolean) {
  const scores = reader.record(key, read);
  const expected = withAll ? [...TRAINING_COLUMNS, "all"] : [...TRAINING_COLUMNS];
  const missing = expected.filter((name) => !(name in scores));
  if (missing.length) reader.fail(`${key} has no ${missing.join(", ")}`);
  return scores;
}

function parseReadout(reader: Reader): TrainingPriorReadout {
  const model = reader.child("model");
  const baselines = reader.child("baselines");
  const perColumn = parseColumnScores(model, "perColumn", (value, where): TrainingPriorColumnReadout => {
    const column = Reader.of(value, where);
    const changeSteps = column.count("changeSteps");
    const changed = {
      changeProbabilityWhereChanged: column.nullableShare("changeProbabilityWhereChanged"),
      top1GivenChange: column.nullableShare("top1GivenChange"), top5GivenChange: column.nullableShare("top5GivenChange"),
    };
    // the change metrics are counted over the steps where the truth says a word: none, when there are none
    const absent = Object.entries(changed).filter(([, share]) => (share === null) !== (changeSteps === 0)).map(([name]) => name);
    if (absent.length > 0) column.fail(`${absent.join(", ")} ${changeSteps === 0 ? "given" : "absent"} with ${changeSteps} change steps`);
    return {
      nllPerStep: column.number("nllPerStep"), changeSteps, firstStepTop1: column.share("firstStepTop1"), ...changed,
      falseChangeShareWhereKept: column.share("falseChangeShareWhereKept"),
    };
  }, false) as Record<TrainingColumn, TrainingPriorColumnReadout>;
  const runway = reader.child("firstStepRunway");
  // the rules by name: the readout's own, no more and no fewer
  const rules = Object.keys(runway.record("rules", (value) => value)).sort();
  const expected = [...TRAINING_PRIOR_RULES].sort();
  if (rules.length !== expected.length || rules.some((rule, index) => rule !== expected[index])) {
    runway.fail(`rules are ${rules.join(", ")}, expected ${TRAINING_PRIOR_RULES.join(", ")}`);
  }
  const readRunway = (value: unknown, where: string): TrainingPriorRunwayReadout => {
    const part = Reader.of(value, where);
    return { top1: part.share("top1"), direction: part.share("direction"), sideGivenDirection: part.nullableShare("sideGivenDirection") };
  };
  return {
    split: reader.string("split"),
    steps: reader.count("steps", 1),
    bestEpoch: reader.count("bestEpoch", 1),
    model: { nllPerStep: model.number("nllPerStep"), perplexityPerStep: model.number("perplexityPerStep"), perColumn },
    baselines: {
      repeat: parseColumnScores(baselines, "repeat", asNumber, true) as Record<TrainingColumn | "all", number>,
      previousWord: parseColumnScores(baselines, "previousWord", asNumber, true) as Record<TrainingColumn | "all", number>,
    },
    firstStepRunway: {
      model: readRunway(runway.raw("model"), runway.at("model")),
      airportFrequency: readRunway(runway.raw("airportFrequency"), runway.at("airportFrequency")),
      rules: runway.record("rules", readRunway),
    },
  };
}

/** Parse a prior overlay against the manifest entry that listed it and the sample it is drawn over — all or nothing. */
export function parseTrainingPriorOverlay(raw: unknown, entry: TrainingOverlayEntry, sample: TrainingSample): Parsed<TrainingPriorOverlay> {
  return attempt(() => {
    const overlay = Reader.of(raw, "prior overlay");
    if (overlay.raw("schema") !== TRAINING_PRIOR_SCHEMA) {
      overlay.fail(`schema is ${JSON.stringify(overlay.raw("schema"))}, expected ${JSON.stringify(TRAINING_PRIOR_SCHEMA)}`);
    }
    overlay.sameNames("columns", TRAINING_COLUMNS);
    const base = parseBase(overlay, entry, sample);
    const prior = overlay.child("prior");
    const model = prior.child("model");
    return {
      overlayId: entry.id,
      airport: sample.airport,
      base,
      prior: {
        checkpointSha256: prior.string("checkpointSha256"), parameters: prior.count("parameters", 1),
        model: { dModel: model.count("dModel", 1), layers: model.count("layers", 1), heads: model.count("heads", 1) },
        method: prior.string("method"),
      },
      readout: parseReadout(overlay.child("readout")),
      flights: eachFlight(overlay, sample, (item, flight) => parsePriorFlight(item, flight, sample)),
    };
  });
}

// ── the prior's own sentences ────────────────────────────────────────────────

/** Tolerance on the exporter's rounded times (3 decimals). */
export const TIME_SLACK = 2e-3;

/** The flown track: point k at the first predicted row's time plus k steps — the views place a word at point
 *  ``row − firstRow`` — but the last, which may end part-way through its step. */
export function parseGeneratedTrack(reader: Reader, firstS: number, stepS: number): TrainingGeneratedTrack {
  const tS = reader.numbers("tS");
  if (tS.length < 1 || Math.abs(tS[0] - firstS) > TIME_SLACK) reader.fail(`does not start at the first predicted row (${firstS} s)`);
  if (tS.some((value, index) => index > 0 && value <= tS[index - 1])) reader.fail("tS does not run forward");
  const n = tS.length;
  const offStep = tS.findIndex((value, index) => index < n - 1 && Math.abs(value - (firstS + index * stepS)) > TIME_SLACK);
  if (offStep >= 0) reader.fail(`tS[${offStep}] is ${tS[offStep]} s, not the step at ${firstS + offStep * stepS} s`);
  if (n > 1 && tS[n - 1] - tS[n - 2] > stepS + TIME_SLACK) reader.fail(`its last point is more than a step after the one before`);
  return {
    tS, lon: reader.numbers("lon", n), lat: reader.numbers("lat", n), altitudeM: reader.numbers("altitudeM", n),
    altitudeHaeM: reader.numbers("altitudeHaeM", n), groundSpeedMps: reader.numbers("groundSpeedMps", n),
    attitude: readAttitude(reader.child("attitude"), n),
  };
}

/** What a model's sentence said, whatever ended it: its words — in (row, column) order, from the first predicted row,
 *  which says every column — and its bookkeeping bound to them (the runway pointed first and last and the changes between,
 *  the go-around words, cleared at the end), and the probability the prior put on what the masks removed. */
export function readSaidWords(item: Reader, index: number, firstRow: number, sample: TrainingSetHead) {
  item.integer("sample", index, index);
  const rows = item.integer("rows", firstRow + 1, Number.MAX_SAFE_INTEGER);
  const counts = TRAINING_COLUMNS.map((column) => trainingClassCount(sample.vocabulary, sample.candidates, column));
  const events: TrainingSentenceEvent[] = item.children("events").map((event) => {
    const column = event.integer("column", 0, TRAINING_COLUMNS.length - 1);
    return { row: event.integer("row", firstRow, rows - 1), column, value: event.integer("value", 0, counts[column] - 1) };
  });
  events.forEach((event, at) => {
    const previous = events[at - 1];
    if (previous && (previous.row > event.row || (previous.row === event.row && previous.column >= event.column))) {
      item.fail(`events are not in (row, column) order at ${at}: one cell, one word`);
    }
  });
  const opening = events.filter((event) => event.row === firstRow).length;
  if (opening !== TRAINING_COLUMNS.length) item.fail(`the first predicted row (${firstRow}) says ${opening} columns, not all six`);
  // its bookkeeping is its words': the runways pointed first and last and the changes between them, the go-around words,
  // and whether the last approach word clears the flight (`prior_free_generation.flight_rows` counts them so)
  const said = (column: TrainingColumn) => events.filter((event) => event.column === TRAINING_COLUMN_INDEX[column]).map((event) => event.value);
  const runways = said("runway");
  const approach = said("approach");
  const runwayCount = counts[TRAINING_COLUMN_INDEX.runway];
  const firstRunway = item.integer("firstRunway", 0, runwayCount - 1);
  const lastRunway = item.integer("lastRunway", 0, runwayCount - 1);
  if (firstRunway !== runways[0] || lastRunway !== runways[runways.length - 1]) {
    item.fail(`points at runways ${firstRunway} → ${lastRunway}, but its words say ${runways[0]} → ${runways[runways.length - 1]}`);
  }
  const runwayChanges = item.count("runwayChanges");
  const goArounds = item.count("goArounds");
  const clearedAtEnd = item.boolean("clearedAtEnd");
  const expected = {
    runwayChanges: runways.filter((value, at) => at > 0 && value !== runways[at - 1]).length,
    goArounds: approach.filter((value) => value === trainingGoAroundValue(sample.vocabulary)).length,
    clearedAtEnd: approach[approach.length - 1] === trainingClearedValue(sample.vocabulary),
  };
  const found = { runwayChanges, goArounds, clearedAtEnd };
  const differ = (Object.keys(expected) as Array<keyof typeof expected>).filter((key) => found[key] !== expected[key]);
  if (differ.length > 0) item.fail(differ.map((key) => `${key} is ${found[key]}, its words give ${expected[key]}`).join("; "));
  const forbiddenMass = item.record("forbiddenMass", (value, where) => {
    const share = asNumber(value, where);
    if (share < 0 || share > 1) throw new Refusal(`${where} is ${share}, not a probability`);
    return share;
  });
  const unknown = Object.keys(forbiddenMass).filter((name) => !(TRAINING_COLUMNS as readonly string[]).includes(name));
  if (unknown.length > 0) item.fail(`forbiddenMass names ${unknown.join(", ")}: not columns`);
  return { sample: index, rows, firstRow, events, firstRunway, lastRunway, runwayChanges, goArounds, clearedAtEnd, forbiddenMass };
}

/** One sample over a set's flight: its words (`readSaidWords`) and its end, its crossing and its track bound to each other:
 *  a crossing exactly for an outcome read at one, the track to the end (a dynamics failure's to the row before the failed
 *  state). ``stoppable``: its model spoke under the procedure's altitudes, whose sentences the glidepath lower edge stops. */
function parseGeneratedSentence(
  item: Reader, index: number, firstRow: number, stepS: number, cycleS: number, sample: TrainingSample, flight: TrainingFlight,
  stoppable: boolean,
): TrainingGeneratedSentence {
  const said = readSaidWords(item, index, firstRow, sample);
  const { rows } = said;
  const outcome = item.oneOf("outcome", TRAINING_FREE_OUTCOMES);
  if (outcome === TRAINING_BELOW_GLIDEPATH && !stoppable) item.fail(`is ${outcome}, but its model spoke without the procedure's altitudes`);
  const endS = item.number("endS");
  const crossingReader = item.nullableChild("crossing");
  const crossing = crossingReader === null ? null : readCrossing(crossingReader);
  if ((crossing !== null) !== TRAINING_CROSSING_OUTCOMES.includes(outcome)) {
    item.fail(`is ${outcome} ${crossing === null ? "with no crossing" : "and carries a crossing"}`);
  }
  const firstS = firstRow * stepS;
  if (endS < firstS || endS > rows * stepS + TIME_SLACK) item.fail(`ends at ${endS} s, outside its rows ${firstS}…${rows * stepS} s`);
  // stopped at its last step said: its end is that step's end
  if (outcome === TRAINING_BELOW_GLIDEPATH && Math.abs(endS - rows * stepS) > TIME_SLACK) {
    item.fail(`is ${outcome} at ${endS} s, not at the end of its last step, ${rows * stepS} s`);
  }
  const track = parseGeneratedTrack(item.child("track"), firstS, stepS);
  requireSetDatum(item.child("track"), track, flight);
  const trackEnd = track.tS[track.tS.length - 1];
  // the track stops at the outcome's state — a dynamics failure's at the state before it
  const expectedEnd = outcome === "dynamics_failure" ? endS - cycleS : endS;
  if (Math.abs(trackEnd - expectedEnd) > TIME_SLACK) item.fail(`its track ends at ${trackEnd} s, not at ${expectedEnd} s (${outcome})`);
  return { ...said, outcome, endS, crossing, track };
}

/** A readout's cells: "all", and each approach kind — null where the draw held no flight of it. */
function parseGenerationCells(reader: Reader): TrainingGenerationReadoutCells {
  const read = (cell: Reader) => ({ flights: cell.count("flights", 1), landed: cell.share("landed") });
  const strata = Object.fromEntries(TRAINING_STRATA.map((stratum) => {
    const cell = reader.nullableChild(stratum);
    return [stratum, cell === null ? null : read(cell)];
  }));
  return { all: read(reader.child("all")), ...strata } as TrainingGenerationReadoutCells;
}

/** The cells at the overlay's airport and over every airport. */
function parsePlaces(reader: Reader) {
  return { here: parseGenerationCells(reader.child("here")), all: parseGenerationCells(reader.child("all")) };
}

/** A round is post-trained and names what it started from; base is neither — each name's round and start model agree. */
function parseGenerationModel(model: Reader): TrainingGenerationModel {
  const name = model.oneOf("name", TRAINING_MODEL_NAMES);
  const round = model.nullableCount("round", 1);
  const tuning = model.nullableChild("fineTuning");
  if ((name === "base") !== (round === null) || (round === null) !== (tuning === null)) {
    model.fail(`is ${name} with round ${round} and ${tuning === null ? "no" : "a"} fineTuning: base alone has neither`);
  }
  const fineTuning = tuning === null ? null : {
    schema: tuning.string("schema"), from: tuning.string("from"), fromName: tuning.oneOf("fromName", TRAINING_MODEL_NAMES),
    fromRound: tuning.nullableCount("fromRound", 1),
  };
  if (fineTuning !== null && (fineTuning.fromName === "base") !== (fineTuning.fromRound === null)) {
    tuning!.fail(`starts from ${fineTuning.fromName} round ${fineTuning.fromRound}: base alone has no round`);
  }
  const run = model.string("run");
  if (!run.includes(PRIOR_RUNS)) model.fail(`run ${JSON.stringify(run)} is not a prior run (under ${PRIOR_RUNS})`);
  return {
    name, round, run, checkpointSha256: model.string("checkpointSha256"), variant: model.string("variant"), fineTuning,
  };
}

/** A flight's augmented start, bound to the draw's rule and to its samples: drawn (1 … tries draws) exactly when it is on its
 *  own dynamics, moved within the limits exactly when it is flown, its moved observed rows a step apart from 0 up to the
 *  first predicted row, where every sample's track starts. */
function parseAugmentedStart(
  item: Reader, augment: NonNullable<TrainingGenerationOverlay["generation"]["augment"]>, flown: boolean, firstRow: number,
  stepS: number, samples: TrainingGeneratedSentence[], flight: TrainingFlight,
): TrainingAugmentedStart {
  const draws = item.nullableInteger("augmentDraws", 1, augment.tries);
  const moved = item.nullableChild("augmentation");
  if (moved !== null && draws === null) item.fail("is moved without a draw");
  if ((moved !== null) !== flown) item.fail(`is ${flown ? "" : "not "}flown ${moved === null ? "without" : "with"} a move`);
  const augmentation = moved === null ? null : {
    rotationDeg: moved.number("rotationDeg"), altitudeM: moved.number("altitudeM"), speedScale: moved.number("speedScale"),
  };
  if (augmentation !== null) {
    const { limits } = augment;
    if (Math.abs(augmentation.rotationDeg) > limits.rotationDeg || Math.abs(augmentation.altitudeM) > limits.altitudeM
      || Math.abs(augmentation.speedScale - 1) > limits.speedFraction + 1e-9) {
      moved!.fail(`moves ${augmentationText(augmentation)}, outside ±${limits.rotationDeg}°, ±${limits.altitudeM} m, ` +
        `×1 ± ${limits.speedFraction}`);
    }
  }
  const rows = item.nullableChild("observed");
  if ((rows !== null) !== (augmentation !== null)) item.fail(`has ${rows === null ? "no " : ""}moved observed rows ${augmentation === null ? "without" : "with"} a move`);
  let observed: TrainingObservedRows | null = null;
  if (rows !== null) {
    const tS = rows.numbers("tS", firstRow);
    const offStep = tS.findIndex((value, index) => Math.abs(value - index * stepS) > TIME_SLACK);
    if (offStep >= 0) rows.fail(`tS[${offStep}] is ${tS[offStep]} s, not the step at ${offStep * stepS} s`);
    observed = { tS, lon: rows.numbers("lon", firstRow), lat: rows.numbers("lat", firstRow),
      altitudeM: rows.numbers("altitudeM", firstRow), altitudeHaeM: rows.numbers("altitudeHaeM", firstRow) };
    requireSetDatum(rows, observed, flight);
    // every sample flies from the one moved start
    const starts = new Set(samples.map((one) => `${one.track.lat[0]},${one.track.lon[0]},${one.track.altitudeM[0]}`));
    if (starts.size > 1) item.fail(`its samples start from ${starts.size} places, not one moved start`);
  }
  return { draws, augmentation, observed };
}

/** A model overlay's head (`TrainingGenerationHead`), bound to the set it is drawn over: its base, its model, and how its
 *  sentences were drawn — ``augmented``: from augmented starts, with the draw's rule. */
export function readGenerationHead(
  overlay: Reader, entry: TrainingOverlayEntry, sample: TrainingSetHead, augmented: boolean,
): TrainingGenerationHead {
  overlay.sameNames("columns", TRAINING_COLUMNS);
  const base = parseBase(overlay, entry, sample);
  const model = parseGenerationModel(overlay.child("model"));
  const generation = overlay.child("generation");
  const stepS = generation.number("stepS");
  if (stepS !== sample.vocabulary.stepS) generation.fail(`stepS is ${stepS}, the set's step ${sample.vocabulary.stepS}`);
  const procedureMasks = generation.children("procedureMasks").map((item) => {
    const dataSha256 = item.string("dataSha256");
    if (!/^[0-9a-f]{64}$/.test(dataSha256)) item.fail(`dataSha256 ${JSON.stringify(dataSha256)} is not a sha256`);
    return { name: item.oneOf("name", TRAINING_PROCEDURE_MASK_SETS) as string, dataSha256 };
  });
  if (new Set(procedureMasks.map((item) => item.name)).size !== procedureMasks.length) {
    generation.fail(`names a set of the procedure's masks twice: ${procedureMasks.map((item) => item.name).join(", ")}`);
  }
  const executor = generation.child("executor");
  const augmentReader = augmented ? generation.child("augment") : null;
  return {
    overlayId: entry.id,
    airport: sample.airport,
    base,
    model,
    generation: {
      samples: generation.count("samples", 1), temperature: generation.number("temperature"), seed: generation.number("seed"),
      firstPredictedRow: generation.count("firstPredictedRow"), procedureMasks,
      executor: { specSha256: executor.string("specSha256"), wordClock: executor.string("wordClock"),
        cycleS: executor.number("cycleS"), timeoutFactor: executor.number("timeoutFactor") },
      augment: augmentReader === null ? null : {
        seed: augmentReader.integer("seed", 0, Number.MAX_SAFE_INTEGER), tries: augmentReader.count("tries", 1),
        limits: { rotationDeg: augmentReader.child("limits").number("rotationDeg"),
          altitudeM: augmentReader.child("limits").number("altitudeM"),
          speedFraction: augmentReader.child("limits").number("speedFraction") },
        timeoutFactor: augmentReader.number("timeoutFactor"),
      },
    },
  };
}

/** Whether a model spoke under the procedure's altitudes, whose sentences the glidepath lower edge stops. */
export function speaksUnderAltitudes(head: TrainingGenerationHead): boolean {
  return head.generation.procedureMasks.some((item) => item.name === TRAINING_PROCEDURE_ALTITUDES);
}

/** Parse a generation overlay against the manifest entry that listed it and the sample it is drawn over — all or
 *  nothing. */
export function parseTrainingGenerationOverlay(
  raw: unknown, entry: TrainingOverlayEntry, sample: TrainingSample,
): Parsed<TrainingGenerationOverlay> {
  return attempt(() => {
    const overlay = Reader.of(raw, "generation overlay");
    const augmented = entry.kind === "prior-generation-augmented";
    const schema = augmented ? TRAINING_AUGMENTED_GENERATION_SCHEMA : TRAINING_GENERATION_SCHEMA;
    if (overlay.raw("schema") !== schema) {
      overlay.fail(`schema is ${JSON.stringify(overlay.raw("schema"))}, expected ${JSON.stringify(schema)}`);
    }
    const head = readGenerationHead(overlay, entry, sample, augmented);
    const { samples, firstPredictedRow, augment } = head.generation;
    const { stepS } = sample.vocabulary;
    const { cycleS } = head.generation.executor;
    // from augmented starts: no readout (stage 2's augmented readouts drew their own moves), the draw's rule instead
    if (augmented && overlay.raw("readout") !== undefined) overlay.fail("carries a readout: one from augmented starts has none");
    const readout = augmented ? null : overlay.nullableChild("readout");
    return {
      ...head,
      readout: readout === null ? null : {
        split: readout.string("split"), writtenUtc: readout.string("writtenUtc"),
        drawn: { flights: readout.child("drawn").count("flights", 1), perAirport: readout.child("drawn").count("perAirport", 0) },
        prior: parsePlaces(readout.child("prior")), labelled: parsePlaces(readout.child("labelled")),
      },
      flights: eachFlight(overlay, sample, (item, flight) => {
        const flown = item.boolean("flown");
        const list = item.children("samples");
        if (flown ? list.length !== samples : list.length !== 0) {
          item.fail(`is ${flown ? "" : "not "}flown and holds ${list.length} samples${flown ? `, not ${samples}` : ""}`);
        }
        if (firstPredictedRow >= flight.rows) item.fail(`has ${flight.rows} rows: the prior speaks from row ${firstPredictedRow}`);
        const samplesRead = list.map((one, index) => parseGeneratedSentence(one, index, firstPredictedRow, stepS, cycleS, sample,
          flight, speaksUnderAltitudes(head)));
        // flown from the set's own start: each sample's track begins at the observed first predicted row
        if (augment === null) {
          samplesRead.forEach((one, index) => requireSetStart(list[index].child("track"), one.track, flight, firstPredictedRow));
        }
        return {
          flightKey: flight.flightKey, datasetId: flight.datasetId, group: item.string("group"), flown, samples: samplesRead,
          start: augment === null ? null : parseAugmentedStart(item, augment, flown, firstPredictedRow, stepS, samplesRead, flight),
        };
      }),
    };
  });
}

// ── where the files live ─────────────────────────────────────────────────────

export function trainingOverlaysPath(airportCode: string): string {
  return `${trainingDirectory(airportCode)}/overlays.json`;
}

export async function fetchTrainingOverlays(airportCode: string): Promise<Parsed<TrainingOverlays>> {
  return parseTrainingOverlays(await fetchJson<unknown>(trainingOverlaysPath(airportCode)));
}

export async function fetchTrainingExecutorOverlay(
  airportCode: string, entry: TrainingOverlayEntry, sample: TrainingSample,
): Promise<Parsed<TrainingExecutorOverlay>> {
  return parseTrainingExecutorOverlay(await fetchJson<unknown>(trainingFilePath(airportCode, entry.file)), entry, sample);
}

export async function fetchTrainingPriorOverlay(
  airportCode: string, entry: TrainingOverlayEntry, sample: TrainingSample,
): Promise<Parsed<TrainingPriorOverlay>> {
  return parseTrainingPriorOverlay(await fetchJson<unknown>(trainingFilePath(airportCode, entry.file)), entry, sample);
}

export async function fetchTrainingGenerationOverlay(
  airportCode: string, entry: TrainingOverlayEntry, sample: TrainingSample,
): Promise<Parsed<TrainingGenerationOverlay>> {
  return parseTrainingGenerationOverlay(await fetchJson<unknown>(trainingFilePath(airportCode, entry.file)), entry, sample);
}
