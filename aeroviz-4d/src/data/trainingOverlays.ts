/**
 * trainingOverlays.ts
 * -------------------
 * What another model makes of a Training set's own flights, drawn over the set: the EXECUTOR's replay (the truth
 * sentence flown from row 0 — its track, its outcome and every word's verdict) and the PRIOR's predictions (at every
 * step of the truth sentence, what it gives each column). Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`
 * §2.6 and §4.6; the writers: `experiments/executor_training_export.py`, `experiments/prior_training_export.py`.
 *
 * AN OVERLAY IS A FILE BESIDE ITS SET, NEVER INSIDE IT. `training/overlays.json` lists them, each naming the set it
 * is drawn over (`base`) and that set's sample by its sha256; the payload repeats the set id, the sample's time of
 * writing and the spec, and holds one entry per flight of the set, in the set's order. The reader binds the two by
 * those fields and by the flights themselves — the same keys, and for the executor the same words, for the prior the
 * same number of steps — and refuses the payload whole if any differs: a verdict drawn on the wrong word is worse
 * than none.
 *
 * NOTHING HERE IS COMPUTED. Every verdict is the executor's judge's, re-flown and checked against its formal replay
 * in Python — a heading word's with its band and a verdict per row on the flown track the judge read — and every
 * probability is the prior's own. The reader checks bookkeeping — lengths, ranges, the binding — and hands numbers to
 * the views.
 *
 * NO COMPATIBILITY: the three schemas are pinned below and anything else is refused by name. A payload is bound to the
 * manifest entry that listed it (its id and set) and to the sample on screen (its set, time of writing, spec and
 * airport), or refused whole.
 *
 * SI units only: metres, m/s, degrees, seconds.
 */

import { fetchJson } from "../utils/fetchJson";
import { asNumber, attempt, parseManifest, Reader, recordOf, type Parsed } from "./trainingReader";
import {
  readHeadingBand,
  TRAINING_COLUMN_INDEX,
  TRAINING_COLUMNS,
  TRAINING_SPEC_SHA256,
  TRAINING_STRATA,
  TRAINING_UNCHANGED,
  trainingClassCount,
  trainingDirectory,
  trainingFilePath,
  type TrainingColumn,
  type TrainingFlight,
  type TrainingHeadingBand,
  type TrainingSample,
  type TrainingSelection,
  type TrainingVocabulary,
} from "./trainingSample";

/** MIRROR of the exporter's `OVERLAYS_SCHEMA` (`instruction_training_export.py`): the manifest of overlays. */
export const TRAINING_OVERLAYS_SCHEMA = "aeroviz-training-overlays-v1";
/** MIRROR of `OVERLAY_KINDS`: what an overlay can be. */
export const TRAINING_OVERLAY_KINDS = ["executor-replay", "prior-prediction"] as const;
export type TrainingOverlayKind = (typeof TRAINING_OVERLAY_KINDS)[number];
/** MIRROR of `executor_training_export.SCHEMA`: v2 (instruction-v3) gives each heading word judged its band and a
 *  verdict per row, and each flight judged its flown track as the judge read it; v1 (turns and holds) is refused. */
export const TRAINING_EXECUTOR_SCHEMA = "aeroviz-training-executor-v2";
/** MIRROR of `executor_training_export.STATUSES`: one per word. */
export const TRAINING_EXECUTOR_STATUSES = [
  "inside", "outside", "not judged", "not reached", "superseded", "no check",
] as const;
export type TrainingExecutorStatus = (typeof TRAINING_EXECUTOR_STATUSES)[number];
/** MIRROR of `ts_transformer.autopilot.judge.OUTCOMES`: how the executor's flight ended — the replay's, and the live
 *  executor's when it flew on to the landing. */
export const TRAINING_EXECUTOR_OUTCOMES = [
  "landed", "crossed_without_capture", "crossed_off_runway", "ground_contact", "timeout", "dynamics_failure",
] as const;
export type TrainingExecutorOutcome = (typeof TRAINING_EXECUTOR_OUTCOMES)[number];
/** MIRROR of `prior_training_export.SCHEMA`. */
export const TRAINING_PRIOR_SCHEMA = "aeroviz-training-prior-v3";

// ── shapes ───────────────────────────────────────────────────────────────────

export interface TrainingOverlayEntry {
  id: string;
  kind: TrainingOverlayKind;
  /** The set it is drawn over. */
  base: string;
  /** The sha256 of that set's sample file when the overlay was written. */
  baseSampleSha256: string;
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

/** What an overlay says about the set it is drawn over. */
export interface TrainingOverlayBase {
  setId: string;
  sampleWrittenUtc: string;
  sampleSha256: string;
  specSha256: string;
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
  /** The horizontal distance the executor flew. */
  distanceM: number[];
}

/** At the threshold: metres right of the centreline and above the threshold, and when. */
export interface TrainingCrossing {
  crossM: number;
  heightM: number;
  atS: number;
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
  /** The prior's first-step runway beside each airport's own runway frequency and the causal rules (B0, B1, B3). */
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

/** What the views draw for the selected flight: its overlay's own data, and the flight's entry. */
export interface TrainingExecutorView {
  overlay: TrainingExecutorOverlay;
  flight: TrainingExecutorFlight;
}

export interface TrainingPriorView {
  overlay: TrainingPriorOverlay;
  flight: TrainingPriorFlight;
}

// ── reading one step ─────────────────────────────────────────────────────────

/** An overlay's view if it is of the flight on screen — drawn over its set, at its airport — else null: in the render
 *  after a switch, the view still published is the last flight's (a flight key alone repeats across sets). */
export function overlayOnScreen<V extends TrainingExecutorView | TrainingPriorView>(
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
    baseSampleSha256: entry.string("baseSampleSha256"),
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
function parseBase(reader: Reader, entry: TrainingOverlayEntry, sample: TrainingSample): TrainingOverlayBase {
  const overlayId = reader.string("overlayId");
  if (overlayId !== entry.id) reader.fail(`overlayId is ${overlayId}, but the manifest lists it as ${entry.id}`);
  const airport = reader.string("airport");
  if (airport !== sample.airport) reader.fail(`is drawn at ${airport}, but the loaded sample is ${sample.airport}'s`);
  const base = reader.child("base");
  const found = {
    setId: base.string("setId"),
    sampleWrittenUtc: base.string("sampleWrittenUtc"),
    sampleSha256: base.string("sampleSha256"),
    specSha256: base.string("specSha256"),
  };
  if (found.specSha256 !== TRAINING_SPEC_SHA256) base.fail(`specSha256 is ${found.specSha256.slice(0, 12)}, not ${TRAINING_SPEC_SHA256.slice(0, 12)}`);
  if (found.setId !== entry.base || found.sampleSha256 !== entry.baseSampleSha256) {
    base.fail(`is set ${found.setId} (sample ${found.sampleSha256.slice(0, 12)}), but the manifest lists it over set ` +
      `${entry.base} (sample ${entry.baseSampleSha256.slice(0, 12)})`);
  }
  if (found.setId !== sample.setId || found.sampleWrittenUtc !== sample.writtenUtc) {
    base.fail(
      `drawn over set ${found.setId} as written ${found.sampleWrittenUtc}, but the loaded sample is ${sample.setId} as ` +
      `written ${sample.writtenUtc}: the set was re-exported after the overlay`,
    );
  }
  return found;
}

/** The payload's flights, one per flight of the sample, in its order. */
function eachFlight<T>(reader: Reader, sample: TrainingSample, read: (item: Reader, flight: TrainingFlight) => T): T[] {
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
  return { crossM: reader.number("crossM"), heightM: reader.number("heightM"), atS: reader.number("atS") };
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
    distanceM: reader.numbers("distanceM", n),
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
  // The judge's counts, from the words' own verdicts. It counts the heading word left to intercept the final on its own
  // once more when its band also had rows judged — the one word with two checks, its band's and that one — and inside
  // once when the band's check passed.
  const twice = words.filter((word) => word.column === TRAINING_COLUMN_INDEX.heading && word.checks.length === 2);
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
