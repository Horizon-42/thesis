/**
 * trainingTraffic.ts
 * ------------------
 * MULTI-AIRCRAFT WINDOWS (the Training module §2.9, §4.11): a window set (`traffic-windows`, `traffic.json`, written by ts
 * `window_training_export`) and a model's sentences in its windows (`window-generation` overlays).
 *
 * A window is 20 minutes of one airport: the arrivals the model commands in it (each a set flight, with its sentence and
 * envelopes as a read-back set's), every other aircraft there replayed as recorded (with a sentence, or a background
 * arrival), and the window as recorded — the commanded aircraft along their records, their losses of separation and
 * their landings. A model's overlay adds, per window and sample (a sample is the whole window), each commanded aircraft's
 * sentence and flown track, the losses under VISUAL (the loop's reading: they end aircraft) and IFR (judged afterwards),
 * and the landings.
 *
 * THE FRONTEND JUDGES NOTHING: losses, ends and landings are the exporter's judge's (`traffic_loop`); this reader checks
 * their bookkeeping — who they name is in the window, an ended aircraft's sentence says so at the same time, the landings
 * are its landed aircraft in order.
 *
 * THE SINGLE-FLIGHT VIEWS ARE REUSED AS THEY ARE: an aircraft on screen is a set flight (`trainingWindowSelection`, on the
 * window's clock), and a model's sentences for it are a `TrainingGenerationView` (`windowGenerationView`) — the sentence
 * bar, the read-back window and the flight's 3D layers draw it as any model sentence. Only the scene — every aircraft at
 * one time — is drawn here (`sceneTrackAt`, `episodesAt`).
 */

import { attempt, Reader, type Parsed } from "./trainingReader";
import {
  parseGeneratedTrack,
  readCrossing,
  readGenerationHead,
  readSaidWords,
  requireSetDatum,
  requireSetStart,
  speaksUnderAltitudes,
  TIME_SLACK,
  TRAINING_BELOW_GLIDEPATH,
  TRAINING_CROSSING_OUTCOMES,
  TRAINING_FREE_OUTCOMES,
  TRAINING_LOST_SEPARATION,
  type TrainingFreeOutcome,
  type TrainingGeneratedSentence,
  type TrainingGenerationHead,
  type TrainingGenerationView,
  type TrainingOverlayEntry,
  type TrainingSource,
} from "./trainingOverlays";
import {
  readSetHead,
  rowAtTime,
  trainingFilePath,
  type TrainingFlight,
  type TrainingSelection,
  type TrainingSetHead,
  type TrainingWindowCohort,
} from "./trainingSample";
import { fetchJson } from "../utils/fetchJson";

/** MIRROR of `training_files.TRAFFIC_SCHEMA`: a window set. */
export const TRAINING_TRAFFIC_SCHEMA = "aeroviz-training-traffic-v1";
/** MIRROR of `window_training_export.SCHEMA`: a model's sentences in a window set's windows. */
export const TRAINING_WINDOW_GENERATION_SCHEMA = "aeroviz-training-window-generation-v1";
/** The two readings of the separation judge a window carries (`inference/separation.py`): VISUAL — the loop's, whose
 *  losses end aircraft — and IFR, the same paths judged afterwards, beside it. */
export const TRAINING_SEPARATION_READINGS = ["visual", "ifr"] as const;
export type TrainingSeparationReading = (typeof TRAINING_SEPARATION_READINGS)[number];
/** MIRROR of `window_training_export.window_payload`'s roles: an aircraft of the window the model does not command is
 *  replayed as recorded — an arrival with a sentence, or a background one without. */
export const TRAINING_OTHER_ROLES = ["replayed", "background"] as const;
export type TrainingOtherRole = (typeof TRAINING_OTHER_ROLES)[number];
/** What an aircraft of a window is to the view, which says how it is drawn: the one ON SCREEN (the sentence bar reads it),
 *  one the model COMMANDS, or one replayed as recorded. The commanded are a SET — every aircraft spoken to at once, as a
 *  live run of several will be — and the one on screen is one of them: nothing here assumes a single commanded aircraft. */
export const TRAINING_AIRCRAFT_ROLES = ["onScreen", "commanded", ...TRAINING_OTHER_ROLES] as const;
export type TrainingAircraftRole = (typeof TRAINING_AIRCRAFT_ROLES)[number];
/** How a window sentence may end: a free sentence's outcomes, or ended by the judge. */
export const TRAINING_WINDOW_OUTCOMES = [...TRAINING_FREE_OUTCOMES, TRAINING_LOST_SEPARATION] as const;

// ── shapes ───────────────────────────────────────────────────────────────────

/** An aircraft's positions on the window's clock: seconds from its opening. */
export interface TrainingSceneTrack {
  tS: number[];
  lon: number[];
  lat: number[];
  altitudeM: number[];
  altitudeHaeM: number[];
}

/** A pair under its minimum on consecutive steps of the window, as the judge recorded it (`traffic_loop.Judging`): their
 *  runways' relation, the steps (from, to — scene time), the kinds of minimum it broke, the closest it came against the
 *  minimum then, who answers for it and whom it ended. */
export interface TrainingLossEpisode {
  pair: [string, string];
  relation: string;
  fromS: number;
  toS: number;
  steps: number;
  kinds: string[];
  closestM: number;
  requiredM: number;
  wakeKnown: boolean;
  responsible: string[];
  ended: string[];
}

/** A follower short of its wake minimum when its leader crossed the threshold. */
export interface TrainingThresholdLoss {
  atS: number;
  leader: string;
  follower: string;
  gapM: number;
  requiredM: number;
  relation: string;
}

/** An aircraft the judge ended, when and why. */
export interface TrainingJudgedEnd {
  datasetId: string;
  atS: number;
  kind: string;
  relation: string;
  with: string;
}

/** A window's losses under one reading. */
export interface TrainingLosses {
  episodes: TrainingLossEpisode[];
  atThreshold: TrainingThresholdLoss[];
  ended: TrainingJudgedEnd[];
}

/** A commanded aircraft's landing, scene time. */
export interface TrainingWindowLanding {
  datasetId: string;
  atS: number;
}

export interface TrainingWindowCommanded {
  /** Its flight in the set (`TrainingSetHead.flights`). */
  flight: TrainingFlight;
  /** Its row 0 on the window's steps (its rows hang on the scene's even-second steps: its recorded rows are up to 1 s off). */
  rowZeroS: number;
  limitS: number;
  /** Its recorded rows, the window's clock. */
  recorded: TrainingSceneTrack;
}

export interface TrainingWindowOther {
  datasetId: string;
  callsign: string;
  /** Its wake category (null: unknown). */
  category: string | null;
  role: TrainingOtherRole;
  track: TrainingSceneTrack;
}

export interface TrainingWindow {
  /** Its place in the set: its key (`trainingWindowKey`) and the overlays' windows are by it. */
  index: number;
  opensUtc: string;
  commanded: TrainingWindowCommanded[];
  others: TrainingWindowOther[];
  /** The window as recorded: the commanded aircraft along their records. */
  recorded: { losses: Record<TrainingSeparationReading, TrainingLosses>; landings: TrainingWindowLanding[] };
}

export interface TrainingTrafficSet extends TrainingSetHead {
  cohort: TrainingWindowCohort;
  windows: TrainingWindow[];
}

/** One aircraft's sentence in one sample of a window: a model sentence (`TrainingGeneratedSentence`, its words, track and
 *  crossing on its own clock), and how the WINDOW ended it — ``outcome`` `lost_separation` when the judge did (``end``: why,
 *  with whom), its own end otherwise; it flies on silent to its own end (``own``, at ``ownEndS``, where its track ends). */
export interface TrainingWindowSentence extends TrainingGeneratedSentence {
  own: TrainingFreeOutcome;
  ownEndS: number;
  end: { kind: string; relation: string; with: string } | null;
}

/** Whether a model's sentence is one in a window (it carries its own end beside the window's). */
export function isWindowSentence(sentence: TrainingGeneratedSentence): sentence is TrainingWindowSentence {
  return "own" in sentence;
}

export interface TrainingWindowSample {
  sample: number;
  /** In the window's commanded order. */
  aircraft: TrainingWindowSentence[];
  losses: Record<TrainingSeparationReading, TrainingLosses>;
  landings: TrainingWindowLanding[];
}

/** The model's formal window readout, per source: the aircraft counted, landed and lost separation (VISUAL, IFR). */
export interface TrainingWindowReadoutCell {
  aircraft: number;
  landed: number;
  lostSeparation: number;
  lostSeparationIfr: number;
}

export interface TrainingWindowGenerationOverlay extends TrainingGenerationHead {
  readout: {
    directory: string;
    writtenUtc: string;
    /** What its cells count: every sample of its own draw's windows (not this overlay's few). */
    windowsPerAirport: number;
    samples: number;
    scene: { here: TrainingWindowReadoutCell; all: TrainingWindowReadoutCell };
    recorded: { here: TrainingWindowReadoutCell; all: TrainingWindowReadoutCell };
  } | null;
  /** One per set window, in its order. */
  windows: Array<{ samples: TrainingWindowSample[] }>;
}

/** The window on screen, as the panel publishes it (`AppContext.trainingWindow`): its set, the window and every model's
 *  sentences in the set's windows. */
export interface TrainingWindowView {
  set: TrainingTrafficSet;
  window: TrainingWindow;
  overlays: TrainingWindowGenerationOverlay[];
  /** Puts one of the window's commanded aircraft on screen (the panel's window session owns which). */
  focus: (datasetId: string) => void;
}

// ── reading ──────────────────────────────────────────────────────────────────

function readSceneTrack(reader: Reader): TrainingSceneTrack {
  const tS = reader.numbers("tS");
  if (tS.length < 1) reader.fail("holds no point");
  if (tS.some((value, at) => at > 0 && value <= tS[at - 1])) reader.fail("tS does not run forward");
  const n = tS.length;
  return { tS, lon: reader.numbers("lon", n), lat: reader.numbers("lat", n), altitudeM: reader.numbers("altitudeM", n),
    altitudeHaeM: reader.numbers("altitudeHaeM", n) };
}

/** A window's losses under one reading, bound to who is in it: every aircraft they name is the window's, an episode runs
 *  forward and ends only aircraft of its pair, and only commanded aircraft are ended (a replayed one never is). */
function readLosses(reader: Reader, present: Set<string>, commanded: Set<string>): TrainingLosses {
  const known = (item: Reader, key: string, id: string) => {
    if (!present.has(id)) item.fail(`${key} ${id} is not in the window`);
    return id;
  };
  const episodes = reader.children("episodes").map((item): TrainingLossEpisode => {
    const pair = item.strings("pair");
    if (pair.length !== 2 || pair[0] >= pair[1]) item.fail(`pair [${pair.join(", ")}] is not two aircraft in order`);
    pair.forEach((id) => known(item, "pair", id));
    const fromS = item.number("fromS");
    const toS = item.number("toS");
    if (toS < fromS) item.fail(`runs from ${fromS} s back to ${toS} s`);
    const ended = item.strings("ended");
    if (ended.some((id) => !pair.includes(id))) item.fail(`ends [${ended.join(", ")}], not of its pair`);
    return { pair: [pair[0], pair[1]], relation: item.string("relation"), fromS, toS, steps: item.count("steps", 1),
      kinds: item.strings("kinds"), closestM: item.number("closestM"), requiredM: item.number("requiredM"),
      wakeKnown: item.boolean("wakeKnown"), responsible: item.strings("responsible").map((id) => known(item, "responsible", id)),
      ended };
  });
  const atThreshold = reader.children("atThreshold").map((item) => ({
    atS: item.number("atS"), leader: known(item, "leader", item.string("leader")),
    follower: known(item, "follower", item.string("follower")), gapM: item.number("gapM"), requiredM: item.number("requiredM"),
    relation: item.string("relation"),
  }));
  const ended = reader.children("ended").map((item): TrainingJudgedEnd => {
    const datasetId = item.string("datasetId");
    if (!commanded.has(datasetId)) item.fail(`ends ${datasetId}, not an aircraft the window commands`);
    return { datasetId, atS: item.number("atS"), kind: item.string("kind"), relation: item.string("relation"),
      with: known(item, "with", item.string("with")) };
  });
  if (new Set(ended.map((item) => item.datasetId)).size !== ended.length) reader.fail("ends an aircraft twice");
  const endedIds = new Set(ended.map((item) => item.datasetId));
  const unnamed = episodes.flatMap((episode) => episode.ended).filter((id) => !endedIds.has(id));
  if (unnamed.length > 0) reader.fail(`episodes end [${unnamed.join(", ")}], which the ended aircraft do not list`);
  return { episodes, atThreshold, ended };
}

function readLandings(reader: Reader, key: string, landed: Set<string>): TrainingWindowLanding[] {
  const landings = reader.children(key).map((item) => ({ datasetId: item.string("datasetId"), atS: item.number("atS") }));
  if (landings.some((item, at) => at > 0 && item.atS < landings[at - 1].atS)) reader.fail(`${key} are not in the order they landed`);
  const named = landings.map((item) => item.datasetId).sort();
  if (named.join() !== [...landed].sort().join()) {
    reader.fail(`${key} name [${named.join(", ")}], but the aircraft that landed are [${[...landed].sort().join(", ")}]`);
  }
  return landings;
}

function readLossesByReading(reader: Reader, present: Set<string>, commanded: Set<string>) {
  return Object.fromEntries(TRAINING_SEPARATION_READINGS.map((reading) =>
    [reading, readLosses(reader.child(reading), present, commanded)])) as Record<TrainingSeparationReading, TrainingLosses>;
}

function readWindow(item: Reader, index: number, set: TrainingSetHead): TrainingWindow {
  const byId = new Map(set.flights.map((flight) => [flight.datasetId, flight]));
  const commanded = item.children("commanded").map((one): TrainingWindowCommanded => {
    const datasetId = one.string("datasetId");
    const flight = byId.get(datasetId);
    if (flight === undefined) return one.fail(`is ${datasetId}, not a flight of the set`);
    const recorded = readSceneTrack(one.child("recorded"));
    requireSetDatum(one.child("recorded"), recorded, flight);
    return { flight, rowZeroS: one.number("rowZeroS"), limitS: one.number("limitS"), recorded };
  });
  if (commanded.length === 0) item.fail("commands no aircraft");
  const others = item.children("others").map((one) => ({
    datasetId: one.string("datasetId"), callsign: one.string("callsign"), category: one.nullableString("category"),
    role: one.oneOf("role", TRAINING_OTHER_ROLES), track: readSceneTrack(one.child("track")),
  }));
  const ids = [...commanded.map((one) => one.flight.datasetId), ...others.map((one) => one.datasetId)];
  if (new Set(ids).size !== ids.length) item.fail("holds an aircraft twice");
  const present = new Set(ids);
  const commandedIds = new Set(commanded.map((one) => one.flight.datasetId));
  const recorded = item.child("recorded");
  const losses = readLossesByReading(recorded, present, commandedIds);
  // the record lands where its own track reaches the threshold: the judge's landings, whichever it ended
  const landed = new Set(recorded.children("landings").map((one) => one.string("datasetId")));
  if ([...landed].some((id) => !commandedIds.has(id))) recorded.fail("lands an aircraft the window does not command");
  return { index, opensUtc: item.string("opensUtc"), commanded, others,
    recorded: { losses, landings: readLandings(recorded, "landings", landed) } };
}

/** Parse a window set — all or nothing, as a sample. */
export function parseTrainingTrafficSet(raw: unknown): Parsed<TrainingTrafficSet> {
  return attempt(() => {
    const set = Reader.of(raw, "window set");
    const head = readSetHead(set, TRAINING_TRAFFIC_SCHEMA);
    const cohort = set.child("cohort");
    const windows = set.children("windows").map((item, index) => readWindow(item, index, head));
    if (windows.length === 0) set.fail("holds no window");
    return { ...head, cohort: { split: cohort.string("split"), windows: cohort.count("windows", 1), seed: cohort.number("seed"),
      drawnFrom: cohort.string("drawnFrom") }, windows };
  });
}

/** One aircraft's sentence in a window sample, bound to its window and its set flight: its words (`readSaidWords`); its
 *  outcome the window's — `lost_separation` exactly when the judge ended it (``end``), its own end otherwise; a crossing
 *  exactly for an own end read at one; its track from the observed state at its first predicted row, a step apart, to its own
 *  end; its words' end no later. */
function readWindowSentence(item: Reader, index: number, head: TrainingGenerationHead, set: TrainingSetHead,
  flight: TrainingFlight): TrainingWindowSentence {
  const firstRow = head.generation.firstPredictedRow;
  const { stepS } = set.vocabulary;
  if (item.string("datasetId") !== flight.datasetId) item.fail(`is ${item.raw("datasetId")}, not the window's ${flight.datasetId}`);
  const said = readSaidWords(item, index, firstRow, set);
  const outcome = item.oneOf("outcome", TRAINING_WINDOW_OUTCOMES);
  const own = item.oneOf("own", TRAINING_FREE_OUTCOMES);
  if (own === TRAINING_BELOW_GLIDEPATH && !speaksUnderAltitudes(head)) {
    item.fail(`is ${own}, but its model spoke without the procedure's altitudes`);
  }
  const endReader = item.nullableChild("end");
  const end = endReader === null ? null
    : { kind: endReader.string("kind"), relation: endReader.string("relation"), with: endReader.string("with") };
  if ((outcome === TRAINING_LOST_SEPARATION) !== (end !== null)) {
    item.fail(`is ${outcome} ${end === null ? "and names no end" : "but names the judge's end"}`);
  }
  if (end === null && outcome !== own) item.fail(`is ${outcome}, not its own end ${own}, and the judge did not end it`);
  const lastWordS = said.events[said.events.length - 1].row * set.vocabulary.stepS;
  if (end !== null && lastWordS > item.number("endS") + TIME_SLACK) {
    item.fail(`says a word at ${lastWordS} s, after the judge ended it at ${item.number("endS")} s`);
  }
  const crossingReader = item.nullableChild("crossing");
  const crossing = crossingReader === null ? null : readCrossing(crossingReader);
  if ((crossing !== null) !== TRAINING_CROSSING_OUTCOMES.includes(own)) {
    item.fail(`ends ${own} ${crossing === null ? "with no crossing" : "and carries a crossing"}`);
  }
  const track = parseGeneratedTrack(item.child("track"), firstRow * stepS, stepS);
  requireSetDatum(item.child("track"), track, flight);
  requireSetStart(item.child("track"), track, flight, firstRow);
  if (track.tS.slice(1).some((value, at) => Math.abs(value - track.tS[at] - stepS) > TIME_SLACK)) {
    item.fail("its track is not a step apart to its own end: a window's states are the scene's steps");
  }
  const ownEndS = item.number("ownEndS");
  const endS = item.number("endS");
  if (Math.abs(track.tS[track.tS.length - 1] - ownEndS) > TIME_SLACK) {
    item.fail(`its track ends at ${track.tS[track.tS.length - 1]} s, not at its own end, ${ownEndS} s`);
  }
  if (endS < firstRow * stepS - TIME_SLACK || endS > ownEndS + TIME_SLACK) {
    item.fail(`ends at ${endS} s, outside its first predicted row ${firstRow * stepS} s … its own end ${ownEndS} s`);
  }
  return { ...said, outcome, endS, crossing, track, own, ownEndS, end };
}

/** A window's samples, each bound to the window: its aircraft in the window's commanded order, the ended ones those
 *  its VISUAL losses end, at the same time, with the same aircraft; its landings its landed aircraft, in order. */
function readWindowSamples(item: Reader, head: TrainingGenerationHead, set: TrainingTrafficSet, window: TrainingWindow) {
  const present = new Set([...window.commanded.map((one) => one.flight.datasetId), ...window.others.map((one) => one.datasetId)]);
  const commandedIds = new Set(window.commanded.map((one) => one.flight.datasetId));
  const list = item.children("samples");
  if (list.length !== head.generation.samples) item.fail(`holds ${list.length} samples, not ${head.generation.samples}`);
  return list.map((one, index): TrainingWindowSample => {
    one.integer("sample", index, index);
    const sentences = one.children("aircraft");
    if (sentences.length !== window.commanded.length) {
      one.fail(`holds ${sentences.length} aircraft, the window commands ${window.commanded.length}`);
    }
    const aircraft = sentences.map((sentence, at) => readWindowSentence(sentence, index, head, set, window.commanded[at].flight));
    const losses = readLossesByReading(one, present, commandedIds);
    const ends = new Map(losses.visual.ended.map((end) => [end.datasetId, end]));
    aircraft.forEach((sentence, at) => {
      const { flight, rowZeroS } = window.commanded[at];
      const judged = ends.get(flight.datasetId);
      if ((judged !== undefined) !== (sentence.end !== null)) {
        one.fail(`${flight.datasetId} is ${sentence.outcome}, but the VISUAL losses ${judged === undefined ? "do not end" : "end"} it`);
      }
      if (judged !== undefined && (Math.abs(judged.atS - (rowZeroS + sentence.endS)) > TIME_SLACK || judged.with !== sentence.end!.with)) {
        one.fail(`${flight.datasetId} is ended at ${rowZeroS + sentence.endS} s with ${sentence.end!.with}, the losses say ` +
          `${judged.atS} s with ${judged.with}`);
      }
    });
    const landed = new Set(aircraft.flatMap((sentence, at) => (sentence.own === "landed" ? [window.commanded[at].flight.datasetId] : [])));
    return { sample: index, aircraft, losses, landings: readLandings(one, "landings", landed) };
  });
}

function readReadoutCell(reader: Reader): TrainingWindowReadoutCell {
  return { aircraft: reader.count("aircraft", 1), landed: reader.share("landed"), lostSeparation: reader.share("lostSeparation"),
    lostSeparationIfr: reader.share("lostSeparationIfr") };
}

function readReadoutPlaces(reader: Reader) {
  return { here: readReadoutCell(reader.child("here")), all: readReadoutCell(reader.child("all")) };
}

/** Parse a model's window overlay against the manifest entry that listed it and the window set it is drawn over — all or
 *  nothing: its windows are the set's (by opening and commanded aircraft), each with every sample. */
export function parseTrainingWindowGenerationOverlay(
  raw: unknown, entry: TrainingOverlayEntry, set: TrainingTrafficSet,
): Parsed<TrainingWindowGenerationOverlay> {
  return attempt(() => {
    const overlay = Reader.of(raw, "window overlay");
    if (overlay.raw("schema") !== TRAINING_WINDOW_GENERATION_SCHEMA) {
      overlay.fail(`schema is ${JSON.stringify(overlay.raw("schema"))}, expected ${JSON.stringify(TRAINING_WINDOW_GENERATION_SCHEMA)}`);
    }
    const head = readGenerationHead(overlay, entry, set, false);
    const readout = overlay.nullableChild("readout");
    const windows = overlay.children("windows");
    if (windows.length !== set.windows.length) overlay.fail(`holds ${windows.length} windows, the set ${set.windows.length}`);
    return {
      ...head,
      readout: readout === null ? null : {
        directory: readout.string("directory"), writtenUtc: readout.string("writtenUtc"),
        windowsPerAirport: readout.count("windowsPerAirport", 1), samples: readout.count("samples", 1),
        scene: readReadoutPlaces(readout.child("scene")), recorded: readReadoutPlaces(readout.child("recorded")),
      },
      windows: windows.map((item, index) => {
        const window = set.windows[index];
        const commanded = item.strings("commanded");
        if (item.string("opensUtc") !== window.opensUtc || commanded.join() !== window.commanded.map((one) => one.flight.datasetId).join()) {
          item.fail(`opens at ${item.raw("opensUtc")} commanding [${commanded.join(", ")}], but the set's window ${index} opens at ` +
            `${window.opensUtc} commanding [${window.commanded.map((one) => one.flight.datasetId).join(", ")}]`);
        }
        return { samples: readWindowSamples(item, head, set, window) };
      }),
    };
  });
}

// ── where the files live ─────────────────────────────────────────────────────

export async function fetchTrainingTrafficSet(airportCode: string, file: string): Promise<Parsed<TrainingTrafficSet>> {
  return parseTrainingTrafficSet(await fetchJson<unknown>(trainingFilePath(airportCode, file)));
}

export async function fetchTrainingWindowGenerationOverlay(
  airportCode: string, entry: TrainingOverlayEntry, set: TrainingTrafficSet,
): Promise<Parsed<TrainingWindowGenerationOverlay>> {
  return parseTrainingWindowGenerationOverlay(await fetchJson<unknown>(trainingFilePath(airportCode, entry.file)), entry, set);
}

// ── one window on screen ─────────────────────────────────────────────────────

/** "08-26 13:40Z": a window's opening, as the views name it. */
export function windowOpening(window: TrainingWindow): string {
  return `${window.opensUtc.slice(5, 10)} ${window.opensUtc.slice(11, 16)}Z`;
}

/** A window as one identity: the cursor keeps its time while it is on screen. */
export function trainingWindowKey(set: TrainingSetHead, window: TrainingWindow): string {
  return `${set.airport}/${set.setId}/window ${window.index}`;
}

/** The window whose aircraft is on screen, or null: the flight on screen is read on that window's clock. */
export function windowOnScreen(view: TrainingWindowView | null, selection: TrainingSelection | null): TrainingWindowView | null {
  return view !== null && selection !== null && trainingWindowKey(view.set, view.window) === selection.clock.scope ? view : null;
}

/** An aircraft of a window on screen: its set flight, read on the window's clock (its own 0 s at its row 0 there). The
 *  live executor does not fly it: the backend opens read-back sets only. */
export function trainingWindowSelection(set: TrainingSetHead, window: TrainingWindow, aircraft: TrainingWindowCommanded): TrainingSelection {
  return { airport: set.airport, setId: set.setId, vocabulary: set.vocabulary, candidates: set.candidates, flight: aircraft.flight,
    clock: { scope: trainingWindowKey(set, window), offsetS: aircraft.rowZeroS }, liveExecutor: false };
}

/** A model's sentences for one commanded aircraft of a window — one per sample — as the single-flight views read a model
 *  (`TrainingGenerationView`): the sentence bar's tab and samples, its 3D layers, the pick. */
export function windowGenerationView(overlay: TrainingWindowGenerationOverlay, window: TrainingWindow,
  aircraft: TrainingWindowCommanded): TrainingGenerationView {
  const at = window.commanded.indexOf(aircraft);
  if (at < 0) throw new Error(`${aircraft.flight.datasetId} is not commanded in window ${window.index}`);
  return {
    overlay,
    flight: { flightKey: aircraft.flight.flightKey, datasetId: aircraft.flight.datasetId, group: "commanded in its window", flown: true,
      samples: overlay.windows[window.index].samples.map((sample) => sample.aircraft[at]), start: null },
  };
}

/** A sentence's track on the window's clock (its own clock shifted by its row 0 there). */
export function sceneTrackOf(sentence: TrainingWindowSentence, aircraft: TrainingWindowCommanded): TrainingSceneTrack {
  return { ...sentence.track, tS: sentence.track.tS.map((value) => value + aircraft.rowZeroS) };
}

/** Where a track is at scene time ``atS`` — linear between its points — or null outside its span. */
export function sceneTrackAt(track: TrainingSceneTrack, atS: number): { lon: number; lat: number; heightHaeM: number } | null {
  const { tS } = track;
  if (atS < tS[0] || atS > tS[tS.length - 1]) return null;
  // the point at or before it (`rowAtTime`'s search), and the one after — the last point's own when it is the last
  const k = Math.min(rowAtTime(tS, atS), tS.length - 2);
  const span = k < 0 ? 0 : tS[k + 1] - tS[k];
  const f = span > 0 ? (atS - tS[k]) / span : 0;
  const at = (values: number[]) => (span > 0 ? values[k] + f * (values[k + 1] - values[k]) : values[Math.max(k, 0)]);
  return { lon: at(track.lon), lat: at(track.lat), heightHaeM: at(track.altitudeHaeM) };
}

/** The episodes going on at scene time ``atS``: a pair under its minimum from its first step until the step after its
 *  last (the judge reads the scene every ``stepS``). */
export function episodesAt(losses: TrainingLosses, atS: number, stepS: number): TrainingLossEpisode[] {
  return losses.episodes.filter((episode) => episode.fromS <= atS && atS < episode.toS + stepS);
}

/** The scene time the commanded aircraft span — their recorded rows and ``read`` (their tracks as the sentence read has
 *  them) — for the window's strip: the others are drawn inside it (one in the air since long before the window would
 *  squeeze the aircraft the window is about into a corner). */
export function windowSpanS(window: TrainingWindow, read: TrainingSceneTrack[]): [number, number] {
  const tracks = [...window.commanded.map((one) => one.recorded), ...read];
  return [Math.min(...tracks.map((track) => track.tS[0])), Math.max(...tracks.map((track) => track.tS[track.tS.length - 1]))];
}

// ── the window as read ───────────────────────────────────────────────────────

/** How a window went under one reading: an aircraft lost separation (the judge ended one), or none did and every
 *  commanded aircraft landed, or none did but not every one landed. */
export type TrainingWindowVerdict = "lost" | "clean" | "short";

/** A window as the sentence bar's source reads it — the record (the truth), or one sample of a model's — every commanded
 *  aircraft's track on the window's clock, the losses under both readings, the landings, and, of a model, its sentences. */
export interface TrainingWindowReading {
  /** The model's overlay and sample read; null: the record. */
  model: { overlay: TrainingWindowGenerationOverlay; sample: TrainingWindowSample } | null;
  tracks: TrainingSceneTrack[];
  losses: Record<TrainingSeparationReading, TrainingLosses>;
  landings: TrainingWindowLanding[];
}

/** The window as ``source`` reads it: a sample of one of the window set's models (the sample number read over a model with
 *  fewer is its last, as a single flight's is), or — the truth, or a model not of this set — the record. */
export function windowReading(view: TrainingWindowView, source: TrainingSource | null): TrainingWindowReading {
  const { window } = view;
  const overlay = source === null ? undefined : view.overlays.find((item) => item.overlayId === source.overlayId);
  if (overlay === undefined) {
    return { model: null, tracks: window.commanded.map((one) => one.recorded), losses: window.recorded.losses,
      landings: window.recorded.landings };
  }
  const samples = overlay.windows[window.index].samples;
  const sample = samples[Math.min(source!.sample, samples.length - 1)];
  return { model: { overlay, sample }, tracks: sample.aircraft.map((sentence, at) => sceneTrackOf(sentence, window.commanded[at])),
    losses: sample.losses, landings: sample.landings };
}

/** A window's verdict under a reading (`TrainingWindowVerdict`). */
export function windowVerdict(window: TrainingWindow, reading: Pick<TrainingWindowReading, "losses" | "landings">): TrainingWindowVerdict {
  if (reading.losses.visual.ended.length > 0) return "lost";
  return reading.landings.length === window.commanded.length ? "clean" : "short";
}

/** What became of one commanded aircraft under a reading: ended by the judge (why, with whom), else landed (its place in
 *  the landing order), else neither — and its place in the record's landing order, to read its own against. */
export function aircraftFate(window: TrainingWindow, reading: Pick<TrainingWindowReading, "losses" | "landings">, datasetId: string) {
  const ended = reading.losses.visual.ended.find((item) => item.datasetId === datasetId) ?? null;
  const place = reading.landings.findIndex((item) => item.datasetId === datasetId);
  const recorded = window.recorded.landings.findIndex((item) => item.datasetId === datasetId);
  return { ended, landed: place >= 0 ? place + 1 : null, recordedLanded: recorded >= 0 ? recorded + 1 : null };
}

/** Over every window of a set, read as the record (``overlay`` null) or every sample of a model: the commanded aircraft
 *  counted, those that landed and were not ended, and those the judge ended under VISUAL (the loop's) and IFR. */
export function windowSetCounts(set: TrainingTrafficSet, overlay: TrainingWindowGenerationOverlay | null) {
  const readings = set.windows.flatMap((window) => (overlay === null
    ? [{ window, losses: window.recorded.losses, landings: window.recorded.landings }]
    : overlay.windows[window.index].samples.map((sample) => ({ window, losses: sample.losses, landings: sample.landings }))));
  const count = (pick: (reading: (typeof readings)[number]) => number) => readings.reduce((sum, reading) => sum + pick(reading), 0);
  return {
    aircraft: count(({ window }) => window.commanded.length),
    landed: count(({ losses, landings }) => landings.filter((item) => !losses.visual.ended.some((end) => end.datasetId === item.datasetId)).length),
    lost: count(({ losses }) => losses.visual.ended.length),
    lostIfr: count(({ losses }) => losses.ifr.ended.length),
  };
}
