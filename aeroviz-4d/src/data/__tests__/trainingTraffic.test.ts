/**
 * trainingTraffic: a window set and a model's sentences in its windows (the Training module §2.9) — read against each
 * other, projected onto the single-flight views, and read as the sentence bar's source reads a window.
 */

import { poseAt } from "../trainingAttitude";
import { describe, expect, it } from "vitest";
import {
  aircraftFate,
  episodesAt,
  parseTrainingTrafficSet,
  parseTrainingWindowGenerationOverlay,
  TRAINING_TRAFFIC_SCHEMA,
  sceneTrackOf,
  trainingWindowKey,
  trainingWindowSelection,
  windowGenerationView,
  windowReading,
  windowSetCounts,
  windowSpanS,
  windowVerdict,
  type TrainingTrafficSet,
  type TrainingWindowGenerationOverlay,
  type TrainingWindowView,
} from "../trainingTraffic";
import { generationOnScreen, parseTrainingOverlays, TRAINING_OVERLAY_KINDS, type TrainingOverlayEntry } from "../trainingOverlays";
import { cursorOnFlight, isOwnClock, trainingSelectionKey } from "../trainingSample";
import { parseOverlayOver, parseTrainingSet, TRAINING_OVERLAYS_OVER } from "../trainingSets";
import { WORD, mockSample } from "./trainingSample.fixture";
import {
  BACKGROUND_ID,
  REPLAYED_ID,
  ROW_ZERO_S,
  STRAIGHT_ID,
  TRAFFIC_SET_ID,
  VECTORED_ID,
  WINDOW_MODEL_ID,
  mockTrafficSet,
  mockWindowOverlay,
  mockWindowOverlays,
} from "./trainingTraffic.fixture";

function trafficSet(change: (raw: any) => void = () => undefined): TrainingTrafficSet {
  const raw: any = mockTrafficSet();
  change(raw);
  const parsed = parseTrainingTrafficSet(raw);
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

function refusal(read: (raw: any) => { ok: boolean; problem?: string }, raw: any, change: (raw: any) => void): string {
  change(raw);
  const parsed = read(raw);
  if (parsed.ok) throw new Error("expected a refusal");
  return parsed.problem!;
}

function entry(): TrainingOverlayEntry {
  const parsed = parseTrainingOverlays(mockWindowOverlays());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value.overlays[0];
}

function overlay(change: (raw: any) => void = () => undefined): TrainingWindowGenerationOverlay {
  const raw = mockWindowOverlay();
  change(raw);
  const parsed = parseTrainingWindowGenerationOverlay(raw, entry(), trafficSet());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

function view(): TrainingWindowView {
  const set = trafficSet();
  return { set, window: set.windows[0], overlays: [overlay()], focus: () => undefined };
}

describe("a window set", () => {
  it("reads its windows: the commanded aircraft as the set's flights, the others by role, the window as recorded", () => {
    const set = trafficSet();
    const [window] = set.windows;
    expect(set.setId).toBe(TRAFFIC_SET_ID);
    expect(set.cohort).toEqual({ split: "select", windows: 1, seed: 1337, drawnFrom: "a test draw" });
    expect(window.commanded.map((one) => [one.flight.datasetId, one.rowZeroS])).toEqual([[VECTORED_ID, 100], [STRAIGHT_ID, 110]]);
    expect(window.commanded[0].flight).toBe(set.flights.find((flight) => flight.datasetId === VECTORED_ID));
    expect(window.others.map((other) => [other.datasetId, other.role])).toEqual([[REPLAYED_ID, "replayed"], [BACKGROUND_ID, "background"]]);
    expect(window.recorded.losses.visual.ended).toEqual([]);
    expect(window.recorded.losses.ifr.episodes[0].pair).toEqual([REPLAYED_ID, VECTORED_ID].sort());
    expect(window.recorded.landings.map((item) => item.datasetId)).toEqual([VECTORED_ID, STRAIGHT_ID]);
  });

  it("is refused by name when it holds what the judge could not have written", () => {
    const read = parseTrainingTrafficSet;
    expect(refusal(read, mockTrafficSet(), (raw) => { raw.schema = "aeroviz-training-traffic-v0"; }))
      .toContain(`expected "${TRAINING_TRAFFIC_SCHEMA}"`);
    expect(refusal(read, mockTrafficSet(), (raw) => { raw.windows[0].commanded[0].datasetId = "KXXX:NOPE"; }))
      .toContain("not a flight of the set");
    expect(refusal(read, mockTrafficSet(), (raw) => { raw.windows[0].recorded.ifr.episodes[0].pair = ["KXXX:A", VECTORED_ID]; }))
      .toContain("KXXX:A is not in the window");
    expect(refusal(read, mockTrafficSet(), (raw) => { raw.windows[0].recorded.ifr.ended[0].datasetId = REPLAYED_ID; }))
      .toContain("not an aircraft the window commands");
    expect(refusal(read, mockTrafficSet(), (raw) => { raw.windows[0].recorded.landings.reverse(); }))
      .toContain("not in the order they landed");
    expect(refusal(read, mockTrafficSet(), (raw) => { raw.windows[0].others.push({ ...raw.windows[0].others[0] }); }))
      .toContain("holds an aircraft twice");
    expect(refusal(read, mockTrafficSet(), (raw) => { raw.windows[0].commanded[0].recorded.altitudeHaeM[3] += 5; }))
      .toContain("HAE − MSL");
  });

  it("opens by its kind, and its overlays only over it", () => {
    const open = parseTrainingSet("traffic-windows", mockTrafficSet());
    const readback = parseTrainingSet("vocabulary-readback", mockSample());
    if (!open.ok || !readback.ok) throw new Error("the fixtures should open");
    expect(parseOverlayOver(entry(), mockWindowOverlay(), open.value)).toEqual({ ok: true, value: { flights: 2 } });
    expect(parseOverlayOver(entry(), mockWindowOverlay(), readback.value))
      .toEqual({ ok: false, problem: "a window-generation overlay is not drawn over a vocabulary-readback set" });
    // every overlay kind is drawn over exactly one kind of set
    const listed = Object.values(TRAINING_OVERLAYS_OVER).flat();
    expect([...listed].sort()).toEqual([...TRAINING_OVERLAY_KINDS].sort());
  });
});

describe("a model's sentences in the windows", () => {
  it("reads every sample of every window, each aircraft's sentence on its own clock, ended as its window's losses say", () => {
    const read = overlay();
    const [first, second] = read.windows[0].samples;
    expect(first.aircraft.map((one) => [one.outcome, one.own, one.endS, one.ownEndS])).toEqual([
      ["lost_separation", "timeout", 20, 60], ["landed", "landed", 50, 50]]);
    expect(first.aircraft[0].end).toEqual({ kind: "in_trail", relation: "same", with: STRAIGHT_ID });
    expect(first.aircraft[0].track.tS[first.aircraft[0].track.tS.length - 1]).toBe(60);
    expect(second.aircraft.every((one) => one.outcome === "timeout")).toBe(true);
  });

  it("is refused by name where its books do not agree", () => {
    const set = trafficSet();
    const read = (raw: any) => parseTrainingWindowGenerationOverlay(raw, entry(), set);
    expect(refusal(read, mockWindowOverlay(), (raw) => { raw.windows[0].commanded.reverse(); }))
      .toContain("but the set's window 0 opens at");
    expect(refusal(read, mockWindowOverlay(), (raw) => { raw.windows[0].samples.pop(); })).toContain("holds 1 samples, not 2");
    // ended by the judge at another time than the losses say
    expect(refusal(read, mockWindowOverlay(), (raw) => { raw.windows[0].samples[0].visual.ended[0].atS = 130; }))
      .toContain("the losses say 130 s");
    // the losses end an aircraft its sentence does not say was ended
    expect(refusal(read, mockWindowOverlay(), (raw) => { raw.windows[0].samples[0].visual.ended.push(
      { datasetId: STRAIGHT_ID, atS: 200, kind: "in_trail", relation: "same", with: VECTORED_ID }); }))
      .toContain(`${STRAIGHT_ID} is landed, but the VISUAL losses end it`);
    expect(refusal(read, mockWindowOverlay(), (raw) => { raw.windows[0].samples[0].aircraft[0].end = null; }))
      .toContain("is lost_separation and names no end");
    expect(refusal(read, mockWindowOverlay(), (raw) => { raw.windows[0].samples[0].aircraft[1].crossing = null; }))
      .toContain("ends landed with no crossing");
    expect(refusal(read, mockWindowOverlay(), (raw) => { raw.windows[0].samples[0].aircraft[0].ownEndS = 58; }))
      .toContain("not at its own end, 58 s");
    expect(refusal(read, mockWindowOverlay(), (raw) => { raw.windows[0].samples[0].landings = []; }))
      .toContain("but the aircraft that landed are");
    expect(refusal(read, mockWindowOverlay(), (raw) => { raw.windows[0].samples[0].aircraft[0].track.lat[0] += 0.01; }))
      .toContain("where it was flown from");
    // a word said after the judge ended it; an episode ending an aircraft the ends do not list
    expect(refusal(read, mockWindowOverlay(), (raw) => { raw.windows[0].samples[0].aircraft[0].events.push({ row: 20, column: 2, value: WORD.heading225 }); }))
      .toContain("says a word at 40 s, after the judge ended it at 20 s");
    expect(refusal(read, mockWindowOverlay(), (raw) => { raw.windows[0].samples[0].visual.episodes[0].ended.push(STRAIGHT_ID); }))
      .toContain("which the ended aircraft do not list");
  });

  it("is projected onto the aircraft on screen as a model's sentences over a flight — a sample per window sample", () => {
    const shown = view();
    const [window] = shown.set.windows;
    const projected = windowGenerationView(shown.overlays[0], window, window.commanded[1]);
    expect(projected.flight.flightKey).toBe(window.commanded[1].flight.flightKey);
    expect(projected.flight.samples).toEqual(shown.overlays[0].windows[0].samples.map((sample) => sample.aircraft[1]));
    const selection = trainingWindowSelection(shown.set, window, window.commanded[1]);
    expect(selection.clock).toEqual({ scope: trainingWindowKey(shown.set, window), offsetS: ROW_ZERO_S[STRAIGHT_ID] });
    expect(selection.liveExecutor).toBe(false);
    expect(isOwnClock(selection)).toBe(false);
    // on a window's clock the cursor is on the aircraft only over its axis; on a flight's own, always (held at its ends)
    expect([cursorOnFlight(selection, -1, 120), cursorOnFlight(selection, 0, 120), cursorOnFlight(selection, 121, 120)])
      .toEqual([false, true, false]);
    const own = { ...selection, clock: { scope: trainingSelectionKey(selection)!, offsetS: 0 } };
    expect([cursorOnFlight(own, -1, 120), cursorOnFlight(own, 500, 120)]).toEqual([true, true]);
    expect(trainingSelectionKey(selection)).toBe(`KXXX/${TRAFFIC_SET_ID}/${window.commanded[1].flight.flightKey}`);
    // the sentence bar reads it as any model's: its sample by number
    expect(generationOnScreen([projected], { overlayId: WINDOW_MODEL_ID, sample: 1 }, selection)?.sentence?.outcome).toBe("timeout");
  });
});

describe("a window as read", () => {
  it("reads the record for the truth, a model's sample for its source (a sample past its last: its last)", () => {
    const shown = view();
    const recorded = windowReading(shown, null);
    expect(recorded.model).toBeNull();
    expect(recorded.tracks).toEqual(shown.window.commanded.map((one) => one.recorded));
    const sample = windowReading(shown, { overlayId: WINDOW_MODEL_ID, sample: 0 });
    expect(sample.model?.sample.sample).toBe(0);
    expect(sample.tracks[0].tS[0]).toBe(ROW_ZERO_S[VECTORED_ID] + 8);
    expect(windowReading(shown, { overlayId: WINDOW_MODEL_ID, sample: 7 }).model?.sample.sample).toBe(1);
    expect(windowReading(shown, { overlayId: "another set's model", sample: 0 }).model).toBeNull();
  });

  it("gives a window its verdict and each aircraft its fate, and counts a set's windows", () => {
    const shown = view();
    const { window } = shown;
    expect(windowVerdict(window, window.recorded)).toBe("clean");
    const [lost, short] = shown.overlays[0].windows[0].samples;
    expect([windowVerdict(window, lost), windowVerdict(window, short)]).toEqual(["lost", "short"]);
    expect(aircraftFate(window, lost, VECTORED_ID).ended?.with).toBe(STRAIGHT_ID);
    expect(aircraftFate(window, lost, STRAIGHT_ID)).toEqual({ ended: null, landed: 1, recordedLanded: 2 });
    expect(windowSetCounts(shown.set, null)).toEqual({ aircraft: 2, landed: 2, lost: 0, lostIfr: 1 });
    expect(windowSetCounts(shown.set, shown.overlays[0])).toEqual({ aircraft: 4, landed: 1, lost: 1, lostIfr: 0 });
  });

  it("puts an aircraft where its track is at a time, and the pairs under their minimum until the step after their last", () => {
    const shown = view();
    const track = sceneTrackOf(shown.overlays[0].windows[0].samples[0].aircraft[0], shown.window.commanded[0]);
    expect(poseAt(track, track.tS[0] - 1)).toBeNull();
    const halfway = poseAt(track, (track.tS[0] + track.tS[1]) / 2)!;
    expect(halfway.lon).toBeCloseTo((track.lon[0] + track.lon[1]) / 2, 12);
    const losses = shown.overlays[0].windows[0].samples[0].losses.visual;
    expect(episodesAt(losses, 119, 2)).toEqual([]);
    expect(episodesAt(losses, 125.9, 2)).toHaveLength(1);
    expect(episodesAt(losses, 126, 2)).toEqual([]);
    expect(windowSpanS(shown.window, [track])).toEqual([100, 228]);
  });
});
