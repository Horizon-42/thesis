/**
 * The models' own sentences (`prior-generation` overlays): what the reader accepts over the fixture's set, how the views
 * find the sentence read, and — the point of it — that a payload whose samples break their own bookkeeping is refused
 * whole and by name.
 */
import { describe, expect, it } from "vitest";

import { parseTrainingSample, sentenceColumnRuns, sentenceWordAt, trainingClearedValue, type TrainingSample } from "../trainingSample";
import {
  generatedRowAt,
  generatedTrackRows,
  generationLanded,
  generationOnScreen,
  parseTrainingGenerationOverlay,
  parseTrainingOverlays,
  trainingModelGroups,
  trainingModelLabel,
  trainingOverlaysOf,
  trainingRunName,
  TRAINING_BELOW_GLIDEPATH,
  TRAINING_GENERATION_SCHEMA,
  TRAINING_PROCEDURE_ALTITUDES,
  type TrainingGenerationOverlay,
  type TrainingGenerationView,
} from "../trainingOverlays";
import { trainingSelectionOf } from "../trainingSample";
import { SET_ID, STRAIGHT_KEY, VECTORED_KEY, WORD, mockSample } from "./trainingSample.fixture";
import {
  AUGMENTED_R1_ID, AUGMENTED_R2_ID, BASE_MODEL_ID, MOCK_GENERATION_FIRST_ROW, POST_TRAINED_ID, mockGenerationEntry, mockGenerationOverlay,
  mockOverlaysWithGenerations,
} from "./trainingOverlays.fixture";
import { trainingModelColour, TRAINING_MODEL_COLOR } from "../../utils/trainingWordColors";
import { trainingModelText } from "../trainingText";

function sample(): TrainingSample {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

function read(id = BASE_MODEL_ID): TrainingGenerationOverlay {
  const parsed = parseTrainingGenerationOverlay(mockGenerationOverlay(id), mockGenerationEntry(id), sample());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

function refusal(change: (raw: any) => void): string {
  const raw: any = mockGenerationOverlay(BASE_MODEL_ID);
  change(raw);
  const result = parseTrainingGenerationOverlay(raw, mockGenerationEntry(BASE_MODEL_ID), sample());
  if (result.ok) throw new Error("the change was accepted");
  return result.problem;
}

/** The fixture's vectored flight's first sample. */
const landedSample = (raw: any) => raw.flights[0].samples[0];

describe("a generation overlay", () => {
  it("is listed beside the set's other overlays, under its own kind", () => {
    const parsed = parseTrainingOverlays(mockOverlaysWithGenerations());
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(trainingOverlaysOf(parsed.value, SET_ID, "prior-generation").map((item) => item.id)).toEqual([BASE_MODEL_ID, POST_TRAINED_ID]);
  });

  it("reads each flight's samples: the words from the first predicted row, the flight's end and its track", () => {
    const overlay = read();
    expect(trainingModelLabel(overlay.model)).toBe("base");
    expect(overlay.generation.firstPredictedRow).toBe(MOCK_GENERATION_FIRST_ROW);
    const [vectored, straight] = overlay.flights;
    expect([vectored.flightKey, straight.flightKey]).toEqual([VECTORED_KEY, STRAIGHT_KEY]);
    expect(straight).toMatchObject({ flown: false, group: "stand-in dynamics", samples: [] });
    const [landed, timedOut] = vectored.samples;
    expect(landed).toMatchObject({ sample: 0, outcome: "landed", endS: 110, rows: 56, firstRow: MOCK_GENERATION_FIRST_ROW });
    expect(timedOut).toMatchObject({ sample: 1, outcome: "timeout", crossing: null, firstRunway: 0, lastRunway: 1 });
    // a sentence is a sentence: its runs tile its rows from its first
    const heading = sentenceColumnRuns(landed, "heading");
    expect(heading.map((run) => [run.row, run.endRow, run.value])).toEqual([
      [MOCK_GENERATION_FIRST_ROW, 12, WORD.heading270], [12, 16, WORD.heading225], [16, 56, WORD.heading180],
    ]);
    expect(sentenceWordAt(landed, "heading", MOCK_GENERATION_FIRST_ROW - 1)).toBeNull();          // observed only
    expect(landed.events.some((event) => event.column === 1 && event.value === trainingClearedValue(sample().vocabulary))).toBe(true);
    expect(overlay.readout?.prior.all.all).toEqual({ flights: 800, landed: 0.9 });
    expect(overlay.readout?.prior.here.all).toEqual({ flights: 800, landed: 0.85 });
    expect(overlay.readout?.labelled.here.vectored).toBeNull();
  });

  it("names its model — base, or a post-trained round and the model it started from — and colours it by name", () => {
    expect(read().model).toMatchObject({ name: "base", round: null, fineTuning: null });
    expect(trainingModelColour(read().model)).toBe(TRAINING_MODEL_COLOR.base);
    const landing = read(POST_TRAINED_ID);
    expect(landing.model).toMatchObject({ name: "landing", round: 1, run: "4dTrajectory/outputs/POOLED/prior/v3_rl/grpo_s1",
      fineTuning: { schema: "ts-prior-landing-reward-v1", fromName: "base", fromRound: null } });
    expect(trainingModelColour(landing.model)).toBe(TRAINING_MODEL_COLOR.landing);
    const augmented = read(AUGMENTED_R2_ID);
    expect(trainingModelLabel(augmented.model)).toBe("augmented r2");
    expect(trainingModelColour(augmented.model)).toBe(TRAINING_MODEL_COLOR.augmented);
    expect(trainingModelText(augmented.model)).toBe(
      "augmented r2 (v3_stage2/aug_s1, post-trained from landing r1 by ts-prior-augmented-reward-v5)");
    expect(trainingModelText(read().model)).toBe("base (v3_step1/full_s1, trained on data alone)");
    // every name its own colour
    expect(new Set(Object.values(TRAINING_MODEL_COLOR)).size).toBe(3);
  });

  it("reads a sentence the glidepath lower edge stopped only from a model that spoke under the procedure's altitudes", () => {
    const stopped = (id: string) => {
      const raw: any = mockGenerationOverlay(id);
      raw.flights[0].samples[1].outcome = TRAINING_BELOW_GLIDEPATH;      // no crossing; stopped at its last step, 150 s
      raw.flights[0].samples[1].rows = 75;
      return parseTrainingGenerationOverlay(raw, mockGenerationEntry(id), sample());
    };
    const augmented = stopped(AUGMENTED_R1_ID);
    if (!augmented.ok) throw new Error(augmented.problem);
    expect(augmented.value.flights[0].samples[1].outcome).toBe(TRAINING_BELOW_GLIDEPATH);
    expect(augmented.value.generation.procedureMasks).toEqual([{ name: TRAINING_PROCEDURE_ALTITUDES, dataSha256: "d".repeat(64) }]);
    expect(read().generation.procedureMasks).toEqual([]);
    const base = stopped(BASE_MODEL_ID);
    expect(base.ok ? "" : base.problem).toMatch("is below_glidepath, but its model spoke without the procedure's altitudes");
    // a stopped sentence carries no crossing
    const crossed: any = mockGenerationOverlay(AUGMENTED_R1_ID);
    crossed.flights[0].samples[0].outcome = TRAINING_BELOW_GLIDEPATH;
    const refused = parseTrainingGenerationOverlay(crossed, mockGenerationEntry(AUGMENTED_R1_ID), sample());
    expect(refused.ok ? "" : refused.problem).toMatch("is below_glidepath and carries a crossing");
    // stopped anywhere but at the end of its last step
    const early: any = mockGenerationOverlay(AUGMENTED_R1_ID);
    Object.assign(early.flights[0].samples[1], { outcome: TRAINING_BELOW_GLIDEPATH, rows: 76 });
    const late = parseTrainingGenerationOverlay(early, mockGenerationEntry(AUGMENTED_R1_ID), sample());
    expect(late.ok ? "" : late.problem).toMatch("is below_glidepath at 150 s, not at the end of its last step, 152 s");
    const set = { name: TRAINING_PROCEDURE_ALTITUDES, dataSha256: "d".repeat(64) };
    expect(refusal((raw) => { raw.generation.procedureMasks = [set, set]; })).toMatch("names a set of the procedure's masks twice");
    expect(refusal((raw) => { raw.generation.procedureMasks = [{ ...set, name: "procedure-altitudes-v1" }]; }))
      .toMatch("not one of procedure-altitudes-v2");
    expect(refusal((raw) => { raw.generation.procedureMasks = [{ ...set, dataSha256: "x" }]; })).toMatch('dataSha256 "x" is not a sha256');
  });

  it("refuses a model whose name, round and start model disagree, by name", () => {
    expect(refusal((raw) => { raw.model.name = "post-trained"; })).toMatch("not one of base, landing, augmented");
    expect(refusal((raw) => { raw.model.round = 1; })).toMatch("is base with round 1 and no fineTuning: base alone has neither");
    expect(refusal((raw) => { delete raw.model.run; })).toMatch("model.run is undefined, not a non-empty string");
    expect(refusal((raw) => { raw.model.run = "somewhere/full_s1"; })).toMatch('run "somewhere/full_s1" is not a prior run');
    const tuned = (change: (model: any) => void) => refusal((raw) => {
      Object.assign(raw.model, mockGenerationOverlay(AUGMENTED_R1_ID).model);
      change(raw.model);
    });
    expect(tuned((model) => { model.fineTuning = null; })).toMatch("is augmented with round 1 and no fineTuning");
    expect(tuned((model) => { model.round = 0; })).toMatch("model.round is 0, not a whole number of at least 1");
    expect(tuned((model) => { model.fineTuning.fromRound = null; })).toMatch("starts from landing round null: base alone has no round");
    expect(tuned((model) => { model.fineTuning.fromName = "sft"; })).toMatch("not one of base, landing, augmented");
  });

  it("refuses another schema, another step or columns in another order by name", () => {
    expect(refusal((raw) => { raw.schema = "aeroviz-training-generation-v0"; })).toMatch(`expected ${JSON.stringify(TRAINING_GENERATION_SCHEMA)}`);
    expect(refusal((raw) => { raw.generation.stepS = 1; })).toMatch("stepS is 1, the set's step 2");
    expect(refusal((raw) => { raw.columns.reverse(); })).toMatch("in that order");
  });

  it("refuses a sample whose first predicted row leaves a column unsaid, or whose words are out of order", () => {
    expect(refusal((raw) => { landedSample(raw).events.splice(2, 1); })).toMatch("says 5 columns, not all six");
    expect(refusal((raw) => { landedSample(raw).events.reverse(); })).toMatch("not in (row, column) order");
    expect(refusal((raw) => { landedSample(raw).events[6].row = MOCK_GENERATION_FIRST_ROW - 1; })).toMatch(/row is 3, not a whole number in 4…55/);
  });

  it("refuses a sample whose runways, crossing or track disagree with its own words and end", () => {
    expect(refusal((raw) => { raw.flights[0].samples[1].lastRunway = 0; })).toMatch("its words say 0 → 1");
    expect(refusal((raw) => { raw.flights[0].samples[1].crossing = { crossM: 0, heightM: 10, atS: 140, runway: 0 }; })).toMatch("is timeout and carries a crossing");
    expect(refusal((raw) => { landedSample(raw).crossing = null; })).toMatch("is landed with no crossing");
    expect(refusal((raw) => { landedSample(raw).endS = 104; })).toMatch("its track ends at 110 s, not at 104 s");
    expect(refusal((raw) => { landedSample(raw).track.tS[0] = 0; })).toMatch("does not start at the first predicted row (8 s)");
    expect(refusal((raw) => { landedSample(raw).track.lat.pop(); })).toMatch("lat has");
  });

  it("refuses a sample whose bookkeeping is not its words': runway changes, go-arounds, cleared at the end", () => {
    expect(refusal((raw) => { raw.flights[0].samples[1].runwayChanges = 0; })).toMatch("runwayChanges is 0, its words give 1");
    expect(refusal((raw) => { landedSample(raw).goArounds = 1; })).toMatch("goArounds is 1, its words give 0");
    expect(refusal((raw) => { landedSample(raw).clearedAtEnd = false; })).toMatch("clearedAtEnd is false, its words give true");
  });

  it("refuses a sample that ends outside its rows, or whose masked mass names no column or is no probability", () => {
    expect(refusal((raw) => { landedSample(raw).endS = 200; })).toMatch("ends at 200 s, outside its rows 8…112 s");
    expect(refusal((raw) => { landedSample(raw).forbiddenMass = { wind: 0.1 }; })).toMatch("forbiddenMass names wind: not columns");
    expect(refusal((raw) => { landedSample(raw).forbiddenMass = { angle: 1.5 }; })).toMatch("is 1.5, not a probability");
  });

  it("refuses a track that does not run forward or whose points are not the steps the views place words at", () => {
    expect(refusal((raw) => { [landedSample(raw).track.tS[3], landedSample(raw).track.tS[4]] = [16, 14]; })).toMatch("tS does not run forward");
    expect(refusal((raw) => { landedSample(raw).track.tS[5] = 18.5; })).toMatch("tS[5] is 18.5 s, not the step at 18 s");
    expect(refusal((raw) => {
      const track = landedSample(raw).track;
      for (const key of Object.keys(track)) track[key].splice(-2, 1);          // a point missing before the last
    })).toMatch("its last point is more than a step after the one before");
  });

  it("refuses a readout cell counted over no flight", () => {
    expect(refusal((raw) => { raw.readout.prior.here.all.flights = 0; })).toMatch("flights is 0, not a whole number of at least 1");
  });

  it("keeps a dynamics failure's track one state short of its end, the failed state left out", () => {
    const ok = parseTrainingGenerationOverlay((() => {
      const raw: any = mockGenerationOverlay(BASE_MODEL_ID);
      Object.assign(landedSample(raw), { outcome: "dynamics_failure", crossing: null, endS: 111 });
      return raw;
    })(), mockGenerationEntry(BASE_MODEL_ID), sample());
    expect(ok.ok).toBe(true);
    expect(refusal((raw) => { Object.assign(landedSample(raw), { outcome: "dynamics_failure", crossing: null }); })).toMatch("not at 109 s");
  });

  it("refuses a flight whose samples are not the overlay's count, or a flight not flown that carries any", () => {
    expect(refusal((raw) => { raw.flights[0].samples.pop(); })).toMatch("is flown and holds 1 samples, not 2");
    expect(refusal((raw) => { raw.flights[1].samples = [landedSample(raw)]; })).toMatch("is not flown and holds 1 samples");
    expect(refusal((raw) => { raw.flights[0].samples[1].sample = 0; })).toMatch("sample is 0, not a whole number in 1…1");
  });
});

describe("the set's approach words", () => {
  it("must name the clearance and the go-around: a model's words are read by them", () => {
    for (const name of ["cleared", "go-around"]) {
      const raw: any = mockSample();
      raw.vocabulary.approachClasses = raw.vocabulary.approachClasses.map((item: string) => (item === name ? "other" : item));
      const parsed = parseTrainingSample(raw);
      expect(parsed.ok ? "" : parsed.problem).toMatch(`name no "${name}" word`);
    }
  });
});

describe("a set's samples, counted", () => {
  it("counts the landed samples on the set's flights, in all and per approach kind — none for a kind it does not fly", () => {
    const landed = generationLanded(read(), sample().flights);
    // the vectored flight flies twice, one landing; the straight-in one is not flown
    expect(landed).toEqual({ all: { flights: 2, landed: 0.5 }, vectored: { flights: 2, landed: 0.5 }, "straight-in": null });
  });

  it("counts nothing — not zero — for a model that flies none of the set's flights", () => {
    const raw: any = mockGenerationOverlay(BASE_MODEL_ID);
    raw.flights[0] = { ...raw.flights[0], flown: false, group: "stand-in dynamics", samples: [] };
    const parsed = parseTrainingGenerationOverlay(raw, mockGenerationEntry(BASE_MODEL_ID), sample());
    if (!parsed.ok) throw new Error(parsed.problem);
    expect(generationLanded(parsed.value, sample().flights)).toEqual({ all: null, "straight-in": null, vectored: null });
  });
});

describe("the sentence read", () => {
  const selection = () => trainingSelectionOf(sample(), sample().flights[0]);
  const views = (): TrainingGenerationView[] => [read(), read(POST_TRAINED_ID)].map((overlay) => ({ overlay, flight: overlay.flights[0] }));

  it("is the truth without a source, or with one no view of the flight on screen publishes", () => {
    expect(generationOnScreen(views(), null, selection())).toBeNull();
    expect(generationOnScreen(views(), { overlayId: "gone", sample: 0 }, selection())).toBeNull();
    expect(generationOnScreen(views(), { overlayId: BASE_MODEL_ID, sample: 0 }, trainingSelectionOf(sample(), sample().flights[1]))).toBeNull();
  });

  it("is the chosen model's chosen sample — its last for a number past its count", () => {
    const shown = generationOnScreen(views(), { overlayId: POST_TRAINED_ID, sample: 1 }, selection());
    expect(shown?.view.overlay.overlayId).toBe(POST_TRAINED_ID);
    expect(shown?.sentence?.outcome).toBe("timeout");
    expect(generationOnScreen(views(), { overlayId: BASE_MODEL_ID, sample: 7 }, selection())?.sentence?.sample).toBe(1);
  });

  it("runs past the observed flight's rows, and a word's stretch of track is cut where the flight ended", () => {
    const timedOut = read().flights[0].samples[1];
    expect(generatedRowAt(2, 145)).toBe(72);                            // the observed flight has 60 rows
    expect(generatedRowAt(2, -5)).toBe(0);
    // past the sentence's last row no word is in force: nothing stays selected there
    expect(sentenceWordAt(timedOut, "heading", generatedRowAt(2, 1e6))).toBeNull();
    const landed = read().flights[0].samples[0];
    // point k is row first + k: the heading word said at 16 lasts to the end, cut at the track's last point
    expect(generatedTrackRows(landed, landed.firstRow, 16, 56)).toEqual({ first: 12, last: landed.track.tS.length - 1 });
    expect(generatedTrackRows(landed, landed.firstRow, 12, 16)).toEqual({ first: 8, last: 12 });
    expect(generatedTrackRows(landed, landed.firstRow, 90, 95)).toBeNull();
  });
});

describe("the models grouped for the views", () => {
  const models = (ids: string[]) => ids.map((id) => read(id));
  /** One of the fixture's overlays under another id and run (another run of its stage). */
  const inRun = (id: string, overlayId: string, run: string): TrainingGenerationOverlay => {
    const overlay = read(id);
    return { ...overlay, overlayId, model: { ...overlay.model, run: `4dTrajectory/outputs/POOLED/prior/${run}` } };
  };
  const named = (overlays: TrainingGenerationOverlay[]) => trainingModelGroups(overlays, (o) => o)
    .map((group) => [group.title, group.members.map(group.memberLabel)]);

  it("orders them by name as they are trained, then run, each model's rounds in order", () => {
    expect(named(models([AUGMENTED_R2_ID, POST_TRAINED_ID, AUGMENTED_R1_ID, BASE_MODEL_ID]))).toEqual([
      ["base", ["base"]], ["landing", ["landing r1"]], ["augmented", ["augmented r1", "augmented r2"]],
    ]);
  });

  it("names two runs of one stage by their runs — a run of one round too — and no other model", () => {
    const restart = inRun(AUGMENTED_R1_ID, "restart_r1", "v3_restart/aug_s1");
    expect(named([...models([BASE_MODEL_ID, AUGMENTED_R2_ID, AUGMENTED_R1_ID]), restart])).toEqual([
      ["base", ["base"]],
      ["augmented · v3_restart", ["augmented r1 · v3_restart"]],
      ["augmented · v3_stage2", ["augmented r1 · v3_stage2", "augmented r2 · v3_stage2"]],
    ]);
    // two seeds of one campaign: the whole run
    const seed = inRun(AUGMENTED_R1_ID, "seed_r1", "v3_stage2/aug_s2");
    expect(named([read(AUGMENTED_R1_ID), seed]).map(([title]) => title)).toEqual(["augmented · v3_stage2/aug_s1", "augmented · v3_stage2/aug_s2"]);
    expect(trainingRunName("4dTrajectory/outputs/POOLED/prior/v3_rl/grpo_s1")).toBe("v3_rl/grpo_s1");
  });

  it("gives a round published twice for its run a group of its own each, named by overlay id — never two same chips", () => {
    const again = { ...read(BASE_MODEL_ID), overlayId: "generation_base_again" };
    const twice = { ...read(AUGMENTED_R2_ID), overlayId: "generation_aug_r2_again" };
    expect(named([...models([BASE_MODEL_ID, AUGMENTED_R1_ID, AUGMENTED_R2_ID]), again, twice])).toEqual([
      ["base · generation_base", ["base · generation_base"]],
      ["base · generation_base_again", ["base · generation_base_again"]],
      ["augmented", ["augmented r1"]],
      ["augmented · generation_aug_r2", ["augmented r2 · generation_aug_r2"]],
      ["augmented · generation_aug_r2_again", ["augmented r2 · generation_aug_r2_again"]],
    ]);
  });
});
