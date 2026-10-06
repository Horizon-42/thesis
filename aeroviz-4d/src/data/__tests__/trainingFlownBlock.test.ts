/**
 * The one reader of a flown sentence's block (D135, outline §6.2 items 7, 8): stage A's closed-loop replay, stage B's
 * prior sentences and stage C's rounds are read by `parseFlownBlock`, each with the envelopes of its words — on the
 * fixtures the exports write.
 */
import { describe, expect, it } from "vitest";
import { parseTrainingPriorSample } from "../trainingPriorSample";
import { parseTrainingSample, TRAINING_SET_SPLITS } from "../trainingSample";
import { parseTrainingWindowSample } from "../trainingWindowSample";
import { stageASampleFile } from "./stageA";
import { stageBSampleFile } from "./stageB";
import { stageCSampleFile } from "./stageC";

describe("the flown block of every stage", () => {
  it("gives every flown sentence its envelopes: stage A's replays, stage B's samples, stage C's rounds", () => {
    const a = parseTrainingSample(stageASampleFile(), TRAINING_SET_SPLITS);
    const b = parseTrainingPriorSample(stageBSampleFile());
    const c = parseTrainingWindowSample(stageCSampleFile());
    if (!a.ok || !b.ok || !c.ok) throw new Error("a fixture does not parse");
    const replays = Object.values(a.value.flights[0].closedLoop).map((closed) => closed.replay.envelopes);
    const samples = b.value.flights.flatMap((flight) => flight.sentences.map((sentence) => sentence.envelopes));
    const rounds = c.value.windows.flatMap((window) => window.rounds.map((round) => round.envelopes));
    for (const [stage, envelopes] of [["A", replays], ["B", samples], ["C", rounds]] as const) {
      expect(envelopes.length, stage).toBeGreaterThan(0);
      for (const one of envelopes) expect(one?.heading.length, stage).toBeGreaterThan(0);
    }
  });

  it("refuses an envelope that ends past the flown track, in any stage", () => {
    const b = stageBSampleFile();
    const sentence = b.flights[0].prior[0];
    const band = sentence.envelopes.heading.find((one: Record<string, number>) => one.stopRow > one.firstRow);
    const longer = sentence.track.rows + 1 - band.stopRow;            // the band runs one row past the track, verdicts kept
    band.stopRow += longer;
    band.inside = [...band.inside, ...Array(longer).fill(1)];
    const parsed = parseTrainingPriorSample(b);
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) expect(parsed.problem).toContain("past the");
    const c = stageCSampleFile();
    const round = c.windows[0].rounds[0];
    const last = round.envelopes.heading.findLast((one: Record<string, number>) => one.stopRow > one.firstRow);
    const more = round.track.rows + 2 - last.stopRow;
    last.stopRow += more;
    last.inside = [...last.inside, ...Array(more).fill(1)];
    const windows = parseTrainingWindowSample(c);
    expect(windows.ok).toBe(false);
    if (!windows.ok) expect(windows.problem).toContain("past the");
  });
});
