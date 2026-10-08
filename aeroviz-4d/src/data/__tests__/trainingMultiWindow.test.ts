/**
 * Stage D's reading of a window of several commanded aircraft (frontend §5, F3) on the file stage C's export writes for a
 * synthetic window of two (`stageD.ts`): both aircraft in the order they join, each on its own flight clock; the anchor
 * silent from the row of the loss it answers (D144) and flying on; the loss written once, between the two commanded
 * aircraft; W the sum of the rewards; where each aircraft is at a window time; a D round's sentences in stage D's colour.
 */
import { describe, expect, it } from "vitest";
import {
  mostTrafficAtOnce, onAircraftClock, silentFromS, trackAt, windowAircraftAt, windowReward,
} from "../trainingWindowSample";
import { roundKind } from "../trainingSentenceKind";
import { stageDSample } from "./stageD";

describe("a window of two commanded aircraft", () => {
  const sample = stageDSample();
  const [window] = sample.windows;
  const [anchor, copy] = window.commanded;

  it("is read as stage D's: both aircraft in the order they join, the copy 32 s after the window's row 0", () => {
    expect(sample.stage).toBe("D");
    expect(window.commanded.map((aircraft) => aircraft.place)).toEqual([0, 1]);
    expect([anchor.joinS, copy.joinS]).toEqual([0, 32]);
    // the same flight copied: its flight clock is the anchor's, 32 s behind on the window's
    expect(anchor.clockS - copy.clockS).toBe(32);
  });

  it("the anchor answers the loss with the copy, once, and is silent from that row, flying on", () => {
    const [end] = window.rounds;
    const said = anchor.rounds[0];
    const answered = end.losses.filter((loss) => loss.answering.includes(anchor.datasetId));
    expect(answered).toHaveLength(1);
    expect(new Set(answered[0].aircraft)).toEqual(new Set([anchor.datasetId, copy.datasetId]));
    expect(said.outcome).toBe("lost_separation");
    expect(said.reward).toBe(0);
    expect(said.silentFromRow).not.toBeNull();
    const fromS = silentFromS(anchor, said, sample.model.rowIntervalS)!;
    expect(fromS).toBe(anchor.firstStepS + said.silentFromRow! * sample.model.rowIntervalS);
    expect(said.flown.tS[said.flown.tS.length - 1]).toBeGreaterThan(fromS);              // it flies on
    // the loss is at the row it went silent on (on the anchor's clock)
    expect(onAircraftClock(anchor, answered[0].timeS)).toBeCloseTo(fromS, 6);
  });

  it("W is the sum of the commanded aircraft's rewards", () => {
    expect(windowReward(window, 0)).toBeCloseTo(anchor.rounds[0].reward + copy.rounds[0].reward, 12);
  });

  it("finds each aircraft at a window time: a commanded one on its round's flown track, on its own clock", () => {
    const t = copy.rounds[0].flown.tS[1];                                    // on the copy's clock (it flew 3 rows)
    const windowS = t - copy.clockS;
    expect(windowAircraftAt(window, 0, copy.datasetId, windowS)).toEqual(trackAt(copy.rounds[0].flown, t));
    expect(windowAircraftAt(window, 0, "nobody", windowS)).toBeNull();
  });

  it("finds a commanded aircraft from the row it joins: on its observed rows before its first predicted step", () => {
    expect(windowAircraftAt(window, 0, copy.datasetId, copy.joinS - 4)).toBeNull();          // not yet joined
    const at = copy.joinS + 8;                                                                // joined, still observed
    expect(onAircraftClock(copy, at)).toBeLessThan(copy.firstStepS);
    expect(windowAircraftAt(window, 0, copy.datasetId, at)).toEqual(trackAt(copy.head.observed, onAircraftClock(copy, at)));
    // so both aircraft of every loss of the round are found at its time (the anchor's loss is at the copy's join)
    for (const loss of window.rounds[0].losses) {
      for (const key of loss.aircraft) expect(windowAircraftAt(window, 0, key, loss.timeS)).not.toBeNull();
    }
  });

  it("counts the most recorded aircraft in the air at one time", () => {
    expect(window.traffic).toEqual([]);                                          // the synthetic window has no other aircraft
    expect(mostTrafficAtOnce(window)).toBe(0);
    const track = (from: number, to: number) => ({ ...window.commanded[0].rounds[0].flown, tS: [from, to] });
    const busy = { ...window, traffic: [track(0, 10), track(5, 20), track(12, 30), track(25, 40)] as any };
    expect(mostTrafficAtOnce(busy)).toBe(2);
  });

  it("a stage D round's sentences are stage D's model's; its start, a stage C round, is post-trained", () => {
    expect(roundKind(2, sample.model.start, "D")).toBe("multi");
    expect(roundKind("start", { campaign: "c", round: 5, checkpointSha256: "0".repeat(64) }, "D")).toBe("postTrained");
    expect(roundKind(2, null, "C")).toBe("postTrained");
  });
});
