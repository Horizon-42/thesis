/**
 * The live executor's answer on a window (`trainingWindowAutopilot.ts`) on the two answers the Python code wrote
 * (`stageC.ts`): refused by name unless it is the schema, the window and the round asked for, then read as stage A's
 * answer; the request names the set's window and round, not the derived flight.
 */
import { describe, expect, it } from "vitest";
import { parseTrainingWindowAutopilot, trainingWindowRequestBody, TRAINING_WINDOW_AUTOPILOT_SCHEMA } from "../trainingWindowAutopilot";
import { TRAINING_AUTOPILOT_SEGMENT_END } from "../trainingAutopilot";
import { stageCAnswers, stageCSample, stageCSelection, windowRequestOf } from "./stageC";

const sample = stageCSample();

describe("the answer", () => {
  it("reads the real window's word and the lost window's last word stopped at its loss", () => {
    const [real, lost] = stageCAnswers();
    for (const [answer, place] of [[real, 0], [lost, 1]] as const) {
      const selection = stageCSelection(sample, place);
      const parsed = parseTrainingWindowAutopilot(answer, windowRequestOf(answer, selection), selection);
      if (!parsed.ok) throw new Error(parsed.problem);
      expect(parsed.value.flightKey).toBe(selection.flight.flightKey);
      expect(parsed.value.stored.horizontalM).toBeLessThan(0.08);
    }
    expect(lost.segment.end).toBe(TRAINING_AUTOPILOT_SEGMENT_END);
    expect(lost.segment.endCycle).toBe(sample.windows[1].rounds[0].flown.tS.length * 2 - 2);
    expect(lost.windowEnd.loss.other).toBe(sample.windows[1].traffic[0].key);
  });

  it("is refused by name for another schema, window or round", () => {
    const [real] = stageCAnswers();
    const selection = stageCSelection(sample, 0);
    const request = windowRequestOf(real, selection);
    const refused = (answer: Record<string, any>, says: string) => {
      const parsed = parseTrainingWindowAutopilot(answer, request, selection);
      expect(parsed.ok).toBe(false);
      if (!parsed.ok) expect(parsed.problem).toContain(says);
    };
    refused({ ...real, schema: "aeroviz-autopilot-prior-segment-v1" }, TRAINING_WINDOW_AUTOPILOT_SCHEMA);
    refused({ ...real, window: 1 }, "window is 1");
    refused({ ...real, round: 0 }, "round is 0");
  });
});

describe("the request", () => {
  it("names the set's window and round, not the derived flight", () => {
    const [real] = stageCAnswers();
    const selection = stageCSelection(sample, 0);
    expect(trainingWindowRequestBody(windowRequestOf(real, selection), selection)).toMatchObject({
      setId: real.setId, window: 0, round: "start", row: real.segment.row,
    });
  });
});
