/**
 * The live executor's answer on a sentence the prior said (`trainingPriorAutopilot.ts`) on the two answers the Python code
 * wrote (`stageB.ts`): refused by name unless it is the schema, the set's flight and the sentence asked for, then read as
 * stage A's answer — a segment of the word the bar shows, on the flight's clock.
 */
import { describe, expect, it } from "vitest";
import { parseTrainingPriorAutopilot, trainingPriorRequestBody, TRAINING_PRIOR_AUTOPILOT_SCHEMA } from "../trainingPriorAutopilot";
import { trainingAutopilotRequest, TRAINING_AUTOPILOT_SEGMENT_END } from "../trainingAutopilot";
import { trainingPriorFlightView } from "../trainingPriorSample";
import { priorRequestOf, stageBAnswers, stageBSample, stageBSelection } from "./stageB";

const sample = stageBSample();
const selection = stageBSelection(sample, 0);

describe("the answer", () => {
  it("reads a segment stopped at the next word and the last word flown to its outcome, on the flight's clock", () => {
    const [stopped, final] = stageBAnswers();
    const first = parseTrainingPriorAutopilot(stopped, priorRequestOf(stopped, selection), selection);
    if (!first.ok) throw new Error(first.problem);
    expect(first.value.segment.end).toBe(TRAINING_AUTOPILOT_SEGMENT_END);
    expect(first.value.flightKey).toBe(selection.flight.flightKey);
    expect(first.value.stored.horizontalM).toBeLessThan(0.08);
    const closed = selection.flight.closedLoop["4"];
    expect(first.value.track.tS[0]).toBe(closed.startS + first.value.segment.startCycle * closed.cycleS);
    const last = parseTrainingPriorAutopilot(final, priorRequestOf(final, selection), selection);
    if (!last.ok) throw new Error(last.problem);
    const sentence = sample.flights[0].sentences[0];
    expect(last.value.segment.end).toBe(sentence.outcome);
    expect(last.value.segment.endCycle).toBe(sentence.endCycle);
    expect(last.value.crossing === null).toBe(sentence.crossing === null);
  });

  it("refuses another schema — stage A's too — another flight or another sentence, by name", () => {
    const [raw] = stageBAnswers();
    const request = priorRequestOf(raw, selection);
    const refused = (changed: Record<string, unknown>, text: string, on = selection) => {
      const parsed = parseTrainingPriorAutopilot({ ...raw, ...changed }, request, on);
      expect(parsed.ok).toBe(false);
      if (!parsed.ok) expect(parsed.problem).toContain(text);
    };
    refused({ schema: "aeroviz-autopilot-segment-v9" }, `not one of ${TRAINING_PRIOR_AUTOPILOT_SCHEMA}`);
    refused({ flightKey: "other" }, "but KXXX");
    refused({ sentence: 1 }, "but 0 was asked for");
    refused({ sentence: "closedLoop" }, "but 0 was asked for");
    // the answer of the prior's sentence 0 on screen with sentence 1
    refused({}, "but 1 was asked for", stageBSelection(sample, 1));
  });
});

describe("the request", () => {
  it("names the set's own flight and the sentence on screen, not the derived flight", () => {
    const [raw] = stageBAnswers();
    const request = priorRequestOf(raw, selection);
    expect(request.flightKey).toContain("~prior-0");
    expect(trainingPriorRequestBody(request, selection)).toEqual({
      airport: "KXXX", setId: "fixture_set", flightKey: sample.flights[0].head.flightKey, sentence: 0, column: "heading", row: raw.segment.row,
    });
    const closed = stageBSelection(sample, "closedLoop");
    const pick = { rowIntervalS: 4, column: "heading" as const, row: 4, attempt: 0 };
    expect(trainingPriorRequestBody(trainingAutopilotRequest(closed, pick), closed).sentence).toBe("closedLoop");
    expect(trainingPriorFlightView(sample, sample.flights[0], 0)).toBe(selection.flight);
  });
});
