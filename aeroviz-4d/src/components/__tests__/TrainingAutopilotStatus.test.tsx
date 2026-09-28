/**
 * The live executor's line in the sentence bar: short enough to share the header's row with its buttons — the verdict,
 * a badly ended flight's tag, the two times; the word in the line only once the selection has moved off it; the word
 * and the full reading, or a refusal's reason, in its tooltip.
 */
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { TrainingAutopilotStatus } from "../TrainingAutopilotStatus";
import { parseTrainingSample, type TrainingSample } from "../../data/trainingSample";
import { parseTrainingAutopilot } from "../../data/trainingAutopilot";
import { VECTORED_KEY, mockSample } from "../../data/__tests__/trainingSample.fixture";
import { mockAutopilotAnswer, mockAutopilotRequest, mockSelection } from "../../data/__tests__/trainingAutopilot.fixture";

function sample(): TrainingSample {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

describe("the sentence bar's line", () => {
  it("is short: a badly ended flight a tag, a refusal's reason and the word in the tooltip — the word in the line only off the selection", () => {
    const set = sample();
    const asked = mockAutopilotRequest(set, VECTORED_KEY, "heading", 8);
    const selection = mockSelection(set, asked);
    const raw = mockAutopilotAnswer(set, asked);
    raw.end = { reason: "timeout", reachedSegmentEnd: false, offsetFromObserved: null, flownS: 8, crossing: null, refused: null };
    const answer = parseTrainingAutopilot(raw, asked, selection);
    if (!answer.ok) throw new Error(answer.problem);
    const view = { status: "ready", request: asked, segment: answer.value, playedAt: 1 } as const;
    const { container, unmount } = render(<TrainingAutopilotStatus selection={selection} view={view} named={false} />);
    const line = container.firstChild as HTMLElement;
    expect(line.textContent).toBe("Autopilot · in envelope · timed out · 8.00 s flown · computed 1.24 s");
    expect(line.title).toBe("heading 225°: in envelope; the flight did not get there within its time limit; 8.00 s flown, " +
      "computed 1.24 s");
    unmount();
    const { container: off, unmount: offDone } = render(<TrainingAutopilotStatus selection={selection} view={view} named />);
    expect(off.textContent).toBe("Autopilot · heading 225° · in envelope · timed out · 8.00 s flown · computed 1.24 s");
    offDone();
    render(<TrainingAutopilotStatus selection={selection} named={false}
      view={{ status: "failed", request: asked, problem: "the backend refused (400): 0 executor specs" }} />);
    expect(screen.getByText("Autopilot · not flown").title).toBe("heading 225° not flown — the backend refused (400): 0 executor specs");
  });
});
