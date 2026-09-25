/**
 * The live executor's card in the Training panel: nothing before a word is flown, a refusal with its reason, and a
 * flown segment written out — the word it flew, its verdict first, the two times, the rest in Details — with "Replay
 * in 3D", which asks the backend nothing; a refusal's "Fly again" picks the word anew.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState, replayTrainingAutopilot, setTrainingPick } = vi.hoisted(() => ({
  appState: { trainingAutopilot: null as unknown, trainingSelection: null as unknown, trainingPick: null as unknown },
  replayTrainingAutopilot: vi.fn(),
  setTrainingPick: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, replayTrainingAutopilot, setTrainingPick }),
}));

import TrainingAutopilotCard from "../TrainingAutopilotCard";
import { parseTrainingSample, type TrainingSample } from "../../data/trainingSample";
import { parseTrainingAutopilot } from "../../data/trainingAutopilot";
import { formatElapsed } from "../../data/trainingText";
import { VECTORED_KEY, mockSample } from "../../data/__tests__/trainingSample.fixture";
import { failedAnswer, mockAutopilotAnswer, mockAutopilotRequest, mockSelection } from "../../data/__tests__/trainingAutopilot.fixture";

function sample(): TrainingSample {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

describe("TrainingAutopilotCard", () => {
  let set: TrainingSample;
  const request = () => mockAutopilotRequest(set, VECTORED_KEY, "heading", 8);

  beforeEach(() => {
    set = sample();
    appState.trainingAutopilot = null;
    appState.trainingSelection = mockSelection(set, request());
    appState.trainingPick = null;
    replayTrainingAutopilot.mockClear();
    setTrainingPick.mockClear();
  });

  it("shows nothing before a word is flown", () => {
    const { container } = render(<TrainingAutopilotCard />);
    expect(container.innerHTML).toBe("");
  });

  it("names the refusal", () => {
    appState.trainingAutopilot = { status: "failed", request: request(), problem: "the backend refused (400): 0 executor specs" };
    render(<TrainingAutopilotCard />);
    expect(screen.getByRole("alert").textContent).toMatch(/did not fly heading 225° from step 8.*0 executor specs/);
  });

  it("flies a refused word again — the word asked for, whichever the bar has selected since", () => {
    appState.trainingAutopilot = { status: "failed", request: request(), problem: "the backend did not answer" };
    appState.trainingPick = { column: "heading", row: 8, attempt: 2 };
    const { rerender } = render(<TrainingAutopilotCard />);
    fireEvent.click(screen.getByRole("button", { name: "Fly again" }));
    expect(setTrainingPick).toHaveBeenCalledWith({ column: "heading", row: 8, attempt: 3 });
    // the pick has moved on to another word since: that word is not flown, the refused one is
    setTrainingPick.mockClear();
    appState.trainingPick = { column: "altitude", row: 20, attempt: 0 };
    rerender(<TrainingAutopilotCard />);
    fireEvent.click(screen.getByRole("button", { name: "Fly again" }));
    expect(setTrainingPick).toHaveBeenCalledWith({ column: "heading", row: 8, attempt: 0 });
  });

  it("says how a flight that failed in its first cycle ended, and offers nothing to replay", () => {
    const asked = request();
    const answer = parseTrainingAutopilot(failedAnswer(mockAutopilotAnswer(set, asked), 1), asked, mockSelection(set, asked));
    if (!answer.ok) throw new Error(answer.problem);
    appState.trainingAutopilot = { status: "ready", request: asked, segment: answer.value, playedAt: 1, roundTripS: 0.5 };
    render(<TrainingAutopilotCard />);
    expect(document.querySelector(".training-autopilot-ended")!.textContent).toMatch(/^The flight left the dynamics/);
    expect(screen.queryByRole("button", { name: "Replay in 3D" })).toBeNull();
  });

  it("shows nothing for another flight's answer", () => {
    appState.trainingAutopilot = { status: "failed", request: { ...request(), setId: "another_set" }, problem: "refused" };
    const { container } = render(<TrainingAutopilotCard />);
    expect(container.innerHTML).toBe("");
  });

  it("writes a flown segment out, and replays it in 3D without asking again", () => {
    const asked = request();
    const answer = parseTrainingAutopilot(mockAutopilotAnswer(set, asked), asked, mockSelection(set, asked));
    if (!answer.ok) throw new Error(answer.problem);
    appState.trainingAutopilot = { status: "ready", request: asked, segment: answer.value, playedAt: 1, roundTripS: 0.5 };
    render(<TrainingAutopilotCard />);
    const shown = screen.getByLabelText("The autopilot, live").textContent!;
    expect(document.querySelector(".training-autopilot-title")!.textContent).toBe("heading 225° · steps 8–10, flown on to step 12");
    // the verdict first, in the segment's colour
    const verdict = document.querySelector(".training-autopilot-verdict") as HTMLElement;
    expect(verdict.textContent).toBe("✓ inside its envelope");
    expect(verdict.style.color).toBe("rgb(37, 99, 235)");
    // the two times, side by side: the simulated flight and the computation
    expect(screen.getByLabelText("Simulated flight time").textContent).toBe("Simulated flight8.00 sobserved 8.00 s");
    expect(screen.getByLabelText("Computation time").textContent).toBe("Computed in1.24 sround trip 500 ms");
    expect(shown).toMatch(/✓ track within the tolerance of the word \(2\/2 rows\)/);
    // the rest folded into Details
    const details = document.querySelector("details.training-autopilot-details")!.textContent!;
    expect(details).toMatch(/reached the point a lead after the next heading word was said, where this word's band ends\./);
    expect(details).toMatch(/120 m from the observed aircraft, 8 m below it, 1\.5 m\/s faster/);
    expect(details).toMatch(/limits bound: bank rate 2/);
    expect(details).toMatch(/executor 210 ms \(8 cycles of 1 s computed\) · judge 60 ms · flight rebuilt 910 ms · set and spec 20 ms · segment set up 10 ms · answer 30 ms · round trip 500 ms/);
    expect(details).toMatch(/spec 999999999999 .* executor code 888888888888 · words said on the track clock · computed at/);
    // a flight that ended well says nothing more above the fold; flying it anew is the sentence bar's button
    expect(document.querySelector(".training-autopilot-ended")).toBeNull();
    expect(screen.queryByText("Fly again")).toBeNull();
    fireEvent.click(screen.getByText("Replay in 3D"));
    expect(replayTrainingAutopilot).toHaveBeenCalledTimes(1);
  });

  it("says a flight kept from an earlier request, and a wait behind the flight before it", () => {
    const asked = request();
    const answer = parseTrainingAutopilot(mockAutopilotAnswer(set, asked), asked, mockSelection(set, asked));
    if (!answer.ok) throw new Error(answer.problem);
    const segment = { ...answer.value, timing: { ...answer.value.timing, flightKept: true, openS: 0.0002, waitS: 1.5 } };
    appState.trainingAutopilot = { status: "ready", request: asked, segment, playedAt: 1, roundTripS: 2 };
    render(<TrainingAutopilotCard />);
    expect(document.querySelector("details.training-autopilot-details")!.textContent).toMatch(
      /flight kept from an earlier request \(0 ms\) · .* · waited 1\.50 s for the flight before it · round trip 2\.00 s\./);
  });

  it("turns the verdict red and says how a flight that ended badly ended", () => {
    const asked = request();
    const raw = mockAutopilotAnswer(set, asked);
    raw.word.status = "outside";
    raw.word.checks[0].ok = false;
    raw.word.checks[0].inside = 1;
    raw.word.heading.inside[1] = 0;
    raw.end = { reason: "timeout", reachedSegmentEnd: false, offsetFromObserved: null, flownS: 8, crossing: null, refused: null };
    const answer = parseTrainingAutopilot(raw, asked, mockSelection(set, asked));
    if (!answer.ok) throw new Error(answer.problem);
    appState.trainingAutopilot = { status: "ready", request: asked, segment: answer.value, playedAt: 1, roundTripS: 0.5 };
    render(<TrainingAutopilotCard />);
    const verdict = document.querySelector(".training-autopilot-verdict") as HTMLElement;
    expect(verdict.textContent).toBe("✗ outside its envelope");
    expect(verdict.style.color).toBe("rgb(255, 45, 45)");
    expect(document.querySelector(".training-autopilot-ended")!.textContent).toBe("The flight did not get there within its time limit.");
  });
});

describe("the times written out", () => {
  it("chooses the unit after rounding", () => {
    expect(formatElapsed(0.0004)).toBe("0 ms");
    expect(formatElapsed(0.305)).toBe("305 ms");
    expect(formatElapsed(0.9996)).toBe("1.00 s");
    expect(formatElapsed(9.996)).toBe("10.0 s");
    expect(formatElapsed(64)).toBe("64.0 s");
  });
});
