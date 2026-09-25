/**
 * The live executor's card in the Training panel: what to do before a word is flown, a refusal with its reason, and a
 * flown segment written out — with "Fly again" (a new attempt at the same pick) and "Replay in 3D".
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState, setTrainingAutopilot, setTrainingPick } = vi.hoisted(() => ({
  appState: { trainingAutopilot: null as unknown, trainingPick: null as unknown },
  setTrainingAutopilot: vi.fn(),
  setTrainingPick: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setTrainingAutopilot, setTrainingPick }),
}));

import TrainingAutopilotCard, { autopilotTimingText, formatDuration } from "../TrainingAutopilotCard";
import { parseTrainingSample, type TrainingSample } from "../../data/trainingSample";
import { nextPick, parseTrainingAutopilot } from "../../data/trainingAutopilot";
import { VECTORED_KEY, mockSample } from "../../data/__tests__/trainingSample.fixture";
import { mockAutopilotAnswer, mockAutopilotRequest } from "../../data/__tests__/trainingAutopilot.fixture";

function sample(): TrainingSample {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

describe("TrainingAutopilotCard", () => {
  let set: TrainingSample;

  beforeEach(() => {
    set = sample();
    appState.trainingAutopilot = null;
    appState.trainingPick = null;
    setTrainingAutopilot.mockClear();
    setTrainingPick.mockClear();
  });

  const card = () => render(
    <TrainingAutopilotCard flight={set.flights[0]} vocabulary={set.vocabulary} candidates={set.candidates} />,
  );

  it("says what to do before a word is flown", () => {
    card();
    expect(screen.getByText(/Select a word in the sentence bar/).textContent).toMatch(/press ▶ Fly this segment above it/);
  });

  it("names the refusal and offers to fly again", () => {
    appState.trainingPick = nextPick(null, VECTORED_KEY, "heading", 8);
    appState.trainingAutopilot = { status: "failed", request: mockAutopilotRequest(set, VECTORED_KEY, "heading", 8),
      problem: "the backend refused (400): 0 executor specs" };
    card();
    expect(screen.getByRole("alert").textContent).toMatch(/did not fly heading 225° from step 8.*0 executor specs/);
    fireEvent.click(screen.getByText("Fly again"));
    expect(setTrainingPick).toHaveBeenLastCalledWith({ flightKey: VECTORED_KEY, column: "heading", row: 8, attempt: 1 });
  });

  it("writes a flown segment out, and replays it in 3D without asking again", () => {
    const request = mockAutopilotRequest(set, VECTORED_KEY, "heading", 8);
    const answer = parseTrainingAutopilot(mockAutopilotAnswer(set, request), request, set);
    if (!answer.ok) throw new Error(answer.problem);
    const view = { status: "ready", request, segment: answer.value, playedAt: 1, roundTripS: 0.5 };
    appState.trainingAutopilot = view;
    card();
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
    // a flight that ended well says nothing more above the fold
    expect(document.querySelector(".training-autopilot-ended")).toBeNull();
    fireEvent.click(screen.getByText("Replay in 3D"));
    const replayed = setTrainingAutopilot.mock.calls[0][0];
    expect(replayed.segment).toBe(answer.value);
    expect(replayed.playedAt).toBeGreaterThan(1);
    expect(setTrainingPick).not.toHaveBeenCalled();
  });
});

describe("the times written out", () => {
  it("chooses the unit after rounding", () => {
    expect(formatDuration(0.0004)).toBe("0 ms");
    expect(formatDuration(0.305)).toBe("305 ms");
    expect(formatDuration(0.9996)).toBe("1.00 s");
    expect(formatDuration(9.996)).toBe("10.0 s");
    expect(formatDuration(64)).toBe("64.0 s");
  });

  it("says a flight kept from an earlier request, and a wait behind the flight before it", () => {
    const set = sample();
    const request = mockAutopilotRequest(set, VECTORED_KEY, "heading", 8);
    const answer = parseTrainingAutopilot(mockAutopilotAnswer(set, request), request, set);
    if (!answer.ok) throw new Error(answer.problem);
    const kept = { ...answer.value, timing: { ...answer.value.timing, flightKept: true, openS: 0.0002, waitS: 1.5 } };
    expect(autopilotTimingText(kept, 2)).toMatch(/flight kept from an earlier request \(0 ms\) · .* · waited 1\.50 s for the flight before it · round trip 2\.00 s$/);
  });
});

describe("a segment outside its envelope, or a flight that ended badly", () => {
  it("turns the verdict red and says how it ended", () => {
    const set = sample();
    const request = mockAutopilotRequest(set, VECTORED_KEY, "heading", 8);
    const raw = mockAutopilotAnswer(set, request);
    raw.word.status = "outside";
    raw.word.checks[0].ok = false;
    raw.word.checks[0].inside = 1;
    raw.word.heading.inside[1] = 0;
    raw.end = { reason: "timeout", reachedSegmentEnd: false, offsetFromObserved: null, flownS: 8, crossing: null, refused: null };
    const answer = parseTrainingAutopilot(raw, request, set);
    if (!answer.ok) throw new Error(answer.problem);
    appState.trainingAutopilot = { status: "ready", request, segment: answer.value, playedAt: 1, roundTripS: 0.5 };
    render(<TrainingAutopilotCard flight={set.flights[0]} vocabulary={set.vocabulary} candidates={set.candidates} />);
    const verdict = document.querySelector(".training-autopilot-verdict") as HTMLElement;
    expect(verdict.textContent).toBe("✗ outside its envelope");
    expect(verdict.style.color).toBe("rgb(255, 45, 45)");
    expect(document.querySelector(".training-autopilot-ended")!.textContent).toBe("The flight did not get there within its time limit.");
  });
});

