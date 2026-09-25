/**
 * The live executor's card in the Training panel: what to do before a word is selected, the flight in progress, a
 * refusal with its reason, and a flown segment written out — with "Fly again" (a new request) and "Replay in 3D".
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState, setTrainingAutopilot } = vi.hoisted(() => ({
  appState: { trainingAutopilot: null as unknown },
  setTrainingAutopilot: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setTrainingAutopilot }),
}));

import TrainingAutopilotCard from "../TrainingAutopilotCard";
import { parseTrainingSample, type TrainingSample } from "../../data/trainingSample";
import { parseTrainingAutopilot } from "../../data/trainingAutopilot";
import { VECTORED_KEY, mockSample } from "../../data/__tests__/trainingSample.fixture";
import { mockAutopilotAnswer, mockAutopilotRequest } from "../../data/__tests__/trainingAutopilot.fixture";

function sample(): TrainingSample {
  const parsed = parseTrainingSample(mockSample());
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

describe("TrainingAutopilotCard", () => {
  let set: TrainingSample;
  const flyAgain = vi.fn();

  beforeEach(() => {
    set = sample();
    appState.trainingAutopilot = null;
    setTrainingAutopilot.mockClear();
    flyAgain.mockClear();
  });

  const card = (selected = true) => render(
    <TrainingAutopilotCard flight={set.flights[0]} vocabulary={set.vocabulary} candidates={set.candidates}
      selected={selected} flyAgain={flyAgain} />,
  );

  it("says what to do before a word is selected", () => {
    card(false);
    expect(screen.getByText(/Click a word.s band in the sentence bar/)).toBeTruthy();
  });

  it("names the refusal and offers to fly again", () => {
    appState.trainingAutopilot = { status: "failed", request: mockAutopilotRequest(set, VECTORED_KEY, "heading", 8),
      problem: "the backend refused (400): 0 executor specs" };
    card();
    expect(screen.getByRole("alert").textContent).toMatch(/did not fly heading 225° from step 8.*0 executor specs/);
    fireEvent.click(screen.getByText("Fly again"));
    expect(flyAgain).toHaveBeenCalledTimes(1);
  });

  it("writes a flown segment out, and replays it in 3D without asking again", () => {
    const request = mockAutopilotRequest(set, VECTORED_KEY, "heading", 8);
    const answer = parseTrainingAutopilot(mockAutopilotAnswer(set, request), request, set);
    if (!answer.ok) throw new Error(answer.problem);
    const view = { status: "ready", request, segment: answer.value, playedAt: 1 };
    appState.trainingAutopilot = view;
    card();
    const shown = screen.getByLabelText("The autopilot, live").textContent!;
    expect(shown).toMatch(/heading 225°, steps 8–10, flown on to step 12/);
    expect(shown).toMatch(/reached the point a lead after the next heading word was said, where this word's band ends after 8 s \(the observed aircraft: 8 s\)/);
    expect(shown).toMatch(/120 m from the observed aircraft, 8 m below it, 1\.5 m\/s faster/);
    expect(shown).toMatch(/The word: inside/);
    expect(shown).toMatch(/✓ track within the tolerance of the word \(2\/2 rows\)/);
    expect(shown).toMatch(/limits bound: bank rate 2/);
    expect(shown).toMatch(/spec 999999999999 .* executor code 888888888888 · words said on the track clock · computed in 0\.42 s at/);
    fireEvent.click(screen.getByText("Replay in 3D"));
    const replayed = setTrainingAutopilot.mock.calls[0][0];
    expect(replayed.segment).toBe(answer.value);
    expect(replayed.playedAt).toBeGreaterThan(1);
    expect(flyAgain).not.toHaveBeenCalled();
  });
});
