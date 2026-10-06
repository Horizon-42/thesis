/**
 * The live executor's line in the sentence bar: short enough to share the header's row with its buttons — how the segment
 * ended, the two times; the word in the line only once the selection has moved off it; the word and the full reading, or a
 * refusal's reason, in its tooltip.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";

import { OPENING_NOTE_AFTER_MS, TrainingAutopilotStatus } from "../TrainingAutopilotStatus";
import { parseTrainingAutopilot } from "../../data/trainingAutopilot";
import { requestOf, stageAAnswers, stageASelection } from "../../data/__tests__/stageA";

const selection = stageASelection();

function ready(which: number) {
  const raw = stageAAnswers()[which];
  const request = requestOf(raw);
  const answer = parseTrainingAutopilot(raw, request, selection);
  if (!answer.ok) throw new Error(answer.problem);
  return { request, view: { status: "ready", request, segment: answer.value, playedAt: 1 } as const };
}

describe("the sentence bar's line", () => {
  it("says a segment flown to the next word, the two times, and the word only off the selection", () => {
    const { view } = ready(0);
    const { container, unmount } = render(<TrainingAutopilotStatus selection={selection} view={view} named={false} />);
    const line = container.firstChild as HTMLElement;
    expect(line.textContent).toBe("Autopilot · flown to the next word · 8.00 s flown · computed 0 ms");
    expect(line.title).toContain("the flight reached the point where the next heading word is said");
    expect(line.title).toContain("0.000 m horizontally and 0.000 m vertically from the exported flown states");
    unmount();
    const { container: off } = render(<TrainingAutopilotStatus selection={selection} view={view} named />);
    expect(off.textContent).toMatch(/^Autopilot · heading [+−]?\d+° · flown to the next word/);
  });

  it("tags a word flown on to the flight's outcome with the judge's outcome, and names a correction", () => {
    const { view } = ready(1);
    const { container } = render(<TrainingAutopilotStatus selection={selection} view={view} named={false} />);
    expect(container.textContent).toContain("unstable at minimums");
    expect((container.firstChild as HTMLElement).title).toContain("(a correction)");
    expect((container.firstChild as HTMLElement).title).toContain("15.5 m right of the centreline, 41.8 m above the threshold");
  });

  it("says flying, and gives a refusal's reason in the tooltip", () => {
    const { request } = ready(0);
    const { unmount } = render(<TrainingAutopilotStatus selection={selection} view={{ status: "flying", request }} named={false} />);
    expect(screen.getByRole("status").textContent).toBe("Autopilot · flying …");
    unmount();
    render(<TrainingAutopilotStatus selection={selection} named={false}
      view={{ status: "failed", request, problem: "the backend refused (404): no flight" }} />);
    expect(screen.getByText("Autopilot · not flown").title).toMatch(/not flown — the backend refused \(404\): no flight$/);
  });
});

describe("an answer that takes long (vocabulary A43)", () => {
  afterEach(() => vi.useRealTimers());

  it("says that the backend is opening the set once the request has flown for two seconds, and no sooner", () => {
    vi.useFakeTimers();
    const { request } = ready(0);
    render(<TrainingAutopilotStatus selection={selection} view={{ status: "flying", request }} named={false} />);
    expect(screen.getByRole("status").textContent).toBe("Autopilot · flying …");
    act(() => { vi.advanceTimersByTime(OPENING_NOTE_AFTER_MS - 1); });
    expect(screen.getByRole("status").textContent).toBe("Autopilot · flying …");
    act(() => { vi.advanceTimersByTime(1); });
    expect(screen.getByRole("status").textContent).toBe("Autopilot · flying … the backend is opening the set");
    expect(screen.getByRole("status").title).toContain("the backend is opening the set");
  });

  it("starts the wait again for a new request at once, and says nothing once the answer is there", () => {
    vi.useFakeTimers();
    const first = ready(0);
    const { rerender } = render(<TrainingAutopilotStatus selection={selection} view={{ status: "flying", request: first.request }} named={false} />);
    act(() => { vi.advanceTimersByTime(OPENING_NOTE_AFTER_MS); });
    expect(screen.getByRole("status").textContent).toContain("opening the set");
    const another = { ...first.request, row: first.request.row + 1 };
    rerender(<TrainingAutopilotStatus selection={selection} view={{ status: "flying", request: another }} named={false} />);
    expect(screen.getByRole("status").textContent).toBe("Autopilot · flying …");          // the first commit, no stale note
    act(() => { vi.advanceTimersByTime(OPENING_NOTE_AFTER_MS - 500); });
    const third = { ...first.request, row: first.request.row + 2 };                     // switched at 1.5 s: the timer restarts
    rerender(<TrainingAutopilotStatus selection={selection} view={{ status: "flying", request: third }} named={false} />);
    act(() => { vi.advanceTimersByTime(600); });                                         // 2.1 s after the second request
    expect(screen.getByRole("status").textContent).toBe("Autopilot · flying …");
    act(() => { vi.advanceTimersByTime(OPENING_NOTE_AFTER_MS); });
    expect(screen.getByRole("status").textContent).toContain("opening the set");
    rerender(<TrainingAutopilotStatus selection={selection} view={first.view} named={false} />);
    act(() => { vi.advanceTimersByTime(10 * OPENING_NOTE_AFTER_MS); });
    expect(document.body.textContent).not.toContain("opening the set");
  });
});
