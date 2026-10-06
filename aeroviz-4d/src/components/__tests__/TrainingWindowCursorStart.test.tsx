/**
 * D129: the Training cursor starts at the window's row 0 when a window comes on screen (the other aircraft show from the
 * start), stays where the user then puts it, and starts again at the next window's row 0. With the real provider.
 */
import { describe, expect, it, vi } from "vitest";
import { act, render } from "@testing-library/react";

vi.mock("../../utils/fetchJson", () => ({
  fetchJson: vi.fn().mockResolvedValue({ defaultAirport: "KXXX", airports: [{ code: "KXXX", name: "Test field", lat: 35, lon: -78 }] }),
}));

import { AppProvider, useApp, useTrainingCursor } from "../../context/AppContext";
import { WindowCursorStart } from "../training/TrainingWindowSession";
import { stageCSample, stageCSelection } from "../../data/__tests__/stageC";

describe("the cursor of a window", () => {
  it("starts at the window's row 0, then is the user's, and starts again with another window", () => {
    const sample = stageCSample();
    let app!: ReturnType<typeof useApp>;
    let cursor!: ReturnType<typeof useTrainingCursor>;
    // the synthetic flight's window starts at its own row 0 (row0S 0): the window's row 0 is given here (a real flight's
    // closed-loop sentence starts after its row 0)
    const first = stageCSelection(sample, 0);
    const third = stageCSelection(sample, 2);
    function Harness({ flightKey, row0S }: { flightKey: string; row0S: number | null }) {
      app = useApp();
      cursor = useTrainingCursor();
      return row0S === null ? null : <WindowCursorStart flightKey={flightKey} row0S={row0S} />;
    }
    const { rerender } = render(<AppProvider><Harness flightKey={first.flight.flightKey} row0S={null} /></AppProvider>);
    act(() => app.setTrainingSelection(first));
    expect(cursor.trainingCursorS).toBe(0);                            // a flight on screen: its row 0
    rerender(<AppProvider><Harness flightKey={first.flight.flightKey} row0S={6} /></AppProvider>);
    expect(cursor.trainingCursorS).toBe(6);
    act(() => cursor.setTrainingCursorS(99));
    rerender(<AppProvider><Harness flightKey={first.flight.flightKey} row0S={6} /></AppProvider>);
    expect(cursor.trainingCursorS).toBe(99);                           // the user's, not set again
    // the next window's leaf before the session has put that window on screen: nothing written on the flight before
    rerender(<AppProvider><Harness flightKey={third.flight.flightKey} row0S={10} /></AppProvider>);
    expect(cursor.trainingCursorS).toBe(99);
    act(() => app.setTrainingSelection(third));
    rerender(<AppProvider><Harness flightKey={third.flight.flightKey} row0S={10} /></AppProvider>);
    expect(cursor.trainingCursorS).toBe(10);
  });
});
