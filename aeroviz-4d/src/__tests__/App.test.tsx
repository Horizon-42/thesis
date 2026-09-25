/**
 * The app shell does not follow the Training cursor: the cursor moves on every mousemove over a chart, and only its
 * readers — the sentence bar and the 3D scene's leaf (`TrainingScene`) — may re-render with it. A hook that reads the
 * cursor put back into `FlightApp` would re-render the whole workbench on every hover; this test fails then.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, waitFor } from "@testing-library/react";

const { counts, cursor, nothing, layer } = vi.hoisted(() => ({
  counts: { shell: 0, scene: 0 },
  cursor: { set: null as null | ((atS: number) => void) },
  nothing: () => ({ default: () => null }),
  layer: () => ({ flightIds: [], flightSummaries: {}, error: null, warning: null, observedEvaluation: null }),
}));

vi.mock("../components/CesiumViewer", nothing);
vi.mock("../components/WorkbenchShell", nothing);
vi.mock("../components/TrainingSentenceBar", nothing);
vi.mock("../components/WorkbenchLeftDock", nothing);
vi.mock("../components/AirportLocalTerrainDemoPage", nothing);
vi.mock("../components/ChartAnnotatedPage", nothing);
vi.mock("../components/AirportLocalTerrainAlert", nothing);
vi.mock("../components/HUD", nothing);
vi.mock("../components/WorkbenchRightInspector", nothing);
vi.mock("../components/WorkbenchBottomBar", nothing);
vi.mock("../components/ProcedureDetailsPage", nothing);
vi.mock("../components/ProcedureAnnotationPopup", nothing);
vi.mock("../components/ProcedurePanel", nothing);
vi.mock("../components/ApproachViewPanel", nothing);
// the 3D scene's leaf: a reader of the real cursor
vi.mock("../components/TrainingScene", async () => {
  const { useTrainingCursor } = await import("../context/AppContext");
  return {
    default: () => {
      cursor.set = useTrainingCursor().setTrainingCursorS;
      counts.scene += 1;
      return null;
    },
  };
});
vi.mock("../hooks/useObservedTrajectoryLayer", () => ({ useObservedTrajectoryLayer: layer }));
vi.mock("../hooks/useComparisonTrajectoryLayer", () => ({ useComparisonTrajectoryLayer: layer }));
// called once per render of the shell (`FlightApp`)
vi.mock("../hooks/useLandingsManifest", () => ({
  useLandingsManifest: () => {
    counts.shell += 1;
    return { manifest: null, status: "loading", error: null };
  },
}));

import App from "../App";
import { AppProvider } from "../context/AppContext";

describe("App", () => {
  beforeEach(() => {
    counts.shell = 0;
    counts.scene = 0;
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true, headers: { get: () => "application/json" },
      text: async () => JSON.stringify({ defaultAirport: "KRDU", airports: [{ code: "KRDU", name: "Raleigh-Durham", lat: 35.88, lon: -78.79 }] }),
    })));
  });

  afterEach(() => vi.unstubAllGlobals());

  it("re-renders the Training scene's leaf, not the shell, when the cursor moves", async () => {
    render(<AppProvider><App /></AppProvider>);
    await waitFor(() => expect(cursor.set).not.toBeNull());
    await act(async () => undefined);
    const before = { ...counts };
    act(() => cursor.set!(12));
    act(() => cursor.set!(24));
    expect(counts.scene).toBe(before.scene + 2);
    expect(counts.shell).toBe(before.shell);
  });
});
