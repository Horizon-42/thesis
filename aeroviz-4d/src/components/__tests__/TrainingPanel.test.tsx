import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

const { appState } = vi.hoisted(() => ({
  appState: { activeAirportCode: "KRDU" as string },
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState }),
}));

import TrainingPanel, { trainingIndexPath } from "../TrainingPanel";

describe("TrainingPanel (T1: the empty state)", () => {
  beforeEach(() => {
    appState.activeAirportCode = "KRDU";
  });

  it("says there is no export yet, for the ACTIVE airport", () => {
    render(<TrainingPanel />);
    expect(screen.getByText(/No Training export for KRDU yet/)).toBeTruthy();
  });

  it("names the exact path it reads, so the message is actionable", () => {
    render(<TrainingPanel />);
    expect(
      screen.getByText("data/airports/KRDU/training/index.json"),
    ).toBeTruthy();
  });

  it("names the command that writes it", () => {
    render(<TrainingPanel />);
    expect(screen.getByText(/run_ts\.py instruction_sample_export/)).toBeTruthy();
  });

  // AV5: vite does not watch public/data, so a directory created after the dev
  // server booted is served as the SPA fallback. Without this line the first
  // person to export hits a "received HTML" error about a file that is on disk.
  it("warns that the dev server must be restarted after the first export", () => {
    render(<TrainingPanel />);
    expect(screen.getByText(/restart the dev server/i)).toBeTruthy();
    expect(screen.getByText(/npm run dev/)).toBeTruthy();
  });

  it("follows the active airport rather than hardcoding one", () => {
    appState.activeAirportCode = "KSJC";
    render(<TrainingPanel />);
    expect(screen.getByText(/No Training export for KSJC yet/)).toBeTruthy();
    expect(
      screen.getByText("data/airports/KSJC/training/index.json"),
    ).toBeTruthy();
  });

  it("exports the path helper the reader will share with the message", () => {
    expect(trainingIndexPath("KSTL")).toBe(
      "data/airports/KSTL/training/index.json",
    );
  });
});
