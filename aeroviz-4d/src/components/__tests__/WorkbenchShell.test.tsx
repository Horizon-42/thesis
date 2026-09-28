import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState, setPresentationMode } = vi.hoisted(() => ({
  appState: { presentationMode: false, mode: "observe", viewer: null as unknown },
  setPresentationMode: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setPresentationMode }),
}));

// Stub the chrome the shell composes so this test is about the shell itself.
vi.mock("../WorkbenchTopBar", () => ({ default: () => <div>TOPBAR</div> }));
vi.mock("../LayersDrawer", () => ({ default: () => <div>DRAWER</div> }));

import WorkbenchShell from "../WorkbenchShell";

function renderShell() {
  const { container } = render(
    <WorkbenchShell left={<div>LEFT</div>} right={<div>RIGHT</div>} bottom={<div>BOTTOM</div>}>
      <div>OVERLAY</div>
    </WorkbenchShell>,
  );
  return container.querySelector(".workbench") as HTMLElement;
}

describe("WorkbenchShell", () => {
  beforeEach(() => {
    appState.presentationMode = false;
    appState.mode = "observe";
    appState.viewer = null;
    vi.clearAllMocks();
  });

  it("places the slot content and chrome by default (no presentation class, no exit button)", () => {
    const root = renderShell();
    expect(root.classList.contains("workbench--presentation")).toBe(false);
    expect(screen.getByText("TOPBAR")).toBeTruthy();
    expect(screen.getByText("LEFT")).toBeTruthy();
    expect(screen.getByText("RIGHT")).toBeTruthy();
    expect(screen.getByText("BOTTOM")).toBeTruthy();
    expect(screen.getByText("OVERLAY")).toBeTruthy();
    expect(screen.queryByText("Exit presentation (Esc)")).toBeNull();
  });

  it("applies the presentation class and shows an exit affordance", () => {
    appState.presentationMode = true;
    const root = renderShell();
    expect(root.classList.contains("workbench--presentation")).toBe(true);

    fireEvent.click(screen.getByText("Exit presentation (Esc)"));
    expect(setPresentationMode).toHaveBeenCalledWith(false);
  });

  it("exits presentation mode on Escape", () => {
    appState.presentationMode = true;
    renderShell();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(setPresentationMode).toHaveBeenCalledWith(false);
  });

  it("hides Cesium's clock console in Training only — the viewer laid out again after each switch — and puts it back on leaving", () => {
    // what the viewer's layout would read: the class at the moment it is asked to lay itself out
    const training = () => document.body.classList.contains("workbench-training-active");
    const laidOut: boolean[] = [];
    appState.viewer = { forceResize: vi.fn(() => laidOut.push(training())) };
    const shell = () => <WorkbenchShell><div /></WorkbenchShell>;
    const { rerender, unmount } = render(shell());
    expect(laidOut).toEqual([false]);
    appState.mode = "training";
    rerender(shell());
    expect(laidOut).toEqual([false, true]);
    appState.mode = "observe";
    rerender(shell());
    expect(laidOut).toEqual([false, true, false]);
    appState.mode = "training";
    rerender(shell());
    unmount();
    expect(training()).toBe(false);
  });

  it("does not listen for Escape when not in presentation mode", () => {
    renderShell();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(setPresentationMode).not.toHaveBeenCalled();
  });
});
