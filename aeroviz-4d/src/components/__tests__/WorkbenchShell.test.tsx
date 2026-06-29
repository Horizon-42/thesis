import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState, setPresentationMode } = vi.hoisted(() => ({
  appState: { presentationMode: false },
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

  it("does not listen for Escape when not in presentation mode", () => {
    renderShell();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(setPresentationMode).not.toHaveBeenCalled();
  });
});
