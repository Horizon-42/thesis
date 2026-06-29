import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

const { appState, setRightInspectorCollapsed } = vi.hoisted(() => ({
  appState: { rightInspectorCollapsed: false },
  setRightInspectorCollapsed: vi.fn(),
}));

vi.mock("../../context/AppContext", () => ({
  useApp: () => ({ ...appState, setRightInspectorCollapsed }),
}));

import WorkbenchRightInspector from "../WorkbenchRightInspector";

describe("WorkbenchRightInspector", () => {
  beforeEach(() => {
    appState.rightInspectorCollapsed = false;
    vi.clearAllMocks();
  });

  it("renders its children when expanded", () => {
    render(
      <WorkbenchRightInspector>
        <div>CAMERA_HUD</div>
      </WorkbenchRightInspector>,
    );
    expect(screen.getByText("CAMERA_HUD")).toBeTruthy();
    fireEvent.click(screen.getByLabelText("Collapse inspector"));
    expect(setRightInspectorCollapsed).toHaveBeenCalledWith(true);
  });

  it("hides its children when collapsed and offers an expand control", () => {
    appState.rightInspectorCollapsed = true;
    render(
      <WorkbenchRightInspector>
        <div>CAMERA_HUD</div>
      </WorkbenchRightInspector>,
    );
    expect(screen.queryByText("CAMERA_HUD")).toBeNull();
    fireEvent.click(screen.getByLabelText("Expand inspector"));
    expect(setRightInspectorCollapsed).toHaveBeenCalledWith(false);
  });
});
