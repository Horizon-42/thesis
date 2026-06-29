/**
 * WorkbenchRightInspector.tsx
 * ---------------------------
 * The collapsible right dock ("what am I looking at"): the camera/HUD controls
 * today, and the natural home for selected-flight detail. Collapsing it frees the
 * globe without losing state. The live-state readout (PilotRealtimeStatePanel) keeps
 * portaling itself into the overlay host, so it is unaffected by this wrapper.
 */

import type { ReactNode } from "react";
import { useApp } from "../context/AppContext";

export default function WorkbenchRightInspector({ children }: { children: ReactNode }) {
  const { rightInspectorCollapsed, setRightInspectorCollapsed } = useApp();
  return (
    <aside
      className={`workbench-right-inspector${rightInspectorCollapsed ? " collapsed" : ""}`}
      aria-label="Inspector"
    >
      <button
        type="button"
        className="workbench-inspector-toggle"
        aria-pressed={rightInspectorCollapsed}
        aria-label={rightInspectorCollapsed ? "Expand inspector" : "Collapse inspector"}
        onClick={() => setRightInspectorCollapsed(!rightInspectorCollapsed)}
      >
        {rightInspectorCollapsed ? "‹" : "›"}
      </button>
      {rightInspectorCollapsed ? null : (
        <div className="workbench-inspector-body">{children}</div>
      )}
    </aside>
  );
}
