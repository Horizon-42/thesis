/**
 * WorkbenchShell.tsx
 * ------------------
 * The workbench layout shell that sits over the full-bleed Cesium globe. It owns the
 * persistent top context bar (WorkbenchTopBar) and a body region that hosts the
 * overlay docks. `pointer-events: none` lets clicks fall through to the globe; each
 * dock re-enables them.
 *
 * Migration note: during the staged restructure the body keeps the original
 * `.cesium-overlay-container` element (now just a transparent portal/overlay host) so
 * the portal-rendered panels (PilotRealtimeStatePanel, the comparison charts) keep
 * working unchanged. Later stages move the docked panels into dedicated grid areas.
 */

import { useEffect, type ReactNode } from "react";
import { useApp } from "../context/AppContext";
import WorkbenchTopBar from "./WorkbenchTopBar";
import LayersDrawer from "./LayersDrawer";

export default function WorkbenchShell({
  left,
  right,
  bottom,
  children,
}: {
  left?: ReactNode;
  right?: ReactNode;
  bottom?: ReactNode;
  children: ReactNode;
}) {
  const { presentationMode, setPresentationMode } = useApp();

  // Presentation mode hides all chrome (including the top bar that toggled it), so
  // Esc is the always-available way back out, alongside the floating exit button.
  useEffect(() => {
    if (!presentationMode) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setPresentationMode(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [presentationMode, setPresentationMode]);

  // Cesium's timeline widget lives outside the .workbench tree, so reach it via a
  // body class to hide the scrubber for a fully clean globe in presentation mode.
  useEffect(() => {
    document.body.classList.toggle("workbench-presentation-active", presentationMode);
    return () => document.body.classList.remove("workbench-presentation-active");
  }, [presentationMode]);

  return (
    <div className={`workbench${presentationMode ? " workbench--presentation" : ""}`}>
      <WorkbenchTopBar />
      <div className="cesium-overlay-container">
        {left ? <div className="left-overlay-panel-stack">{left}</div> : null}
        {right}
        {children}
      </div>
      {bottom}
      <LayersDrawer />
      {presentationMode ? (
        <button
          type="button"
          className="workbench-presentation-exit"
          onClick={() => setPresentationMode(false)}
        >
          Exit presentation (Esc)
        </button>
      ) : null}
    </div>
  );
}
