/**
 * WorkbenchTopBar.tsx
 * -------------------
 * The persistent top context bar of the workbench shell. It holds the app's
 * orthogonal global context — Active Airport + Landing Runway — and the single
 * task switcher (Observe / Procedures / Fly / Optimize / Compare) that replaces the
 * app's former two competing "mode" systems. It also exposes the on-demand Layers
 * drawer and the Presentation-mode toggle.
 *
 * Everything goes through AppContext; this component never touches Cesium directly.
 */

import { useApp, type WorkbenchMode } from "../context/AppContext";
import { useLandingsManifest } from "../hooks/useLandingsManifest";

const TASK_TABS: Array<{ mode: WorkbenchMode; label: string }> = [
  { mode: "observe", label: "Observe" },
  { mode: "procedures", label: "Procedures" },
  { mode: "fly", label: "Fly" },
  { mode: "optimize", label: "Optimize" },
  { mode: "compare", label: "Compare" },
];

export default function WorkbenchTopBar() {
  const {
    airports,
    activeAirportCode,
    setActiveAirportCode,
    selectedRunway,
    setSelectedRunway,
    mode,
    setMode,
    layersDrawerOpen,
    setLayersDrawerOpen,
    presentationMode,
    setPresentationMode,
  } = useApp();
  const { manifest: landingsManifest } = useLandingsManifest(activeAirportCode);
  const landingRunways = landingsManifest?.runways ?? [];

  return (
    <header className="workbench-topbar">
      <h1 className="workbench-topbar-title">AeroViz-4D</h1>

      <div className="workbench-topbar-context">
        <label className="workbench-topbar-field">
          <span>Active Airport</span>
          <select
            value={activeAirportCode}
            onChange={(event) => setActiveAirportCode(event.target.value)}
            disabled={airports.length === 0}
          >
            {airports.map((airport) => (
              <option key={airport.code} value={airport.code}>
                {airport.code} - {airport.name}
              </option>
            ))}
          </select>
        </label>

        {landingRunways.length > 0 ? (
          <label className="workbench-topbar-field">
            <span>Landing Runway</span>
            <select
              value={selectedRunway ?? ""}
              onChange={(event) => setSelectedRunway(event.target.value || null)}
            >
              <option value="">All runways</option>
              {landingRunways.map((entry) => (
                <option key={entry.runway} value={entry.runway}>
                  {entry.runway} ({entry.count})
                </option>
              ))}
            </select>
          </label>
        ) : null}
      </div>

      <nav className="workbench-task-switcher" role="group" aria-label="Task">
        {TASK_TABS.map((tab) => (
          <button
            key={tab.mode}
            type="button"
            className={`workbench-task-tab${mode === tab.mode ? " active" : ""}`}
            aria-pressed={mode === tab.mode}
            onClick={() => setMode(tab.mode)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <div className="workbench-topbar-actions">
        <button
          type="button"
          className={`workbench-topbar-button${layersDrawerOpen ? " active" : ""}`}
          aria-pressed={layersDrawerOpen}
          onClick={() => setLayersDrawerOpen(!layersDrawerOpen)}
        >
          ⚙ Layers
        </button>
        <button
          type="button"
          className={`workbench-topbar-button${presentationMode ? " active" : ""}`}
          aria-pressed={presentationMode}
          onClick={() => setPresentationMode(!presentationMode)}
        >
          ▣ Present
        </button>
      </div>
    </header>
  );
}
