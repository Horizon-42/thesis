/**
 * WorkbenchTopBar.tsx
 * -------------------
 * The persistent top context bar of the workbench shell. It holds the app's
 * orthogonal global context — Active Airport + Landing Runway — and the single
 * task switcher (Fly | Optimize / Learning / Evaluate, plus the Procedures toggle) that replaces the
 * app's former two competing "mode" systems. It also exposes the on-demand Layers
 * drawer and the Presentation-mode toggle.
 *
 * Everything goes through AppContext; this component never touches Cesium directly.
 */

import { useApp, type WorkbenchMode } from "../context/AppContext";
import { useLandingsManifest } from "../hooks/useLandingsManifest";

interface TaskTab {
  mode: WorkbenchMode;
  label: string;
}

/**
 * The tasks in GROUPS, left to right (the user's order, 2026-10-01): Fly on its own, then
 * Optimize, Learning, Evaluate. A gap sets each group apart (`workbench-task-group-start`,
 * the same gap that sets the Procedures toggle apart).
 */
const TASK_GROUPS: TaskTab[][] = [
  [{ mode: "fly", label: "Fly" }],
  [
    { mode: "optimize", label: "Optimize" },
    // Only the tab is named Learning: what it shows is still the training of the models
    // (its sets, rounds and details), so its mode and code keep the name `training`.
    { mode: "training", label: "Learning" },
    // Only the tab reads "Evaluate"; the task (mode `evaluation`) and its summaries keep "Evaluation".
    { mode: "evaluation", label: "Evaluate" },
  ],
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
    proceduresOpen,
    setProceduresOpen,
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
        {TASK_GROUPS.flatMap((group, groupIndex) => group.map((tab, index) => (
          <button
            key={tab.mode}
            type="button"
            className={`workbench-task-tab${groupIndex > 0 && index === 0 ? " workbench-task-group-start" : ""}${
              mode === tab.mode ? " active" : ""}`}
            aria-pressed={mode === tab.mode}
            onClick={() => setMode(tab.mode)}
          >
            {tab.label}
          </button>
        )))}
        {/* Procedures is an independent toggle, not a task — it coexists with the
            active task above (separated here to signal that). */}
        <button
          type="button"
          className={`workbench-task-tab workbench-task-group-start workbench-task-tab-toggle${proceduresOpen ? " active" : ""}`}
          aria-pressed={proceduresOpen}
          onClick={() => setProceduresOpen(!proceduresOpen)}
        >
          Procedures
        </button>
      </nav>

      <div className="workbench-topbar-actions">
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
