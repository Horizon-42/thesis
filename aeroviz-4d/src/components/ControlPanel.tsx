/**
 * ControlPanel.tsx
 * ----------------
 * Observe-mode trajectory controls: the Trajectories layer toggle and its options
 * (optimizer 3-colour comparison + category, sample count). Playback transport lives
 * in the bottom bar and airport/runway selection in the top bar; this panel only
 * owns the trajectory-view options.
 *
 * All interactions go through AppContext — this component never touches Cesium.
 */

import { useApp, type ComparisonKind } from "../context/AppContext";
import { useComparisonCategories } from "../hooks/useComparisonCategories";
import { useForcedProcedureDisplay } from "../hooks/useForcedProcedureDisplay";
import { COMPARISON_KIND_COLORS } from "../utils/trajectoryRenderModel";
import ApproachViewToggle from "./ApproachViewToggle";
import { useEffect } from "react";

/**
 * The comparison trajectories, each a colour-keyed visibility checkbox. The kind keys
 * ("optimizer"/"simulator"/"predicted") stay as the backend's entity-id prefixes; the swatch
 * colour comes from the shared COMPARISON_KIND_COLORS (the same source the rendered paths are
 * recoloured to), so the legend and the tracks can never disagree.
 *
 * Which kinds a category actually contains depends on its producer: an optimizer batch emits
 * reference + optimizer + simulator, a ts_transformer batch emits reference + predicted. A
 * checkbox for a kind the loaded category has none of is harmless — it toggles nothing — so
 * the list stays static rather than being derived per category.
 */
const COMPARISON_KINDS: Array<{ kind: ComparisonKind; label: string }> = [
  { kind: "reference", label: "Reference" },
  { kind: "simulator", label: "Optimize results" },
  { kind: "optimizer", label: "Optimize states" },
  { kind: "predicted", label: "Predicted" },
];

export default function ControlPanel() {
  const {
    layers,
    toggleLayer,
    activeAirportCode,
    selectedRunway,
    trajectoryComparison,
    setTrajectoryComparison,
    trajectoryComparisonCategory,
    setTrajectoryComparisonCategory,
    trajectoryComparisonKinds,
    setTrajectoryComparisonKind,
    trajectorySampleCount,
    setTrajectorySampleCount,
  } = useApp();
  const { categories: comparisonCategories } = useComparisonCategories(activeAirportCode);

  // Default the comparison category to the first available, and keep it valid as airports change.
  useEffect(() => {
    if (comparisonCategories.length === 0) {
      if (trajectoryComparisonCategory !== null) setTrajectoryComparisonCategory(null);
      return;
    }
    const stillValid = comparisonCategories.some((c) => c.dir === trajectoryComparisonCategory);
    if (!stillValid) setTrajectoryComparisonCategory(comparisonCategories[0].dir);
  }, [comparisonCategories, trajectoryComparisonCategory, setTrajectoryComparisonCategory]);

  // ── Constraint-scoped procedure display (drive & restore) ────────────────────
  // When the comparison overlay is showing a CONSTRAINED category (the manifest's
  // explicit `constrained` field — its solves enforce the runway's RNAV procedure) and a
  // global landing runway is selected, drive the procedure display on for that runway —
  // the same affordance Optimize gives, so you see the corridors / glidepath the shown
  // solves flew. `forceRunway: null`: the runway IS the user-owned global selectedRunway,
  // so the hook only opens the panel + layer and never touches the runway (it must not
  // fight the top-bar selector nor revert it on exit).
  const activeComparisonCategory =
    comparisonCategories.find((c) => c.dir === trajectoryComparisonCategory) ?? null;
  useForcedProcedureDisplay({
    active:
      layers.trajectories &&
      trajectoryComparison &&
      activeComparisonCategory?.constrained === true &&
      selectedRunway !== null,
    forceRunway: null,
  });

  return (
    <div className="control-panel">
      <div className="control-panel-approach-view">
        <div className="control-panel-approach-view-row">
          <span>Approach view</span>
          <ApproachViewToggle runwayIdent={selectedRunway} />
        </div>
        {selectedRunway === null ? (
          <p className="control-panel-approach-view-hint">
            No landing runway selected — pick one in the top bar to view its approach view.
          </p>
        ) : null}
      </div>
      <section className="control-panel-trajectory-layer">
        <label>
          <input
            type="checkbox"
            checked={layers.trajectories}
            onChange={() => toggleLayer("trajectories")}
          />
          Trajectories
        </label>

        {layers.trajectories ? (
          <div className="control-panel-trajectory-options" aria-label="Trajectory options">
            <label>
              <input
                type="checkbox"
                checked={trajectoryComparison}
                onChange={(event) => setTrajectoryComparison(event.target.checked)}
              />
              Optimizer comparison (3-colour)
            </label>
            {trajectoryComparison ? (
              comparisonCategories.length > 0 ? (
                <label className="control-panel-airport-selector">
                  <span>Optimization category</span>
                  <select
                    className="control-panel-airport-selector-input"
                    value={trajectoryComparisonCategory ?? ""}
                    onChange={(event) => setTrajectoryComparisonCategory(event.target.value || null)}
                  >
                    {comparisonCategories.map((category) => (
                      <option key={category.key} value={category.dir}>
                        {category.label} ({category.groups})
                      </option>
                    ))}
                  </select>
                </label>
              ) : (
                <p className="control-panel-comparison-empty">
                  No comparison data found for this airport.
                </p>
              )
            ) : null}
            <label className="control-panel-airport-selector">
              <span>Sample count (0 = all)</span>
              <input
                type="number"
                min={0}
                step={1}
                className="control-panel-airport-selector-input"
                value={trajectorySampleCount}
                onChange={(event) => {
                  const parsed = Number.parseInt(event.target.value, 10);
                  setTrajectorySampleCount(Number.isFinite(parsed) && parsed > 0 ? parsed : 0);
                }}
              />
            </label>
            {trajectoryComparison ? (
              <div className="control-panel-comparison-kinds">
                {COMPARISON_KINDS.map(({ kind, label }) => (
                  <label key={kind}>
                    <input
                      type="checkbox"
                      checked={trajectoryComparisonKinds[kind]}
                      onChange={(event) => setTrajectoryComparisonKind(kind, event.target.checked)}
                    />
                    <i style={{ background: COMPARISON_KIND_COLORS[kind] }} />
                    {label}
                  </label>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
      </section>
    </div>
  );
}
