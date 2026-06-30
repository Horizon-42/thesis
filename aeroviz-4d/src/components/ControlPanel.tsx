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
import { COMPARISON_KIND_COLORS } from "../utils/trajectoryRenderModel";
import { useEffect } from "react";

/**
 * The three comparison trajectories, each a colour-keyed visibility checkbox. The kind keys
 * ("optimizer"/"simulator") stay as the backend's entity-id prefixes; the swatch colour comes
 * from the shared COMPARISON_KIND_COLORS (the same source the rendered paths are recoloured to),
 * so the legend and the tracks can never disagree.
 */
const COMPARISON_KINDS: Array<{ kind: ComparisonKind; label: string }> = [
  { kind: "reference", label: "Reference" },
  { kind: "simulator", label: "Optimize results" },
  { kind: "optimizer", label: "Optimize states" },
];

export default function ControlPanel() {
  const {
    layers,
    toggleLayer,
    activeAirportCode,
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

  return (
    <div className="control-panel">
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
