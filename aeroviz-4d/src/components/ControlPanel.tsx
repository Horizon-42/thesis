/**
 * ControlPanel.tsx
 * ----------------
 * Observe-mode trajectory controls: the Trajectories layer toggle and its options
 * (category-aware comparison legend + sample count). Playback transport lives
 * in the bottom bar and airport/runway selection in the top bar; this panel only
 * owns the trajectory-view options.
 *
 * All interactions go through AppContext — this component never touches Cesium.
 */

import { useApp } from "../context/AppContext";
import type { ObservedVerdictState } from "../hooks/useObservedVerdictColors";
import {
  OBSERVED_VERDICT_COLORS,
  OBSERVED_VERDICT_HINTS,
  OBSERVED_VERDICT_LABELS,
} from "../utils/observedVerdictColors";
import { useComparisonCategories } from "../hooks/useComparisonCategories";
import { useComparisonLegend } from "../hooks/useComparisonLegend";
import type { ComparisonLegendKind } from "../utils/comparisonLegend";
import { isDrawableComparisonCategory } from "../data/airportData";
import { useForcedProcedureDisplay } from "../hooks/useForcedProcedureDisplay";
import {
  COMPARISON_KIND_COLORS,
  COMPARISON_KIND_ALPHA,
  COMPARISON_STATUS_STYLES,
} from "../utils/trajectoryRenderModel";
import ApproachViewToggle from "./ApproachViewToggle";
import { useEffect } from "react";
import {
  activeTrajectoryResultSource,
  categoriesForResultSource,
  categoryForExperimentSplit,
  experimentOptions,
  type TrajectoryResultSource,
} from "../utils/trajectoryResultSources";

const COMPARISON_KIND_LABELS: Record<ComparisonLegendKind, string> = {
  reference: "Reference",
  simulator: "Optimize results",
  predicted: "Predicted",
  lookback: "Predictor input",
};

interface ControlPanelProps {
  /**
   * Gate-verdict tally for the plain observed tracks. Optional and defaulted: it drives
   * a legend only, so the panel stays constructible (and testable) without it.
   */
  observedVerdicts?: ObservedVerdictState;
}

const NO_VERDICTS: ObservedVerdictState = {
  counts: null,
  matched: 0,
  total: 0,
  loading: false,
  missing: false,
};

export default function ControlPanel({ observedVerdicts = NO_VERDICTS }: ControlPanelProps) {
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
  const drawableComparisonCategories = comparisonCategories.filter(
    isDrawableComparisonCategory,
  );
  const predictionCategories = categoriesForResultSource(
    drawableComparisonCategories,
    "prediction",
  );
  const experimentCategories = categoriesForResultSource(
    drawableComparisonCategories,
    "experiment",
  );
  const heldOutTestCategories = predictionCategories.filter(
    (category) => category.datasetSplit === "test",
  );
  const trainingCategories = predictionCategories.filter(
    (category) => category.datasetSplit === "train",
  );
  const otherComparisonCategories = predictionCategories.filter(
    (category) => category.datasetSplit !== "train" && category.datasetSplit !== "test",
  );
  const activeComparisonCategory =
    drawableComparisonCategories.find((c) => c.dir === trajectoryComparisonCategory) ?? null;
  const resultSource = activeTrajectoryResultSource(
    trajectoryComparison,
    activeComparisonCategory,
  );
  const experiments = experimentOptions(experimentCategories);
  const activeExperimentId = activeComparisonCategory?.experiment?.id ?? "";
  const activeExperimentCategories = experimentCategories.filter(
    (category) => category.experiment?.id === activeExperimentId,
  );
  const experimentGroups = [...new Set(experiments.map((experiment) => experiment.group))];
  const comparisonLegend = useComparisonLegend(
    activeAirportCode,
    activeComparisonCategory?.dir ?? null,
    selectedRunway,
    layers.trajectories && trajectoryComparison,
  );

  // A report-only category (notably Observed ADS-B) has no comparison index/CZML and
  // therefore cannot own this selector. Default to the first drawable category and
  // keep the selection valid as airports change.
  useEffect(() => {
    const drawable = comparisonCategories.filter(isDrawableComparisonCategory);
    if (drawable.length === 0) {
      if (trajectoryComparisonCategory !== null) setTrajectoryComparisonCategory(null);
      return;
    }
    const stillValid = drawable.some((c) => c.dir === trajectoryComparisonCategory);
    if (stillValid) return;
    setTrajectoryComparisonCategory(drawable[0].dir);
  }, [comparisonCategories, trajectoryComparisonCategory, setTrajectoryComparisonCategory]);

  // ── Constraint-scoped procedure display (drive & restore) ────────────────────
  // When the comparison overlay is showing a CONSTRAINED category (the manifest's
  // explicit `constrained` field — its solves enforce the runway's RNAV procedure) and a
  // global landing runway is selected, drive the procedure display on for that runway —
  // the same affordance Optimize gives, so you see the corridors / glidepath the shown
  // solves flew. `forceRunway: null`: the runway IS the user-owned global selectedRunway,
  // so the hook only opens the panel + layer and never touches the runway (it must not
  // fight the top-bar selector nor revert it on exit).
  useForcedProcedureDisplay({
    active:
      layers.trajectories &&
      trajectoryComparison &&
      activeComparisonCategory?.constrained === true &&
      selectedRunway !== null,
    forceRunway: null,
  });

  function selectResultSource(source: TrajectoryResultSource): void {
    if (source === "baseline") {
      setTrajectoryComparison(false);
      return;
    }
    const candidates = source === "experiment" ? experimentCategories : predictionCategories;
    const currentStillMatches = candidates.some(
      (category) => category.dir === trajectoryComparisonCategory,
    );
    if (!currentStillMatches) {
      setTrajectoryComparisonCategory(candidates[0]?.dir ?? null);
    }
    setTrajectoryComparison(true);
  }

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
            <label className="control-panel-airport-selector">
              <span>Result source</span>
              <select
                className="control-panel-airport-selector-input"
                value={resultSource}
                onChange={(event) =>
                  selectResultSource(event.target.value as TrajectoryResultSource)}
              >
                <option value="baseline">Baseline</option>
                <option value="prediction" disabled={predictionCategories.length === 0}>
                  Prediction
                </option>
                <option value="experiment" disabled={experimentCategories.length === 0}>
                  Experiments
                </option>
              </select>
            </label>
            {resultSource === "baseline" && observedVerdicts.counts ? (
              <div className="control-panel-verdict-legend" aria-label="Approach verdict legend">
                <div className="control-panel-verdict-title">
                  Approach verdict (FAA 8260.58D gates at the threshold)
                </div>
                {(["pass", "fail", "undecided"] as const).map((verdict) => (
                  <div key={verdict} className="control-panel-verdict-row">
                    <span
                      className="control-panel-verdict-swatch"
                      style={{ background: OBSERVED_VERDICT_COLORS[verdict] }}
                      aria-hidden="true"
                    />
                    <span className="control-panel-verdict-label">
                      {OBSERVED_VERDICT_LABELS[verdict]}
                      <strong>{observedVerdicts.counts?.[verdict] ?? 0}</strong>
                    </span>
                    <span className="control-panel-verdict-hint">
                      {OBSERVED_VERDICT_HINTS[verdict]}
                    </span>
                  </div>
                ))}
                {observedVerdicts.matched < observedVerdicts.total ? (
                  <div className="control-panel-verdict-note">
                    {observedVerdicts.total - observedVerdicts.matched} of {observedVerdicts.total}{" "}
                    tracks have no published verdict and are shown as Undecidable — their runway
                    may lack a published gate target, or the report may not cover this harvest.
                  </div>
                ) : null}
              </div>
            ) : null}
            {resultSource === "prediction" ? (
              predictionCategories.length > 0 ? (
                <label className="control-panel-airport-selector">
                  <span>Prediction result</span>
                  <select
                    className="control-panel-airport-selector-input"
                    value={trajectoryComparisonCategory ?? ""}
                    onChange={(event) => setTrajectoryComparisonCategory(event.target.value || null)}
                  >
                    {heldOutTestCategories.length > 0 ? (
                      <optgroup label="Held-out test results">
                        {heldOutTestCategories.map((category) => (
                          <option key={category.key} value={category.dir}>
                            {category.label} ({category.groups})
                          </option>
                        ))}
                      </optgroup>
                    ) : null}
                    {trainingCategories.length > 0 ? (
                      <optgroup label="Training results (in-sample)">
                        {trainingCategories.map((category) => (
                          <option key={category.key} value={category.dir}>
                            {category.label} ({category.groups})
                          </option>
                        ))}
                      </optgroup>
                    ) : null}
                    {otherComparisonCategories.length > 0 ? (
                      <optgroup label="Other evaluation results">
                        {otherComparisonCategories.map((category) => (
                          <option key={category.key} value={category.dir}>
                            {category.label} ({category.groups})
                          </option>
                        ))}
                      </optgroup>
                    ) : null}
                  </select>
                </label>
              ) : (
                <p className="control-panel-comparison-empty">
                  No comparison data found for this airport.
                </p>
              )
            ) : null}
            {resultSource === "experiment" ? (
              experimentCategories.length > 0 ? (
                <div className="control-panel-experiment-selectors">
                  <label className="control-panel-airport-selector">
                    <span>Experiment model</span>
                    <select
                      className="control-panel-airport-selector-input"
                      value={activeExperimentId}
                      onChange={(event) => {
                        const next = categoryForExperimentSplit(
                          experimentCategories,
                          event.target.value,
                          activeComparisonCategory?.datasetSplit === "train" ? "train" : "val",
                        );
                        setTrajectoryComparisonCategory(next?.dir ?? null);
                      }}
                    >
                      {experimentGroups.map((group) => (
                        <optgroup key={group} label={group}>
                          {experiments
                            .filter((experiment) => experiment.group === group)
                            .map((experiment) => (
                              <option key={experiment.id} value={experiment.id}>
                                {experiment.label} · {experiment.predictionOutput ?? "state"}
                                {experiment.horizonMode === "normalized" ? " · normalized time" : ""}
                                {experiment.horizonMode === "full" ? " · full horizon" : ""}
                                {experiment.horizonMode === "window" ? " · recursive window" : ""}
                                {experiment.seed == null ? "" : ` · seed ${experiment.seed}`}
                              </option>
                            ))}
                        </optgroup>
                      ))}
                    </select>
                  </label>
                  <label className="control-panel-airport-selector">
                    <span>Dataset split</span>
                    <select
                      className="control-panel-airport-selector-input"
                      value={activeComparisonCategory?.dir ?? ""}
                      onChange={(event) =>
                        setTrajectoryComparisonCategory(event.target.value || null)}
                    >
                      {activeExperimentCategories.map((category) => (
                        <option key={category.key} value={category.dir}>
                          {category.datasetSplit === "train" ? "Training (in-sample)" :
                            category.datasetSplit === "val" ? "Validation (model selection)" :
                              "Held-out test"} ({category.groups})
                        </option>
                      ))}
                    </select>
                  </label>
                  {activeComparisonCategory?.experiment ? (
                    <p className="control-panel-experiment-checkpoint">
                      {activeComparisonCategory.experiment.checkpoint}
                    </p>
                  ) : null}
                </div>
              ) : (
                <p className="control-panel-comparison-empty">
                  No experiment trajectories have been published for this airport.
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
            {trajectoryComparison && activeComparisonCategory ? (
              <div
                className="control-panel-comparison-kinds"
                aria-label="Comparison trajectory legend"
              >
                {comparisonLegend.kinds.map((kind) => (
                  <label key={kind}>
                    <input
                      type="checkbox"
                      checked={trajectoryComparisonKinds[kind]}
                      onChange={(event) => setTrajectoryComparisonKind(kind, event.target.checked)}
                    />
                    {/* Opacity, not just hue: "Predicted" and "Predictor input" share a colour
                        and are told apart by their alpha, so a solid swatch would make the two
                        rows look identical. Same source the paths are drawn with. */}
                    <i style={{
                      background: COMPARISON_KIND_COLORS[kind],
                      opacity: COMPARISON_KIND_ALPHA[kind],
                    }} />
                    {COMPARISON_KIND_LABELS[kind]}
                  </label>
                ))}
                {comparisonLegend.statuses.length > 0 ? (
                  <div
                    className="control-panel-comparison-statuses"
                    aria-label="Outcome colour overrides"
                  >
                    <span className="control-panel-comparison-status-title">Outcome colours</span>
                    {comparisonLegend.statuses.map((status) => {
                      const style = COMPARISON_STATUS_STYLES[status];
                      return (
                        <span key={status} className="control-panel-comparison-status-row">
                          <i style={{ background: style.color, opacity: style.alpha }} />
                          {style.label}
                        </span>
                      );
                    })}
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
        ) : null}
      </section>
    </div>
  );
}
