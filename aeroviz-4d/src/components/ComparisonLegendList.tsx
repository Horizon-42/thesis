/**
 * ComparisonLegendList.tsx
 * ------------------------
 * The comparison legend: one switch per path kind the category draws, the outcome colours that override a kind's colour,
 * and — for a traffic category — the recorded aircraft that are not controlled. Used by the Evaluate task's trajectory
 * options and by the Optimize task's multi-aircraft panel, which draw the same layer.
 */

import { useApp } from "../context/AppContext";
import {
  TRAFFIC_OFF_TARGET_LABEL,
  comparisonKindLabel,
  recordedTrafficLabel,
  type ComparisonLegendModel,
} from "../utils/comparisonLegend";
import {
  COMPARISON_KIND_ALPHA,
  COMPARISON_STATUS_STYLES,
  COMPARISON_TRAFFIC_COLOR,
  comparisonKindSwatch,
} from "../utils/trajectoryRenderModel";

export default function ComparisonLegendList({ legend }: { legend: ComparisonLegendModel }) {
  const { trajectoryComparisonKinds, setTrajectoryComparisonKind } = useApp();
  const offTarget = COMPARISON_STATUS_STYLES.offTargetResult;
  // In a traffic category the off-target yellow is explained at the optimized-path entry, and not listed again below.
  const trafficOffTarget = legend.traffic !== null && legend.statuses.includes("offTargetResult");
  const statuses = trafficOffTarget ? legend.statuses.filter((status) => status !== "offTargetResult") : legend.statuses;
  return (
    <div className="control-panel-comparison-kinds" aria-label="Comparison trajectory legend">
      {legend.kinds.map((kind) => (
        <span key={kind} className="comparison-legend-kind">
          <label>
            <input
              type="checkbox"
              checked={trajectoryComparisonKinds[kind]}
              onChange={(event) => setTrajectoryComparisonKind(kind, event.target.checked)}
            />
            {/* Opacity, not just hue: "Predicted" and "Predictor input" draw the same
                outcome colours and are told apart by their alpha alone, so the swatch
                has to carry that alpha too. Same source the paths are drawn with. */}
            <i style={{
              background: comparisonKindSwatch(kind),
              opacity: COMPARISON_KIND_ALPHA[kind],
            }} />
            {comparisonKindLabel(kind, legend.traffic)}
          </label>
          {kind === "simulator" && trafficOffTarget ? (
            <span className="control-panel-comparison-status-row comparison-legend-explains">
              <i style={{ background: offTarget.color, opacity: offTarget.alpha }} />
              {TRAFFIC_OFF_TARGET_LABEL}
            </span>
          ) : null}
        </span>
      ))}
      {legend.traffic ? (
        /* The recorded neighbours follow the record switch above: they are recorded tracks too. */
        <span className="control-panel-comparison-status-row" title="Shown with the record switch">
          <i style={{ background: COMPARISON_TRAFFIC_COLOR, opacity: COMPARISON_KIND_ALPHA.reference }} />
          {recordedTrafficLabel(legend.traffic)}
        </span>
      ) : null}
      {statuses.length > 0 ? (
        <div className="control-panel-comparison-statuses" aria-label="Outcome colour overrides">
          <span className="control-panel-comparison-status-title">Outcome colours</span>
          {statuses.map((status) => {
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
  );
}
