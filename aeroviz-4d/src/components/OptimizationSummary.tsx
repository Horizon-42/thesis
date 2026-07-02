/**
 * OptimizationSummary.tsx
 * -----------------------
 * A small Observe-dock block (below the flight list) summarising the loaded category's
 * optimization results. Only the **solve rate** is real for now — read from the comparison
 * index (which took it straight from the run's summary.json, not recomputed). Success rate,
 * average state error and average flight time are placeholders until the evaluation package
 * computes them (they render automatically once the index carries those fields).
 */

import { useFlightOptimizerData } from "../hooks/useFlightOptimizerData";
import { formatPercent, formatDuration } from "../utils/flightListFormat";

function formatMetres(m: number | null | undefined): string {
  if (m == null || !Number.isFinite(m)) return "—";
  return `${Math.round(m)} m`;
}

export default function OptimizationSummary() {
  const { stats } = useFlightOptimizerData();

  const rows: Array<{ label: string; value: string; pending: boolean }> = [
    { label: "Solve rate", value: formatPercent(stats?.solveRate), pending: stats?.solveRate == null },
    { label: "Success rate", value: formatPercent(stats?.successRate), pending: stats?.successRate == null },
    { label: "Avg state error", value: formatMetres(stats?.avgStateErrorM), pending: stats?.avgStateErrorM == null },
    { label: "Avg flight time", value: formatDuration(stats?.avgTimeS ?? null), pending: stats?.avgTimeS == null },
  ];

  return (
    <section className="optimization-summary" aria-label="Optimization results">
      <h4>Optimization</h4>
      <dl>
        {rows.map((row) => (
          <div key={row.label} className="optimization-summary-row">
            <dt>{row.label}</dt>
            <dd
              className={row.pending ? "optimization-summary-pending" : undefined}
              title={row.pending ? "pending the evaluation package" : undefined}
            >
              {row.value}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
