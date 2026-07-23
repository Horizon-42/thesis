/**
 * OptimizationSummary.tsx
 * -----------------------
 * A small Observe-dock block (below the flight list) summarising the loaded category's
 * optimization results — solve rate from the run's summary.json plus the evaluation
 * package's batch metrics (success rate, average final lateral deviation, average
 * flight time), all read from the comparison index (never recomputed here).
 *
 * "Details" opens the full evaluation report (EvaluationReportWindow): the published
 * immutable report artifact named by the committed comparison index — a verbatim
 * copy of the backend `python -m evaluation` output — fetched lazily on first open.
 * The frontend only visualizes it.
 */

import { useEffect, useState } from "react";
import { useApp } from "../context/AppContext";
import { useFlightOptimizerData } from "../hooks/useFlightOptimizerData";
import { airportEvaluationReportUrl } from "../data/airportData";
import { isEvaluationReport, type EvaluationReport } from "../data/evaluationReport";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import { formatPercent, formatDuration } from "../utils/flightListFormat";
import EvaluationReportWindow from "./EvaluationReportWindow";

function formatMetres(m: number | null | undefined): string {
  if (m == null || !Number.isFinite(m)) return "—";
  return `${Math.round(m)} m`;
}

export default function OptimizationSummary() {
  const { activeAirportCode } = useApp();
  const { stats, categoryDir, reportFile } = useFlightOptimizerData();

  // The report is per committed generation, not merely per category. Including the
  // immutable filename prevents a refreshed index from reusing the previous generation's
  // cached metrics under the same airport/category.
  const reportKey = activeAirportCode && categoryDir && reportFile
    ? `${activeAirportCode}/${categoryDir}/${reportFile}`
    : null;
  const [cached, setCached] = useState<{ key: string; report: EvaluationReport } | null>(null);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setOpen(false);
    setError(null);
  }, [reportKey]);
  const report = cached && cached.key === reportKey ? cached.report : null;

  const rows: Array<{ label: string; value: string; pending: boolean }> = [
    { label: "Solve rate", value: formatPercent(stats?.solveRate), pending: stats?.solveRate == null },
    { label: "Success rate", value: formatPercent(stats?.successRate), pending: stats?.successRate == null },
    { label: "Avg state error", value: formatMetres(stats?.avgStateErrorM), pending: stats?.avgStateErrorM == null },
    { label: "Avg flight time", value: formatDuration(stats?.avgTimeS ?? null), pending: stats?.avgTimeS == null },
  ];

  function openDetails() {
    if (!activeAirportCode || !categoryDir || !reportFile || !reportKey) return;
    setError(null);
    if (report) {
      setOpen(true);
      return;
    }
    setLoading(true);
    fetchJson<unknown>(
      airportEvaluationReportUrl(
        activeAirportCode,
        categoryDir,
        reportFile,
      ),
    )
      .then((data) => {
        if (!isEvaluationReport(data)) {
          throw new Error("evaluation report is malformed");
        }
        setCached({ key: reportKey, report: data });
        setOpen(true);
      })
      .catch((fetchError) => {
        setError(
          isMissingJsonAsset(fetchError)
            ? "No evaluation report published for this category — re-run the pipeline's czml tail."
            : String(fetchError instanceof Error ? fetchError.message : fetchError),
        );
      })
      .finally(() => setLoading(false));
  }

  return (
    <section className="optimization-summary" aria-label="Optimization results">
      <div className="optimization-summary-head">
        <h4>Optimization</h4>
        <button
          type="button"
          className="optimization-summary-details"
          onClick={openDetails}
          disabled={!activeAirportCode || !categoryDir || !reportFile || loading}
          title="Open the full evaluation report (per-flight verdicts + charts)"
        >
          {loading ? "Loading…" : "Details"}
        </button>
      </div>
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
      {error ? <p className="optimization-summary-error">{error}</p> : null}
      {open && report ? (
        <EvaluationReportWindow
          report={report}
          subtitle={`${activeAirportCode ?? ""} · ${categoryDir ?? ""} · ${report.total} trajectories`}
          onClose={() => setOpen(false)}
        />
      ) : null}
    </section>
  );
}
