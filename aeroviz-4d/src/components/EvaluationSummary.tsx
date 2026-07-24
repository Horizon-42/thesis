/**
 * EvaluationSummary.tsx
 * ---------------------
 * Subject-aware evaluation summary for the active Observe comparison category.
 *
 * The three producers answer different questions, so they deliberately do not share
 * one generic set of "solve/success" labels:
 *   • observed       — quality of the measured final approach at the runway threshold;
 *   • optimization   — solver outcome and error relative to the selected target;
 *   • data-driven    — ADE/FDE against the observed trajectory.
 *
 * Observed is a report-only category and therefore reads its fixed report directly.
 * Modelled categories first read their immutable comparison index; Details then follows
 * the exact evaluation-report filename committed by that index.
 */

import { useEffect, useMemo, useState } from "react";
import { useApp } from "../context/AppContext";
import {
  OBSERVED_CATEGORY_KEY,
  OBSERVED_EVALUATION_REPORT_FILE,
  airportComparisonIndexUrl,
  airportEvaluationReportUrl,
  isComparisonIndex,
  type ComparisonCategory,
  type OptimizationStats,
  type PredictionAccuracyStats,
} from "../data/airportData";
import { isEvaluationReport, type EvaluationReport } from "../data/evaluationReport";
import { useComparisonCategories } from "../hooks/useComparisonCategories";
import { fetchJson, isMissingJsonAsset } from "../utils/fetchJson";
import { formatDuration, formatPercent } from "../utils/flightListFormat";
import EvaluationReportWindow from "./EvaluationReportWindow";

type EvaluationKind = "observed" | "optimization" | "dataDriven";

interface LoadedSummary {
  key: string;
  reportFile: string;
  stats: OptimizationStats | null;
  prediction: PredictionAccuracyStats | null;
  report: EvaluationReport | null;
}

interface SummaryRow {
  label: string;
  value: string;
  pending: boolean;
}

interface Presentation {
  title: string;
  context: string;
  note: string;
  rows: SummaryRow[];
}

function formatMetres(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${Math.round(value)} m`;
}

function row(label: string, value: string, available: boolean): SummaryRow {
  return { label, value, pending: !available };
}

function evaluationKind(category: ComparisonCategory): EvaluationKind {
  if (category.key === OBSERVED_CATEGORY_KEY) return "observed";
  if (category.key.startsWith("ts_")) return "dataDriven";
  return "optimization";
}

function isFittedAdsbTarget(category: ComparisonCategory): boolean {
  return category.key === "fitted_adsb";
}

function optimizationContext(category: ComparisonCategory): string {
  if (isFittedAdsbTarget(category)) return "Target: Fitted ADS-B crossing";
  return category.constrained
    ? "Target: Runway threshold · Procedure constrained"
    : "Target: Runway threshold";
}

function passRateAmongSolved(stats: OptimizationStats | null): number | null {
  const successful = stats?.successful;
  const solved = stats?.solved;
  if (
    successful == null ||
    solved == null ||
    !Number.isFinite(successful) ||
    !Number.isFinite(solved) ||
    solved <= 0
  ) {
    return null;
  }
  return successful / solved;
}

function observedPresentation(report: EvaluationReport | null): Presentation {
  const establishedRate = report?.observed?.established_rate;
  const measured = report?.measured;
  const crossingPassRate =
    report && measured != null && Number.isFinite(measured) && measured > 0
      ? report.successful / measured
      : null;
  const lateralMean = report?.lateral_m?.mean;
  const verticalMeanAbs = report?.vertical_m?.mean_abs;

  return {
    title: "Observed Baseline Evaluation",
    context: "Observed ADS-B trajectories",
    note:
      "Real ADS-B tracks are evaluated by fitting the established final-approach segment " +
      "and extrapolating it to the runway threshold. No optimizer is involved. The " +
      "threshold-crossing pass rate is calculated over established approaches; measurement " +
      "uncertainty near a gate boundary is reported in Details.",
    rows: [
      row(
        "Established approach rate",
        formatPercent(establishedRate ?? null),
        establishedRate != null,
      ),
      row(
        "Threshold-crossing pass rate",
        formatPercent(crossingPassRate),
        crossingPassRate != null,
      ),
      row(
        "Mean lateral deviation at threshold",
        formatMetres(lateralMean),
        lateralMean != null,
      ),
      row(
        "Mean absolute vertical deviation at threshold",
        formatMetres(verticalMeanAbs),
        verticalMeanAbs != null,
      ),
    ],
  };
}

function optimizationPresentation(
  category: ComparisonCategory,
  stats: OptimizationStats | null,
): Presentation {
  const fitted = isFittedAdsbTarget(category);
  const solveRate = stats?.solveRate;
  const targetPassRate = passRateAmongSolved(stats);
  const lateralError = stats?.avgStateErrorM;
  const averageTime = stats?.avgTimeS;
  const targetName = fitted ? "fitted target" : "runway threshold";

  return {
    title: "Optimization Evaluation",
    context: optimizationContext(category),
    note: fitted
      ? "The optimizer targets the threshold-crossing state fitted from each flight's ADS-B " +
        "track, not the nominal runway threshold. Fitted-target pass rate is calculated over " +
        "solved runs."
      : "The optimizer targets the nominal runway-threshold state. Runway-threshold pass " +
        "rate is calculated over solved runs." +
        (category.constrained
          ? " This category also enforces the selected procedure as path constraints."
          : ""),
    rows: [
      row("Solve rate", formatPercent(solveRate ?? null), solveRate != null),
      row(
        fitted ? "Fitted-target pass rate" : "Runway-threshold pass rate",
        formatPercent(targetPassRate),
        targetPassRate != null,
      ),
      row(
        `Mean lateral error to ${targetName}`,
        formatMetres(lateralError),
        lateralError != null,
      ),
      row(
        "Mean optimized flight time",
        formatDuration(averageTime ?? null),
        averageTime != null,
      ),
    ],
  };
}

function predictionPresentation(
  category: ComparisonCategory,
  prediction: PredictionAccuracyStats | null,
): Presentation {
  const adeMean = prediction?.adeM?.mean;
  const adeP95 = prediction?.adeM?.p95;
  const fdeMean = prediction?.fdeM?.mean;
  const fdeP95 = prediction?.fdeM?.p95;

  return {
    title: "Data-Driven Model Evaluation",
    context: category.label,
    note:
      "ADE is the mean position error across the predicted trajectory; FDE is the position " +
      "error at the final predicted sample. Both compare the prediction with the observed " +
      "track over their overlapping samples. A solver success rate does not apply.",
    rows: [
      row("Mean ADE", formatMetres(adeMean), adeMean != null),
      row("95th-percentile ADE", formatMetres(adeP95), adeP95 != null),
      row("Mean FDE", formatMetres(fdeMean), fdeMean != null),
      row("95th-percentile FDE", formatMetres(fdeP95), fdeP95 != null),
    ],
  };
}

export default function EvaluationSummary() {
  const {
    activeAirportCode,
    trajectoryComparison,
    trajectoryComparisonCategory,
  } = useApp();
  const { categories } = useComparisonCategories(activeAirportCode);
  const category = trajectoryComparison
    ? categories.find((candidate) => candidate.dir === trajectoryComparisonCategory) ?? null
    : categories.find((candidate) => candidate.key === OBSERVED_CATEGORY_KEY) ?? null;
  const sourceKey =
    activeAirportCode && category
      ? `${activeAirportCode}/${category.dir}/${category.key}`
      : null;

  const [loaded, setLoaded] = useState<LoadedSummary | null>(null);
  const [cachedReport, setCachedReport] =
    useState<{ key: string; report: EvaluationReport } | null>(null);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setOpen(false);
    setError(null);
    setLoaded(null);
    if (!activeAirportCode || !category || !sourceKey) return;

    let cancelled = false;
    const kind = evaluationKind(category);
    if (kind === "observed") {
      const reportKey = `${sourceKey}/${OBSERVED_EVALUATION_REPORT_FILE}`;
      setLoading(true);
      fetchJson<unknown>(
        airportEvaluationReportUrl(
          activeAirportCode,
          category.dir,
          OBSERVED_EVALUATION_REPORT_FILE,
        ),
      )
        .then((data) => {
          if (!isEvaluationReport(data)) {
            throw new Error("evaluation report is malformed");
          }
          if (cancelled) return;
          setCachedReport({ key: reportKey, report: data });
          setLoaded({
            key: sourceKey,
            reportFile: OBSERVED_EVALUATION_REPORT_FILE,
            stats: null,
            prediction: null,
            report: data,
          });
        })
        .catch((fetchError) => {
          if (cancelled) return;
          setError(
            isMissingJsonAsset(fetchError)
              ? "No observed evaluation report is published for this airport."
              : String(fetchError instanceof Error ? fetchError.message : fetchError),
          );
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    } else {
      setLoading(true);
      fetchJson<unknown>(airportComparisonIndexUrl(activeAirportCode, category.dir))
        .then((data) => {
          if (!isComparisonIndex(data)) {
            throw new Error(
              `comparison index for ${activeAirportCode}/${category.dir} does not use ` +
                "comparison-v2-generation",
            );
          }
          if (cancelled) return;
          setLoaded({
            key: sourceKey,
            reportFile: data.evaluationReport,
            stats: data.optimization ?? null,
            prediction: data.prediction ?? null,
            report: null,
          });
        })
        .catch((fetchError) => {
          if (cancelled) return;
          setError(
            isMissingJsonAsset(fetchError)
              ? "No evaluation data is published for this category."
              : String(fetchError instanceof Error ? fetchError.message : fetchError),
          );
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    }

    return () => {
      cancelled = true;
    };
  }, [activeAirportCode, category, sourceKey]);

  const presentation = useMemo<Presentation>(() => {
    if (!category) {
      return {
        title: "Evaluation",
        context: "Select a comparison category",
        note: "Select a comparison category to view its evaluation.",
        rows: [],
      };
    }
    const kind = evaluationKind(category);
    if (kind === "observed") {
      return observedPresentation(loaded?.key === sourceKey ? loaded.report : null);
    }
    if (kind === "dataDriven") {
      return predictionPresentation(
        category,
        loaded?.key === sourceKey ? loaded.prediction : null,
      );
    }
    return optimizationPresentation(
      category,
      loaded?.key === sourceKey ? loaded.stats : null,
    );
  }, [category, loaded, sourceKey]);

  const currentLoaded = loaded?.key === sourceKey ? loaded : null;
  const reportKey =
    sourceKey && currentLoaded
      ? `${sourceKey}/${currentLoaded.reportFile}`
      : null;
  const report =
    currentLoaded?.report ??
    (cachedReport && cachedReport.key === reportKey ? cachedReport.report : null);
  const reportTitle = `${presentation.title} Report`;

  function openDetails() {
    if (
      !activeAirportCode ||
      !category ||
      !loaded ||
      loaded.key !== sourceKey ||
      !reportKey
    ) {
      return;
    }
    setError(null);
    if (report) {
      setOpen(true);
      return;
    }
    setLoading(true);
    fetchJson<unknown>(
      airportEvaluationReportUrl(
        activeAirportCode,
        category.dir,
        loaded.reportFile,
      ),
    )
      .then((data) => {
        if (!isEvaluationReport(data)) {
          throw new Error("evaluation report is malformed");
        }
        setCachedReport({ key: reportKey, report: data });
        setOpen(true);
      })
      .catch((fetchError) => {
        setError(
          isMissingJsonAsset(fetchError)
            ? "No evaluation report is published for this category."
            : String(fetchError instanceof Error ? fetchError.message : fetchError),
        );
      })
      .finally(() => setLoading(false));
  }

  return (
    <section className="evaluation-summary" aria-label={presentation.title}>
      <div className="evaluation-summary-head">
        <div>
          <h4>{presentation.title}</h4>
          <p className="evaluation-summary-context">{presentation.context}</p>
        </div>
        <button
          type="button"
          className="evaluation-summary-details"
          onClick={openDetails}
          disabled={!loaded || loaded.key !== sourceKey || loading}
          title="Open the full evaluation report (per-flight results and charts)"
        >
          {loading ? "Loading…" : "Details"}
        </button>
      </div>
      <dl>
        {presentation.rows.map((summaryRow) => (
          <div key={summaryRow.label} className="evaluation-summary-row">
            <dt>{summaryRow.label}</dt>
            <dd
              className={summaryRow.pending ? "evaluation-summary-pending" : undefined}
              title={summaryRow.pending ? "Evaluation data is not available" : undefined}
            >
              {summaryRow.value}
            </dd>
          </div>
        ))}
      </dl>
      <aside className="evaluation-summary-notes" aria-label="Evaluation notes">
        <strong>Evaluation Notes</strong>
        <p>{presentation.note}</p>
      </aside>
      {error ? <p className="evaluation-summary-error">{error}</p> : null}
      {open && report ? (
        <EvaluationReportWindow
          report={report}
          title={reportTitle}
          subtitle={`${activeAirportCode ?? ""} · ${presentation.context} · ${report.total} trajectories`}
          onClose={() => setOpen(false)}
        />
      ) : null}
    </section>
  );
}
