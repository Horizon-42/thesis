/**
 * EvaluationSummary.tsx
 * ---------------------
 * Subject-aware evaluation summary for the active Observe comparison category.
 *
 * The three producers answer different questions, so they deliberately do not share
 * one generic set of "solve/success" labels:
 *   • observed       — quality of the measured final approach at the runway threshold;
 *   • optimization   — solver outcome and error relative to the selected target;
 *   • data-driven    — runway-threshold gate pass plus ADE/FDE against the observed trajectory.
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
  type EvaluationBatchStats,
  type OptimizationStats,
  type PredictionAccuracyStats,
  type PredictionErrorSpread,
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
  evaluation: EvaluationBatchStats | null;
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
  if (category.resultSource === "experiment" || category.key.startsWith("ts_")) {
    return "dataDriven";
  }
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
  const eventRate = report?.observed?.event_estimated_rate;
  const crossingPassRate = report && report.total > 0
    ? report.verdict_counts.pass / report.total
    : null;
  const lateralMean = report?.lateral_m?.mean;
  const verticalMeanAbs = report?.vertical_m?.mean_abs;

  return {
    title: "Observed Baseline Evaluation",
    context: "Observed ADS-B trajectories",
    note:
      "Runway assignment produces one policy-free threshold-event estimate from the final " +
      "approach fit; evaluation consumes that estimate without refitting ADS-B. LPV vertical " +
      "uses the published-TCH path with the 7.5 m half-FSD threshold bound.",
    rows: [
      row(
        "Threshold-event availability",
        formatPercent(eventRate ?? null),
        eventRate != null,
      ),
      row(
        "Terminal-verdict pass rate",
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
  stats: OptimizationStats | null,
  evaluation: EvaluationBatchStats | null,
): Presentation {
  const targetPassRate = evaluation?.successRate ?? passRateAmongSolved(stats);
  const formatCount = (value: number | null | undefined) =>
    value == null || !Number.isFinite(value) ? "—" : Math.round(value).toLocaleString();
  const formatScalar = (
    value: number | null | undefined,
    unit: string,
    digits = 1,
  ) => value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(digits)} ${unit}`;
  const optionalRow = (
    label: string,
    value: number | null | undefined,
    formatted: string,
  ): SummaryRow[] => value == null ? [] : [row(label, formatted, true)];
  const spreadRows = (name: string, spread: PredictionErrorSpread | null | undefined) => [
    ...optionalRow(`Median ${name}`, spread?.median, formatMetres(spread?.median)),
    ...optionalRow(`Mean ${name}`, spread?.mean, formatMetres(spread?.mean)),
    ...optionalRow(
      `95th-percentile ${name}`,
      spread?.p95,
      formatMetres(spread?.p95),
    ),
    ...optionalRow(`Maximum ${name}`, spread?.max, formatMetres(spread?.max)),
  ];
  const raw = prediction?.rawKinematics?.predicted;
  const lateral = evaluation?.lateralM;
  const vertical = evaluation?.verticalM;
  const evaluatedTime = evaluation?.finalTimeS;

  return {
    title: category.resultSource === "experiment"
      ? "Experiment Evaluation"
      : "Data-Driven Model Evaluation",
    context: category.label,
    note:
      "ADE is the mean position error across the predicted trajectory; FDE is the position " +
      "error at the final predicted sample. Both compare the prediction with the observed " +
      "track over their overlapping samples. The runway-threshold pass rate grades the " +
      "prediction's final state against both evaluation gates. Train results are in-sample; " +
      "validation results are the development/model-selection partition. A solver solve rate " +
      "does not apply.",
    rows: [
      row("Trajectories", formatCount(prediction?.flights), prediction?.flights != null),
      row(
        "Without temporal overlap",
        formatCount(prediction?.flightsWithoutOverlap),
        prediction?.flightsWithoutOverlap != null,
      ),
      row(
        "Runway-threshold pass rate",
        formatPercent(targetPassRate),
        targetPassRate != null,
      ),
      ...optionalRow(
        "Indeterminate terminal verdicts",
        evaluation?.indeterminate,
        formatCount(evaluation?.indeterminate),
      ),
      ...spreadRows("ADE", prediction?.adeM),
      ...spreadRows("FDE", prediction?.fdeM),
      ...optionalRow(
        "Final-time error MAE",
        prediction?.finalTimeS?.mae,
        formatScalar(prediction?.finalTimeS?.mae, "s"),
      ),
      ...optionalRow(
        "Final-time error p95",
        prediction?.finalTimeS?.p95Abs,
        formatScalar(prediction?.finalTimeS?.p95Abs, "s"),
      ),
      ...optionalRow(
        "Final-time signed bias",
        prediction?.finalTimeS?.meanSigned,
        formatScalar(prediction?.finalTimeS?.meanSigned, "s"),
      ),
      ...optionalRow(
        "Mean per-flight cross-track p95",
        prediction?.crossTrackP95M?.mean,
        formatMetres(prediction?.crossTrackP95M?.mean),
      ),
      ...optionalRow(
        "Mean per-flight altitude p95",
        prediction?.altitudeP95M?.mean,
        formatMetres(prediction?.altitudeP95M?.mean),
      ),
      ...optionalRow(
        "Mean lateral deviation at threshold",
        lateral?.mean,
        formatMetres(lateral?.mean),
      ),
      ...optionalRow(
        "Lateral deviation p95 at threshold",
        lateral?.p95,
        formatMetres(lateral?.p95),
      ),
      ...optionalRow(
        "Mean absolute vertical deviation at threshold",
        vertical?.mean_abs,
        formatMetres(vertical?.mean_abs),
      ),
      ...optionalRow(
        "Vertical deviation p95 at threshold",
        vertical?.p95_abs,
        formatMetres(vertical?.p95_abs),
      ),
      ...optionalRow(
        "Mean predicted duration",
        evaluatedTime?.mean,
        formatScalar(evaluatedTime?.mean, "s"),
      ),
      ...optionalRow(
        "Position–velocity consistency p95",
        raw?.positionVelocityRmseMps?.p95,
        formatScalar(raw?.positionVelocityRmseMps?.p95, "m/s"),
      ),
      ...optionalRow(
        "Heading consistency p95",
        raw?.headingConsistencyP95Deg?.p95,
        formatScalar(raw?.headingConsistencyP95Deg?.p95, "deg"),
      ),
      ...optionalRow(
        "Turn-rate p95",
        raw?.turnRateP95DegS?.p95,
        formatScalar(raw?.turnRateP95DegS?.p95, "deg/s"),
      ),
      ...optionalRow(
        "Acceleration p95",
        raw?.accelerationP95Mps2?.p95,
        formatScalar(raw?.accelerationP95Mps2?.p95, "m/s²"),
      ),
      ...optionalRow(
        "Jerk p95",
        raw?.jerkP95Mps3?.p95,
        formatScalar(raw?.jerkP95Mps3?.p95, "m/s³"),
      ),
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
            evaluation: null,
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
            evaluation: data.evaluation ?? null,
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
        loaded?.key === sourceKey ? loaded.stats : null,
        loaded?.key === sourceKey ? loaded.evaluation : null,
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
  const detailsAvailable = category?.resultSource !== "experiment";

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
        {detailsAvailable ? (
          <button
            type="button"
            className="evaluation-summary-details"
            onClick={openDetails}
            disabled={!loaded || loaded.key !== sourceKey || loading}
            title="Open the full evaluation report (per-flight results and charts)"
          >
            {loading ? "Loading…" : "Details"}
          </button>
        ) : (
          <span className="evaluation-summary-summary-only">Aggregate statistics</span>
        )}
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
