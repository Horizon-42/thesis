/**
 * ExperimentDetails.tsx
 * ---------------------
 * What one published experiment run IS: why it exists (the run's intent, the campaign's
 * question — stamped by the publisher from the tracked intent registry) and every parameter as
 * a named row under its section (`run_naming.run_parameter_rows`). Shared by the experiment
 * browser's detail pane and the Trajectories panel's summary card; `compact` (the narrow
 * panel) keeps the `Model` rows in view and folds the other sections behind a disclosure.
 */

import type { ExperimentParameterRow } from "../data/airportData";
import type { ExperimentOption } from "../utils/trajectoryResultSources";

/** Where a missing intent is written — the publisher reads it, and refuses to publish without. */
export const INTENT_REGISTRY_PATH = "4dTrajectory/ts_transformer/docs/experiments/intents.json";

/** Rows grouped by section, sections in the order the publisher emitted them. */
function parameterSections(rows: ExperimentParameterRow[]): [string, ExperimentParameterRow[]][] {
  const sections = new Map<string, ExperimentParameterRow[]>();
  for (const row of rows) {
    sections.set(row.section, [...(sections.get(row.section) ?? []), row]);
  }
  return [...sections.entries()];
}

function ParameterSection({ title, rows }: { title: string; rows: ExperimentParameterRow[] }) {
  return (
    <section className="experiment-details-section" aria-label={title}>
      <h5>{title}</h5>
      <dl>
        {rows.map((row) => (
          <div key={row.name} className="experiment-details-row">
            <dt title={row.field ?? row.name}>{row.name}</dt>
            <dd>{row.value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

interface ExperimentDetailsProps {
  experiment: ExperimentOption;
  compact?: boolean;
}

export default function ExperimentDetails({ experiment, compact = false }: ExperimentDetailsProps) {
  const { intent } = experiment;
  const sections = parameterSections(experiment.parameters);
  const shown = compact ? sections.slice(0, 1) : sections;
  const folded = compact ? sections.slice(1) : [];
  const foldedRows = folded.reduce((count, [, rows]) => count + rows.length, 0);

  return (
    <div className={`experiment-details${compact ? " is-compact" : ""}`}>
      {compact ? null : (
        <header className="experiment-details-header">
          <span className="experiment-details-group">
            {experiment.groupTitle}
            {experiment.groupTitle !== experiment.group ? <code>{experiment.group}</code> : null}
          </span>
          <h4>{experiment.runName}</h4>
          {experiment.variantLabel ? (
            <p className="experiment-details-variant">{experiment.variantLabel}</p>
          ) : null}
        </header>
      )}

      {intent ? (
        <div className="experiment-details-intent" aria-label="Experiment intent">
          <p>
            <span className="experiment-details-tag">This run</span>
            {intent.run}
          </p>
          {intent.variant ? (
            <p>
              <span className="experiment-details-tag">This variant</span>
              {intent.variant}
            </p>
          ) : null}
          <p className="experiment-details-campaign">
            <span className="experiment-details-tag">Campaign</span>
            {intent.group}
          </p>
          {intent.design && !compact ? (
            <p className="experiment-details-design">
              Design: <code>{intent.design}</code>
            </p>
          ) : null}
        </div>
      ) : (
        <p className="experiment-details-missing">
          No intent recorded — this run was published before intents were stamped. Add it to{" "}
          <code>{INTENT_REGISTRY_PATH}</code> and refresh the published labels.
        </p>
      )}

      {sections.length === 0 ? (
        // Published before the structured rows existed: the flat run name is all there is.
        <p className="experiment-details-legacy">{experiment.label}</p>
      ) : (
        <div className="experiment-details-parameters">
          {shown.map(([title, rows]) => <ParameterSection key={title} title={title} rows={rows} />)}
          {folded.length > 0 ? (
            <details className="experiment-details-more">
              <summary>All parameters ({foldedRows} more)</summary>
              {folded.map(([title, rows]) => (
                <ParameterSection key={title} title={title} rows={rows} />
              ))}
            </details>
          ) : null}
        </div>
      )}

      <p className="experiment-details-checkpoint">
        <span>Checkpoint</span>
        <code>{experiment.checkpoint}</code>
      </p>
    </div>
  );
}
