/**
 * TrainingResults.tsx
 * -------------------
 * The readouts behind the Training overlays, as the details page's sections (`TrainingDetails`): the executor's replay
 * gate (the formal val replay's own table, for this airport and all airports), the prior's val readout (per column,
 * against the two baselines, and what it says where the truth changes a word) and the models' own sentences (how many
 * landed: on this set's flights, counted from the samples on screen, beside each model's formal val free generation; and
 * how each model's sentences were drawn and flown). The formal numbers are copied from the artefacts by the exporters;
 * the only thing counted here is the set's own samples.
 *
 * Each readout also has a one-line SUMMARY — its conclusion, without its name — which is what the panel lists beside the
 * readout's name under "Details".
 */

import { formatSeconds, TRAINING_COLUMNS, TRAINING_STRATA, type TrainingFlight } from "../data/trainingSample";
import {
  generationLanded,
  trainingModelGroups,
  trainingRunName,
  type TrainingExecutorOverlay,
  type TrainingGateCell,
  type TrainingGenerationCell,
  type TrainingGenerationOverlay,
  type TrainingGenerationReadoutCells,
  type TrainingGenerationSetCells,
  type TrainingPriorOverlay,
} from "../data/trainingOverlays";
import { checkMark, shortSha, trainingModelOrigin, trainingModelText } from "../data/trainingText";
import { trainingModelColour } from "../utils/trainingWordColors";

function share(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function utcText(written: string): string {
  return `${written.slice(0, 16).replace("T", " ")} UTC`;
}

/** A share, and beside it — muted — whether its gate clears. */
function ShareCell({ value, clears }: { value: number | null; clears: boolean | null }) {
  return (
    <td>
      {share(value)}
      {clears === null ? null : (
        <span className={`training-results-mark ${clears ? "is-pass" : "is-fail"}`} role="img"
          aria-label={clears ? "clears the gate" : "misses the gate"}>{checkMark(clears)}</span>
      )}
    </td>
  );
}

function GateRow({ name, cell, total }: { name: string; cell: TrainingGateCell; total: boolean }) {
  // a gated cell marks each share it clears or not; a cell reported but not gated marks none
  const clears = (key: "landed" | "words" | "evaluation") => (cell.clears === null ? null : cell.clears[key]);
  return (
    <tr className={total ? "training-results-total" : undefined}>
      <th scope="row">{name}</th>
      <td>{cell.flights.toLocaleString("en")}</td>
      <ShareCell value={cell.landed} clears={clears("landed")} />
      <ShareCell value={cell.wordsInside} clears={clears("words")} />
      <ShareCell value={cell.evaluationPaired} clears={clears("evaluation")} />
    </tr>
  );
}

const STRATA = [...TRAINING_STRATA, "all"] as const;

export function executorGateSummary(overlay: TrainingExecutorOverlay): string {
  return `${overlay.replay.split} · spec ${shortSha(overlay.executor.specSha256)}`;
}

export function TrainingExecutorGate({ overlay }: { overlay: TrainingExecutorOverlay }) {
  const airport = overlay.airport;
  const gateShare = `${(overlay.replay.gateShare * 100).toFixed(0)} %`;
  return (
    <>
      <p className="training-details-lede">
        Executor spec <code>{shortSha(overlay.executor.specSha256)}</code> on its {overlay.replay.split} replay: every
        flyable {overlay.replay.split} flight flown ({overlay.replay.drawn.flights.toLocaleString("en")}), the table the
        formal replay's own, written {utcText(overlay.replay.writtenUtc)}.
      </p>
      {Object.entries(overlay.gate).map(([group, places]) => {
        const { notGated } = places.here.all;
        return (
          <table key={group} className="training-results-table">
            <caption>
              <strong>{group}</strong> at {airport}
              <span className="training-results-caption-note">
                {notGated === null ? `gated — each share ≥ ${gateShare}` : `reported, ${notGated}`}
              </span>
            </caption>
            <thead>
              <tr>
                <th scope="col">approach</th><th scope="col">flights</th><th scope="col">landed</th>
                <th scope="col">words inside</th><th scope="col">evaluation paired</th>
              </tr>
            </thead>
            <tbody>
              {STRATA.flatMap((stratum) => {
                const cell = places.here[stratum];
                return cell ? [<GateRow key={`${airport}-${stratum}`} name={stratum} cell={cell} total={stratum === "all"} />] : [];
              })}
              <GateRow name="all airports" cell={places.all.all} total />
            </tbody>
          </table>
        );
      })}
      <dl className="training-details-terms">
        <div><dt>landed</dt><dd>on the pointed runway, as the harvest and evaluation judge a landing</dd></div>
        <div><dt>words inside</dt><dd>of the words the judge judged, those flown inside their envelopes, each envelope
          re-drawn from where the executor was told the word</dd></div>
        <div><dt>evaluation paired</dt><dd>of the flights whose observed track passes evaluation, the replays that pass
          too</dd></div>
      </dl>
    </>
  );
}

const nll = (value: number) => value.toFixed(4);
const percent = (value: number | null, digits = 1) => (value === null ? "—" : `${(value * 100).toFixed(digits)}%`);

export function priorReadoutSummary(overlay: TrainingPriorOverlay): string {
  const { readout } = overlay;
  return `${readout.split} · ${nll(readout.model.nllPerStep)} per step against ` +
    `${nll(readout.baselines.repeat.all)} / ${nll(readout.baselines.previousWord.all)}`;
}

/** A row of the likelihood table: the prior and the two baselines, the lowest of the three marked. */
function NllRow({ name, values, total }: { name: string; values: [number, number, number]; total: boolean }) {
  const best = Math.min(...values);
  return (
    <tr className={total ? "training-results-total" : undefined}>
      <th scope="row">{name}</th>
      {values.map((value, index) => (
        <td key={index} className={value === best ? "training-results-best" : undefined}>{nll(value)}</td>
      ))}
    </tr>
  );
}

export function TrainingPriorReadout({ overlay, stepS }: { overlay: TrainingPriorOverlay; stepS: number }) {
  const { readout } = overlay;
  const runway = readout.firstStepRunway;
  return (
    <>
      <p className="training-details-lede">
        {readout.steps.toLocaleString("en")} predicted {readout.split} steps ({formatSeconds(stepS)} s each), best
        epoch {readout.bestEpoch}; teacher-forced — every step sees the truth sentence before it. Prior{" "}
        <code>{shortSha(overlay.prior.checkpointSha256)}</code>: {overlay.prior.parameters.toLocaleString("en")} parameters,
        d {overlay.prior.model.dModel}, {overlay.prior.model.layers} layers, {overlay.prior.model.heads} heads.
      </p>
      <table className="training-results-table">
        <caption>
          <strong>Negative log-likelihood per step</strong>
          <span className="training-results-caption-note">nats, lower is better; the lowest of each row is bold</span>
        </caption>
        <thead>
          <tr><th scope="col">column</th><th scope="col">prior</th><th scope="col">repeat</th><th scope="col">previous word</th></tr>
        </thead>
        <tbody>
          {TRAINING_COLUMNS.map((column) => (
            <NllRow key={column} name={column} total={false} values={[readout.model.perColumn[column].nllPerStep,
              readout.baselines.repeat[column], readout.baselines.previousWord[column]]} />
          ))}
          <NllRow name="all" total values={[readout.model.nllPerStep, readout.baselines.repeat.all, readout.baselines.previousWord.all]} />
        </tbody>
      </table>
      <table className="training-results-table">
        <caption>
          <strong>Where the truth says a word</strong>
          <span className="training-results-caption-note">the prior's own words, column by column</span>
        </caption>
        <thead>
          <tr>
            <th scope="col">column</th>
            <th scope="col">first step top-1</th>
            <th scope="col">word changes</th>
            <th scope="col">p(a word) there</th>
            <th scope="col">top-1 there</th>
            <th scope="col">top-5 there</th>
            <th scope="col">says a word where none is</th>
          </tr>
        </thead>
        <tbody>
          {TRAINING_COLUMNS.map((column) => {
            const own = readout.model.perColumn[column];
            return (
              <tr key={column}>
                <th scope="row">{column}</th>
                <td>{percent(own.firstStepTop1)}</td>
                <td>{own.changeSteps.toLocaleString("en")}</td>
                <td>{own.changeProbabilityWhereChanged === null ? "—" : own.changeProbabilityWhereChanged.toFixed(3)}</td>
                <td>{percent(own.top1GivenChange)}</td>
                <td>{percent(own.top5GivenChange)}</td>
                <td>{percent(own.falseChangeShareWhereKept, 2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <dl className="training-details-terms">
        <div><dt>first step top-1</dt><dd>at the first predicted step, its most likely word is the truth's</dd></div>
        <div><dt>word changes</dt><dd>after the first step, the steps where the truth says a word</dd></div>
        <div><dt>p(a word) there</dt><dd>the probability it gives saying a word there, on average</dd></div>
        <div><dt>top-1 there</dt><dd>its most likely word is the one said</dd></div>
        {/* "first five": MIRROR of `prior.readout.TOP_K` (5), the readout's `top5GivenChange` */}
        <div><dt>top-5 there</dt><dd>the word said is among its first five</dd></div>
        <div><dt>says a word where none is</dt><dd>on the steps where nothing is said, how often it says a word</dd></div>
      </dl>
      <table className="training-results-table">
        <caption>
          <strong>The runway at the first predicted step</strong>
          <span className="training-results-caption-note">right, against each airport's own frequency and the rules; the direction is the prior's alone</span>
        </caption>
        <thead><tr><th scope="col">who</th><th scope="col">right</th><th scope="col">right direction</th></tr></thead>
        <tbody>
          <tr><th scope="row">prior</th><td>{percent(runway.model.top1)}</td><td>{percent(runway.model.direction)}</td></tr>
          <tr><th scope="row">airport frequency</th><td>{percent(runway.airportFrequency.top1)}</td><td /></tr>
          {Object.entries(runway.rules).map(([rule, part]) => (
            <tr key={rule}><th scope="row">{rule.split("_")[0]}</th><td>{percent(part.top1)}</td><td /></tr>
          ))}
        </tbody>
      </table>
      <p className="training-results-note">
        The baselines are counted from train. At the first predicted step, where every column is said, both give each
        column's word its train frequency at that step (the runway: its airport's own frequency over its candidates); after
        it, "repeat" says a word with the column's train frequency, the word by its frequency among changes; "previous word"
        the word by its train frequency after the column's word in force.
      </p>
    </>
  );
}

/** A landed share, its count beneath it. */
function LandedCell({ cell }: { cell: TrainingGenerationCell | null }) {
  if (cell === null) return <td className="training-results-empty">—</td>;
  return (
    <td>
      {(cell.landed * 100).toFixed(1)}%
      <span className="training-results-count"> of {cell.flights.toLocaleString("en")}</span>
    </td>
  );
}

const READ_STRATA = ["all", ...TRAINING_STRATA] as const;

/** A model's sentences in the views' order (by name in training order, then run, then round), each named as the bar and
 *  the panel name it. */
function namedModels(published: TrainingGenerationOverlay[]) {
  return trainingModelGroups(published, (overlay) => overlay)
    .flatMap((group) => group.members.map((overlay) => ({ overlay, label: group.memberLabel(overlay) })));
}

export function generationSummary(published: TrainingGenerationOverlay[], flights: TrainingFlight[]): string {
  return namedModels(published).map(({ overlay, label }) => {
    const own = generationLanded(overlay, flights).all;
    return `${label} ${own === null ? "—" : `${(own.landed * 100).toFixed(0)}%`}` +
      (overlay.readout === null ? "" : ` (${overlay.readout.split} ${(overlay.readout.prior.all.all.landed * 100).toFixed(1)}%)`);
  }).join(" · ");
}

/** How the sentences were drawn and flown, one field of the models' table. A field every model shares is said once
 *  above the table instead of repeated down a column. */
interface DrawnField {
  name: string;
  value: (overlay: TrainingGenerationOverlay) => string;
}

const DRAWN_FIELDS: DrawnField[] = [
  { name: "run", value: (overlay) => trainingRunName(overlay.model.run) },
  { name: "trained", value: ({ model }) => trainingModelOrigin(model) },
  { name: "procedure masks", value: ({ generation }) => (generation.procedureMasks.length === 0 ? "none (the vocabulary's rules alone)"
    : generation.procedureMasks.map((item) => item.name).join(", ")) },
  { name: "samples a flight", value: ({ generation }) => String(generation.samples) },
  { name: "temperature", value: ({ generation }) => String(generation.temperature) },
  { name: "seed", value: ({ generation }) => String(generation.seed) },
  { name: "executor spec", value: ({ generation }) => shortSha(generation.executor.specSha256) },
  { name: "time limit", value: ({ generation }) => `${generation.executor.timeoutFactor}× the observed remaining time` },
  { name: "formal readout", value: ({ readout }) => (readout === null ? "not given"
    : `${readout.drawn.flights.toLocaleString("en")} ${readout.split} flights ` +
      `(${readout.drawn.perAirport === 0 ? "every flight" : `${readout.drawn.perAirport} an airport`}), written ${utcText(readout.writtenUtc)}`) },
];

export function TrainingGenerationReadout({ overlays: published, flights }: { overlays: TrainingGenerationOverlay[]; flights: TrainingFlight[] }) {
  const named = namedModels(published);
  const airport = published[0].airport;
  const common = DRAWN_FIELDS.filter((field) => new Set(published.map(field.value)).size === 1);
  const varying = DRAWN_FIELDS.filter((field) => !common.includes(field));
  // with no formal readout given, every row is "this set": the column would say nothing
  const withReadout = published.some((overlay) => overlay.readout !== null);
  const rowsOf = (overlay: TrainingGenerationOverlay): Array<[string, TrainingGenerationReadoutCells | TrainingGenerationSetCells]> => [
    ["this set", generationLanded(overlay, flights)],
    ...(overlay.readout === null ? [] : [
      [`${overlay.readout.split} · ${airport}`, overlay.readout.prior.here],
      [`${overlay.readout.split} · all airports`, overlay.readout.prior.all],
      [`labelled words · ${airport}`, overlay.readout.labelled.here],
      ["labelled words · all", overlay.readout.labelled.all],
    ] as Array<[string, TrainingGenerationReadoutCells]>),
  ];
  return (
    <>
      <p className="training-details-lede">
        Each model speaks from its first predicted step (the steps before are observed), a sentence of its own, and the
        executor flies each step as it is said. Landed: on the runway pointed at the end, as the executor's judge reads a
        landing. Only flights on their own aircraft dynamics are flown, as in the formal readout.
      </p>
      <table className="training-results-table training-results-models">
        <caption>
          <strong>Landed</strong>
          <span className="training-results-caption-note">
            {withReadout ? "each sample on this set's flights, and in the model's formal free generation"
              : "each sample on this set's flights (no formal readout was given)"}
          </span>
        </caption>
        <thead>
          <tr>
            <th scope="col">model</th>{withReadout ? <th scope="col" className="training-results-where">sentences of</th> : null}
            {READ_STRATA.map((key) => <th key={key} scope="col">{key}</th>)}
          </tr>
        </thead>
        {named.map(({ overlay, label }) => {
          const rows = rowsOf(overlay);
          return (
            <tbody key={overlay.overlayId}>
              {rows.map(([where, cells], index) => (
                <tr key={where}>
                  {index === 0 ? (
                    <th scope="rowgroup" rowSpan={rows.length} title={trainingModelText(overlay.model)}>
                      <span className="training-model-swatch" style={{ background: trainingModelColour(overlay.model) }} />
                      {label}
                    </th>
                  ) : null}
                  {withReadout ? <td className="training-results-where">{where}</td> : null}
                  {READ_STRATA.map((key) => <LandedCell key={key} cell={cells[key]} />)}
                </tr>
              ))}
            </tbody>
          );
        })}
      </table>
      <p className="training-results-note">
        "labelled words": the truth sentence flown the same way from the same step — how far the executor alone gets. Each
        model speaks as it was trained to: under the vocabulary's rules and the procedure's masks it was post-trained under,
        if any; under the procedure's altitudes (the floor before the join, no climbing back, the glidepath's lower edge) a
        sentence stops at the first step that sinks below the edge — "below glidepath".
      </p>

      <h4 className="training-details-subhead">How each model's sentences were drawn and flown</h4>
      {common.length > 0 ? (
        <dl className="training-details-facts" aria-label="Shared by every model">
          {common.map((field) => (
            <div key={field.name}><dt>{field.name}</dt><dd>{field.value(published[0])}</dd></div>
          ))}
        </dl>
      ) : null}
      {varying.length > 0 ? (
        <table className="training-results-table training-results-text">
          <thead>
            <tr><th scope="col">model</th>{varying.map((field) => <th key={field.name} scope="col">{field.name}</th>)}</tr>
          </thead>
          <tbody>
            {named.map(({ overlay, label }) => (
              <tr key={overlay.overlayId}>
                <th scope="row">
                  <span className="training-model-swatch" style={{ background: trainingModelColour(overlay.model) }} />
                  {label}
                </th>
                {varying.map((field) => <td key={field.name}>{field.value(overlay)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </>
  );
}
