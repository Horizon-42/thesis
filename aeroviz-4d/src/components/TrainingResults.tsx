/**
 * TrainingResults.tsx
 * -------------------
 * The two readouts behind the Training overlays, as plain tables in the Training panel: the executor's replay gate
 * (the formal val replay's own table, for this airport and all airports) and the prior's val readout (per column,
 * against the two baselines). Both are copied from the artefacts by the exporters; nothing here is recomputed.
 */

import { formatSeconds, TRAINING_COLUMNS, TRAINING_STRATA } from "../data/trainingSample";
import type { TrainingExecutorOverlay, TrainingGateCell, TrainingPriorOverlay } from "../data/trainingOverlays";
import { checkMark, shortSha } from "../data/trainingText";

function share(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function GateRow({ name, cell }: { name: string; cell: TrainingGateCell }) {
  // a gated cell marks each share it clears or not; a cell reported but not gated marks none
  const mark = (key: "landed" | "words" | "evaluation") => (cell.clears === null ? "" : ` ${checkMark(cell.clears[key])}`);
  return (
    <tr>
      <th scope="row">{name}</th>
      <td>{cell.flights.toLocaleString("en")}</td>
      <td>{share(cell.landed)}{mark("landed")}</td>
      <td>{share(cell.wordsInside)}{mark("words")}</td>
      <td>{share(cell.evaluationPaired)}{mark("evaluation")}</td>
    </tr>
  );
}

const STRATA = [...TRAINING_STRATA, "all"] as const;

export function TrainingExecutorGate({ overlay }: { overlay: TrainingExecutorOverlay }) {
  const airport = overlay.airport;
  const gateShare = `${(overlay.replay.gateShare * 100).toFixed(0)} %`;
  return (
    <details className="training-results" aria-label="The executor's replay gate">
      <summary>The executor's {overlay.replay.split} replay gate · spec {shortSha(overlay.executor.specSha256)}</summary>
      {Object.entries(overlay.gate).map(([group, places]) => {
        const { notGated } = places.here.all;
        return (
          <table key={group} className="training-results-table">
            <caption>
              {group} at {airport} — {notGated === null ? `gated, each share ≥ ${gateShare}` : `reported, ${notGated}`}
            </caption>
            <thead>
              <tr><th scope="col" /><th scope="col">flights</th><th scope="col">landed</th><th scope="col">words</th><th scope="col">eval.</th></tr>
            </thead>
            <tbody>
              {STRATA.flatMap((stratum) => {
                const cell = places.here[stratum];
                return cell ? [<GateRow key={`${airport}-${stratum}`} name={stratum} cell={cell} />] : [];
              })}
              <GateRow name="all airports" cell={places.all.all} />
            </tbody>
          </table>
        );
      })}
      <p className="training-results-note">
        landed: on the pointed runway, as the harvest and evaluation judge a landing · words: of the words the judge
        judged, those flown inside their envelopes, each envelope re-drawn from where the executor was told the word ·
        eval.: evaluation paired — of the flights whose observed track passes evaluation, the replays that pass too. Every
        flyable {overlay.replay.split} flight was flown ({overlay.replay.drawn.flights.toLocaleString("en")}); the table is the
        formal replay's, written {overlay.replay.writtenUtc.slice(0, 16).replace("T", " ")} UTC.
      </p>
    </details>
  );
}

export function TrainingPriorReadout({ overlay, stepS }: { overlay: TrainingPriorOverlay; stepS: number }) {
  const { readout } = overlay;
  const nll = (value: number) => value.toFixed(4);
  const percent = (value: number | null, digits = 1) => (value === null ? "—" : `${(value * 100).toFixed(digits)}%`);
  const runway = readout.firstStepRunway;
  return (
    <details className="training-results" aria-label="The prior's readout">
      <summary>The prior's {readout.split} readout · {nll(readout.model.nllPerStep)} per step against {nll(readout.baselines.repeat.all)} / {nll(readout.baselines.previousWord.all)}</summary>
      <table className="training-results-table">
        <caption>negative log-likelihood per step (nats), lower is better; hover a row for its word changes</caption>
        <thead>
          <tr><th scope="col">column</th><th scope="col">prior</th><th scope="col">repeat</th><th scope="col">prev. word</th></tr>
        </thead>
        <tbody>
          {TRAINING_COLUMNS.map((column) => {
            const own = readout.model.perColumn[column];
            const title =
              `first predicted step: its most likely word is the truth's ${percent(own.firstStepTop1)}; after it, ` +
              `${own.changeSteps.toLocaleString("en")} steps where the truth says a word: the prior gives a word ` +
              `${own.changeProbabilityWhereChanged === null ? "—" : own.changeProbabilityWhereChanged.toFixed(3)} on average there; ` +
              // "first five": MIRROR of `prior.readout.TOP_K` (5), the readout's `top5GivenChange`
              `its most likely word is the one said ${percent(own.top1GivenChange)} of the time, among its first five ` +
              `${percent(own.top5GivenChange)}; on the steps where nothing is said it says a word ` +
              `${percent(own.falseChangeShareWhereKept, 2)} of the time`;
            return (
              <tr key={column} title={title}>
                <th scope="row">{column}</th>
                <td>{nll(own.nllPerStep)}</td>
                <td>{nll(readout.baselines.repeat[column])}</td>
                <td>{nll(readout.baselines.previousWord[column])}</td>
              </tr>
            );
          })}
          <tr>
            <th scope="row">all</th>
            <td>{nll(readout.model.nllPerStep)}</td>
            <td>{nll(readout.baselines.repeat.all)}</td>
            <td>{nll(readout.baselines.previousWord.all)}</td>
          </tr>
        </tbody>
      </table>
      <p className="training-results-note">
        {readout.steps.toLocaleString("en")} predicted {readout.split} steps ({formatSeconds(stepS)} s each), best epoch {readout.bestEpoch}; teacher-forced — every
        step sees the truth sentence before it. The runway at the first predicted step: prior {percent(runway.model.top1)} right
        ({percent(runway.model.direction)} in the right direction), each airport's own frequency {percent(runway.airportFrequency.top1)},{" "}
        {Object.entries(runway.rules).map(([rule, part]) => `${rule.split("_")[0]} ${percent(part.top1)}`).join(", ")}. The baselines are
        counted from train. At the first predicted step, where every column is said, both give each column's word its train
        frequency at that step (the runway: its airport's own frequency over its candidates); after it, "repeat" says a word
        with the column's train frequency, the word by its frequency among changes; "previous word" the word by its train
        frequency after the column's word in force. Prior {shortSha(overlay.prior.checkpointSha256)}: {overlay.prior.parameters.toLocaleString("en")}{" "}
        parameters, d {overlay.prior.model.dModel}, {overlay.prior.model.layers} layers, {overlay.prior.model.heads} heads.
      </p>
    </details>
  );
}
