/**
 * ResultSections.tsx
 * ------------------
 * The details page's sections of the experiment's results (outline §6.2 item 3, D134), from the backend's answer
 * (`data/trainingSetResults.ts`): stage A's labelling and closed loop; stage B's free generation, training, validation,
 * choice and speed; stage C's rounds and checks. A section whose answer has no fields is listed with its reason
 * (`resultSection`). Tables only; nothing is computed here but sums and shares of the numbers shown.
 */

import type { ReactElement, ReactNode } from "react";
import type {
  Counts,
  TrainingChoiceConfiguration,
  TrainingChoiceStep,
  TrainingChoiceVariant,
  TrainingFreeGenerationCell,
  TrainingLabellingSplit,
  TrainingReplayResult,
  TrainingResultSection,
  TrainingRoundResult,
  TrainingSetResultsState,
  TrainingSpeedSetting,
} from "../../data/trainingSetResults";
import type { TrainingDetailsSection } from "./TrainingDetails";

/** A section of the details page from a section of the results: its body, or — no set, loading, the answer absent, the
 *  section without fields — why not. ``own``: what the set itself shows beside the results (its own sentences or rounds,
 *  from its sample): shown under the reason when the results have none. */
export function resultSection<T>(id: string, title: string, state: TrainingSetResultsState,
  part: TrainingResultSection<T> | null, render: (value: T, own: ReactNode) => ReactElement, own?: ReactNode): TrainingDetailsSection {
  const reason = state.status === "none" ? "no set is on screen" : state.status === "loading" ? "the results are loading"
    : state.status === "absent" ? state.problem : part === null ? "not a section of this stage" : part.ok ? null : part.problem;
  if (reason === null) return { id, title, body: render((part as { ok: true; value: T }).value, own) };
  if (own === undefined) return { id, title, body: null, absent: reason };
  return { id, title, body: <><p className="experiment-details-missing">{reason}</p>{own}</> };
}

function share(value: number): string {
  return `${(value * 100).toFixed(1)} %`;
}

function outcomesText(outcomes: Counts): string {
  return Object.entries(outcomes).sort((a, b) => b[1] - a[1]).map(([name, n]) => `${name} ${n}`).join(", ");
}

function landedOf(outcomes: Counts): string {
  const all = Object.values(outcomes).reduce((sum, n) => sum + n, 0);
  return `${outcomes.landed ?? 0}/${all}`;
}

function Table({ label, head, rows }: { label: string; head: ReactNode[]; rows: ReactNode[][] }) {
  return (
    <table className="training-flown-table training-results-table" aria-label={label}>
      <thead><tr>{head.map((cell, i) => <th key={i} scope="col">{cell}</th>)}</tr></thead>
      <tbody>{rows.map((row, r) => <tr key={r}>{row.map((cell, i) => (i === 0 ? <th key={i} scope="row">{cell}</th> : <td key={i}>{cell}</td>))}</tr>)}</tbody>
    </table>
  );
}

// ── stage A ──────────────────────────────────────────────────────────────────
export function LabellingSection({ splits }: { splits: Record<string, TrainingLabellingSplit> }) {
  const airports = [...new Set(Object.values(splits).flatMap((split) => Object.keys(split.labelledByAirport)))].sort();
  return (
    <>
      <p className="training-details-lede">The labeller's reading of the artefact's flights, train and select (the val days are not shown).</p>
      <Table label="Labelled and refused, by airport" head={["split", "labelled", "refused", ...airports]}
        rows={Object.entries(splits).map(([name, split]) => [name, split.labelled, split.refused,
          ...airports.map((code) => `${split.labelledByAirport[code] ?? 0} · ${Object.values(split.refusedByAirport[code] ?? {}).reduce((a, b) => a + b, 0)} refused`)])} />
      <Table label="Why flights were refused" head={["reason", ...Object.keys(splits)]}
        rows={[...new Set(Object.values(splits).flatMap((split) => Object.keys(split.refusalReasons)))].map((reason) =>
          [reason, ...Object.values(splits).map((split) => split.refusalReasons[reason] ?? 0)])} />
    </>
  );
}

export function ClosedLoopSection({ replays, airport }: { replays: TrainingReplayResult[]; airport: string }) {
  return (
    <>
      <p className="training-details-lede">
        The executor's replay of every closed-loop sentence of the set's splits at each Δ (the formal replay of the artefact),
        at {airport} and at all airports, by the flight's dynamics.
      </p>
      <Table label="The closed loop's replay" head={["split · Δ", "dynamics", `landed at ${airport}`, `outcomes at ${airport}`, "landed, all airports"]}
        rows={replays.flatMap((replay) => Object.entries(replay.groups).map(([group, at]) => [
          `${replay.split} · ${replay.intervalS} s`, group,
          at[airport] === undefined ? "–" : `${landedOf(at[airport].outcomes)} (${share(at[airport].landed)})`,
          at[airport] === undefined ? "–" : outcomesText(at[airport].outcomes),
          `${landedOf(at.all.outcomes)} (${share(at.all.landed)})`]))} />
    </>
  );
}

// ── stage B ──────────────────────────────────────────────────────────────────
const SIDE_TEXT: Record<string, string> = {
  inside: "inside the prior's selection",
  outside_fault: "outside it: a faulty observed track",
  outside_outcome: "outside it: the stored outcome",
};

export function FreeGenerationSection({ split, selection, sides, own }: {
  split: string; selection: string; sides: Record<string, Record<string, Record<string, TrainingFreeGenerationCell>>>; own: ReactNode;
}) {
  return (
    <>
      <p className="training-details-lede">
        The free-generation readout the set was made from: every sentence the prior said on the {split} days (selection
        "{selection}"), by airport and stratum, apart for the flights its selection keeps and those it leaves out.
      </p>
      {Object.entries(sides).map(([side, airports]) => (
        <section key={side}>
          <h4 className="training-details-subhead">{SIDE_TEXT[side] ?? side}</h4>
          <Table label={`Free generation, ${side}`} head={["airport · stratum", "landed", "outcomes", "go-arounds", "timed out", "heading words (labelled)"]}
            rows={Object.entries(airports).flatMap(([code, strata]) => Object.entries(strata).map(([stratum, cell]) => [
              `${code} · ${stratum}`, landedOf(cell.outcomes), outcomesText(cell.outcomes), cell.goArounds, cell.timedOut,
              `${cell.wordsPerSentence.heading.toFixed(1)} (${cell.labelledWordsPerSentence.heading.toFixed(1)})`]))} />
        </section>
      ))}
      <h4 className="training-details-subhead">This set's own sentences</h4>
      {own}
    </>
  );
}

export function TrainingSection({ bestEpoch, epochs }: { bestEpoch: number; epochs: Array<{ epoch: number; trainLossPerStep: number; selectLossPerStep: number }> }) {
  return (
    <>
      <p className="training-details-lede">The prior's loss per step at each epoch, on train and select; the epoch kept is {bestEpoch}.</p>
      <Table label="Loss per epoch" head={["epoch", "train", "select"]}
        rows={epochs.map((e) => [e.epoch === bestEpoch ? `${e.epoch} (kept)` : e.epoch, e.trainLossPerStep.toFixed(4), e.selectLossPerStep.toFixed(4)])} />
    </>
  );
}

export function ValidationSection({ lossPerStep, perColumn, masksOnLabelledWords }: {
  lossPerStep: number; perColumn: Counts; masksOnLabelledWords: Record<string, Record<string, Counts>>;
}) {
  const maskColumns = [...new Set(Object.values(masksOnLabelledWords).flatMap((airports) => Object.values(airports).flatMap(Object.keys)))];
  return (
    <>
      <p className="training-details-lede">The base's teacher-forced loss on the val days: {lossPerStep.toFixed(4)} per step.</p>
      <Table label="Loss per column" head={["column", "loss per step"]} rows={Object.entries(perColumn).map(([c, v]) => [c, v.toFixed(4)])} />
      <h4 className="training-details-subhead">Share of the labelled words the procedure masks block</h4>
      <Table label="Masks on the labelled words" head={["side · airport", ...maskColumns]}
        rows={Object.entries(masksOnLabelledWords).flatMap(([side, airports]) => Object.entries(airports).map(([code, columns]) =>
          [`${side} · ${code}`, ...maskColumns.map((column) => (column in columns ? share(columns[column]) : "–"))]))} />
    </>
  );
}

function ChoiceTable({ name, step, said }: { name: string; step: TrainingChoiceStep; said: string }) {
  const folds = [...new Set(Object.values(step.arms).flatMap((arm) => Object.keys(arm.folds)))].sort();
  return (
    <>
      <h4 className="training-details-subhead">The {name}: {step.chosen} ({said}, seed scale {step.seedScale.toFixed(4)})</h4>
      <Table label={`The choice of the ${name}`} head={["arm", "score", ...folds]}
        rows={Object.entries(step.arms).map(([arm, value]) => [arm, value.score.toFixed(4), ...folds.map((f) => value.folds[f]?.toFixed(4) ?? "–")])} />
    </>
  );
}

export function ChoiceSection({ configuration, variant }: { configuration: TrainingChoiceConfiguration; variant: TrainingChoiceVariant }) {
  return (
    <>
      <p className="training-details-lede">The campaign's two choices, each arm's score over its folds (the held-out airports).</p>
      <ChoiceTable name="configuration" step={configuration}
        said={`best ${configuration.bestScore.toFixed(4)}, within twice the seed scale of it: ${configuration.within.join(", ")}`} />
      <ChoiceTable name="variant" step={variant}
        said={Object.entries(variant.scores).map(([name, score]) => `${name} ${score.toFixed(4)}`).join(", ")} />
    </>
  );
}

function ms(values: Counts): string {
  return `${values.p50.toFixed(1)} / ${values.p95.toFixed(1)} / ${values.max.toFixed(1)}`;
}

export function SpeedSection({ model, split, warmupRows, smoke, settings }: {
  model: string; split: string; warmupRows: number; smoke: boolean; settings: TrainingSpeedSetting[];
}) {
  return (
    <>
      <p className="training-details-lede">
        How fast the model speaks: <code>{model}</code> timed on {split}-day items (the first {warmupRows} rows not
        timed){smoke ? " — a SMOKE readout" : ""}. Each row of the table is one setting; times are p50 / p95 / largest.
      </p>
      <Table label="The model's speed" head={["setting", "prior's step of a row (ms)", "executor's steps (ms)", "a row (ms)",
        "a sentence (s)", "flight rows a second", "a row's share of Δ (p50)"]}
        rows={settings.map((s) => [
          `${s.device}${s.host.name === undefined ? "" : ` (${s.host.name})`}, ${s.host.threads} thread${s.host.threads === 1 ? "" : "s"}, batch ${s.batch} (${s.loops} loops, ${s.rowsTimed} rows)`,
          ms(s.priorStepMs), ms(s.executorStepsMs), ms(s.rowMs),
          `${s.sentenceS.p50.toFixed(2)} / ${s.sentenceS.p95.toFixed(2)} / ${s.sentenceS.max.toFixed(2)}`,
          s.flightRowsPerS.toFixed(0), `${share(s.shareOfInterval.row.p50)} of ${s.shareOfInterval.intervalS} s`])} />
    </>
  );
}

// ── stage C ──────────────────────────────────────────────────────────────────
export function RoundsSection({ started, rounds, own }: { started: string; rounds: TrainingRoundResult[]; own: ReactNode }) {
  return (
    <>
      <p className="training-details-lede">The campaign (started {started}): each round, the windows it spoke and its selection readout.</p>
      {rounds.length === 0 ? <p className="experiment-details-missing">The campaign holds no round yet.</p> : (
        <Table label="The campaign's rounds" head={["round", "spoken: landed", "spoken: outcomes", "spoken: reward", "selection: landed", "selection: mean reward"]}
          rows={rounds.map((r) => {
            const selected = Object.values(r.selection);
            const outcomes = selected.reduce<Counts>((all, cell) => {
              for (const [name, n] of Object.entries(cell.outcomes)) all[name] = (all[name] ?? 0) + n;
              return all;
            }, {});
            const windows = selected.reduce((sum, cell) => sum + cell.windows, 0);
            return [`r${r.round}`, landedOf(r.speaking.outcomes), outcomesText(r.speaking.outcomes),
              `${r.speaking.rewardSum.toFixed(2)} over ${r.speaking.windows}`, landedOf(outcomes),
              (selected.reduce((sum, cell) => sum + cell.rewardMean * cell.windows, 0) / Math.max(windows, 1)).toFixed(3)];
          })} />
      )}
      <h4 className="training-details-subhead">This set's own rounds</h4>
      {own}
    </>
  );
}

export function ChecksSection({ checks }: { checks: Record<string, unknown> }) {
  return (
    <>
      <p className="training-details-lede">The checks the campaign ran at its start (the labeller's, the executor's, the closed loop's), as it recorded them.</p>
      <pre className="training-details-pre">{JSON.stringify(checks, null, 1)}</pre>
    </>
  );
}
