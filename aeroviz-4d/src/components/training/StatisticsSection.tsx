/**
 * StatisticsSection.tsx
 * ---------------------
 * The details page's "The models' statistics" (frontend §3 item 3, D159): one table, a row for each sentence kind of the
 * set with its colour's swatch beside its name (the name in the body text's colour, §3 item 11), and two groups of
 * columns — this set, the formal readout (`data/trainingStatistics.ts`). Each cell is a share with its count beneath; a
 * count the readout does not hold is "—". The go-arounds are the go-arounds said per sentence and the mean reward of a
 * window stage a mean, each with its count beneath (frontend D160 (11)).
 */

import type { ReactNode } from "react";
import { TRAINING_SENTENCE_COLOR } from "../../utils/trainingWordColors";
import type { TrainingSetResultsState } from "../../data/trainingSetResults";
import type { TrainingStatsCells, TrainingStatsTable } from "../../data/trainingStatistics";

/** A number per sentence and its count beneath, or "—". */
function PerSentence({ count, of }: { count: number | null; of: number }) {
  if (count === null || of === 0) return <>—</>;
  return <><span className="training-stats-share">{(count / of).toFixed(3)}</span><span className="training-stats-count">{count}/{of}</span></>;
}

/** A share and its count beneath, or "—". */
function Share({ count, of }: { count: number | null; of: number }) {
  if (count === null || of === 0) return <>—</>;
  return <><span className="training-stats-share">{((count / of) * 100).toFixed(1)} %</span><span className="training-stats-count">{count}/{of}</span></>;
}

function cells(group: TrainingStatsCells | null, windows: boolean): ReactNode[] {
  const columns: ReactNode[] = [
    <Share count={group?.landed ?? null} of={group?.sentences ?? 0} />,
    <PerSentence count={group?.goArounds ?? null} of={group?.sentences ?? 0} />,
    <Share count={group?.timedOut ?? null} of={group?.sentences ?? 0} />,
  ];
  if (!windows) return columns;
  const reward = group === null || group.rewardSum === null || group.sentences === 0 ? <>—</> : (
    <><span className="training-stats-share">{(group.rewardSum / group.sentences).toFixed(3)}</span>
      <span className="training-stats-count">over {group.sentences}</span></>
  );
  return [...columns, <Share count={group?.lost ?? null} of={group?.sentences ?? 0} />, reward];
}

export default function StatisticsSection({ table, results }: { table: TrainingStatsTable; results: TrainingSetResultsState }) {
  const columns = ["landed", "go-arounds / sentence", "timed out", ...(table.windows ? ["lost separation", "mean reward"] : [])];
  const readoutNote = results.status === "ready" ? table.readoutProblem : results.status === "absent" ? results.problem
    : results.status === "loading" ? "the formal readout is loading" : "no set is on screen";
  return (
    <>
      <p className="training-details-lede">
        Each kind of sentence of the set, in the order of the bar's tabs: on the set's own sentences, and on the formal readout
        the set was made from. A share with its count beneath; "—" where the readout holds no count. The go-arounds are
        the go-arounds said, per sentence.
      </p>
      {readoutNote === null ? null : <p className="experiment-details-missing">The formal readout: {readoutNote}</p>}
      {table.notes.map((note) => <p key={note} className="training-details-lede">{note}</p>)}
      <table className="training-flown-table training-stats-table" aria-label="The models' statistics">
        <thead>
          <tr><th scope="col" rowSpan={2}>sentences</th><th scope="colgroup" colSpan={columns.length}>this set</th>
            <th scope="colgroup" colSpan={columns.length}>the formal readout</th></tr>
          <tr>{[...columns, ...columns].map((name, i) => <th key={i} scope="col">{name}</th>)}</tr>
        </thead>
        <tbody>
          {table.rows.map((row) => (
            <tr key={row.key}>
              <th scope="row">
                <span className="training-stats-swatch" style={{ background: TRAINING_SENTENCE_COLOR[row.kind] }} aria-hidden="true" />
                {row.label}
              </th>
              {[...cells(row.set, table.windows), ...cells(row.readout, table.windows)].map((cell, i) => <td key={i}>{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
