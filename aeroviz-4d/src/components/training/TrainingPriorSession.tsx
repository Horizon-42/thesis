/**
 * TrainingPriorSession.tsx
 * ------------------------
 * The Training panel over a prior set of stage B (`data/trainingPriorSample.ts`): its flights one at a time, and of the
 * flight on screen the sentences it has side by side — the closed-loop sentence of the reading (stage A's) and every
 * sentence the prior said in free generation — one on screen at a time (the sentence bar, the read-back window and the 3D
 * layers draw it, as for stage A: a sentence is published as a closed-loop sentence at the prior's Δ, `trainingPriorFlightView`).
 *
 * Under the table, for a sentence the prior said: the ROW INSPECTOR — at the row the cursor is on (a slider moves the cursor)
 * the probability the prior gave "go-around", whether the procedure permitted one and whether the flight was on final, and
 * the words the procedure masked BLOCKED in each column it rules (altitude, angle), written as what is still allowed. The
 * procedure's limits and the flight's other sentences are drawn in 3D (`useTrainingProcedureLayer`; the Draw switches here).
 *
 * A click on a word of the sentence on screen flies its segment live (`useTrainingPriorAutopilot`).
 */

import { useEffect, useState } from "react";
import { useApp, useTrainingCursor } from "../../context/AppContext";
import useTrainingPriorAutopilot from "../../hooks/useTrainingPriorAutopilot";
import { setTrainingPriorLayer, useTrainingPriorLayers } from "../../data/trainingPriorLayers";
import {
  fetchTrainingPriorSample,
  trainingPriorFlightView,
  trainingPriorOriginOf,
  trainingPriorSelectionOf,
  TRAINING_PRIOR_BLOCKED_COLUMNS,
  type TrainingPriorFlight,
  type TrainingPriorSample,
  type TrainingPriorSentence,
  type TrainingPriorSetEntry,
  type TrainingPriorWhich,
  type TrainingPriorBlockedColumn,
} from "../../data/trainingPriorSample";
import { readingRowAt, readingRowTimeS, trainingReadingOf, type TrainingVocabulary } from "../../data/trainingSample";
import { checkMark, TRAINING_OUTCOME_TAG, TRAINING_OUTCOME_TEXT, decisionText, crossingText } from "../../data/trainingText";
import {
  TRAINING_DECISION_FAIL_COLOR,
  TRAINING_DECISION_PASS_COLOR,
  trainingOutcomeColour,
} from "../../utils/trainingWordColors";
import ProblemBox from "./ProblemBox";

type SetState =
  | { status: "loading" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; sample: TrainingPriorSample };

/** The most allowed words a blocked column lists before it counts the rest. */
const ALLOWED_SHOWN = 10;

/** What a word of a masked column is, in words: an altitude word is a level above the airport elevation E (not MSL, as the 3D labels are) or "no level-off";
 *  an angle word is a class. */
export function blockedWordName(column: TrainingPriorBlockedColumn, value: number, vocabulary: TrainingVocabulary): string {
  if (column === "angle") return vocabulary.angleClasses[value].name;
  return value === vocabulary.noLevelOff ? "no level-off" : `${vocabulary.altitudeLevelsM[value]} m`;
}

/** The words of a masked column the procedure masks leave permitted at a row (the grammar may still forbid some), given the ones it blocked. */
export function allowedWords(column: TrainingPriorBlockedColumn, blocked: number[], vocabulary: TrainingVocabulary): number[] {
  const total = column === "angle" ? vocabulary.angleClasses.length : vocabulary.noLevelOff + 1;
  const out = new Set(blocked);
  return Array.from({ length: total }, (_, value) => value).filter((value) => !out.has(value));
}

/** How a sentence ended, as one line of the table: the outcome in its colour, the DA check, the go-arounds, its length. */
function SentenceLine({ outcome, crossing, goArounds, words, endS, label, active, onSelect, title }: {
  outcome: TrainingPriorSentence["outcome"]; crossing: TrainingPriorSentence["crossing"]; goArounds: number | null; words: number;
  endS: number; label: string; active: boolean; onSelect: () => void; title: string;
}) {
  const decision = crossing?.decision ?? null;
  return (
    <tr className={active ? "training-prior-sentence active" : "training-prior-sentence"}>
      <th scope="row">
        <button type="button" aria-pressed={active} title={title} onClick={onSelect}>{label}</button>
      </th>
      <td style={{ color: trainingOutcomeColour(outcome) }} title={TRAINING_OUTCOME_TEXT[outcome]}>
        {TRAINING_OUTCOME_TAG[outcome]} · {endS.toFixed(0)} s
      </td>
      <td title={decision === null ? "no decision-altitude check was made" : decisionText(decision)}>
        {decision === null ? "–" : (
          <span style={{ color: decision.passed ? TRAINING_DECISION_PASS_COLOR : TRAINING_DECISION_FAIL_COLOR }}>DA {checkMark(decision.passed)}</span>
        )}
      </td>
      <td>{goArounds === null ? "–" : goArounds}</td>
      <td>{words}</td>
    </tr>
  );
}

/** The flight's sentences side by side: the closed-loop one first, then each the prior said. */
function SentenceTable({ sample, flight, which, onSelect }: {
  sample: TrainingPriorSample; flight: TrainingPriorFlight; which: TrainingPriorWhich; onSelect: (which: TrainingPriorWhich) => void;
}) {
  const closed = flight.head.closedLoop[String(sample.model.rowIntervalS)];
  const endS = (startS: number, endCycle: number) => startS + endCycle * closed.cycleS - closed.startS;
  return (
    <table className="training-flown-table training-prior-table" aria-label="The flight's sentences">
      <thead>
        <tr><th scope="col">sentence</th><th scope="col">ended (from the first predicted step)</th><th scope="col">DA</th><th scope="col">go-arounds</th><th scope="col">words</th></tr>
      </thead>
      <tbody>
        <SentenceLine outcome={closed.replay.outcome} crossing={closed.replay.crossing} goArounds={null} words={closed.events.length}
          endS={endS(closed.startS, closed.replay.endCycle)} label="closed-loop reading" active={which === "closedLoop"}
          onSelect={() => onSelect("closedLoop")} title="the words the closed-loop reading says to the executor (stage A's sentence)" />
        {flight.sentences.map((sentence) => (
          <SentenceLine key={sentence.sample} outcome={sentence.outcome} crossing={sentence.crossing} goArounds={sentence.goArounds}
            words={sentence.events.length} endS={endS(closed.startS, sentence.endCycle)} label={`prior · sample ${sentence.sample}`}
            active={which === sentence.sample} onSelect={() => onSelect(sentence.sample)}
            title={`a sentence the prior said (temperature ${sample.model.temperature}), flown by the executor`} />
        ))}
      </tbody>
    </table>
  );
}

/** The inspector of one word row of a sentence the prior said. A leaf: it reads the Training cursor, which moves on every chart
 *  hover. */
function RowInspector({ sample, flight, sentence }: { sample: TrainingPriorSample; flight: TrainingPriorFlight; sentence: TrainingPriorSentence }) {
  const { trainingCursorS, setTrainingCursorS } = useTrainingCursor();
  const view = trainingPriorFlightView(sample, flight, sentence.sample);
  const reading = trainingReadingOf(view, sample.vocabulary.stepS, sample.model.rowIntervalS);
  const row = Math.max(readingRowAt(reading, trainingCursorS) ?? 0, 0);
  const probability = sentence.goAroundProbability;
  const wide = 200;
  const point = (r: number, p: number) => `${(r / Math.max(probability.length - 1, 1)) * wide},${34 - p * 32}`;
  return (
    <section className="training-prior-row" aria-label="The words at a row">
      <label className="training-field">
        <span>Row</span>
        <input type="range" min={0} max={sentence.words.length - 1} value={row} aria-label="Word row"
          onChange={(event) => setTrainingCursorS(readingRowTimeS(reading, Number(event.target.value)))} />
        <output>{row} · {(row * sample.model.rowIntervalS).toFixed(0)} s</output>
      </label>
      <svg className="training-prior-probability" viewBox={`0 0 ${wide} 36`} role="img" preserveAspectRatio="none"
        aria-label="Probability of go-around at each word row" onClick={(event) => {
          const box = event.currentTarget.getBoundingClientRect();
          const at = Math.round(((event.clientX - box.left) / box.width) * (probability.length - 1));
          setTrainingCursorS(readingRowTimeS(reading, Math.min(Math.max(at, 0), probability.length - 1)));
        }}>
        <polyline points={probability.map((p, r) => point(r, p)).join(" ")} fill="none" stroke="#fb923c" strokeWidth={1} vectorEffect="non-scaling-stroke" />
        <line x1={(row / Math.max(probability.length - 1, 1)) * wide} x2={(row / Math.max(probability.length - 1, 1)) * wide} y1={0} y2={36}
          stroke="#facc15" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      </svg>
      <dl className="training-flown-block">
        <div>
          <dt>Go-around</dt>
          <dd>
            {(probability[row] * 100).toFixed(1)} % said · {sentence.goAroundPermitted[row] ? "permitted" : "not permitted"} by the procedure
            · {sentence.onFinal[row] ? "on final" : "not on final"}
          </dd>
        </div>
        {TRAINING_PRIOR_BLOCKED_COLUMNS.map((column) => {
          const blocked = sentence.blocked[column][row];
          const allowed = allowedWords(column, blocked, sample.vocabulary);
          const listed = allowed.slice(0, ALLOWED_SHOWN).map((value) => blockedWordName(column, value, sample.vocabulary)).join(", ");
          return (
            <div key={column}>
              <dt>Blocked · {column}{column === "altitude" ? " (levels above airport elevation)" : ""}</dt>
              <dd title={`blocked: ${blocked.map((value) => blockedWordName(column, value, sample.vocabulary)).join(", ") || "none"}`}>
                {blocked.length === 0 ? "none blocked" : `${blocked.length} blocked`} · {allowed.length === 0 ? "none permitted" : `permitted by the procedure masks: ${listed}${allowed.length > ALLOWED_SHOWN ? ` … +${allowed.length - ALLOWED_SHOWN}` : ""}`}
              </dd>
            </div>
          );
        })}
        {sentence.crossing === null ? null : (
          <div><dt>Threshold</dt><dd>{crossingText(sentence.crossing)}</dd></div>
        )}
      </dl>
    </section>
  );
}

export default function TrainingPriorSession({ airport, sets }: { airport: string; sets: TrainingPriorSetEntry[] }) {
  const { setTrainingSelection, setTrainingIntervalS } = useApp();
  const layers = useTrainingPriorLayers();
  useTrainingPriorAutopilot();
  const [setId, setSetId] = useState<string | null>(sets[0]?.id ?? null);
  const [state, setState] = useState<SetState>({ status: "loading" });
  const [flightKey, setFlightKey] = useState<string | null>(null);
  const [which, setWhich] = useState<TrainingPriorWhich>(0);
  const entry = sets.find((item) => item.id === setId) ?? null;

  useEffect(() => {
    if (entry === null) return;
    let live = true;
    setState({ status: "loading" });
    fetchTrainingPriorSample(airport, entry.file, entry.id)
      .then((parsed) => {
        if (live) setState(parsed.ok ? { status: "ready", sample: parsed.value } : { status: "invalid", problem: parsed.problem });
      })
      .catch((error: unknown) => {
        if (live) setState({ status: "invalid", problem: error instanceof Error ? error.message : String(error) });
      });
    return () => {
      live = false;
    };
  }, [airport, entry]);

  const sample = state.status === "ready" ? state.sample : null;
  const flight = sample?.flights.find((item) => item.head.flightKey === flightKey) ?? null;
  // a set that has just opened may still hold the last set's choice for one render (the reset below runs after it): the
  // sentence shown is one the flight has, else its first
  const shown: TrainingPriorWhich = flight === null || which === "closedLoop" || flight.sentences.some((item) => item.sample === which)
    ? which : flight.sentences[0].sample;

  // a set opens on its first flight and that flight's first sentence, read at the prior's Δ
  useEffect(() => {
    if (sample === null) return;
    setFlightKey(sample.flights[0]?.head.flightKey ?? null);
    setWhich(sample.flights[0]?.sentences[0].sample ?? "closedLoop");
    setTrainingIntervalS(sample.model.rowIntervalS);
  }, [sample, setTrainingIntervalS]);

  // publish what the sentence bar, the read-back window and the 3D layers draw
  useEffect(() => {
    if (sample === null || flight === null) {
      setTrainingSelection(null);
      return;
    }
    setTrainingSelection(trainingPriorSelectionOf(sample, trainingPriorFlightView(sample, flight, shown)));
  }, [sample, flight, shown, setTrainingSelection]);
  useEffect(() => () => setTrainingSelection(null), [setTrainingSelection]);

  const origin = flight === null || sample === null ? undefined : trainingPriorOriginOf(trainingPriorFlightView(sample, flight, shown));
  const said = origin?.sentence ?? null;

  return (
    <>
      {sets.length > 1 ? (
        <label className="training-field" title={entry?.title}>
          <span>Set</span>
          <select value={setId ?? ""} onChange={(event) => setSetId(event.target.value)}>
            {sets.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.flights} flights · {item.sentences} sentences</option>)}
          </select>
        </label>
      ) : null}
      {entry === null ? null : (
        <p className="training-note" title={entry.title}>
          {entry.title}{entry.source.smoke ? " · SMOKE (not a result)" : ""}
        </p>
      )}
      {state.status === "loading" ? <p className="training-note" role="status">Loading {entry?.file} …</p> : null}
      {state.status === "invalid" ? <ProblemBox title={`Set ${setId} cannot be read.`} detail={state.problem} /> : null}

      {sample === null ? null : (
        <>
          <ul className="training-flight-list">
            {sample.flights.map((item) => (
              <li key={item.head.flightKey}>
                <button type="button" className={item.head.flightKey === flightKey ? "active" : undefined} title={item.head.flightKey}
                  onClick={() => { setFlightKey(item.head.flightKey); setWhich(item.sentences[0].sample); }}>
                  <span className="training-flight-callsign">{item.head.callsign}</span>
                  <span className="training-flight-runway">{item.head.runway}</span>
                  <span className="training-flight-stratum">{item.head.stratum}</span>
                  <span className="training-flight-executor"
                    title={item.sentences.map((s) => `sample ${s.sample}: ${TRAINING_OUTCOME_TAG[s.outcome]}`).join("; ")}>
                    {item.sentences.filter((s) => s.outcome === "landed").length}/{item.sentences.length} landed
                  </span>
                </button>
              </li>
            ))}
          </ul>
          {flight === null ? null : <SentenceTable sample={sample} flight={flight} which={shown} onSelect={setWhich} />}
          {flight === null || said === null ? (
            <p className="training-note">The closed-loop sentence is the reading's, with no probabilities or masks: pick a prior sentence for them.</p>
          ) : <RowInspector sample={sample} flight={flight} sentence={said} />}
          <fieldset className="training-layers">
            <legend>Draw (prior)</legend>
            <label title="each runway's region outline, glidepath lower edge, DA point and the entry point at the FAF">
              <input type="checkbox" checked={layers.procedure} onChange={(event) => setTrainingPriorLayer("procedure", event.target.checked)} />
              Procedure limits
            </label>
            <label title="the flown paths of the flight's other sentences, thin beside the one on screen">
              <input type="checkbox" checked={layers.otherSentences} onChange={(event) => setTrainingPriorLayer("otherSentences", event.target.checked)} />
              Other sentences
            </label>
          </fieldset>
        </>
      )}
    </>
  );
}
