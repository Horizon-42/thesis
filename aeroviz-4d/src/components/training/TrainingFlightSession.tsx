/**
 * TrainingFlightSession.tsx
 * -------------------------
 * The Training panel over a stage-A set: its flights one at a time. It publishes the selected flight through
 * `trainingSelection` — the sentence bar, the read-back window and the 3D layer draw it — and keeps the row interval Δ
 * the sentence is read at (`trainingIntervalS`: the set's first Δ when a set opens; the sentence bar's tabs choose).
 *
 * THE DOCK ENDS ABOVE THE SENTENCE BAR (the bar measures itself, `--training-bar-height`): the flight list takes the
 * height left over — never less than a few rows, the dock scrolling below that — so the switches under it are never
 * covered. Its readouts are one line each; their tables are on the details page (`TrainingDetails`), whose sections this
 * session supplies.
 */

import { useEffect, useRef, useState } from "react";
import { useApp } from "../../context/AppContext";
import TrainingVocabularyNotes from "../TrainingVocabularyNotes";
import TrainingDetails, { type TrainingDetailsSection } from "./TrainingDetails";
import { NotesList } from "./NotesToggle";
import { DetailsLink, LAYER_SWITCHES, LayerSwitches, type DetailsPage } from "./PanelParts";
import {
  TRAINING_CORRECTION_COLOR,
  TRAINING_DECISION_FAIL_COLOR,
  TRAINING_DECISION_PASS_COLOR,
  trainingOutcomeColour,
} from "../../utils/trainingWordColors";
import {
  closedCycleTimeS,
  correctionCount,
  trainingSelectionOf,
  type TrainingClosedLoop,
  type TrainingFlight,
  type TrainingSample,
  type TrainingSetEntry,
} from "../../data/trainingSample";
import { checkMark, crossingText, decisionText, TRAINING_OUTCOME_TAG, TRAINING_OUTCOME_TEXT } from "../../data/trainingText";

const OVERVIEW_SECTION = "overview";
const FLOWN_SECTION = "flown";

/** How a flight's closed-loop sentence at Δ ended, in the flight list's tag: the judge's outcome in its colour, the DA
 *  check's mark, and how many words the reading added. */
function FlownTag({ closed }: { closed: TrainingClosedLoop }) {
  const { replay } = closed;
  const decision = replay.crossing?.decision ?? null;
  const added = correctionCount(closed.events);
  return (
    <span className="training-flight-executor" style={{ color: trainingOutcomeColour(replay.outcome) }}
      title={`${TRAINING_OUTCOME_TEXT[replay.outcome]}${decision === null ? "" : ` — ${decisionText(decision)}`}; ` +
        `${added} of the ${closed.events.length} words were added by the closed-loop reading`}>
      {TRAINING_OUTCOME_TAG[replay.outcome]}
      {decision === null ? null : (
        <span style={{ color: decision.passed ? TRAINING_DECISION_PASS_COLOR : TRAINING_DECISION_FAIL_COLOR }}> DA{checkMark(decision.passed)}</span>
      )}
      <span style={{ color: TRAINING_CORRECTION_COLOR }}> {added}c</span>
    </span>
  );
}

/** The flown flight of the selected flight at Δ: how it ended, where it crossed the threshold and the DA check's values —
 *  a few lines in the dock; the same values are in the bar's chips and the read-back window. */
function FlownBlock({ closed }: { closed: TrainingClosedLoop }) {
  const { replay } = closed;
  const crossing = replay.crossing;
  const decision = crossing?.decision ?? null;
  return (
    <dl className="training-flown-block" aria-label={`The flown flight at Δ ${closed.rowIntervalS} s`}>
      <div>
        <dt>Flown flight · Δ {closed.rowIntervalS} s</dt>
        <dd style={{ color: trainingOutcomeColour(replay.outcome) }} title={TRAINING_OUTCOME_TEXT[replay.outcome]}>
          {TRAINING_OUTCOME_TAG[replay.outcome]} at {closedCycleTimeS(closed, replay.endCycle)} s
        </dd>
      </div>
      <div>
        <dt>Threshold</dt>
        <dd>{crossing === null ? "not crossed" : crossingText(crossing)}</dd>
      </div>
      <div>
        <dt>DA check</dt>
        {decision === null ? <dd>none made</dd> : (
          <dd title={decisionText(decision)}>
            <strong style={{ color: decision.passed ? TRAINING_DECISION_PASS_COLOR : TRAINING_DECISION_FAIL_COLOR }}>
              {decision.passed ? "passed" : "failed"}
            </strong>
            {" "}· lateral {Math.abs(decision.rightM).toFixed(1)} m {decision.rightM >= 0 ? "right" : "left"} (cone ±
            {decision.coneHalfWidthM.toFixed(1)} m) {checkMark(decision.lateralOk)} · vertical {Math.abs(decision.aboveGlidepathM).toFixed(1)} m{" "}
            {decision.aboveGlidepathM >= 0 ? "above" : "below"} the glidepath {checkMark(decision.verticalOk)} · {decision.heightMslM.toFixed(0)} m MSL
          </dd>
        )}
      </div>
    </dl>
  );
}

/** The flown flights of the set at each Δ, counted: landed, DA passed, words added. */
function flownSummary(sample: TrainingSample): string {
  return sample.vocabulary.rowIntervalsS.map((interval) => {
    const closed = sample.flights.map((flight) => flight.closedLoop[String(interval)]);
    const landed = closed.filter((item) => item.replay.outcome === "landed").length;
    return `Δ ${interval} s: ${landed}/${closed.length} landed`;
  }).join(" · ");
}

/** The flown flights table: every flight at every Δ — outcome, DA check, words added. */
function FlownTable({ sample }: { sample: TrainingSample }) {
  const intervals = sample.vocabulary.rowIntervalsS;
  return (
    <table className="training-flown-table">
      <thead>
        <tr>
          <th scope="col">flight</th>
          {intervals.map((interval) => <th key={interval} scope="col">Δ {interval} s</th>)}
        </tr>
      </thead>
      <tbody>
        {sample.flights.map((flight) => (
          <tr key={flight.flightKey}>
            <th scope="row" title={flight.flightKey}>{flight.callsign} · {flight.runway} · {flight.stratum}</th>
            {intervals.map((interval) => {
              const closed = flight.closedLoop[String(interval)];
              const decision = closed.replay.crossing?.decision ?? null;
              return (
                <td key={interval} style={{ color: trainingOutcomeColour(closed.replay.outcome) }}
                  title={decision === null ? undefined : decisionText(decision)}>
                  {TRAINING_OUTCOME_TAG[closed.replay.outcome]}
                  {decision === null ? "" : ` · DA ${checkMark(decision.passed)}`} · {correctionCount(closed.events)} added
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** ``sample``: the set open — null while the next one loads (the session is kept: its switches and choices per set
 *  outlive a switch of sets, and it publishes no flight meanwhile). */
export default function TrainingFlightSession({ airport, sample, entry, details }: {
  airport: string; sample: TrainingSample | null; entry: TrainingSetEntry | null; details: DetailsPage;
}) {
  const { setTrainingSelection, trainingIntervalS, setTrainingIntervalS } = useApp();
  const [flightKey, setFlightKey] = useState<string | null>(null);

  // Keep the selection on the same flight across a reload when it is still there; otherwise the first, so the
  // sentence bar is never blank beside a list.
  useEffect(() => {
    if (sample === null) {
      setFlightKey(null);
      return;
    }
    setFlightKey((previous) =>
      previous && sample.flights.some((flight) => flight.flightKey === previous) ? previous : (sample.flights[0]?.flightKey ?? null));
  }, [sample]);

  // The Δ read: the set's first when the first set opens; after that kept across sets (the labelled sentence too) when the
  // new set has it.
  const intervalChosen = useRef(false);
  useEffect(() => {
    if (sample === null) return;
    const intervals = sample.vocabulary.rowIntervalsS;
    const keep = intervalChosen.current && (trainingIntervalS === null || intervals.includes(trainingIntervalS));
    intervalChosen.current = true;
    if (!keep) setTrainingIntervalS(intervals[0]);
    // only when a set opens: the sentence bar's tabs own the choice after that
  }, [sample, setTrainingIntervalS]);

  // ── publish what the other views draw ─────────────────────────────────────
  useEffect(() => {
    const flight = sample?.flights.find((item) => item.flightKey === flightKey) ?? null;
    setTrainingSelection(flight && sample ? trainingSelectionOf(sample, flight) : null);
  }, [sample, flightKey, setTrainingSelection]);
  useEffect(() => () => setTrainingSelection(null), [setTrainingSelection]);
  if (sample === null) return null;

  const selected: TrainingFlight | undefined = sample.flights.find((flight) => flight.flightKey === flightKey);
  const selectedClosed = selected === undefined || trainingIntervalS === null ? null : selected.closedLoop[String(trainingIntervalS)];

  // ── the details page ──────────────────────────────────────────────────────
  const sections: TrainingDetailsSection[] = [
    { id: OVERVIEW_SECTION, title: "What this view shows", body: (
      <>
        <p className="training-details-lede">
          Each arrival read as the instructions a controller could have given — five columns: runway (or go-around), heading
          relative to the course of the runway in force, a level above the airport elevation, an angle, a speed. The labelled
          sentence is the labeller's reading of the observed flight; the closed-loop sentence at each row interval Δ is what the
          closed-loop reading says to the executor from the first predicted step, the words it added marked, and the path the
          executor flew from them beside the observed track, with the judge's outcome and the decision-altitude (DA) check.
        </p>
        <h4 className="training-details-subhead">What each switch draws</h4>
        <NotesList items={LAYER_SWITCHES.map(({ layer, colour, text, title }) => ({ key: layer, text: title, name: (
          <><span className="training-model-swatch" style={{ background: colour }} />{text}</>) }))} />
        <h4 className="training-details-subhead">The set</h4>
        <NotesList items={[
          { key: "id", name: <code>{sample.setId}</code>, text: entry?.title ?? sample.setId },
          { key: "cohort", name: "Flights", text: `${sample.flights.length} (${Object.entries(sample.cohort.splits).map(([split, count]) =>
            `${split} ${count}`).join(", ")}; ${sample.cohort.perStratum} per stratum, seed ${sample.cohort.seed}) — ${sample.cohort.drawnFrom}` },
          { key: "source", name: "Made from", text: `${sample.source.instructions} and ${sample.source.executor}, commit ` +
            `${sample.source.git.head.slice(0, 10)}${sample.source.git.dirty ? " (dirty tree)" : ""}` },
        ]} />
      </>
    ) },
    { id: "vocabulary", title: "Vocabulary", body: <TrainingVocabularyNotes sample={sample} /> },
    { id: FLOWN_SECTION, title: "Flown flights", body: (
      <>
        <p className="training-details-lede">
          Every flight of the set at every row interval: how its flown path ended (the judge's outcome), whether the DA check of
          its threshold crossing passed, and how many words the closed-loop reading added.
        </p>
        <FlownTable sample={sample} />
      </>
    ) },
  ];

  return (
    <>
      {details.shown !== null ? (
        <TrainingDetails context={[airport, sample.setId, `${sample.flights.length} flights`].join(" · ")}
          sections={sections} sectionId={details.shown.section} onSection={details.show} onClose={details.close}
          opener={details.shown.opener} />
      ) : null}

      <p className="training-note" title={`${entry?.title ?? sample.setId} — ${sample.cohort.drawnFrom}`}>
        {sample.flights.length} flights · {Object.entries(sample.cohort.splits).map(([split, count]) => `${split} ${count}`).join(", ")},{" "}
        {sample.cohort.perStratum} per stratum
        {entry && entry.flights !== sample.flights.length
          ? ` · the index says ${entry.flights}: this set's two files are from different exports` : ""}
      </p>
      <ul className="training-flight-list">
        {sample.flights.map((flight) => {
          const closed = trainingIntervalS === null ? null : flight.closedLoop[String(trainingIntervalS)];
          return (
            <li key={flight.flightKey}>
              <button type="button" className={flight.flightKey === flightKey ? "active" : undefined}
                title={flight.flightKey} onClick={() => setFlightKey(flight.flightKey)}>
                <span className="training-flight-callsign">{flight.callsign}</span>
                <span className="training-flight-runway">{flight.runway}</span>
                <span className="training-flight-stratum">{flight.stratum}</span>
                {closed ? <FlownTag closed={closed} />
                  : <span className="training-flight-events">{flight.openLoop.events.length} words</span>}
              </button>
            </li>
          );
        })}
      </ul>

      {selectedClosed !== null ? <FlownBlock closed={selectedClosed} /> : null}

      <fieldset className="training-layers">
        <legend>Draw</legend>
        <LayerSwitches />
      </fieldset>

      <ul className="training-details-links" aria-label="Readouts">
        <DetailsLink name="Flown flights" summary={flownSummary(sample)} onOpen={details.open(FLOWN_SECTION)} />
      </ul>
    </>
  );
}
