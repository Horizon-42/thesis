/**
 * TrainingPriorSession.tsx
 * ------------------------
 * The Training panel over a prior set of stage B (`data/trainingPriorSample.ts`), in the layout every stage shares (outline
 * §6.2): the set chooser with the set's intent line, the list of flights, one line per readout and the Draw box. Of the
 * flight on screen the SENTENCE BAR'S TABS choose the sentence — Labelled (the labeller's open-loop reading of the observed
 * flight), Closed loop (stage A's closed-loop sentence at the prior's Δ) and each sentence the prior said in free
 * generation, Sample 0, 1 … (`data/trainingTabs.ts`); the bar, the read-back window and the 3D layers draw it as stage A
 * draws a sentence (`trainingPriorFlightView`). The left panel has no second chooser.
 *
 * Its readouts, one line each, open the DETAILS PAGE (modal): "This sentence" and "Sentences" → Sentences; "At the cursor"
 * (the probability the prior gave "go-around", whether the procedure permitted one, on final or not, the words the masks
 * blocked) → Row inspector. Under "At the cursor" stays the probability strip of the sentence on screen (the page is
 * modal: what follows the cursor stays in the panel). The procedure's limits and the flight's other sentences are drawn in
 * 3D (`useTrainingProcedureLayer`; the Draw switches here).
 *
 * A click on a word of the sentence on screen flies its segment live (`useTrainingPriorAutopilot`).
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useApp, useTrainingCursor } from "../../context/AppContext";
import useTrainingPriorAutopilot from "../../hooks/useTrainingPriorAutopilot";
import useTrainingSet from "../../hooks/useTrainingSet";
import { setTrainingPriorLayer, useTrainingPriorLayers } from "../../data/trainingPriorLayers";
import {
  fetchTrainingPriorSample,
  trainingPriorFlightView,
  trainingPriorSelectionOf,
  TRAINING_PRIOR_BLOCKED_COLUMNS,
  type TrainingPriorFlight,
  type TrainingPriorSample,
  type TrainingPriorSentence,
  type TrainingPriorSetEntry,
  type TrainingPriorWhich,
  type TrainingPriorBlockedColumn,
} from "../../data/trainingPriorSample";
import { readingRowAt, readingRowTimeS, trainingReadingOf, type TrainingFlownEnd, type TrainingVocabulary } from "../../data/trainingSample";
import { checkMark, crossingText, TRAINING_OUTCOME_TAG, TRAINING_OUTCOME_TEXT } from "../../data/trainingText";
import { publishTrainingTabs, useTrainingTabs, type TrainingTab } from "../../data/trainingTabs";
import { useTrainingSetIntent } from "../../data/trainingSetIntent";
import { trainingOutcomeColour } from "../../utils/trainingWordColors";
import ProblemBox from "./ProblemBox";
import TrainingDetails, { type TrainingDetailsSection } from "./TrainingDetails";
import { DetailsLink, DrawBox, EXPERIMENT_SECTION, OVERVIEW_SECTION, type DetailsPage } from "./PanelParts";
import { ExperimentSection, ItemList, SetChooser } from "./SetParts";
import { NotesList } from "./NotesToggle";

const SENTENCES_SECTION = "sentences";
const INSPECTOR_SECTION = "row-inspector";
/** The tabs of a flight: its labelled sentence, its closed-loop sentence and each sample the prior said. */
const LABELLED = "labelled";
const CLOSED_LOOP = "closed-loop";
const sampleTab = (sample: number) => `sample-${sample}`;

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

/** The seconds from the first predicted step to a flown sentence's end (the executor's cycles). */
function endS(flight: TrainingPriorFlight, intervalS: number, endCycle: number): number {
  return endCycle * flight.head.closedLoop[String(intervalS)].cycleS;
}

/** A flown sentence's ending as the tabs, lines and tables say it: the outcome, its time, the DA check, the go-arounds. */
function ending(outcome: TrainingFlownEnd, seconds: number, decision: { passed: boolean } | null, goArounds: number | null): string {
  return `${TRAINING_OUTCOME_TAG[outcome]} at ${seconds.toFixed(0)} s` +
    (decision === null ? "" : ` · DA ${checkMark(decision.passed)}`) + (goArounds === null ? "" : ` · ${goArounds} go-around${goArounds === 1 ? "" : "s"}`);
}

/** The flight's tabs (outline §6.2 item 2). */
function flightTabs(sample: TrainingPriorSample, flight: TrainingPriorFlight): TrainingTab[] {
  const intervalS = sample.model.rowIntervalS;
  const closed = flight.head.closedLoop[String(sample.model.rowIntervalS)];
  return [
    { id: LABELLED, label: "Labelled", outcome: null,
      title: "The labeller's reading of the observed flight (open loop): its words on the 2 s rows; not flown" },
    { id: CLOSED_LOOP, label: "Closed loop", outcome: closed.replay.outcome,
      title: `Stage A's closed-loop sentence of the flight at Δ = ${intervalS} s, flown by the executor — ` +
        ending(closed.replay.outcome, endS(flight, sample.model.rowIntervalS, closed.replay.endCycle), closed.replay.crossing?.decision ?? null, null) },
    ...flight.sentences.map((sentence) => ({
      id: sampleTab(sentence.sample), label: `Sample ${sentence.sample}`, outcome: sentence.outcome,
      title: `A sentence the prior said (sample ${sentence.sample}, temperature ${sample.model.temperature}), flown by the executor — ` +
        ending(sentence.outcome, endS(flight, sample.model.rowIntervalS, sentence.endCycle), sentence.crossing?.decision ?? null, sentence.goArounds),
    })),
  ];
}

/** The sentence a tab stands for. */
function whichOf(tab: string): TrainingPriorWhich {
  return tab === LABELLED || tab === CLOSED_LOOP ? "closedLoop" : Number(tab.slice("sample-".length));
}

/** The inspector of one word row of a sentence the prior said (the details page's Row inspector). A leaf: it reads the
 *  Training cursor, which moves on every chart hover. */
function RowInspector({ sample, flight, sentence }: { sample: TrainingPriorSample; flight: TrainingPriorFlight; sentence: TrainingPriorSentence }) {
  const { trainingCursorS, setTrainingCursorS } = useTrainingCursor();
  const view = trainingPriorFlightView(sample, flight, sentence.sample);
  const reading = trainingReadingOf(view, sample.vocabulary.stepS, sample.model.rowIntervalS);
  const row = Math.max(readingRowAt(reading, trainingCursorS) ?? 0, 0);
  return (
    <section className="training-prior-row" aria-label="The words at a row">
      <label className="training-field">
        <span>Row</span>
        <input type="range" min={0} max={sentence.words.length - 1} value={row} aria-label="Word row"
          onChange={(event) => setTrainingCursorS(readingRowTimeS(reading, Number(event.target.value)))} />
        <output>{row} · {(row * sample.model.rowIntervalS).toFixed(0)} s</output>
      </label>
      <dl className="training-flown-block">
        <div>
          <dt>Go-around</dt>
          <dd>
            {(sentence.goAroundProbability[row] * 100).toFixed(1)} % said · {sentence.goAroundPermitted[row] ? "permitted" : "not permitted"} by the procedure
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
                {blocked.length === 0 ? "none blocked" : `${blocked.length} blocked: ${blocked.map((value) => blockedWordName(column, value, sample.vocabulary)).join(", ")}`}
                {" · "}{allowed.length === 0 ? "none permitted" : `permitted by the procedure masks: ${listed}${allowed.length > ALLOWED_SHOWN ? ` … +${allowed.length - ALLOWED_SHOWN}` : ""}`}
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

/** "At the cursor": the readout's one line and, under it, the probability strip of the sentence on screen (one line high; a
 *  click moves the cursor). A leaf: it reads the Training cursor. */
function AtTheCursor({ sample, flight, sentence, onOpen }: {
  sample: TrainingPriorSample; flight: TrainingPriorFlight; sentence: TrainingPriorSentence | null;
  onOpen: ReturnType<DetailsPage["open"]>;
}) {
  const { trainingCursorS, setTrainingCursorS } = useTrainingCursor();
  if (sentence === null) {
    return <DetailsLink name="At the cursor" summary="a sample's records: choose a Sample tab" onOpen={onOpen} />;
  }
  const view = trainingPriorFlightView(sample, flight, sentence.sample);
  const reading = trainingReadingOf(view, sample.vocabulary.stepS, sample.model.rowIntervalS);
  const row = Math.max(readingRowAt(reading, trainingCursorS) ?? 0, 0);
  const probability = sentence.goAroundProbability;
  const blocked = TRAINING_PRIOR_BLOCKED_COLUMNS.map((column) => `${column} ${sentence.blocked[column][row].length}`).join(", ");
  const summary = `row ${row}: go-around ${(probability[row] * 100).toFixed(1)} %, ${sentence.goAroundPermitted[row] ? "permitted" : "not permitted"}, ` +
    `${sentence.onFinal[row] ? "on final" : "not on final"} · blocked ${blocked}`;
  const wide = 200;
  const x = (r: number) => (r / Math.max(probability.length - 1, 1)) * wide;
  return (
    <>
      <DetailsLink name="At the cursor" summary={summary} onOpen={onOpen} />
      <li className="training-prior-strip-item">
        <svg className="training-prior-probability" viewBox={`0 0 ${wide} 12`} role="img" preserveAspectRatio="none"
          aria-label="Probability of go-around at each word row (a click moves the cursor)" onClick={(event) => {
            const box = event.currentTarget.getBoundingClientRect();
            const at = Math.round(((event.clientX - box.left) / box.width) * (probability.length - 1));
            setTrainingCursorS(readingRowTimeS(reading, Math.min(Math.max(at, 0), probability.length - 1)));
          }}>
          <polyline points={probability.map((p, r) => `${x(r)},${11 - p * 10}`).join(" ")} fill="none" stroke="#fb923c" strokeWidth={1}
            vectorEffect="non-scaling-stroke" />
          <line x1={x(row)} x2={x(row)} y1={0} y2={12} stroke="#facc15" strokeWidth={1} vectorEffect="non-scaling-stroke" />
        </svg>
      </li>
    </>
  );
}

/** Every flight's sentences (the details page's Sentences): the closed-loop one and each sample, and the landed of each sample
 *  by stratum. */
function SentencesTable({ sample }: { sample: TrainingPriorSample }) {
  const intervalS = sample.model.rowIntervalS;
  const samples = [...new Set(sample.flights.flatMap((flight) => flight.sentences.map((s) => s.sample)))].sort((a, b) => a - b);
  const strata = [...new Set(sample.flights.map((flight) => flight.head.stratum))].sort();
  const cell = (outcome: TrainingFlownEnd, text: string) => (
    <td style={{ color: trainingOutcomeColour(outcome) }} title={TRAINING_OUTCOME_TEXT[outcome]}>{text}</td>
  );
  return (
    <>
      <table className="training-flown-table training-prior-table" aria-label="Every flight's sentences">
        <thead>
          <tr><th scope="col">flight</th><th scope="col">closed loop</th>{samples.map((s) => <th key={s} scope="col">sample {s}</th>)}</tr>
        </thead>
        <tbody>
          {sample.flights.map((flight) => {
            const closed = flight.head.closedLoop[String(intervalS)];
            return (
              <tr key={flight.head.flightKey}>
                <th scope="row" title={flight.head.flightKey}>{flight.head.callsign} · {flight.head.stratum}</th>
                {cell(closed.replay.outcome, ending(closed.replay.outcome, endS(flight, intervalS, closed.replay.endCycle), closed.replay.crossing?.decision ?? null, null))}
                {samples.map((s) => {
                  const said = flight.sentences.find((item) => item.sample === s);
                  return said === undefined ? <td key={s}>–</td> : (
                    <td key={s} style={{ color: trainingOutcomeColour(said.outcome) }} title={TRAINING_OUTCOME_TEXT[said.outcome]}>
                      {ending(said.outcome, endS(flight, intervalS, said.endCycle), said.crossing?.decision ?? null, said.goArounds)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <table className="training-flown-table" aria-label="Landed by sample and by stratum">
        <thead><tr><th scope="col">landed</th>{strata.map((s) => <th key={s} scope="col">{s}</th>)}<th scope="col">all</th></tr></thead>
        <tbody>
          {samples.map((s) => {
            const count = (flights: TrainingPriorFlight[]) => {
              const said = flights.flatMap((flight) => flight.sentences.filter((item) => item.sample === s));
              return `${said.filter((item) => item.outcome === "landed").length}/${said.length}`;
            };
            return (
              <tr key={s}>
                <th scope="row">sample {s}</th>
                {strata.map((stratum) => <td key={stratum}>{count(sample.flights.filter((flight) => flight.head.stratum === stratum))}</td>)}
                <td>{count(sample.flights)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}

export default function TrainingPriorSession({ airport, sets, details }: {
  airport: string; sets: TrainingPriorSetEntry[]; details: DetailsPage;
}) {
  const { setTrainingSelection, setTrainingIntervalS } = useApp();
  const layers = useTrainingPriorLayers();
  useTrainingPriorAutopilot();
  const [setId, setSetId] = useState<string | null>(sets[0]?.id ?? null);
  const [flightKey, setFlightKey] = useState<string | null>(null);
  const entry = sets.find((item) => item.id === setId) ?? null;
  const state = useTrainingSet(entry === null ? null : `${airport}/${entry.id}/${entry.file}`,
    () => fetchTrainingPriorSample(airport, entry!.file, entry!.id));
  const intent = useTrainingSetIntent(setId);

  const sample = state.status === "ready" ? state.sample : null;
  // the flight on screen: the one chosen if this set has it, else the set's first
  const flight = sample === null ? null : sample.flights.find((item) => item.head.flightKey === flightKey) ?? sample.flights[0] ?? null;

  // ── the tabs: the session gives them, the bar chooses (outline §6.2 item 2) ──
  const scope = sample === null || flight === null ? null : `B/${airport}/${sample.setId}/${flight.head.flightKey}`;
  const tabList = useMemo(() => (sample === null || flight === null ? [] : flightTabs(sample, flight)), [sample, flight]);
  const shared = useTrainingTabs();
  const last = useRef<string>(sampleTab(0));
  // the bar's choice in this scope; on a new flight or set, the last tab chosen where it has one, else its first sample
  const firstSample = tabList.find((tab) => tab.id.startsWith("sample-"))?.id ?? CLOSED_LOOP;
  const chosen = shared !== null && shared.scope === scope && tabList.some((tab) => tab.id === shared.chosen) ? shared.chosen
    : tabList.some((tab) => tab.id === last.current) ? last.current : firstSample;
  // only a choice made among the tabs of an item on screen is remembered (not the fallback while a set loads)
  useEffect(() => {
    if (scope !== null) last.current = chosen;
  }, [scope, chosen]);
  useEffect(() => {
    if (scope !== null) publishTrainingTabs({ scope, tabs: tabList, chosen });
  }, [scope, tabList, chosen]);
  useEffect(() => () => publishTrainingTabs(null), []);

  // the sentence on screen, read at the prior's Δ (Labelled: the open-loop reading)
  const which = whichOf(chosen);
  useEffect(() => {
    if (sample !== null) setTrainingIntervalS(chosen === LABELLED ? null : sample.model.rowIntervalS);
  }, [sample, chosen, setTrainingIntervalS]);
  useEffect(() => {
    if (sample === null || flight === null) {
      setTrainingSelection(null);
      return;
    }
    setTrainingSelection(trainingPriorSelectionOf(sample, trainingPriorFlightView(sample, flight, which)));
  }, [sample, flight, which, setTrainingSelection]);
  useEffect(() => () => setTrainingSelection(null), [setTrainingSelection]);

  const said = flight === null || which === "closedLoop" ? null : flight.sentences.find((item) => item.sample === which) ?? null;


  // ── the details page ──────────────────────────────────────────────────────
  const absent = state.status === "invalid" ? `the set cannot be read: ${state.problem}` : "the set is loading";
  const sections: TrainingDetailsSection[] = [
    // the first section always has a body: the set's intent and facts come from its index entry, not its sample
    entry === null ? { id: EXPERIMENT_SECTION, title: "The set and the experiment", body: <p className="training-details-lede">The index lists no set.</p> } : {
      id: EXPERIMENT_SECTION, title: "The set and the experiment", body: (
        <ExperimentSection setId={entry.id} intent={intent} facts={[
          { key: "made", name: "Made from", text: `the free-generation readout ${entry.source.readout} of the prior ${entry.model.prior} ` +
            `(selection ${entry.model.selection}, Δ ${entry.model.rowIntervalS} s, temperature ${entry.model.temperature}), on ` +
            `${entry.source.instructions} and ${entry.source.executor}, commit ${entry.source.git.head.slice(0, 10)}` +
            `${entry.source.git.dirty ? " (dirty tree)" : ""}${entry.source.smoke ? " — a SMOKE readout: not a result" : ""}` },
          { key: "items", name: "Flights", text: `${entry.flights} of the ${entry.cohort.split} days, ${entry.cohort.perAirport} ` +
            `drawn at this airport (seed ${entry.cohort.seed}) from the readout's ${entry.cohort.readoutFlights}, ${entry.cohort.samples} ` +
            `sentences said for each — ${entry.cohort.drawnFrom}` },
          ...(entry.source.validationClaim === null ? [] : [{ key: "claim", name: "Validation readout",
            text: `the base's one validation readout ${entry.source.validationClaim.readout}, claimed by ` +
              `${entry.source.validationClaim.reader} of ${entry.source.validationClaim.prior} (its flights are val)` }]),
        ]} />
      ) },
    { id: OVERVIEW_SECTION, title: "What this view shows", body: (
      <>
        <p className="training-details-lede">
          Each flight of a free-generation readout of the prior: the observed flight, stage A's closed-loop sentence of it at the
          prior's Δ and every sentence the prior said itself, each flown by the executor and judged.
        </p>
        <h4 className="training-details-subhead">The tabs of the sentence bar</h4>
        <NotesList items={[
          { key: "labelled", name: "Labelled", text: "the labeller's open-loop reading of the observed flight (not flown)" },
          { key: "closed", name: "Closed loop", text: "stage A's closed-loop sentence of the flight at the prior's Δ, flown by the executor" },
          { key: "sample", name: "Sample 0, 1 …", text: "a sentence the prior said in free generation, flown by the executor; the dot is its outcome's colour" },
        ]} />
        <h4 className="training-details-subhead">What each switch draws</h4>
        <NotesList items={[
          { key: "procedure", name: "Procedure limits", text: "each runway's region outline, glidepath lower edge, DA point and the entry point at the FAF" },
          { key: "other", name: "Other sentences", text: "the flown paths of the flight's other sentences, thin beside the one on screen" },
        ]} />
      </>
    ) },
    sample === null ? { id: SENTENCES_SECTION, title: "Sentences", body: null, absent }
      : { id: SENTENCES_SECTION, title: "Sentences", body: <SentencesTable sample={sample} /> },
    sample === null || flight === null || said === null
      ? { id: INSPECTOR_SECTION, title: "Row inspector", body: null, absent: sample === null ? absent : "a prior sample's records: choose a Sample tab" }
      : { id: INSPECTOR_SECTION, title: "Row inspector", body: <RowInspector sample={sample} flight={flight} sentence={said} /> },
  ];

  const choices = sets.map((item) => ({ id: item.id, count: `${item.flights} flights · ${item.sentences} sentences`, title: item.title,
    smoke: item.source.smoke }));
  return (
    <>
      {details.shown !== null ? (
        <TrainingDetails context={[airport, setId ?? "no set", sample === null ? "" : `${sample.flights.length} flights`].filter(Boolean).join(" · ")}
          sections={sections} sectionId={details.shown.section} onSection={details.show} onClose={details.close}
          opener={details.shown.opener} />
      ) : null}
      <SetChooser sets={choices} setId={setId} onChange={(id) => setSetId(id)} intent={intent} onOpenExperiment={details.open(EXPERIMENT_SECTION)} />
      {state.status === "loading" ? <p className="training-note" role="status">Loading {entry?.file} …</p> : null}
      {state.status === "invalid" ? <ProblemBox title={`Set ${setId} cannot be read.`} detail={state.problem} /> : null}

      {sample === null ? null : (
        <>
          <ItemList label="The set's flights" active={flight?.head.flightKey ?? null} onSelect={setFlightKey}
            items={sample.flights.map((item) => ({
              key: item.head.flightKey, callsign: item.head.callsign, runway: item.head.runway, stratum: item.head.stratum,
              landed: item.sentences.filter((s) => s.outcome === "landed").length, of: item.sentences.length, title: item.head.flightKey,
              outcomes: item.sentences.map((s) => `sample ${s.sample}: ${TRAINING_OUTCOME_TAG[s.outcome]}`).join("; "),
            }))} />
          {flight === null ? null : (
            <ul className="training-details-links" aria-label="Readouts">
              <DetailsLink name="This sentence" onOpen={details.open(SENTENCES_SECTION)} summary={
                chosen === LABELLED ? "the labelled sentence (not flown)"
                  : said === null ? `closed loop: ${ending(flight.head.closedLoop[String(sample.model.rowIntervalS)].replay.outcome,
                    endS(flight, sample.model.rowIntervalS, flight.head.closedLoop[String(sample.model.rowIntervalS)].replay.endCycle),
                    flight.head.closedLoop[String(sample.model.rowIntervalS)].replay.crossing?.decision ?? null, null)}`
                    : `sample ${said.sample}: ${ending(said.outcome, endS(flight, sample.model.rowIntervalS, said.endCycle), said.crossing?.decision ?? null, said.goArounds)}`} />
              <DetailsLink name="Sentences" onOpen={details.open(SENTENCES_SECTION)} summary={
                [...new Set(sample.flights.flatMap((item) => item.sentences.map((s) => s.sample)))].sort((a, b) => a - b).map((s) => {
                  const all = sample.flights.flatMap((item) => item.sentences.filter((x) => x.sample === s));
                  return `sample ${s}: ${all.filter((x) => x.outcome === "landed").length}/${all.length} landed`;
                }).join(" · ")} />
              <AtTheCursor sample={sample} flight={flight} sentence={said} onOpen={details.open(INSPECTOR_SECTION)} />
            </ul>
          )}
          <DrawBox legend="Draw (prior)">
            <label title="each runway's region outline, glidepath lower edge, DA point and the entry point at the FAF">
              <input type="checkbox" checked={layers.procedure} onChange={(event) => setTrainingPriorLayer("procedure", event.target.checked)} />
              Procedure limits
            </label>
            <label title="the flown paths of the flight's other sentences, thin beside the one on screen">
              <input type="checkbox" checked={layers.otherSentences} onChange={(event) => setTrainingPriorLayer("otherSentences", event.target.checked)} />
              Other sentences
            </label>
          </DrawBox>
        </>
      )}
    </>
  );
}
