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
 * Its readouts, one line each: "This sentence" and "Sentences" open the DETAILS PAGE (modal) on Free generation; "At the
 * cursor" (the probability the prior gave "go-around", whether the procedure permitted one, on final or not, the words the
 * masks blocked) opens nothing — it follows the cursor, so it stays in the panel. Under the readouts, the cursor slider
 * (frontend §6.1) carries, on a sample's tab, the probability of "go-around" along the sentence. The procedure's limits and
 * the flight's other sentences are drawn in 3D (`useTrainingProcedureLayer`; the Draw switches here).
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
} from "../../data/trainingPriorSample";
import { readingRowAt, trainingReadingOf, type TrainingFlownEnd } from "../../data/trainingSample";
import { firstStepMark, goAroundLine } from "../../data/trainingSlider";
import { checkMark, TRAINING_OUTCOME_TAG, TRAINING_OUTCOME_TEXT } from "../../data/trainingText";
import { publishTrainingTabs, useTrainingTabs, type TrainingTab } from "../../data/trainingTabs";
import { useTrainingSetIntent } from "../../data/trainingSetIntent";
import { useTrainingSetResults } from "../../data/trainingSetResults";
import { trainingOutcomeColour } from "../../utils/trainingWordColors";
import CursorSlider from "./CursorSlider";
import ProblemBox from "./ProblemBox";
import TrainingDetails, { type TrainingDetailsSection } from "./TrainingDetails";
import { DetailsLink, DrawBox, EXPERIMENT_SECTION, LayerSwitches, ReadoutLine, type DetailsPage } from "./PanelParts";
import { ChoiceSection, FreeGenerationSection, resultSection, SpeedSection, TrainingSection, ValidationSection } from "./ResultSections";
import { ExperimentSection, ItemList, SetChooser } from "./SetParts";

const FREE_GENERATION_SECTION = "free-generation";
/** The tabs of a flight: its labelled sentence, its closed-loop sentence and each sample the prior said. */
const LABELLED = "labelled";
const CLOSED_LOOP = "closed-loop";
const sampleTab = (sample: number) => `sample-${sample}`;


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
      // the tab says the sample's number alone; its tooltip says what it is (the user, 2026-10-06)
      id: sampleTab(sentence.sample), label: `${sentence.sample}`, outcome: sentence.outcome,
      title: `A sentence the prior said (sample ${sentence.sample}, temperature ${sample.model.temperature}), flown by the executor — ` +
        ending(sentence.outcome, endS(flight, sample.model.rowIntervalS, sentence.endCycle), sentence.crossing?.decision ?? null, sentence.goArounds),
    })),
  ];
}

/** The sentence a tab stands for. */
function whichOf(tab: string): TrainingPriorWhich {
  return tab === LABELLED || tab === CLOSED_LOOP ? "closedLoop" : Number(tab.slice("sample-".length));
}

/** "At the cursor": the readout's one line (the probability along the sentence is on the cursor slider's track). A leaf: it
 *  reads the Training cursor. */
function AtTheCursor({ sample, flight, sentence }: {
  sample: TrainingPriorSample; flight: TrainingPriorFlight; sentence: TrainingPriorSentence | null;
}) {
  const { trainingCursorS } = useTrainingCursor();
  if (sentence === null) return <ReadoutLine name="At the cursor" summary="a sample's records: choose a sample's tab (0, 1 …)" />;
  const view = trainingPriorFlightView(sample, flight, sentence.sample);
  const reading = trainingReadingOf(view, sample.vocabulary.stepS, sample.model.rowIntervalS);
  const row = Math.max(readingRowAt(reading, trainingCursorS) ?? 0, 0);
  const probability = sentence.goAroundProbability;
  const blocked = TRAINING_PRIOR_BLOCKED_COLUMNS.map((column) => `${column} ${sentence.blocked[column][row].length}`).join(", ");
  const summary = `row ${row}: go-around ${(probability[row] * 100).toFixed(1)} %, ${sentence.goAroundPermitted[row] ? "permitted" : "not permitted"}, ` +
    `${sentence.onFinal[row] ? "on final" : "not on final"} · blocked ${blocked}`;
  return <ReadoutLine name="At the cursor" summary={summary} />;
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
  // the slider's marks (frontend §6.1): the first predicted step (the prior's Δ) on every tab; on a sample's tab, the
  // probability of "go-around" at each of its rows
  const marks = useMemo(() => {
    if (sample === null || flight === null) return [];
    const first = firstStepMark(flight.head.closedLoop[String(sample.model.rowIntervalS)].startS);
    if (said === null) return [first];
    const reading = trainingReadingOf(trainingPriorFlightView(sample, flight, said.sample), sample.vocabulary.stepS, sample.model.rowIntervalS);
    return [first, goAroundLine(reading, said.goAroundProbability)];
  }, [sample, flight, said]);


  // ── the details page: the experiment's results (outline §6.2 item 3, D134) ──
  const results = useTrainingSetResults("B", airport, setId);
  const b = results.status === "ready" && results.results.stage === "B" ? results.results : null;
  const absent = state.status === "invalid" ? `the set cannot be read: ${state.problem}` : "the set is loading";
  const sections: TrainingDetailsSection[] = [
    // the first section always has a body: the set's intent and provenance come from its index entry, not its sample
    entry === null ? { id: EXPERIMENT_SECTION, title: "The set and the experiment", body: <p className="training-details-lede">The index lists no set.</p> } : {
      id: EXPERIMENT_SECTION, title: "The set and the experiment",
      body: <ExperimentSection setId={entry.id} intent={intent} provenance={entry.source.readout} /> },
    resultSection(FREE_GENERATION_SECTION, "Free generation", results, b === null ? null : b.freeGeneration,
      (value, own) => <FreeGenerationSection {...value} own={own} />,
      sample === null ? <p className="experiment-details-missing">{absent}</p> : <SentencesTable sample={sample} />),
    resultSection("training", "Training", results, b === null ? null : b.training, (value) => <TrainingSection {...value} />),
    resultSection("validation", "Validation", results, b === null ? null : b.validation, (value) => <ValidationSection {...value} />),
    resultSection("speed", "Speed", results, b === null ? null : b.speed, (value) => <SpeedSection {...value} />),
    resultSection("choice", "The choice", results, b === null ? null : b.choice, (value) => <ChoiceSection {...value} />),
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
              <DetailsLink name="This sentence" onOpen={details.open(FREE_GENERATION_SECTION)} summary={
                chosen === LABELLED ? "the labelled sentence (not flown)"
                  : said === null ? `closed loop: ${ending(flight.head.closedLoop[String(sample.model.rowIntervalS)].replay.outcome,
                    endS(flight, sample.model.rowIntervalS, flight.head.closedLoop[String(sample.model.rowIntervalS)].replay.endCycle),
                    flight.head.closedLoop[String(sample.model.rowIntervalS)].replay.crossing?.decision ?? null, null)}`
                    : `sample ${said.sample}: ${ending(said.outcome, endS(flight, sample.model.rowIntervalS, said.endCycle), said.crossing?.decision ?? null, said.goArounds)}`} />
              <DetailsLink name="Sentences" onOpen={details.open(FREE_GENERATION_SECTION)} summary={
                [...new Set(sample.flights.flatMap((item) => item.sentences.map((s) => s.sample)))].sort((a, b) => a - b).map((s) => {
                  const all = sample.flights.flatMap((item) => item.sentences.filter((x) => x.sample === s));
                  return `sample ${s}: ${all.filter((x) => x.outcome === "landed").length}/${all.length} landed`;
                }).join(" · ")} />
              <AtTheCursor sample={sample} flight={flight} sentence={said} />
            </ul>
          )}
          <CursorSlider marks={marks} />
          <DrawBox legend="Draw (prior)">
            <LayerSwitches />
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
