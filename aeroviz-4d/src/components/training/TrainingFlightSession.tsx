/**
 * TrainingFlightSession.tsx
 * -------------------------
 * The Training panel over a stage-A set, in the layout every stage shares (outline §6.2): the list of flights (four
 * columns: the callsign, the runway, the stratum, the closed-loop sentences that landed, one for each Δ), one line per
 * readout, the cursor slider (frontend §6.1: its marks the first predicted step and the corrections) and the Draw box; the set chooser above it is the panel's. Of the flight on screen the SENTENCE BAR'S TABS choose
 * the sentence — Labelled and one per Δ (`data/trainingTabs.ts`); the session maps the choice to `trainingIntervalS` (the
 * set's first Δ when the first set opens; kept across sets when the new set has it) and publishes the flight through
 * `trainingSelection` — the sentence bar, the read-back window and the 3D layer draw it.
 *
 * THE DOCK ENDS ABOVE THE SENTENCE BAR (the bar measures itself, `--training-bar-height`): the flight list takes the
 * height left over — never less than a few rows, the dock scrolling below that — so the switches under it are never
 * covered. Its readouts are one line each; the details page (`TrainingDetails`) has two sections (frontend §3 item 3,
 * D159): "The set and the experiment" and "The models' statistics" (a row for each Δ's closed loop).
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "../../context/AppContext";
import CursorSlider from "./CursorSlider";
import TrainingDetails, { type TrainingDetailsSection } from "./TrainingDetails";
import { DetailsLink, DrawBox, EXPERIMENT_SECTION, LayerSwitches, STATISTICS_SECTION, type DetailsPage } from "./PanelParts";
import StatisticsSection from "./StatisticsSection";
import { ExperimentSection, ItemList } from "./SetParts";
import {
  closedCycleTimeS,
  correctionCount,
  trainingReadingOf,
  trainingSelectionOf,
  type TrainingFlight,
  type TrainingSample,
  type TrainingSetEntry,
} from "../../data/trainingSample";
import { checkMark, TRAINING_OUTCOME_TAG } from "../../data/trainingText";
import { publishTrainingTabs, useTrainingTabs, type TrainingTab } from "../../data/trainingTabs";
import type { TrainingSetIntentState } from "../../data/trainingSetIntent";
import { useTrainingSetResults } from "../../data/trainingSetResults";
import { correctionMarks, firstStepMark } from "../../data/trainingSlider";
import { stageAStatistics } from "../../data/trainingStatistics";

const LABELLED = "labelled";
const intervalTab = (interval: number) => `interval-${interval}`;

/** The flight's tabs: Labelled, then the closed-loop sentence at each Δ with its outcome. */
function flightTabs(sample: TrainingSample, flight: TrainingFlight): TrainingTab[] {
  return [
    { id: LABELLED, label: "Labelled", outcome: null, kind: null,
      title: "The labeller's reading of the observed flight (open loop): its words on the 2 s rows, and the envelopes of its words on the observed track; not flown" },
    ...sample.vocabulary.rowIntervalsS.map((interval) => {
      const closed = flight.closedLoop[String(interval)];
      const decision = closed.replay.crossing?.decision ?? null;
      return {
        id: intervalTab(interval), label: `Δ ${interval} s`, outcome: closed.replay.outcome, kind: "closedLoop" as const,
        title: `The closed-loop sentence at Δ = ${interval} s, flown by the executor — ${TRAINING_OUTCOME_TAG[closed.replay.outcome]} at ` +
          `${closedCycleTimeS(closed, closed.replay.endCycle)} s${decision === null ? "" : ` · DA ${checkMark(decision.passed)}`} · ` +
          `${correctionCount(closed.events)} words added by the reading`,
      };
    }),
  ];
}

/** The flown flights of the set at each Δ, counted: landed, DA passed, words added. */
function flownSummary(sample: TrainingSample): string {
  return sample.vocabulary.rowIntervalsS.map((interval) => {
    const closed = sample.flights.map((flight) => flight.closedLoop[String(interval)]);
    const landed = closed.filter((item) => item.replay.outcome === "landed").length;
    return `Δ ${interval} s: ${landed}/${closed.length} landed`;
  }).join(" · ");
}

/** ``sample``: the set open — null while the next one loads (the session is kept: its choices per set outlive a switch of
 *  sets, and it publishes no flight meanwhile). */
export default function TrainingFlightSession({ airport, sample, entry, details, intent }: {
  airport: string; sample: TrainingSample | null; entry: TrainingSetEntry | null; details: DetailsPage; intent: TrainingSetIntentState;
}) {
  const { setTrainingSelection, setTrainingIntervalS } = useApp();
  const [flightKey, setFlightKey] = useState<string | null>(null);
  const flight = sample === null ? null : sample.flights.find((item) => item.flightKey === flightKey) ?? sample.flights[0] ?? null;

  // ── the tabs: the session gives them, the bar chooses (outline §6.2 item 2) ──
  const scope = sample === null || flight === null ? null : `A/${airport}/${sample.setId}/${flight.flightKey}`;
  const tabList = useMemo(() => (sample === null || flight === null ? [] : flightTabs(sample, flight)), [sample, flight]);
  const shared = useTrainingTabs();
  const last = useRef<string | null>(null);
  // the set's first Δ when the first set opens; after that the tab last chosen when the new set or flight has it
  const first = tabList[1]?.id ?? LABELLED;
  const chosen = shared !== null && shared.scope === scope && tabList.some((tab) => tab.id === shared.chosen) ? shared.chosen
    : last.current !== null && tabList.some((tab) => tab.id === last.current) ? last.current : first;
  // only a choice made among the tabs of an item on screen is remembered (not the fallback while a set loads)
  useEffect(() => {
    if (scope !== null) last.current = chosen;
  }, [scope, chosen]);
  useEffect(() => {
    if (scope !== null) publishTrainingTabs({ scope, tabs: tabList, chosen });
  }, [scope, tabList, chosen]);
  useEffect(() => () => publishTrainingTabs(null), []);
  const intervalS = chosen === LABELLED ? null : Number(chosen.slice("interval-".length));
  useEffect(() => {
    if (scope !== null) setTrainingIntervalS(intervalS);
  }, [scope, intervalS, setTrainingIntervalS]);

  // ── publish what the other views draw ─────────────────────────────────────
  useEffect(() => {
    setTrainingSelection(flight && sample ? trainingSelectionOf(sample, flight) : null);
  }, [sample, flight, setTrainingSelection]);
  useEffect(() => () => setTrainingSelection(null), [setTrainingSelection]);
  const closed = flight === null || intervalS === null ? null : flight.closedLoop[String(intervalS)];
  // the slider's marks (frontend §6.1): on a closed-loop sentence, its first predicted step and the rows of the words the
  // reading added; the labelled sentence has no first predicted step of its own (each Δ has its own), so no mark
  const marks = useMemo(() => {
    if (sample === null || flight === null || intervalS === null) return [];
    const reading = trainingReadingOf(flight, sample.vocabulary.stepS, intervalS);
    return [firstStepMark(reading.originS), ...correctionMarks(reading)];
  }, [sample, flight, intervalS]);

  // ── the details page: the set and the models' statistics (frontend §3 item 3, D159) ──
  const results = useTrainingSetResults("A", airport, entry === null ? null : entry.id);
  const a = results.status === "ready" ? results.results : null;
  const sections: TrainingDetailsSection[] = [
    // the first section always has a body: the set's intent and provenance come from its index entry, not its sample
    entry === null ? { id: EXPERIMENT_SECTION, title: "The set and the experiment", body: <p className="training-details-lede">The index lists no set.</p> } : {
      id: EXPERIMENT_SECTION, title: "The set and the experiment",
      body: <ExperimentSection setId={entry.id} intent={intent} provenance={entry.source.instructions} windows={null} /> },
    sample === null ? { id: STATISTICS_SECTION, title: "The models' statistics", body: null, absent: "the set is loading or cannot be read" }
      : { id: STATISTICS_SECTION, title: "The models' statistics",
        body: <StatisticsSection table={stageAStatistics(sample, a)} results={results} /> },
  ];
  return (
    <>
      {details.shown !== null ? (
        <TrainingDetails context={[airport, entry?.id ?? "no set", sample === null ? "" : `${sample.flights.length} flights`].filter(Boolean).join(" · ")}
          sections={sections} sectionId={details.shown.section} onSection={details.show} onClose={details.close}
          opener={details.shown.opener} />
      ) : null}
      {sample === null ? null : (
        <>
          {entry && entry.flights !== sample.flights.length
            ? <p className="training-note">The index says {entry.flights} flights: this set's two files are from different exports.</p> : null}
          <ItemList label="The set's flights" active={flight?.flightKey ?? null} onSelect={setFlightKey}
            items={sample.flights.map((item) => {
              const ends = sample.vocabulary.rowIntervalsS.map((interval) => [interval, item.closedLoop[String(interval)].replay.outcome] as const);
              return {
                key: item.flightKey, callsign: item.callsign, runway: item.runway, stratum: item.stratum, title: item.flightKey,
                landed: ends.filter(([, outcome]) => outcome === "landed").length, of: ends.length,
                outcomes: ends.map(([interval, outcome]) => `Δ ${interval} s: ${TRAINING_OUTCOME_TAG[outcome]}`).join("; "),
              };
            })} />
          <ul className="training-details-links" aria-label="Readouts">
            <DetailsLink name="Flown flight" onOpen={details.open(STATISTICS_SECTION)} summary={closed === null
              ? "the labelled sentence (not flown): choose a Δ tab"
              : `Δ ${closed.rowIntervalS} s: ${TRAINING_OUTCOME_TAG[closed.replay.outcome]} at ${closedCycleTimeS(closed, closed.replay.endCycle)} s` +
                (closed.replay.crossing?.decision ? ` · DA ${checkMark(closed.replay.crossing.decision.passed)}` : " · no DA check")} />
            <DetailsLink name="Flown flights" summary={flownSummary(sample)} onOpen={details.open(STATISTICS_SECTION)} />
          </ul>
          <CursorSlider marks={marks} />
          <DrawBox>
            <LayerSwitches />
          </DrawBox>
        </>
      )}
    </>
  );
}
