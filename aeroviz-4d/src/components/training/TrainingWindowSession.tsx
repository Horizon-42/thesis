/**
 * TrainingWindowSession.tsx
 * -------------------------
 * The Training panel over a window set of stage C (`data/trainingWindowSample.ts`), in the layout every stage shares
 * (outline §6.2): the set chooser with the set's intent line, the list of windows, one line per readout, the cursor slider
 * (frontend §6.1: its marks the first predicted step and the round's loss of separation) and the Draw box.
 * A window holds a list of commanded aircraft (frontend §5.7, the window format of stages C and D); stage C's windows
 * hold one, and this view shows it (stage D's aircraft strip chooses among several, F3). Of the window on screen the
 * SENTENCE BAR'S TABS choose the sentence — Labelled (the labeller's open-loop reading of the recorded flight of the
 * commanded aircraft) and each round's sentence for it, Start (what the campaign starts from: the base, or another
 * campaign's round, D162), r0, r1 …
 * (`data/trainingTabs.ts`); the bar, the read-back window and the 3D layers draw it as stage A draws a sentence (a round's
 * sentence is published as a closed-loop sentence at the set's Δ, `trainingWindowFlightView`). The left panel has no
 * second chooser.
 *
 * Its readouts, one line each: "This round" and "Rounds" open the DETAILS PAGE (two sections, frontend §3 item 3, D159:
 * the set and the experiment, the models' statistics) on the statistics; "The window" (its kind and its other aircraft by
 * role) opens nothing. A window set holds no records of the speaker: no row inspector (post-training D125). The other aircraft, the
 * loss and the other rounds are drawn in 3D (`useTrainingWindowLayer`; the Draw switches here).
 *
 * A click on a word of the sentence on screen flies its segment live (`useTrainingWindowAutopilot`).
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useApp, useTrainingCursor } from "../../context/AppContext";
import useTrainingWindowAutopilot from "../../hooks/useTrainingWindowAutopilot";
import useTrainingSet from "../../hooks/useTrainingSet";
import { setTrainingWindowLayer, useTrainingWindowLayers, useTrainingWindowPick } from "../../data/trainingWindowLayers";
import {
  fetchTrainingWindowSample,
  lossesOf,
  mostTrafficAtOnce,
  roundLabel,
  onAircraftClock,
  otherOf,
  silentFromS,
  startName,
  TRAINING_WINDOW_START,
  trainingWindowFlightView,
  trainingWindowSelectionOf,
  type TrainingWindow,
  type TrainingWindowAircraft,
  type TrainingWindowRound,
  type TrainingWindowRoundEnd,
  type TrainingWindowSample,
  type TrainingWindowSentence,
  type TrainingWindowSetEntry,
  type TrainingWindowStage,
  type TrainingWindowStart,
  windowReward,
} from "../../data/trainingWindowSample";
import { readingAxisEndS, trainingReadingOf } from "../../data/trainingSample";
import { TRAINING_OUTCOME_TAG } from "../../data/trainingText";
import { firstStepMark, lossMark, silenceMark } from "../../data/trainingSlider";
import { publishTrainingTabs, useTrainingTabs, type TrainingTab } from "../../data/trainingTabs";
import { useTrainingSetIntent } from "../../data/trainingSetIntent";
import { useTrainingSetResults } from "../../data/trainingSetResults";
import { windowStatistics } from "../../data/trainingStatistics";
import { roundKind } from "../../data/trainingSentenceKind";
import CursorSlider from "./CursorSlider";
import ProblemBox from "./ProblemBox";
import TrainingDetails, { type TrainingDetailsSection } from "./TrainingDetails";
import { DetailsLink, DrawBox, EXPERIMENT_SECTION, LayerSwitches, ReadoutLine, STATISTICS_SECTION, type DetailsPage } from "./PanelParts";
import StatisticsSection from "./StatisticsSection";
import AircraftStrip from "./AircraftStrip";
import { ExperimentSection, ItemList, SetChooser } from "./SetParts";

const LABELLED = "labelled";
const roundTab = (round: TrainingWindowRound) => (round === "start" ? "start" : `round-${round}`);

/** What a window's kind is, in words. */
export const TRAINING_WINDOW_KIND_TEXT: Record<TrainingWindow["kind"], string> = {
  real: "real: the recorded traffic as it was",
  A: "A: one recorded flight of another time inserted",
  D: "D: the aircraft ahead moved in time",
  B: "B: the commanded aircraft's start moved",
};

/** The cursor of a window. When a window comes on screen, at its row 0 (D129), so the other aircraft show from the start
 *  (the cursor resets to the flight's row 0, which is before the window's, with each flight on screen). On a change of the
 *  round or of the selected aircraft (stage D), at the window's instant it was at (frontend §6.1: kept on the window's
 *  clock, each aircraft's flight clock ``clockS`` from it), or at the nearer end of the new sentence's axis (0 … ``endS``)
 *  when that does not reach it — also between Labelled and the round whose flight it is read on (one derived flight, so
 *  the provider keeps the cursor: it is only clamped). Between those changes the cursor is the user's. A LEAF: it reads
 *  the Training cursor, which moves on every chart hover. ``windowKey``: the window's identity (its set and place);
 *  ``flightKey``: the derived flight of the window, aircraft and round — the cursor is set only once that flight is the
 *  one on screen (the session publishes it after this leaf's effect has run), never on the flight before; ``labelled``:
 *  the labelled sentence is on screen; ``clockS``: the aircraft's flight time at the window's row 0. */
export function WindowCursor({ windowKey, flightKey, labelled, clockS, endS }: {
  windowKey: string; flightKey: string; labelled: boolean; clockS: number; endS: number;
}) {
  const { trainingSelection } = useApp();
  const { trainingCursorS, setTrainingCursorS } = useTrainingCursor();
  const onScreen = trainingSelection?.flight.flightKey === flightKey;
  /** The sentence last on screen and its cursor's last instant on the WINDOW's clock. */
  const kept = useRef<{ windowKey: string; flightKey: string; labelled: boolean; windowS: number } | null>(null);
  useEffect(() => {
    if (!onScreen) return;
    const last = kept.current;
    if (last !== null && last.flightKey === flightKey) {
      if (last.labelled !== labelled) {
        // the same flight read as another sentence: the provider kept the cursor; clamp it to this sentence's axis
        last.labelled = labelled;
        const atS = Math.min(trainingCursorS, endS);
        last.windowS = atS - clockS;
        if (atS !== trainingCursorS) setTrainingCursorS(atS);
        return;
      }
      last.windowS = trainingCursorS - clockS;
      return;
    }
    // another flight on screen: the setter is its own (a new flight, a new setter), set once
    const atS = Math.min(Math.max(last !== null && last.windowKey === windowKey ? last.windowS + clockS : clockS, 0), endS);
    kept.current = { windowKey, flightKey, labelled, windowS: atS - clockS };
    setTrainingCursorS(atS);
  }, [onScreen, windowKey, flightKey, labelled, clockS, endS, trainingCursorS, setTrainingCursorS]);
  return null;
}

/** A round's ending for a commanded aircraft as the tabs, lines and tables say it: the outcome, its time, the reward, the
 *  losses of separation it is in. */
function ending(aircraft: TrainingWindowAircraft, sentence: TrainingWindowSentence, end: TrainingWindowRoundEnd): string {
  const seconds = sentence.flown.tS[sentence.flown.tS.length - 1] - aircraft.firstStepS;
  const losses = lossesOf(end, aircraft.datasetId);
  return `${TRAINING_OUTCOME_TAG[sentence.outcome]} at ${seconds.toFixed(0)} s · reward ${sentence.reward.toFixed(2)} · ` +
    (losses.length === 0 ? "no loss of separation"
      : `loss of separation with ${losses.map((loss) => otherOf(loss, aircraft.datasetId)).join(", ")}`);
}

/** A round's tab label (outline §6.2 item 2): short, r1, r2 … (the user, 2026-10-06), the start by what it is
 *  (`startName`, frontend §4.3); its tooltip says what it is. */
export function roundTabLabel(round: TrainingWindowRound, start: TrainingWindowStart | null): string {
  return round === TRAINING_WINDOW_START ? `Start (${startName(start)})` : `r${round}`;
}

/** The tabs of a commanded aircraft of a window of ``stage``'s set. */
function windowTabs(window: TrainingWindow, aircraft: TrainingWindowAircraft, start: TrainingWindowStart | null,
  stage: TrainingWindowStage): TrainingTab[] {
  return [
    { id: LABELLED, label: "Labelled", outcome: null, kind: null,
      title: "The labeller's open-loop reading of the recorded flight of the commanded aircraft; not flown" +
        (window.kind === "B" ? ". This window starts from a moved start (window B): the reading is of the flight as recorded" : "") },
    ...aircraft.rounds.map((sentence, place) => ({
      id: roundTab(sentence.round), label: roundTabLabel(sentence.round, start), outcome: sentence.outcome,
      kind: roundKind(sentence.round, start, stage),
      title: `The sentence ${roundLabel(sentence.round, start)}'s model said for the commanded aircraft, flown by the executor among the ` +
        `window's other aircraft — ${ending(aircraft, sentence, window.rounds[place])} · ${sentence.goArounds} go-around${sentence.goArounds === 1 ? "" : "s"}`,
    })),
  ];
}

/** A window as the list shows it (outline §6.2, frontend §5.1): its anchor's callsign ("+k" for its k other commanded
 *  aircraft), recorded runway and kind; its commanded aircraft landed of all in its last round, each round's in the
 *  tooltip. */
function windowItem(item: TrainingWindow, sample: TrainingWindowSample) {
  const [anchor] = item.commanded;
  const landedIn = (place: number) => item.commanded.filter((aircraft) => aircraft.rounds[place].outcome === "landed").length;
  const several = item.commanded.length > 1;
  const last = item.rounds.length - 1;
  return {
    key: String(item.index), callsign: several ? `${anchor.head.callsign} +${item.commanded.length - 1}` : anchor.head.callsign,
    runway: `recorded ${anchor.head.runway}`,
    runwayTitle: "the runway the flight landed on in the record (a round's sentence may say another)",
    stratum: item.c === null ? item.kind : `${item.kind} · c ${item.c}`,
    landed: several ? landedIn(last) : anchor.rounds.filter((s) => s.outcome === "landed").length,
    of: several ? item.commanded.length : anchor.rounds.length,
    more: `${item.traffic.length} other`, title: anchor.head.flightKey,
    outcomes: several
      ? item.rounds.map((end, place) => `${roundLabel(end.round, sample.model.start)}: ${landedIn(place)}/${item.commanded.length} landed`).join("; ")
      : anchor.rounds.map((s) => `${roundLabel(s.round, sample.model.start)}: ${TRAINING_OUTCOME_TAG[s.outcome]}`).join("; "),
  };
}

/** Stage D's "this round" (frontend §5.1): W and the count of commanded aircraft, landed, go-arounds, silent aircraft,
 *  and the round's losses of separation, between commanded aircraft and with recorded ones. */
function roundReadout(window: TrainingWindow, place: number): string {
  const said = window.commanded.map((aircraft) => aircraft.rounds[place]);
  const ids = new Set(window.commanded.map((aircraft) => aircraft.datasetId));
  const losses = window.rounds[place].losses;
  const between = losses.filter((loss) => loss.aircraft.every((key) => ids.has(key))).length;
  return `W ${windowReward(window, place).toFixed(2)} of ${said.length} · ${said.filter((s) => s.outcome === "landed").length} landed · ` +
    `${said.reduce((sum, s) => sum + s.goArounds, 0)} go-arounds · ${said.filter((s) => s.silentFromRow !== null).length} silent · ` +
    `${between} losses of separation between commanded aircraft, ${losses.length - between} with recorded`;
}

/** Stage D's "this aircraft" (frontend §5.1): its callsign, its outcome and reward r in the round, the row it is silent
 *  from, its join time in the window (``sentence`` null: the labelled sentence on screen). */
function aircraftReadout(aircraft: TrainingWindowAircraft, sentence: TrainingWindowSentence | null): string {
  const said = sentence === null ? "the labelled sentence"
    : `${TRAINING_OUTCOME_TAG[sentence.outcome]} · r ${sentence.reward.toFixed(2)}` +
      (sentence.silentFromRow === null ? "" : ` · silent from row ${sentence.silentFromRow}`);
  return `${aircraft.head.callsign} · ${said} · joins at ${aircraft.joinS.toFixed(0)} s`;
}

/** The landed of each round over the set's windows. */
function roundsSummary(sample: TrainingWindowSample): string {
  return sample.model.rounds.map((round) => {
    const said = sample.windows.flatMap((window) => window.commanded.flatMap((aircraft) => aircraft.rounds.filter((item) => item.round === round)));
    return `${roundLabel(round, sample.model.start)}: ${said.filter((item) => item.outcome === "landed").length}/${said.length} landed`;
  }).join(" · ");
}

export default function TrainingWindowSession({ airport, stage, sets, details }: {
  airport: string; stage: TrainingWindowStage; sets: TrainingWindowSetEntry[]; details: DetailsPage;
}) {
  const { mode, setTrainingSelection, setTrainingIntervalS } = useApp();
  const layers = useTrainingWindowLayers();
  useTrainingWindowAutopilot();
  const [setId, setSetId] = useState<string | null>(sets[0]?.id ?? null);
  /** The window on screen: its set and its place there (a choice made in another set never selects in this one). */
  const [placed, setPlaced] = useState<{ setId: string; index: number } | null>(null);
  const entry = sets.find((item) => item.id === setId) ?? null;
  const state = useTrainingSet(entry === null ? null : `${airport}/${entry.id}/${entry.file}`,
    () => fetchTrainingWindowSample(airport, entry!.file, entry!.id, stage));
  const intent = useTrainingSetIntent(setId);

  const sample = state.status === "ready" ? state.sample : null;
  const place = sample === null ? null : placed !== null && placed.setId === sample.setId ? placed.index : 0;
  const window = sample === null || place === null ? null : sample.windows[place] ?? null;
  const windowKey = sample === null || window === null ? null : `${sample.setId}/${window.index}`;
  // the commanded aircraft on screen (stage C's windows command one; stage D's strip, keys and scene choose among several,
  // frontend §5.2): the one chosen in this window, else its first
  const [picked, setPicked] = useState<{ windowKey: string; member: number } | null>(null);
  const member = window !== null && picked !== null && picked.windowKey === windowKey && picked.member < window.commanded.length
    ? picked.member : 0;
  const aircraft = window === null ? null : window.commanded[member];
  const select = (next: number) => {
    if (windowKey !== null) setPicked({ windowKey, member: next });
  };
  const count = window?.commanded.length ?? 0;
  useEffect(() => {
    // the panel stays mounted, hidden, in the other tasks (AV28): the keys are Training's only there
    if (count < 2 || mode !== "training") return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "[" && event.key !== "]") return;
      const target = event.target as HTMLElement | null;
      if (target !== null && (target.isContentEditable || ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName))) return;
      event.preventDefault();
      setPicked((last) => {
        const now = last !== null && last.windowKey === windowKey ? last.member : 0;
        return { windowKey: windowKey!, member: event.key === "]" ? Math.min(now + 1, count - 1) : Math.max(now - 1, 0) };
      });
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [count, windowKey, mode]);
  // a commanded aircraft clicked in the scene (`useTrainingWindowLayer`) is selected
  const scenePick = useTrainingWindowPick();
  const lastPick = useRef<number | null>(scenePick?.count ?? null);
  useEffect(() => {
    if (scenePick === null || scenePick.count === lastPick.current) return;
    lastPick.current = scenePick.count;
    const found = window?.commanded.findIndex((item) => item.datasetId === scenePick.datasetId) ?? -1;
    if (found >= 0 && windowKey !== null) setPicked({ windowKey, member: found });
  }, [scenePick, window, windowKey]);

  // ── the tabs: the session gives them, the bar chooses (outline §6.2 item 2) ──
  const scope = sample === null || window === null ? null : `${stage}/${airport}/${sample.setId}/window-${window.index}`;
  const tabList = useMemo(() => (sample === null || window === null || aircraft === null ? []
    : windowTabs(window, aircraft, sample.model.start, sample.stage)), [sample, window, aircraft]);
  const shared = useTrainingTabs();
  const last = useRef<string | null>(null);
  // a window opens on the last tab chosen where it has one, else on its last round
  const lastRound = tabList.length > 0 ? tabList[tabList.length - 1].id : LABELLED;
  const chosen = shared !== null && shared.scope === scope && tabList.some((tab) => tab.id === shared.chosen) ? shared.chosen
    : last.current !== null && tabList.some((tab) => tab.id === last.current) ? last.current : lastRound;
  // only a choice made among the tabs of an item on screen is remembered (not the fallback while a set loads)
  useEffect(() => {
    if (scope !== null) last.current = chosen;
  }, [scope, chosen]);
  useEffect(() => {
    if (scope !== null) publishTrainingTabs({ scope, tabs: tabList, chosen });
  }, [scope, tabList, chosen]);
  useEffect(() => () => publishTrainingTabs(null), []);

  // the round on screen (Labelled reads the open-loop sentence of the round last flown on screen: the same head)
  const shown: TrainingWindowRound | null = window === null ? null
    : chosen === LABELLED ? window.rounds[window.rounds.length - 1].round
      : window.rounds.find((item) => roundTab(item.round) === chosen)!.round;
  const shownPlace = window === null || shown === null ? -1 : window.rounds.findIndex((item) => item.round === shown);
  useEffect(() => {
    if (sample !== null) setTrainingIntervalS(chosen === LABELLED ? null : sample.model.rowIntervalS);
  }, [sample, chosen, setTrainingIntervalS]);
  useEffect(() => {
    if (sample === null || window === null || aircraft === null || shown === null) {
      setTrainingSelection(null);
      return;
    }
    setTrainingSelection(trainingWindowSelectionOf(sample, trainingWindowFlightView(sample, window, aircraft, shown)));
  }, [sample, window, aircraft, shown, setTrainingSelection]);
  useEffect(() => () => setTrainingSelection(null), [setTrainingSelection]);

  const sentence = aircraft === null || shownPlace < 0 ? null : aircraft.rounds[shownPlace];
  const end = window === null || shownPlace < 0 ? null : window.rounds[shownPlace];
  // the round's sentence on screen as the views read it, and the slider's marks (frontend §6.1): the first predicted step
  // on every tab; on a round's tab, the losses of separation the aircraft is in
  const view = sample === null || window === null || aircraft === null || shown === null ? null
    : trainingWindowFlightView(sample, window, aircraft, shown);
  const endS = sample === null || view === null ? 0
    : readingAxisEndS(trainingReadingOf(view, sample.vocabulary.stepS, chosen === LABELLED ? null : sample.model.rowIntervalS));
  const silentS = sample === null || aircraft === null || sentence === null || chosen === LABELLED ? null
    : silentFromS(aircraft, sentence, sample.model.rowIntervalS);
  const marks = useMemo(() => {
    if (aircraft === null || end === null) return [];
    const losses = chosen === LABELLED ? [] : lossesOf(end, aircraft.datasetId);
    return [firstStepMark(aircraft.firstStepS),
      ...losses.map((loss) => lossMark(onAircraftClock(aircraft, loss.timeS), otherOf(loss, aircraft.datasetId))),
      ...(silentS === null ? [] : [silenceMark(silentS, endS)])];
  }, [aircraft, end, chosen, silentS, endS]);

  // ── the details page: the set and the models' statistics (frontend §3 item 3, D159) ──
  const results = useTrainingSetResults(stage, airport, setId);
  const c = results.status === "ready" ? results.results : null;
  const absent = state.status === "invalid" ? `the set cannot be read: ${state.problem}` : "the set is loading";
  const roles = { recorded: 0, inserted: 0, moved: 0 };
  for (const aircraft of window?.traffic ?? []) roles[aircraft.role] += 1;
  const sections: TrainingDetailsSection[] = [
    // the first section always has a body: the set's intent and provenance come from its index entry, not its sample
    entry === null ? { id: EXPERIMENT_SECTION, title: "The set and the experiment", body: <p className="training-details-lede">The index lists no set.</p> } : {
      id: EXPERIMENT_SECTION, title: "The set and the experiment",
      body: <ExperimentSection setId={entry.id} intent={intent} provenance={entry.model.campaign} /> },
    sample === null ? { id: STATISTICS_SECTION, title: "The models' statistics", body: null, absent }
      : { id: STATISTICS_SECTION, title: "The models' statistics",
        body: <StatisticsSection table={windowStatistics(sample, c)} results={results} /> },
  ];

  const choices = sets.map((item) => ({ id: item.id, count: `${item.windows} windows`, title: item.title,
    smoke: item.source.smoke || item.source.campaignSmoke }));
  return (
    <>
      {details.shown !== null ? (
        <TrainingDetails context={[airport, setId ?? "no set", sample === null ? "" : `${sample.windows.length} windows`].filter(Boolean).join(" · ")}
          sections={sections} sectionId={details.shown.section} onSection={details.show} onClose={details.close}
          opener={details.shown.opener} />
      ) : null}
      <SetChooser sets={choices} setId={setId} onChange={(id) => setSetId(id)} intent={intent} onOpenExperiment={details.open(EXPERIMENT_SECTION)} />
      {state.status === "loading" ? <p className="training-note" role="status">Loading {entry?.file} …</p> : null}
      {state.status === "invalid" ? <ProblemBox title={`Set ${setId} cannot be read.`} detail={state.problem} /> : null}

      {sample === null ? null : (
        <>
          <ItemList label="The set's windows" active={window === null ? null : String(window.index)}
            onSelect={(key) => setPlaced({ setId: sample.setId, index: Number(key) })}
            items={sample.windows.map((item) => windowItem(item, sample))} />
          {window === null || aircraft === null || sentence === null || end === null ? null : (
            <ul className="training-details-links" aria-label="Readouts">
              {sample.stage === "D" ? (
                <>
                  <DetailsLink name="This round" onOpen={details.open(STATISTICS_SECTION)}
                    summary={chosen === LABELLED ? "the labelled sentences (not flown)" : roundReadout(window, shownPlace)} />
                  <ReadoutLine name="This aircraft" summary={aircraftReadout(aircraft, chosen === LABELLED ? null : sentence)} />
                  <ReadoutLine name="The window" summary={`${TRAINING_WINDOW_KIND_TEXT[window.kind]}${window.c === null ? "" : ` · c ${window.c}`} · ` +
                    `${window.commanded.length} commanded · at most ${mostTrafficAtOnce(window)} recorded at once`} />
                </>
              ) : (
                <>
                  <DetailsLink name="This round" onOpen={details.open(STATISTICS_SECTION)}
                    summary={chosen === LABELLED ? "the labelled sentence (not flown)" : `${roundLabel(sentence.round, sample.model.start)}: ${ending(aircraft, sentence, end)}`} />
                  <DetailsLink name="Rounds" onOpen={details.open(STATISTICS_SECTION)} summary={roundsSummary(sample)} />
                  <ReadoutLine name="The window" summary={`${TRAINING_WINDOW_KIND_TEXT[window.kind]} · ` +
                    (window.traffic.length === 0 ? "no other aircraft"
                      : (["recorded", "inserted", "moved"] as const).filter((role) => roles[role] > 0).map((role) => `${roles[role]} ${role}`).join(", "))} />
                </>
              )}
            </ul>
          )}
          {window === null ? null : (
            <AircraftStrip aircraft={window.commanded} place={chosen === LABELLED ? null : shownPlace} selected={member} onSelect={select} />
          )}
          <CursorSlider marks={marks} />
          {window === null || aircraft === null || view === null ? null : (
            <WindowCursor windowKey={windowKey!} flightKey={view.flightKey} labelled={chosen === LABELLED}
              clockS={aircraft.clockS} endS={endS} />
          )}
          <DrawBox legend="Draw (window)">
            <LayerSwitches />
            {window !== null && window.commanded.length > 1 ? (
              <label title="the flown tracks of the window's other commanded aircraft in the round (their points stay)">
                <input type="checkbox" checked={layers.otherCommanded} onChange={(event) => setTrainingWindowLayer("otherCommanded", event.target.checked)} />
                Other commanded tracks
              </label>
            ) : null}
            <label title="the other aircraft where their records have them at the cursor's time">
              <input type="checkbox" checked={layers.traffic} onChange={(event) => setTrainingWindowLayer("traffic", event.target.checked)} />
              Other aircraft
            </label>
            <label title="each other aircraft's track over the window, in its role's colour">
              <input type="checkbox" checked={layers.trafficTracks} onChange={(event) => setTrainingWindowLayer("trafficTracks", event.target.checked)} />
              Their tracks
            </label>
            <label title="the two aircraft at the loss of separation that ended the round, joined">
              <input type="checkbox" checked={layers.loss} onChange={(event) => setTrainingWindowLayer("loss", event.target.checked)} />
              Loss of separation
            </label>
            <label title="the flown paths of the window's other rounds, thin beside the one on screen">
              <input type="checkbox" checked={layers.otherRounds} onChange={(event) => setTrainingWindowLayer("otherRounds", event.target.checked)} />
              Other rounds
            </label>
          </DrawBox>
        </>
      )}
    </>
  );
}
