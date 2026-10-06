/**
 * TrainingWindowSession.tsx
 * -------------------------
 * The Training panel over a window set of stage C (`data/trainingWindowSample.ts`), in the layout every stage shares
 * (outline §6.2): the set chooser with the set's intent line, the list of windows, one line per readout, the cursor slider
 * (frontend §6.1: its marks the first predicted step and the round's loss of separation) and the Draw box.
 * A window holds a list of commanded aircraft (frontend §5.7, the window format of stages C and D); stage C's windows
 * hold one, and this view shows it (stage D's aircraft strip chooses among several, F3). Of the window on screen the
 * SENTENCE BAR'S TABS choose the sentence — Labelled (the labeller's open-loop reading of the recorded flight of the
 * commanded aircraft) and each round's sentence for it, Start (base), Round 1 …
 * (`data/trainingTabs.ts`); the bar, the read-back window and the 3D layers draw it as stage A draws a sentence (a round's
 * sentence is published as a closed-loop sentence at the set's Δ, `trainingWindowFlightView`). The left panel has no
 * second chooser.
 *
 * Its readouts, one line each, open the DETAILS PAGE: "This round" and "Rounds" → Rounds; "The window" → The window (its
 * kind, its moves, its other aircraft, the round's end, the threshold, the rows where the speed-word mask acted, the faulty
 * points). A window set holds no records of the speaker: no row inspector (post-training D125). The other aircraft, the
 * loss and the other rounds are drawn in 3D (`useTrainingWindowLayer`; the Draw switches here).
 *
 * A click on a word of the sentence on screen flies its segment live (`useTrainingWindowAutopilot`).
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useApp, useTrainingCursor } from "../../context/AppContext";
import useTrainingWindowAutopilot from "../../hooks/useTrainingWindowAutopilot";
import useTrainingSet from "../../hooks/useTrainingSet";
import { setTrainingWindowLayer, useTrainingWindowLayers } from "../../data/trainingWindowLayers";
import {
  fetchTrainingWindowSample,
  lossesOf,
  onAircraftClock,
  otherOf,
  trainingWindowFlightView,
  trainingWindowSelectionOf,
  windowShiftText,
  type TrainingWindow,
  type TrainingWindowAircraft,
  type TrainingWindowRound,
  type TrainingWindowRoundEnd,
  type TrainingWindowSample,
  type TrainingWindowSentence,
  type TrainingWindowSetEntry,
} from "../../data/trainingWindowSample";
import { readingAxisEndS, trainingReadingOf } from "../../data/trainingSample";
import { checkMark, crossingText, TRAINING_OUTCOME_TAG, TRAINING_OUTCOME_TEXT } from "../../data/trainingText";
import { firstStepMark, lossMark } from "../../data/trainingSlider";
import { publishTrainingTabs, useTrainingTabs, type TrainingTab } from "../../data/trainingTabs";
import { useTrainingSetIntent } from "../../data/trainingSetIntent";
import { useTrainingSetResults } from "../../data/trainingSetResults";
import { trainingOutcomeColour } from "../../utils/trainingWordColors";
import { TRAINING_WINDOW_ROLE_COLOR } from "../../hooks/useTrainingWindowLayer";
import CursorSlider from "./CursorSlider";
import ProblemBox from "./ProblemBox";
import TrainingDetails, { type TrainingDetailsSection } from "./TrainingDetails";
import { DetailsLink, DrawBox, EXPERIMENT_SECTION, LayerSwitches, type DetailsPage } from "./PanelParts";
import { ChecksSection, resultSection, RoundsSection, SpeedSection } from "./ResultSections";
import { ExperimentSection, ItemList, SetChooser } from "./SetParts";

const ROUNDS_SECTION = "rounds";
const WINDOW_SECTION = "window";
const LABELLED = "labelled";
const roundTab = (round: TrainingWindowRound) => (round === "start" ? "start" : `round-${round}`);

/** What a window's kind is, in words. */
export const TRAINING_WINDOW_KIND_TEXT: Record<TrainingWindow["kind"], string> = {
  real: "real: the recorded traffic as it was",
  A: "A: one recorded flight of another time inserted",
  D: "D: the aircraft ahead moved in time",
  B: "B: the commanded aircraft's start moved",
};

/** A round as its table names it. */
export function roundLabel(round: TrainingWindowRound): string {
  return round === "start" ? "start (base)" : `round ${round}`;
}

/** The window on screen and its round's end for the commanded aircraft on screen. */
function WindowEnd({ window, aircraft, sentence, end }: {
  window: TrainingWindow; aircraft: TrainingWindowAircraft; sentence: TrainingWindowSentence; end: TrainingWindowRoundEnd;
}) {
  const losses = lossesOf(end, aircraft.datasetId);
  const move = aircraft.startMove;
  const roles = { recorded: 0, inserted: 0, moved: 0 };
  for (const aircraft of window.traffic) roles[aircraft.role] += 1;
  return (
    <dl className="training-flown-block" aria-label="The window and its end">
      <div><dt>Window</dt><dd>{TRAINING_WINDOW_KIND_TEXT[window.kind]}</dd></div>
      {window.kind === "B" ? (
        <div><dt>Start moved</dt><dd>turned {move.turnDeg.toFixed(1)}° about the airport · {move.heightM >= 0 ? "+" : ""}{move.heightM.toFixed(0)} m · speed × {move.speedScale.toFixed(3)}</dd></div>
      ) : null}
      {window.moved.length > 0 ? (
        <div><dt>Moved</dt><dd>{window.moved.map(([key, shift]) => `${key} by ${windowShiftText(shift)}`).join("; ")}</dd></div>
      ) : null}
      <div>
        <dt>Other aircraft</dt>
        <dd>
          {window.traffic.length === 0 ? "none in the window" : (["recorded", "inserted", "moved"] as const).filter((role) => roles[role] > 0).map((role) => (
            <span key={role} style={{ color: TRAINING_WINDOW_ROLE_COLOR[role] }}>{roles[role]} {role} </span>
          ))}
        </dd>
      </div>
      <div>
        <dt>End</dt>
        <dd style={{ color: trainingOutcomeColour(sentence.outcome) }}>
          {TRAINING_OUTCOME_TEXT[sentence.outcome]} · reward {sentence.reward.toFixed(2)}
        </dd>
      </div>
      {losses.map((loss) => (
        <div key={`${loss.step}-${otherOf(loss, aircraft.datasetId)}`}>
          <dt>Loss of separation</dt>
          <dd>
            with {otherOf(loss, aircraft.datasetId)} at {(onAircraftClock(aircraft, loss.timeS) - aircraft.firstStepS).toFixed(0)} s
            ({loss.kind}, {loss.relation}) ·{" "}
            {loss.distanceM.toFixed(0)} m of {loss.requiredM.toFixed(0)} m required · {loss.verticalM.toFixed(0)} m apart in height
            {loss.wakeKnown ? "" : " · wake category unknown"}
            {loss.readsFault ? " · the other aircraft read a faulty point near it" : ""}
          </dd>
        </div>
      ))}
      {sentence.crossing === null ? null : <div><dt>Threshold</dt><dd>{crossingText(sentence.crossing)}</dd></div>}
      <div>
        <dt>Speed-word mask</dt>
        <dd>acted on {sentence.speedMaskRows} row{sentence.speedMaskRows === 1 ? "" : "s"}</dd>
      </div>
      <div>
        <dt>Faulty points</dt>
        <dd>{end.faultySteps === 0 ? `${checkMark(true)} no step read one` : `${end.faultySteps} steps where a recorded aircraft read one`}</dd>
      </div>
    </dl>
  );
}

/** The cursor of a window. When a window comes on screen, at its row 0 (D129), so the other aircraft show from the start
 *  (the cursor resets to the flight's row 0, which is before the window's, with each flight on screen). On a change of the
 *  round, at the window's instant it was at (frontend §6.1: the rounds of a window share its flight's clock), or at the
 *  nearer end of the new sentence's axis (0 … ``endS``) when that does not reach it — also between Labelled and the round
 *  whose flight it is read on (one derived flight, so the provider keeps the cursor: it is only clamped). Between those
 *  changes the cursor is the user's. A LEAF: it reads the Training cursor, which moves on every chart hover.
 *  ``windowKey``: the window's identity (its set and place); ``flightKey``: the derived flight of the window and round —
 *  the cursor is set only once that flight is the one on screen (the session publishes it after this leaf's effect has
 *  run), never on the flight before; ``labelled``: the labelled sentence is on screen. */
export function WindowCursor({ windowKey, flightKey, labelled, row0S, endS }: {
  windowKey: string; flightKey: string; labelled: boolean; row0S: number; endS: number;
}) {
  const { trainingSelection } = useApp();
  const { trainingCursorS, setTrainingCursorS } = useTrainingCursor();
  const onScreen = trainingSelection?.flight.flightKey === flightKey;
  /** The sentence last on screen and its cursor's last instant. */
  const kept = useRef<{ windowKey: string; flightKey: string; labelled: boolean; atS: number } | null>(null);
  useEffect(() => {
    if (!onScreen) return;
    const last = kept.current;
    if (last !== null && last.flightKey === flightKey) {
      if (last.labelled !== labelled) {
        // the same flight read as another sentence: the provider kept the cursor; clamp it to this sentence's axis
        last.labelled = labelled;
        const atS = Math.min(trainingCursorS, endS);
        last.atS = atS;
        if (atS !== trainingCursorS) setTrainingCursorS(atS);
        return;
      }
      last.atS = trainingCursorS;
      return;
    }
    // another flight on screen: the setter is its own (a new flight, a new setter), set once
    const atS = last !== null && last.windowKey === windowKey ? Math.min(last.atS, endS) : row0S;
    kept.current = { windowKey, flightKey, labelled, atS };
    setTrainingCursorS(atS);
  }, [onScreen, windowKey, flightKey, labelled, row0S, endS, trainingCursorS, setTrainingCursorS]);
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

/** A round's tab label (outline §6.2 item 2): short, r1, r2 … (the user, 2026-10-06); its tooltip says what it is. */
export function roundTabLabel(round: TrainingWindowRound): string {
  return round === "start" ? "Start (base)" : `r${round}`;
}

/** The tabs of a commanded aircraft of a window. */
function windowTabs(window: TrainingWindow, aircraft: TrainingWindowAircraft): TrainingTab[] {
  return [
    { id: LABELLED, label: "Labelled", outcome: null,
      title: "The labeller's open-loop reading of the recorded flight of the commanded aircraft; not flown" +
        (window.kind === "B" ? ". This window starts from a moved start (window B): the reading is of the flight as recorded" : "") },
    ...aircraft.rounds.map((sentence, place) => ({
      id: roundTab(sentence.round), label: roundTabLabel(sentence.round), outcome: sentence.outcome,
      title: `The sentence ${roundLabel(sentence.round)}'s model said for the commanded aircraft, flown by the executor among the ` +
        `window's other aircraft — ${ending(aircraft, sentence, window.rounds[place])} · ${sentence.goArounds} go-around${sentence.goArounds === 1 ? "" : "s"}`,
    })),
  ];
}

/** Every window's rounds (the details page's Rounds). */
function RoundsTable({ sample }: { sample: TrainingWindowSample }) {
  const rounds = sample.model.rounds;
  return (
    <>
      <table className="training-flown-table training-prior-table" aria-label="Every window's rounds">
        <thead>
          <tr><th scope="col">window</th>{rounds.map((round) => <th key={String(round)} scope="col">{roundLabel(round)}</th>)}</tr>
        </thead>
        <tbody>
          {sample.windows.map((window) => {
            const [aircraft] = window.commanded;
            return (
              <tr key={window.index}>
                <th scope="row" title={aircraft.head.flightKey}>{aircraft.head.callsign} · {window.kind}</th>
                {rounds.map((round, place) => {
                  const sentence = aircraft.rounds[place];
                  return (
                    <td key={String(round)} style={{ color: trainingOutcomeColour(sentence.outcome) }} title={TRAINING_OUTCOME_TEXT[sentence.outcome]}>
                      {ending(aircraft, sentence, window.rounds[place])} · {sentence.goArounds} go-arounds · {sentence.events.length} words
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}

/** The landed of each round over the set's windows. */
function roundsSummary(sample: TrainingWindowSample): string {
  return sample.model.rounds.map((round) => {
    const said = sample.windows.flatMap((window) => window.commanded.flatMap((aircraft) => aircraft.rounds.filter((item) => item.round === round)));
    return `${roundLabel(round)}: ${said.filter((item) => item.outcome === "landed").length}/${said.length} landed`;
  }).join(" · ");
}

export default function TrainingWindowSession({ airport, sets, details }: {
  airport: string; sets: TrainingWindowSetEntry[]; details: DetailsPage;
}) {
  const { setTrainingSelection, setTrainingIntervalS } = useApp();
  const layers = useTrainingWindowLayers();
  useTrainingWindowAutopilot();
  const [setId, setSetId] = useState<string | null>(sets[0]?.id ?? null);
  /** The window on screen: its set and its place there (a choice made in another set never selects in this one). */
  const [placed, setPlaced] = useState<{ setId: string; index: number } | null>(null);
  const entry = sets.find((item) => item.id === setId) ?? null;
  const state = useTrainingSet(entry === null ? null : `${airport}/${entry.id}/${entry.file}`,
    () => fetchTrainingWindowSample(airport, entry!.file, entry!.id));
  const intent = useTrainingSetIntent(setId);

  const sample = state.status === "ready" ? state.sample : null;
  const place = sample === null ? null : placed !== null && placed.setId === sample.setId ? placed.index : 0;
  const window = sample === null || place === null ? null : sample.windows[place] ?? null;
  // the commanded aircraft on screen: stage C's windows command one (stage D's strip chooses among several, F3)
  const aircraft = window === null ? null : window.commanded[0];

  // ── the tabs: the session gives them, the bar chooses (outline §6.2 item 2) ──
  const scope = sample === null || window === null ? null : `C/${airport}/${sample.setId}/window-${window.index}`;
  const tabList = useMemo(() => (window === null || aircraft === null ? [] : windowTabs(window, aircraft)), [window, aircraft]);
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
  const marks = useMemo(() => {
    if (aircraft === null || end === null) return [];
    const losses = chosen === LABELLED ? [] : lossesOf(end, aircraft.datasetId);
    return [firstStepMark(aircraft.firstStepS),
      ...losses.map((loss) => lossMark(onAircraftClock(aircraft, loss.timeS), otherOf(loss, aircraft.datasetId)))];
  }, [aircraft, end, chosen]);

  // ── the details page: the experiment's results (outline §6.2 item 3, D134) ──
  const results = useTrainingSetResults("C", airport, setId);
  const c = results.status === "ready" && results.results.stage === "C" ? results.results : null;
  const absent = state.status === "invalid" ? `the set cannot be read: ${state.problem}` : "the set is loading";
  const roles = { recorded: 0, inserted: 0, moved: 0 };
  for (const aircraft of window?.traffic ?? []) roles[aircraft.role] += 1;
  const sections: TrainingDetailsSection[] = [
    // the first section always has a body: the set's intent and provenance come from its index entry, not its sample
    entry === null ? { id: EXPERIMENT_SECTION, title: "The set and the experiment", body: <p className="training-details-lede">The index lists no set.</p> } : {
      id: EXPERIMENT_SECTION, title: "The set and the experiment",
      body: <ExperimentSection setId={entry.id} intent={intent} provenance={entry.model.campaign} /> },
    resultSection(ROUNDS_SECTION, "Rounds", results, c === null ? null : c.rounds, (value, own) => <RoundsSection {...value} own={own} />,
      sample === null ? <p className="experiment-details-missing">{absent}</p> : <RoundsTable sample={sample} />),
    resultSection("speed", "Speed", results, c === null ? null : c.speed, (value) => <SpeedSection {...value} />),
    resultSection("checks", "The checks", results, c === null ? null : c.checks, (value) => <ChecksSection {...value} />),
    window === null || aircraft === null || sentence === null || end === null ? { id: WINDOW_SECTION, title: "The window", body: null, absent }
      : { id: WINDOW_SECTION, title: "The window", body: <WindowEnd window={window} aircraft={aircraft} sentence={sentence} end={end} /> },
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
            items={sample.windows.map((item) => {
              const [anchor] = item.commanded;
              return {
                key: String(item.index), callsign: anchor.head.callsign, runway: `recorded ${anchor.head.runway}`,
                runwayTitle: "the runway the flight landed on in the record (a round's sentence may say another)", stratum: item.kind,
                landed: anchor.rounds.filter((s) => s.outcome === "landed").length, of: anchor.rounds.length,
                more: `${item.traffic.length} other`, title: anchor.head.flightKey,
                outcomes: anchor.rounds.map((s) => `${roundLabel(s.round)}: ${TRAINING_OUTCOME_TAG[s.outcome]}`).join("; "),
              };
            })} />
          {window === null || aircraft === null || sentence === null || end === null ? null : (
            <ul className="training-details-links" aria-label="Readouts">
              <DetailsLink name="This round" onOpen={details.open(ROUNDS_SECTION)}
                summary={chosen === LABELLED ? "the labelled sentence (not flown)" : `${roundLabel(sentence.round)}: ${ending(aircraft, sentence, end)}`} />
              <DetailsLink name="Rounds" onOpen={details.open(ROUNDS_SECTION)} summary={roundsSummary(sample)} />
              <DetailsLink name="The window" onOpen={details.open(WINDOW_SECTION)} summary={`${TRAINING_WINDOW_KIND_TEXT[window.kind]} · ` +
                (window.traffic.length === 0 ? "no other aircraft"
                  : (["recorded", "inserted", "moved"] as const).filter((role) => roles[role] > 0).map((role) => `${roles[role]} ${role}`).join(", "))} />
            </ul>
          )}
          <CursorSlider marks={marks} />
          {window === null || aircraft === null || view === null ? null : (
            <WindowCursor windowKey={`${sample.setId}/${window.index}`} flightKey={view.flightKey} labelled={chosen === LABELLED}
              row0S={aircraft.clockS} endS={endS} />
          )}
          <DrawBox legend="Draw (window)">
            <LayerSwitches />
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
