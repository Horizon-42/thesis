/**
 * TrainingWindowSession.tsx
 * -------------------------
 * The Training panel over a window set of stage C (`data/trainingWindowSample.ts`), in the layout every stage shares
 * (outline §6.2): the set chooser with the set's intent line, the list of windows, one line per readout and the Draw box.
 * Of the window on screen the SENTENCE BAR'S TABS choose the sentence — Labelled (the labeller's open-loop reading of the
 * recorded flight of the commanded aircraft) and each round's sentence for it, Start (base), Round 1 …
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
  trainingWindowFlightView,
  trainingWindowSelectionOf,
  windowShiftText,
  type TrainingWindow,
  type TrainingWindowRound,
  type TrainingWindowSample,
  type TrainingWindowSentence,
  type TrainingWindowSetEntry,
} from "../../data/trainingWindowSample";
import { checkMark, crossingText, TRAINING_OUTCOME_TAG, TRAINING_OUTCOME_TEXT } from "../../data/trainingText";
import { publishTrainingTabs, useTrainingTabs, type TrainingTab } from "../../data/trainingTabs";
import { useTrainingSetIntent } from "../../data/trainingSetIntent";
import { trainingOutcomeColour } from "../../utils/trainingWordColors";
import { TRAINING_WINDOW_ROLE_COLOR } from "../../hooks/useTrainingWindowLayer";
import ProblemBox from "./ProblemBox";
import TrainingDetails, { type TrainingDetailsSection } from "./TrainingDetails";
import { DetailsLink, DrawBox, EXPERIMENT_SECTION, OVERVIEW_SECTION, type DetailsPage } from "./PanelParts";
import { ExperimentSection, ItemList, SetChooser } from "./SetParts";
import { NotesList } from "./NotesToggle";

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

/** The window on screen and its round's end. */
function WindowEnd({ window, sentence }: { window: TrainingWindow; sentence: TrainingWindowSentence }) {
  const loss = sentence.end.loss;
  const move = window.startMove;
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
          {TRAINING_OUTCOME_TEXT[sentence.outcome]} · reward {sentence.end.reward.toFixed(2)}
        </dd>
      </div>
      {loss === null ? null : (
        <div>
          <dt>Loss of separation</dt>
          <dd>
            with {loss.other} at {(loss.timeS - window.firstStepS).toFixed(0)} s ({loss.kind}, {loss.relation}) ·{" "}
            {loss.distanceM.toFixed(0)} m of {loss.requiredM.toFixed(0)} m required · {loss.verticalM.toFixed(0)} m apart in height
            {loss.wakeKnown ? "" : " · wake category unknown"}
            {sentence.end.lossReadsFault ? " · the other aircraft read a faulty point near it" : ""}
          </dd>
        </div>
      )}
      {sentence.crossing === null ? null : <div><dt>Threshold</dt><dd>{crossingText(sentence.crossing)}</dd></div>}
      <div>
        <dt>Speed-word mask</dt>
        <dd>acted on {sentence.end.speedMaskRows} row{sentence.end.speedMaskRows === 1 ? "" : "s"}</dd>
      </div>
      <div>
        <dt>Faulty points</dt>
        <dd>{sentence.end.faultySteps === 0 ? `${checkMark(true)} no step read one` : `${sentence.end.faultySteps} steps where a recorded aircraft read one`}</dd>
      </div>
    </dl>
  );
}

/** The cursor at the window's row 0 when a window or round comes on screen (D129), so the other aircraft show from the
 *  start (the cursor resets to the flight's row 0, which is before the window's, with the flight on screen). A LEAF: it
 *  reads the Training cursor, which moves on every chart hover; once set, the cursor is the user's. ``flightKey``: the
 *  derived flight of the window and round — the cursor is set only once that flight is the one on screen (the session
 *  publishes it after this leaf's effect has run), never on the flight before. */
export function WindowCursorStart({ flightKey, row0S }: { flightKey: string; row0S: number }) {
  const { trainingSelection } = useApp();
  const { setTrainingCursorS } = useTrainingCursor();
  const onScreen = trainingSelection?.flight.flightKey === flightKey;
  // the setter belongs to the flight on screen (a new window or round, a new setter): set once for each
  useEffect(() => {
    if (onScreen) setTrainingCursorS(row0S);
  }, [onScreen, setTrainingCursorS, row0S]);
  return null;
}

/** A round's ending as the tabs, lines and tables say it: the outcome, its time, the reward, the loss of separation. */
function ending(window: TrainingWindow, sentence: TrainingWindowSentence): string {
  const seconds = sentence.flown.tS[sentence.flown.tS.length - 1] - window.firstStepS;
  const loss = sentence.end.loss;
  return `${TRAINING_OUTCOME_TAG[sentence.outcome]} at ${seconds.toFixed(0)} s · reward ${sentence.end.reward.toFixed(2)} · ` +
    (loss === null ? "no loss of separation" : `loss of separation with ${loss.other}`);
}

/** A round's tab label (outline §6.2 item 2). */
function roundTabLabel(round: TrainingWindowRound): string {
  return round === "start" ? "Start (base)" : `Round ${round}`;
}

/** The window's tabs. */
function windowTabs(window: TrainingWindow): TrainingTab[] {
  return [
    { id: LABELLED, label: "Labelled", outcome: null,
      title: "The labeller's open-loop reading of the recorded flight of the commanded aircraft; not flown" +
        (window.kind === "B" ? ". This window starts from a moved start (window B): the reading is of the flight as recorded" : "") },
    ...window.rounds.map((sentence) => ({
      id: roundTab(sentence.round), label: roundTabLabel(sentence.round), outcome: sentence.outcome,
      title: `The sentence ${roundLabel(sentence.round)}'s model said for the commanded aircraft, flown by the executor among the ` +
        `window's other aircraft — ${ending(window, sentence)} · ${sentence.goArounds} go-around${sentence.goArounds === 1 ? "" : "s"}`,
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
          {sample.windows.map((window) => (
            <tr key={window.index}>
              <th scope="row" title={window.head.flightKey}>{window.head.callsign} · {window.kind}</th>
              {rounds.map((round) => {
                const sentence = window.rounds.find((item) => item.round === round);
                return sentence === undefined ? <td key={String(round)}>–</td> : (
                  <td key={String(round)} style={{ color: trainingOutcomeColour(sentence.outcome) }} title={TRAINING_OUTCOME_TEXT[sentence.outcome]}>
                    {ending(window, sentence)} · {sentence.goArounds} go-arounds · {sentence.events.length} words
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

/** The landed of each round over the set's windows. */
function roundsSummary(sample: TrainingWindowSample): string {
  return sample.model.rounds.map((round) => {
    const said = sample.windows.flatMap((window) => window.rounds.filter((item) => item.round === round));
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

  // ── the tabs: the session gives them, the bar chooses (outline §6.2 item 2) ──
  const scope = sample === null || window === null ? null : `C/${airport}/${sample.setId}/window-${window.index}`;
  const tabList = useMemo(() => (window === null ? [] : windowTabs(window)), [window]);
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
  useEffect(() => {
    if (sample !== null) setTrainingIntervalS(chosen === LABELLED ? null : sample.model.rowIntervalS);
  }, [sample, chosen, setTrainingIntervalS]);
  useEffect(() => {
    if (sample === null || window === null || shown === null) {
      setTrainingSelection(null);
      return;
    }
    setTrainingSelection(trainingWindowSelectionOf(sample, trainingWindowFlightView(sample, window, shown)));
  }, [sample, window, shown, setTrainingSelection]);
  useEffect(() => () => setTrainingSelection(null), [setTrainingSelection]);

  const sentence = window === null || shown === null ? null : window.rounds.find((item) => item.round === shown) ?? null;

  // ── the details page ──────────────────────────────────────────────────────
  const absent = state.status === "invalid" ? `the set cannot be read: ${state.problem}` : "the set is loading";
  const roles = { recorded: 0, inserted: 0, moved: 0 };
  for (const aircraft of window?.traffic ?? []) roles[aircraft.role] += 1;
  const sections: TrainingDetailsSection[] = [
    // the first section always has a body: the set's intent and facts come from its index entry, not its sample
    entry === null ? { id: EXPERIMENT_SECTION, title: "The set and the experiment", body: <p className="training-details-lede">The index lists no set.</p> } : {
      id: EXPERIMENT_SECTION, title: "The set and the experiment", body: (
        <ExperimentSection setId={entry.id} intent={intent} facts={[
          { key: "made", name: "Made from", text: `the post-training campaign ${entry.source.campaign} (rounds ` +
            `${entry.model.rounds.map(roundLabel).join(", ")}; Δ ${entry.model.rowIntervalS} s), on ${entry.source.instructions} and ` +
            `${entry.source.executor}, commit ${entry.source.git.head.slice(0, 10)}${entry.source.git.dirty ? " (dirty tree)" : ""}` +
            `${entry.source.smoke || entry.source.campaignSmoke ? " — SMOKE: not a result" : ""}` },
          { key: "items", name: "Windows", text: `${entry.windows} of the ${entry.cohort.split} days, ${entry.cohort.perAirport} ` +
            `drawn at this airport (seed ${entry.cohort.seed}; kinds ${entry.cohort.kinds.join(", ")}) from ${entry.cohort.pool} real ` +
            `windows — ${entry.cohort.drawnFrom}` },
        ]} />
      ) },
    { id: OVERVIEW_SECTION, title: "What this view shows", body: (
      <>
        <p className="training-details-lede">
          Each window of recorded traffic with one commanded aircraft: its recorded flight and, for each round of the post-training,
          the sentence that round's model said for it, flown by the executor among the window's other aircraft and judged.
        </p>
        <h4 className="training-details-subhead">The tabs of the sentence bar</h4>
        <NotesList items={[
          { key: "labelled", name: "Labelled", text: "the labeller's open-loop reading of the commanded aircraft's recorded flight (not flown)" },
          { key: "start", name: "Start (base)", text: "the sentence the base model said, before any round" },
          { key: "round", name: "Round 1 …", text: "the sentence each round's model said; the dot is its outcome's colour" },
        ]} />
        <h4 className="training-details-subhead">Window kinds</h4>
        <NotesList items={Object.entries(TRAINING_WINDOW_KIND_TEXT).map(([kind, text]) => ({ key: kind, name: kind, text }))} />
        <h4 className="training-details-subhead">What each switch draws</h4>
        <NotesList items={[
          { key: "traffic", name: "Other aircraft", text: "the other aircraft where their records have them at the cursor's time" },
          { key: "tracks", name: "Their tracks", text: "each other aircraft's track over the window, in its role's colour" },
          { key: "loss", name: "Loss of separation", text: "the two aircraft at the loss of separation that ended the round, joined" },
          { key: "rounds", name: "Other rounds", text: "the flown paths of the window's other rounds, thin beside the one on screen" },
        ]} />
      </>
    ) },
    sample === null ? { id: ROUNDS_SECTION, title: "Rounds", body: null, absent }
      : { id: ROUNDS_SECTION, title: "Rounds", body: <RoundsTable sample={sample} /> },
    window === null || sentence === null ? { id: WINDOW_SECTION, title: "The window", body: null, absent }
      : { id: WINDOW_SECTION, title: "The window", body: <WindowEnd window={window} sentence={sentence} /> },
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
            items={sample.windows.map((item) => ({
              key: String(item.index), callsign: item.head.callsign, runway: `recorded ${item.head.runway}`,
              runwayTitle: "the runway the flight landed on in the record (a round's sentence may say another)", stratum: item.kind,
              landed: item.rounds.filter((s) => s.outcome === "landed").length, of: item.rounds.length,
              more: `${item.traffic.length} other`, title: item.head.flightKey,
              outcomes: item.rounds.map((s) => `${roundLabel(s.round)}: ${TRAINING_OUTCOME_TAG[s.outcome]}`).join("; "),
            }))} />
          {window === null || sentence === null ? null : (
            <ul className="training-details-links" aria-label="Readouts">
              <DetailsLink name="This round" onOpen={details.open(ROUNDS_SECTION)}
                summary={chosen === LABELLED ? "the labelled sentence (not flown)" : `${roundLabel(sentence.round)}: ${ending(window, sentence)}`} />
              <DetailsLink name="Rounds" onOpen={details.open(ROUNDS_SECTION)} summary={roundsSummary(sample)} />
              <DetailsLink name="The window" onOpen={details.open(WINDOW_SECTION)} summary={`${TRAINING_WINDOW_KIND_TEXT[window.kind]} · ` +
                (window.traffic.length === 0 ? "no other aircraft"
                  : (["recorded", "inserted", "moved"] as const).filter((role) => roles[role] > 0).map((role) => `${roles[role]} ${role}`).join(", "))} />
            </ul>
          )}
          {window === null || shown === null ? null : (
            <WindowCursorStart flightKey={trainingWindowFlightView(sample, window, shown).flightKey} row0S={window.row0S} />
          )}
          <DrawBox legend="Draw (window)">
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
