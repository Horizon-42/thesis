/**
 * TrainingWindowSession.tsx
 * -------------------------
 * The Training panel over a window set of stage C (`data/trainingWindowSample.ts`): its windows one at a time, and of the
 * window on screen the rounds of the post-training side by side — each round's sentence for the commanded aircraft, one on
 * screen at a time (the sentence bar, the read-back window and the 3D layers draw it, as for stage A: a round's sentence is
 * published as a closed-loop sentence at the set's Δ, `trainingWindowFlightView`).
 *
 * Under the table: the WINDOW — its kind (real; A, an aircraft inserted; D, the aircraft ahead moved; B, the commanded
 * aircraft's start moved), its moves, its other aircraft — and the round's END: the reward, the loss of separation with its
 * other aircraft and the minimum it broke, the rows the speed-word mask acted (D101), the steps where a recorded aircraft
 * read a faulty point (D114). The other aircraft, the loss and the other rounds are drawn in 3D (`useTrainingWindowLayer`;
 * the Draw switches here).
 *
 * A click on a word of the sentence on screen flies its segment live (`useTrainingWindowAutopilot`).
 */

import { useEffect, useState } from "react";
import { useApp, useTrainingCursor } from "../../context/AppContext";
import useTrainingWindowAutopilot from "../../hooks/useTrainingWindowAutopilot";
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
import { trainingOutcomeColour } from "../../utils/trainingWordColors";
import { TRAINING_WINDOW_ROLE_COLOR } from "../../hooks/useTrainingWindowLayer";
import ProblemBox from "./ProblemBox";

type SetState =
  | { status: "loading" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; sample: TrainingWindowSample };

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

/** The window's rounds side by side. */
function RoundTable({ window, round, onSelect }: {
  window: TrainingWindow; round: TrainingWindowRound; onSelect: (round: TrainingWindowRound) => void;
}) {
  const endS = (sentence: TrainingWindowSentence) => sentence.flown.tS[sentence.flown.tS.length - 1] - window.firstStepS;
  return (
    <table className="training-flown-table training-prior-table" aria-label="The window's rounds">
      <thead>
        <tr><th scope="col">round</th><th scope="col">ended (from the first predicted step)</th><th scope="col">reward</th><th scope="col">go-arounds</th><th scope="col">words</th></tr>
      </thead>
      <tbody>
        {window.rounds.map((sentence) => {
          const active = sentence.round === round;
          return (
            <tr key={String(sentence.round)} className={active ? "training-prior-sentence active" : "training-prior-sentence"}>
              <th scope="row">
                <button type="button" aria-pressed={active} title={`the sentence ${roundLabel(sentence.round)}'s model said for the commanded aircraft`}
                  onClick={() => onSelect(sentence.round)}>{roundLabel(sentence.round)}</button>
              </th>
              <td style={{ color: trainingOutcomeColour(sentence.outcome) }} title={TRAINING_OUTCOME_TEXT[sentence.outcome]}>
                {TRAINING_OUTCOME_TAG[sentence.outcome]} · {endS(sentence).toFixed(0)} s
              </td>
              <td>{sentence.end.reward.toFixed(2)}</td>
              <td>{sentence.goArounds}</td>
              <td>{sentence.events.length}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
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

/** One line of the window list: the commanded flight, the kind, and each round's end. */
function windowSummary(window: TrainingWindow): string {
  return window.rounds.map((sentence) => `${roundLabel(sentence.round)}: ${TRAINING_OUTCOME_TAG[sentence.outcome]}`).join("; ");
}

export default function TrainingWindowSession({ airport, sets }: { airport: string; sets: TrainingWindowSetEntry[] }) {
  const { setTrainingSelection, setTrainingIntervalS } = useApp();
  const layers = useTrainingWindowLayers();
  useTrainingWindowAutopilot();
  const [setId, setSetId] = useState<string | null>(sets[0]?.id ?? null);
  const [state, setState] = useState<SetState>({ status: "loading" });
  /** The window on screen: its set and its place there (a choice made in another set never selects in this one). */
  const [placed, setPlaced] = useState<{ setId: string; index: number } | null>(null);
  const [round, setRound] = useState<TrainingWindowRound | null>(null);
  const entry = sets.find((item) => item.id === setId) ?? null;

  useEffect(() => {
    if (entry === null) return;
    let live = true;
    setState({ status: "loading" });
    fetchTrainingWindowSample(airport, entry.file, entry.id)
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
  const place = sample !== null && placed !== null && placed.setId === sample.setId ? placed.index : null;
  const window = sample === null || place === null ? null : sample.windows[place] ?? null;
  // a set that has just opened may still hold the last set's choice for one render: the round shown is one the set has
  const shown: TrainingWindowRound | null = sample === null ? null
    : round !== null && sample.model.rounds.includes(round) ? round : sample.model.rounds[sample.model.rounds.length - 1];

  // a set opens on its first window and its last round, read at the set's Δ
  useEffect(() => {
    if (sample === null) return;
    setPlaced(sample.windows.length > 0 ? { setId: sample.setId, index: 0 } : null);
    setRound(sample.model.rounds[sample.model.rounds.length - 1]);
    setTrainingIntervalS(sample.model.rowIntervalS);
  }, [sample, setTrainingIntervalS]);

  // publish what the sentence bar, the read-back window and the 3D layers draw
  useEffect(() => {
    if (sample === null || window === null || shown === null) {
      setTrainingSelection(null);
      return;
    }
    setTrainingSelection(trainingWindowSelectionOf(sample, trainingWindowFlightView(sample, window, shown)));
  }, [sample, window, shown, setTrainingSelection]);
  useEffect(() => () => setTrainingSelection(null), [setTrainingSelection]);

  const sentence = window === null || shown === null ? null : window.rounds.find((item) => item.round === shown) ?? null;

  return (
    <>
      {sets.length > 1 ? (
        <label className="training-field" title={entry?.title}>
          <span>Set</span>
          <select value={setId ?? ""} onChange={(event) => setSetId(event.target.value)}>
            {sets.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.windows} windows</option>)}
          </select>
        </label>
      ) : null}
      {entry === null ? null : (
        <p className="training-note" title={entry.title}>
          {entry.title}{entry.source.smoke || entry.source.campaignSmoke ? " · SMOKE (not a result)" : ""}
        </p>
      )}
      {state.status === "loading" ? <p className="training-note" role="status">Loading {entry?.file} …</p> : null}
      {state.status === "invalid" ? <ProblemBox title={`Set ${setId} cannot be read.`} detail={state.problem} /> : null}

      {sample === null ? null : (
        <>
          <ul className="training-flight-list" aria-label="The set's windows">
            {sample.windows.map((item) => (
              <li key={item.index}>
                <button type="button" className={item.index === place ? "active" : undefined} title={`${item.head.flightKey} · ${windowSummary(item)}`}
                  onClick={() => setPlaced({ setId: sample.setId, index: item.index })}>
                  <span className="training-flight-callsign">{item.head.callsign}</span>
                  <span className="training-flight-runway" title="the runway the flight landed on in the record (a round's sentence may say another)">
                    recorded {item.head.runway}
                  </span>
                  <span className="training-flight-stratum">{item.kind}</span>
                  <span className="training-flight-executor">
                    {item.rounds.filter((s) => s.outcome === "landed").length}/{item.rounds.length} landed · {item.traffic.length} other
                  </span>
                </button>
              </li>
            ))}
          </ul>
          {window === null || shown === null ? null : (
            <RoundTable window={window} round={shown} onSelect={setRound} />
          )}
          {window === null || sentence === null ? null : <WindowEnd window={window} sentence={sentence} />}
          {window === null || shown === null ? null : (
            <WindowCursorStart flightKey={trainingWindowFlightView(sample, window, shown).flightKey} row0S={window.row0S} />
          )}
          <fieldset className="training-layers">
            <legend>Draw (window)</legend>
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
          </fieldset>
        </>
      )}
    </>
  );
}
