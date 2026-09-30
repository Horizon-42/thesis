/**
 * TrainingFlightSession.tsx
 * -------------------------
 * The Training panel over a READ-BACK SET (`vocabulary-readback`): its flights one at a time. It publishes the selected
 * flight through `trainingSelection` — the sentence bar, the read-back window and the 3D layer draw it — and offers the
 * OVERLAYS published over the set (`useTrainingOverlays`): the executor's replay and the prior's predictions, each behind
 * its own switch (a set with none says so and folds away the command that writes one). THE MODELS' OWN SENTENCES are
 * downloaded here and CHOSEN IN THE SENTENCE BAR (its tabs, `trainingSource`); while one is read, each flight in the list
 * shows its samples, filled where the flight landed; one that cannot be read is named here, with Retry.
 *
 * THE DOCK ENDS ABOVE THE SENTENCE BAR (the bar measures itself, `--training-bar-height`): the flight list takes the
 * height left over — never less than a few rows, the dock scrolling below that — so the switches under it are never
 * covered. Its readouts are one line each; their tables are on the details page (`TrainingDetails`), whose sections this
 * session supplies.
 */

import { useEffect, useMemo, useState } from "react";
import { useApp } from "../../context/AppContext";
import useTrainingOverlays, { type GenerationItem, type OverlayKindState, type OverlaysManifestState } from "../../hooks/useTrainingOverlays";
import TrainingVocabularyNotes from "../TrainingVocabularyNotes";
import ProblemBox from "./ProblemBox";
import TrainingDetails, { type TrainingDetailsSection } from "./TrainingDetails";
import { NotesList } from "./NotesToggle";
import { DetailsLink, LAYER_SWITCHES, LayerSwitches, manifestAbsence, noneReadable, type DetailsPage } from "./PanelParts";
import { TRAINING_EXECUTOR_COLOR, TRAINING_REPLAY_COLOR, trainingModelColour } from "../../utils/trainingWordColors";
import {
  generationUnflownReason,
  replayVerdict,
  trainingOverlaysPath,
  type TrainingExecutorFlight,
  type TrainingGenerationFlight,
} from "../../data/trainingOverlays";
import { TRAINING_OUTCOME_TAG } from "../../data/trainingText";
import {
  executorGateSummary,
  generationSummary,
  priorReadoutSummary,
  TrainingExecutorGate,
  TrainingGenerationReadout,
  TrainingPriorReadout,
} from "../TrainingResults";
import {
  TRAINING_COLUMNS,
  trainingSelectionOf,
  trainingVerdicts,
  type TrainingSample,
  type TrainingSetEntry,
} from "../../data/trainingSample";

/** The overlay switches' names, as the Draw box and the details page's reasons say them. */
const EXECUTOR_SWITCH = "Executor replay";
const PRIOR_SWITCH = "Prior predictions";

/** The details page's readout sections, by title. */
const MODELS_TITLE = "The models' own sentences";
const EXECUTOR_TITLE = "The executor's replay gate";
const PRIOR_TITLE = "The prior's readout";

/** The commands that write the overlays, folded under a switch whose set has none. */
const TRAINING_OVERLAY_COMMAND = {
  "executor-replay": "python run_ts.py executor_training_export --executor <spec dir> --replay <spec dir>/replay-val " +
    "--instructions <artefact> --airports-root aeroviz-4d/public/data/airports --set ",
  "prior-prediction": "python run_ts.py prior_training_export --prior <prior dir> --instructions <artefact> " +
    "--airports-root aeroviz-4d/public/data/airports --set ",
} as const;

/** One overlay kind's switch: on / off, which overlay (when several), and what failed. */
function OverlaySwitch<T>({ state, colour, text, setId, airport }: {
  state: OverlayKindState<T>; colour: string | undefined; text: string; setId: string; airport: string;
}) {
  const { entry, entries, shown, setShown, load, choose } = state;
  return (
    <div className="training-overlay-switch">
      <label style={entry ? { color: colour } : undefined} className={entry ? undefined : "training-slot"}
        title={entry?.title}>
        <input type="checkbox" checked={entry !== null && shown} disabled={entry === null}
          onChange={(event) => setShown(event.target.checked)} />
        {text}
        {entry !== null && shown && load.status === "loading" ? <span className="training-overlay-note" role="status"> loading …</span> : null}
      </label>
      {entry === null ? (
        <details className="training-overlay-note">
          <summary>none published</summary>
          <code>{TRAINING_OVERLAY_COMMAND[state.kind]}{setId} --airport {airport}</code>
        </details>
      ) : null}
      {entry !== null && entries.length > 1 ? (
        <select aria-label={`which ${state.kind} overlay`} value={entry.id} onChange={(event) => choose(event.target.value)}>
          {entries.map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}
        </select>
      ) : null}
      {entry !== null && load.status === "invalid" ? <ProblemBox title={`Overlay ${entry.id} cannot be read.`} detail={load.problem} /> : null}
    </div>
  );
}

/** How the flight list colours a flight's replay (`replayVerdict`), for its tooltip and the details page. */
const REPLAY_COLOURS_TEXT = "teal: landed, every judged word inside its envelope; amber: landed, but words outside (or " +
  "its track refused by the labeller's gate); red: did not land";

/** What the flight list says of a flight's replay: its outcome, and its words inside of those judged — in the colour of
 *  its verdict: landed with every word inside, landed with words outside, not landed. */
function ExecutorTag({ flight }: { flight: TrainingExecutorFlight }) {
  if (!flight.flown) {
    return <span className="training-flight-executor" title={`the replay does not fly it: ${flight.group}`}>not flown</span>;
  }
  const { wordsInside, wordsJudged } = flight.counts;
  return (
    <span className="training-flight-executor" style={{ color: TRAINING_REPLAY_COLOR[replayVerdict(flight).kind] }}
      title={`the executor on ${flight.group}: ${TRAINING_OUTCOME_TAG[flight.outcome]}; ${wordsInside} of ${wordsJudged} words ` +
        `inside their envelopes, as the replay gate counts them${flight.flewTheSentence ? " — it flew the sentence" : ""} ` +
        `(${REPLAY_COLOURS_TEXT})`}>
      {TRAINING_OUTCOME_TAG[flight.outcome]} · {wordsInside}/{wordsJudged}
    </span>
  );
}

/** A flight's samples of the chosen model, one mark each: filled where the flight landed, hollow where it did not. */
function SampleMarks({ flight, colour }: { flight: TrainingGenerationFlight; colour: string }) {
  if (!flight.flown) {
    return <span className="training-flight-samples" title={`the model's sentences do not fly it: ${generationUnflownReason(flight)}`}>—</span>;
  }
  const landed = flight.samples.filter((item) => item.outcome === "landed").length;
  return (
    <span className="training-flight-samples" style={{ color: colour }}
      title={`${landed} of ${flight.samples.length} of the model's sentences landed: ` +
        flight.samples.map((item) => `#${item.sample + 1} ${TRAINING_OUTCOME_TAG[item.outcome]}`).join(", ")}>
      {flight.samples.map((item) => (item.outcome === "landed" ? "●" : "○")).join("")}
    </span>
  );
}

/** Why an overlay's readout has nothing to show: the details page's disabled section says it. */
function overlayAbsence<T>(manifest: OverlaysManifestState, state: OverlayKindState<T>, switchName: string): string {
  const unread = manifestAbsence(manifest);
  if (unread !== null) return unread;
  if (state.entry === null) return noneReadable(manifest);
  if (!state.shown) return `switch on ${switchName} under Draw`;
  return state.load.status === "invalid" ? "cannot be read" : "loading …";
}

/** Why the models' sentences have nothing to show. */
function generationAbsence(manifest: OverlaysManifestState, items: GenerationItem[]): string {
  const unread = manifestAbsence(manifest);
  if (unread !== null) return unread;
  if (items.length === 0) return noneReadable(manifest);
  return items.every(({ load }) => load.status === "invalid") ? "cannot be read" : "loading …";
}

/** ``sample``: the read-back set open — null while the next one loads (the session is kept: its switches and choices per
 *  set outlive a switch of sets, and it publishes no flight meanwhile). */
export default function TrainingFlightSession({ airport, sample, entry, details }: {
  airport: string; sample: TrainingSample | null; entry: TrainingSetEntry | null; details: DetailsPage;
}) {
  const { setTrainingSelection, trainingSource } = useApp();
  const [flightKey, setFlightKey] = useState<string | null>(null);
  const overlays = useTrainingOverlays(airport, sample, flightKey);
  const executorOverlay = overlays.executor.shown && overlays.executor.load.status === "ready" ? overlays.executor.load.overlay : null;
  const priorOverlay = overlays.prior.shown && overlays.prior.load.status === "ready" ? overlays.prior.load.overlay : null;
  const executorFlights = useMemo(
    () => new Map((executorOverlay?.flights ?? []).map((flight) => [flight.flightKey, flight])),
    [executorOverlay],
  );
  // the model whose sentences are read, when it is loaded for this set: its samples beside each flight
  const readModel = useMemo(() => {
    const item = overlays.generations.find(({ entry: listed }) => listed.id === trainingSource?.overlayId);
    return item?.load.status === "ready" ? item.load.overlay : null;
  }, [overlays.generations, trainingSource]);
  const readFlights = useMemo(
    () => new Map((readModel?.flights ?? []).map((flight) => [flight.flightKey, flight])),
    [readModel],
  );

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

  // ── publish what the other views draw ─────────────────────────────────────
  useEffect(() => {
    const flight = sample?.flights.find((item) => item.flightKey === flightKey) ?? null;
    setTrainingSelection(flight && sample ? trainingSelectionOf(sample, flight) : null);
  }, [sample, flightKey, setTrainingSelection]);
  useEffect(() => () => setTrainingSelection(null), [setTrainingSelection]);
  if (sample === null) return null;

  // ── the details page ──────────────────────────────────────────────────────
  const generations = overlays.generations.flatMap(({ load }) => (load.status === "ready" ? [load.overlay] : []));
  const sections: TrainingDetailsSection[] = [
    { id: "overview", title: "What this view shows", body: (
      <>
        <p className="training-details-lede">
          Each arrival read as the instructions a controller could have given — {TRAINING_COLUMNS.length} columns per{" "}
          {sample.vocabulary.stepS} s step — with what every word allows: a heading word's band over the
          rows it is judged on, the capture turn and corridor, the altitude tube, the speed band.
        </p>
        <h4 className="training-details-subhead">What each switch draws</h4>
        <NotesList items={[
          ...LAYER_SWITCHES.map(({ layer, colour, text, title }) => ({ key: layer, text: title, name: (
            <><span className="training-model-swatch" style={{ background: colour }} />{text}</>) })),
          { key: "executor", text: `The executor's replay of the truth sentence. In the flight list, its outcome and its ` +
            `words inside of those judged — ${REPLAY_COLOURS_TEXT}; in the sentence bar, a dot at each word's band and, ` +
            "when it did not land or flew words outside, a short note in the header.", name: (
            <>{Object.values(TRAINING_REPLAY_COLOR).map((colour) => (
              <span key={colour} className="training-model-swatch" style={{ background: colour }} />))}{EXECUTOR_SWITCH}</>) },
        ]} />
        <h4 className="training-details-subhead">The overlays published over this set</h4>
        {overlays.executor.entries.length + overlays.prior.entries.length + overlays.generations.length === 0 ? (
          <p className="training-details-lede">None.</p>
        ) : (
          <NotesList items={[...overlays.executor.entries, ...overlays.prior.entries, ...overlays.generations.map((item) => item.entry)]
            .map((item) => ({ key: item.id, name: <code>{item.id}</code>, text: item.title }))} />
        )}
      </>
    ) },
    { id: "vocabulary", title: "Vocabulary", body: <TrainingVocabularyNotes sample={sample} /> },
    generations.length > 0
      ? { id: "models", title: MODELS_TITLE, body: <TrainingGenerationReadout flights={sample.flights} overlays={generations} /> }
      : { id: "models", title: MODELS_TITLE, body: null, absent: generationAbsence(overlays.manifest, overlays.generations) },
    executorOverlay
      ? { id: "executor", title: EXECUTOR_TITLE, body: <TrainingExecutorGate overlay={executorOverlay} /> }
      : { id: "executor", title: EXECUTOR_TITLE, body: null, absent: overlayAbsence(overlays.manifest, overlays.executor, EXECUTOR_SWITCH) },
    priorOverlay
      ? { id: "prior", title: PRIOR_TITLE, body: <TrainingPriorReadout overlay={priorOverlay} stepS={sample.vocabulary.stepS} /> }
      : { id: "prior", title: PRIOR_TITLE, body: null, absent: overlayAbsence(overlays.manifest, overlays.prior, PRIOR_SWITCH) },
  ];

  return (
    <>
      {details.shown !== null ? (
        <TrainingDetails context={[airport, sample.setId, `${sample.flights.length} flights`].join(" · ")}
          sections={sections} sectionId={details.shown.section} onSection={details.show} onClose={details.close}
          opener={details.shown.opener} />
      ) : null}

      <p className="training-note" title={`${entry?.title ?? sample.setId} — ${sample.cohort.drawnFrom}`}>
        {sample.flights.length} flights · {sample.cohort.split}, {sample.cohort.perStratum} per stratum
        {entry && entry.flights !== sample.flights.length
          ? ` · the manifest says ${entry.flights}: this set's two files are from different exports` : ""}
      </p>
      <ul className="training-flight-list">
        {sample.flights.map((flight) => {
          const replay = executorFlights.get(flight.flightKey);
          const read = readFlights.get(flight.flightKey);
          return (
            <li key={flight.flightKey}>
              <button type="button" className={flight.flightKey === flightKey ? "active" : undefined}
                onClick={() => setFlightKey(flight.flightKey)}>
                <span className="training-flight-callsign">{flight.callsign}</span>
                <span className="training-flight-runway">{flight.runway}</span>
                <span className="training-flight-stratum">{flight.stratum}</span>
                <span className="training-flight-events">{trainingVerdicts(flight).instructionsAfterStep0} words</span>
                {read !== undefined && readModel !== null
                  ? <SampleMarks flight={read} colour={trainingModelColour(readModel.model)} /> : null}
                {replay !== undefined ? <ExecutorTag flight={replay} /> : null}
              </button>
            </li>
          );
        })}
      </ul>

      <fieldset className="training-layers">
        <legend>Draw</legend>
        <LayerSwitches />
        {/* THE OVERLAYS over this set: another model's output on its own flights. */}
        <OverlaySwitch state={overlays.executor} colour={TRAINING_EXECUTOR_COLOR} setId={sample.setId} airport={airport}
          text={EXECUTOR_SWITCH} />
        <OverlaySwitch state={overlays.prior} colour={undefined} setId={sample.setId} airport={airport}
          text={PRIOR_SWITCH} />
      </fieldset>

      {overlays.manifest.status === "invalid" ? (
        <ProblemBox title={`${trainingOverlaysPath(airport)} cannot be read.`} detail={overlays.manifest.problem} />
      ) : null}
      {overlays.manifest.status === "ready" ? overlays.manifest.overlays.rejected.map((item) => (
        <ProblemBox key={`overlay-${item.id}`} title={`Overlay ${item.id} was rejected.`} detail={item.problem} />
      )) : null}
      {/* a model's sentences that cannot be read: named, and asked for again (the sentence bar lists those read) */}
      {overlays.generations.flatMap(({ entry: listed, load, retry }) => (load.status !== "invalid" ? [] : [
        <div key={`generation-${listed.id}`}>
          <ProblemBox title={`Overlay ${listed.id} cannot be read.`} detail={load.problem} />
          <button type="button" className="training-sentence-readback-button" onClick={retry}>Retry</button>
        </div>,
      ]))}
      {/* THE READOUTS: one line each here, their tables on the details page */}
      {generations.length > 0 || executorOverlay || priorOverlay ? (
        <ul className="training-details-links" aria-label="Readouts">
          {generations.length > 0 ? (
            <DetailsLink name="Models' sentences" summary={generationSummary(generations, sample.flights)}
              onOpen={details.open("models")} />
          ) : null}
          {executorOverlay ? (
            <DetailsLink name="Replay gate" summary={executorGateSummary(executorOverlay)} onOpen={details.open("executor")} />
          ) : null}
          {priorOverlay ? (
            <DetailsLink name="Prior readout" summary={priorReadoutSummary(priorOverlay)} onOpen={details.open("prior")} />
          ) : null}
        </ul>
      ) : null}
    </>
  );
}
