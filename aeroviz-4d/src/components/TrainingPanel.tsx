/**
 * TrainingPanel.tsx
 * -----------------
 * The Training task's left-dock panel: which exported set, which flight, which envelopes are drawn — for the
 * instruction vocabulary (`TRAINING_READING_RULE`). Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`.
 *
 * It owns the only fetch and publishes the selected flight through `trainingSelection`, which the sentence bar, the
 * read-back window and the 3D layer draw. The dock keeps it mounted from its first visit on, only ``hidden`` in the
 * other tasks (`WorkbenchLeftDock`): its session — set, sample, flight, overlays, live answer — outlives a task switch.
 *
 * The states all name what failed:
 *   ① no `index.json`                → the path, the command, the dev-server restart (AV5)
 *   ② a listed set this reader refuses → WHY, by name, from the manifest alone (no download):
 *                                       a superseded vocabulary, another spec, a prior set
 *   ③ a readable set that fails to parse → THAT set alone, with the field
 *   ④ an entry that is not a set entry  → greyed out on its own; the others still load (AV6)
 *
 * KEPT SHORT, TOP TO BOTTOM: the set, the flight list, the live executor, the Draw switches (short labels, the full
 * reading in each tooltip), then the readouts behind the overlays. What is read once — what the module is, the
 * vocabulary's numbers, the shas — is behind the header's ⓘ (`TrainingVocabularyNotes`).
 *
 * Over the open set it offers the OVERLAYS published for it (`useTrainingOverlays`): the executor's replay and the
 * prior's predictions, each behind its own switch; a set with none says so and folds away the command that writes one.
 * And THE MODELS' OWN SENTENCES (`prior-generation`): each model published for the set is a line of "Model sentences" —
 * its colour, its name, how many of its samples landed on this set — which chooses it as the sentence read
 * (`trainingSource`, as the sentence bar's tabs do); while one is chosen, each flight in the list shows its samples,
 * filled where the flight landed.
 *
 * THE DOCK ENDS ABOVE THE SENTENCE BAR (the bar measures itself, `--training-bar-height`): the flight list takes the
 * height left over — never less than a few rows, the dock scrolling below that — so the switches under it are never
 * covered.
 * And THE EXECUTOR, LIVE (`useTrainingAutopilot`): a word picked — the sentence bar's Fly button, or a band clicked
 * while its switch is on — is flown by the backend now, never read from an overlay (`TrainingAutopilotCard`).
 */

import { useEffect, useMemo, useState } from "react";
import { useApp, type TrainingLayers } from "../context/AppContext";
import useTrainingOverlays, { type OverlayKindState } from "../hooks/useTrainingOverlays";
import useTrainingAutopilot from "../hooks/useTrainingAutopilot";
import TrainingAutopilotCard from "./TrainingAutopilotCard";
import TrainingVocabularyNotes from "./TrainingVocabularyNotes";
import ProblemBox from "./training/ProblemBox";
import { NotesList } from "./training/NotesToggle";
import { isMissingJsonAsset } from "../utils/fetchJson";
import {
  TRAINING_AUTOPILOT_COLOR,
  TRAINING_CANDIDATE_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TUBE_COLOR,
  trainingModelColour,
} from "../utils/trainingWordColors";
import {
  generationLanded,
  trainingOverlaysPath,
  type TrainingExecutorFlight,
  type TrainingGenerationFlight,
} from "../data/trainingOverlays";
import { TRAINING_OUTCOME_TAG } from "../data/trainingText";
import { TrainingExecutorGate, TrainingGenerationReadout, TrainingPriorReadout } from "./TrainingResults";
import type { GenerationItem } from "../hooks/useTrainingOverlays";
import {
  fetchTrainingIndex,
  fetchTrainingSample,
  trainingIndexPath,
  trainingSelectionOf,
  trainingSetRefusal,
  trainingVerdicts,
  type TrainingIndex,
  type TrainingSample,
  type TrainingFlight,
  type TrainingSetEntry,
} from "../data/trainingSample";

/** The Draw switches, in drawing order: a short name in its own colour, what it shows in its tooltip. */
const LAYER_SWITCHES: Array<{ layer: keyof TrainingLayers; colour: string; text: string; title: string }> = [
  { layer: "headingBands", colour: TRAINING_HEADING_BAND_COLOR, text: "Heading bands",
    title: "each heading word's band over the rows it is judged on, and those rows outside it in red" },
  { layer: "corridor", colour: TRAINING_CORRIDOR_COLOR, text: "Capture turn + corridor",
    title: "the capture turn from the clearance onto the course, the capture corridor, and the course band after it" },
  { layer: "vertical", colour: TRAINING_TUBE_COLOR, text: "Altitude tubes + speed bands",
    title: "each altitude word's tube, and on the speed chart each speed word's transition and band" },
  { layer: "candidates", colour: TRAINING_CANDIDATE_COLOR, text: "Other candidate runways",
    title: "every runway the runway pointer can point at; the designated one is always drawn" },
];

/** What the live executor's switch does. */
const FLY_ON_CLICK = "On: clicking a word's band in the sentence bar flies its segment at once. Off: only the bar's Fly button does.";

type IndexState =
  | { status: "loading" }
  | { status: "absent" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; index: TrainingIndex };

type SampleState =
  | { status: "idle" }
  | { status: "refused"; problem: string }
  | { status: "loading" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; sample: TrainingSample };

/** The command that writes the export, as the empty state shows it. */
const TRAINING_EXPORT_COMMAND =
  "python run_ts.py instruction_training_export --dir 4dTrajectory/outputs/POOLED/instruction_language/<an instruction-v3 artefact> " +
  "--airports-root aeroviz-4d/public/data/airports --airport ";

/** The commands that write the overlays, folded under a switch whose set has none. */
const TRAINING_OVERLAY_COMMAND = {
  "executor-replay": "python run_ts.py executor_training_export --executor <spec dir> --replay <spec dir>/replay-val " +
    "--instructions <artefact> --airports-root aeroviz-4d/public/data/airports --set ",
  "prior-prediction": "python run_ts.py prior_training_export --prior <prior dir> --instructions <artefact> " +
    "--airports-root aeroviz-4d/public/data/airports --set ",
  "prior-generation": "python run_ts.py prior_generation_training_export --prior <prior dir> --label <its name> " +
    "--instructions <artefact> --executor <executor spec dir> --readout <its val free generation> " +
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

/** What the flight list says of a flight's replay: its outcome, and its words inside of those judged. */
function ExecutorTag({ flight }: { flight: TrainingExecutorFlight }) {
  if (!flight.flown) {
    return <span className="training-flight-executor" title={`the replay does not fly it: ${flight.group}`}>not flown</span>;
  }
  const { wordsInside, wordsJudged } = flight.counts;
  const ok = flight.outcome === "landed" && wordsInside === wordsJudged;
  return (
    <span className="training-flight-executor" style={{ color: ok ? TRAINING_EXECUTOR_COLOR : TRAINING_OUTSIDE_COLOR }}
      title={`the executor on ${flight.group}: ${TRAINING_OUTCOME_TAG[flight.outcome]}; ${wordsInside} of ${wordsJudged} words ` +
        `inside their envelopes, as the replay gate counts them${flight.flewTheSentence ? " — it flew the sentence" : ""}`}>
      {TRAINING_OUTCOME_TAG[flight.outcome]} · {wordsInside}/{wordsJudged}
    </span>
  );
}

/** A flight's samples of the chosen model, one mark each: filled where the flight landed, hollow where it did not. */
function SampleMarks({ flight, colour }: { flight: TrainingGenerationFlight; colour: string }) {
  if (!flight.flown) {
    return <span className="training-flight-samples" title={`the model's sentences do not fly it: ${flight.group}`}>—</span>;
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

/** THE MODELS' OWN SENTENCES published for the set: each a line that chooses it as the sentence read (from its first
 *  sample); the truth first — and checked when the chosen model is not one of this set's. */
function ModelSentences({ items, setId, airport, flights }: {
  items: GenerationItem[]; setId: string; airport: string; flights: TrainingFlight[];
}) {
  const { trainingSource, setTrainingSource } = useApp();
  if (items.length === 0) {
    return (
      <div className="training-models">
        <span className="training-slot">Model sentences</span>
        <details className="training-overlay-note">
          <summary>none published</summary>
          <code>{TRAINING_OVERLAY_COMMAND["prior-generation"]}{setId} --airport {airport}</code>
        </details>
      </div>
    );
  }
  const chosen = items.some(({ entry }) => entry.id === trainingSource?.overlayId) ? trainingSource!.overlayId : null;
  return (
    <div className="training-models" role="radiogroup" aria-label="Which sentence is read">
      <span className="training-models-title">Sentences read</span>
      <label className="training-model">
        <input type="radio" name="training-source" checked={chosen === null} onChange={() => setTrainingSource(null)} />
        <span className="training-model-swatch training-model-swatch-truth" />
        truth (labelled)
      </label>
      {items.map(({ entry, load, retry }) => {
        const overlay = load.status === "ready" ? load.overlay : null;
        const count = overlay === null ? null : generationLanded(overlay, flights).all;
        return (
          <div key={entry.id}>
            <label className="training-model" title={entry.title}
              style={overlay === null ? undefined : { color: trainingModelColour(overlay.model) }}>
              <input type="radio" name="training-source" checked={chosen === entry.id} disabled={overlay === null}
                onChange={() => setTrainingSource({ overlayId: entry.id, sample: 0 })} />
              <span className="training-model-swatch"
                style={overlay === null ? undefined : { background: trainingModelColour(overlay.model) }} />
              {overlay?.model.label ?? entry.id}
              {count !== null ? (
                <span className="training-model-count" title={`of the ${count.flights} sentences it said over the set's flights ` +
                  "(each flown by the executor from its first predicted step), those that landed"}>
                  {" "}{Math.round(count.landed * count.flights)}/{count.flights} landed
                </span>
              ) : null}
              {load.status === "loading" || load.status === "idle" ? <span className="training-overlay-note" role="status"> loading …</span> : null}
            </label>
            {load.status === "invalid" ? (
              <>
                <ProblemBox title={`Overlay ${entry.id} cannot be read.`} detail={load.problem} />
                <button type="button" className="training-sentence-readback-button" onClick={retry}>Retry</button>
              </>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function EmptyState({ airport }: { airport: string }) {
  return (
    <div className="training-empty" role="status">
      <p className="training-empty-title">No Training export for {airport} yet.</p>
      <p className="training-empty-label">This panel reads</p>
      <code className="training-empty-path">{trainingIndexPath(airport)}</code>
      <p className="training-empty-label">Written by</p>
      <code className="training-empty-path">{TRAINING_EXPORT_COMMAND}{airport}</code>
      <p className="training-empty-note">
        After the first export, restart the dev server — vite does not watch{" "}
        <code>public/data</code>, so a directory created after boot is served as the SPA
        fallback instead of JSON. Kill the <code>vite</code> node process, not the{" "}
        <code>npm run dev</code> wrapper.
      </p>
    </div>
  );
}

export default function TrainingPanel({ hidden }: { hidden: boolean }) {
  const {
    activeAirportCode, setTrainingSelection, trainingLayers, setTrainingLayer, trainingAutopilotAuto, setTrainingAutopilotAuto,
    trainingSource,
  } = useApp();
  const airport = activeAirportCode || "—";

  const [indexState, setIndexState] = useState<IndexState>({ status: "loading" });
  const [setId, setSetId] = useState<string | null>(null);
  const [sampleState, setSampleState] = useState<SampleState>({ status: "idle" });
  const [flightKey, setFlightKey] = useState<string | null>(null);
  const [aboutOpen, setAboutOpen] = useState<boolean>(false);

  // ── the manifest ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!activeAirportCode) return;
    let live = true;
    setIndexState({ status: "loading" });
    setSetId(null);
    fetchTrainingIndex(activeAirportCode)
      .then((parsed) => {
        if (!live) return;
        if (!parsed.ok) {
          setIndexState({ status: "invalid", problem: parsed.problem });
          return;
        }
        setIndexState({ status: "ready", index: parsed.value });
        // Open on a set this reader CAN read: the manifest states each set's reading rule and spec, so which ones are
        // current is known before anything is downloaded.
        const readable = parsed.value.sets.find((item) => trainingSetRefusal(item) === null);
        setSetId((readable ?? parsed.value.sets[0])?.id ?? null);
      })
      .catch((error: unknown) => {
        if (!live) return;
        // "Not exported yet" and "exported but broken" are different answers; `isMissingJsonAsset` also catches vite's
        // SPA fallback HTML, which is what a 404 under `public/data` looks like.
        if (isMissingJsonAsset(error)) setIndexState({ status: "absent" });
        else setIndexState({ status: "invalid", problem: error instanceof Error ? error.message : String(error) });
      });
    return () => {
      live = false;
    };
  }, [activeAirportCode]);

  const entry: TrainingSetEntry | null = useMemo(() => {
    if (indexState.status !== "ready") return null;
    return indexState.index.sets.find((item) => item.id === setId) ?? null;
  }, [indexState, setId]);

  // ── the chosen set ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!activeAirportCode || !entry) {
      setSampleState({ status: "idle" });
      return;
    }
    const refusal = trainingSetRefusal(entry);
    if (refusal !== null) {
      setSampleState({ status: "refused", problem: refusal });
      return;
    }
    let live = true;
    setSampleState({ status: "loading" });
    fetchTrainingSample(activeAirportCode, entry.file)
      .then((parsed) => {
        if (!live) return;
        if (parsed.ok) setSampleState({ status: "ready", sample: parsed.value });
        else setSampleState({ status: "invalid", problem: parsed.problem });
      })
      .catch((error: unknown) => {
        if (!live) return;
        setSampleState({ status: "invalid", problem: error instanceof Error ? error.message : String(error) });
      });
    return () => {
      live = false;
    };
  }, [activeAirportCode, entry]);

  const sample = sampleState.status === "ready" ? sampleState.sample : null;
  const overlays = useTrainingOverlays(activeAirportCode || null, sample, flightKey);
  const executorOverlay = overlays.executor.shown && overlays.executor.load.status === "ready" ? overlays.executor.load.overlay : null;
  const priorOverlay = overlays.prior.shown && overlays.prior.load.status === "ready" ? overlays.prior.load.overlay : null;
  useTrainingAutopilot();
  const executorFlights = useMemo(
    () => new Map((executorOverlay?.flights ?? []).map((flight) => [flight.flightKey, flight])),
    [executorOverlay],
  );
  // the model whose sentences are read, when it is loaded for this set: its samples beside each flight
  const readModel = useMemo(() => {
    const item = overlays.generations.find(({ entry }) => entry.id === trainingSource?.overlayId);
    return item?.load.status === "ready" ? item.load.overlay : null;
  }, [overlays.generations, trainingSource]);
  const readFlights = useMemo(
    () => new Map((readModel?.flights ?? []).map((flight) => [flight.flightKey, flight])),
    [readModel],
  );

  // Keep the selection on the same flight across a reload when it is still there; otherwise the first, so the
  // sentence bar is never blank beside a list.
  useEffect(() => {
    if (!sample) {
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

  return (
    <section className="training-panel" aria-label="Training" hidden={hidden}>
      <header className="training-panel-header">
        <h2>Training</h2>
        {/* Everything read ONCE — what the module is, the vocabulary, the shas — folds away, so the flight list keeps
            the height the sentence bar would otherwise take. */}
        <button type="button" className="training-about-toggle" aria-expanded={aboutOpen}
          aria-label={aboutOpen ? "Hide what this panel shows" : "What does this panel show?"}
          onClick={() => setAboutOpen((open) => !open)}>
          {aboutOpen ? "×" : "ⓘ"}
        </button>
      </header>

      {aboutOpen ? (
        <>
          <p className="training-panel-lede">
            Each arrival read as the instructions a controller could have given — six columns per{" "}
            {sample ? sample.vocabulary.stepS : "—"} s step — with what every word allows: a heading word's band over the
            rows it is judged on, the capture turn and corridor, the altitude tube, the speed band.
          </p>
          <NotesList items={[
            ...LAYER_SWITCHES.map(({ layer, text, title }) => ({ key: layer, name: text, text: title })),
            { key: "autopilot", name: "Fly on band click",
              text: FLY_ON_CLICK },
            ...[...overlays.executor.entries, ...overlays.prior.entries, ...overlays.generations.map(({ entry }) => entry)]
              .map((item) => ({ key: item.id, name: item.id, text: item.title })),
          ]} />
          {sample ? <TrainingVocabularyNotes sample={sample} /> : null}
        </>
      ) : null}

      {indexState.status === "loading" ? <p className="training-note" role="status">Reading {trainingIndexPath(airport)} …</p> : null}
      {indexState.status === "absent" ? <EmptyState airport={airport} /> : null}
      {indexState.status === "invalid" ? (
        <ProblemBox title={`${trainingIndexPath(airport)} cannot be read.`} detail={indexState.problem} />
      ) : null}

      {indexState.status === "ready" ? (
        <>
          {indexState.index.sets.length > 1 ? (
            <label className="training-field" title={entry?.title}>
              <span>Set</span>
              <select value={setId ?? ""} onChange={(event) => setSetId(event.target.value)}>
                {indexState.index.sets.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.id} · {item.flights} flights, {item.cohort.split}
                    {trainingSetRefusal(item) === null ? "" : ` · ${item.readingRule} — refused`}
                  </option>
                ))}
              </select>
            </label>
          ) : null}

          {/* ④ an entry that is not a set entry names itself and its field; the others load */}
          {indexState.index.rejected.map((item) => (
            <ProblemBox key={item.id} title={`Entry ${item.id} was rejected.`} detail={item.problem} />
          ))}
          {/* ② refused by name, from the manifest alone */}
          {sampleState.status === "refused" ? <ProblemBox title={`Set ${setId} is not read.`} detail={sampleState.problem} /> : null}
          {sampleState.status === "loading" ? <p className="training-note" role="status">Loading {entry?.file} …</p> : null}
          {/* ③ a readable set that fails; the others are untouched */}
          {sampleState.status === "invalid" ? <ProblemBox title={`Set ${setId} cannot be read.`} detail={sampleState.problem} /> : null}

          {sample ? (
            <>
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

              {/* THE MODELS' OWN SENTENCES: which sentence every view reads */}
              <ModelSentences items={overlays.generations} setId={sample.setId} airport={airport} flights={sample.flights} />

              {/* THE EXECUTOR, LIVE: flown by the backend when a word is picked, never read from an overlay */}
              <fieldset className="training-layers training-autopilot-section" aria-label="Autopilot (live)">
                <legend style={{ color: TRAINING_AUTOPILOT_COLOR }}>Autopilot (live)</legend>
                <label style={{ color: TRAINING_AUTOPILOT_COLOR }}
                  title={FLY_ON_CLICK}>
                  <input type="checkbox" checked={trainingAutopilotAuto} onChange={(event) => setTrainingAutopilotAuto(event.target.checked)} />
                  Fly on band click
                </label>
                <TrainingAutopilotCard />
              </fieldset>
            </>
          ) : null}

          <fieldset className="training-layers">
            <legend>Draw</legend>
            {LAYER_SWITCHES.map(({ layer, colour, text, title }) => (
              <label key={layer} style={{ color: colour }} title={title}>
                <input type="checkbox" checked={trainingLayers[layer]} onChange={(event) => setTrainingLayer(layer, event.target.checked)} />
                {text}
              </label>
            ))}
            {/* THE OVERLAYS over this set: another model's output on its own flights. */}
            {sample ? (
              <>
                <OverlaySwitch state={overlays.executor} colour={TRAINING_EXECUTOR_COLOR} setId={sample.setId} airport={airport}
                  text="Executor replay" />
                <OverlaySwitch state={overlays.prior} colour={undefined} setId={sample.setId} airport={airport}
                  text="Prior predictions" />
              </>
            ) : null}
          </fieldset>

          {overlays.manifest.status === "invalid" ? (
            <ProblemBox title={`${trainingOverlaysPath(airport)} cannot be read.`} detail={overlays.manifest.problem} />
          ) : null}
          {overlays.manifest.status === "ready" ? overlays.manifest.overlays.rejected.map((item) => (
            <ProblemBox key={`overlay-${item.id}`} title={`Overlay ${item.id} was rejected.`} detail={item.problem} />
          )) : null}
          {executorOverlay ? <TrainingExecutorGate overlay={executorOverlay} /> : null}
          {priorOverlay && sample ? <TrainingPriorReadout overlay={priorOverlay} stepS={sample.vocabulary.stepS} /> : null}
          {sample && overlays.generations.some(({ load }) => load.status === "ready") ? (
            <TrainingGenerationReadout flights={sample.flights}
              overlays={overlays.generations.flatMap(({ load }) => (load.status === "ready" ? [load.overlay] : []))} />
          ) : null}
        </>
      ) : null}
    </section>
  );
}
