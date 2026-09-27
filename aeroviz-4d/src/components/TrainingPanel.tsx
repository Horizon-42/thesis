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
 * reading in each tooltip), then "Details": one line per readout behind the overlays, its conclusion. Everything longer —
 * what the module is, the vocabulary's numbers, the readouts' tables — is on the DETAILS PAGE (`TrainingDetails`), opened
 * by the header's "Details" or by a readout's line, on its section; the dock never unfolds it.
 *
 * Over the open set it offers the OVERLAYS published for it (`useTrainingOverlays`): the executor's replay and the
 * prior's predictions, each behind its own switch; a set with none says so and folds away the command that writes one.
 * And THE MODELS' OWN SENTENCES (`prior-generation`): each model round published for the set is a line of "Sentences
 * read" — grouped by model (base, landing, augmented), a model published at several rounds a heading with a line per
 * round; its colour, its name, how many of its samples landed on this set — which chooses it as the sentence read
 * (`trainingSource`, as the sentence bar's tabs do); while one is chosen, each flight in the list shows its samples,
 * filled where the flight landed.
 *
 * THE DOCK ENDS ABOVE THE SENTENCE BAR (the bar measures itself, `--training-bar-height`): the flight list takes the
 * height left over — never less than a few rows, the dock scrolling below that — so the switches under it are never
 * covered.
 * And THE EXECUTOR, LIVE (`useTrainingAutopilot`): a word picked — the sentence bar's Fly button, or a band clicked
 * while its switch is on — is flown by the backend now, never read from an overlay (`TrainingAutopilotCard`).
 */

import { useCallback, useEffect, useMemo, useState, type MouseEvent } from "react";
import { useApp, type TrainingLayers } from "../context/AppContext";
import useTrainingOverlays, { type OverlayKindState, type OverlaysManifestState } from "../hooks/useTrainingOverlays";
import useTrainingAutopilot from "../hooks/useTrainingAutopilot";
import TrainingAutopilotCard from "./TrainingAutopilotCard";
import TrainingVocabularyNotes from "./TrainingVocabularyNotes";
import ProblemBox from "./training/ProblemBox";
import TrainingDetails, { type TrainingDetailsSection } from "./training/TrainingDetails";
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
  trainingModelGroups,
  trainingOverlaysPath,
  type TrainingExecutorFlight,
  type TrainingGenerationFlight,
  type TrainingGenerationOverlay,
} from "../data/trainingOverlays";
import { TRAINING_OUTCOME_TAG, trainingModelText } from "../data/trainingText";
import {
  executorGateSummary,
  generationSummary,
  priorReadoutSummary,
  TrainingExecutorGate,
  TrainingGenerationReadout,
  TrainingPriorReadout,
} from "./TrainingResults";
import type { GenerationItem } from "../hooks/useTrainingOverlays";
import {
  fetchTrainingIndex,
  fetchTrainingSample,
  TRAINING_COLUMNS,
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

/** The overlay switches' names, as the Draw box and the details page's reasons say them. */
const EXECUTOR_SWITCH = "Executor replay";
const PRIOR_SWITCH = "Prior predictions";

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
  "prior-generation": "python run_ts.py prior_generation_training_export --prior <prior dir, or one round of a post-training run> " +
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

/** One model round as a line of the list: its radio (read from its first sample), its name, and how many of its
 *  samples over the set's flights landed. */
function ModelLine({ overlay, text, chosen, flights, indent }: {
  overlay: TrainingGenerationOverlay; text: string; chosen: boolean; flights: TrainingFlight[]; indent: boolean;
}) {
  const { setTrainingSource } = useApp();
  const count = generationLanded(overlay, flights).all;
  return (
    <label className={`training-model${indent ? " training-model-round" : ""}`} title={trainingModelText(overlay.model)}>
      <input type="radio" name="training-source" checked={chosen}
        onChange={() => setTrainingSource({ overlayId: overlay.overlayId, sample: 0 })} />
      {indent ? null : <span className="training-model-swatch" style={{ background: trainingModelColour(overlay.model) }} />}
      <span className="training-model-name">{text}</span>
      {count !== null ? (
        <span className="training-model-count" title={`of the ${count.flights} sentences it said over the set's flights ` +
          "(each flown by the executor from its first predicted step), those that landed"}>
          {" "}{Math.round(count.landed * count.flights)}/{count.flights} landed
        </span>
      ) : null}
    </label>
  );
}

/** THE MODELS' OWN SENTENCES published for the set, by model (`trainingModelGroups`: base, landing, augmented; a model
 *  published at several rounds is a heading with one line per round): each line chooses it as the sentence read (from its
 *  first sample); the truth first — and checked when the chosen model is not one of this set's. An overlay still loading
 *  or unreadable is listed after them by its id. */
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
  const groups = trainingModelGroups(items.flatMap(({ load }) => (load.status === "ready" ? [load.overlay] : [])),
    (overlay) => overlay);
  return (
    <div className="training-models" role="radiogroup" aria-label="Which sentence is read">
      <span className="training-models-title">Sentences read</span>
      <label className="training-model">
        <input type="radio" name="training-source" checked={chosen === null} onChange={() => setTrainingSource(null)} />
        <span className="training-model-swatch training-model-swatch-truth" />
        truth (labelled)
      </label>
      {groups.map((group) => (group.members.length === 1 ? (
        <ModelLine key={group.key} overlay={group.members[0]} text={group.memberLabel(group.members[0])}
          chosen={chosen === group.members[0].overlayId} flights={flights} indent={false} />
      ) : (
        <div key={group.key} className="training-model-group" role="group" aria-label={`${group.title}: the rounds published`}>
          <span className="training-model training-model-heading">
            <span className="training-model-swatch" style={{ background: trainingModelColour(group) }} />
            <span className="training-model-name">{group.title}</span>
          </span>
          {group.members.map((overlay) => (
            <ModelLine key={overlay.overlayId} overlay={overlay} text={`r${overlay.model.round}`}
              chosen={chosen === overlay.overlayId} flights={flights} indent />
          ))}
        </div>
      )))}
      {items.filter(({ load }) => load.status !== "ready").map(({ entry, load, retry }) => (
        <div key={entry.id}>
          <label className="training-model" title={entry.title}>
            <input type="radio" name="training-source" checked={false} disabled />
            <span className="training-model-swatch" />
            {entry.id}
            {load.status === "loading" || load.status === "idle" ? <span className="training-overlay-note" role="status"> loading …</span> : null}
          </label>
          {load.status === "invalid" ? (
            <>
              <ProblemBox title={`Overlay ${entry.id} cannot be read.`} detail={load.problem} />
              <button type="button" className="training-sentence-readback-button" onClick={retry}>Retry</button>
            </>
          ) : null}
        </div>
      ))}
    </div>
  );
}

/** Why no overlay can be read from the manifest, or null when it was read. */
function manifestAbsence(manifest: OverlaysManifestState): string | null {
  if (manifest.status === "ready") return null;
  if (manifest.status === "absent") return "none published for this set";
  return manifest.status === "invalid" ? "the overlays manifest cannot be read" : "loading …";
}

/** None of this set's — and how many entries the manifest rejected (a rejected entry names no kind or set; the dock
 *  names each). */
function noneReadable(manifest: OverlaysManifestState): string {
  const rejected = manifest.status === "ready" ? manifest.overlays.rejected.length : 0;
  return rejected === 0 ? "none published for this set"
    : `none published for this set; ${rejected} ${rejected === 1 ? "entry" : "entries"} of the manifest rejected`;
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

/** One readout in the panel: its name and its conclusion on one line (in full in its tooltip); it opens the details
 *  page on its section. */
function DetailsLink({ name, summary, onOpen }: { name: string; summary: string; onOpen: (event: MouseEvent<HTMLElement>) => void }) {
  return (
    <li>
      <button type="button" className="training-details-link" aria-haspopup="dialog" title={`${name}: ${summary}`} onClick={onOpen}>
        <span className="training-details-link-name">{name}</span>
        <span className="training-details-link-summary">{summary}</span>
        <span className="training-details-link-open" aria-hidden="true">›</span>
      </button>
    </li>
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
  /** The details page while it is open: its section, and the control that opened it (the focus goes back there). */
  const [details, setDetails] = useState<{ section: string; opener: HTMLElement } | null>(null);
  const openDetails = (section: string) => (event: MouseEvent<HTMLElement>) => setDetails({ section, opener: event.currentTarget });
  const showSection = useCallback((section: string) => setDetails((open) => (open === null ? null : { ...open, section })), []);
  const closeDetails = useCallback(() => setDetails(null), []);
  // a panel hidden (another task) closes its page: it does not come back unasked
  useEffect(() => {
    if (hidden) setDetails(null);
  }, [hidden]);

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

  // ── the details page ──────────────────────────────────────────────────────
  const generations = overlays.generations.flatMap(({ load }) => (load.status === "ready" ? [load.overlay] : []));
  const detailsSections: TrainingDetailsSection[] = [
    { id: "overview", title: "What this view shows", body: (
      <>
        <p className="training-details-lede">
          Each arrival read as the instructions a controller could have given — {TRAINING_COLUMNS.length} columns per{" "}
          {sample ? sample.vocabulary.stepS : "—"} s step — with what every word allows: a heading word's band over the
          rows it is judged on, the capture turn and corridor, the altitude tube, the speed band.
        </p>
        <h4 className="training-details-subhead">What each switch draws</h4>
        <NotesList items={[
          ...LAYER_SWITCHES.map(({ layer, colour, text, title }) => ({ key: layer, text: title, name: (
            <><span className="training-model-swatch" style={{ background: colour }} />{text}</>) })),
          { key: "autopilot", text: FLY_ON_CLICK, name: (
            <><span className="training-model-swatch" style={{ background: TRAINING_AUTOPILOT_COLOR }} />Fly on band click</>) },
        ]} />
        <h4 className="training-details-subhead">The overlays published over this set</h4>
        {overlays.executor.entries.length + overlays.prior.entries.length + overlays.generations.length === 0 ? (
          <p className="training-details-lede">None.</p>
        ) : (
          <NotesList items={[...overlays.executor.entries, ...overlays.prior.entries, ...overlays.generations.map(({ entry }) => entry)]
            .map((item) => ({ key: item.id, name: <code>{item.id}</code>, text: item.title }))} />
        )}
      </>
    ) },
    sample ? { id: "vocabulary", title: "Vocabulary", body: <TrainingVocabularyNotes sample={sample} /> }
      : { id: "vocabulary", title: "Vocabulary", body: null, absent: "no set read" },
    sample && generations.length > 0
      ? { id: "models", title: MODELS_TITLE, body: <TrainingGenerationReadout flights={sample.flights} overlays={generations} /> }
      : { id: "models", title: MODELS_TITLE, body: null, absent: sample ? generationAbsence(overlays.manifest, overlays.generations) : "no set read" },
    executorOverlay
      ? { id: "executor", title: EXECUTOR_TITLE, body: <TrainingExecutorGate overlay={executorOverlay} /> }
      : { id: "executor", title: EXECUTOR_TITLE, body: null,
          absent: sample ? overlayAbsence(overlays.manifest, overlays.executor, EXECUTOR_SWITCH) : "no set read" },
    priorOverlay && sample
      ? { id: "prior", title: PRIOR_TITLE, body: <TrainingPriorReadout overlay={priorOverlay} stepS={sample.vocabulary.stepS} /> }
      : { id: "prior", title: PRIOR_TITLE, body: null,
          absent: sample ? overlayAbsence(overlays.manifest, overlays.prior, PRIOR_SWITCH) : "no set read" },
  ];

  return (
    <section className="training-panel" aria-label="Training" hidden={hidden}>
      <header className="training-panel-header">
        <h2>Training</h2>
        {/* Everything read ONCE — what the module is, the vocabulary, the readouts — is on the details page, so the
            flight list keeps the dock's height. */}
        <button type="button" className="training-details-open" aria-haspopup="dialog" onClick={openDetails("overview")}>
          Details
        </button>
      </header>

      {details !== null && !hidden ? (
        <TrainingDetails context={[airport, sample?.setId, sample ? `${sample.flights.length} flights` : null]
          .filter((part) => part).join(" · ")}
          sections={detailsSections} sectionId={details.section} onSection={showSection} onClose={closeDetails}
          opener={details.opener} />
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
                  text={EXECUTOR_SWITCH} />
                <OverlaySwitch state={overlays.prior} colour={undefined} setId={sample.setId} airport={airport}
                  text={PRIOR_SWITCH} />
              </>
            ) : null}
          </fieldset>

          {overlays.manifest.status === "invalid" ? (
            <ProblemBox title={`${trainingOverlaysPath(airport)} cannot be read.`} detail={overlays.manifest.problem} />
          ) : null}
          {overlays.manifest.status === "ready" ? overlays.manifest.overlays.rejected.map((item) => (
            <ProblemBox key={`overlay-${item.id}`} title={`Overlay ${item.id} was rejected.`} detail={item.problem} />
          )) : null}
          {/* THE READOUTS: one line each here, their tables on the details page */}
          {sample && (generations.length > 0 || executorOverlay || priorOverlay) ? (
            <ul className="training-details-links" aria-label="Readouts">
              {generations.length > 0 ? (
                <DetailsLink name="Models' sentences" summary={generationSummary(generations, sample.flights)}
                  onOpen={openDetails("models")} />
              ) : null}
              {executorOverlay ? (
                <DetailsLink name="Replay gate" summary={executorGateSummary(executorOverlay)} onOpen={openDetails("executor")} />
              ) : null}
              {priorOverlay ? (
                <DetailsLink name="Prior readout" summary={priorReadoutSummary(priorOverlay)} onOpen={openDetails("prior")} />
              ) : null}
            </ul>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
