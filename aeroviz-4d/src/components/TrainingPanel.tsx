/**
 * TrainingPanel.tsx
 * -----------------
 * The Training task's left-dock panel: which exported set, which flight, which envelopes are
 * drawn — for the instruction vocabulary (`TRAINING_READING_RULE`). Design:
 * `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`.
 *
 * It owns the only fetch and publishes the selected flight through `trainingSelection`, which the
 * sentence bar, the read-back window and the 3D layer draw.
 *
 * The states all name what failed:
 *   ① no `index.json`                → the path, the command, the dev-server restart (AV5)
 *   ② a listed set this reader refuses → WHY, by name, from the manifest alone (no download):
 *                                       a superseded vocabulary, another spec, a prior set
 *   ③ a readable set that fails to parse → THAT set alone, with the field
 *   ④ an entry that is not a set entry  → greyed out on its own; the others still load (AV6)
 *
 * Over the open set it offers the OVERLAYS published for it (`useTrainingOverlays`): the executor's replay of each
 * flight's truth sentence and the prior's predictions over it, each behind its own switch, with their readouts —
 * the replay's gate table and the prior's val readout — folded below the switches. A set with none says so and names
 * the command that writes one; nothing is drawn in its place.
 *
 * And THE EXECUTOR, LIVE (`useTrainingAutopilot`), in its own section above Draw: a word picked — the sentence bar's
 * "Fly this segment" button, or a band clicked while this section's switch is on — is flown by the backend now, not read
 * from any overlay; its answer is shown here (`TrainingAutopilotCard`) and drawn in the sentence bar, the read-back window
 * and 3D.
 */

import { useEffect, useMemo, useState } from "react";
import { useApp, type TrainingLayers } from "../context/AppContext";
import useTrainingOverlays, { type OverlayKindState } from "../hooks/useTrainingOverlays";
import useTrainingAutopilot from "../hooks/useTrainingAutopilot";
import TrainingAutopilotCard from "./TrainingAutopilotCard";
import { isMissingJsonAsset } from "../utils/fetchJson";
import {
  TRAINING_AUTOPILOT_COLOR,
  TRAINING_CANDIDATE_COLOR,
  TRAINING_CORRIDOR_COLOR,
  TRAINING_EXECUTOR_COLOR,
  TRAINING_HEADING_BAND_COLOR,
  TRAINING_OUTSIDE_COLOR,
  TRAINING_TUBE_COLOR,
} from "../utils/trainingWordColors";
import { trainingOverlaysPath, type TrainingExecutorFlight } from "../data/trainingOverlays";
import { TrainingExecutorGate, TrainingPriorReadout } from "./TrainingResults";
import {
  fetchTrainingIndex,
  fetchTrainingSample,
  trainingIndexPath,
  trainingSetRefusal,
  trainingVerdicts,
  TRAINING_COLUMNS,
  TRAINING_SPEC_SHA256,
  type TrainingIndex,
  type TrainingSample,
  type TrainingSetEntry,
} from "../data/trainingSample";

/** The Draw switches, in drawing order: what each shows, in its own colour. */
const LAYER_SWITCHES: Array<{ layer: keyof TrainingLayers; colour: string; text: string }> = [
  { layer: "headingBands", colour: TRAINING_HEADING_BAND_COLOR, text: "heading words: each word's band over the rows it is judged on" },
  { layer: "corridor", colour: TRAINING_CORRIDOR_COLOR, text: "the capture: its turn and the corridor" },
  { layer: "vertical", colour: TRAINING_TUBE_COLOR, text: "vertical: the altitude tubes" },
  { layer: "candidates", colour: TRAINING_CANDIDATE_COLOR, text: "every candidate runway" },
];

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

const SHA_SHOWN = 12;

/** The command that writes the export, as the empty state shows it. */
export const TRAINING_EXPORT_COMMAND =
  "python run_ts.py instruction_training_export --dir 4dTrajectory/outputs/POOLED/instruction_language/<an instruction-v3 artefact> " +
  "--airports-root aeroviz-4d/public/data/airports --airport ";

/** The commands that write the overlays, as the switches show them when a set has none. */
export const TRAINING_OVERLAY_COMMAND = {
  "executor-replay": "python run_ts.py executor_training_export --executor <spec dir> --replay <spec dir>/replay-val " +
    "--instructions <artefact> --airports-root aeroviz-4d/public/data/airports --set ",
  "prior-prediction": "python run_ts.py prior_training_export --prior <prior dir> --instructions <artefact> " +
    "--airports-root aeroviz-4d/public/data/airports --set ",
} as const;

/** One overlay kind's switch: on / off, which overlay (when several), and what failed. */
function OverlaySwitch<T>({ state, colour, text, setId, airport }: {
  state: OverlayKindState<T>; colour: string; text: string; setId: string; airport: string;
}) {
  const { entry, entries, shown, setShown, load, choose } = state;
  return (
    <div className="training-overlay-switch">
      <label style={entry ? { color: colour } : undefined} className={entry ? undefined : "training-slot"}>
        <input type="checkbox" checked={entry !== null && shown} disabled={entry === null}
          onChange={(event) => setShown(event.target.checked)} />
        {text}
      </label>
      {entry === null ? (
        <p className="training-overlay-note">
          none published over {setId}: <code>{TRAINING_OVERLAY_COMMAND[state.kind]}{setId} --airport {airport}</code>
        </p>
      ) : null}
      {entry !== null && entries.length > 1 ? (
        <select aria-label={`which ${state.kind} overlay`} value={entry.id} onChange={(event) => choose(event.target.value)}>
          {entries.map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}
        </select>
      ) : null}
      {entry !== null ? <p className="training-overlay-note">{entry.title}</p> : null}
      {entry !== null && shown && load.status === "loading" ? (
        <p className="training-overlay-note" role="status">Loading {entry.file} …</p>
      ) : null}
      {entry !== null && load.status === "invalid" ? (
        <div className="training-problem" role="alert">
          <p className="training-empty-title">Overlay {entry.id} cannot be read.</p>
          <p className="training-problem-detail">{load.problem}</p>
        </div>
      ) : null}
    </div>
  );
}

/** What the flight list says of a flight's replay: its outcome, and its words inside of those judged. */
export function executorTag(flight: TrainingExecutorFlight): { text: string; ok: boolean; title: string } {
  if (!flight.flown) return { text: "not flown", ok: false, title: `the replay does not fly it: ${flight.group}` };
  const { wordsInside, wordsJudged } = flight.counts;
  const outcome = flight.outcome.replace(/_/g, " ");
  return {
    text: `${outcome} · ${wordsInside}/${wordsJudged}`,
    ok: flight.outcome === "landed" && wordsInside === wordsJudged,
    title: `the executor on ${flight.group}: ${outcome}; ${wordsInside} of ${wordsJudged} words inside their envelopes, ` +
      `as the replay gate counts them${flight.flewTheSentence ? " — it flew the sentence" : ""}`,
  };
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

export default function TrainingPanel() {
  const {
    activeAirportCode, setTrainingSelection, trainingLayers, setTrainingLayer, trainingAutopilotAuto, setTrainingAutopilotAuto,
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
        // Open on a set this reader CAN read: the manifest states each set's reading rule and
        // spec, so which ones are current is known before anything is downloaded.
        const readable = parsed.value.sets.find((item) => trainingSetRefusal(item) === null);
        setSetId((readable ?? parsed.value.sets[0])?.id ?? null);
      })
      .catch((error: unknown) => {
        if (!live) return;
        // "Not exported yet" and "exported but broken" are different answers; `isMissingJsonAsset`
        // also catches vite's SPA fallback HTML, which is what a 404 under `public/data` looks like.
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
  useTrainingAutopilot(activeAirportCode || null, sample);
  const executorFlights = useMemo(
    () => new Map((executorOverlay?.flights ?? []).map((flight) => [flight.flightKey, flight])),
    [executorOverlay],
  );

  // Keep the selection on the same flight across a reload when it is still there; otherwise the
  // first, so the sentence bar is never blank beside a list.
  useEffect(() => {
    if (!sample) {
      setFlightKey(null);
      return;
    }
    setFlightKey((previous) =>
      previous && sample.flights.some((flight) => flight.flightKey === previous)
        ? previous
        : (sample.flights[0]?.flightKey ?? null),
    );
  }, [sample]);

  // ── publish what the other views draw ─────────────────────────────────────
  useEffect(() => {
    const flight = sample?.flights.find((item) => item.flightKey === flightKey) ?? null;
    setTrainingSelection(
      flight && sample ? { vocabulary: sample.vocabulary, candidates: sample.candidates, flight } : null,
    );
  }, [sample, flightKey, setTrainingSelection]);

  useEffect(() => () => setTrainingSelection(null), [setTrainingSelection]);

  return (
    <section className="training-panel" aria-label="Training">
      <header className="training-panel-header">
        <h2>Training</h2>
        {/* Everything read ONCE — what the module is, the vocabulary, the shas — folds away, so
            the flight list keeps the height the sentence bar would otherwise take. */}
        <button
          type="button"
          className="training-about-toggle"
          aria-expanded={aboutOpen}
          aria-label={aboutOpen ? "Hide what this panel shows" : "What does this panel show?"}
          onClick={() => setAboutOpen((open) => !open)}
        >
          {aboutOpen ? "×" : "ⓘ"}
        </button>
      </header>

      {aboutOpen ? (
        <p className="training-panel-lede">
          Each arrival read as the instructions a controller could have given — six columns per
          2 s step — with what every word allows: a heading word's band over the rows it is judged
          on, the capture turn and corridor, the altitude tube, the speed band.
        </p>
      ) : null}

      {indexState.status === "loading" ? (
        <p className="training-note" role="status">Reading {trainingIndexPath(airport)} …</p>
      ) : null}

      {indexState.status === "absent" ? <EmptyState airport={airport} /> : null}

      {indexState.status === "invalid" ? (
        <div className="training-problem" role="alert">
          <p className="training-empty-title">{trainingIndexPath(airport)} cannot be read.</p>
          <p className="training-problem-detail">{indexState.problem}</p>
        </div>
      ) : null}

      {indexState.status === "ready" ? (
        <>
          {indexState.index.sets.length > 1 ? (
            <label className="training-field">
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
          {entry ? <p className="training-note">{entry.title}</p> : null}

          {/* THE EXECUTOR, LIVE: flown by the backend when a word is picked, never read from an overlay */}
          {sample ? (
            <fieldset className="training-layers training-autopilot-section" aria-label="Autopilot (live)">
              <legend style={{ color: TRAINING_AUTOPILOT_COLOR }}>Autopilot (live)</legend>
              <label style={{ color: TRAINING_AUTOPILOT_COLOR }}>
                <input type="checkbox" checked={trainingAutopilotAuto}
                  onChange={(event) => setTrainingAutopilotAuto(event.target.checked)} />
                fly a word's segment as soon as its band is clicked
              </label>
              <TrainingAutopilotCard
                flight={sample.flights.find((item) => item.flightKey === flightKey) ?? null}
                vocabulary={sample.vocabulary} candidates={sample.candidates} />
            </fieldset>
          ) : null}

          <fieldset className="training-layers">
            <legend>Draw</legend>
            {LAYER_SWITCHES.map(({ layer, colour, text }) => (
              <label key={layer} style={{ color: colour }}>
                <input
                  type="checkbox"
                  checked={trainingLayers[layer]}
                  onChange={(event) => setTrainingLayer(layer, event.target.checked)}
                />
                {text}
              </label>
            ))}
            {/* THE OVERLAYS over this set: another model's output on its own flights. */}
            {sample ? (
              <>
                <OverlaySwitch state={overlays.executor} colour={TRAINING_EXECUTOR_COLOR} setId={sample.setId} airport={airport}
                  text="the executor's replay: its flown track and each word's verdict" />
                <OverlaySwitch state={overlays.prior} colour={TRAINING_EXECUTOR_COLOR} setId={sample.setId} airport={airport}
                  text="the prior's predictions at each step (teacher-forced)" />
              </>
            ) : null}
          </fieldset>

          {overlays.manifest.status === "invalid" ? (
            <div className="training-problem" role="alert">
              <p className="training-empty-title">{trainingOverlaysPath(airport)} cannot be read.</p>
              <p className="training-problem-detail">{overlays.manifest.problem}</p>
            </div>
          ) : null}
          {overlays.manifest.status === "ready" ? overlays.manifest.overlays.rejected.map((item) => (
            <div className="training-problem" role="alert" key={`overlay-${item.id}`}>
              <p className="training-empty-title">Overlay {item.id} was rejected.</p>
              <p className="training-problem-detail">{item.problem}</p>
            </div>
          )) : null}
          {executorOverlay ? <TrainingExecutorGate overlay={executorOverlay} /> : null}
          {priorOverlay ? <TrainingPriorReadout overlay={priorOverlay} /> : null}

          {/* ④ an entry that is not a set entry names itself and its field; the others load */}
          {indexState.index.rejected.map((item) => (
            <div className="training-problem" role="alert" key={item.id}>
              <p className="training-empty-title">Entry {item.id} was rejected.</p>
              <p className="training-problem-detail">{item.problem}</p>
            </div>
          ))}

          {/* ② refused by name, from the manifest alone */}
          {sampleState.status === "refused" ? (
            <div className="training-problem" role="alert">
              <p className="training-empty-title">Set {setId} is not read.</p>
              <p className="training-problem-detail">{sampleState.problem}</p>
            </div>
          ) : null}

          {sampleState.status === "loading" ? (
            <p className="training-note" role="status">Loading {entry?.file} …</p>
          ) : null}

          {/* ③ a readable set that fails; the others are untouched */}
          {sampleState.status === "invalid" ? (
            <div className="training-problem" role="alert">
              <p className="training-empty-title">Set {setId} cannot be read.</p>
              <p className="training-problem-detail">{sampleState.problem}</p>
            </div>
          ) : null}

          {sample ? (
            <>
              {aboutOpen ? (
                <dl className="training-vocabulary">
                  <div>
                    <dt>Vocabulary</dt>
                    <dd>
                      {sample.vocabulary.readingRule} · spec {sample.vocabulary.specSha256.slice(0, SHA_SHOWN)}
                      {sample.vocabulary.specSha256 === TRAINING_SPEC_SHA256 ? " (frozen)" : ""} · labeller{" "}
                      {sample.vocabulary.labellerSourceSha256.slice(0, SHA_SHOWN)}
                    </dd>
                  </div>
                  <div>
                    <dt>Columns</dt>
                    <dd>
                      {TRAINING_COLUMNS.map((column) =>
                        column === "runway"
                          ? `runway ${sample.candidates.length}`
                          : `${column} ${sample.vocabulary.classCounts[column]}`,
                      ).join(" · ")}{" "}
                      classes, each with "unchanged"
                    </dd>
                  </div>
                  <div>
                    <dt>Runway pointer</dt>
                    <dd>
                      {sample.candidates.map((candidate) => candidate.ident).join(" ")} (sha{" "}
                      {sample.candidatesSha256.slice(0, SHA_SHOWN)})
                    </dd>
                  </div>
                  <div>
                    <dt>Landing</dt>
                    <dd>
                      passing the threshold within {sample.vocabulary.landingMaxHeightM} m of its height and,
                      off the centreline, within{" "}
                      {sample.candidates.map((candidate) => `${candidate.ident} ${candidate.landingCrossLimitM} m`).join(", ")}{" "}
                      ({sample.vocabulary.landingCrossLimitM} m, or half the spacing to a parallel runway)
                    </dd>
                  </div>
                  <div>
                    <dt>Heading</dt>
                    <dd>
                      {sample.vocabulary.headingTargetsDeg[1] - sample.vocabulary.headingTargetsDeg[0]}° grid, read
                      step by step: a word says where the track is {sample.vocabulary.headingLeadS} s later, and from
                      then to the next word's the track stays within ±{sample.vocabulary.headingToleranceDeg}° of it
                    </dd>
                  </div>
                  <div>
                    <dt>Capture turn</dt>
                    <dd>
                      from the clearance onto the course, begun where the track turns toward it faster than{" "}
                      {sample.vocabulary.turnOnsetRateDegS}°/s: monotone, {sample.vocabulary.turnRateMinDegS}–
                      {sample.vocabulary.turnRateMaxDegS}°/s (the lowest rate only for turns of{" "}
                      {sample.vocabulary.turnRateMinFromDeg}° or more), at most {sample.vocabulary.turnBankMaxDeg}° of bank
                    </dd>
                  </div>
                  <div>
                    <dt>Corridor</dt>
                    <dd>
                      {sample.vocabulary.corridorHalfWidthM} m at the threshold, widening{" "}
                      {sample.vocabulary.corridorWideningDeg}°, course ±{sample.vocabulary.corridorCourseToleranceDeg}°;
                      intercept {sample.vocabulary.interceptAngleDeg}°
                    </dd>
                  </div>
                  <div>
                    <dt>Altitude</dt>
                    <dd>
                      {sample.vocabulary.altitudeTargetsM[1] - sample.vocabulary.altitudeTargetsM[0]} m MSL grid to{" "}
                      {sample.vocabulary.altitudeTargetsM[sample.vocabulary.altitudeTargetsM.length - 1]} m, or
                      "descend to land"; tube ±{sample.vocabulary.altitudeToleranceM} m
                    </dd>
                  </div>
                  <div>
                    <dt>Angle</dt>
                    <dd>
                      {sample.vocabulary.angleClasses
                        .map((angle) => angle.value === sample.vocabulary.angleLevelValue
                          ? angle.name
                          : `${angle.name} ${angle.nominalDeg}° (${angle.lowDeg}…${angle.steepDeg}°)`)
                        .join(" · ")}
                    </dd>
                  </div>
                  <div>
                    <dt>Speed</dt>
                    <dd>
                      ground speed {sample.vocabulary.speedTargetsMps[0]}…
                      {sample.vocabulary.speedTargetsMps[sample.vocabulary.speedTargetsMps.length - 1]} m/s or
                      "unspecified"; band ±{sample.vocabulary.speedToleranceMps} m/s, at most{" "}
                      {sample.vocabulary.speedAccelMaxMps2} m/s² between
                    </dd>
                  </div>
                  <div>
                    <dt>Read on</dt>
                    <dd>
                      signals smoothed over {sample.vocabulary.smoothingS.track} s (track),{" "}
                      {sample.vocabulary.smoothingS.altitude} s (altitude), {sample.vocabulary.smoothingS.speed} s
                      (speed), one step every {sample.vocabulary.stepS} s
                    </dd>
                  </div>
                </dl>
              ) : null}

              <p className="training-note">
                {sample.flights.length} flights · {sample.cohort.drawnFrom}.
                {entry && entry.flights !== sample.flights.length
                  ? ` The manifest says ${entry.flights}: this set's two files are from different exports.`
                  : ""}
              </p>

              <ul className="training-flight-list">
                {sample.flights.map((flight) => (
                  <li key={flight.flightKey}>
                    <button
                      type="button"
                      className={flight.flightKey === flightKey ? "active" : undefined}
                      onClick={() => setFlightKey(flight.flightKey)}
                    >
                      <span className="training-flight-callsign">{flight.callsign}</span>
                      <span className="training-flight-runway">{flight.runway}</span>
                      <span className="training-flight-stratum">{flight.stratum}</span>
                      <span className="training-flight-events">
                        {trainingVerdicts(flight).instructionsAfterStep0} words
                      </span>
                      {executorFlights.has(flight.flightKey) ? (() => {
                        const tag = executorTag(executorFlights.get(flight.flightKey)!);
                        return (
                          <span className="training-flight-executor" title={tag.title}
                            style={{ color: tag.ok ? TRAINING_EXECUTOR_COLOR : TRAINING_OUTSIDE_COLOR }}>
                            {tag.text}
                          </span>
                        );
                      })() : null}
                    </button>
                  </li>
                ))}
              </ul>
            </>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
