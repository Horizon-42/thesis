/**
 * TrainingPanel.tsx
 * -----------------
 * The Training task's left-dock panel: which exported set, which flight, which envelopes are drawn — for the stage-A
 * vocabulary (`TRAINING_READING_RULE`, five columns). Design: `4dTrajectory/ts_transformer/docs/two_tier/design/vocabulary.md`
 * §12.1 A23.
 *
 * It owns the index and the set's fetch, and opens the set (`TrainingFlightSession`) — the session publishes the flight on
 * screen through `trainingSelection`, which the sentence bar, the read-back window and the 3D layer draw. The dock keeps
 * it mounted from its first visit on, only ``hidden`` in the other tasks (`WorkbenchLeftDock`): its session — set, flight,
 * Δ, live answer — outlives a task switch.
 *
 * It reads `training/index_v5.json` and nothing else: never the instruction-v3 view's `training/index.json`. The states
 * all name what failed:
 *   ① no index                      → the path, the command, the dev-server restart (AV5)
 *   ② an index of another schema    → the schema found and the one this view reads
 *   ③ a set that fails to parse     → THAT set alone, with the field (a sample of another schema: its name and the expected)
 *   ④ an entry that is not a set entry → named on its own; the others still load (AV6)
 *
 * ONE LAYOUT FOR THE THREE STAGES (outline §6.2), KEPT SHORT, TOP TO BOTTOM: the stage switch; the set chooser (the set
 * and its own line of the intent registry, `GET /experiments/intent`); the session — its list, one line per readout, the
 * Draw switches. Everything longer — what the view shows, the set and its experiment, the vocabulary's numbers, the
 * readouts' tables — is on the DETAILS PAGE (`TrainingDetails`), opened by the header's ⓘ (never disabled: on its first
 * section, "The set and the experiment"), by a readout's line or by the sentence bar's notes; the dock never unfolds it.
 * The panel owns the page's state (`useDetailsPage`) and gives it to every stage's session, which supplies its sections.
 * The sentence on screen is chosen by the sentence bar's tabs, which the session gives (`data/trainingTabs.ts`).
 *
 * And THE EXECUTOR, LIVE (`useTrainingAutopilot`, run here): a word picked in the sentence bar — its Fly button, or a band
 * clicked — is flown by the backend now, never read from the file; the bar says the answer (`TrainingAutopilotStatus`).
 */

import { useEffect, useMemo, useState } from "react";
import { useApp } from "../context/AppContext";
import useTrainingAutopilot from "../hooks/useTrainingAutopilot";
import useTrainingPriorIndex from "../hooks/useTrainingPriorIndex";
import useTrainingWindowIndex from "../hooks/useTrainingWindowIndex";
import TrainingPriorSession from "./training/TrainingPriorSession";
import TrainingWindowSession from "./training/TrainingWindowSession";
import ProblemBox from "./training/ProblemBox";
import TrainingFlightSession from "./training/TrainingFlightSession";
import { EXPERIMENT_SECTION, useDetailsPage } from "./training/PanelParts";
import TrainingDetails from "./training/TrainingDetails";
import { SetChooser } from "./training/SetParts";
import useTrainingSet from "../hooks/useTrainingSet";
import { useTrainingSetIntent } from "../data/trainingSetIntent";
import { isMissingJsonAsset } from "../utils/fetchJson";
import {
  fetchTrainingIndex,
  fetchTrainingSample,
  TRAINING_SPLITS,
  trainingIndexPath,
  type TrainingIndex,
  type TrainingSample,
  type TrainingSetEntry,
} from "../data/trainingSample";
import { trainingPriorIndexPath } from "../data/trainingPriorSample";
import { trainingWindowIndexPath } from "../data/trainingWindowSample";

type IndexState =
  | { status: "loading" }
  | { status: "absent" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; index: TrainingIndex };

/** The command that writes the export, as the empty state shows it. */
const TRAINING_EXPORT_COMMAND =
  "python run_ts.py training_export --instructions 4dTrajectory/outputs/POOLED/instruction_language/<artefact> " +
  "--executor 4dTrajectory/outputs/POOLED/executor/<spec> --set-id <set id> --airports ";

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
  const { activeAirportCode } = useApp();
  const airport = activeAirportCode || "—";

  const [indexState, setIndexState] = useState<IndexState>({ status: "loading" });
  const [setId, setSetId] = useState<string | null>(null);
  /** Whose sets the panel shows: stage A's (`index_v5.json`), stage B's prior sets (`index_prior_v3.json`) or stage C's
   *  window sets (`index_post_v2.json`) — B and C offered only where the airport has the file. */
  const [viewing, setViewing] = useState<"stageA" | "prior" | "window">("stageA");
  const priorIndex = useTrainingPriorIndex(activeAirportCode || null);
  const windowIndex = useTrainingWindowIndex(activeAirportCode || null);
  // a switch is offered where the airport has such sets (or an index to complain about); an airport without the one chosen
  // at the last airport shows stage A
  const priorOffered = (priorIndex.status === "ready" && priorIndex.index.sets.length > 0) || priorIndex.status === "invalid";
  const windowOffered = (windowIndex.status === "ready" && windowIndex.index.sets.length > 0) || windowIndex.status === "invalid";
  const showing = (viewing === "prior" && priorOffered) || (viewing === "window" && windowOffered) ? viewing : "stageA";
  // the details page's state, given to every stage's session (a panel hidden in another task keeps no page open)
  const details = useDetailsPage(hidden);
  useTrainingAutopilot();

  // ── the index ─────────────────────────────────────────────────────────────
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
        setSetId(parsed.value.sets[0]?.id ?? null);
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

  // ── the chosen set (the one loader of every stage, outline §6.2 item 5) ─────
  const setState = useTrainingSet<TrainingSample>(activeAirportCode && entry ? `${activeAirportCode}/${entry.id}/${entry.file}` : null,
    () => fetchTrainingSample(activeAirportCode, entry!.file, entry!.id, TRAINING_SPLITS));      // stage A's sets (D109)
  const intentA = useTrainingSetIntent(showing === "stageA" ? entry?.id ?? null : null);

  const sample = setState.status === "ready" ? setState.sample : null;
  // whether a stage's session is on screen (it draws the details page then); otherwise why not
  const windowShown = showing === "window" && windowIndex.status === "ready" && !!activeAirportCode && windowIndex.index.airport === activeAirportCode;
  const priorShown = showing === "prior" && priorIndex.status === "ready" && !!activeAirportCode;
  const stageAShown = showing === "stageA" && indexState.status === "ready" && entry !== null && !!activeAirportCode;
  const sessionShown = windowShown || priorShown || stageAShown;
  const nothingShown = showing === "window" ? `no window set of ${airport} can be read (${trainingWindowIndexPath(airport)})`
    : showing === "prior" ? `no prior set of ${airport} can be read (${trainingPriorIndexPath(airport)})`
      : indexState.status === "loading" ? `reading ${trainingIndexPath(airport)}`
        : indexState.status === "ready" ? "the index lists no set" : `no Training export for ${airport} (${trainingIndexPath(airport)})`;

  return (
    <section className="training-panel" aria-label="Learning" hidden={hidden}>
      <header className="training-panel-header">
        {/* the module's name (its tab's); what it shows is still training — "Training details" below */}
        <h2>Learning</h2>
        {/* Everything read ONCE — what the module is, the vocabulary, the readouts — is on the details page, so the
            list keeps the dock's height. */}
        <button type="button" className="training-details-open" aria-haspopup="dialog" aria-label="Training details"
          title="Training details: the set and the experiment" onClick={details.open(EXPERIMENT_SECTION)}>
          ⓘ
        </button>
      </header>

      {priorOffered || windowOffered ? (
        <label className="training-field" title="Stage A: flights read back through the closed loop. Stage B: the sentences the prior said, flown by the executor. Stage C: windows of recorded traffic, the commanded aircraft flown on each round's words.">
          <span>Sets of</span>
          <select value={showing} onChange={(event) => setViewing(event.target.value as "stageA" | "prior" | "window")}>
            <option value="stageA">Stage A · read-back</option>
            {priorOffered ? <option value="prior">Stage B · prior</option> : null}
            {windowOffered ? <option value="window">Stage C · windows</option> : null}
          </select>
        </label>
      ) : null}
      {showing === "window" && windowIndex.status === "invalid" ? (
        <ProblemBox title={`${trainingWindowIndexPath(airport)} cannot be read.`} detail={windowIndex.problem} />
      ) : null}
      {windowShown ? (
        <>
          {windowIndex.index.rejected.map((item) => (
            <ProblemBox key={item.id} title={`Entry ${item.id} was rejected.`} detail={item.problem} />
          ))}
          {/* keyed by the airport: another airport's index never opens with this one's set, window or round */}
          <TrainingWindowSession key={`${activeAirportCode}:${windowIndex.index.airport}`} airport={activeAirportCode} sets={windowIndex.index.sets}
            details={details} />
        </>
      ) : null}
      {showing === "prior" && priorIndex.status === "invalid" ? (
        <ProblemBox title={`${trainingPriorIndexPath(airport)} cannot be read.`} detail={priorIndex.problem} />
      ) : null}
      {priorShown ? (
        <>
          {priorIndex.index.rejected.map((item) => (
            <ProblemBox key={item.id} title={`Entry ${item.id} was rejected.`} detail={item.problem} />
          ))}
          <TrainingPriorSession key={activeAirportCode} airport={activeAirportCode} sets={priorIndex.index.sets} details={details} />
        </>
      ) : null}

      {/* no session on screen (no set to open, an index that cannot be read): the page still opens — its ⓘ is never
          disabled — and says why there is nothing */}
      {details.shown !== null && !sessionShown ? (
        <TrainingDetails context={airport} sectionId={details.shown.section} onSection={details.show} onClose={details.close}
          opener={details.shown.opener} sections={[
            { id: EXPERIMENT_SECTION, title: "The set and the experiment", body: <p className="training-details-lede">{nothingShown}</p> },
          ]} />
      ) : null}
      {showing !== "stageA" ? null : <>
      {indexState.status === "loading" ? <p className="training-note" role="status">Reading {trainingIndexPath(airport)} …</p> : null}
      {indexState.status === "absent" ? <EmptyState airport={airport} /> : null}
      {indexState.status === "invalid" ? (
        <ProblemBox title={`${trainingIndexPath(airport)} cannot be read.`} detail={indexState.problem} />
      ) : null}

      {indexState.status === "ready" ? (
        <>
          {indexState.index.sets.length > 0 ? (
            <SetChooser sets={indexState.index.sets.map((item) => ({ id: item.id, count: `${item.flights} flights`, title: item.title, smoke: false }))}
              setId={setId} onChange={setSetId} intent={intentA} onOpenExperiment={details.open(EXPERIMENT_SECTION)} />
          ) : null}

          {/* ④ an entry that is not a set entry names itself and its field; the others load */}
          {indexState.index.rejected.map((item) => (
            <ProblemBox key={item.id} title={`Entry ${item.id} was rejected.`} detail={item.problem} />
          ))}
          {indexState.index.sets.length === 0 && indexState.index.rejected.length === 0
            ? <p className="training-note">The index lists no set.</p> : null}
          {setState.status === "loading" ? <p className="training-note" role="status">Loading {entry?.file} …</p> : null}
          {/* ③ a set that fails; the others are untouched */}
          {setState.status === "invalid" ? <ProblemBox title={`Set ${setId} cannot be read.`} detail={setState.problem} /> : null}

          {activeAirportCode && entry !== null ? (
            <TrainingFlightSession airport={activeAirportCode} sample={sample} entry={entry} details={details} intent={intentA} />
          ) : null}
        </>
      ) : null}
      </>}
    </section>
  );
}
