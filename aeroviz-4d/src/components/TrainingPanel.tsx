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
 * It reads `training/index_v4.json` and nothing else: never the instruction-v3 view's `training/index.json`. The states
 * all name what failed:
 *   ① no index                      → the path, the command, the dev-server restart (AV5)
 *   ② an index of another schema    → the schema found and the one this view reads
 *   ③ a set that fails to parse     → THAT set alone, with the field (a sample of another schema: its name and the expected)
 *   ④ an entry that is not a set entry → named on its own; the others still load (AV6)
 *
 * KEPT SHORT, TOP TO BOTTOM: the set, then the session — its list, the Draw switches (short labels, the full reading in
 * each tooltip), one line per readout. Everything longer — what the module is, the vocabulary's numbers, the readouts'
 * tables — is on the DETAILS PAGE (`TrainingDetails`), opened by the header's ⓘ or by a readout's line, on its section;
 * the dock never unfolds it. The panel owns the page's state; the session supplies its sections.
 *
 * And THE EXECUTOR, LIVE (`useTrainingAutopilot`, run here): a word picked in the sentence bar — its Fly button, or a band
 * clicked — is flown by the backend now, never read from the file; the bar says the answer (`TrainingAutopilotStatus`).
 */

import { useCallback, useEffect, useMemo, useState, type MouseEvent } from "react";
import { useApp } from "../context/AppContext";
import useTrainingAutopilot from "../hooks/useTrainingAutopilot";
import ProblemBox from "./training/ProblemBox";
import TrainingFlightSession from "./training/TrainingFlightSession";
import type { DetailsPage } from "./training/PanelParts";
import { isMissingJsonAsset } from "../utils/fetchJson";
import {
  fetchTrainingIndex,
  fetchTrainingSample,
  trainingIndexPath,
  type TrainingIndex,
  type TrainingSample,
  type TrainingSetEntry,
} from "../data/trainingSample";

type IndexState =
  | { status: "loading" }
  | { status: "absent" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; index: TrainingIndex };

type SetState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; sample: TrainingSample };

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
  const [setState, setSetState] = useState<SetState>({ status: "idle" });
  /** The details page while it is open: its section, and the control that opened it (the focus goes back there). */
  const [shownDetails, setShownDetails] = useState<{ section: string; opener: HTMLElement } | null>(null);
  const show = useCallback((section: string) => setShownDetails((open) => (open === null ? null : { ...open, section })), []);
  const close = useCallback(() => setShownDetails(null), []);
  const details: DetailsPage = {
    // a panel hidden (another task) keeps no page open
    shown: hidden ? null : shownDetails,
    open: (section: string) => (event: MouseEvent<HTMLElement>) => setShownDetails({ section, opener: event.currentTarget }),
    show,
    close,
  };
  // a panel hidden (another task) closes its page: it does not come back unasked
  useEffect(() => {
    if (hidden) setShownDetails(null);
  }, [hidden]);
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

  // ── the chosen set ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!activeAirportCode || !entry) {
      setSetState({ status: "idle" });
      return;
    }
    let live = true;
    setSetState({ status: "loading" });
    fetchTrainingSample(activeAirportCode, entry.file)
      .then((parsed) => {
        if (!live) return;
        if (parsed.ok) setSetState({ status: "ready", sample: parsed.value });
        else setSetState({ status: "invalid", problem: parsed.problem });
      })
      .catch((error: unknown) => {
        if (!live) return;
        setSetState({ status: "invalid", problem: error instanceof Error ? error.message : String(error) });
      });
    return () => {
      live = false;
    };
  }, [activeAirportCode, entry]);

  const sample = setState.status === "ready" ? setState.sample : null;

  return (
    <section className="training-panel" aria-label="Learning" hidden={hidden}>
      <header className="training-panel-header">
        {/* the module's name (its tab's); what it shows is still training — "Training details" below */}
        <h2>Learning</h2>
        {/* Everything read ONCE — what the module is, the vocabulary, the readouts — is on the details page, so the
            list keeps the dock's height. */}
        <button type="button" className="training-details-open" aria-haspopup="dialog" aria-label="Training details"
          title="Training details" onClick={details.open("overview")} disabled={sample === null}>
          ⓘ
        </button>
      </header>

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
                  <option key={item.id} value={item.id}>{item.id} · {item.flights} flights</option>
                ))}
              </select>
            </label>
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
            <TrainingFlightSession airport={activeAirportCode} sample={sample} entry={entry} details={details} />
          ) : null}
        </>
      ) : null}
    </section>
  );
}
