/**
 * TrainingPanel.tsx
 * -----------------
 * The Training task's left-dock panel: which exported set, which flight, which envelopes are drawn — for the
 * instruction vocabulary (`TRAINING_READING_RULE`). Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`.
 *
 * It owns the manifest and the set's fetch, and opens the set by its KIND (`trainingSets.ts`): a read-back set's flights
 * one at a time (`TrainingFlightSession`) or a window set's multi-aircraft windows (`TrainingWindowSession`) — the
 * session publishes the flight on screen through `trainingSelection`, which the sentence bar, the read-back window and the
 * 3D layer draw. The dock keeps it mounted from its first visit on, only ``hidden`` in the other tasks
 * (`WorkbenchLeftDock`): its session — set, flight or window, overlays, live answer — outlives a task switch.
 *
 * The states all name what failed:
 *   ① no `index.json`                → the path, the command, the dev-server restart (AV5)
 *   ② a listed set this reader refuses → WHY, by name, from the manifest alone (no download):
 *                                       a superseded vocabulary, another spec, a prior set
 *   ③ a readable set that fails to parse → THAT set alone, with the field
 *   ④ an entry that is not a set entry  → greyed out on its own; the others still load (AV6)
 *
 * KEPT SHORT, TOP TO BOTTOM: the set, then the session — its list, the Draw switches (short labels, the full reading in
 * each tooltip), one line per readout, its conclusion. Everything longer — what the module is, the vocabulary's numbers,
 * the readouts' tables — is on the DETAILS PAGE (`TrainingDetails`), opened by the header's ⓘ (as every ⓘ in the module
 * opens its notes) or by a readout's line, on its section; the dock never unfolds it. The panel owns the page's state; the
 * session supplies its sections.
 *
 * And THE EXECUTOR, LIVE (`useTrainingAutopilot`, run here): a word picked in the sentence bar — its Fly button, or a band
 * clicked — is flown by the backend now, never read from an overlay; the bar says the answer (`TrainingAutopilotStatus`).
 */

import { useCallback, useEffect, useMemo, useState, type MouseEvent } from "react";
import { useApp } from "../context/AppContext";
import useTrainingAutopilot from "../hooks/useTrainingAutopilot";
import ProblemBox from "./training/ProblemBox";
import TrainingFlightSession from "./training/TrainingFlightSession";
import TrainingWindowSession from "./training/TrainingWindowSession";
import type { DetailsPage } from "./training/PanelParts";
import { isMissingJsonAsset } from "../utils/fetchJson";
import {
  fetchTrainingIndex,
  TRAINING_READBACK_SET_KIND,
  TRAINING_TRAFFIC_SET_KIND,
  trainingIndexPath,
  trainingSetRefusal,
  type TrainingIndex,
  type TrainingSetEntry,
} from "../data/trainingSample";
import { fetchTrainingSet, openable, type TrainingOpenSet } from "../data/trainingSets";

type IndexState =
  | { status: "loading" }
  | { status: "absent" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; index: TrainingIndex };

type SetState =
  | { status: "idle" }
  | { status: "refused"; problem: string }
  | { status: "loading" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; open: TrainingOpenSet };

/** The command that writes the export, as the empty state shows it. */
const TRAINING_EXPORT_COMMAND =
  "python run_ts.py instruction_training_export --dir 4dTrajectory/outputs/POOLED/instruction_language/<an instruction-v3 artefact> " +
  "--airports-root aeroviz-4d/public/data/airports --airport ";

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

  // ── the chosen set, opened by its kind ────────────────────────────────────
  useEffect(() => {
    if (!activeAirportCode || !entry) {
      setSetState({ status: "idle" });
      return;
    }
    const { kind, refusal } = openable(entry);
    if (kind === null) {
      setSetState({ status: "refused", problem: refusal });
      return;
    }
    let live = true;
    setSetState({ status: "loading" });
    fetchTrainingSet(activeAirportCode, kind, entry.file)
      .then((parsed) => {
        if (!live) return;
        if (parsed.ok) setSetState({ status: "ready", open: parsed.value });
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

  const open = setState.status === "ready" ? setState.open : null;
  // the session of the KIND chosen stays mounted while another set of that kind loads (its switches and choices per set
  // outlive a switch of sets); it reads nothing while none of its kind is open
  const chosenKind = entry === null ? null : openable(entry).kind;

  return (
    <section className="training-panel" aria-label="Learning" hidden={hidden}>
      <header className="training-panel-header">
        {/* the module's name (its tab's); what it shows is still training — "Training details" below */}
        <h2>Learning</h2>
        {/* Everything read ONCE — what the module is, the vocabulary, the readouts — is on the details page, so the
            list keeps the dock's height. */}
        <button type="button" className="training-details-open" aria-haspopup="dialog" aria-label="Training details"
          title="Training details" onClick={details.open("overview")} disabled={open === null}>
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
                  <option key={item.id} value={item.id}>
                    {item.id} · {item.kind === TRAINING_TRAFFIC_SET_KIND ? `${item.cohort.windows} windows` : `${item.flights} flights`}, {item.cohort.split}
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
          {setState.status === "refused" ? <ProblemBox title={`Set ${setId} is not read.`} detail={setState.problem} /> : null}
          {setState.status === "loading" ? <p className="training-note" role="status">Loading {entry?.file} …</p> : null}
          {/* ③ a readable set that fails; the others are untouched */}
          {setState.status === "invalid" ? <ProblemBox title={`Set ${setId} cannot be read.`} detail={setState.problem} /> : null}

          {chosenKind === TRAINING_READBACK_SET_KIND && activeAirportCode ? (
            <TrainingFlightSession airport={activeAirportCode}
              sample={open?.kind === TRAINING_READBACK_SET_KIND ? open.sample : null} entry={entry} details={details} />
          ) : null}
          {chosenKind === TRAINING_TRAFFIC_SET_KIND && activeAirportCode ? (
            <TrainingWindowSession airport={activeAirportCode}
              traffic={open?.kind === TRAINING_TRAFFIC_SET_KIND ? open.traffic : null} entry={entry} details={details} />
          ) : null}
        </>
      ) : null}
    </section>
  );
}
