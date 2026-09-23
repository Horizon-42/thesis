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
 * Two things do not exist yet, and the panel says so instead of drawing anything in their place:
 * the executor's replay of a sentence (the executor is being designed) and a sentence a trained
 * prior says (no prior is trained on this vocabulary).
 */

import { useEffect, useMemo, useState } from "react";
import { useApp } from "../context/AppContext";
import { isMissingJsonAsset } from "../utils/fetchJson";
import {
  TRAINING_CANDIDATE_COLOR,
  TRAINING_FUNNEL_COLOR,
  TRAINING_TUBE_COLOR,
  TRAINING_TURN_COLOR,
} from "../utils/trainingWordColors";
import {
  fetchTrainingIndex,
  fetchTrainingSample,
  trainingIndexPath,
  trainingSetRefusal,
  trainingVerdicts,
  TRAINING_COLUMNS,
  TRAINING_READING_RULE,
  TRAINING_SPEC_SHA256,
  type TrainingIndex,
  type TrainingSample,
  type TrainingSetEntry,
} from "../data/trainingSample";

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
  "python run_ts.py instruction_training_export --dir 4dTrajectory/outputs/POOLED/instruction_language/v1_20260923 " +
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

export default function TrainingPanel() {
  const { activeAirportCode, setTrainingSelection, trainingLayers, setTrainingLayer } = useApp();
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
          2 s step — with the region every word allows from where it was issued: the turn and
          the hold funnel of a heading, the capture corridor, the altitude tube, the speed band.
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

          <fieldset className="training-layers">
            <legend>Draw</legend>
            <label style={{ color: TRAINING_FUNNEL_COLOR }}>
              <input
                type="checkbox"
                checked={trainingLayers.lateral}
                onChange={(event) => setTrainingLayer("lateral", event.target.checked)}
              />
              lateral: turn regions, hold funnels, the capture corridor
            </label>
            <label style={{ color: TRAINING_TURN_COLOR }}>
              <input
                type="checkbox"
                checked={trainingLayers.turnPaths}
                onChange={(event) => setTrainingLayer("turnPaths", event.target.checked)}
              />
              turn paths: the fastest and the slowest turn
            </label>
            <label style={{ color: TRAINING_TUBE_COLOR }}>
              <input
                type="checkbox"
                checked={trainingLayers.vertical}
                onChange={(event) => setTrainingLayer("vertical", event.target.checked)}
              />
              vertical: the altitude tubes
            </label>
            <label style={{ color: TRAINING_CANDIDATE_COLOR }}>
              <input
                type="checkbox"
                checked={trainingLayers.candidates}
                onChange={(event) => setTrainingLayer("candidates", event.target.checked)}
              />
              every candidate runway
            </label>
            {/* TWO EMPTY SLOTS, on purpose: neither exists yet, and nothing is drawn in their
                place. They arrive with the executor (stage 3) and the prior (stage 5). */}
            <label className="training-slot" title="The executor is being designed (stage 3): nothing to replay yet.">
              <input type="checkbox" checked={false} disabled readOnly />
              executor replay — not built yet
            </label>
            <label className="training-slot" title={`No prior is trained on ${TRAINING_READING_RULE} yet (stage 5).`}>
              <input type="checkbox" checked={false} disabled readOnly />
              prior-generated sentence — no prior yet
            </label>
          </fieldset>

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
                      {sample.vocabulary.headingTargetsDeg[1] - sample.vocabulary.headingTargetsDeg[0]}° grid,
                      hold ±{sample.vocabulary.headingToleranceDeg}°, at most {sample.vocabulary.headingMaxTurnDeg}° per
                      word; a turn at {sample.vocabulary.turnRateMinDegS}–{sample.vocabulary.turnRateMaxDegS}°/s (the
                      lowest rate only for turns of {sample.vocabulary.turnRateMinFromDeg}° or more) and at most{" "}
                      {sample.vocabulary.turnBankMaxDeg}° of bank, begun up to {sample.vocabulary.turnStartDelayMaxS} s
                      after the word
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
