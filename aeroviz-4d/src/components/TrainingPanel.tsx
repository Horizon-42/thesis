/**
 * TrainingPanel.tsx
 * -----------------
 * The Training task's left-dock panel: the intermediate results of the two-tier
 * plan's stage B — the instruction words read off a real track (the "sentence"),
 * the track those words alone describe, and later the sentence a trained prior
 * says. Design: `aeroviz-4d/docs/36-2026-09-20-training-module.zh.md`.
 *
 * T4a: the panel now READS the export. It owns the only fetch and publishes the
 * selected flight through `trainingSelection`, which the full-width sentence bar
 * draws — the bar is a sibling of the dock, not a child, so a second fetch there
 * would load the same megabytes twice and could show a different flight.
 *
 * The three states of design §4.5 are all here and all name what failed:
 *   ① no `index.json` at all         → the path, the command, the restart (AV5)
 *   ② no `prior-generated` set       → the second layer has no artefact until B3′
 *   ③ one set's `sample.json` is bad → THAT set alone reports, by field name
 *
 * ③ is a deliberate divergence from the comparison picker's `.every(...)`, which
 * empties an airport over one bad entry (AV6): a half-written export is normal in
 * a development view, so a bad set must not take the good ones down with it.
 */

import { useEffect, useMemo, useState } from "react";
import { useApp } from "../context/AppContext";
import { isMissingJsonAsset } from "../utils/fetchJson";
import {
  fetchTrainingIndex,
  fetchTrainingSample,
  trainingIndexPath,
  trainingWordBandLabel,
  trainingWordCounts,
  TRAINING_KINDS,
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
  | { status: "loading" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; sample: TrainingSample };

const SHA_SHOWN = 12;

function EmptyState({ airport }: { airport: string }) {
  return (
    <div className="training-empty" role="status">
      <p className="training-empty-title">No Training export for {airport} yet.</p>

      <p className="training-empty-label">This panel reads</p>
      <code className="training-empty-path">{trainingIndexPath(airport)}</code>

      <p className="training-empty-label">Written by</p>
      <code className="training-empty-path">
        python run_ts.py instruction_sample_export --vocabulary &lt;…/vocabulary_tau10/instruction_vocabulary.json&gt; --executor &lt;…/checkpoint.pt&gt; --out aeroviz-4d/public/data/airports/{airport}/training/…
      </code>

      <p className="training-empty-note">
        After the first export, restart the dev server — vite does not watch{" "}
        <code>public/data</code>, so a directory created after boot is served as
        the SPA fallback instead of JSON. Kill the <code>vite</code> node
        process, not the <code>npm run dev</code> wrapper.
      </p>
    </div>
  );
}

export default function TrainingPanel() {
  const { activeAirportCode, setTrainingSelection } = useApp();
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
        if (!parsed.ok) setIndexState({ status: "invalid", problem: parsed.problem });
        else {
          setIndexState({ status: "ready", index: parsed.value });
          setSetId(parsed.value.sets[0]?.id ?? null);
        }
      })
      .catch((error: unknown) => {
        if (!live) return;
        // "Not exported yet" and "exported but broken" are different answers and
        // get different screens; `isMissingJsonAsset` also catches vite's SPA
        // fallback HTML, which is what a 404 under `public/data` looks like.
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

  // Keep the selection on the same flight across a set reload when it is still
  // there; otherwise fall to the first, so the bar is never blank beside a list.
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

  // ── publish what the sentence bar draws ───────────────────────────────────
  useEffect(() => {
    const flight = sample?.flights.find((item) => item.flightKey === flightKey) ?? null;
    setTrainingSelection(
      flight && sample
        ? { vocabulary: sample.vocabulary, flight, geometry: sample.geometry }
        : null,
    );
  }, [sample, flightKey, setTrainingSelection]);

  useEffect(() => () => setTrainingSelection(null), [setTrainingSelection]);

  const counts = sample ? trainingWordCounts(sample.vocabulary) : null;
  const hasPrior =
    indexState.status === "ready" &&
    indexState.index.sets.some((item) => item.kind === "prior-generated");

  return (
    <section className="training-panel" aria-label="Training">
      <header className="training-panel-header">
        <h2>Training</h2>
        {/* The prose folds away by default. The sentence bar is docked across the
            bottom and the flight list is what it would cover, so everything that
            is read ONCE — what the module is, which vocabulary, which shas — sits
            behind this, and the list keeps the height. */}
        <button
          type="button"
          className="training-about-toggle"
          aria-expanded={aboutOpen}
          aria-label={aboutOpen ? "Hide what this panel shows" : "What does this panel show?"}
          onClick={() => setAboutOpen((open) => !open)}
        >
          {aboutOpen ? "\u00d7" : "\u24d8"}
        </button>
      </header>

      {aboutOpen ? (
        <p className="training-panel-lede">
          The instruction words read off each arrival, the track those words alone
          describe, and — once the second layer is trained — the sentence it says.
        </p>
      ) : null}

      {indexState.status === "loading" ? (
        <p className="training-note" role="status">
          Reading {trainingIndexPath(airport)} …
        </p>
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
              <span>Sample set</span>
              <select value={setId ?? ""} onChange={(event) => setSetId(event.target.value)}>
                {indexState.index.sets.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.title} ({item.flights} flights)
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <p className="training-note">{entry?.title ?? "No sample set in this manifest."}</p>
          )}

          {/* AV6 in reverse: a rejected entry names itself and its field, and the
              sets beside it still load. */}
          {indexState.index.rejected.map((item) => (
            <div className="training-problem" role="alert" key={item.id}>
              <p className="training-empty-title">Set {item.id} was rejected.</p>
              <p className="training-problem-detail">{item.problem}</p>
            </div>
          ))}

          {/* ② the second layer has no artefact yet */}
          {!hasPrior && aboutOpen ? (
            <p className="training-note">
              No prior-generated set for {airport}: the second layer's own
              sentences arrive with stage B3′. The read-back set below is what the
              vocabulary makes of the observed tracks.
            </p>
          ) : null}

          {sampleState.status === "loading" ? (
            <p className="training-note" role="status">Loading {entry?.file} …</p>
          ) : null}

          {/* ③ one set is bad; the others are untouched */}
          {sampleState.status === "invalid" ? (
            <div className="training-problem" role="alert">
              <p className="training-empty-title">Set {setId} cannot be read.</p>
              <p className="training-problem-detail">{sampleState.problem}</p>
            </div>
          ) : null}

          {sample && counts ? (
            <>
              {aboutOpen ? (
              <dl className="training-vocabulary">
                <div>
                  <dt>Reading rule</dt>
                  <dd>{sample.vocabulary.readingRule}</dd>
                </div>
                <div>
                  <dt>Vocabulary sha</dt>
                  <dd>{sample.vocabulary.sha256.slice(0, SHA_SHOWN)}…</dd>
                </div>
                <div>
                  {/* The runway classes are NOT inside the spec's sha, so the two
                      shas are shown separately: same spec, different runway list
                      is a real possibility (design §4.4-2). */}
                  <dt>Runway sha</dt>
                  <dd>{sample.vocabulary.runwaySha256.slice(0, SHA_SHOWN)}…</dd>
                </div>
                <div>
                  <dt>Runway words</dt>
                  <dd>{sample.vocabulary.runwayIdents.join(" ")}</dd>
                </div>
                {/* The two kinds that carry a tolerance. A word here means "stay
                    inside this band", so a panel that listed only the centres
                    would be stating half of what the vocabulary says. */}
                <div>
                  {/* How the labeller cut the profile, not a word count — the
                      vertical words are the slopes of these segments. */}
                  <dt>Vertical segments</dt>
                  <dd>{sample.vocabulary.verticalSegments} per approach</dd>
                </div>
                <div>
                  <dt>Vertical words</dt>
                  <dd>
                    {sample.vocabulary.verticalModesDeg
                      .map((_, word) => trainingWordBandLabel(sample.vocabulary, "vertical", word))
                      .join(" · ")}
                  </dd>
                </div>
                <div>
                  <dt>Speed words</dt>
                  <dd>
                    {sample.vocabulary.speedCentresMps[0]}…
                    {sample.vocabulary.speedCentresMps[sample.vocabulary.speedCentresMps.length - 1]} m/s,
                    {" "}±{(sample.vocabulary.speedToleranceFraction * 100).toFixed(0)} %
                  </dd>
                </div>
                <div>
                  <dt>Words per kind</dt>
                  <dd>{TRAINING_KINDS.map((kind) => `${kind} ${counts[kind]}`).join(" · ")}</dd>
                </div>
                {/* The flown tracks' assumptions. The exporter writes them BECAUSE
                    a view shows them: an approximation nobody can see stated is
                    worse than none, and "flown by rule" means nothing until the
                    rule's numbers are on screen. */}
                <div>
                  <dt>Flown by</dt>
                  <dd>{sample.geometry.method}, {sample.geometry.dtS} s steps</dd>
                </div>
                <div>
                  <dt>Bank · angle limits · accel</dt>
                  <dd>
                    {sample.geometry.bankDeg}° ·{" "}
                    −{sample.geometry.descentMaxDeg}°/+{sample.geometry.climbMaxDeg}° ·{" "}
                    {sample.geometry.accelMaxMps2} m/s²
                  </dd>
                </div>
                {/* Where the corridor on screen comes from, and the one thing a
                    reader would otherwise assume wrongly: the two bands are
                    separate, not the joint envelope (V32). */}
                <div>
                  <dt>Bands</dt>
                  <dd>
                    {sample.geometry.verticalBandFrom} · {sample.geometry.speedBandFrom} ·{" "}
                    {sample.geometry.bandsAreJoint
                      ? "drawn jointly"
                      : "one kind at a time, not the joint envelope"}
                  </dd>
                </div>
                {/* The fan closes onto this floor, so its widest point is before
                    the threshold rather than at it — the executor's doing, not
                    the vocabulary's (V34). */}
                <div>
                  <dt>Levels at</dt>
                  <dd>
                    {sample.geometry.heightFloorM} m above the threshold
                    {sample.geometry.verticalIsCommandedAngle
                      ? "; the vertical word is the commanded angle, so nothing stops it on its own"
                      : ""}
                  </dd>
                </div>
                <div>
                  <dt>Not modelled</dt>
                  <dd>
                    {[
                      sample.geometry.windModelled ? null : "wind",
                      sample.geometry.aircraftTypeModelled ? null : "aircraft type",
                    ]
                      .filter(Boolean)
                      .join(" · ") || "—"}
                  </dd>
                </div>
                <div>
                  <dt>Starts / stops</dt>
                  <dd>{sample.geometry.startsAt}; {sample.geometry.stopRule}</dd>
                </div>
                {/* Every constant above is imported from these files rather than
                    typed into the kinematics — which is only worth saying if the
                    files are named where the numbers are shown. */}
                <div>
                  <dt>Constants from</dt>
                  <dd>{sample.geometry.constantsFrom.join(" · ")}</dd>
                </div>
              </dl>
              ) : null}

              <p className="training-note">
                {sample.flights.length} flights, drawn from the hand check's own
                pages so the two views show the same aircraft.
                {/* The manifest states a count too. They are written by one run of
                    the exporter, so a disagreement means the two files came from
                    different runs — which is worth saying, not smoothing over. */}
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
                        {flight.sentence.eventTimesS.length} events
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
