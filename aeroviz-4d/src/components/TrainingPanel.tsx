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
import { TRAINING_FLOWN_COLOR, TRAINING_MODEL_COLOR, TRAINING_TARGET_COLOR } from "../utils/trainingWordColors";
import {
  fetchTrainingIndex,
  fetchTrainingSample,
  headingWordAt,
  trainingIndexPath,
  trainingWordBandLabel,
  trainingWordCounts,
  TRAINING_KINDS,
  TRAINING_READING_RULE,
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
        python run_ts.py instruction_sample_export --vocabulary &lt;…/vocabulary_box_v3_five_airports&gt; --out aeroviz-4d/public/data/airports/{airport}/training/…
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
        if (!parsed.ok) setIndexState({ status: "invalid", problem: parsed.problem });
        else {
          setIndexState({ status: "ready", index: parsed.value });
          // Open on a set this reader can READ. The manifest states each set's
          // reading rule, so which ones are current is known before any sample is
          // fetched — and a vocabulary bump leaves the superseded sets listed
          // (they say why when picked) rather than opening the panel on one.
          const readable = parsed.value.sets.find(
            (item) => item.readingRule === TRAINING_READING_RULE,
          );
          setSetId((readable ?? parsed.value.sets[0])?.id ?? null);
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
    // The manifest's kind decides what the sample must carry; the reader keys the
    // model's words on it rather than on whether the field happens to be there.
    fetchTrainingSample(activeAirportCode, entry.file, entry.kind)
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
        ? {
            vocabulary: sample.vocabulary,
            flight,
            reading: sample.reading,
            ...(sample.prior ? { prior: sample.prior } : {}),
          }
        : null,
    );
  }, [sample, flightKey, setTrainingSelection]);

  useEffect(() => () => setTrainingSelection(null), [setTrainingSelection]);

  const counts = sample ? trainingWordCounts(sample.vocabulary) : null;
  // Two different questions, and the switch wants the second one: does this
  // AIRPORT have a prior set anywhere (the note below), and does the set that is
  // OPEN carry a model (the switch). A switch enabled by a set nobody is looking
  // at toggles a line that is not there.
  const hasPrior =
    indexState.status === "ready" &&
    indexState.index.sets.some((item) => item.kind === "prior-generated");
  const openSetHasPrior = entry?.kind === "prior-generated";

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
          {/* THE SET IS THE EXPERIMENT. One set is one vocabulary read, or one
              vocabulary read plus one model's answers to it, so switching set is
              how two experiments are compared — and the option has to say WHICH
              model without anyone loading the set to find out. The manifest
              carries that (`entry.prior`); the sample is ten megabytes. */}
          {indexState.index.sets.length > 1 ? (
            <label className="training-field">
              <span>Experiment</span>
              <select value={setId ?? ""} onChange={(event) => setSetId(event.target.value)}>
                {indexState.index.sets.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.prior
                      ? `${item.id} · prior seed ${item.prior.seed} (${item.prior.sha256.slice(0, 8)}…)`
                      : `${item.id} · the words alone`}
                    {` · ${item.flights} flights, ${item.cohort.split}`}
                    {item.readingRule === TRAINING_READING_RULE
                      ? ""
                      : ` · ${item.readingRule}, superseded`}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <p className="training-note">{entry?.title ?? "No sample set in this manifest."}</p>
          )}
          {entry ? (
            <p className="training-note">
              {entry.title}
              {entry.prior
                ? ` · model ${entry.prior.sha256.slice(0, SHA_SHOWN)}… asked ${entry.prior.method}`
                : " · no model in this set: the words alone"}
            </p>
          ) : null}

          {/* WHAT IS DRAWN BESIDE THE TRACK. The observed track has no switch:
              it is the aircraft that was actually there and everything else is
              read against it. These two are the REGIONS a sentence allows — the
              wedge wall and the chain of boxes — and each switch reaches every
              view at once. */}
          <fieldset className="training-layers">
            <legend>Draw</legend>
            <label style={{ color: TRAINING_FLOWN_COLOR }}>
              <input
                type="checkbox"
                checked={trainingLayers.flown}
                onChange={(event) => setTrainingLayer("flown", event.target.checked)}
              />
              the envelope the words allow
            </label>
            <label
              style={{ color: openSetHasPrior ? TRAINING_MODEL_COLOR : undefined }}
              title={openSetHasPrior ? undefined : "This set carries no model's words."}
            >
              <input
                type="checkbox"
                checked={trainingLayers.model}
                disabled={!openSetHasPrior}
                onChange={(event) => setTrainingLayer("model", event.target.checked)}
              />
              the envelope the model said
            </label>
          </fieldset>
          {trainingLayers.flown ? (
            <p className="training-note" style={{ color: TRAINING_TARGET_COLOR }}>
              Cyan planes mark target altitudes above the runway threshold.
            </p>
          ) : null}

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
                {/* WHAT A WORD IS. Every one of these numbers is inside the
                    vocabulary's sha, because the tolerance IS the word here
                    (§2.6) — a panel that listed only the class counts would be
                    stating the shape of the vocabulary and none of its meaning. */}
                <div>
                  <dt>Redundancy</dt>
                  <dd>
                    ±{(sample.vocabulary.redundancyFraction * 100).toFixed(0)} %, with a
                    {" "}{sample.vocabulary.headingFloorDeg}° floor on the heading box
                  </dd>
                </div>
                <div>
                  <dt>Heading boxes</dt>
                  <dd>
                    {counts.heading} tiling{" "}
                    {sample.vocabulary.headingEdgesDeg[0]}…
                    {sample.vocabulary.headingEdgesDeg[sample.vocabulary.headingEdgesDeg.length - 1]}°
                    {" "}· on the course {trainingWordBandLabel(sample.vocabulary, "heading", headingWordAt(sample.vocabulary, 0))}
                  </dd>
                </div>
                <div>
                  <dt>Speed boxes</dt>
                  <dd>
                    {counts.speed} tiling {sample.vocabulary.speedEdgesMps[0]}…
                    {sample.vocabulary.speedEdgesMps[sample.vocabulary.speedEdgesMps.length - 1]} m/s
                  </dd>
                </div>
                <div>
                  {/* The one kind whose box is not an interval on a table: it is
                      a target plus the set that target is reachable from, so the
                      two angles and the ladder's ends are what say what it means. */}
                  <dt>Altitude ladder</dt>
                  <dd>
                    {counts.altitude} targets, {Math.round(sample.vocabulary.altitudeTargetsM[0])}…
                    {Math.round(sample.vocabulary.altitudeTargetsM[sample.vocabulary.altitudeTargetsM.length - 1])} m
                    {" "}above the threshold (h₀ {sample.vocabulary.altitudeH0M} m)
                  </dd>
                </div>
                <div>
                  <dt>Altitude wedge</dt>
                  <dd>
                    {sample.vocabulary.altitudeDownDeg}° above / {sample.vocabulary.altitudeUpDeg}° below,
                    {" "}on the remaining path — {sample.reading.altitudeForm}
                  </dd>
                </div>
                <div>
                  <dt>Read</dt>
                  <dd>{sample.reading.altitudeReading}</dd>
                </div>
                <div>
                  <dt>Hold</dt>
                  <dd>
                    {sample.vocabulary.durationBinS} s bins to {sample.vocabulary.durationMaxS} s,
                    {" "}written on the box it describes
                  </dd>
                </div>
                <div>
                  <dt>Words per kind</dt>
                  <dd>{TRAINING_KINDS.map((kind) => `${kind} ${counts[kind]}`).join(" · ")}</dd>
                </div>
                {/* HOW THE VERDICT WAS COMPUTED. The same track against the same
                    boxes is 100 % inside on the smoothed signals and 93 % on the
                    raw ones, so the signal is not a detail — it is the number. */}
                <div>
                  <dt>Signals judged</dt>
                  <dd>
                    course {sample.vocabulary.courseSmoothingS} s · speed and height{" "}
                    {sample.vocabulary.smoothingS} s · {sample.reading.windowRows}
                  </dd>
                </div>
                <div>
                  <dt>Remaining path to</dt>
                  <dd>{sample.reading.remainingPathTo}</dd>
                </div>
                <div>
                  {/* THE PRODUCER IS NOT IN THIS REPOSITORY. The box labeller ran
                      outside the tree, so the boxes here are a reconstruction
                      from the artefact's spec — which is why the containment
                      verdict is computed twice and compared. */}
                  <dt>Boxes rebuilt by</dt>
                  <dd>{sample.reading.producedBy}</dd>
                </div>
                {sample.prior ? (
                  <>
                    <div>
                      {/* WHICH model, and HOW it was asked. The method is the line
                          that stops the whole panel being read as free generation. */}
                      <dt>Model</dt>
                      <dd>
                        prior {sample.prior.sha256.slice(0, SHA_SHOWN)}… · seed {sample.prior.seed} ·
                        best epoch {sample.prior.bestEpoch} · {sample.prior.method}
                      </dd>
                    </div>
                    <div>
                      <dt>Seen these flights</dt>
                      <dd>
                        {sample.prior.trainedOnTheseFlights === 0
                          ? "never — none of the drawn flights was in its training split"
                          : `${sample.prior.trainedOnTheseFlights} of the drawn flights were in its training split`}
                      </dd>
                    </div>
                    {Object.entries(sample.prior.readout).map(([split, table]) => (
                      <div key={`readout-${split}`}>
                        <dt>Next-word NLL ({split})</dt>
                        <dd>
                          {TRAINING_KINDS.filter((kind) => kind in table)
                            .map((kind) => `${kind} ${table[kind].toFixed(2)}`)
                            .join(" · ")}
                        </dd>
                      </div>
                    ))}
                  </>
                ) : null}
                <div>
                  <dt>Constants from</dt>
                  <dd>{sample.reading.constantsFrom.join(" · ")}</dd>
                </div>
              </dl>
              ) : null}

              <p className="training-note">
                {/* WHERE THE DRAW CAME FROM, in the file's own words. It used to be
                    the hand check's pages (so the screen showed the aircraft a human
                    had marked); the first `segment-v12` artefacts carried no pages, so the
                    exporter draws and stratifies for itself and records the rule
                    here. A view that kept claiming the old provenance would be
                    citing a document that does not exist. */}
                {sample.flights.length} flights ·{" "}
                {entry ? entry.cohort.drawnFrom : "no manifest entry"}.
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
                        {flight.sentence.eventTimesS.length} boxes
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
