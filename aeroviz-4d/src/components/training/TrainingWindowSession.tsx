/**
 * TrainingWindowSession.tsx
 * -------------------------
 * The Training panel over a WINDOW SET (`traffic-windows`, the Training module §4.11): its multi-aircraft windows. A
 * window is picked in the list, then one of its commanded aircraft — the aircraft ON SCREEN, a set flight the sentence
 * bar, the read-back window and the flight's 3D layers draw as any flight (`trainingWindowSelection`, on the window's
 * clock, so another aircraft of the window comes up at the same moment). The models' sentences in the windows
 * (`window-generation`) are downloaded here and chosen in the sentence bar as any model's: each is projected onto the
 * aircraft on screen (`windowGenerationView`); a sample is the whole window. The window itself — every aircraft, the
 * losses, the landings — is published for the window strip and 3D (`trainingWindow`).
 *
 * The list says how each window went under the sentence read: a mark per sample of the model read (one for the record),
 * teal where every commanded aircraft landed with no loss of separation, amber where none lost separation but one did not
 * land, red where one did. The frontend judges nothing: the losses and landings are the exporter's judge's.
 */

import { useEffect, useMemo, useState } from "react";
import { useApp } from "../../context/AppContext";
import { useTrainingWindowOverlays, type OverlaysManifestState } from "../../hooks/useTrainingOverlays";
import TrainingVocabularyNotes from "../TrainingVocabularyNotes";
import ProblemBox from "./ProblemBox";
import TrainingDetails, { type TrainingDetailsSection } from "./TrainingDetails";
import { NotesList } from "./NotesToggle";
import { DetailsLink, LAYER_SWITCHES, LayerSwitches, manifestAbsence, noneReadable, type DetailsPage } from "./PanelParts";
import { TRAINING_OTHER_AIRCRAFT_COLOR, TRAINING_WINDOW_VERDICT_COLOR, trainingModelColour } from "../../utils/trainingWordColors";
import { trainingModelLabel, trainingOverlaysPath } from "../../data/trainingOverlays";
import { TRAINING_OUTCOME_TAG, trainingModelText } from "../../data/trainingText";
import {
  aircraftFate,
  trainingWindowSelection,
  windowGenerationView,
  windowReading,
  windowSetCounts,
  windowVerdict,
  type TrainingTrafficSet,
  type TrainingWindow,
  type TrainingWindowGenerationOverlay,
  type TrainingWindowReading,
  type TrainingWindowVerdict,
} from "../../data/trainingTraffic";
import type { TrainingSetEntry } from "../../data/trainingSample";

/** How the list and the strip word a window's verdict (`windowVerdict`). */
export const TRAINING_WINDOW_VERDICT_TEXT: Record<TrainingWindowVerdict, string> = {
  clean: "every commanded aircraft landed, no loss of separation",
  short: "no loss of separation, but not every commanded aircraft landed",
  lost: "an aircraft lost separation (the judge ended it)",
};

const TRAFFIC_TITLE = "The windows";

/** Why no model's sentences are read in the windows: none published, or not loaded yet, or not readable. */
function modelsAbsence(manifest: OverlaysManifestState, items: Array<{ load: { status: string } }>): string {
  const unread = manifestAbsence(manifest);
  if (unread !== null) return unread;
  if (items.length === 0) return noneReadable(manifest);
  return items.every(({ load }) => load.status === "invalid") ? "cannot be read" : "loading …";
}

/** "08-26 13:40Z": a window's opening, as the list names it. */
export function windowOpening(window: TrainingWindow): string {
  return `${window.opensUtc.slice(5, 10)} ${window.opensUtc.slice(11, 16)}Z`;
}

/** "3rd": a place in a landing order. */
function ordinal(place: number): string {
  const tens = place % 100;
  const suffix = tens >= 11 && tens <= 13 ? "th" : ["th", "st", "nd", "rd"][place % 10] ?? "th";
  return `${place}${suffix}`;
}

/** What became of a commanded aircraft under the reading, in a few words: "lost separation · DAL12", "landed 2nd (recorded
 *  3rd)", or its own end. */
function fateText(window: TrainingWindow, reading: TrainingWindowReading, at: number): string {
  const aircraft = window.commanded[at];
  const fate = aircraftFate(window, reading, aircraft.flight.datasetId);
  const callsign = (id: string) => window.others.find((other) => other.datasetId === id)?.callsign
    ?? window.commanded.find((one) => one.flight.datasetId === id)?.flight.callsign ?? id;
  if (fate.ended !== null) return `lost separation · ${callsign(fate.ended.with)}`;
  if (fate.landed !== null) {
    return `landed ${ordinal(fate.landed)}` + (fate.recordedLanded === null || reading.model === null ? ""
      : ` (recorded ${ordinal(fate.recordedLanded)})`);
  }
  return reading.model === null ? "did not land" : TRAINING_OUTCOME_TAG[reading.model.sample.aircraft[at].own];
}

/** A window's marks under the model read: one per sample (one for the record), each in its verdict's colour. */
function WindowMarks({ window, overlay }: { window: TrainingWindow; overlay: TrainingWindowGenerationOverlay | null }) {
  const readings = overlay === null ? [window.recorded] : overlay.windows[window.index].samples;
  const verdicts = readings.map((reading) => windowVerdict(window, reading));
  return (
    <span className="training-flight-samples"
      title={(overlay === null ? "as recorded: " : `${trainingModelText(overlay.model)}, sample by sample: `) +
        verdicts.map((verdict, k) => `${overlay === null ? "" : `#${k + 1} `}${TRAINING_WINDOW_VERDICT_TEXT[verdict]}`).join("; ")}>
      {verdicts.map((verdict, k) => <span key={k} style={{ color: TRAINING_WINDOW_VERDICT_COLOR[verdict] }}>●</span>)}
    </span>
  );
}

/** The set's windows read by each model and as recorded: the commanded aircraft counted (every sample), landed and not
 *  ended, and ended for a loss of separation under VISUAL (and IFR, beside it). */
function TrafficReadout({ set, overlays }: { set: TrainingTrafficSet; overlays: TrainingWindowGenerationOverlay[] }) {
  const share = (part: number, whole: number) => `${((part / whole) * 100).toFixed(1)} %`;
  const rows = [
    { key: "recorded", name: <>as recorded</>, counts: windowSetCounts(set, null) },
    ...overlays.map((overlay) => ({ key: overlay.overlayId, name: (
      <><span className="training-model-swatch" style={{ background: trainingModelColour(overlay.model) }} />
        {trainingModelText(overlay.model)}</>), counts: windowSetCounts(set, overlay) })),
  ];
  return (
    <>
      <p className="training-details-lede">
        {set.windows.length} windows, {set.flights.length} commanded flights · {set.cohort.split} · {set.cohort.drawnFrom}.
        A model's counts are over every sample: each sample commands every aircraft of its window together.
      </p>
      <table className="training-details-table">
        <thead>
          <tr><th scope="col">read as</th><th scope="col">aircraft</th><th scope="col">landed</th>
            <th scope="col" title="ended by the judge under VISUAL: the loop's reading">lost separation</th>
            <th scope="col" title="the same paths judged afterwards under IFR">IFR</th></tr>
        </thead>
        <tbody>
          {rows.map(({ key, name, counts }) => (
            <tr key={key}>
              <th scope="row">{name}</th><td>{counts.aircraft}</td><td>{share(counts.landed, counts.aircraft)}</td>
              <td>{share(counts.lost, counts.aircraft)}</td><td>{share(counts.lostIfr, counts.aircraft)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {overlays.some((overlay) => overlay.readout !== null) ? (
        <>
          <h4 className="training-details-subhead">The formal window readouts (other draws, the same windows' draw)</h4>
          <NotesList items={overlays.flatMap((overlay) => (overlay.readout === null ? [] : [{
            key: overlay.overlayId, name: <>{trainingModelText(overlay.model)}</>,
            text: `${overlay.readout.directory} (${overlay.readout.windowsPerAirport} windows an airport, ${overlay.readout.samples} ` +
              `samples each): in the model's scene ${share(overlay.readout.scene.all.lostSeparation, 1)} lost ` +
              `separation (IFR ${share(overlay.readout.scene.all.lostSeparationIfr, 1)}), ${share(overlay.readout.scene.all.landed, 1)} ` +
              `landed over ${overlay.readout.scene.all.aircraft} aircraft; recorded ${share(overlay.readout.recorded.all.lostSeparation, 1)}`,
          }]))} />
        </>
      ) : null}
    </>
  );
}

export default function TrainingWindowSession({ airport, traffic, entry, details }: {
  airport: string; traffic: TrainingTrafficSet; entry: TrainingSetEntry | null; details: DetailsPage;
}) {
  const { setTrainingSelection, setTrainingGenerations, setTrainingWindow, trainingSource } = useApp();
  const overlays = useTrainingWindowOverlays(airport, traffic);
  const loaded = useMemo(() => overlays.windows.flatMap(({ load }) => (load.status === "ready" ? [load.overlay] : [])),
    [overlays.windows]);
  const [windowIndex, setWindowIndex] = useState<number>(0);
  const [focus, setFocus] = useState<string | null>(null);
  const current = traffic.windows[Math.min(windowIndex, traffic.windows.length - 1)];
  const aircraft = current.commanded.find((one) => one.flight.datasetId === focus) ?? current.commanded[0];
  const readModel = loaded.find((overlay) => overlay.overlayId === trainingSource?.overlayId) ?? null;
  const view = useMemo(() => ({ set: traffic, window: current, overlays: loaded, focus: setFocus }), [traffic, current, loaded]);
  const reading = windowReading(view, trainingSource);

  // ── publish what the other views draw: the aircraft on screen, the models' sentences for it, the window ─────────
  useEffect(() => {
    setTrainingSelection(trainingWindowSelection(traffic, current, aircraft));
  }, [traffic, current, aircraft, setTrainingSelection]);
  useEffect(() => {
    setTrainingGenerations(loaded.map((overlay) => windowGenerationView(overlay, current, aircraft)));
  }, [loaded, current, aircraft, setTrainingGenerations]);
  useEffect(() => {
    setTrainingWindow(view);
  }, [view, setTrainingWindow]);
  useEffect(() => () => {
    setTrainingSelection(null);
    setTrainingGenerations([]);
    setTrainingWindow(null);
  }, [setTrainingSelection, setTrainingGenerations, setTrainingWindow]);

  const pick = (index: number) => {
    setWindowIndex(index);
    setFocus(null);
  };

  const sections: TrainingDetailsSection[] = [
    { id: "overview", title: "What this view shows", body: (
      <>
        <p className="training-details-lede">
          Multi-aircraft windows: 20 minutes of one airport, every arrival entering it with a sentence commanded by the model
          at once — each by its own executor, judged together for losses of separation — and every other aircraft there
          replayed as recorded. Pick a window, then one of its commanded aircraft: the sentence bar reads it as any flight.
          The Truth tab reads the window as recorded; a model's sample is the whole window's, its aircraft commanded
          together. The strip above the bar is the window's clock: every aircraft, its landing, the pairs under their
          minimum (VISUAL solid — the reading that ends an aircraft — IFR dashed).
        </p>
        <h4 className="training-details-subhead">What each switch draws</h4>
        <NotesList items={LAYER_SWITCHES.map(({ layer, colour, text, title }) => ({ key: layer, text: title, name: (
          <><span className="training-model-swatch" style={{ background: colour }} />{text}</>) }))} />
        <h4 className="training-details-subhead">How the list marks a window</h4>
        <NotesList items={(Object.keys(TRAINING_WINDOW_VERDICT_TEXT) as TrainingWindowVerdict[]).map((verdict) => ({
          key: verdict, text: TRAINING_WINDOW_VERDICT_TEXT[verdict],
          name: <span className="training-model-swatch" style={{ background: TRAINING_WINDOW_VERDICT_COLOR[verdict] }} /> }))} />
        <h4 className="training-details-subhead">The models published over this set</h4>
        {overlays.windows.length === 0 ? <p className="training-details-lede">None.</p> : (
          <NotesList items={overlays.windows.map(({ entry: listed }) => ({ key: listed.id, name: <code>{listed.id}</code>, text: listed.title }))} />
        )}
      </>
    ) },
    { id: "vocabulary", title: "Vocabulary", body: <TrainingVocabularyNotes sample={traffic} /> },
    { id: "traffic", title: TRAFFIC_TITLE, body: <TrafficReadout set={traffic} overlays={loaded} /> },
  ];
  const others = current.others.length;
  const background = current.others.filter((other) => other.role === "background").length;

  return (
    <>
      {details.shown !== null ? (
        <TrainingDetails context={[airport, traffic.setId, `${traffic.windows.length} windows`].join(" · ")}
          sections={sections} sectionId={details.shown.section} onSection={details.show} onClose={details.close}
          opener={details.shown.opener} />
      ) : null}

      <p className="training-note" title={`${entry?.title ?? traffic.setId} — ${traffic.cohort.drawnFrom}`}>
        {traffic.windows.length} windows · {traffic.cohort.split}
      </p>
      <ul className="training-flight-list training-window-list" aria-label="Windows">
        {traffic.windows.map((item) => (
          <li key={item.index}>
            <button type="button" className={item === current ? "active" : undefined} onClick={() => pick(item.index)}>
              <span className="training-flight-callsign">{windowOpening(item)}</span>
              <span className="training-flight-events">{item.commanded.length} by model · {item.others.length} others</span>
              <WindowMarks window={item} overlay={readModel} />
            </button>
          </li>
        ))}
      </ul>

      <h3 className="training-window-heading" title={`opens ${current.opensUtc}; ${others} other aircraft, ${background} of them ` +
        "background arrivals without a sentence"}>
        Window {windowOpening(current)} · aircraft
      </h3>
      <ul className="training-window-aircraft" aria-label="Aircraft of the window">
        {current.commanded.map((one, at) => (
          <li key={one.flight.datasetId}>
            <button type="button" className={one === aircraft ? "active" : undefined} onClick={() => setFocus(one.flight.datasetId)}
              title={`${one.flight.callsign} · ${one.flight.typecode ?? "type unknown"} · runway ${one.flight.runway} — commanded by ` +
                "the model; put it on screen"}>
              <span className="training-flight-callsign">{one.flight.callsign}</span>
              <span className="training-flight-runway">{one.flight.runway}</span>
              <span className="training-window-fate">{fateText(current, reading, at)}</span>
            </button>
          </li>
        ))}
        {current.others.map((other) => (
          <li key={other.datasetId} className="training-window-other" style={{ color: TRAINING_OTHER_AIRCRAFT_COLOR[other.role] }}
            title={`${other.callsign}: ${other.role === "replayed" ? "replayed as recorded" : "a background arrival without a sentence, "
              + "replayed as recorded"}${other.category === null ? "" : ` · wake category ${other.category}`}`}>
            <span className="training-flight-callsign">{other.callsign}</span>
            <span className="training-flight-stratum">{other.role}</span>
          </li>
        ))}
      </ul>

      <fieldset className="training-layers">
        <legend>Draw</legend>
        <LayerSwitches />
      </fieldset>

      {overlays.manifest.status === "invalid" ? (
        <ProblemBox title={`${trainingOverlaysPath(airport)} cannot be read.`} detail={overlays.manifest.problem} />
      ) : null}
      {overlays.manifest.status === "ready" ? overlays.manifest.overlays.rejected.map((item) => (
        <ProblemBox key={`overlay-${item.id}`} title={`Overlay ${item.id} was rejected.`} detail={item.problem} />
      )) : null}
      {overlays.windows.flatMap(({ entry: listed, load, retry }) => (load.status !== "invalid" ? [] : [
        <div key={`window-${listed.id}`}>
          <ProblemBox title={`Overlay ${listed.id} cannot be read.`} detail={load.problem} />
          <button type="button" className="training-sentence-readback-button" onClick={retry}>Retry</button>
        </div>,
      ]))}
      <ul className="training-details-links" aria-label="Readouts">
        <DetailsLink name="Windows" onOpen={details.open("traffic")}
          summary={[null, ...loaded].map((overlay) => {
            const counts = windowSetCounts(traffic, overlay);
            return `${overlay === null ? "recorded" : trainingModelLabel(overlay.model)} ${((counts.lost / counts.aircraft) * 100).toFixed(1)} %`;
          }).join(" · ") + ` lost separation${loaded.length > 0 ? "" : ` (models: ${modelsAbsence(overlays.manifest, overlays.windows)})`}`} />
      </ul>
    </>
  );
}
