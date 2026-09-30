/**
 * trainingWordColors.ts
 * ---------------------
 * The Training views' one palette. The sentence bar, the read-back charts and the 3D scene must
 * agree about what "the heading colour" or "a heading band" looks like — two copies of a hex string is
 * how they stop agreeing.
 *
 * Only colours that CARRY MEANING live here; the panel's chrome is styled by class in
 * `index.css`. These are chosen per column or per envelope at render time.
 */

import type { TrainingColumn } from "../data/trainingSample";
import type { TrainingModelName, TrainingReplayKind } from "../data/trainingOverlays";
import type { TrainingOtherRole, TrainingWindowVerdict } from "../data/trainingTraffic";

/** The sentence bar's surface — every colour here is validated against it; `index.css` draws `.training-sentence-bar` in it
 *  at 0.94 opacity (MIRROR: CSS cannot import it). The bar shades with it. */
export const TRAINING_SURFACE_COLOR = "#0f131e";

/** One colour per column, in `TRAINING_COLUMNS` order.
 *
 *  Speed is a purple (2026-09-24): its amber was the selection's yellow to the eye (OKLab ΔE 3.4).
 *  Checked with the dataviz palette validator on the bar's surface (#0f131e): ΔE 39 from the
 *  selection, 23 from the verdict red, 18 from the raw grey, ≥ 10 from every other colour here under
 *  simulated colour blindness — a teal read 4.6 from the approach pink and 5.3 from the raw grey for
 *  a deuteranope. Altitude and angle are near each other (ΔE 4.3); their rows are labelled. */
export const TRAINING_COLUMN_COLOR: Record<TrainingColumn, string> = {
  runway: "#86efac",
  approach: "#f472b6",
  heading: "#7dd3fc",
  altitude: "#a5b4fc",
  angle: "#c4b5fd",
  speed: "#be76ff",
};

/** The SELECTED word (one column's, at the cursor), in ONE colour across every view: its band's
 *  stroke, its envelope's edge and the rows it is in force — the eye should be able to follow it
 *  from the bar to the charts to the scene. An envelope keeps its own hue as its fill. */
export const TRAINING_WORD_COLOR = "#facc15";

/** The smoothed signal the labeller read — the bright line every envelope is judged against. */
export const TRAINING_TRACE_COLOR = "#e2e8f0";
/** The raw rows behind it: what the aircraft did, before the reading's smoothing. */
export const TRAINING_RAW_COLOR = "#94a3b8";

/** A heading word's BAND: its target ± the heading tolerance over the rows it is judged on — on the heading chart, and
 *  those rows of the ground trace in 3D (instruction-v3). The blue of the heading family, deeper than the column's. */
export const TRAINING_HEADING_BAND_COLOR = "#38bdf8";
/** The CAPTURE TURN: its rows, from the clearance onto the course. */
export const TRAINING_CAPTURE_TURN_COLOR = "#fb923c";
/** The CAPTURE CORRIDOR from the capture to the threshold (and the course band after it). */
export const TRAINING_CORRIDOR_COLOR = "#4ade80";
/** An altitude word's TUBE — the altitude column's own colour. */
export const TRAINING_TUBE_COLOR = TRAINING_COLUMN_COLOR.altitude;
/** A speed word's transition and band — the speed column's own colour. */
export const TRAINING_SPEED_COLOR = TRAINING_COLUMN_COLOR.speed;

/** How opaque each envelope is in the 3D scene at rest, and a fill when it is the selected word's — one table, so the
 *  legend's swatches are the scene's. The heading bands and the capture turn are lines on the ground. */
export const TRAINING_ENVELOPE_ALPHA = {
  headingBand: 0.85, captureTurn: 0.85, corridor: 0.3, tube: 0.28, selected: 0.45,
} as const;

/** Every candidate runway (the pointer's choices) — and the one pointed at. */
export const TRAINING_CANDIDATE_COLOR = "#94a3b8";
export const TRAINING_DESIGNATED_COLOR = TRAINING_COLUMN_COLOR.runway;

/** The signal where it sits OUTSIDE the envelope of the word in force — the one thing on these
 *  views that is a disagreement rather than a drawing. */
export const TRAINING_OUTSIDE_COLOR = "#f87171";

/** THE EXECUTOR's flown track and its verdicts "inside" (the replay of the truth sentence, `trainingOverlays.ts`).
 *  Teal (2026-09-24): OKLab ΔE on the palette above is ≥ 11.9 from every colour here (nearest: the raw grey 11.9,
 *  the heading band's blue 12.3, the corridor green 13.8) and 53 from the bar's surface (#0f131e); every candidate hue
 *  between them sat under 10 from one of the columns. */
export const TRAINING_EXECUTOR_COLOR = "#14b8a6";

/** THE REPLAY'S VERDICT ON A FLIGHT (`replayVerdict`) where it is said in one colour — the flight list, the sentence bar's
 *  header: the replay's teal when it landed clean, amber when it landed flawed (words outside, or its track refused), the
 *  outside red when it did not land (the user, 2026-09-28: a landing with two words out is not a flight that failed).
 *  The amber #f59e0b, OKLab ΔE on the bar's surface (#0f131e): 14.6 from the outside red (≥ 8.6 under simulated colour
 *  blindness), 24.9 from the teal (≥ 14.1), 11.2 from the selection's yellow (≥ 9.1); contrast 8.6:1, so it carries
 *  text. 4.2 from the capture turn's orange, which is only drawn — in the charts and in 3D — never text. */
export const TRAINING_REPLAY_COLOR: Record<TrainingReplayKind, string> = {
  clean: TRAINING_EXECUTOR_COLOR,
  flawed: "#f59e0b",
  "not landed": TRAINING_OUTSIDE_COLOR,
};

/** THE EXECUTOR FLOWN LIVE: the selected word's segment, flown by the backend when it is selected
 *  (`trainingAutopilot.ts`) — the replay's teal would read as the precomputed replay. Royal blue (2026-09-25): the dataviz
 *  validator on the bar's surface (#0f131e) puts it ≥ 21.8 OKLab ΔE from every colour above under normal vision (nearest:
 *  the speed purple) and ≥ 9.3 under simulated colour blindness, contrast ≥ 3:1; the violet #7c3aed passed too (17.3 /
 *  12.5) but sits in the speed column's hue. */
export const TRAINING_AUTOPILOT_COLOR = "#2563eb";
/** A FAILURE: where a flight failed, drawn as the answer to the question it was flown or read for — louder than the
 *  per-row `TRAINING_OUTSIDE_COLOR`: the live executor's segment whose word flew outside its envelope (the user,
 *  2026-09-25), a model's flight that did not land (its end on the sentence bar), a loss of separation and where the judge
 *  ended an aircraft (the user, 2026-09-30: the window strip's pale red was not seen). The validator puts it ΔE 41.6 from
 *  the autopilot blue (29.0 under CVD); contrast 5.0:1. */
export const TRAINING_FAILURE_COLOR = "#ff2d2d";
/** The live executor's segment when the selected word flew OUTSIDE its envelope: the whole flown line turns red. */
export const TRAINING_AUTOPILOT_OUTSIDE_COLOR = TRAINING_FAILURE_COLOR;

/** THE PRIOR'S OWN SENTENCES (`trainingOverlays.TrainingGenerationOverlay`): each model in one colour — its tab in the
 *  sentence bar, its flown tracks in 3D, its samples in the flight list — by NAME (`TRAINING_MODEL_NAMES`): every round
 *  of a model shares its colour, the round is said in words. The truth is the observed track's own near-white
 *  (`TRAINING_TRACE_COLOR`), always drawn.
 *  The palette above leaves two hue regions free (2026-09-25, the dataviz validator's OKLab ΔE on the bar's surface
 *  #0f131e): base's fuchsia is ≥ 14.4 from every colour drawn in 3D (nearest: the approach pink, a marker there) and 9.2
 *  from the speed purple, which draws nothing in 3D; landing's lime is ≥ 9.5 from every colour (nearest: the corridor's
 *  translucent fill on the ground, and the runway green), contrast ≥ 5:1 both. augmented's raspberry (2026-09-26, the
 *  best of a search over hue × saturation × lightness against every colour above) is ≥ 17.7 from every one of them under
 *  normal vision (nearest: the live executor's outside red; 18.5 from base's fuchsia) and ≥ 11.6 under simulated colour
 *  blindness, but its contrast is only 3.3:1 — enough for a mark (lines, swatches, borders, the flight's end time on the
 *  bar's axis, which the user asked to see in the model's colour), too little for running text: names and counts stay in
 *  the text colour beside a swatch. traffic's azure (the user, 2026-09-30: fresher than the olive gold it replaces; the
 *  best of a search over the green-to-blue hues at contrast ≥ 4.5:1 against every colour above) is ≥ 11.5 OKLab ΔE from
 *  every one of them (nearest: the heading band's blue 11.5, the autopilot blue 12.4), contrast 5.8:1. Two models are never
 *  drawn together (the bar reads one at a time). */
export const TRAINING_MODEL_COLOR: Record<TrainingModelName, string> = {
  base: "#d946ef",
  landing: "#a3e635",
  augmented: "#b82e7a",
  traffic: "#2b93ee",
};

/** How opaque a model's samples other than the one read are drawn in 3D (thin), and in the legend. */
export const TRAINING_OTHER_SAMPLE_ALPHA = 0.35;

/** A model's colour, by its name. */
export function trainingModelColour(model: { name: TrainingModelName }): string {
  return TRAINING_MODEL_COLOR[model.name];
}

/** A multi-aircraft window's verdict (`trainingTraffic.windowVerdict`), in the replay's three colours — the same reading of
 *  "clean / something wrong / did not get there": every commanded aircraft landed with no loss of separation, none lost
 *  separation but one did not land, one lost separation. */
export const TRAINING_WINDOW_VERDICT_COLOR = {
  clean: TRAINING_REPLAY_COLOR.clean,
  short: TRAINING_REPLAY_COLOR.flawed,
  lost: TRAINING_REPLAY_COLOR["not landed"],
} as const satisfies Record<TrainingWindowVerdict, string>;

/** The aircraft of a window the model does not command, replayed as recorded: the raw track's slate for an arrival with a
 *  sentence, darker for a background arrival without one — neither is a hue of the words, the models or the verdicts. */
export const TRAINING_OTHER_AIRCRAFT_COLOR = {
  replayed: "#94a3b8",
  background: "#64748b",
} as const satisfies Record<TrainingOtherRole, string>;
/** …and how opaque their tracks are in 3D (and in the legend): faded under the commanded aircraft's, which are opaque. */
export const TRAINING_OTHER_AIRCRAFT_ALPHA = {
  replayed: 0.6,
  background: 0.45,
} as const satisfies Record<TrainingOtherRole, number>;

/** The pair of aircraft under their minimum at the cursor, drawn between them in 3D and on the window strip, and where
 *  the judge ended an aircraft for it. */
export const TRAINING_LOSS_COLOR = TRAINING_FAILURE_COLOR;
