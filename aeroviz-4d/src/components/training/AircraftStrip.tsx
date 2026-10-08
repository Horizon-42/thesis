/**
 * AircraftStrip.tsx
 * -----------------
 * Stage D's aircraft strip (frontend §5.2): one line of the left panel that shows a window's commanded aircraft and chooses
 * one. A small square for each, in the order they join the window, filled with its outcome in the round on screen
 * (`trainingOutcomeColour`; neutral on the Labelled tab); a silent aircraft's square (D144) has one diagonal stroke; the
 * selected one is outlined. After the squares: the selected aircraft's callsign and "k of n". A click selects; the keys
 * `[` and `]` (the session's) select the one before and after. A window of one commanded aircraft shows no strip.
 */

import type { TrainingWindowAircraft } from "../../data/trainingWindowSample";
import { TRAINING_OUTCOME_TAG } from "../../data/trainingText";
import { trainingOutcomeColour } from "../../utils/trainingWordColors";

/** A square on the Labelled tab: no round's outcome. */
const NEUTRAL = "#475569";

export default function AircraftStrip({ aircraft, place, selected, onSelect }: {
  aircraft: TrainingWindowAircraft[];
  /** The round on screen (its place in each aircraft's rounds), or null on the Labelled tab. */
  place: number | null;
  selected: number;
  onSelect: (member: number) => void;
}) {
  if (aircraft.length < 2) return null;
  const chosen = aircraft[selected];
  return (
    <div className="training-aircraft-strip" role="group" aria-label="The window's commanded aircraft">
      {aircraft.map((item, member) => {
        const sentence = place === null ? null : item.rounds[place];
        const silent = sentence !== null && sentence.silentFromRow !== null;
        const said = sentence === null ? "the labelled sentence"
          : `${TRAINING_OUTCOME_TAG[sentence.outcome]} · r ${sentence.reward.toFixed(2)}${silent ? ` · silent from row ${sentence.silentFromRow}` : ""}`;
        return (
          <button key={item.datasetId} type="button" className="training-aircraft-square" aria-pressed={member === selected}
            aria-label={`${item.head.callsign}, ${member + 1} of ${aircraft.length}: ${said}`} title={`${item.head.callsign} (${item.datasetId}): ${said}`}
            data-silent={silent || undefined} onClick={() => onSelect(member)}>
            <svg viewBox="0 0 10 10" aria-hidden="true">
              <rect x={0.5} y={0.5} width={9} height={9} fill={sentence === null ? NEUTRAL : trainingOutcomeColour(sentence.outcome)}
                stroke={member === selected ? "#f8fafc" : "none"} strokeWidth={1} />
              {silent ? <line x1={1} y1={9} x2={9} y2={1} stroke="#0f172a" strokeWidth={1.5} /> : null}
            </svg>
          </button>
        );
      })}
      <span className="training-aircraft-strip-name">{chosen.head.callsign} · {selected + 1} of {aircraft.length}</span>
    </div>
  );
}
