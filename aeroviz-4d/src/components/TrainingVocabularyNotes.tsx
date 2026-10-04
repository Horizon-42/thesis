/**
 * TrainingVocabularyNotes.tsx
 * ---------------------------
 * The vocabulary of the open set, read once: its reading rule, the five columns and every number a word's envelope is
 * drawn with — the one place in the Training views that states them (the sentence bar, the windows and the legend name
 * the envelopes and point here). On the details page's "Vocabulary" section (`training/TrainingDetails`).
 */

import { TRAINING_COLUMNS, type TrainingSample } from "../data/trainingSample";
import { shortSha } from "../data/trainingText";

export default function TrainingVocabularyNotes({ sample }: { sample: TrainingSample }) {
  const { vocabulary, candidates, source } = sample;
  const levels = vocabulary.altitudeLevelsM;
  const tolerances = vocabulary.altitudeTolerancesM;
  const rows: Array<[string, string]> = [
    ["Vocabulary", `${vocabulary.readingRule} · spec ${shortSha(source.specSha256)} · executor spec ${shortSha(source.executorSpecSha256)}` +
      ` · five columns, ${TRAINING_COLUMNS.join(", ")}, each with "unchanged"; row 0 says every column`],
    ["Runway", `${candidates.map((candidate) => candidate.ident).join(" ")}, or go-around (candidates sha ${shortSha(sample.candidatesSha256)})`],
    ["Heading", `relative to the course of the runway in force, on a ${vocabulary.headingStepDeg}° grid: a word says where the track ` +
      `is ${vocabulary.headingLeadS} s after it is said, and from then on the track stays within ±${vocabulary.headingToleranceDeg}° of it`],
    ["Altitude", `a level above the airport elevation E, ${levels[0]}…${levels[levels.length - 1]} m in ${levels.length} classes (tube ` +
      `±${Math.min(...tolerances)}…${Math.max(...tolerances)} m), or "no level-off" (class ${vocabulary.noLevelOff})`],
    ["Angle", vocabulary.angleClasses.map((angle) => `${angle.name} ${angle.nominalDeg}° (${angle.lowDeg}…${angle.highDeg}°)`).join(" · ")],
    ["Speed", `ground speed ${vocabulary.speed.minMps} m/s in steps of ${vocabulary.speed.stepMps} m/s (${vocabulary.speed.levels} levels), ` +
      `or "unspecified" (class ${vocabulary.speed.unspecified}); band ±${vocabulary.speed.toleranceMps} m/s`],
    ["Closed loop", `the executor is told the words from the first predicted step, one row every Δ = ` +
      `${vocabulary.rowIntervalsS.join(", ")} s; the reading adds a word (a correction) when the flown path is more than ` +
      `${vocabulary.closedLoopLateralM} m lateral or ${vocabulary.closedLoopVerticalM} m vertical from the observed one`],
    ["Read on", `the 2 s rows of the data (${vocabulary.stepS} s); heights MSL, the 3D scene's ellipsoid heights add each flight's runway's HAE − MSL`],
  ];
  return (
    <dl className="training-vocabulary">
      {rows.map(([term, detail]) => (
        <div key={term}>
          <dt>{term}</dt>
          <dd>{detail}</dd>
        </div>
      ))}
    </dl>
  );
}
