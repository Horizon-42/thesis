/**
 * TrainingVocabularyNotes.tsx
 * ---------------------------
 * The vocabulary of the open set, read once: its spec and labeller, the columns and their classes, and every number a
 * word's envelope is drawn with — the one place in the Training views that states them (the sentence bar, the windows
 * and the legend name the envelopes and point here). Behind the panel's ⓘ.
 */

import { TRAINING_COLUMNS, TRAINING_SPEC_SHA256, type TrainingSample } from "../data/trainingSample";
import { shortSha } from "../data/trainingText";

export default function TrainingVocabularyNotes({ sample }: { sample: TrainingSample }) {
  const { vocabulary, candidates } = sample;
  const rows: Array<[string, string]> = [
    ["Vocabulary", `${vocabulary.readingRule} · spec ${shortSha(vocabulary.specSha256)}` +
      `${vocabulary.specSha256 === TRAINING_SPEC_SHA256 ? " (frozen)" : ""} · labeller ${shortSha(vocabulary.labellerSourceSha256)}`],
    ["Columns", `${TRAINING_COLUMNS.map((column) => (column === "runway" ? `runway ${candidates.length}`
      : `${column} ${vocabulary.classCounts[column]}`)).join(" · ")} classes, each with "unchanged"`],
    ["Runway pointer", `${candidates.map((candidate) => candidate.ident).join(" ")} (sha ${shortSha(sample.candidatesSha256)})`],
    ["Landing", `passing the threshold within ${vocabulary.landingMaxHeightM} m of its height and, off the centreline, within ` +
      `${candidates.map((candidate) => `${candidate.ident} ${candidate.landingCrossLimitM} m`).join(", ")} ` +
      `(${vocabulary.landingCrossLimitM} m, or half the spacing to a parallel runway)`],
    ["Heading", `${vocabulary.headingTargetsDeg[1] - vocabulary.headingTargetsDeg[0]}° grid, read step by step: a word says ` +
      `where the track is ${vocabulary.headingLeadS} s later, and from then to the next word's the track stays within ` +
      `±${vocabulary.headingToleranceDeg}° of it`],
    ["Capture turn", `from the clearance onto the course, begun where the track turns toward it faster than ` +
      `${vocabulary.turnOnsetRateDegS}°/s: monotone, ${vocabulary.turnRateMinDegS}–${vocabulary.turnRateMaxDegS}°/s (the lowest ` +
      `rate only for turns of ${vocabulary.turnRateMinFromDeg}° or more), at most ${vocabulary.turnBankMaxDeg}° of bank`],
    ["Corridor", `${vocabulary.corridorHalfWidthM} m at the threshold, widening ${vocabulary.corridorWideningDeg}°, course ` +
      `±${vocabulary.corridorCourseToleranceDeg}°; intercept ${vocabulary.interceptAngleDeg}°`],
    ["Altitude", `${vocabulary.altitudeTargetsM[1] - vocabulary.altitudeTargetsM[0]} m MSL grid to ` +
      `${vocabulary.altitudeTargetsM[vocabulary.altitudeTargetsM.length - 1]} m, or "descend to land"; tube ` +
      `±${vocabulary.altitudeToleranceM} m`],
    ["Angle", vocabulary.angleClasses.map((angle) => (angle.value === vocabulary.angleLevelValue ? angle.name
      : `${angle.name} ${angle.nominalDeg}° (${angle.lowDeg}…${angle.steepDeg}°)`)).join(" · ")],
    ["Speed", `ground speed ${vocabulary.speedTargetsMps[0]}…${vocabulary.speedTargetsMps[vocabulary.speedTargetsMps.length - 1]} ` +
      `m/s or "unspecified"; band ±${vocabulary.speedToleranceMps} m/s, at most ${vocabulary.speedAccelMaxMps2} m/s² between`],
    ["Read on", `signals smoothed over ${vocabulary.smoothingS.track} s (track), ${vocabulary.smoothingS.altitude} s (altitude), ` +
      `${vocabulary.smoothingS.speed} s (speed), one step every ${vocabulary.stepS} s`],
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
