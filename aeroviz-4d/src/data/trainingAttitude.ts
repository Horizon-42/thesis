/**
 * trainingAttitude.ts
 * -------------------
 * The attitude an aircraft is DRAWN in by the Training scene (Training module design, doc 36 §4.12): its heading, path
 * angle and right bank at each point of a track, and a reading of its angle of attack — every one computed in Python
 * (`experiments/training_attitude.py`) and exported beside the track's points; nothing here computes an attitude, it is
 * read and interpolated between points.
 *
 * A flight the dynamics has no airframe for (C31) has its heading and path angle and NO bank or attack (both null):
 * it is drawn wings level and says so. The angle of attack is a reading through a clean-wing lift curve — high on a
 * flapped final — so it is written in the label, never into the drawn pitch (the user, 2026-09-30).
 */

import type { Reader } from "./trainingReader";

/** A track's attitude, one value per point of the track it sits beside (`training_attitude.ATTITUDE_FIELDS`: the reader
 *  refuses a field missing by name). */
export interface TrainingAttitude {
  /** Compass degrees. */
  headingDeg: number[];
  /** The flight path's angle to the horizontal, climbing positive. */
  pathAngleDeg: number[];
  /** Right wing down positive; null: no airframe to read it with. */
  bankRightDeg: number[] | null;
  /** A reading (the module docstring); null with the bank. */
  attackDeg: number[] | null;
}

/** A track an aircraft is drawn on: its points' times (any one clock), positions and attitude. */
export interface TrainingAttitudeTrack {
  tS: number[];
  lon: number[];
  lat: number[];
  altitudeHaeM: number[];
  attitude: TrainingAttitude;
}

/** Where an aircraft is and how it sits, at one instant. */
export interface TrainingAircraftPose {
  lon: number;
  lat: number;
  heightHaeM: number;
  headingDeg: number;
  pathAngleDeg: number;
  bankRightDeg: number | null;
  attackDeg: number | null;
}

/** A track's `attitude` block, ``n`` values a field; the bank and the attack are both there or both null. */
export function readAttitude(reader: Reader, n: number): TrainingAttitude {
  const bankRightDeg = reader.nullableNumbers("bankRightDeg", n);
  const attackDeg = reader.nullableNumbers("attackDeg", n);
  if ((bankRightDeg === null) !== (attackDeg === null)) reader.fail("has a bank without an attack reading, or one without the other");
  return { headingDeg: reader.numbers("headingDeg", n), pathAngleDeg: reader.numbers("pathAngleDeg", n), bankRightDeg, attackDeg };
}

/** Where ``track`` is at ``atS`` (its own clock) — linear between its points, the heading the shorter way round — or null
 *  outside its span. */
export function poseAt(track: TrainingAttitudeTrack, atS: number): TrainingAircraftPose | null {
  const { tS, attitude } = track;
  if (atS < tS[0] || atS > tS[tS.length - 1]) return null;
  // the last point at or before it, and the one after (the last point's own when it is the last)
  let k = 0;
  let high = tS.length - 1;
  while (k < high) {
    const mid = (k + high + 1) >> 1;
    if (tS[mid] <= atS) k = mid;
    else high = mid - 1;
  }
  const next = Math.min(k + 1, tS.length - 1);
  const f = next > k ? (atS - tS[k]) / (tS[next] - tS[k]) : 0;
  const at = (values: number[]) => values[k] + f * (values[next] - values[k]);
  const turn = ((attitude.headingDeg[next] - attitude.headingDeg[k] + 540) % 360) - 180;
  return {
    lon: at(track.lon), lat: at(track.lat), heightHaeM: at(track.altitudeHaeM),
    headingDeg: (attitude.headingDeg[k] + f * turn + 360) % 360,
    pathAngleDeg: at(attitude.pathAngleDeg),
    bankRightDeg: attitude.bankRightDeg === null ? null : at(attitude.bankRightDeg),
    attackDeg: attitude.attackDeg === null ? null : at(attitude.attackDeg),
  };
}

/** The pose in words, for an aircraft's label: "hdg 230° · bank 12° R · path −3.0° · α 15° (reading)". */
export function poseText(pose: TrainingAircraftPose): string {
  const bank = pose.bankRightDeg === null ? "bank — (no airframe)"
    : `bank ${Math.abs(pose.bankRightDeg).toFixed(0)}°${Math.abs(pose.bankRightDeg) < 0.5 ? "" : pose.bankRightDeg > 0 ? " R" : " L"}`;
  const attack = pose.attackDeg === null ? "" : ` · α ${pose.attackDeg.toFixed(0)}° (reading)`;
  return `hdg ${Math.round(pose.headingDeg) % 360}° · ${bank} · path ${pose.pathAngleDeg.toFixed(1)}°${attack}`;
}
