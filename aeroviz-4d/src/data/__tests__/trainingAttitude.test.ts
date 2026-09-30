/**
 * The attitude the Training scene draws an aircraft in (`trainingAttitude.ts`): read beside a track, interpolated between
 * its points — the heading the shorter way round — and said in words; a flight without an airframe has no bank.
 */
import { describe, expect, it } from "vitest";
import { poseAt, poseText, readAttitude, type TrainingAttitudeTrack } from "../trainingAttitude";
import { Reader, Refusal } from "../trainingReader";

function track(headingDeg: number[], bankRightDeg: number[] | null): TrainingAttitudeTrack {
  return { tS: [0, 2], lon: [-78, -77.99], lat: [35, 35], altitudeHaeM: [1000, 990],
    attitude: { headingDeg, pathAngleDeg: [-3, -2], bankRightDeg, attackDeg: bankRightDeg === null ? null : [6, 8] } };
}

describe("trainingAttitude", () => {
  it("reads a track's attitude, its bank and attack both there or both null", () => {
    const block = { headingDeg: [90, 91], pathAngleDeg: [-3, -3], bankRightDeg: null, attackDeg: null };
    expect(readAttitude(Reader.of(block, "track.attitude"), 2).bankRightDeg).toBeNull();
    expect(() => readAttitude(Reader.of({ ...block, attackDeg: [6, 6] }, "track.attitude"), 2)).toThrow(Refusal);
    expect(() => readAttitude(Reader.of(block, "track.attitude"), 3)).toThrow(/has 2 values, expected 3/);
  });

  it("interpolates between points, the heading across north the shorter way, and nothing outside the track", () => {
    const half = poseAt(track([350, 10], [10, 20]), 1)!;
    expect(half.headingDeg).toBeCloseTo(0, 9);
    expect([half.pathAngleDeg, half.bankRightDeg, half.attackDeg, half.heightHaeM]).toEqual([-2.5, 15, 7, 995]);
    expect(poseAt(track([350, 10], [10, 20]), 2)!.headingDeg).toBeCloseTo(10, 9);
    expect(poseAt(track([350, 10], [10, 20]), 2.1)).toBeNull();
    expect(poseAt(track([90, 90], null), 1)!.bankRightDeg).toBeNull();
  });

  it("says the pose in words: the side it banks to, the attack as a reading, no bank without an airframe", () => {
    expect(poseText(poseAt(track([230, 230], [-12, -12]), 0)!)).toBe("hdg 230° · bank 12° L · path -3.0° · α 6° (reading)");
    expect(poseText(poseAt(track([90, 90], null), 0)!)).toBe("hdg 90° · bank — (no airframe) · path -3.0°");
  });
});
