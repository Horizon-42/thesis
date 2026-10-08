/**
 * The models' statistics (frontend §3 item 3, D159) on the sets and the results answers the Python code writes: a row for
 * each sentence kind of an A, a B and a C set, counted from the set's own sentences and from the formal readout the
 * results route answers (a missing count is null, the page's "—"), and the page's cells.
 */
import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { stageAStatistics, stageBStatistics, windowStatistics } from "../trainingStatistics";
import { parseTrainingSetResults, type TrainingSetResults } from "../trainingSetResults";
import { resultsAnswer, type ResultsAnswer } from "./trainingResults";
import { stageASample } from "./stageA";
import { stageBSample } from "./stageB";
import { stageCSample, stageCSampleFromRound } from "./stageC";
import { stageDSample } from "./stageD";
import { startName } from "../trainingWindowSample";
import StatisticsSection from "../../components/training/StatisticsSection";

function results(name: ResultsAnswer): TrainingSetResults {
  const parsed = parseTrainingSetResults(true, resultsAnswer(name), name);
  if (!parsed.ok) throw new Error(parsed.problem);
  return parsed.value;
}

describe("the rows of each stage", () => {
  it("A: a row for each Δ's closed loop; the readout: the replays of the set's splits at that Δ at its airport", () => {
    const sample = stageASample();
    const table = stageAStatistics(sample, results("A"));
    expect(table.windows).toBe(false);
    expect(table.rows.map((row) => [row.label, row.kind])).toEqual(
      [["closed loop · Δ 2 s", "closedLoop"], ["closed loop · Δ 4 s", "closedLoop"], ["closed loop · Δ 8 s", "closedLoop"]]);
    const [first] = table.rows;
    const outcomes = sample.flights.map((flight) => flight.closedLoop["2"].replay.outcome);
    expect(first.set).toEqual({ sentences: outcomes.length, landed: outcomes.filter((o) => o === "landed").length, goArounds: null,
      timedOut: outcomes.filter((o) => o === "timeout").length, lost: null, rewardSum: null });
    // the fixture's set is of train flights only: the train replay at Δ 2 s, not the select one
    expect(first.readout).toEqual({ sentences: 4, landed: 3, goArounds: null, timedOut: 1, lost: null, rewardSum: null });
    expect(stageAStatistics(sample, null).rows.every((row) => row.readout === null)).toBe(true);
    // every dynamics group at the set's airport, summed; never the "all" airports cell; never the other split
    const varied = structuredClone(results("A"));
    if (varied.stage !== "A" || !varied.closedLoop.ok) throw new Error("not stage A's answer");
    const train2 = varied.closedLoop.value.find((replay) => replay.split === "train" && replay.intervalS === 2)!;
    const select2 = varied.closedLoop.value.find((replay) => replay.split === "select" && replay.intervalS === 2)!;
    train2.groups["stand-in dynamics"] = { KXXX: { flights: 2, landed: 0.5, outcomes: { landed: 1, ground_contact: 1 } },
      all: { flights: 9, landed: 0, outcomes: { ground_contact: 9 } } };
    train2.groups["own dynamics"].all = { flights: 50, landed: 0, outcomes: { timeout: 50 } };
    select2.groups["own dynamics"].KXXX = { flights: 7, landed: 1, outcomes: { landed: 7 } };
    expect(stageAStatistics(sample, varied).rows[0].readout).toEqual(
      { sentences: 6, landed: 4, goArounds: null, timedOut: 1, lost: null, rewardSum: null });
  });

  it("B: the closed loop (no readout) and the prior's samples, with their go-arounds; the readout over every side of the selection", () => {
    const sample = stageBSample();
    const table = stageBStatistics(sample, results("B"));
    expect(table.rows.map((row) => [row.label, row.kind])).toEqual([["closed loop · Δ 4 s", "closedLoop"], ["the prior's samples", "base"]]);
    expect(table.rows[0].readout).toBeNull();
    const said = sample.flights.flatMap((flight) => flight.sentences);
    expect(table.rows[1].set.sentences).toBe(said.length);
    expect(table.rows[1].set.goArounds).toBe(said.reduce((sum, s) => sum + s.goArounds, 0));
    const cell = resultsAnswer("B").sections.freeGeneration.sides.inside.KXXX.vectored;
    expect(table.rows[1].readout).toMatchObject({ sentences: cell.sentences, landed: cell.outcomes.landed ?? 0, goArounds: cell.goArounds });
    // every side and every stratum at the set's airport (the set is drawn from all of them), never another airport
    const varied = structuredClone(results("B"));
    if (varied.stage !== "B" || !varied.freeGeneration.ok) throw new Error("not stage B's answer");
    const { sides } = varied.freeGeneration.value;
    const extra = (landed: number, timeout: number, goArounds: number) => ({ ...structuredClone(cell), sentences: landed + timeout,
      outcomes: { landed, timeout }, goArounds, timedOut: timeout });
    sides.inside.KXXX["straight-in"] = extra(2, 0, 0);
    sides.outside_fault = { KXXX: { vectored: extra(0, 1, 2) }, KYYY: { vectored: extra(40, 0, 0) } };
    expect(stageBStatistics(sample, varied).rows[1].readout).toEqual({
      sentences: cell.sentences + 3, landed: (cell.outcomes.landed ?? 0) + 2, goArounds: cell.goArounds + 2,
      timedOut: (cell.outcomes.timeout ?? 0) + 1, lost: null, rewardSum: null });
  });

  it("C: a row for each round; the readout each round's selection at the airport, the start's its source round's, or why not", () => {
    // a campaign from the base: the start has no selection readout
    const sample = stageCSample();
    const answer = results("C");
    if (answer.stage !== "C" || !answer.rounds.ok) throw new Error("not stage C's answer");
    const fromBase = structuredClone(answer) as typeof answer;
    if (!fromBase.rounds.ok) throw new Error("not stage C's answer");
    fromBase.rounds.value.start = null;
    const table = windowStatistics(sample, fromBase);
    expect(table.windows).toBe(true);
    expect(table.rows.map((row) => [row.label, row.kind])).toEqual([["start (base)", "base"]]);
    const said = sample.windows.map((window) => window.commanded[0].rounds[0]);
    expect(table.rows[0].set).toMatchObject({
      sentences: said.length, lost: said.filter((s) => s.outcome === "lost_separation").length,
      rewardSum: said.reduce((sum, s) => sum + s.reward, 0),
    });
    expect(table.rows[0].readout).toBeNull();                                     // the base has no selection readout
    expect(table.notes).toEqual([]);
    expect(table.readoutProblem).toBeNull();
    expect(windowStatistics(sample, results("Cmissing")).readoutProblem).toContain("campaign.json");
    // a campaign from another campaign's round (the route's fixture): the start's row reads that round's own readout
    const from = stageCSampleFromRound();
    from.model.rounds = ["start", 0];
    for (const window of from.windows) {
      for (const aircraft of window.commanded) aircraft.rounds.push({ ...aircraft.rounds[0], round: 0 });
      window.rounds.push({ ...window.rounds[0], round: 0 });
    }
    const two = windowStatistics(from, answer);
    expect(two.rows.map((row) => [row.label, row.kind]))
      .toEqual([["start (post_source_fixture r0)", "postTrained"], ["r0", "postTrained"]]);
    const source = resultsAnswer("C").sections.rounds.start.selection.KXXX;
    expect(two.rows[0].readout).toEqual({ sentences: source.windows, landed: 0, goArounds: null, timedOut: 0,
      lost: source.outcomes.lost_separation, rewardSum: source.rewardMean * source.windows });
    expect(two.notes).toEqual(["The start's readout is post_source_fixture r0's own, read on the same select windows."]);
    const round = resultsAnswer("C").sections.rounds.rounds[0].selection.KXXX;
    expect(two.rows[1].readout).toEqual({ sentences: round.windows, landed: round.outcomes.landed,
      goArounds: null, timedOut: 0, lost: 0, rewardSum: round.rewardMean * round.windows });
    // a source read on other windows: no start readout, and why
    const other = structuredClone(answer) as typeof answer;
    if (!other.rounds.ok || other.rounds.value.start === null) throw new Error("no start in the fixture");
    other.rounds.value.start = { ...other.rounds.value.start, selection: null, why: "the source campaign read other select windows" };
    const apart = windowStatistics(from, other);
    expect(apart.rows[0].readout).toBeNull();
    expect(apart.notes).toEqual(["The start's readout is not shown: the source campaign read other select windows."]);
  });
});

describe("stage D's rows (frontend §5.6)", () => {
  it("a row for each round, a sentence each commanded aircraft; the readout its airport's all cell; losses by pair in notes", () => {
    const sample = stageDSample();
    const table = windowStatistics(sample, results("D"));
    expect(table.windows).toBe(true);
    const [start] = table.rows;
    expect([start.label, start.kind]).toEqual([`start (${startName(sample.model.start)})`, sample.model.start === null ? "base" : "postTrained"]);
    const said = sample.windows.flatMap((window) => window.commanded.map((aircraft) => aircraft.rounds[0]));
    expect(start.set).toMatchObject({ sentences: said.length, lost: said.filter((s) => s.outcome === "lost_separation").length,
      rewardSum: said.reduce((sum, s) => sum + s.reward, 0) });
    expect(start.readout).toBeNull();                                     // stage C's round: read on stage C's windows
    expect(table.notes[0]).toContain("stage C's windows");
    // windows with a loss of each kind (the readout's unit): the fixture's one window, its two losses between the same two
    expect(table.notes[1]).toMatch(/windows with a loss of separation — this set: 1 between commanded aircraft, 0 with a recorded one$/);
    // a stage D round: stage D's colour, its readout the airport's all cell, the readout's loss windows by pair
    sample.model.rounds = ["start", 0];
    for (const window of sample.windows) {
      for (const aircraft of window.commanded) aircraft.rounds.push({ ...aircraft.rounds[0], round: 0 });
      window.rounds.push({ ...window.rounds[0], round: 0 });
    }
    const two = windowStatistics(sample, results("D"));
    const cell = resultsAnswer("D").sections.rounds.rounds[0].selection.KXXX;
    expect([two.rows[1].label, two.rows[1].kind]).toEqual(["r0", "multi"]);
    expect(two.rows[1].readout).toEqual({ sentences: cell.aircraft, landed: cell.outcomes.landed, goArounds: cell.goArounds,
      timedOut: 0, lost: cell.outcomes.lost_separation, rewardSum: cell.rewardSum });
    expect(two.notes[2]).toContain("; the readout: ");
    for (const pair of ["commanded commanded 1", "commanded answers 0", "records kept 0", "recorded only 2"]) expect(two.notes[2]).toContain(pair);
  });
});

describe("the page's cells", () => {
  it("a share with its count beneath, a mean reward over its count, and — where there is no count", () => {
    const sample = stageCSample();
    const table = windowStatistics(sample, null);
    render(<StatisticsSection table={table} results={{ status: "none" }} />);
    const grid = screen.getByRole("table", { name: "The models' statistics" });
    expect(within(grid).getAllByRole("columnheader").map((cell) => cell.textContent)).toEqual([
      "sentences", "this set", "the formal readout",
      ...Array(2).fill(["landed", "go-arounds / sentence", "timed out", "lost separation", "mean reward"]).flat()]);
    const cells = within(within(grid).getAllByRole("row")[2]).getAllByRole("cell").map((cell) => cell.textContent);
    const { set } = table.rows[0];
    expect(cells[0]).toBe(`${((set.landed / set.sentences) * 100).toFixed(1)} %${set.landed}/${set.sentences}`);
    expect(cells[1]).toBe(`${(set.goArounds! / set.sentences).toFixed(3)}${set.goArounds}/${set.sentences}`);   // per sentence
    expect(cells[4]).toBe(`${(set.rewardSum! / set.sentences).toFixed(3)}over ${set.sentences}`);
    expect(cells.slice(5)).toEqual(["—", "—", "—", "—", "—"]);
    expect(screen.getByText(/The formal readout: no set is on screen/)).toBeTruthy();
  });
});
