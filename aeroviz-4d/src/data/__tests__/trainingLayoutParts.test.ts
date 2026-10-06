/**
 * The shared parts of the Training views (outline §6.2 item 5) that hold no view: the tabs' one state (the session gives
 * the tabs, the bar chooses, the arrow keys, Home and End move), the intent's answer read by name, and the one loader of a
 * set (an answer of an older set never shows).
 */
import { describe, expect, it } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { chooseTrainingTab, publishTrainingTabs, tabForKey, useTrainingTabs, type TrainingTab } from "../trainingTabs";
import { parseTrainingSetIntent } from "../trainingSetIntent";
import useTrainingSet from "../../hooks/useTrainingSet";
import answers from "./fixtures/training_intent/answers.json";
import { requestTrainingDetails, useDetailsPage } from "../../components/training/PanelParts";

const TABS: TrainingTab[] = [
  { id: "labelled", label: "Labelled", title: "", outcome: null, kind: null },
  { id: "closed-loop", label: "Closed loop", title: "", outcome: "landed", kind: "closedLoop" },
  { id: "sample-0", label: "Sample 0", title: "", outcome: "timeout", kind: "base" },
];

describe("the tabs' one state", () => {
  it("holds the session's tabs and the bar's choice, refuses a tab that is none of them, and clears", () => {
    const { result } = renderHook(() => useTrainingTabs());
    act(() => publishTrainingTabs({ scope: "a", tabs: TABS, chosen: "sample-0" }));
    expect(result.current?.chosen).toBe("sample-0");
    act(() => chooseTrainingTab("closed-loop"));
    expect(result.current?.chosen).toBe("closed-loop");
    act(() => chooseTrainingTab("round-3"));                                // not a tab on screen: nothing changes
    expect(result.current?.chosen).toBe("closed-loop");
    expect(() => publishTrainingTabs({ scope: "a", tabs: TABS, chosen: "round-3" })).toThrow(/round-3 is none of/);
    act(() => publishTrainingTabs(null));
    expect(result.current).toBeNull();
    act(() => chooseTrainingTab("labelled"));                              // no session: nothing to choose
    expect(result.current).toBeNull();
  });

  it("moves with the arrow keys (wrapping), Home and End, and ignores other keys", () => {
    expect(tabForKey(TABS, "labelled", "ArrowRight")).toBe("closed-loop");
    expect(tabForKey(TABS, "sample-0", "ArrowRight")).toBe("labelled");
    expect(tabForKey(TABS, "labelled", "ArrowLeft")).toBe("sample-0");
    expect(tabForKey(TABS, "closed-loop", "ArrowDown")).toBe("sample-0");
    expect(tabForKey(TABS, "closed-loop", "Home")).toBe("labelled");
    expect(tabForKey(TABS, "labelled", "End")).toBe("sample-0");
    expect(tabForKey(TABS, "labelled", "Enter")).toBeNull();
    expect(tabForKey(TABS, "gone", "ArrowRight")).toBeNull();
  });
});

describe("a set's intent", () => {
  it("reads the backend's answers (the fixture its test writes): one campaign in the picker's form, none or several named", () => {
    const one = parseTrainingSetIntent(true, answers.one.body, "set_a");
    expect(one).toEqual({ ok: true, value: { campaign: "one", groupTitle: "T", group: "I", run: "line a", design: "D" } });
    const none = parseTrainingSetIntent(answers.none.status === 200, answers.none.body, "nowhere");
    expect(none.ok ? "" : none.problem).toContain("No intent for nowhere: nowhere is a run of no campaign");
    const several = parseTrainingSetIntent(answers.several.status === 200, answers.several.body, "shared");
    expect(several.ok ? "" : several.problem).toContain("2 campaigns (one, two)");
    const broken = parseTrainingSetIntent(true, { ok: true, campaign: "c", title: 3 }, "s");
    expect(broken.ok ? "" : broken.problem).toContain("not readable: title, intent, design, line");
  });
});

describe("the one loader of a set", () => {
  it("shows the answer of the set asked last, never an older one that arrives late", async () => {
    const answers: Record<string, (value: { ok: true; value: string }) => void> = {};
    const load = (key: string) => () => new Promise<{ ok: true; value: string }>((resolve) => { answers[key] = resolve; });
    const { result, rerender } = renderHook(({ key }) => useTrainingSet<string>(key, load(key ?? "")), { initialProps: { key: "a" as string | null } });
    expect(result.current.status).toBe("loading");
    rerender({ key: "b" });
    act(() => answers.a({ ok: true, value: "set a" }));                      // the older set's answer, late
    expect(result.current.status).toBe("loading");
    act(() => answers.b({ ok: true, value: "set b" }));
    await waitFor(() => expect(result.current).toEqual({ status: "ready", sample: "set b" }));
    rerender({ key: null });
    expect(result.current.status).toBe("idle");
  });
});

describe("the details page's state", () => {
  it("opens once for each request — never again after a close, a task switch or a remount", () => {
    const opener = document.createElement("button");
    act(() => requestTrainingDetails("before", opener));                   // made before the panel mounted: not opened
    const { result, rerender, unmount } = renderHook(({ hidden }) => useDetailsPage(hidden), { initialProps: { hidden: false } });
    expect(result.current.shown).toBeNull();
    act(() => requestTrainingDetails("experiment", opener));
    expect(result.current.shown?.section).toBe("experiment");
    act(() => result.current.close());
    expect(result.current.shown).toBeNull();
    rerender({ hidden: true });                                            // another task, and back to Learning
    rerender({ hidden: false });
    expect(result.current.shown).toBeNull();
    unmount();
    const again = renderHook(() => useDetailsPage(false));                // a remount
    expect(again.result.current.shown).toBeNull();
  });
});
