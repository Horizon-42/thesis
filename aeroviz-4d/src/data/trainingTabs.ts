/**
 * trainingTabs.ts
 * ---------------
 * WHICH SENTENCE IS ON SCREEN, in every stage (outline §6.2 item 2): the session of the set on screen gives the sentence
 * bar its tabs — stage A: Labelled and one per Δ; B: Labelled, Closed loop and the samples; C: Labelled, Start (base) and
 * the rounds — and the bar's tabs choose. One state holds the choice: the session publishes its tabs and the tab chosen
 * (`publishTrainingTabs`), the bar calls `chooseTrainingTab`, and the session reads the choice back (`useTrainingTabs`)
 * and shows that sentence. The left panel has no second chooser.
 *
 * A store of its own (as `trainingPriorLayers`): the sessions live in the left dock and the bar is a sibling of it.
 */

import { useSyncExternalStore } from "react";
import type { TrainingFlownEnd } from "./trainingSample";

/** One tab of the bar: a sentence of the item on screen. */
export interface TrainingTab {
  /** Unique among the tabs (e.g. "labelled", "interval-4", "closed-loop", "sample-0", "start", "round-2"). */
  id: string;
  label: string;
  /** What the sentence is, and — for a sentence that was flown — its outcome, time and go-arounds. */
  title: string;
  /** The judge's outcome of a flown sentence, drawn as a dot in its colour; null for a sentence not flown (Labelled). */
  outcome: TrainingFlownEnd | null;
}

export interface TrainingTabs {
  /** Whose tabs: the session's scope (the set and the item); a choice is kept only within it. */
  scope: string;
  tabs: TrainingTab[];
  chosen: string;
}

let state: TrainingTabs | null = null;
const listeners = new Set<() => void>();

function emit(): void {
  listeners.forEach((listener) => listener());
}

/** The session's tabs for the item on screen, and the tab chosen: kept when the scope and the tab are unchanged (a session
 *  publishes again on each render of a new item), else the session's own choice. */
export function publishTrainingTabs(next: TrainingTabs | null): void {
  if (next !== null && next.tabs.every((tab) => tab.id !== next.chosen)) {
    throw new Error(`tab ${next.chosen} is none of ${next.tabs.map((tab) => tab.id).join(", ")}`);
  }
  if (sameTabs(state, next)) return;
  state = next;
  emit();
}

function sameTabs(a: TrainingTabs | null, b: TrainingTabs | null): boolean {
  if (a === null || b === null) return a === b;
  return a.scope === b.scope && a.chosen === b.chosen && a.tabs.length === b.tabs.length &&
    a.tabs.every((tab, index) => {
      const other = b.tabs[index];
      return tab.id === other.id && tab.label === other.label && tab.title === other.title && tab.outcome === other.outcome;
    });
}

/** The bar's choice: one of the tabs on screen (a tab of another scope, or none, changes nothing). */
export function chooseTrainingTab(id: string): void {
  if (state === null || state.chosen === id || state.tabs.every((tab) => tab.id !== id)) return;
  state = { ...state, chosen: id };
  emit();
}

/** The tabs on screen and the tab chosen; null when no session gives any. */
export function useTrainingTabs(): TrainingTabs | null {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => state,
  );
}

/** The tab a key moves to (outline §6.2 item 2: the arrow keys between the tabs, Home and End to the first and the last);
 *  null for any other key. */
export function tabForKey(tabs: TrainingTab[], chosen: string, key: string): string | null {
  const index = tabs.findIndex((tab) => tab.id === chosen);
  if (tabs.length === 0 || index < 0) return null;
  if (key === "ArrowRight" || key === "ArrowDown") return tabs[(index + 1) % tabs.length].id;
  if (key === "ArrowLeft" || key === "ArrowUp") return tabs[(index - 1 + tabs.length) % tabs.length].id;
  if (key === "Home") return tabs[0].id;
  if (key === "End") return tabs[tabs.length - 1].id;
  return null;
}
