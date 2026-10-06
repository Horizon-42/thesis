/**
 * useTrainingSet.ts
 * -----------------
 * The one loader of a Training set (outline §6.2 item 5): stage A's panel and B's and C's sessions open their set through
 * it. ``key`` names the set (its airport, id and file); a new key starts a new load, and an answer of an older key is
 * never shown (a set of another airport never opens with this one's). ``load`` reads the set with the stage's own reader.
 */

import { useEffect, useRef, useState } from "react";
import type { Parsed } from "../data/trainingReader";

export type TrainingSetState<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; sample: T };

export default function useTrainingSet<T>(key: string | null, load: () => Promise<Parsed<T>>): TrainingSetState<T> {
  const [state, setState] = useState<{ key: string | null; value: TrainingSetState<T> }>({ key: null, value: { status: "idle" } });
  const loader = useRef(load);
  loader.current = load;
  useEffect(() => {
    if (key === null) {
      setState({ key, value: { status: "idle" } });
      return;
    }
    let live = true;
    setState({ key, value: { status: "loading" } });
    loader.current()
      .then((parsed) => {
        if (live) setState({ key, value: parsed.ok ? { status: "ready", sample: parsed.value } : { status: "invalid", problem: parsed.problem } });
      })
      .catch((error: unknown) => {
        if (live) setState({ key, value: { status: "invalid", problem: error instanceof Error ? error.message : String(error) } });
      });
    return () => {
      live = false;
    };
  }, [key]);
  if (state.key === key) return state.value;
  return key === null ? { status: "idle" } : { status: "loading" };
}
