/**
 * useTrainingWindowIndex.ts
 * ------------------------
 * The airport's index of window sets (`training/index_post_v2.json`): absent (none exported — the panel offers nothing),
 * invalid (named on screen), or ready. Stage A's index is the panel's own; this one is read beside it and stage B's.
 */

import { useEffect, useState } from "react";
import { isMissingJsonAsset } from "../utils/fetchJson";
import { fetchTrainingWindowIndex, type TrainingWindowIndex } from "../data/trainingWindowSample";

export type TrainingWindowIndexState =
  | { status: "loading" }
  | { status: "absent" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; index: TrainingWindowIndex };

export default function useTrainingWindowIndex(airport: string | null): TrainingWindowIndexState {
  const [state, setState] = useState<TrainingWindowIndexState>({ status: "loading" });
  useEffect(() => {
    if (!airport) return;
    let live = true;
    setState({ status: "loading" });
    fetchTrainingWindowIndex(airport)
      .then((parsed) => {
        if (live) setState(parsed.ok ? { status: "ready", index: parsed.value } : { status: "invalid", problem: parsed.problem });
      })
      .catch((error: unknown) => {
        if (!live) return;
        if (isMissingJsonAsset(error)) setState({ status: "absent" });
        else setState({ status: "invalid", problem: error instanceof Error ? error.message : String(error) });
      });
    return () => {
      live = false;
    };
  }, [airport]);
  return state;
}
