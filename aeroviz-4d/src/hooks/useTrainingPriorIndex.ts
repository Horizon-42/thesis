/**
 * useTrainingPriorIndex.ts
 * ------------------------
 * The airport's index of prior sets (`training/index_prior_v1.json`): absent (none exported — the panel offers nothing),
 * invalid (named on screen), or ready. Stage A's index is the panel's own; this one is read beside it.
 */

import { useEffect, useState } from "react";
import { isMissingJsonAsset } from "../utils/fetchJson";
import { fetchTrainingPriorIndex, type TrainingPriorIndex } from "../data/trainingPriorSample";

export type TrainingPriorIndexState =
  | { status: "loading" }
  | { status: "absent" }
  | { status: "invalid"; problem: string }
  | { status: "ready"; index: TrainingPriorIndex };

export default function useTrainingPriorIndex(airport: string | null): TrainingPriorIndexState {
  const [state, setState] = useState<TrainingPriorIndexState>({ status: "loading" });
  useEffect(() => {
    if (!airport) return;
    let live = true;
    setState({ status: "loading" });
    fetchTrainingPriorIndex(airport)
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
