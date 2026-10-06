/**
 * The answers of `GET /training/results` the backend WRITES (`aeroviz_backend/tests/test_training_results.py` writes
 * `fixtures/training_results/answers.json`; never edited by hand): stage A's, stage B's (the set and the claimed val set),
 * stage C's, and C's with its campaign's file missing.
 */
import answersFile from "./fixtures/training_results/answers.json";
import { AEROVIZ_BACKEND_URL } from "../../pilot/pilotClient";
import { TRAINING_RESULTS_PATH, type TrainingStage } from "../trainingSetResults";

export type ResultsAnswer = "A" | "B" | "Bval" | "C" | "Cmissing";
export const resultsAnswer = (name: ResultsAnswer): Record<string, any> =>
  structuredClone((answersFile as Record<string, unknown>)[name]) as Record<string, any>;
export const resultsUrl = (stage: TrainingStage, airport: string, setId: string): string =>
  `${AEROVIZ_BACKEND_URL.replace(/\/+$/, "")}${TRAINING_RESULTS_PATH}?${new URLSearchParams({ stage, airport, set: setId })}`;
