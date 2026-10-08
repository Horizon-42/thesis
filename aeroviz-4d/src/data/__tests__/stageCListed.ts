/**
 * The listed set the Python code WRITES (post-training D176, frontend F5: stage C's export with `--windows`, a window list
 * of the synthetic flight's selection window — `4dTrajectory/ts_transformer/tests/test_post_training_export.py` writes
 * `fixtures/stage_c_listed/`; never edited by hand): its cohort is the list's.
 */
import indexFile from "./fixtures/stage_c_listed/index_post_v4.json";
import sampleFile from "./fixtures/stage_c_listed/listed/sample.json";

export const stageCListedIndex = (): Record<string, any> => structuredClone(indexFile) as Record<string, any>;
export const stageCListedSampleFile = (): Record<string, any> => structuredClone(sampleFile) as Record<string, any>;
