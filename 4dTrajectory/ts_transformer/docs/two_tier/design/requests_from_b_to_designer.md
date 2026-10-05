# Requests from stage B to the designer

What Claude, as stage B's implementer, asks of the designer: the readings that wait for the user's decision, what stage
A must change for stage B, and the design text that stage B's work now needs. It holds only the current state: it is
rewritten in full each time, never appended to, and an item leaves when it is done. The evidence and the commits are in
the implementation log (`readouts/2026-10-05_stage_b_implementation_log.md` §1).

State of 2026-10-05. Branch `dev-two-tier-v4-prior` at `89845fa6`: B0–B4, B6, B8, B9, B10 and B11 done and reviewed;
stage A's line (A37–A41) merged and followed; full suite passed. Stopped before B5.

## 1 Readings for the user to decide

Each is built as written. None changes a result the user has seen.

1. **A faulty flight that also did not land** (B11, D111) is counted as left out for its fault, not for its outcome:
   `prior/selection.py` `left_out` asks the mark first.
2. **The identity also holds each split's stored signals** (`signals_files`, their sha256; B11). The faulty-track
   marks are read from the signals, so this binds the val marks before the claim of the val read, without reading them
   (D85). It changes §8 item 1's list (section 3 below).
3. **The claim of the val read names its readout relative to the repository** (`checkpoint.claim_validation_read`),
   so it stays valid after the worktree the readout ran in is removed.
4. **The claimed validation set flies on a second stage A service** (D109): `AutopilotSegmentBackend(splits=
   SEALED_READINGS, …)` inside stage B's `aeroviz_backend/autopilot_segment/prior.py`, so stage A's `backend.py` is not
   changed. The frontend gives a claimed set val alone (`TRAINING_SEALED_SPLITS`).
5. **After the claim, the val readers recount val's selection** (`source.require_selection_of`) as a last guard. A
   refusal there spends the one val read; with item 2 it can only come from a change of stage A's fault rule that
   moves no train or select mark.
6. **A flight the caller ends** (post-training D93) has the caller's outcome: `SpeakingLoop.generated` refuses it by
   name, and `said` and `states` give its sentence.
7. **The behaviour check of D108** (`experiments/prior_behaviour.py`) uses the artefact's vocabulary spec and its first
   airport's finals as its fixed inputs. Its answer covers two training steps, the speaker, the input functions, the
   variant `constants`, the first-step runway and `SpeakingLoop` on a straight-flying stand-in. The campaign compares
   its own settings too: the seeds, the selection, the configurations, free generation, the temperature and D68's bound.
8. **Two readings of B9** still stand:
   - a flight's random numbers in free generation come from numpy `default_rng([seed, sample, index])`;
   - `masked_log_probability` gives 0 at a row not asked, and an asked row reads the record of the row said at its
     time, with its own-state inputs checked against the record.

## 2 For stage A

- `aeroviz_backend/autopilot_segment/backend.py` (A23's file), line 107: the comment on stage B's own hook
  (`self.prior = PriorSegments(self)`, added by B6's `aac93945` and listed in the log as a hook in A23's files) still
  names `index_prior_v1.json`; stage B's index is now `index_prior_v2.json` (`prior.training_files.INDEX_FILE`). Stage
  B has not edited the file since the rule (stage B does not edit stage A's code): the designer decides whether stage B
  updates its own hook's comment or stage A does.
- The synthetic artefact of the tests (`tests/support.py`, `labelled_instruction_artefact`) writes no
  `runway_ends_from` in `signals.json`, which the real artefacts record and A37's
  `training_export.candidate_hae_minus_msl_m` reads. Stage B's export tests add a stand-in
  (`tests/test_prior_training_export.py` `with_runway_ends`). If the helper wrote it, the stand-in could go.

## 3 For the design text

- **§7, the names of this round** (B10's are written already):
  - item 1:
    - `prior/checkpoint.py` `readable_identity`, `validation_claim`, `holds_claim`, `CHECKPOINT_SCHEMA` =
      `ts-prior-checkpoint-v8`;
    - `prior/source.py` `require_selection_of`.
  - item 4: `prior/selection.py`:
    - `kept(rule, outcome, faulty)`, `left_out`, `side`, `SIDES`, `REASONS`, `CELL`;
    - the record's cells are `kept`, `left_out_fault` and `left_out_outcome`.
  - item 7: `experiments/prior_speaking_loop.py` `SpeakingLoop.copy(flights)` (on A38's `Loop.copy`), `said(b)`,
    `states(b)`, `generated(flights)`.
- **§8 item 1:** the identity of a prior's data also holds the sha256 of each split's stored signals (section 1, item 2).
- **Formats changed in this round**, each under a new name:
  - `ts-prior-checkpoint-v8`, `ts-prior-free-generation-v4`, `ts-prior-validation-v3`;
  - `aeroviz-training-prior-index-v2` / `aeroviz-training-prior-sample-v2`, with the index file `index_prior_v2.json`;
  - `procedure-masks-v5` and `ts-prior-campaign-v2` (both from B10).

  A prior stamped with an earlier name is refused. Every such prior is a smoke run: no formal prior exists yet.
- **D85 in the readouts:** free generation and the validation readout write the identity without the val counts
  (`readable_identity`) into their `config.json`. Before B10 they copied the whole identity.

## 4 The plan

- **B5** waits for the user's go. The campaign's dry run (`--smoke 50`, before B10) ran every step through to the base
  and its readouts.
- **B6's publication** of the folds and the base comes after B5.
- **B7** adds `docs/reference/runners.md` entries for the new runner `prior_behaviour` and for the changes to
  `prior_campaign`, `prior_free_generation`, `prior_validation` and `prior_training_export`.
