# Resume: KSJC v5 imitation ladder

Stopped 2026-08-21 13:09 at the user's request, mid-`v5_w16`. Nothing is wrong with the
experiment; it was halted to free the machine.

## State

| arm | weight | ×position | status |
|---|---:|---:|---|
| `v5_baseline` | 0 | — | **DONE** — checkpoint, 1083 predictions, report, 4 CZML |
| `v5_w16` | 16 | 11.8 | not started (stub cleared) |
| `v5_w28` | 28 | ~20.6 | not started |
| `v5_w64` | 64 | 47 | not started |

`run_ts_control_arms.py` has **no resume or skip logic** — pointing it at the original
four-arm declaration would retrain the finished baseline. Use the three-arm declaration
instead, writing into the SAME campaign directory so all four score together.

## Resume command

```bash
cd /home/supercomputing/studys/thesis
conda run --no-capture-output -n aeroviz python -u run_ts_control_arms.py \
  --arms 4dTrajectory/ts_transformer/docs/experiments/imitation_ksjc_v5_remaining_arms.json \
  --campaign 4dTrajectory/outputs/KSJC/experiments/imitation_v5 --airport KSJC
```

≈35 min per arm on the RTX 4060, so ≈1.8 h for the three.

## Check BEFORE resuming

`arrivals/lateral_pass_eligibility.json` must exist. Any harvest rebuild deletes it
(`arrivals._clear()` unlinks every `*.json` in that directory — see
`trajectory_data_process/CLAUDE.md`), and its absence surfaces only at predict, after an
arm has already trained. Rebuild with:

```bash
conda run -n aeroviz python -c "
import sys; sys.path[:0]=['.','4dTrajectory/ts_transformer']
from lateral_eligibility import ensure_lateral_pass_roster
print(ensure_lateral_pass_roster('trajectory_data_process/outputs/harvest/KSJC/arrivals/manifest.json'))"
```

Expected state: manifest schema `harvest-arrivals-v5-takeoff-excluded`, 11,082 included,
64 `takeoff_in_segment`; roster 11,076 eligible; train 5,023 / val 1,083.

## The question this ladder answers

`config.py` claims a new airport should recalibrate `control_imitation_loss_weight` off the
box-normalised channel split rather than inherit KRDU's 64.0. Bank carries **41 %** of the
imitation term at KRDU against **18 %** at KSJC, which predicts KSJC's equivalent of KRDU's
47× is 47 / (41/18) ≈ 20.6×, i.e. **weight 28**.

**Pre-registered reading** (fixed before the run, do not revise it afterwards):

- If **28** lands near KSJC's flown straight-in bank of **0.53°** with FDE recovered toward
  baseline and most of the skill kept → the recipe in `config.py` stands.
- If **16** is needed instead → the channel-share rule under-corrects, and `config.py` must
  be rewritten to say so rather than keep a rule that only looks right.

## Baseline already measured (v5 cohort)

| metric | flown truth | `v5_baseline` |
|---|---:|---:|
| per-flight bank skill | — | **0.157** |
| — random-flight floor | | 0.313 |
| — same-runway twin | | 0.543 |
| common-profile share | 3.2 % | 17.0 % |
| straight-ref bank RMS | 0.53° | 0.75° |
| straight-ref reversals | 0 | 2.0 |
| ADE median | | 327.3 m |
| FDE mean | | 853.6 m |

Skill **0.157 is below the 0.313 floor**, so the defect this whole investigation is about
reproduces on the clean cohort: the predicted bank says less about the flown bank than a
randomly chosen other flight does.

## Analysis once the three arms land

```bash
# dose ladder, with each airport's own flown truth printed under every metric
conda run -n aeroviz python 4dTrajectory/ts_transformer/docs/score_control_arms.py \
  4dTrajectory/outputs/KSJC/experiments/imitation_v5

# paired sign tests against the baseline (read MAGNITUDES, not p: at n=1083 pure seed
# noise returns p=3e-16)
D=4dTrajectory/outputs/KSJC/experiments/imitation_v5
for a in v5_w16 v5_w28 v5_w64; do
  conda run -n aeroviz python 4dTrajectory/ts_transformer/docs/compare_control_arms_paired.py \
    $D/v5_baseline_pred_val $D/${a}_pred_val
done

# inside a matched route-mix stratum — v5 predictions carry the approach_difficulty
# covariates, which the earlier v4 run did not
conda run -n aeroviz python 4dTrajectory/ts_transformer/docs/compare_control_arms_stratified.py \
  4dTrajectory/outputs/KSJC/experiments/imitation_v5
```

## Context worth not relearning

- **KRDU is unaffected by the v5 schema bump** — verified by content hash, not by count:
  eligible 14,378, sha `3a8d9e3e6376b7bf`, identical before and after. Every KRDU
  conclusion, including `simple-v3`, stands.
- **v4 and v5 share an identical val set** (1,083 flights, zero difference), so the earlier
  v4 KSJC pair is a like-for-like reference on the evaluation side and differs only in
  training cohort. v4 baseline skill was 0.197 against v5's 0.157.
- The discarded first attempt at this ladder was not a code failure: a
  `--reclassify-existing` run killed with SIGTERM finished anyway several minutes later,
  cleared `arrivals/` a second time, and replaced the manifest under a training arm.
