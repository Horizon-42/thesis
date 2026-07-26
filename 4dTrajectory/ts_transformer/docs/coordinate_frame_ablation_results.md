# Pooled Coordinate-Frame Ablation: Results and Interpretation

## Experiment status

This document analyzes the pooled iTransformer/full-horizon coordinate-frame ablation
recorded on 2026-07-25 UTC. The experiment has completed paired cross-validation, selected
the winning frame, and trained the final winning model. It has **not** evaluated the held-out
outer-test set:

- result status: `trained`;
- `outer_test_used_for_selection: false`;
- `outer_test_evaluation_started: false`;
- `outer_test_evaluated: false`.

Consequently, this report can support a coordinate-frame selection decision and describe
outer-validation behavior, but it cannot yet make a held-out test-generalization claim.

## Executive conclusion

The experiment selected **ENU**, not `runway-aligned`.

| Coordinate frame | Best mean CV airport-macro normalized MSE | Fold standard deviation | Selected trial |
|---|---:|---:|---:|
| ENU | **0.193617** | 0.029265 | 1 |
| Runway-aligned | 0.203823 | 0.028626 | 1 |

ENU reduced the registered selection objective by **0.010206**, or **5.01% relative to
runway-aligned**. This was not caused by the two frames selecting different architectures:
both selected the same hyperparameter configuration. ENU also scored better in all three
paired folds of the winning trial and under all four matched hyperparameter trials.

The defensible conclusion is therefore:

> For this pooled, full-horizon iTransformer experiment, with this data roster, seed,
> candidate grid, normalization, and CV budget, ENU is the preferred coordinate frame under
> the registered normalized-MSE selection objective.

This does **not** establish that ENU is universally better, or even that its physical-distance
error is lower. The current CV artifact does not record a common physical-unit metric for
both frames, and the final test set is still untouched.

## Experimental design and leakage controls

The fixed usable-flight population contained 19,741 flights from KMSY, KRDU, KSJC, KSMF,
and KSTL:

| Outer split | Flights | Approximate share | Used for coordinate selection? |
|---|---:|---:|---|
| Train | 13,807 | 69.94% | Yes, through inner CV only |
| Validation | 2,951 | 14.95% | No; final-training early stopping only |
| Test | 2,983 | 15.11% | No; still sealed |

The two coordinate runs recorded identical manifest hashes, outer-split hashes, CV-fold
hashes, seed, candidate ordering, sampling budget, and resolved batch sizes. The paired
comparison used:

- 3 airport-stratified folds constructed only from outer-train;
- 4 matched hyperparameter trials;
- 12 epoch maximum and patience 4 per CV fit;
- 100,000 airport → flight → anchor samples per CV epoch;
- airport-macro validation loss, so large airports did not directly dominate selection;
- batch size 512 in every recorded fold;
- seed 1337.

The losing runway-aligned model was not subsequently trained against outer-validation. Only
the selected ENU configuration was given the final outer-train/outer-validation fit.

## Cross-validation results

### Results across all matched trials

| Trial | Main configuration | ENU mean | Runway-aligned mean | ENU reduction relative to runway-aligned |
|---:|---|---:|---:|---:|
| 0 | Baseline: 128 width, 3 layers, LR 1e-4, dropout 0.10 | 0.235974 | 0.246454 | 4.25% |
| 1 | 256 width, 2 layers, LR 3e-4, dropout 0.05 | **0.193617** | **0.203823** | **5.01%** |
| 2 | 256 width, 3 layers, LR 5e-4, dropout 0.20, WD 1e-4 | 0.203783 | 0.215777 | 5.56% |
| 3 | 256 width, 2 layers, LR 3e-4, dropout 0.20, WD 1e-4 | 0.221379 | 0.232163 | 4.65% |

ENU wins under every sampled configuration. The absolute aligned-minus-ENU differences are
0.010480, 0.010206, 0.011994, and 0.010784, which is a notably consistent direction and
magnitude within this search. It is stronger evidence than a win from only one isolated
configuration, although it is still a one-seed experiment rather than a statistical
replication study.

Trial 1 improved substantially over the baseline for both frames: approximately 17.95% for
ENU and 17.30% for runway-aligned. The selected hyperparameters were:

```json
{
  "learning_rate": 0.0003,
  "d_model": 256,
  "d_ff": 512,
  "e_layers": 2,
  "n_heads": 8,
  "dropout": 0.05,
  "weight_decay": 0.0
}
```

### Paired folds for the winning trial

| Fold | ENU | Runway-aligned | Aligned − ENU | Winner |
|---:|---:|---:|---:|---|
| 0 | 0.156358 | 0.165946 | 0.009589 | ENU |
| 1 | 0.196644 | 0.210387 | 0.013743 | ENU |
| 2 | 0.227850 | 0.235137 | 0.007287 | ENU |

All three paired folds favor ENU. Fold difficulty varies substantially for both frames, but
the coordinate-frame ordering does not change with the fold.

### Airport-level CV behavior

The following values average each airport's validation loss across the three folds of the
winning trial:

| Airport | ENU | Runway-aligned | ENU reduction relative to aligned |
|---|---:|---:|---:|
| KMSY | 0.163105 | 0.165331 | 1.35% |
| KRDU | 0.148093 | 0.162638 | 8.94% |
| KSJC | 0.089639 | 0.094638 | 5.28% |
| KSMF | 0.139603 | 0.159247 | 12.34% |
| KSTL | 0.427645 | 0.437263 | 2.20% |

ENU is better at every airport, so its macro win is not produced by one large airport.
KSTL is much harder than the other airports under both frames and is the largest source of
fold-to-fold variation. Because the metric is airport-macro and the sampler is
airport-balanced, this is a genuine per-airport modeling issue rather than a direct result of
KSTL having more or fewer flights.

## Final selected-model training

The final ENU model has 1,147,436 parameters and was trained on CUDA with batch size 512 and
250,000 balanced samples per epoch. It ran the full 50-epoch cap; early stopping did not fire.
The best outer-validation macro loss occurred at epoch 49:

| Quantity | Value |
|---|---:|
| Best epoch | 49 |
| Train loss at best epoch | 0.125888 |
| Outer-validation macro loss | 0.233140 |
| Total recorded epoch time | 471.9 s (7.9 min) |

The best epoch's outer-validation loss by airport was:

| Airport | Normalized loss |
|---|---:|
| KMSY | 0.103460 |
| KRDU | 0.161928 |
| KSJC | 0.048695 |
| KSMF | 0.182773 |
| KSTL | 0.668845 |

The fixed outer-validation set is clearly harder than the average CV folds, primarily at
KSTL. Its loss should not be compared mechanically with the CV mean: it contains a different
flight population and its normalizer was fitted on the complete outer-train rather than a CV
fold's training subset.

### Outer-validation errors in physical units

These values describe the selected ENU model on outer-validation, not outer-test:

| Metric | Mean | P95 | Maximum |
|---|---:|---:|---:|
| ADE | 1,674.8 m | 5,762.1 m | — |
| FDE | 1,703.6 m | 5,148.9 m | — |
| Horizontal error | 1,669.7 m | 5,756.8 m | 33,166.1 m |
| Along-track absolute error | 1,158.1 m | 4,602.4 m | 28,055.4 m |
| Cross-track absolute error | 770.3 m | 3,314.5 m | 32,978.4 m |
| Altitude absolute error | 84.6 m | 304.2 m | 14,677.6 m |

The long error tails and extreme maxima warrant flight-level outlier inspection before the
model is described as operationally reliable. The signed along-track mean is -148.0 m, the
signed cross-track mean is -37.8 m, and the signed altitude mean is -6.6 m, so the large mean
absolute errors are not explained by one simple global directional bias.

Selected mean horizontal error by lead time is:

| Lead time | Mean | P95 | Remaining flights |
|---:|---:|---:|---:|
| 60 s | 688.0 m | 1,781.0 m | 2,951 |
| 120 s | 988.4 m | 3,080.2 m | 2,889 |
| 300 s | 3,042.7 m | 8,329.5 m | 1,354 |
| 600 s | 1,889.4 m | 7,755.9 m | 201 |

The apparent reduction from 300 to 600 seconds is not evidence that the forecast improves at
longer horizons: only 201 long-duration flights contribute at 600 seconds, versus 1,354 at
300 seconds. This changing survivor population makes cross-horizon values non-comparable
without a fixed-flight cohort.

## Interpretation

The original runway-alignment hypothesis was plausible: removing runway orientation could
make geometrically similar approaches look alike across airports. The observed result does
not support that hypothesis for this experiment. Several mechanisms could explain why, but
the current artifacts cannot distinguish among them:

1. The threshold-anchored ENU representation already removes absolute geographic position,
   and iTransformer may learn the remaining rotations without difficulty.
2. Absolute east/north orientation may act as a useful implicit airport/runway-context cue.
   Rotating every approach into one canonical direction removes that cue.
3. iTransformer treats variables as tokens. Rotating the `e/n` and `edot/ndot` channel pairs
   changes their marginal distributions and correlations, so the transformation is not
   neutral to this architecture.
4. Each frame fits its own per-channel normalizer. The registered loss is therefore a valid
   training/selection objective, but it is not a rotation-invariant physical-distance metric.

Items 1–4 are explanatory hypotheses, not findings proven by this run.

## Limitations and threats to validity

1. **No test result yet.** The held-out test set is still sealed; only selection and
   outer-validation behavior are available.
2. **One random seed.** Three paired folds give consistent results, but they are not three
   independent repetitions of the entire data split and training process.
3. **Small hyperparameter search.** Only four configurations were tested.
4. **Short CV fits.** All three ENU winning folds achieved their best score at the 12-epoch
   cap, while the final fit continued improving until epoch 49. The coordinate comparison is
   fair because both frames had the same cap, but the hyperparameter ranking may reflect
   undertrained CV models.
5. **Coordinate-specific normalized objective.** A lower normalized MSE does not by itself
   prove lower metre-scale trajectory error across coordinate systems.
6. **Scope is narrow.** The result applies to pooled iTransformer in full-horizon mode. It
   should not be generalized to PatchTST, window forecasting, or per-airport training.
7. **Airport heterogeneity remains.** KSTL performance is substantially worse than the other
   airports and deserves a data-quality/trajectory-pattern analysis on non-test data.

## Recommended decision

Keep **ENU** as the selected coordinate frame for this experiment. Before opening test, the
strongest optional robustness checks would be paired multi-seed CV, a longer CV epoch budget,
and recording a common physical-unit validation metric inside each fold. Those checks must
remain confined to outer-train if they are used to revisit the selection.

Once the experimental design is frozen, evaluate only the selected ENU checkpoint on
outer-test once. Do not train or test the losing runway-aligned configuration after examining
that test result.

## Source artifacts

- [Ablation decision](../../outputs/POOLED/ts_itransformer_full_coordinate_frame_ablation/coordinate_frame_ablation.json)
- [ENU CV results](../../outputs/POOLED/ts_itransformer_full_coordinate_frame_ablation/enu/cross_validation/cv_results.json)
- [Runway-aligned CV results](../../outputs/POOLED/ts_itransformer_full_coordinate_frame_ablation/runway-aligned/cross_validation/cv_results.json)
- [Selected ENU training history](../../outputs/POOLED/ts_itransformer_full_coordinate_frame_ablation/enu/history.json)
