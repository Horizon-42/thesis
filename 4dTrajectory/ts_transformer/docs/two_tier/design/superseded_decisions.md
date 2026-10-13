# Superseded decisions and passages: an index for review

**What this file is.** The two-tier design documents hold the current design only. On 2026-10-12 they were rewritten in
place to match the code of the branch `two-tier2-eval-landing` (commits `04378395`, `ea632d20`, `51a68b8e`), and on
2026-10-13 again for commits `e00c8d08` and `d8f21c77`, for the prior's campaign on the new labels (`7e72a059`,
B15), and for the final's lateral corrections (D201, the branch `two-tier2-centreline`). This file
keeps each text that the rewrite replaced, word for word, beside the text that replaced it, so that the user can compare
them. It is not part of the design: no code and no design document cites it. An entry leaves this file when the user
has reviewed it.

**The user's decisions behind the rewrite (2026-10-12, the user's words).**

1. A landing is the evaluation package's verdict (vocabulary D196): "修改降落判定，直接引用evaluation 的降落判定 不要再单独维护
   一套不同的降落判定函数". Where a flight ends does not change: "evaluation只是判定是否降落，飞行结束的判定逻辑还包括复飞，失去
   间隔，这些为什么要变？". The DA check stays as a record: "保留，只做记录".
2. The closed loop's tolerances in the final descent are half of the others, Y_final = 15 m and H_final = 7.5 m
   (vocabulary D66): "直接用一般 如果垂直的还是10m 也改成7.5 m".
3. The corrections read the errors ahead by the lead (vocabulary D197): "标注器是看提前量的吧？如果是 那就按1".
4. Stage A's labelling reads its val days; the test days come at the end (outline D198): "在标注阶段，val可以读。我们要保证的
   是词表标注的上限足够高 test可以放在最后，用来验证整个标注的结果是不是可以泛化的。"
5. A correction ends at the row with the smallest error left, aimed at the path, not under half the tolerance
   (vocabulary D197; 2026-10-13): the user chose "改成偏差最小的那行撤回".
6. "Unspecified" is the middle of the evaluation's speed window (executor design §8.1; 2026-10-13): "未指定速度 应该是一个
   范围速度 在 v_ref*sqrt(n) 到 v_ref+20kt之间 为什么会在速度门下沿 它们应该是一致的啊", and then "飞中点". No extra
   limit: "必须保证这个速度不会小于 stall 速度 不会大于最大速度 不过如果v_ref 本身已经保证了这一点 或者dynamic 模型本身会处理这一点
   那么没必要加额外的限制或判定" (the stall floor of §8.1 holds; the window's upper edge is far below the flap and
   structural limits).
7. A near-level piece is a level, a vertical correction moves two classes far off the path, and a new observed heading
   word does not stop a heading correction (vocabulary D199; 2026-10-13): the user chose "A + C + D1 都采用".
8. The prior is trained on the new labels with a smaller cross-validation (prior D39, D40, B15; 2026-10-13): "should run
   cv, but maybe not 30 runs; see what's is necessary and what's not", then the choice "A + C, with seed pair".
9. The new labels' base reads the val days once (outline D85; 2026-10-13): "allow; and we difinately have whole new
   dataset, why this only read once check still work for this new dataset?".
10. On the final, once the observed aircraft is lined up, the lateral corrections aim at the runway's centreline, and
   the labels and the prior are built again (vocabulary D201; 2026-10-13): after the diagnosis of B15's held-out
   folds, the user chose "采用，做 v18 并重跑 B（推荐）".

The new decision rows (vocabulary D196, D197, D201; outline D198) are new text, not replacements, so they are not entries
below; the rewritten rows that cite them are.

## Index

| # | Document | Where | What changed |
|---|---|---|---|
| 1 | `vocabulary.md` | D3 | D3 rewritten: the LPV minimums are a record (D196) |
| 2 | `vocabulary.md` | D32 | D32 rewritten: Y_final, H_final 7.5 m (D66) and the errors ahead (D197) |
| 3 | `vocabulary.md` | D38 | D38: the DA check is a record (D196) |
| 4 | `vocabulary.md` | D50 | D50 rewritten: a record, not a rule kept on every row (D197) |
| 5 | `vocabulary.md` | D53 | D53 rewritten: the overshoot reads the end error (D197); 'the rule of D50 then holds on every row' removed |
| 6 | `vocabulary.md` | D66 | D66 rewritten: Y_final = 15 m added, H_final 10 m → 7.5 m; the reason is the evaluation's verdict (D196) |
| 7 | `vocabulary.md` | §0.3 the val days | §0.3: the val days are read in the labelling stage (outline D198) |
| 8 | `vocabulary.md` | §0.4 item 3 | §0.4 item 3: the test days at the end (outline D198) |
| 9 | `vocabulary.md` | §4.9 lateral and vertical correction | §4.9 lateral and vertical corrections: the tolerance in force (Y_final) and the start / end errors (D197); a correction ends where the end error reaches zero, not under half the tolerance (the user, 2026-10-13) |
| 10 | `vocabulary.md` | §4.9 vertical items 2–3 | §4.9 vertical items 2–3: the start / end errors (D197) |
| 11 | `vocabulary.md` | §4.9 tolerances | §4.9 tolerances: Y_final 15 m, H_final 7.5 m; the reason is the evaluation's verdict (D196) |
| 12 | `vocabulary.md` | §5.8 runway limit | §5.8: the runway limit tells only a crossing of another candidate (D196) |
| 13 | `vocabulary.md` | §5.8 outcome table | §5.8 outcomes: crossed_too_high, crossed_off_runway, unstable_at_minimums deleted; landed / landing_failed by the evaluation's verdict (D196) |
| 14 | `vocabulary.md` | §5.8 the DA check | §5.8: the DA check becomes the DA record; the landing paragraph (D196) |
| 15 | `vocabulary.md` | §5.8 quality | §5.8: 'Quality belongs to the evaluation module; the judge does not repeat it' removed (the verdict is the landing, D196) |
| 16 | `vocabulary.md` | §6 item 6 | §6 item 6: the landing verdict, the DA record |
| 17 | `vocabulary.md` | §8 constants | §8: 'Landed lateral limit' → runway limit of another candidate; Landing row (D196); DA check → DA record |
| 18 | `vocabulary.md` | §8 tolerances | §8 closed-loop tolerances: Y_final, H_final 7.5 m, the errors ahead (D197) |
| 19 | `outline.md` | D85 | outline D85: stage A's labelling reads its val days (D198) |
| 20 | `prior.md` | §4 Decoding (to correct or to go around) | prior: the landing verdict judges it (vocabulary D196) |
| 21 | `post_training.md` | §2 Post-training (the reward's reasons, D30) | post-training: `unstable_at_minimums` → `landing_failed` (vocabulary D196) |
| 22 | `multi_control.md` | §2.1 One aircraft: D30, unchanged (D140) | multi-control: `unstable_at_minimums` → `landing_failed` (vocabulary D196) |
| 23 | `executor_first_principles.zh.md` | §7.8 为什么删除草案 3 的 γ̇_c, item 1 | executor first principles: H_final 10 → 7.5 m (vocabulary D66) |
| 24 | `executor_first_principles.zh.md` | §10.2 待验证项, V14 pass condition | executor first principles V14: the pass condition reads H_final = 7.5 m (vocabulary D66) |
| 25 | `vocabulary.md` | §5.8 the DA record, lateral | §5.8 DA record, lateral: the reason 'narrower than the closed loop's 30 m tolerance' removed (Y_final = 15 m is in force at the DA point, D66); the evaluation's lateral gate is read at the threshold (D196) |
| 26 | `vocabulary.md` | §2 Terms, Judge | §2 Terms: a go-around answers an approach that would fail the landing verdict, not a failed DA check (D196) |
| 27 | `vocabulary.md` | §4.9 item 2 (who is flown) | the approach speed a flight needs is the flown type's (the user, 2026-10-13) |
| 28 | `vocabulary.md` | §5.2 what the executor reads | the executor reads the flown type's published speeds for the middle of the evaluation's window (the user, 2026-10-13) |
| 29 | `vocabulary.md` | §5.6 speed law, "unspecified" | "unspecified" is the middle of the evaluation's speed window, from the flown type (the user, 2026-10-13) |
| 30 | `vocabulary.md` | O22 | O22 closed: the executor reads the flown type's window, the one the evaluation grades (the user, 2026-10-13) |
| 31 | `executor_first_principles.zh.md` | §0.2 I3 | I3: the flown type's published approach speeds (two flap configurations) and MALW |
| 32 | `executor_first_principles.zh.md` | §8.1 目标速度 V* | "unspecified" = the middle of the evaluation's speed window, flown type, mass and n now (the user, 2026-10-13) |
| 33 | `executor_first_principles.zh.md` | §8.2.2 目标表 | the table follows §8.1 |
| 34 | `executor_first_principles.zh.md` | §10.2 V2 | V2 checks the middle against the lower edge (v15) |
| 35 | `executor_first_principles.zh.md` | §13 与评分的关系 | the executor reads the evaluation's speed window, and no other number of it |
| 36 | `vocabulary.md` | §5.5 Vertical law (the modes and the inner loop) | the vertical law as the executor of 2026-10-12 flies it (executor design §7, D5, D6): "level" flies 0° and does not fly to T, a ramp over L_γ, a physical stop and a captured hold; no τ_h, no γ̇_max. This text had not followed the executor |
| 37 | `vocabulary.md` | §4.4 Pieces, item 1 (Level) | a near-level piece (within 0.5° of level, at least 20 s) is a level too (D199) |
| 38 | `vocabulary.md` | §4.9 Lateral correction, items 1 and 3 | a new observed heading word neither ends nor stops a heading correction (D199) |
| 39 | `vocabulary.md` | §4.9 Vertical correction, item 2 | a vertical correction moves two classes while its start error is more than 2 H (D199) |
| 40 | `vocabulary.md` | §4.9 Tolerances (D32, D66) | the factor of the two-class step is a constant of the spec too |
| 41 | `vocabulary.md` | §9.7, the item on one correction class | the rule this measurement ran under is marked as the rule before D199 |
| 42 | `prior.md` | D39 | the variant is `full`; `constants` is not run (B5's score, D24) |
| 43 | `prior.md` | D40 | configurations A and C only; B and D are not run (B5's scores); the tie rule, which only A and D could need, goes |
| 44 | `prior.md` | §5 Cross-validation, the steps | two configurations (A, C), no variant step: 15 runs and the base |
| 45 | `prior.md` | §5 Cross-validation, the rules | no tie rule; the variant is given, not chosen |
| 46 | `prior.md` | §5 The base's one validation readout | the base reads the val days once for each labelled artefact (D85) |
| 47 | `prior.md` | §12 Rules, the formal runs | the formal runs read the labels of D196–D199 (B15) |
| 48 | `prior.md` | §0.2 Open items | the results page reads a variant choice that B15's campaign does not write |
| 49 | `outline.md` | D85 | one val read for each stage and each labelled artefact |
| 50 | `outline.md` | §1 `vocabulary.md` state | the formal artefact is v17_20261013 / v22_20261013 |
| 51 | `outline.md` | §1 `prior.md` state | B15 is built |
| 52 | `vocabulary.md` | §0.3 the formal artefact | the formal artefact is v17_20261013 / v22_20261013 |
| 53 | `vocabulary.md` | §0.3 D196–D199 row, its last sentences | the frontend and backend part is committed (288b779b) |
| 54 | `vocabulary.md` | §4.9 The comparison, the lateral error e_y | on the final, once the observed aircraft is lined up, the lateral corrections aim at the centreline (D201) |
| 55 | `vocabulary.md` | §4.9 The errors ahead (D197), ė_y and the end error | on the final, once the observed aircraft is lined up, the lateral corrections aim at the centreline (D201) |
| 56 | `vocabulary.md` | §4.9 The errors ahead (D197), the end error | on the final, once the observed aircraft is lined up, the lateral corrections aim at the centreline (D201) |
| 57 | `vocabulary.md` | §4.9 Lateral correction, item 1 | on the final, once the observed aircraft is lined up, the lateral corrections aim at the centreline (D201) |
| 58 | `vocabulary.md` | §4.9 Lateral correction, item 3 | on the final, once the observed aircraft is lined up, the lateral corrections aim at the centreline (D201) |
| 59 | `vocabulary.md` | §0.1 D83, its first sentence | on the final, once the observed aircraft is lined up, the lateral corrections aim at the centreline (D201) |

## Text kept unchanged on purpose

- `vocabulary.md` §9 (the evidence of stage A's milestones, for example the replay outcomes `crossed_off_runway`,
  `unstable_at_minimums` and `crossed_too_high` of A15–A24, and the DA check failures of select at Δ = 2 s): these are
  measurements made with the judge of their time. They stay as measured.
- `frontend.md`: the Training view still draws the DA point, its chip and its label as before; the evaluation's verdict
  is added as text only. That frontend change is committed (`288b779b`); `frontend.md` is not rewritten for it yet.
- `executor_first_principles.zh.md` §17.4 ("角度类别、名义角、高度网格、ε、H、H_final 都不变"): it says what that design kept;
  it stays true.
- The readouts (`docs/two_tier/readouts/`) are records of their day and are not rewritten.

## Entries

### 1. `vocabulary.md`: D3

**What changed:** D3 rewritten: the LPV minimums are a record (D196)

**Old text:**

```text
| D3 | After the aircraft is on the final, the executor continues to fly the words of the model. The LPV minimums are a judgement only. The model must learn to land or to go around | Decided | User, 2026-10-03 |
```

**New text:**

```text
| D3 | After the aircraft is on the final, the executor continues to fly the words of the model. The executor does not fly the LPV minimums: the judge records the DA check, and the evaluation's verdict judges the landing (D196). The model must learn to land or to go around | Decided | User, 2026-10-03; the landing: user, 2026-10-12 (D196) |
```

### 2. `vocabulary.md`: D32

**What changed:** D32 rewritten: Y_final, H_final 7.5 m (D66) and the errors ahead (D197)

**Old text:**

```text
| D32 | Closed-loop reading. The labeller flies its sentence with the executor. When the flown path leaves the observed path by more than a tolerance, it says a correction word, and the observed word again when the flown path is back. These sentences and their flown states are the training sentences of the artefact (§4.9, §6). Tolerances: lateral 30 m, vertical 15 m (in the final descent H_final, D66) | Decided | User, 2026-10-03 |
```

**New text:**

```text
| D32 | Closed-loop reading. The labeller flies its sentence with the executor. When the flown path leaves the observed path by more than a tolerance, it says a correction word, and the observed word again when the flown path is back. These sentences and their flown states are the training sentences of the artefact (§4.9, §6). Tolerances: lateral Y = 30 m, vertical H = 15 m; in the final descent Y_final = 15 m and H_final = 7.5 m (D66). The labeller reads each error ahead by the lead of its column (D197) | Decided | User, 2026-10-03; the final descent and the lead: user, 2026-10-12 |
```

### 3. `vocabulary.md`: D38

**What changed:** D38: the DA check is a record (D196)

**Old text:**

```text
| D38 | The DA check uses one definition with the evaluation module. Vertical: within ±22 m of the published glidepath of R, the bound of `evaluation/thresholds.py` (`RNAV_TERMINAL_VERTICAL_BOUND_M`, ICAO Doc 9613). Lateral: inside the FAS cone at the distance of the DA point. Neither is a parameter (§5.8) | Decided | User, 2026-10-04 |
```

**New text:**

```text
| D38 | The DA check uses one definition with the evaluation module. Vertical: within ±22 m of the published glidepath of R, the bound of `evaluation/thresholds.py` (`RNAV_TERMINAL_VERTICAL_BOUND_M`, ICAO Doc 9613). Lateral: inside the FAS cone at the distance of the DA point. Neither is a parameter (§5.8). The DA check is a record on every approach crossing; it decides no outcome (D196) | Decided | User, 2026-10-04; a record: user, 2026-10-12 |
```

### 4. `vocabulary.md`: D50

**What changed:** D50 rewritten: a record, not a rule kept on every row (D197)

**Old text:**

```text
| D50 | The closed-loop reading does not keep every row inside the tolerances of D32. A correction is one class and needs time to bring the flown path back (§9.7: a 100 m offset takes approximately 20 s at one 5° class); in the stage A smoke 10–25 % of the correctable rows are more than 30 m off laterally. The check is a rule on the stored data: on every row where §4.9 permits a correction, \|e_y\| > Y is followed by a heading correction in force after that row, and \|e_h\| > H by an angle correction; with D53 the reading keeps this rule on every row. The p90 of \|e_y\| at the last row is information. The share of the correctable rows outside the tolerances is a reading of the ablation (D34), not a check (§4.8) | Decided | User, 2026-10-04 (check §3 item 4) |
```

**New text:**

```text
| D50 | The closed-loop reading does not keep every row inside the tolerances of D32. A correction is one class and needs time to bring the flown path back (§9.7: a 100 m offset takes approximately 20 s at one 5° class); in the stage A smoke 10–25 % of the correctable rows are more than 30 m off laterally. A readout counts, on the rows where §4.9 permits a correction, the rows with \|e_y\| greater than the lateral tolerance in force and no heading correction toward the path in force after the row, and the same for \|e_h\| and the angle corrections (`outside_rows`). This count is a record, not a check: the labeller reads the errors ahead (D197), so a row outside the tolerance whose error closes fast gets no correction. The p90 of \|e_y\| at the last row is information. The share of the correctable rows outside the tolerances is a reading of the ablation (D34), not a check (§4.8) | Decided | User, 2026-10-04 (check §3 item 4); a record: user, 2026-10-12 (D197) |
```

### 5. `vocabulary.md`: D53

**What changed:** D53 rewritten: the overshoot reads the end error (D197); 'the rule of D50 then holds on every row' removed

**Old text:**

```text
| D53 | An overshoot. When a correction is in force and the error changes its sign while it is more than the tolerance (the flown aircraft crossed the observed path in one row), the labeller says the opposite correction in the same row: one class toward the path from the observed word. When the error changes its sign and is within the tolerance, it says the observed word again. Laterally and vertically alike. The rule of D50 then holds on every row (§4.9, §12.2). Why: a correction ends at a sign change so that it does not push the aircraft further on the other side; waiting one row to correct that side adds a lag of one row Δ that comes from the reading, not from Δ, and the ablation compares Δ (A15 smoke: every break of D50 at Δ = 4 and 8 s was such a row) | Decided | User, 2026-10-04 |
```

**New text:**

```text
| D53 | An overshoot. When a correction is in force and its end error (D197) has the other sign and is more than the tolerance (the flown aircraft crosses the observed path within the row), the labeller says the opposite correction in the same row: one class toward the path from the observed word. When the end error has the other sign and is within the tolerance, it says the observed word again. Laterally and vertically alike (§4.9). Why: a correction ends at a sign change so that it does not push the aircraft further on the other side; waiting one row to correct that side adds a lag of one row Δ that comes from the reading, not from Δ, and the ablation compares Δ (A15 smoke: every break of D50 at Δ = 4 and 8 s was such a row) | Decided | User, 2026-10-04; the end error: user, 2026-10-12 (D197) |
```

### 6. `vocabulary.md`: D66

**What changed:** D66 rewritten: Y_final = 15 m added, H_final 10 m → 7.5 m; the reason is the evaluation's verdict (D196)

**Old text:**

```text
| D66 | While "no level-off" is in force (the final descent), the vertical tolerance of the closed-loop reading is H_final = 10 m, smaller than H; elsewhere H = 15 m. The final descent is told by the words, not by procedure data (principle 3). The user chose H_final from a measurement (D55, milestone A24): the closed-loop reading and its replay at Δ = 2, 4, 8 s for H_final = 15, 10, 7.5 and 5 m, on the train sample of D34 and on every labelled select flight. A correction ends below half the tolerance in force; the rule of D50 and the readings of D34 read the tolerance in force at each row. Why: the DA check allows 22 m (D38); the real tracks lie up to +8 m (p90) above the glidepath at the DA point; with H = 15 m the flown path can stay up to 15 m above the observed one before a correction starts, and the correction then needs time. 67 of 1,999 train replays at Δ = 2 s ended `unstable_at_minimums`, 61 of them high, and 57 of these 67 flights passed the same check on their real track (`readouts/2026-10-04_stage_a_check_a15_a22.zh.md` §7). These sentences are the prior's training sentences | Decided; H_final = 10 m, the user's choice from A24 | User, 2026-10-04, on Claude's reading of the check; 10 m: user, 2026-10-04 |
```

**New text:**

```text
| D66 | While "no level-off" is in force (the final descent), the tolerances of the closed-loop reading are half of the others: lateral Y_final = 15 m (Y = 30 m elsewhere) and vertical H_final = 7.5 m (H = 15 m elsewhere). The final descent is told by the words, not by procedure data (principle 3). Y_final and H_final are fields of the spec, given on the command line of `instruction_spec`. A correction ends below half the tolerance in force; the rule of D50 and the readings of D34 read the tolerance in force at each row. Why: a landing is the evaluation's verdict at the threshold (D196): the lateral offset within half the runway's width (15.24–22.86 m) and the height within ±22 m of LTP + TCH. With Y = 30 m in the final descent the flown path can cross the threshold up to 30 m off the observed path, outside the lateral gate (v13, train, Δ = 2 s: 85.5 % landed, the lateral gate failed in most failed crossings, |e_y| 25.6 m median; with Y_final and H_final, v14: 95.1 %; `readouts/2026-10-12_landing_verdict_labels.zh.md`). These sentences are the prior's training sentences | Decided; Y_final = 15 m, H_final = 7.5 m | User, 2026-10-04 (a tolerance of its own in the final descent); 15 m and 7.5 m: user, 2026-10-12 |
```

### 7. `vocabulary.md`: §0.3 the val days

**What changed:** §0.3: the val days are read in the labelling stage (outline D198)

**Old text:**

```text
| The replay of the val days | Not run; it waits for the user (outline D85) |
```

**New text:**

```text
| D196, D66 (Y_final, H_final 7.5 m), D197 | Built on the branch `two-tier2-eval-landing` (from `two-tier2`): `04378395` (D196), `ea632d20` (D66), `51a68b8e` (D197), each reviewed. Artefacts, each from its commit in the run worktree `run-v8`, read-only: `instruction_language/v13_20261012` with `executor/v18_20261012` (D196), `v14_20261012` with `v19_20261012` (and D66), `v15_20261012` with `v20_20261012` (and D197). Readings: `readouts/2026-10-12_landing_verdict_labels.zh.md`. The frontend and backend part of D196 is not committed: the user's last stage. Not merged: the user merges |
| The val days | Read in the labelling stage (outline D198): the closed-loop landed shares of v13–v15 |
```

### 8. `vocabulary.md`: §0.4 item 3

**What changed:** §0.4 item 3: the test days at the end (outline D198)

**Old text:**

```text
3. The replay of the val days, the stage's one validation readout (outline D85), waits for the user.
```

**New text:**

```text
3. The test days check the whole labelling at the end (outline D198).
```

### 9. `vocabulary.md`: §4.9 lateral and vertical correction

**What changed:** §4.9 lateral and vertical corrections: the tolerance in force (Y_final) and the start / end errors (D197); a correction ends where the end error reaches zero, not under half the tolerance (the user, 2026-10-13)

**Old text:**

```text
**Lateral correction.**

1. When |e_y| > Y (the lateral tolerance), and the row says no new observed heading word, the labeller says the
   heading class one step (5°) from the observed word in force, toward the observed path.
2. When |e_y| < Y / 2, or e_y changes its sign, the labeller says the observed word in force again — except an
   overshoot (D53): when e_y changes its sign and |e_y| > Y, it says the opposite correction in the same row (item 1
   toward the other side).
3. A new observed heading word ends a correction. The labeller says the observed word, and the comparison continues.

**Vertical correction.** The vertical tolerance in force is H_final while "no level-off" is in force (the final
descent) and H elsewhere (D66). In the rules below, H is the tolerance in force.
```

**New text:**

```text
**The errors ahead (D197).** A word said at a Δ row has its full effect one lead later (L for a heading word, L_γ for
an angle word, §5), and the next row comes Δ later. Thus the labeller reads each error ahead, at the rate ė at which it
changes now: ė_y is the flown velocity across the matched segment of the observed path (positive to the right), ė_h the
flown vertical rate minus the segment's slope times the flown velocity along it. The START ERROR is e + ė (Δ/2 + L): a
word said now comes at the Δ row nearest the time at which it must be said, as an observed word does (D45). The END
ERROR is e + ė (Δ/2 + L/2): the turn back to the observed word takes L, and in that time the aircraft moves half as far
as at the rate now. A correction ends at the Δ row nearest the time at which the turn back leaves no error, the row
where the end error is zero or has the other sign: the end aims at the observed path. At ė = 0 the start error is the
error now.

**Lateral correction.** The lateral tolerance in force is Y_final while "no level-off" is in force (the final descent)
and Y elsewhere (D66). In the rules below, Y is the tolerance in force, e_y,on the start error and e_y,off the end error.

1. When |e_y,on| > Y, and the row says no new observed heading word, the labeller says the heading class one step (5°)
   from the observed word in force, toward the observed path.
2. When e_y,off is zero or has the other sign, the labeller says the observed word in force again — except an
   overshoot (D53): when |e_y,off| > Y, it says the opposite correction in the same row (item 1 toward the other side).
3. A new observed heading word ends a correction. The labeller says the observed word, and the comparison continues.

**Vertical correction.** The vertical tolerance in force is H_final while "no level-off" is in force (the final
descent) and H elsewhere (D66). In the rules below, H is the tolerance in force, e_h,on the start error and e_h,off the
end error.
```

### 10. `vocabulary.md`: §4.9 vertical items 2–3

**What changed:** §4.9 vertical items 2–3: the start / end errors (D197)

**Old text:**

```text
2. When e_h > H (too high), the labeller says the next steeper descent class. When e_h < −H (too low), it says the next
   shallower descent class. From descent 4 there is no steeper class, and from descent 1 no shallower one: then there is
   no correction.
3. When |e_h| < H / 2, or e_h changes its sign, the labeller says the observed angle class in force again — except an
   overshoot (D53): when e_h changes its sign and |e_h| > H, it says the opposite correction in the same row (item 2,
   where that class exists). A new observed altitude word or angle word ends a correction.
```

**New text:**

```text
2. When e_h,on > H (too high), the labeller says the next steeper descent class. When e_h,on < −H (too low), it says
   the next shallower descent class. From descent 4 there is no steeper class, and from descent 1 no shallower one: then
   there is no correction.
3. When e_h,off is zero or has the other sign, the labeller says the observed angle class in force again — except an
   overshoot (D53): when |e_h,off| > H, it says the opposite correction in the same row (item 2, where that class
   exists). A new observed altitude word or angle word ends a correction.
```

### 11. `vocabulary.md`: §4.9 tolerances

**What changed:** §4.9 tolerances: Y_final 15 m, H_final 7.5 m; the reason is the evaluation's verdict (D196)

**Old text:**

```text
**Tolerances (D32, D66).** Y = 30 m, H = 15 m and H_final, constants of the labeller in the spec. With Y = 30 m the
flown path ends within approximately 30 m of the observed path, which ends 2 m from the centreline (median), far inside
the runway limit of 106.7 m. H = 15 m is larger than the fit residual of the altitude pieces (10 m, §4.4), so that the
noise of the fit starts no correction. In the final descent, H = 15 m is too wide for the DA check (D38: 22 m at the DA
point, with the real tracks up to +8 m above the glidepath there, p90), so H_final is smaller there: the user chose
H_final = 10 m from the measurement of A24, which also counts the correction words that each value adds (D66).
```

**New text:**

```text
**Tolerances (D32, D66).** Y = 30 m, H = 15 m, Y_final = 15 m and H_final = 7.5 m, constants of the labeller in the
spec. H = 15 m is larger than the fit residual of the altitude pieces (10 m, §4.4), so that the noise of the fit starts
no correction. In the final descent the tolerances are half of these: the landing is the evaluation's verdict at the
threshold (D196), with a lateral gate of half the runway's width (15.24–22.86 m) and a vertical gate of ±22 m of
LTP + TCH; with Y = 30 m there, the flown path can cross the threshold up to 30 m off the observed path, outside the
lateral gate.
```

### 12. `vocabulary.md`: §5.8 runway limit

**What changed:** §5.8: the runway limit tells only a crossing of another candidate (D196)

**Old text:**

```text
spacing to a parallel). The runway limit is the FAS half-width at the threshold (106.7 m), and not more than half the
spacing to a parallel. The judge gives the first outcome that occurs. At one row it uses the order of the table.
```

**New text:**

```text
spacing to a parallel). The runway limit of a candidate is the FAS half-width at the threshold (106.7 m), and not more
than half the spacing to a parallel; it tells a crossing of another candidate. The judge gives the first outcome that
occurs. At one row it uses the order of the table.
```

### 13. `vocabulary.md`: §5.8 outcome table

**What changed:** §5.8 outcomes: crossed_too_high, crossed_off_runway, unstable_at_minimums deleted; landed / landing_failed by the evaluation's verdict (D196)

**Old text:**

```text
| `crossed_too_high` | An approach crossing higher than 100 m above the threshold |
| `crossed_off_runway` | An approach crossing at ≤ 100 m, outside the runway limit |
| `unstable_at_minimums` | An approach crossing at ≤ 100 m, inside the runway limit, after a failed DA check (or with no DA point: the aircraft crossed above the DA) |
| `landed` | An approach crossing at ≤ 100 m, inside the runway limit, after a DA check that passed |
```

**New text:**

```text
| `landing_failed` | An approach crossing that the evaluation's verdict fails (D196), with the gates that failed |
| `landed` | An approach crossing that the evaluation's verdict passes (D196) |
```

### 14. `vocabulary.md`: §5.8 the DA check

**What changed:** §5.8: the DA check becomes the DA record; the landing paragraph (D196)

**Old text:**

```text
**The DA check (D3, D38).** At the DA point the judge checks that the aircraft is stable:
```

**New text:**

```text
**The landing (D196).** At an approach crossing the judge gives the flown state at the interpolated crossing to the
evaluation package (`evaluation.metrics.evaluate_record`) with the candidate's assessment context (`candidates.json`,
built by `evaluation.context.assessment_for_runway`) and the flown type. The verdict passes when the lateral offset is
within half the runway's width, the height within ±22 m of LTP + TCH, and the speed within the flown type's published
window (the evaluation's gates, `evaluation/docs/EVALUATION_REFERENCE.md`). The judge adds no bound of its own.

**The DA record (D3, D38).** At the DA point the judge records whether the aircraft is stable. The record is kept on
every approach crossing beside the verdict; it decides no outcome (D196):
```

### 15. `vocabulary.md`: §5.8 quality

**What changed:** §5.8: 'Quality belongs to the evaluation module; the judge does not repeat it' removed (the verdict is the landing, D196)

**Old text:**

```text
**Quality** belongs to the evaluation module (lateral, vertical and speed gates). The judge does not repeat it.
```

**New text:**

(deleted)

### 16. `vocabulary.md`: §6 item 6

**What changed:** §6 item 6: the landing verdict, the DA record

**Old text:**

```text
| 6 | The judge | The outcome of a flight and its order (§5.8), the DA check, the time limit
```

**New text:**

```text
| 6 | The judge | The outcome of a flight and its order (§5.8): a landing is the evaluation's verdict (D196); the DA record (D38), the time limit
```

### 17. `vocabulary.md`: §8 constants

**What changed:** §8: 'Landed lateral limit' → runway limit of another candidate; Landing row (D196); DA check → DA record

**Old text:**

```text
| Landed lateral limit | FAS half-width at the threshold, 106.7 m | FAA Order 8260.58D Formula 3-1-1 |
| Lined up | track within 30° of the course | 7110.65BB 5-9-2, TBL 5-9-1 |
| DA check | vertical ±22 m of the published glidepath; lateral inside the FAS cone at the DA distance | D38 (`evaluation/thresholds.py`; FAA Order 8260.58D Formula 3-1-1) |
```

**New text:**

```text
| Runway limit (a crossing of another candidate) | FAS half-width at the threshold, 106.7 m | FAA Order 8260.58D Formula 3-1-1 |
| Lined up | track within 30° of the course | 7110.65BB 5-9-2, TBL 5-9-1 |
| Landing | the evaluation's verdict on the approach crossing: lateral within half the runway's width, height within ±22 m of LTP + TCH, speed within the flown type's window | D196 (`evaluation/metrics.py` `evaluate_record`) |
| DA record | vertical ±22 m of the published glidepath; lateral inside the FAS cone at the DA distance; decides no outcome | D38, D196 (`evaluation/thresholds.py`; FAA Order 8260.58D Formula 3-1-1) |
```

### 18. `vocabulary.md`: §8 tolerances

**What changed:** §8 closed-loop tolerances: Y_final, H_final 7.5 m, the errors ahead (D197)

**Old text:**

```text
| Closed-loop reading: tolerances | lateral Y = 30 m, vertical H = 15 m; vertical H_final = 10 m while "no level-off" is in force (the user's choice from A24); a correction ends below half the tolerance in force or at a change of sign; a change of sign beyond the tolerance says the opposite correction in the same row | D32, D53, D66 |
```

**New text:**

```text
| Closed-loop reading: tolerances | lateral Y = 30 m, vertical H = 15 m; while "no level-off" is in force lateral Y_final = 15 m, vertical H_final = 7.5 m; a correction starts on the error Δ/2 + L ahead and ends where the error Δ/2 + L/2 ahead reaches zero or changes sign; a change of sign beyond the tolerance says the opposite correction in the same row | D32, D53, D66, D197 |
```

### 19. `outline.md`: D85

**What changed:** outline D85: stage A's labelling reads its val days (D198)

**Old text:**

```text
| D85 | A readout that serves a choice of the user reports the train and select days only. Where a build needs the validation days (the closed-loop sentences of val), it writes them, but no report, summary or printed text shows a reading of them before the stage's one validation readout. Why: the user chose Δ = 4 s (vocabulary D11) on a report that showed the val rows of the readings of D34 (`readouts/2026-10-04_stage_a_a25_report.zh.md`); the train and select rows give the same order, but the validation days are read one time for each stage | Decided | User, 2026-10-05, on Claude's review of stage A |
```

**New text:**

```text
| D85 | A readout that serves a choice of the user reports the train and select days only. Where a build needs the validation days (the closed-loop sentences of val), it writes them, but no report, summary or printed text shows a reading of them before the stage's one validation readout. Why: the user chose Δ = 4 s (vocabulary D11) on a report that showed the val rows of the readings of D34 (`readouts/2026-10-04_stage_a_a25_report.zh.md`); the train and select rows give the same order, but the validation days are read one time for each stage. Stage A's labelling reads its val days (D198) | Decided | User, 2026-10-05, on Claude's review of stage A |
```

### 20. `prior.md`: §4 Decoding (to correct or to go around)

**What changed:** prior: the landing verdict judges it (vocabulary D196)

**Old text:**

```text
executor does not have (D9). Whether to correct or to go around is the model's decision, and the DA check judges it
(vocabulary §6, item 6).
```

**New text:**

```text
executor does not have (D9). Whether to correct or to go around is the model's decision, and the judge's landing
verdict judges it (vocabulary §6, item 6; D196).
```

### 21. `post_training.md`: §2 Post-training (the reward's reasons, D30)

**What changed:** post-training: `unstable_at_minimums` → `landing_failed` (vocabulary D196)

**Old text:**

```text
   The reward gives a go-around that the DA check needs without a term of its own. An unstable approach that continues
   ends as `unstable_at_minimums`: 0.
```

**New text:**

```text
   The reward gives a go-around that a failing approach needs without a term of its own. An approach that continues
   and fails the evaluation's verdict ends as `landing_failed`: 0 (vocabulary D196).
```

### 22. `multi_control.md`: §2.1 One aircraft: D30, unchanged (D140)

**What changed:** multi-control: `unstable_at_minimums` → `landing_failed` (vocabulary D196)

**Old text:**

```text
landing (0.9·p < p); a chain of go-arounds costs; an unstable approach that continues ends as `unstable_at_minimums`.
```

**New text:**

```text
landing (0.9·p < p); a chain of go-arounds costs; an approach that continues and fails the evaluation's verdict ends as
`landing_failed` (vocabulary D196).
```

### 23. `executor_first_principles.zh.md`: §7.8 为什么删除草案 3 的 γ̇_c, item 1

**What changed:** executor first principles: H_final 10 → 7.5 m (vocabulary D66)

**Old text:**

```text
飞机比词说的线高约 18 米，大于 H_final = 10 米。
```

**New text:**

```text
飞机比词说的线高约 18 米，大于 H_final = 7.5 米（词表 D66）。
```

### 24. `executor_first_principles.zh.md`: §10.2 待验证项, V14 pass condition

**What changed:** executor first principles V14: the pass condition reads H_final = 7.5 m (vocabulary D66)

**Old text:**

```text
过冲在 τ_n 取 0.4 到 1.6 秒时都小于 H_final = 10 米 |
```

**New text:**

```text
过冲在 τ_n 取 0.4 到 1.6 秒时都小于 H_final = 7.5 米（词表 D66） |
```

### 25. `vocabulary.md`: §5.8 the DA record, lateral

**What changed:** §5.8 DA record, lateral: the reason 'narrower than the closed loop's 30 m tolerance' removed (Y_final = 15 m is in force at the DA point, D66); the evaluation's lateral gate is read at the threshold (D196)

**Old text:**

```text
- **Lateral:** the lateral offset inside the FAS cone at the distance of the DA point (FAA Order 8260.58D Formula 3-1-1,
  `flight_scenarios/fas_geometry.py`), the same geometry as the runway limit of the crossing. The lateral bound of the
  evaluation module (half the runway width, 15.24–22.86 m) is a landing-geometry criterion at the threshold (evaluation
  EV1). At the DA point, 0.9–2.1 km before the threshold, it would be narrower than the lateral tolerance of the
  closed-loop reading (30 m, D32): a sentence that follows a real flight could then fail the check.
```

**New text:**

```text
- **Lateral:** the lateral offset inside the FAS cone at the distance of the DA point (FAA Order 8260.58D Formula 3-1-1,
  `flight_scenarios/fas_geometry.py`), the same geometry as the runway limit of a candidate. The lateral gate of the
  evaluation module (half the runway width, 15.24–22.86 m) is a landing-geometry criterion at the threshold (evaluation
  EV1); the landing verdict reads it there (D196).
```

### 26. `vocabulary.md`: §2 Terms, Judge

**What changed:** §2 Terms: a go-around answers an approach that would fail the landing verdict, not a failed DA check (D196)

**Old text:**

```text
A go-around before the DA point answers a failed DA check.
```

**New text:**

```text
A go-around before the DA point answers an approach that would fail the landing verdict (D196).
```

### 27. `vocabulary.md`: §4.9 item 2 (who is flown)

**What changed:** the approach speed a flight needs is the flown type's (the user, 2026-10-13)

**Old text:**

```text
   `group_of`). A flight that the replay does not fly (no identified type, no aircraft dynamics, or no published
   approach speed) gives no training sentence.
```

**New text:**

```text
   `group_of`). A flight that the replay does not fly (no identified type, no aircraft dynamics, or no published
   approach speed of the type it is flown on) gives no training sentence.
```

### 28. `vocabulary.md`: §5.2 what the executor reads

**What changed:** the executor reads the flown type's published speeds for the middle of the evaluation's window (the user, 2026-10-13)

**Old text:**

```text
| The aircraft: aerodynamic data, installed thrust, published maximum landing mass, published approach speed | Dynamics and "unspecified" speed |
```

**New text:**

```text
| The aircraft flown: aerodynamic data, installed thrust, published maximum landing mass, published approach speeds | Dynamics and "unspecified" speed (the middle of the evaluation's speed window, §5.6) |
```

### 29. `vocabulary.md`: §5.6 speed law, "unspecified"

**What changed:** "unspecified" is the middle of the evaluation's speed window, from the flown type (the user, 2026-10-13)

**Old text:**

```text
the replay counts it. "Unspecified" is the published approach speed of the type (an indicated airspeed at the maximum
landing mass), scaled by √(m / maximum landing mass) to the mass of the aircraft and flown as the true airspeed at the
present density. A flight on a stand-in's dynamics has the stand-in's mass, so it takes the published speed of its own
type unscaled.
```

**New text:**

```text
the replay counts it. "Unspecified" is the middle of the evaluation's speed window (`evaluation.speed_gate`; executor
design §8.1; the user, 2026-10-13), from the published approach speeds of the type flown (a stand-in's on a stand-in's
dynamics, as the evaluation reads it): the lower edge V_REF,low·√(m / MALW)·√max(n, 1), the upper edge
V_REF,high·√(m / MALW) + 20 kt, at the mass and the load factor now, flown as the true airspeed at the present
density.
```

### 30. `vocabulary.md`: O22

**What changed:** O22 closed: the executor reads the flown type's window, the one the evaluation grades (the user, 2026-10-13)

**Old text:**

```text
| O22 | A flight on a stand-in's dynamics flies the identified type's approach speed, but the evaluation's speed gate reads the flown stand-in's window (v13 replay sample, train, Δ = 2 s: 23 of 98 stand-in flights fail the speed gate) | The user's choice: the speed gate on the identified type, or the stand-in's own approach speed | 5.8 |
```

**New text:**

(deleted)

### 31. `executor_first_principles.zh.md`: §0.2 I3

**What changed:** I3: the flown type's published approach speeds (two flap configurations) and MALW

**Old text:**

```text
| I3 | 机体数据 | 气动参数、安装推力、型号的进近速度 |
```

**New text:**

```text
| I3 | 机体数据 | 所飞机型的气动参数、安装推力、公布的进近速度（低、高两种襟翼形态）与最大着陆重量 |
```

### 32. `executor_first_principles.zh.md`: §8.1 目标速度 V*

**What changed:** "unspecified" = the middle of the evaluation's speed window, flown type, mass and n now (the user, 2026-10-13)

**Old text:**

```text
| "unspecified" | max(V_REF · √max(n,1) · √(ρ_0/ρ)，V_stall(n)) |
| "unspecified" 且复飞状态 G | 听到复飞时的空速，保持不变 |

- V_REF：型号公布的进近速度（指示空速），按 √(m / MALW) 缩放。乘 √(ρ_0/ρ) 换成真实空速。
- √n 的依据：失速速度与 √n 成正比。V_REF 是 1 g 的参考速度。这是物理，不是设定值。
- V_stall(n)：失速速度，物理极限。本设计不加裕度。
```

**New text:**

```text
| "unspecified" | max(V_U · √(ρ_0/ρ)，V_stall(n)) |
| "unspecified" 且复飞状态 G | 听到复飞时的空速，保持不变 |

- V_U：评分速度窗口的中点（`evaluation.speed_gate`；用户 2026-10-13："它们应该是一致的"）。V_U = (V_下 + V_上) / 2，
  V_下 = V_REF,低 · √(m / MALW) · √max(n,1)，V_上 = V_REF,高 · √(m / MALW) + 20 节。m 是当前质量，n 是当前载荷系数。
  乘 √(ρ_0/ρ) 换成真实空速。
- V_REF,低、V_REF,高：所飞机型公布的进近速度（指示空速，最大着陆重量下），低、高两种襟翼形态；只公布一个值的机型两者相同
  （193 种里 183 种）。替身动力学的航班用替身机型的速度和质量：评分读的也是它。
- √n 的依据：失速速度与 √n 成正比。V_REF 是 1 g 的参考速度。这是物理，不是设定值。
- 为什么取中点：下沿 V_REF·√n 是窗口的边。以下沿为目标时，入口处一点坡度（√n 抬高下沿）、质量的变化或 0.1 m/s 的波动
  就出窗（v15 回放：速度失败 83 % 差不到 0.5 m/s）。中点两侧各留约 10 节。
- V_stall(n)：失速速度，物理极限。本设计不加裕度。V_U 在窗口两沿之间，上沿 V_REF + 20 节是稳定进近的上限，远低于襟翼
  限速和 VMO；动力学没有 VMO，上限只有推力范围。
```

### 33. `executor_first_principles.zh.md`: §8.2.2 目标表

**What changed:** the table follows §8.1

**Old text:**

```text
| "unspecified" | max(V_REF·√max(n,1)·√(ρ_0/ρ)，V_stall(n)) |
```

**New text:**

```text
| "unspecified" | max(V_U·√(ρ_0/ρ)，V_stall(n))（§8.1） |
```

### 34. `executor_first_principles.zh.md`: §10.2 V2

**What changed:** V2 checks the middle against the lower edge (v15)

**Old text:**

```text
| V2 | "unspecified" 目标 V_REF·√n，不加余量 | 速度门失败数 | 失败数不高于加 2 m/s 的版本 |
```

**New text:**

```text
| V2 | "unspecified" 目标为评分速度窗口的中点（§8.1） | 速度门失败数（闭环与回放） | 失败数低于以下沿为目标的版本（v15） |
```

### 35. `executor_first_principles.zh.md`: §13 与评分的关系

**What changed:** the executor reads the evaluation's speed window, and no other number of it

**Old text:**

```text
执行器不读评分模块的数。管子半宽 ε 也只用于验收（§7.1）。

**说明：** "unspecified" 的目标用 V_REF·√n。评分窗口的下沿也是这个值。这是同一个物理公式的两次使用。执行器不读窗口的上沿（V_REF + 20 节）。
```

**New text:**

```text
执行器只读评分模块的一样东西：速度窗口（`evaluation.speed_gate` 的下沿公式和上沿的 20 节）。"unspecified" 的目标是它的
中点（§8.1，用户 2026-10-13）。其余的评分数不读。管子半宽 ε 也只用于验收（§7.1）。
```

### 36. `vocabulary.md`: §5.5 Vertical law (the modes and the inner loop)

**What changed:** the vertical law as the executor of 2026-10-12 flies it (executor design §7, D5, D6): "level" flies 0° and does not fly to T, a ramp over L_γ, a physical stop and a captured hold; no τ_h, no γ̇_max. This text had not followed the executor

**Old text:**

```text
The inner path-angle loop: γ̇* = sat((γ_ref − γ) / τ_γ, ±γ̇_max), τ_γ = 2 s. γ̇_max is twice the smallest rate that
keeps an entry into the steepest descent class inside its tube: 2 × V·γ_lo² / (2ε), with γ_lo the lower edge of the
steepest class and ε the narrowest level band of the grid. A level T is the height T + E MSL (D58); h is the height of
the aircraft above E. The modes choose γ_ref (climbing positive):

| Words in force | Mode | γ_ref |
|---|---|---|
| Level T + level | Hold | −(h − T) / (V·τ_h), τ_h = 4·τ_γ = 8 s; within the nominal angle of the steepest descent class and the climb angle (γ_GA while G is true) |
| Level T + descent class | Descend to T | −nominal angle of the class; level-off starts at V·γ²/(2·γ̇_max) above T |
| Level T + climb, G false | Climb to T | +nominal angle of the climb class (D15, D28); level-off as above |
| Level T + climb, G true | Go-around climb to T | +γ_GA, the go-around angle (below); level-off as above |
| "No level-off" + descent class | Descend | −nominal angle of the class; no level-off |

A target that the executor has captured (its level-off started) stays captured until a new altitude word or angle word,
so the mode does not change back and forth at the level.
```

**New text:**

```text
The executor design §7 gives the law (`autopilot/vertical.py`). The words give a class angle γ_k: the nominal angle
of the descent class, the nominal angle of the climb class (D15), γ_GA for a climb while G is true (below), or 0 for the
angle word "level". Unless "no level-off" is in force, the words also give a level T, flown at T + E MSL (D58); h is the
height of the aircraft above E. T and h give the direction; the angle word gives the steepness. The goal angle
(climbing positive):

| Words in force | Goal angle |
|---|---|
| The angle word "level" | 0. The executor does not fly to T: "level" is said only at the first step, and its T is the present height on the grid (executor design D5) |
| "No level-off" + descent class | −γ_k. No level-off |
| Level T + a class | −sign(h − T)·γ_k |

1. **Ramp.** A new altitude word or angle word starts a clock. The executor reaches the goal angle L_γ after it hears
   the word and spreads the change over the time that is left (one cycle at least).
2. **Stop (executor design D6).** Toward a level, the path angle is not more than the largest angle from which the
   aircraft can still stop before T at its physical limits: a pull to min(n_max, n_stall), a push to n_min, with the
   lag of the load factor and the cycle. A closed form gives this angle. Each approximation in it makes the stop
   earlier, not later.
3. **Captured.** A level whose stop has started, and the angle word "level", stays captured until a new altitude word
   or angle word. The executor then flies −sign(h − T)·min(γ_k, γ_stop). While a level is captured, the closed-loop
   reading starts no angle correction (§4.9).
4. The inner loop flies γ̇* = (γ_target − γ) / Δt. The inverse solves the load factor for it, within its physical
   limits. The loop has no path-rate limit of its own.
```

### 37. `vocabulary.md`: §4.4 Pieces, item 1 (Level)

**What changed:** a near-level piece (within 0.5° of level, at least 20 s) is a level too (D199)

**Old text:**

```text
1. **Level.** A piece is level when it lasts at least 20 s and every row is within 25 m of the piece's own median. This
   test is physical and does not use the grid: a test against the 60–450 m steps would miss a level between two grid
   levels (a level at 630 m is 30 m from both neighbours) or take a slow descent for a level. The word of a level is the
   grid level nearest to its median. Consecutive level pieces with the same word are one level.
```

**New text:**

```text
1. **Level.** A piece is level when it lasts at least 20 s and every row is within 25 m of the piece's own median, or
   when its path angle is within 0.5° of level (D199). This test is physical and does not use the grid: a test against
   the 60–450 m steps would miss a level between two grid levels (a level at 630 m is 30 m from both neighbours) or take
   a slow descent for a level. The angle test reads a near-level piece as a level, not as a descent of class 1 (nominal
   1.5°). The word of a level is the grid level nearest to its median. Consecutive level pieces with the same word are
   one level.
```

### 38. `vocabulary.md`: §4.9 Lateral correction, items 1 and 3

**What changed:** a new observed heading word neither ends nor stops a heading correction (D199)

**Old text:**

```text
1. When |e_y,on| > Y, and the row says no new observed heading word, the labeller says the heading class one step (5°)
   from the observed word in force, toward the observed path.
2. When e_y,off is zero or has the other sign, the labeller says the observed word in force again — except an
   overshoot (D53): when |e_y,off| > Y, it says the opposite correction in the same row (item 1 toward the other side).
3. A new observed heading word ends a correction. The labeller says the observed word, and the comparison continues.
```

**New text:**

```text
1. When |e_y,on| > Y, the labeller says the heading class one step (5°) from the observed word in force, toward the
   observed path.
2. When e_y,off is zero or has the other sign, the labeller says the observed word in force again — except an
   overshoot (D53): when |e_y,off| > Y, it says the opposite correction in the same row (item 1 toward the other side).
3. A new observed heading word does not end a correction and does not stop one from starting (D199). While a
   correction is in force, the row says the new observed word one class toward the observed path, and this word is a
   correction. When the correction ends at that row, the row says the new observed word.
```

### 39. `vocabulary.md`: §4.9 Vertical correction, item 2

**What changed:** a vertical correction moves two classes while its start error is more than 2 H (D199)

**Old text:**

```text
2. When e_h,on > H (too high), the labeller says the next steeper descent class. When e_h,on < −H (too low), it says
   the next shallower descent class. From descent 4 there is no steeper class, and from descent 1 no shallower one: then
   there is no correction.
```

**New text:**

```text
2. When e_h,on > H (too high), the labeller says the next steeper descent class. When e_h,on < −H (too low), it says
   the next shallower descent class. From descent 4 there is no steeper class, and from descent 1 no shallower one: then
   there is no correction. While a correction is in force and its start error on the side it answers is more than 2 H,
   the labeller says the class two steps from the observed class, where that class exists; when the start error is
   within 2 H again, one step (D199).
```

### 40. `vocabulary.md`: §4.9 Tolerances (D32, D66)

**What changed:** the factor of the two-class step is a constant of the spec too

**Old text:**

```text
**Tolerances (D32, D66).** Y = 30 m, H = 15 m, Y_final = 15 m and H_final = 7.5 m, constants of the labeller in the
spec.
```

**New text:**

```text
**Tolerances (D32, D66).** Y = 30 m, H = 15 m, Y_final = 15 m and H_final = 7.5 m, and the factor 2 of the two-class
step (D199), constants of the labeller in the spec.
```

### 41. `vocabulary.md`: §9.7, the item on one correction class

**What changed:** the rule this measurement ran under is marked as the rule before D199

**Old text:**

```text
- One correction class (5°) closes 6–9 m/s at a ground speed of 70–100 m/s: 2 km in 4–5 minutes. A new observed heading
  word ends a correction (§4.9).
```

**New text:**

```text
- One correction class (5°) closes 6–9 m/s at a ground speed of 70–100 m/s: 2 km in 4–5 minutes. A new observed heading
  word ended a correction then (§4.9 before D199).
```

### 42. `prior.md`: D39

**What changed:** the variant is `full`; `constants` is not run (B5's score, D24)

**Old text:**

```text
| D39 | The prior's design is chosen by leave-one-airport-out cross-validation (5 folds: train on four airports, read the fifth). Two variants: `full` (§2) and `constants` (`full` and, in each candidate vector, the runway's length and threshold elevation). `constants` is chosen only if it is better by more than twice the seed scale (§5) | Decided | User, 2026-10-04 |
```

**New text:**

```text
| D39 | The prior's design is chosen by leave-one-airport-out cross-validation (5 folds: train on four airports, read the fifth). The variant is `full` (§2). The variant `constants` (`full` and, in each candidate vector, the runway's length and threshold elevation) is not run: in B5 (on `v12_20261005`) it scored worse than `full` by 6 seed scales, and a runway constant is an airport identity (D24) | Decided | User, 2026-10-04; `constants` not run: user, 2026-10-13 |
```

### 43. `prior.md`: D40

**What changed:** configurations A and C only; B and D are not run (B5's scores); the tie rule, which only A and D could need, goes

**Old text:**

```text
| D40 | Hyperparameters: configuration A (§5) is the start. The same folds compare four configurations (A, a smaller model, a larger model, a stronger regularization); the one with the fewest parameters within twice the seed scale of the best is chosen (§5); a tie of parameters among them goes to the lower score (A and D have one shape). Every configuration has attention heads 32 wide: A 6 heads, B (d_model 128) 4, C (d_model 256) 8 | Decided | User, 2026-10-04; the head width: user, 2026-10-04, on Claude's proposal; the tie: user, 2026-10-05 |
```

**New text:**

```text
| D40 | Hyperparameters: configuration A (§5) is the start. The same folds compare A and a larger model, C; the one with the fewest parameters within twice the seed scale of the best is chosen (§5). Every configuration has attention heads 32 wide: A 6 heads, C (d_model 256) 8. A smaller model (B, d_model 128) and a stronger regularization (D) are not run: in B5 (on `v12_20261005`) B scored worst, by 21 seed scales, and D as A (`prior_base_20261006/choice_configuration.json`) | Decided | User, 2026-10-04; the head width: user, 2026-10-04, on Claude's proposal; A and C only: user, 2026-10-13 |
```

### 44. `prior.md`: §5 Cross-validation, the steps

**What changed:** two configurations (A, C), no variant step: 15 runs and the base

**Old text:**

```text
| 1 | Variant `full`, four configurations, each on the 5 folds, seed 1337. A: the start. B: d_model 128, 4 heads, feed-forward 512. C: d_model 256, 8 heads, feed-forward 1,024. D: dropout 0.2, weight decay 0.05. B, C and D have the layers and the learning rate of A; every head is 32 wide | 20 |
| 2 | The seed scale: configuration A with the seed 2024, on the 5 folds | 5 |
| 3 | Variant `constants` with the configuration chosen after step 2, on the 5 folds, seed 1337 | 5 |
| 4 | The base: the chosen configuration and variant on all five airports, seed 1337, stopped on their select days; then its one validation readout | 1 |
```

**New text:**

```text
| 1 | Two configurations, each on the 5 folds, seed 1337. A: the start. C: d_model 256, 8 heads, feed-forward 1,024, the layers and the learning rate of A; every head is 32 wide | 10 |
| 2 | The seed scale: configuration A with the seed 2024, on the 5 folds | 5 |
| 3 | The base: the chosen configuration on all five airports, seed 1337, stopped on their select days; then its one validation readout | 1 |
```

### 45. `prior.md`: §5 Cross-validation, the rules

**What changed:** no tie rule; the variant is given, not chosen

**Old text:**

```text
- **Configuration:** among the configurations whose score is within twice the seed scale of the best (the bound
  included), the one with the fewest parameters; a tie of parameters goes to the lower score (A and D have one shape)
  (D40).
- **Variant:** `constants` only if its score is lower than that of `full` (same configuration) by more than twice the
  seed scale; else `full`.
```

**New text:**

```text
- **Configuration:** among the configurations whose score is within twice the seed scale of the best (the bound
  included), the one with the fewest parameters (D40).
- **Variant:** `full` in every run (D39).
```

### 46. `prior.md`: §5 The base's one validation readout

**What changed:** the base reads the val days once for each labelled artefact (D85)

**Old text:**

```text
- **The base's one validation readout (D85):** on the validation days of every airport,
```

**New text:**

```text
- **The base's one validation readout (D85: one read for each labelled artefact):** on the validation days of every
  airport,
```

### 47. `prior.md`: §12 Rules, the formal runs

**What changed:** the formal runs read the labels of D196–D199 (B15)

**Old text:**

```text
The formal runs read `v12_20261005` / `v17_20261005` at Δ = 4 s. The rules of outline §5 apply.
```

**New text:**

```text
The formal runs read `v17_20261013` / `v22_20261013` at Δ = 4 s (B15); B5 read `v12_20261005` / `v17_20261005`. The rules
of outline §5 apply.
```

### 48. `prior.md`: §0.2 Open items

**What changed:** the results page reads a variant choice that B15's campaign does not write

**Old text:**

```text
### 0.2 Open items

None.
```

**New text:**

```text
### 0.2 Open items

- The Training results page reads a campaign's `choice_variant.json` (`aeroviz_backend/training_results.py`); a B15
  campaign (`ts-prior-campaign-v4`) has none. A change of the backend and the frontend: the user's last stage, with
  the user's permission.
```

### 49. `outline.md`: D85

**What changed:** one val read for each stage and each labelled artefact

**Old text:**

```text
the train and select rows give the same order, but the validation days are read one time for each stage. Stage A's labelling reads its val days (D198) | Decided | User, 2026-10-05, on Claude's review of stage A |
```

**New text:**

```text
the train and select rows give the same order, but the validation days are read one time for each stage and each labelled artefact (new labels are a new data set: their base has its own one read). Stage A's labelling reads its val days (D198) | Decided | User, 2026-10-05, on Claude's review of stage A; one read for each labelled artefact: user, 2026-10-13 |
```

### 50. `outline.md`: §1 `vocabulary.md` state

**What changed:** the formal artefact is v17_20261013 / v22_20261013

**Old text:**

```text
the formal artefact `v12_20261005` / `v17_20261005` and the Training view (`closed_loop_v12_20261005`) done,
```

**New text:**

```text
the formal artefact `v17_20261013` / `v22_20261013` (D196–D199, on `two-tier2`; before it `v12_20261005` / `v17_20261005`, with its Training view `closed_loop_v12_20261005`) done,
```

### 51. `outline.md`: §1 `prior.md` state

**What changed:** B15 is built

**Old text:**

```text
B6's sets published (`prior_sets_20261006`);
```

**New text:**

```text
B6's sets published (`prior_sets_20261006`); B15, the campaign on `v17_20261013` (configurations A and C), built on `two-tier2`;
```

### 52. `vocabulary.md`: §0.3 the formal artefact

**What changed:** the formal artefact is v17_20261013 / v22_20261013

**Old text:**

```text
| The formal artefact | `instruction_language/v12_20261005`, `executor/v17_20261005` (A34; read-only, `SHA256SUMS`);
```

**New text:**

```text
| The formal artefact | `instruction_language/v17_20261013`, `executor/v22_20261013` (D199, the next row; read-only, `SHA256SUMS`), read by the prior's B15. Before it: `instruction_language/v12_20261005`, `executor/v17_20261005` (A34; read-only, `SHA256SUMS`);
```

### 53. `vocabulary.md`: §0.3 D196–D199 row, its last sentences

**What changed:** the frontend and backend part is committed (288b779b)

**Old text:**

```text
The frontend and backend part of D196 is not committed: the user's last stage. Not merged: the user merges |
```

**New text:**

```text
The frontend and backend part: `288b779b`. The local branch `two-tier2` holds the branch; not pushed |
```

### 54. `vocabulary.md`: §4.9 The comparison, the lateral error e_y

**What changed:** on the final, once the observed aircraft is lined up, the lateral corrections aim at the centreline (D201)

**Old text:**

```text
  segment where the path turns there by more than 170° (D83);
```

**New text:**

```text
  segment where the path turns there by more than 170° (D83); on the final, while "no level-off" is in force and the
  observed heading word in force says the course of the runway in force (the observed aircraft is lined up), the
  signed distance of the flown position from that runway's extended centreline, positive to the right (D201);
```

### 55. `vocabulary.md`: §4.9 The errors ahead (D197), ė_y and the end error

**What changed:** on the final, once the observed aircraft is lined up, the lateral corrections aim at the centreline (D201)

**Old text:**

```text
changes now: ė_y is the flown velocity across the matched segment of the observed path (positive to the right), ė_h the
```

**New text:**

```text
changes now: ė_y is the flown velocity across the matched segment of the observed path, or on the final across the
centreline (D201) (positive to the right), ė_h the
```

### 56. `vocabulary.md`: §4.9 The errors ahead (D197), the end error

**What changed:** on the final, once the observed aircraft is lined up, the lateral corrections aim at the centreline (D201)

**Old text:**

```text
where the end error is zero or has the other sign: the end aims at the observed path. At ė = 0 the start error is the
```

**New text:**

```text
where the end error is zero or has the other sign: the end aims at the path e_y is read from (the observed path; on
the final, the centreline, D201). At ė = 0 the start error is the
```

### 57. `vocabulary.md`: §4.9 Lateral correction, item 1

**What changed:** on the final, once the observed aircraft is lined up, the lateral corrections aim at the centreline (D201)

**Old text:**

```text
1. When |e_y,on| > Y, the labeller says the heading class one step (5°) from the observed word in force, toward the
   observed path.
```

**New text:**

```text
1. When |e_y,on| > Y, the labeller says the heading class one step (5°) from the observed word in force, toward the
   observed path (on the final, toward the centreline: D201).
```

### 58. `vocabulary.md`: §4.9 Lateral correction, item 3

**What changed:** on the final, once the observed aircraft is lined up, the lateral corrections aim at the centreline (D201)

**Old text:**

```text
   correction is in force, the row says the new observed word one class toward the observed path, and this word is a
```

**New text:**

```text
   correction is in force, the row says the new observed word one class toward the path e_y is read from, and this word is a
```

### 59. `vocabulary.md`: §0.1 D83, its first sentence

**What changed:** on the final, once the observed aircraft is lined up, the lateral corrections aim at the centreline (D201)

**Old text:**

```text
The lateral error e_y of the closed-loop reading is the signed distance of the flown position from the observed path: from the line through the observed 2 s rows near the matched point (its segments, not their extensions), positive to the right of the path (§4.9).
```

**New text:**

```text
The lateral error e_y of the closed-loop reading is the signed distance of the flown position from the observed path: from the line through the observed 2 s rows near the matched point (its segments, not their extensions), positive to the right of the path (§4.9); on the final, once the observed aircraft is lined up, e_y is read from the runway's centreline instead (D201).
```
