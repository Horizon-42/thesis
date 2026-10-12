# Superseded decisions and passages: an index for review

**What this file is.** The two-tier design documents hold the current design only. On 2026-10-12 they were rewritten in
place to match the code of the branch `two-tier2-eval-landing` (commits `04378395`, `ea632d20`, `51a68b8e`). This file
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

The new decision rows (vocabulary D196, D197; outline D198) are new text, not replacements, so they are not entries
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
| 9 | `vocabulary.md` | §4.9 lateral and vertical correction | §4.9 lateral and vertical corrections: the tolerance in force (Y_final) and the start / end errors (D197) |
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

## Text kept unchanged on purpose

- `vocabulary.md` §9 (the evidence of stage A's milestones, for example the replay outcomes `crossed_off_runway`,
  `unstable_at_minimums` and `crossed_too_high` of A15–A24, and the DA check failures of select at Δ = 2 s): these are
  measurements made with the judge of their time. They stay as measured.
- `frontend.md`: the Training view still draws the DA point, its chip and its label as before; the evaluation's verdict
  is added as text only. That frontend change is the user's last stage and is not committed, so the view's design is
  not rewritten yet.
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

**What changed:** §4.9 lateral and vertical corrections: the tolerance in force (Y_final) and the start / end errors (D197)

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
as at the rate now. At ė = 0 both are the error now.

**Lateral correction.** The lateral tolerance in force is Y_final while "no level-off" is in force (the final descent)
and Y elsewhere (D66). In the rules below, Y is the tolerance in force, e_y,on the start error and e_y,off the end error.

1. When |e_y,on| > Y, and the row says no new observed heading word, the labeller says the heading class one step (5°)
   from the observed word in force, toward the observed path.
2. When |e_y,off| < Y / 2, or e_y,off has the other sign, the labeller says the observed word in force again — except an
   overshoot (D53): when e_y,off has the other sign and |e_y,off| > Y, it says the opposite correction in the same row
   (item 1 toward the other side).
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
3. When |e_h,off| < H / 2, or e_h,off has the other sign, the labeller says the observed angle class in force again —
   except an overshoot (D53): when e_h,off has the other sign and |e_h,off| > H, it says the opposite correction in the
   same row (item 2, where that class exists). A new observed altitude word or angle word ends a correction.
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
| Closed-loop reading: tolerances | lateral Y = 30 m, vertical H = 15 m; while "no level-off" is in force lateral Y_final = 15 m, vertical H_final = 7.5 m; a correction starts on the error Δ/2 + L ahead and ends below half the tolerance in force, or at a change of sign, on the error Δ/2 + L/2 ahead; a change of sign beyond the tolerance says the opposite correction in the same row | D32, D53, D66, D197 |
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
