# Superseded decisions and passages: an index for review

**What this file is.** The two-tier design documents hold the current design only. On 2026-10-12 they were rewritten in
place to match the code of the branch `two-tier2-eval-landing` (commits `04378395`, `ea632d20`, `51a68b8e`), and on
2026-10-13 again for commits `e00c8d08` and `d8f21c77`, for the prior's campaign on the new labels (`7e72a059`,
B15), for the final's lateral corrections (D201, the branch `two-tier2-centreline`), and for the branch rule and
the frontend and backend rule (outline §5 rules 1, 8, 10, 11; the branch `two-tier2-backend-orphans`); on 2026-10-14
for the final's vertical corrections (D203, the branch `two-tier2-glidepath`), the prior's lateral input
scale (D204, the same branch), the prior's stop (D205, the same branch), the prior's landings input (D206, the
same branch) the reward's landing direction (post-training D207, the branch `two-tier2-reward`), the loops' landings
(post-training D208, the branch `two-tier2-no-loop-landings`), the landings in a prior's identity (prior D63, the
branch `two-tier2-no-landings-digest`) and the entries a prior's checkpoint opens on (prior D209, the same branch). This
file
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
11. One branch is kept, `two-tier2`; every other branch is temporary (outline §5 rules 1, 8, 10; 2026-10-13): "为啥要碰
   dev-two-tier？ 还有，two-tier2 维护这个就行 其他的都是临时分支。注意这很重要".
12. The frontend and the backend change last, each change with the user's permission (outline §5 rule 11; 2026-10-12):
   "以后不要做任何前端改动 服务的后端，也要我的允许才能改。因为前后端改动意味着重要的设计修改", and "为什么之前就开始改前后端了 …
   这是最后阶段的事情". On 2026-10-13 the user allowed the results route's change and its reader's ("finish backend";
   "允许（推荐）").
13. On the final the angle corrections aim at the runway's published glidepath, and the corrections go on to the
   threshold (vocabulary D203; 2026-10-14): after the check of the labels' height at the threshold, the user chose
   "先修改垂直标注，同时把标注词的修正延长至通过跑道入口".
14. The prior's offset right of a final goes in on the scale of its height above the glidepath, 100 m (prior D204;
   2026-10-13 and 10-14): "should algin lateral and vertical input, since they have similar accuracy requirement",
   then, for the cross-validation on the labels of D203, "/100 是定了的，一起上".
15. The prior's training stops after 15 epochs without a better select loss, at most 150 epochs (prior D205;
   2026-10-14): "改成连续8轮不变小；或者你建议一个更合理的数字；另外评估一下，当前base训练是不是停早了", then the choices
   "15 轮，上限 150（推荐）" and "新规则重跑整个活动（推荐）".
16. The prior reads no landings; the loops keep them for the post-training's reward (prior D206; 2026-10-14): "这样 你先做个
   试验，直接把这个输入去掉 … 理论上为了增加泛化能力，最好能去掉它", then, after the trial, "直接采用，重跑整个活动".
17. The runway of a landing does not change the reward (post-training D207; 2026-10-14): "降落方向 不对没用奖励的根据是
   什么？", then the choice "直接去掉".
18. The landings that nothing reads go (post-training O26, D208; 2026-10-14): the choice "现在就删". What D208 keeps
   (the landings' digest in a prior's identity, the window's landings that the diagnostic readout reads) is Claude's,
   for the user to confirm.
19. The diagnostic readout keeps the window's landings, and the identity of a prior's data holds no landings
   (post-training O26, D175; prior D63; 2026-10-14): for D175, the choice "Keep (Recommended)"; for D63, after "你给我
   解释清楚 … 把模型的checkpoint和数据硬绑定的目的是什么 是一般做法吗？单纯的作为配置描述不行吗？", the choice "Drop it: not
   written, not compared (Recommended)" (B21's stored digest is kept as written and not compared).
20. A prior's checkpoint opens on the meaning and the day split (prior D209; 2026-10-14): after "Write me a proposal
   first" for the other entries of the identity, the choices "都保持必须一致（推荐）" for the spec, the candidate table, Δ and
   the day split, "改为描述" for the sha256 of the sentence and signal files, and "数目改描述，val 前仍重数（推荐）" for the
   selection's counts.

The new decision rows (vocabulary D196, D197, D201, D203; prior D204–D206, D209; post-training D207, D208; outline D198) are new text, not replacements, so they are not entries
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
| 60 | `outline.md` | §5 rule 1, the branches and the implementers | one branch, `two-tier2`; every other branch is temporary (the user, 2026-10-13) |
| 61 | `outline.md` | §5 rule 10, where the log is committed | one branch, `two-tier2`; every other branch is temporary (the user, 2026-10-13) |
| 62 | `outline.md` | §5 rule 11, the Training view and the merge | the frontend and the backend change last, each change with the user's permission (the user, 2026-10-12) |
| 63 | `outline.md` | §5 rule 8, the checkouts not to touch | one branch, `two-tier2`; every other branch is temporary (the user, 2026-10-13) |
| 64 | `prior.md` | §12 Rules, the artefact of the formal runs | the labels and the prior are built again with the final's corrections at the centreline (vocabulary D201) |
| 65 | `multi_control.md` | §0.1 D200, its last sentence | the backend's stage D route reads a start from the base (the user's permission, 2026-10-13) |
| 66 | `frontend.md` | §3 item 2, the readouts of an item (stage A, stage B) | a landing is the evaluation package's verdict on the crossing (vocabulary D196); the DA check stays a record (`288b779b`) |
| 67 | `frontend.md` | §3 item 7, one flown sentence | a landing is the evaluation package's verdict on the crossing (vocabulary D196); the DA check stays a record (`288b779b`) |
| 68 | `frontend.md` | §4.1 stage A, the sets | a landing is the evaluation package's verdict on the crossing (vocabulary D196); the DA check stays a record (`288b779b`) |
| 69 | `frontend.md` | §4.1 stage A, the view | a landing is the evaluation package's verdict on the crossing (vocabulary D196); the DA check stays a record (`288b779b`) |
| 70 | `frontend.md` | §3 item 3, the results route, stage B | the prior's campaign has no variant step since B15; the route answers the configuration choice only (`be3cdd49`) |
| 71 | `frontend.md` | §5.6 the details page of stage D | the backend's stage D route reads a start from the base (the user's permission, 2026-10-13) |
| 72 | `frontend.md` | §7, the formats | a landing is the evaluation package's verdict on the crossing (vocabulary D196); the DA check stays a record (`288b779b`) |
| 73 | `vocabulary.md` | §0.1 D44 | on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203) |
| 74 | `vocabulary.md` | §0.1 D201, the code names, the end of the path and the format | on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203) |
| 75 | `vocabulary.md` | §4.9 The comparison, the vertical error e_h | on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203) |
| 76 | `vocabulary.md` | §4.9 Past the end of the observed path (D44) | on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203) |
| 77 | `vocabulary.md` | §4.9 The errors ahead (D197), ė_h and the end error | on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203) |
| 78 | `vocabulary.md` | §4.9 The errors ahead (D197), the end error | on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203) |
| 79 | `vocabulary.md` | §4.9 Vertical correction, item 2 | on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203) |
| 80 | `vocabulary.md` | §8 Values, closed-loop reading: correction | on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203) |
| 81 | `vocabulary.md` | §8 Values, past the end of the observed path | on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203) |
| 82 | `prior.md` | §2 Inputs, Scales | the offset right of a final's centreline goes in on the glidepath's scale, asinh(d / 100 m) (D204) |
| 83 | `prior.md` | §9 Values, input scales | the offset right of a final's centreline goes in on the glidepath's scale, asinh(d / 100 m) (D204) |
| 84 | `prior.md` | §5 Training, the loop | the stop: 15 epochs without a better select loss, at most 150 epochs (D205) |
| 85 | `prior.md` | §5 Training, hyperparameters | the stop: 15 epochs without a better select loss, at most 150 epochs (D205) |
| 86 | `prior.md` | §9 Values, configuration A | the stop: 15 epochs without a better select loss, at most 150 epochs (D205) |
| 87 | `prior.md` | §2 Inputs, D23's test | the prior reads no landings; the loops keep them for the reward (D206) |
| 88 | `prior.md` | §2 Inputs, Candidates | the prior reads no landings; the loops keep them for the reward (D206) |
| 89 | `prior.md` | §2 Inputs, Landings | the prior reads no landings; the loops keep them for the reward (D206) |
| 90 | `prior.md` | §2 Inputs, Scales | the prior reads no landings; the loops keep them for the reward (D206) |
| 91 | `prior.md` | §6 A new airport, what it needs | the prior reads no landings; the loops keep them for the reward (D206) |
| 92 | `prior.md` | §7 Public interface, item 2 | the prior reads no landings; the loops keep them for the reward (D206) |
| 93 | `prior.md` | §7 Public interface, item 7 | the prior reads no landings; the loops keep them for the reward (D206) |
| 94 | `prior.md` | §9 Values, landings | the prior reads no landings; the loops keep them for the reward (D206) |
| 95 | `prior.md` | §9 Values, input scales | the prior reads no landings; the loops keep them for the reward (D206) |
| 96 | `post_training.md` | §3 Inputs, landing context | the prior reads no landings; the loops keep them for the reward (prior D206) |
| 97 | `multi_control.md` | §6.2 Compressed windows | the prior reads no landings; the loops keep them for the reward (prior D206) |
| 98 | `multi_control.md` | §9 Changes of other stages, item 3 | the prior reads no landings; the loops keep them for the reward (prior D206) |
| 99 | `post_training.md` | D105 | the runway of a landing does not change the reward (post-training D207) |
| 100 | `post_training.md` | D112 | the runway of a landing does not change the reward (post-training D207) |
| 101 | `post_training.md` | §2 item 2, Reward | the runway of a landing does not change the reward (post-training D207) |
| 102 | `post_training.md` | §3 Multi-aircraft, Landing context | the runway of a landing does not change the reward (post-training D207) |
| 103 | `post_training.md` | §5 Values, Present landing direction | the runway of a landing does not change the reward (post-training D207) |
| 104 | `post_training.md` | §9 Public interface, item 6 | the runway of a landing does not change the reward (post-training D207) |
| 105 | `multi_control.md` | D140 | the runway of a landing does not change the reward (post-training D207) |
| 106 | `multi_control.md` | D147 | the runway of a landing does not change the reward (post-training D207) |
| 107 | `multi_control.md` | §2.1 One aircraft | the runway of a landing does not change the reward (post-training D207) |
| 108 | `prior.md` | §2 Inputs, No landings | the runway of a landing does not change the reward (post-training D207) |
| 109 | `prior.md` | D206 | the runway of a landing does not change the reward (post-training D207) |
| 110 | `vocabulary.md` | O23 | the runway of a landing does not change the reward (post-training D207) |
| 111 | `post_training.md` | O26 | the loops keep no landings (post-training D208) |
| 112 | `post_training.md` | D31 | the loops keep no landings (post-training D208) |
| 113 | `post_training.md` | D105 | the loops keep no landings (post-training D208) |
| 114 | `post_training.md` | §3 Inputs, Landing context | the loops keep no landings (post-training D208) |
| 115 | `post_training.md` | §9 Public interface, item 2 | the loops keep no landings (post-training D208) |
| 116 | `post_training.md` | §9 Public interface, item 8 | the loops keep no landings (post-training D208) |
| 117 | `multi_control.md` | D147 | the loops keep no landings (post-training D208) |
| 118 | `multi_control.md` | D166 (6) | the loops keep no landings (post-training D208) |
| 119 | `multi_control.md` | D186 (1) | the loops keep no landings (post-training D208) |
| 120 | `multi_control.md` | D189 (0) | the loops keep no landings (post-training D208) |
| 121 | `multi_control.md` | §1 Scope | the loops keep no landings (post-training D208) |
| 122 | `multi_control.md` | §2.6 Compressed windows | the loops keep no landings (post-training D208) |
| 123 | `multi_control.md` | §3 The step of a window, items 5 and 6 | the loops keep no landings (post-training D208) |
| 124 | `multi_control.md` | §6.1 | the loops keep no landings (post-training D208) |
| 125 | `multi_control.md` | §6.2 row 3 | the loops keep no landings (post-training D208) |
| 126 | `multi_control.md` | §6.3 item 5 | the loops keep no landings (post-training D208) |
| 127 | `multi_control.md` | §9.3 | the loops keep no landings (post-training D208) |
| 128 | `prior.md` | D63 | the loops keep no landings (post-training D208) |
| 129 | `prior.md` | D105 | the loops keep no landings (post-training D208) |
| 130 | `prior.md` | D206 | the loops keep no landings (post-training D208) |
| 131 | `prior.md` | §2 Inputs, No landings | the loops keep no landings (post-training D208) |
| 132 | `prior.md` | §7 Public interface, item 2 | the loops keep no landings (post-training D208) |
| 133 | `prior.md` | §7 Public interface, item 7 | the loops keep no landings (post-training D208) |
| 134 | `prior.md` | §9 Values, the landings that the loops keep | the loops keep no landings (post-training D208) |
| 135 | `prior.md` | §11 Code index, the landings | the loops keep no landings (post-training D208) |
| 136 | `vocabulary.md` | O23 | the loops keep no landings (post-training D208) |
| 137 | `outline.md` | §3.2 Index | the loops keep no landings (post-training D208) |
| 138 | `outline.md` | §3.2 Index | the loops keep no landings (post-training D208) |
| 139 | `outline.md` | §3.2 Index | the loops keep no landings (post-training D208) |
| 140 | `multi_control.md` | D150 | the loops keep no landings (post-training D208) |
| 141 | `prior.md` | D63 | the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed) |
| 142 | `prior.md` | §2, No landings | the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed) |
| 143 | `prior.md` | §8 item 1 | the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed) |
| 144 | `prior.md` | code index, the landings | the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed) |
| 145 | `prior.md` | code index, the identity | the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed) |
| 146 | `prior.md` | code index, the artefact | the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed) |
| 147 | `prior.md` | code index, the checkpoint | the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed) |
| 148 | `post_training.md` | D208 | the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed) |
| 149 | `post_training.md` | O26 | the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed) |
| 150 | `post_training.md` | §3, Landings | the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed) |
| 151 | `outline.md` | index, post-training | the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed) |
| 152 | `prior.md` | D63 | a checkpoint opens on the meaning and the day split; the files' sha256 and the selection's counts describe the data (prior D209) |
| 153 | `prior.md` | D118 | a checkpoint opens on the meaning and the day split; the files' sha256 and the selection's counts describe the data (prior D209) |
| 154 | `prior.md` | §8 item 1 | a checkpoint opens on the meaning and the day split; the files' sha256 and the selection's counts describe the data (prior D209) |

## Text kept unchanged on purpose

- `vocabulary.md` §9 (the evidence of stage A's milestones, for example the replay outcomes `crossed_off_runway`,
  `unstable_at_minimums` and `crossed_too_high` of A15–A24, and the DA check failures of select at Δ = 2 s): these are
  measurements made with the judge of their time. They stay as measured.
- `frontend.md`: the Training view still draws the DA point, its chip and its label as before; the evaluation's verdict
  is added as text only (`288b779b`), and only the passages that name the verdict are rewritten.
- The status tables and plans (each document's §0.2–§0.4, the outline's §1 and §4) were brought up to the state of
  2026-10-13; git keeps their old text, and they are not entries.
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

### 60. `outline.md`: §5 rule 1, the branches and the implementers

**What changed:** one branch, `two-tier2`; every other branch is temporary (the user, 2026-10-13)

**Old text:**

```text
1. The branches. The main parts of stages A, B and C are merged into `dev-two-tier` (2026-10-06). From then on, every
   stage develops on one branch, `dev-two-tier-v4`, in the worktree `.claude/worktrees/two-tier-v4` (the user,
   2026-10-06), so that a review reads one difference (`dev-two-tier...dev-two-tier-v4`):
   - before its first commit, and before a milestone that a new design commit holds, the implementer brings
     `dev-two-tier-v4` level with `dev-two-tier` (a fast-forward, or a merge of `dev-two-tier` when both moved);
   - work begun on another branch (stage B's `dev-training-layout`) is merged into `dev-two-tier-v4` and goes on there;
     a stage branch whose work is all in `dev-two-tier` (`dev-two-tier-v4-prior`) is deleted with
     its worktree;
   - the user merges `dev-two-tier-v4` into `dev-two-tier` (rule 11).

   One implementer develops for every stage: stage D's (the user, 2026-10-06; it takes over stage B's implementer's
   remaining steps, §4 items 5 and 6). Stages A's and B's implementers have no work. Stage C's builds and runs C14, C10's
   later rounds (the user, 2026-10-06), on `dev-two-tier-v4-post` in `.claude/worktrees/two-tier-v4-post`, first
   brought level with `dev-two-tier` (a fast-forward), and changes no other code. Stage D's implementer keeps two branches: the work of §4 items 5 and 6 on `dev-two-tier-v4`
   (worktree `.claude/worktrees/two-tier-v4`), and the multi-aircraft control on its own branch `dev-multi-control` in
   the worktree `.claude/worktrees/multi-control`, made from `dev-two-tier-v4`, so that the merge of
   `dev-two-tier-v4` does not wait for the multi-aircraft control's review. It
   brings `dev-multi-control` level with `dev-two-tier-v4` (a merge) before each milestone and before its report; it
   changes the code of stages A, B and C only where multi-aircraft control §6 says, each change written in that
   stage's public interface first; the user merges `dev-multi-control`. On the host, the measurements of §4 items 5
   and 6 after C10 (rule 13) come before the multi-aircraft control's runs on real data. The implementer stages files by explicit paths and reads
   `git diff --cached --stat` before each commit. A formal build or run starts from a run worktree at the commit that
   the user merged (D163), never from a development worktree or the main checkout; it reads recorded paths through
   `this_checkout` (post-training D157).
   After the merge of §4 item 9, stage D's implementer removes the worktrees and deletes the branches whose work is all
   in `dev-two-tier` (each worktree's data links unlinked first; `git branch -d`, never `-D`).

   The Training view (`frontend.md`) has an implementer of its own, fronter (the user, 2026-10-06; frontend D154), on
   its own branch `dev-frontend` in the worktree `.claude/worktrees/frontend`, made from `dev-two-tier` and merged
   level with it before each milestone; the user merges `dev-frontend`. Fronter changes the Training view's files and
   the Training exports with their files' modules (frontend §8), and no other code of a stage. Stage D's implementer
   does not change those files; it runs the exports of A's and B's sets after C10 (§4 item 5).
```

**New text:**

```text
1. The branches (the user, 2026-10-13: "two-tier2 维护这个就行 其他的都是临时分支"). The user keeps one branch,
   `two-tier2`. A change is built on a temporary branch from `two-tier2`, in its own worktree; after its review and
   tests, `two-tier2` is moved to it by its ref (`git update-ref refs/heads/two-tier2 <new> <old>`, never a merge
   inside a checkout), and the temporary branch and its worktree are deleted (each worktree's data links unlinked
   first; `git branch -d`, never `-D`). No other branch moves: not `dev-two-tier` (the main checkout, from which the frontend and the backend
   run), not a stage branch. A push of `two-tier2` waits for the user's word. The implementer stages files by
   explicit paths and reads `git diff --cached --stat` before each commit. A formal build or run starts from a run
   worktree at a `two-tier2` commit (D163), never from a development worktree or the main checkout; it reads
   recorded paths through `this_checkout` (post-training D157).
```

### 61. `outline.md`: §5 rule 10, where the log is committed

**What changed:** one branch, `two-tier2`; every other branch is temporary (the user, 2026-10-13)

**Old text:**

```text
The log is committed to `dev-two-tier` (the user's checkout), with explicit paths.
```

**New text:**

```text
The log is committed to `two-tier2` (rule 1), with explicit paths.
```

### 62. `outline.md`: §5 rule 11, the Training view and the merge

**What changed:** the frontend and the backend change last, each change with the user's permission (the user, 2026-10-12)

**Old text:**

```text
11. The user merges a branch into `dev-two-tier`, not before the Training view of its stage (`frontend.md`): before it, the
    backend's live executor (`aeroviz_backend/autopilot_segment/`) and the Training view of the branch do not work. A
    merge also moves the main checkout's services to the new formats; the sets published in the old formats stay on
    disk, and the new frontend does not read them (principle 8). Until the user merges, the user sees a stage on a test
    stack from its worktree (frontend §2 item 5).
```

**New text:**

```text
11. The frontend and the backend change last (the user, 2026-10-12: "以后不要做任何前端改动 服务的后端，也要我的
    允许才能改。因为前后端改动意味着重要的设计修改"; "这是最后阶段的事情"). A change of `aeroviz-4d` (its code, tests and
    fixtures) or of `aeroviz_backend` waits for the user's permission, after the numbers of the main task. A change of
    a format that the Training view or the backend's live executor (`aeroviz_backend/autopilot_segment/`) reads first
    lists what it forces there; until the user allows it, its mirror tests stay red and are named in the commit. The
    sets published in an old format stay on disk, and the new frontend does not read them (principle 8). The
    services run from the main checkout, which the user moves.
```

### 63. `outline.md`: §5 rule 8, the checkouts not to touch

**What changed:** one branch, `two-tier2`; every other branch is temporary (the user, 2026-10-13)

**Old text:**

```text
8. Do not touch the checkouts of running experiments (`.claude/worktrees/step9-run` and others) or the main checkout.
```

**New text:**

```text
8. Do not touch the run worktrees of running builds and campaigns, or the main checkout.
```

### 64. `prior.md`: §12 Rules, the artefact of the formal runs

**What changed:** the labels and the prior are built again with the final's corrections at the centreline (vocabulary D201)

**Old text:**

```text
The formal runs read `v17_20261013` / `v22_20261013` at Δ = 4 s (B15); B5 read `v12_20261005` / `v17_20261005`.
```

**New text:**

```text
The formal runs read `v18_20261013` / `v23_20261013` at Δ = 4 s (B16); B15 read `v17_20261013` / `v22_20261013`, B5 `v12_20261005` / `v17_20261005`.
```

### 65. `multi_control.md`: §0.1 D200, its last sentence

**What changed:** the backend's stage D route reads a start from the base (the user's permission, 2026-10-13)

**Old text:**

```text
The backend's stage D route reads the start as a round of stage C; a campaign from the base shown in the frontend needs that route changed (the last stage, the user's permission).
```

**New text:**

```text
The backend's stage D route answers a start from the base with why it has no readout (`be3cdd49`, the user's permission).
```

### 66. `frontend.md`: §3 item 2, the readouts of an item (stage A, stage B)

**What changed:** a landing is the evaluation package's verdict on the crossing (vocabulary D196); the DA check stays a record (`288b779b`)

**Old text:**

```text
     DA check); B — this sentence (the outcome and its time, the DA check, the go-arounds) and, at the cursor, the
```

**New text:**

```text
     evaluation's verdict, the DA check); B — this sentence (the outcome and its time, the evaluation's verdict, the DA
     check, the go-arounds) and, at the cursor, the
```

### 67. `frontend.md`: §3 item 7, one flown sentence

**What changed:** a landing is the evaluation package's verdict on the crossing (vocabulary D196); the DA check stays a record (`288b779b`)

**Old text:**

```text
     the block of every sentence that the executor flew, in every stage: the outcome, the end cycle, the crossing and
     the DA check, the track
```

**New text:**

```text
     the block of every sentence that the executor flew, in every stage: the outcome, the end cycle, the crossing with
     the evaluation's verdict (`crossing.evaluation`: the verdict, the failed gates, each gate's value with its bounds)
     and the DA check, the track
```

### 68. `frontend.md`: §4.1 stage A, the sets

**What changed:** a landing is the evaluation package's verdict on the crossing (vocabulary D196); the DA check stays a record (`288b779b`)

**Old text:**

```text
  word marked, and its flown states; the judge's outcome and the DA check; the attitudes.
```

**New text:**

```text
  word marked, and its flown states; the judge's outcome (`landed` is the evaluation's verdict, vocabulary D196),
  the verdict and the DA check; the attitudes.
```

### 69. `frontend.md`: §4.1 stage A, the view

**What changed:** a landing is the evaluation package's verdict on the crossing (vocabulary D196); the DA check stays a record (`288b779b`)

**Old text:**

```text
  correction words marked; a tab for each Δ; the flown path beside the observed one; the outcome and the DA point.
```

**New text:**

```text
  correction words marked; a tab for each Δ; the flown path beside the observed one; the outcome, the evaluation's
  verdict in one line (the read-back window, the replay line) and the DA point.
```

### 70. `frontend.md`: §3 item 3, the results route, stage B

**What changed:** the prior's campaign has no variant step since B15; the route answers the configuration choice only (`be3cdd49`)

**Old text:**

```text
· **The choice**: each configuration's and variant's score over its folds, the seed scale, the chosen configuration and variant | the free-generation `readout.json` of `source.readout`; `history.json` of the prior; `validation/readout.json` beside the base; `choice_configuration.json`, `choice_variant.json` of the campaign; the speed readout that `source.speed` names |
```

**New text:**

```text
· **The choice**: each configuration's score over its folds, the seed scale, the chosen configuration (the variant is `full`, prior B15) | the free-generation `readout.json` of `source.readout`; `history.json` of the prior; `validation/readout.json` beside the base; `choice_configuration.json` of the campaign; the speed readout that `source.speed` names |
```

### 71. `frontend.md`: §5.6 the details page of stage D

**What changed:** the backend's stage D route reads a start from the base (the user's permission, 2026-10-13)

**Old text:**

```text
counted once a pair; the results route takes `stage=D` and reads stage D's campaign.
```

**New text:**

```text
counted once a pair; the results route takes `stage=D` and reads stage D's campaign. A campaign's start has no
readout of stage D's windows: a round of stage C was read on stage C's windows, and the base (multi-aircraft control
D200) on none; the route answers why.
```

### 72. `frontend.md`: §7, the formats

**What changed:** a landing is the evaluation package's verdict on the crossing (vocabulary D196); the DA check stays a record (`288b779b`)

**Old text:**

```text
replace v4: the one published set of v4, `windows_seg60_r5_20261008`, is exported again in v5 (the user,
2026-10-08).
```

**New text:**

```text
replace v4: the one published set of v4, `windows_seg60_r5_20261008`, is exported again in v5 (the user,
2026-10-08). The evaluation's landing verdict (vocabulary D196, `288b779b`) renames the samples and the live
segments: stage A's sample v13, stage B's v6, the window sample v7; the segments `aeroviz-autopilot-segment-v11`,
`aeroviz-autopilot-prior-segment-v2`, `aeroviz-autopilot-window-segment-v4`. The sets published in the older formats
are not read.
```

### 73. `vocabulary.md`: §0.1 D44

**What changed:** on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203)

**Old text:**

```text
| D44 | No correction past the end of the observed path (the observed slice ends before the threshold; the flown aircraft flies on to it). There the closed-loop reading measures the lateral error against the line of the last segment, for the readouts only, and no vertical error (there is no observed height); it says no lateral or vertical correction: a correction in force ends, and the observed word is said again; the aircraft flies the last observed words to the threshold. The rows count as rows without correction (D34). The DA point always lies before the end of the observed path (§4.9) | Decided | User, 2026-10-04 |
```

**New text:**

```text
| D44 | No correction past the end of the observed path off the final (the observed slice ends before the threshold; the flown aircraft flies on to it). There the closed-loop reading measures the lateral error against the line of the last segment, for the readouts only, and no vertical error (there is no observed height); it says no lateral or vertical correction: a correction in force ends, and the observed word is said again. The rows count as rows without correction (D34). On the final the errors are read from the centreline and the glidepath, which go on past the end, and the corrections go on to the threshold (D203) | Decided | User, 2026-10-04; on the final: user, 2026-10-14 (D203) |
```

### 74. `vocabulary.md`: §0.1 D201, the code names, the end of the path and the format

**What changed:** on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203)

**Old text:**

```text
| D201 | **On the final the lateral corrections aim at the runway's centreline.** While "no level-off" is in force and the observed heading word in force says the course of the runway in force (the observed aircraft is lined up on its final), e_y is the signed distance of the flown position from that runway's extended centreline (positive to the right) and ė_y the flown velocity across it (`closed_loop.Corrector.lateral_reference`, `closed_loop.centreline_error`); the rules of the lateral correction (§4.9) and Y_final (D66) do not change. Elsewhere e_y is D83's. Past the end of the observed path no correction is said (D44); the e_y the readouts read there is the centreline's while the last observed row is on the lined-up final. The stored e_y is the one the corrections read, so the rule of D50 and the readings of D34 read it too; the closed-loop format is `ts-instruction-closed-loop-v10`. Why: the prior sees the centreline and never the observed path. With the corrections aimed at the observed path, the labels answer the centreline offset softly: on the select days' final (2–8 km before the threshold, the course in force), rows 15–25 m off the centreline turned back in 34–38 % net (88–90 % against the observed path), because the observed tracks scatter about the centreline (median −0.7 m, 10 %–90 % −11.8 to +8.8 m) inside Y_final; B15's folds then settled 20–40 m off the centreline on an airport they did not train on and failed the evaluation's lateral gate (`readouts/2026-10-13_b15_prior.zh.md`). Measured in memory on 5,000 random train flights of v17 (1,000 an airport, seed 1337, Δ = 4 s): landed 97.66 → 98.92 % (72 flights landed that did not, 9 the reverse), heading words 48.5 → 43.1 a sentence, rows 15–25 m off the centreline turned back 98 % net; Y_final 7.5 m was worse (94.08 % alone, 95.10 % with the centreline) (`outputs/POOLED/analyses/final_centreline_labels_20261013`). The cost: the labels lie farther from the observed track (|e_y| from the observed path, the median of the flights' medians 8.3 → 11.4 m). The condition "lined up" is needed: from "no level-off" alone the turn onto the final is pulled toward the centreline (100 flights: 8 timeouts) | Decided | User, 2026-10-13 ("采用，做 v18 并重跑 B") |
```

**New text:**

```text
| D201 | **On the final the lateral corrections aim at the runway's centreline.** While "no level-off" is in force and the observed heading word in force says the course of the runway in force (the observed aircraft is lined up on its final), e_y is the signed distance of the flown position from that runway's extended centreline (positive to the right) and ė_y the flown velocity across it (`closed_loop.Corrector.final_runway`, `closed_loop.final_errors`); the rules of the lateral correction (§4.9) and Y_final (D66) do not change. Elsewhere e_y is D83's. The centreline goes on past the end of the observed path, and so do the corrections (D203). The stored e_y is the one the corrections read, so the rule of D50 and the readings of D34 read it too; the closed-loop format was `ts-instruction-closed-loop-v10` (D203: v11). Why: the prior sees the centreline and never the observed path. With the corrections aimed at the observed path, the labels answer the centreline offset softly: on the select days' final (2–8 km before the threshold, the course in force), rows 15–25 m off the centreline turned back in 34–38 % net (88–90 % against the observed path), because the observed tracks scatter about the centreline (median −0.7 m, 10 %–90 % −11.8 to +8.8 m) inside Y_final; B15's folds then settled 20–40 m off the centreline on an airport they did not train on and failed the evaluation's lateral gate (`readouts/2026-10-13_b15_prior.zh.md`). Measured in memory on 5,000 random train flights of v17 (1,000 an airport, seed 1337, Δ = 4 s): landed 97.66 → 98.92 % (72 flights landed that did not, 9 the reverse), heading words 48.5 → 43.1 a sentence, rows 15–25 m off the centreline turned back 98 % net; Y_final 7.5 m was worse (94.08 % alone, 95.10 % with the centreline) (`outputs/POOLED/analyses/final_centreline_labels_20261013`). The cost: the labels lie farther from the observed track (|e_y| from the observed path, the median of the flights' medians 8.3 → 11.4 m). The condition "lined up" is needed: from "no level-off" alone the turn onto the final is pulled toward the centreline (100 flights: 8 timeouts) | Decided | User, 2026-10-13 ("采用，做 v18 并重跑 B") |
```

### 75. `vocabulary.md`: §4.9 The comparison, the vertical error e_h

**What changed:** on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203)

**Old text:**

```text
- the vertical error e_h is the flown height minus the observed height.
```

**New text:**

```text
- the vertical error e_h is the flown height minus the observed height; on the final (where e_y is read from the
  centreline), the flown height above that runway's published glidepath: its TCH and angle on the straight line
  with the earth's curvature (§9.3), so LTP + TCH at the threshold, the evaluation's reference there (D203).
```

### 76. `vocabulary.md`: §4.9 Past the end of the observed path (D44)

**What changed:** on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203)

**Old text:**

```text
**Past the end of the observed path (D44).** The observed slice ends before the threshold, and the flown aircraft flies
on to it: approximately 150 m (median), at most approximately 750 m (§9.9). There the path continues along the line of
its last segment, and the lateral error is measured against that line for the readouts; no vertical error is measured
there. The labeller says no correction there, lateral or vertical: a correction in force ends at the end of the path,
and the observed word of its column is said again (in the readouts this word counts as a correction word; a speaker and
the executor read every word alike). The aircraft flies the last observed words to the threshold. These rows count as
rows without correction (D34). Why: there is no observation there; the slope of one 2 s segment is not a measurement
(the ADS-B altitude has steps of 7.6 m, so one segment can be approximately ±3° wrong, ±20 m over 400 m); and the DA
point, 0.9–2.1 km before the threshold, always lies before the end of the observed path, so the DA record (D38) does
not change.
```

**New text:**

```text
**Past the end of the observed path (D44, D203).** The observed slice ends before the threshold, and the flown
aircraft flies on to it: approximately 150 m (median), at most approximately 750 m (§9.9). On the final the errors are
read from the centreline and the glidepath, which go on past the end: there the labeller corrects as before the end, to
the threshold (D203). Off the final the path continues along the line of its last segment, and the lateral error is
measured against that line for the readouts; no vertical error is measured there. The labeller says no correction there,
lateral or vertical: a correction in force ends at the end of the path, and the observed word of its column is said
again (in the readouts this word counts as a correction word; a speaker and the executor read every word alike). The
aircraft flies the last observed words on. These rows count as rows without correction (D34). Why: off the final there is
no reference there; the slope of one 2 s segment is not a measurement (the ADS-B altitude has steps of 7.6 m, so one
segment can be approximately ±3° wrong, ±20 m over 400 m). On the final the references are the runway's, not the
observed path's, and without a correction past the end the labels crossed the threshold 1–4 m higher than at the end of
the observed path (D203).
```

### 77. `vocabulary.md`: §4.9 The errors ahead (D197), ė_h and the end error

**What changed:** on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203)

**Old text:**

```text
centreline (D201) (positive to the right), ė_h the
flown vertical rate minus the segment's slope times the flown velocity along it. The START ERROR
```

**New text:**

```text
centreline (D201) (positive to the right), ė_h the
flown vertical rate minus the segment's slope times the flown velocity along it, or on the final minus the rate at which
the glidepath's height under the aircraft changes (D203). The START ERROR
```

### 78. `vocabulary.md`: §4.9 The errors ahead (D197), the end error

**What changed:** on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203)

**Old text:**

```text
where the end error is zero or has the other sign: the end aims at the path e_y is read from (the observed path; on
the final, the centreline, D201). At ė = 0 the start error is the
```

**New text:**

```text
where the end error is zero or has the other sign: the end aims at the path the error is read from (the observed
path; on the final, the centreline and the glidepath, D201, D203). At ė = 0 the start error is the
```

### 79. `vocabulary.md`: §4.9 Vertical correction, item 2

**What changed:** on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203)

**Old text:**

```text
2. When e_h,on > H (too high), the labeller says the next steeper descent class.
```

**New text:**

```text
2. When e_h,on > H (too high; on the final, above the glidepath: D203), the labeller says the next steeper descent
   class.
```

### 80. `vocabulary.md`: §8 Values, closed-loop reading: correction

**What changed:** on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203)

**Old text:**

```text
| Closed-loop reading: correction | heading: one class (5°) toward the observed path; angle: the next descent class | D32 |
```

**New text:**

```text
| Closed-loop reading: correction | heading: one class (5°) toward the observed path; angle: the next descent class toward it; on the final toward the centreline and the published glidepath | D32, D201, D203 |
```

### 81. `vocabulary.md`: §8 Values, past the end of the observed path

**What changed:** on the final the angle corrections aim at the glidepath, and the corrections go on to the threshold (D203)

**Old text:**

```text
| Closed-loop reading past the end of the observed path | no correction (one in force ends); lateral error for the readouts, no vertical error | D44 |
```

**New text:**

```text
| Closed-loop reading past the end of the observed path | on the final, corrections from the centreline and the glidepath to the threshold; off the final, no correction (one in force ends), lateral error for the readouts, no vertical error | D44, D203 |
```

### 82. `prior.md`: §2 Inputs, Scales

**What changed:** the offset right of a final's centreline goes in on the glidepath's scale, asinh(d / 100 m) (D204)

**Old text:**

```text
- **Scales (D41, D65).** Every input has a fixed scale in SI units (§9): the distances along and across a candidate
  asinh(d / 1 km), linear within a few hundred metres of a final; the heights (above E, above a threshold) h / 1 km; the
  height above a glidepath asinh(h / 100 m), linear near the glidepath, where the value has a meaning, and compressed
  far from it (an aircraft high above an opposite runway's extended glidepath is thousands of metres above it); the
```

**New text:**

```text
- **Scales (D41, D65, D204).** Every input has a fixed scale in SI units (§9): the distance before a candidate's
  threshold asinh(d / 1 km), linear within a few hundred metres of it; the heights (above E, above a threshold)
  h / 1 km; the two offsets from a final's path, right of its centreline and above its glidepath, asinh(d / 100 m),
  linear near the path, where the value has a meaning, and compressed far from it (an aircraft high above an opposite
  runway's extended glidepath is thousands of metres above it); the
```

### 83. `prior.md`: §9 Values, input scales

**What changed:** the offset right of a final's centreline goes in on the glidepath's scale, asinh(d / 100 m) (D204)

**Old text:**

```text
| Input scales | Distances asinh(d / 1 km); heights / 1 km; height above a glidepath asinh(h / 100 m); ground speed / 100 m/s; vertical rate / 10 m/s; landings in 30 min / 10; length / 1 km and threshold elevation MSL / 1 km (`constants`) | D41, D65 |
```

**New text:**

```text
| Input scales | Distance before a threshold asinh(d / 1 km); offset right of a final and height above a glidepath asinh(d / 100 m); heights / 1 km; ground speed / 100 m/s; vertical rate / 10 m/s; landings in 30 min / 10; length / 1 km and threshold elevation MSL / 1 km (`constants`) | D41, D65, D204 |
```

### 84. `prior.md`: §5 Training, the loop

**What changed:** the stop: 15 epochs without a better select loss, at most 150 epochs (D205)

**Old text:**

```text
run keeps the weights of the epoch with the smallest one and stops after 3 epochs without a smaller one, or at 30
epochs. The seed sets the first weights, the batches and the dropout.
```

**New text:**

```text
run keeps the weights of the epoch with the smallest one and stops after 15 epochs without a smaller one, or at 150
epochs (D205). The seed sets the first weights, the batches and the dropout.
```

### 85. `prior.md`: §5 Training, hyperparameters

**What changed:** the stop: 15 epochs without a better select loss, at most 150 epochs (D205)

**Old text:**

```text
batch, at most 30 epochs, a stop after 3 epochs without a better select loss. The RoPE base is 10,000 (D65); a test
```

**New text:**

```text
batch, at most 150 epochs, a stop after 15 epochs without a better select loss (D205). The RoPE base is 10,000
(D65); a test
```

### 86. `prior.md`: §9 Values, configuration A

**What changed:** the stop: 15 epochs without a better select loss, at most 150 epochs (D205)

**Old text:**

```text
| Configuration A | d_model 192, 4 layers, 6 heads, feed-forward 768, dropout 0.1, weight decay 0.01, learning rate 3·10⁻⁴ (linear warm-up over 500 steps, then constant), clip 1.0, 16,384 padded rows a batch, at most 30 epochs, stop after 3 | D40 |
```

**New text:**

```text
| Configuration A | d_model 192, 4 layers, 6 heads, feed-forward 768, dropout 0.1, weight decay 0.01, learning rate 3·10⁻⁴ (linear warm-up over 500 steps, then constant), clip 1.0, 16,384 padded rows a batch, at most 150 epochs, stop after 15 (D205) | D40 |
```

### 87. `prior.md`: §2 Inputs, D23's test

**What changed:** the prior reads no landings; the loops keep them for the reward (D206)

**Old text:**

```text
  loop the same input does not exist. A test: a change of the runway word of a sentence, of its flight's record (its
  runway and landing time) and of its own landing in the index leaves the inputs of rows 0 to the first predicted step
  the same, bit for bit.
```

**New text:**

```text
  loop the same input does not exist. A test: a change of the runway word of a sentence and of its flight's record (its
  runway and landing time) leaves the inputs of rows 0 to the first predicted step the same, bit for bit.
```

### 88. `prior.md`: §2 Inputs, Candidates

**What changed:** the prior reads no landings; the loops keep them for the reward (D206)

**Old text:**

```text
  the glidepath height below, and the landings on the candidate in the 30 min before the step. It has no constant of
  its runway.
```

**New text:**

```text
  and the glidepath height below. It has no count of landings (D206) and no constant of its runway.
```

### 89. `prior.md`: §2 Inputs, Landings

**What changed:** the prior reads no landings; the loops keep them for the reward (D206)

**Old text:**

```text
- **Landings (D63, D105).** The number of landings on each candidate in the 30 min before the step: in [t − 30 min, t),
  t the row's UTC time. Offline, the landings come from the airport's tracks roster: every record that the harvest
  assigned to a candidate, with a sentence or without one, less the landings on the sealed test days (C32). The index of
  landings refuses a landing on a test day or on a day outside the day split by itself. A flight never counts its own
  landing: a closed-loop sentence can fly past the time at which the observed aircraft landed, and the runway of that
  landing is the answer of the runway word; the own landing must be in the index. In a loop, the caller gives each
  aircraft the landings that the loop knows (D105). The roster is outside the artefact, so the identity of a prior's
  data holds a digest of the landings (§8, item 1).
```

**New text:**

```text
- **No landings (D206).** The prior does not read the landings before the step. The loops keep each aircraft's
  landings for their caller (D105): the post-training's reward reads the airport's present landing direction from
  them. Offline, the landings come from the airport's tracks roster: every record that the harvest assigned to a
  candidate, with a sentence or without one, less the landings on the sealed test days (C32). The index of landings
  refuses a landing on a test day or on a day outside the day split by itself, and counts the landings in
  [t − 30 min, t) without a given flight's own landing. The roster is outside the artefact, so the identity of a
  prior's data holds a digest of the landings (§8, item 1; D63).
```

### 90. `prior.md`: §2 Inputs, Scales

**What changed:** the prior reads no landings; the loops keep them for the reward (D206)

**Old text:**

```text
  ground speed v / 100 m/s; the vertical rate v / 10 m/s; the landings in 30 min n / 10; in the variant `constants` the
  length l / 1 km and the threshold elevation MSL h / 1 km.
```

**New text:**

```text
  ground speed v / 100 m/s; the vertical rate v / 10 m/s; in the variant `constants` the length l / 1 km and the
  threshold elevation MSL h / 1 km.
```

### 91. `prior.md`: §6 A new airport, what it needs

**What changed:** the prior reads no landings; the loops keep them for the reward (D206)

**Old text:**

```text
| The landings in the 30 min before the step | The airport's tracks (offline) or a live surveillance feed |
```

**New text:**

(deleted)

### 92. `prior.md`: §7 Public interface, item 2

**What changed:** the prior reads no landings; the loops keep them for the reward (D206)

**Old text:**

```text
| 2 | The inputs of a row | One function from the states on the data's 2 s rows (observed before the first predicted step, flown from it), the words said, the candidates and the landings before the step to the inputs of a row (§2). Training, free generation and a loop of several aircraft use the same function: for the sentences of the artefact; and in a loop, from each aircraft's states, its own landings (D105) and the speaker's words in force, the inputs of the next Δ row and the position that the masks read (D96). The landings: from the tracks roster, less the sealed test days (the index refuses them, D105), counted before a time without a given flight's own landing (D63). The motion of rows: the 2 s displacement, with the velocities east and north (D106) Built (multi-aircraft control D150, MC0): `LoopRows` takes each aircraft's join tick in place of one first predicted step for the batch (aircraft b's inputs at tick t are those of its own row t − j_b; an absent aircraft's row is not present), and a landing added to chosen aircraft's landings while the loop runs (each one's `LandingIndex` replaced by one that holds it, with the index's checks); with every join tick 0 and no landing added, today's inputs, bit for bit | `prior/inputs.py` `state_inputs`, `sentence_rows`, `own_flight_key`, `motion` (the motion of rows); `prior/loop.py` `LoopRows` (`select`; one `LandingIndex` for each aircraft; one first predicted step for the batch, an int); `prior/landings.py` `Landing`, `LandingIndex`; `prior/source.py` `airport_landings`; stage D's: `LoopRows(…, join_ticks=…)`, `LoopRows.join_ticks`, `LoopRows.add_landing`, `LandingIndex.with_landing`, `batch.row_tensors(…, present=…)`; `LandingIndex` refuses landings out of time-then-flight-key order (multi-aircraft control D186 (1): a landing added is checked and inserted alone); `LoopRows.select(indices, device=)` (multi-aircraft control D182: a copy on another device) |
```

**New text:**

```text
| 2 | The inputs of a row | One function from the states on the data's 2 s rows (observed before the first predicted step, flown from it), the words said and the candidates to the inputs of a row (§2; no landings, D206). Training, free generation and a loop of several aircraft use the same function: for the sentences of the artefact; and in a loop, from each aircraft's states and the speaker's words in force, the inputs of the next Δ row and the position that the masks read (D96). The landings, which the loops keep for their caller (D105, D206): from the tracks roster, less the sealed test days (the index refuses them), counted before a time without a given flight's own landing (D63). The motion of rows: the 2 s displacement, with the velocities east and north (D106) Built (multi-aircraft control D150, MC0): `LoopRows` takes each aircraft's join tick in place of one first predicted step for the batch (aircraft b's inputs at tick t are those of its own row t − j_b; an absent aircraft's row is not present); with every join tick 0, today's inputs, bit for bit | `prior/inputs.py` `state_inputs`, `sentence_rows`, `own_flight_key`, `motion` (the motion of rows); `prior/loop.py` `LoopRows` (`select`; one first predicted step for the batch, an int); `prior/landings.py` `Landing`, `LandingIndex`; `prior/source.py` `airport_landings`; stage D's: `LoopRows(…, join_ticks=…)`, `LoopRows.join_ticks`, `LandingIndex.with_landing`, `batch.row_tensors(…, present=…)`; `LandingIndex` refuses landings out of time-then-flight-key order (multi-aircraft control D186 (1): a landing added is checked and inserted alone); `LoopRows.select(indices, device=)` (multi-aircraft control D182: a copy on another device) |
```

### 93. `prior.md`: §7 Public interface, item 7

**What changed:** the prior reads no landings; the loops keep them for the reward (D206)

**Old text:**

```text
| 7 | The step of a speaker's closed loop | The closed loop of a speaker and the executor, one Δ row at a time (D106): the observed rows that the start of the closed loop gives back (vocabulary §6 item 5: `start_moved`'s, a moved start's moved rows; never read from the stored sentence), then the inputs of each row, the speaker with the caller's numbers, masks and input of the added modules, the executor's step through the start of a closed loop, the flights done halted, a flight ended by the caller; a copy of its state for chosen flights; free generation's random numbers of a flight. Free generation and the post-training's window loop use it. Built (multi-aircraft control D150, MC0): each aircraft joins at its own tick — `step` advances one tick for every aircraft, absent, observed or said by its own clock; each aircraft's records, words, states and sentence are of its own rows from its row 0, as today; a landing added to chosen aircraft (item 2); the copy keeps the join ticks. With every join tick 0, today's loop, bit for bit | `experiments/prior_speaking_loop.py` `SpeakingLoop` (`SpeakingLoop(model, loop, order, sentences, observed, flights, geometries, landings, finals, words, *, interval_s, variant, device, temperature)`: `observed` the start's rows; the executor flies a row inside the speaker's `accept`, so a refused row leaves the loop as it was; `copy(flights)` on the loop's own `Loop.copy`; `said(b)`, `states(b)`; `generated(flights)`, which refuses a flight the caller ended, D118), `Generated`; free generation's numbers `flight_numbers`; stage D's: `SpeakingLoop.join_ticks`, `roles`, `own_row`, `joined`, `said_now`, `add_landing`; `held`, `left` (each left flight's frozen record), `inside`, `outcome(flight)`, `landings(flight)`, `in_force(flight)`, `permitted(flights)` (one record a flight; multi-aircraft control D193 (3)); `SpeakingLoop.copy(flights, device=)` (multi-aircraft control D182); `SpeakingLoop(…, executor_device=)`, the executor's device, kept by a copy (multi-aircraft control D193 (1)) |
```

**New text:**

```text
| 7 | The step of a speaker's closed loop | The closed loop of a speaker and the executor, one Δ row at a time (D106): the observed rows that the start of the closed loop gives back (vocabulary §6 item 5: `start_moved`'s, a moved start's moved rows; never read from the stored sentence), then the inputs of each row, the speaker with the caller's numbers, masks and input of the added modules, the executor's step through the start of a closed loop, the flights done halted, a flight ended by the caller; a copy of its state for chosen flights; free generation's random numbers of a flight. Free generation and the post-training's window loop use it. Built (multi-aircraft control D150, MC0): each aircraft joins at its own tick — `step` advances one tick for every aircraft, absent, observed or said by its own clock; each aircraft's records, words, states and sentence are of its own rows from its row 0, as today; a landing added to chosen aircraft's landings, which the loop keeps for its caller (D105, D206); the copy keeps the join ticks. With every join tick 0, today's loop, bit for bit | `experiments/prior_speaking_loop.py` `SpeakingLoop` (`SpeakingLoop(model, loop, order, sentences, observed, flights, geometries, landings, finals, words, *, interval_s, variant, device, temperature)`: `observed` the start's rows; the executor flies a row inside the speaker's `accept`, so a refused row leaves the loop as it was; `copy(flights)` on the loop's own `Loop.copy`; `said(b)`, `states(b)`; `generated(flights)`, which refuses a flight the caller ended, D118), `Generated`; free generation's numbers `flight_numbers`; stage D's: `SpeakingLoop.join_ticks`, `roles`, `own_row`, `joined`, `said_now`, `add_landing`; `held`, `left` (each left flight's frozen record), `inside`, `outcome(flight)`, `landings(flight)`, `in_force(flight)`, `permitted(flights)` (one record a flight; multi-aircraft control D193 (3)); `SpeakingLoop.copy(flights, device=)` (multi-aircraft control D182); `SpeakingLoop(…, executor_device=)`, the executor's device, kept by a copy (multi-aircraft control D193 (1)) |
```

### 94. `prior.md`: §9 Values, landings

**What changed:** the prior reads no landings; the loops keep them for the reward (D206)

**Old text:**

```text
| Landings of a candidate | In [t − 30 min, t), the flight's own landing not counted | §2, D63 |
```

**New text:**

```text
| The landings that the loops keep (no input, D206) | Counted in [t − 30 min, t), the flight's own landing not counted | §2, D63 |
```

### 95. `prior.md`: §9 Values, input scales

**What changed:** the prior reads no landings; the loops keep them for the reward (D206)

**Old text:**

```text
| Input scales | Distance before a threshold asinh(d / 1 km); offset right of a final and height above a glidepath asinh(d / 100 m); heights / 1 km; ground speed / 100 m/s; vertical rate / 10 m/s; landings in 30 min / 10; length / 1 km and threshold elevation MSL / 1 km (`constants`) | D41, D65, D204 |
```

**New text:**

```text
| Input scales | Distance before a threshold asinh(d / 1 km); offset right of a final and height above a glidepath asinh(d / 100 m); heights / 1 km; ground speed / 100 m/s; vertical rate / 10 m/s; length / 1 km and threshold elevation MSL / 1 km (`constants`) | D41, D65, D204 |
```

### 96. `post_training.md`: §3 Inputs, landing context

**What changed:** the prior reads no landings; the loops keep them for the reward (prior D206)

**Old text:**

```text
- **Landing context (D31, D98, D105).** Only the commanded aircraft has inputs. Its landing context counts the landings
  of the window's scene before the step: the recorded landings of the recorded aircraft and of the aircraft before the
  window, with the changes of an augmented window (D105), never its own recorded landing (that landing is the future of
  the commanded aircraft) and never a landing on a sealed test day. The commanded aircraft's own landing in the loop
  ends its window, so it never counts it either.
```

**New text:**

```text
- **Landing context (D31, D98, D105; prior D206).** The prior reads no landings; the loop keeps the commanded
  aircraft's landings for the reward's present landing direction (§2 item 2). They are the landings of the window's
  scene before the step: the recorded landings of the recorded aircraft and of the aircraft before the window, with
  the changes of an augmented window (D105), never its own recorded landing (that landing is the future of the
  commanded aircraft) and never a landing on a sealed test day. The commanded aircraft's own landing in the loop
  ends its window, so it never counts it either.
```

### 97. `multi_control.md`: §6.2 Compressed windows

**What changed:** the prior reads no landings; the loops keep them for the reward (prior D206)

**Old text:**

```text
  and words do not change (the executor and the inputs read no absolute time, except the landings it counts, which are
  read at its new time: its moved entry time in `LoopRows`, §6.3 item 2).
```

**New text:**

```text
  and words do not change (the executor and the inputs read no absolute time; the reward's landings are counted at its
  new time).
```

### 98. `multi_control.md`: §9 Changes of other stages, item 3

**What changed:** the prior reads no landings; the loops keep them for the reward (prior D206)

**Old text:**

```text
| 3 | Prior §7 item 2, the landings of a loop | A landing added to chosen aircraft's landings while the loop runs; copied with the loop (§6.3 item 5) | No landing added: today's inputs, bit for bit; a landing added at a time changes only the counts of the rows after it |
```

**New text:**

```text
| 3 | Prior §7 item 7, the landings of a loop | A landing added to chosen aircraft's landings while the loop runs; copied with the loop (§6.3 item 5) | A landing goes to the chosen aircraft still held, and changes no input (prior D206) |
```

### 99. `post_training.md`: D105

**What changed:** the runway of a landing does not change the reward (post-training D207)

**Old text:**

```text
| D105 | The landings that the commanded aircraft's inputs count follow the window's scene (§3): the tracks roster's landings with the window's changes — an inserted aircraft's landing (A) added at its shifted time, a moved aircraft's landing (D) moved by its shift — less the commanded aircraft's own landing (D31) and never a landing on a sealed test day (C32). A real window counts the roster's landings, so a window without other aircraft reads what free generation reads (§2 item 1). The present landing direction of the reward (§2 item 2) reads the same landings. The window loop gives them to the prior's function of a loop's row for each aircraft (prior D105). Why: with one index for each airport, the commanded aircraft read a value of the record that the augmentation changed — one landing on one candidate, up to 120 s in D and to the end of the window in A (`readouts/2026-10-05_stage_b_check.zh.md`) | Decided | User, 2026-10-05, on Claude's check of stage B |
```

**New text:**

```text
| D105 | The landings that the commanded aircraft's inputs count follow the window's scene (§3): the tracks roster's landings with the window's changes — an inserted aircraft's landing (A) added at its shifted time, a moved aircraft's landing (D) moved by its shift — less the commanded aircraft's own landing (D31) and never a landing on a sealed test day (C32). A real window counts the roster's landings, so a window without other aircraft reads what free generation reads (§2 item 1). Since prior D206 and D207 no input and no reward reads them (O26). The window loop gives them to the prior's function of a loop's row for each aircraft (prior D105). Why: with one index for each airport, the commanded aircraft read a value of the record that the augmentation changed — one landing on one candidate, up to 120 s in D and to the end of the window in A (`readouts/2026-10-05_stage_b_check.zh.md`) | Decided | User, 2026-10-05, on Claude's check of stage B |
```

### 100. `post_training.md`: D112

**What changed:** the runway of a landing does not change the reward (post-training D207)

**Old text:**

```text
| D112 | With no landing in the 30 min before the first predicted step, every runway is of the airport's present landing direction (§2 item 2): there is no direction to break. Why: read literally, no runway would be of it and every landing would earn 0; 1,119 of 40,530 train windows (2.8 %; select 3.0 %) of A34's artefact have no landing in those 30 min | Decided | User, 2026-10-05, on the implementer's proposal P19 |
```

**New text:**

(deleted)

### 101. `post_training.md`: §2 item 2, Reward

**What changed:** the runway of a landing does not change the reward (post-training D207)

**Old text:**

```text
   | `landed`, no go-around, on a runway of the airport's present landing direction | 1 |
   | `landed` after n go-arounds, on such a runway | 0.9ⁿ |
   | Every other outcome; every loss of separation (the separation judge, §3) | 0 |

   "The airport's present landing direction": a runway within 90° of a runway with a landing in the 30 min before the
   first predicted step, among the window's landings (D105); with no landing in those 30 min, every runway (D112).
```

**New text:**

```text
   | `landed`, no go-around | 1 |
   | `landed` after n go-arounds | 0.9ⁿ |
   | Every other outcome; every loss of separation (the separation judge, §3) | 0 |

   The runway of the landing does not change the reward: a landing on any candidate runway counts (D207).
```

### 102. `post_training.md`: §3 Multi-aircraft, Landing context

**What changed:** the runway of a landing does not change the reward (post-training D207)

**Old text:**

```text
- **Landing context (D31, D98, D105; prior D206).** The prior reads no landings; the loop keeps the commanded
  aircraft's landings for the reward's present landing direction (§2 item 2). They are the landings of the window's
```

**New text:**

```text
- **Landing context (D31, D98, D105; prior D206; D207).** The prior reads no landings, and the reward reads none
  (D207); the loop still keeps the commanded aircraft's landings (O26). They are the landings of the window's
```

### 103. `post_training.md`: §5 Values, Present landing direction

**What changed:** the runway of a landing does not change the reward (post-training D207)

**Old text:**

```text
| Present landing direction | A runway within 90° of a runway with a landing in the 30 min before the first predicted step; every runway with no such landing | §2 item 2, D112 |
```

**New text:**

(deleted)

### 104. `post_training.md`: §9 Public interface, item 6

**What changed:** the runway of a landing does not change the reward (post-training D207)

**Old text:**

```text
| 6 | The reward | D30's reward of an aircraft; the present landing direction (D112) | `post/reward.py` `reward`, `present_runways` |
```

**New text:**

```text
| 6 | The reward | D30's reward of an aircraft; the runway of a landing does not change it (D207) | `post/reward.py` `reward` |
```

### 105. `multi_control.md`: D140

**What changed:** the runway of a landing does not change the reward (post-training D207)

**Old text:**

```text
| D140 | **The reward of an aircraft** r_i is post-training D30, unchanged, for each commanded aircraft: 1 for `landed` with no go-around, 0.9ⁿ after n go-arounds, on a runway of the airport's present landing direction (D147); 0 for every other outcome and when the aircraft answers for a loss of separation (the judge's responsibility, §2.4). Why: D30's reasons hold for each aircraft (§2.1) | Decided | User, 2026-10-06, on Claude's proposal |
```

**New text:**

```text
| D140 | **The reward of an aircraft** r_i is post-training D30, unchanged, for each commanded aircraft: 1 for `landed` with no go-around, 0.9ⁿ after n go-arounds, on any runway (post-training D207); 0 for every other outcome and when the aircraft answers for a loss of separation (the judge's responsibility, §2.4). Why: D30's reasons hold for each aircraft (§2.1) | Decided | User, 2026-10-06, on Claude's proposal |
```

### 106. `multi_control.md`: D147

**What changed:** the runway of a landing does not change the reward (post-training D207)

**Old text:**

```text
| D147 | **The landings that a window counts.** Two quantities count landings: the input of the landings on each candidate in the 30 min before a row (prior §7 item 2) and the present landing direction of the reward (post-training D112). In a window: (1) a commanded aircraft's landing in the data is not counted (it is that aircraft's future); (2) a commanded aircraft whose outcome in the loop is `landed` is counted for every other aircraft of its window, on its landed runway, from its crossing time; (3) a commanded aircraft with any other outcome is not counted; (4) a recorded aircraft is counted at its landing time in the data, as in stage C (post-training D105). Why: the inputs give only what a controller knows before the step (principle 7); with one commanded aircraft this is post-training D31 and D105 | Decided | User, 2026-10-06, on Claude's proposal |
```

**New text:**

```text
| D147 | **The landings that a window counts.** Nothing reads them since prior D206 (the prior's input) and post-training D207 (the reward's landing direction): post-training O26. In a window: (1) a commanded aircraft's landing in the data is not counted (it is that aircraft's future); (2) a commanded aircraft whose outcome in the loop is `landed` is counted for every other aircraft of its window, on its landed runway, from its crossing time; (3) a commanded aircraft with any other outcome is not counted; (4) a recorded aircraft is counted at its landing time in the data, as in stage C (post-training D105). Why: the inputs give only what a controller knows before the step (principle 7); with one commanded aircraft this is post-training D31 and D105 | Decided | User, 2026-10-06, on Claude's proposal |
```

### 107. `multi_control.md`: §2.1 One aircraft

**What changed:** the runway of a landing does not change the reward (post-training D207)

**Old text:**

```text
| `landed`, no go-around, on a runway of the present landing direction | 1 |
| `landed` after n go-arounds, on such a runway | 0.9ⁿ |
| every other outcome of the judge; a loss of separation that the aircraft answers for | 0 |

The present landing direction (post-training D112, §9 item 6) is read for each aircraft at its own first predicted
step, from the landings of its window (D147). The bound of two go-arounds (post-training D91) holds for each aircraft.
```

**New text:**

```text
| `landed`, no go-around | 1 |
| `landed` after n go-arounds | 0.9ⁿ |
| every other outcome of the judge; a loss of separation that the aircraft answers for | 0 |

The runway of the landing does not change r_i (post-training D207). The bound of two go-arounds (post-training D91)
holds for each aircraft.
```

### 108. `prior.md`: §2 Inputs, No landings

**What changed:** the runway of a landing does not change the reward (post-training D207)

**Old text:**

```text
- **No landings (D206).** The prior does not read the landings before the step. The loops keep each aircraft's
  landings for their caller (D105): the post-training's reward reads the airport's present landing direction from
  them. Offline,
```

**New text:**

```text
- **No landings (D206).** The prior does not read the landings before the step. The loops keep each aircraft's
  landings for their caller (D105); since post-training D207 the reward does not read them either (post-training
  O26). Offline,
```

### 109. `prior.md`: D206

**What changed:** the runway of a landing does not change the reward (post-training D207)

**Old text:**

```text
| D206 | The prior reads no landings: the candidate vector loses the number of landings on the candidate in the 30 min before the step (`prior/batch.py` `CANDIDATE_FEATURES`: 6 in `full`, 8 in `constants`). The loops keep each aircraft's landings for their caller (`SpeakingLoop.landings`, `add_landing`; D105): the post-training's reward reads the airport's present landing direction from them (post-training §2 item 2). The identity of a prior's data still holds their digest (D63): the identities of stages C and D hold the prior's, and their reward reads the same landings. The checkpoint format is `ts-prior-checkpoint-v10`; a v9 checkpoint read the landings and is refused by name. Why: the user's reason, a better generalisation ("理论上为了增加泛化能力，最好能去掉它"). The input counted every landing of the roster, light aircraft too. At KRDU, runway 32 is a runway of light aircraft: on the train days 1,540 landings, 918 of them P28A, 72 with a closed-loop sentence (all C172). B20's fold without KRDU (configuration A, seed 1337) said 32 as its first runway in 60 of 400 sentences (B19's: 17), and of the 59 sentences with 32 last in force 21 landed. A trial without the input (the variant `no_landings`, branch `trial-no-landings` at `54dc46aa`; `outputs/POOLED/prior/trial_no_landings_20261014`, read-only), the folds KRDU and KMSY of A seed 1337 against B20's: KRDU landed 84.8 → 90.8 % of its held-out sentences (B19 85.2 %), 32 first in 19 sentences; KMSY 92.0 → 91.5 %; the first-step runway right KRDU 69.1 → 70.1 %, KMSY 92.6 → 86.5 % | Decided | User, 2026-10-14 ("直接把这个输入去掉", then after the trial "直接采用，重跑整个活动") |
```

**New text:**

```text
| D206 | The prior reads no landings: the candidate vector loses the number of landings on the candidate in the 30 min before the step (`prior/batch.py` `CANDIDATE_FEATURES`: 6 in `full`, 8 in `constants`). The loops keep each aircraft's landings for their caller (`SpeakingLoop.landings`, `add_landing`; D105); since post-training D207 the reward does not read them either (post-training O26). The identity of a prior's data still holds their digest (D63). The checkpoint format is `ts-prior-checkpoint-v10`; a v9 checkpoint read the landings and is refused by name. Why: the user's reason, a better generalisation ("理论上为了增加泛化能力，最好能去掉它"). The input counted every landing of the roster, light aircraft too. At KRDU, runway 32 is a runway of light aircraft: on the train days 1,540 landings, 918 of them P28A, 72 with a closed-loop sentence (all C172). B20's fold without KRDU (configuration A, seed 1337) said 32 as its first runway in 60 of 400 sentences (B19's: 17), and of the 59 sentences with 32 last in force 21 landed. A trial without the input (the variant `no_landings`, branch `trial-no-landings` at `54dc46aa`; `outputs/POOLED/prior/trial_no_landings_20261014`, read-only), the folds KRDU and KMSY of A seed 1337 against B20's: KRDU landed 84.8 → 90.8 % of its held-out sentences (B19 85.2 %), 32 first in 19 sentences; KMSY 92.0 → 91.5 %; the first-step runway right KRDU 69.1 → 70.1 %, KMSY 92.6 → 86.5 % | Decided | User, 2026-10-14 ("直接把这个输入去掉", then after the trial "直接采用，重跑整个活动") |
```

### 110. `vocabulary.md`: O23

**What changed:** the runway of a landing does not change the reward (post-training D207)

**Old text:**

```text
| O23 | Stage D's landing context: a commanded aircraft adds its landing to the others' context only when the judge says `landed`; D196 passes fewer crossings than the judge before it, so the other aircraft's reward can read another present landing direction (the prior reads no landings since prior D206) | Measure with `window_behaviour --against` (R68) on a stage D campaign | 5.8 |
```

**New text:**

```text
| O23 | Stage D's landing context: a commanded aircraft adds its landing to the others' context only when the judge says `landed`; D196 passes fewer crossings than the judge before it. No input (prior D206) and no reward (post-training D207) reads that context now (post-training O26) | Closes with post-training O26 | 5.8 |
```

### 111. `post_training.md`: O26

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| O26 | The window's landings are read by nothing since prior D206 (the prior's input) and D207 (the reward): the landings of a window (D105, `post/landings.py`), a commanded aircraft's landing added in the loop (multi-aircraft control D147, `WindowLoop` → `SpeakingLoop.add_landing`), the landings that a loop holds (multi-aircraft control D189 (0), `SpeakingLoop.landings`), and the landings' digest in the identity of a prior's data (prior D63) | Delete them in stages B–D (code, tests, decisions), one change; the identity of a prior's data then changes its format. The user decides | §3 |
```

**New text:**

```text
| O26 | What D208 kept of the landings that no input, no reward and no loop reads: the landings of a window (D105, `post/landings.py` `window_landings`), read only by the diagnostic readout's field 1 (D175, `other_class`), and their digest in the identity of a prior's data (prior D63, `prior/source.py` `artefact_identity`) | The digest goes with the next prior campaign (the identity then changes its format, and the checkpoints before it are refused); D175's field 1 without the window's landings, or the landings kept for it alone. The user decides | §3 |
```

### 112. `post_training.md`: D31

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| D31 | Multi-aircraft inputs and judgements (§3): the landing context of an aircraft of a scene counts the landings of the closed loop (with one commanded aircraft, only that aircraft has inputs: D98, D105); D23 holds for every aircraft of a scene, with a test; "established on the final" is a function of one row, the same for every aircraft (its rule: D92). (The part of the prior: D31 there) | Decided | User, 2026-10-03 |
```

**New text:**

```text
| D31 | Multi-aircraft inputs and judgements (§3): with one commanded aircraft, only that aircraft has inputs (D98); no input counts landings (prior D206), and the loops keep none (D208); D23 holds for every aircraft of a scene, with a test; "established on the final" is a function of one row, the same for every aircraft (its rule: D92). (The part of the prior: D31 there) | Decided | User, 2026-10-03 |
```

### 113. `post_training.md`: D105

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| D105 | The landings that the commanded aircraft's inputs count follow the window's scene (§3): the tracks roster's landings with the window's changes — an inserted aircraft's landing (A) added at its shifted time, a moved aircraft's landing (D) moved by its shift — less the commanded aircraft's own landing (D31) and never a landing on a sealed test day (C32). A real window counts the roster's landings, so a window without other aircraft reads what free generation reads (§2 item 1). Since prior D206 and D207 no input and no reward reads them (O26). The window loop gives them to the prior's function of a loop's row for each aircraft (prior D105). Why: with one index for each airport, the commanded aircraft read a value of the record that the augmentation changed — one landing on one candidate, up to 120 s in D and to the end of the window in A (`readouts/2026-10-05_stage_b_check.zh.md`) | Decided | User, 2026-10-05, on Claude's check of stage B |
```

**New text:**

```text
| D105 | The landings of a window follow the window's scene (§3): the tracks roster's landings with the window's changes — an inserted aircraft's landing (A) added at its shifted time, a moved aircraft's landing (D) moved by its shift — less the commanded aircraft's own landing (D31) and never a landing on a sealed test day (C32). A real window has the roster's landings. No input (prior D206), no reward (D207) and no loop (D208) reads them; the diagnostic readout does (D175; O26). Why: with one index for each airport, the commanded aircraft read a value of the record that the augmentation changed — one landing on one candidate, up to 120 s in D and to the end of the window in A (`readouts/2026-10-05_stage_b_check.zh.md`) | Decided | User, 2026-10-05, on Claude's check of stage B |
```

### 114. `post_training.md`: §3 Inputs, Landing context

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
- **Landing context (D31, D98, D105; prior D206; D207).** The prior reads no landings, and the reward reads none
  (D207); the loop still keeps the commanded aircraft's landings (O26). They are the landings of the window's
  scene before the step: the recorded landings of the recorded aircraft and of the aircraft before the window, with
  the changes of an augmented window (D105), never its own recorded landing (that landing is the future of the
  commanded aircraft) and never a landing on a sealed test day. The commanded aircraft's own landing in the loop
  ends its window, so it never counts it either.
```

**New text:**

```text
- **Landings (D31, D98, D105; prior D206; D207; D208).** No input of the prior, no reward and no loop reads
  landings. A window's landings are the roster's landings with the changes of an augmented window (D105), never
  a commanded aircraft's own recorded landing and never a landing on a sealed test day; only the diagnostic
  readout reads them (D175; O26).
```

### 115. `post_training.md`: §9 Public interface, item 2

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| 2 | The landings of a window | The roster's landings with the window's changes, less the commanded aircraft's own (D105). Built (multi-aircraft control D149, MC0): less every commanded aircraft's own | `post/landings.py` `window_landings`; stage D's: `landings.commanded_landings` |
```

**New text:**

```text
| 2 | The landings of a window | The roster's landings with the window's changes, less the commanded aircraft's own (D105). Read only by the diagnostic readout (D175); the loops keep none (D208) | `post/landings.py` `window_landings` |
```

### 116. `post_training.md`: §9 Public interface, item 8

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| 8 | The window loop | The commanded aircraft of a batch of windows through the speaker's closed loop (prior §7 item 7), the scene at each step, the judge, the masks of a caller, the copy of a window at a branch point. Built (multi-aircraft control D149, MC0): a window is one or more rows of the batch, each joining at its own tick (prior §7 item 7); an aircraft made silent (a caller's mask that permits only "unchanged" in every column); a window ends when every commanded aircraft is done or silent; each aircraft's result; a landing in the loop added to the other aircraft's landings; the copy copies every aircraft of the window. A reader of what a value network reads at each row, kept apart from the tokens (D171; stage D gives none). A public reader of the steps at which a recorded aircraft read a faulty point (D114), for the Training export (frontend D177 (15); C26, to be built) | `experiments/post_window_loop.py` `WindowLoop`, `WindowResult`, `moved_commanded`, `WindowLoop(…, value_reader=)`, `WindowLoop.values`, `WindowLoop.samples(split, rows)` (multi-aircraft control D181; on `SpeakingLoop.sentences(split, flights)`, prior §7 item 7); stage D's: `WindowLoop(…, answering=…, token_part=…, part_width=…)`, `responsible`, `Answering`, `TokenPartOf`, `WindowLoop.members`, `window_of`, `member_of`, `records`, `silent`, `landings`; `WindowLoop.copy(windows, device=)` (multi-aircraft control D182); `WindowLoop(…, executor_device=)` (D193 (1)); `WindowLoop.results(rows=None)`, by row, a flight's landings read from `SpeakingLoop.landings(flight)` (D193 (3)) |
```

**New text:**

```text
| 8 | The window loop | The commanded aircraft of a batch of windows through the speaker's closed loop (prior §7 item 7), the scene at each step, the judge, the masks of a caller, the copy of a window at a branch point. Built (multi-aircraft control D149, MC0): a window is one or more rows of the batch, each joining at its own tick (prior §7 item 7); an aircraft made silent (a caller's mask that permits only "unchanged" in every column); a window ends when every commanded aircraft is done or silent; each aircraft's result; the copy copies every aircraft of the window. A reader of what a value network reads at each row, kept apart from the tokens (D171; stage D gives none). A public reader of the steps at which a recorded aircraft read a faulty point (D114), for the Training export (frontend D177 (15); C26, to be built) | `experiments/post_window_loop.py` `WindowLoop`, `WindowResult`, `moved_commanded`, `WindowLoop(…, value_reader=)`, `WindowLoop.values`, `WindowLoop.samples(split, rows)` (multi-aircraft control D181; on `SpeakingLoop.sentences(split, flights)`, prior §7 item 7); stage D's: `WindowLoop(…, answering=…, token_part=…, part_width=…)`, `responsible`, `Answering`, `TokenPartOf`, `WindowLoop.members`, `window_of`, `member_of`, `records`, `silent`; `WindowLoop.copy(windows, device=)` (multi-aircraft control D182); `WindowLoop(…, executor_device=)` (D193 (1)); `WindowLoop.results(rows=None)`, by row (D193 (3)) |
```

### 117. `multi_control.md`: D147

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| D147 | **The landings that a window counts.** Nothing reads them since prior D206 (the prior's input) and post-training D207 (the reward's landing direction): post-training O26. In a window: (1) a commanded aircraft's landing in the data is not counted (it is that aircraft's future); (2) a commanded aircraft whose outcome in the loop is `landed` is counted for every other aircraft of its window, on its landed runway, from its crossing time; (3) a commanded aircraft with any other outcome is not counted; (4) a recorded aircraft is counted at its landing time in the data, as in stage C (post-training D105). Why: the inputs give only what a controller knows before the step (principle 7); with one commanded aircraft this is post-training D31 and D105 | Decided | User, 2026-10-06, on Claude's proposal |
```

**New text:**

(deleted)

### 118. `multi_control.md`: D166 (6)

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| D166 | **Stage D's implementer's readings of MC0–MC4, accepted** (`requests_from_d_to_designer.md`, 2026-10-07; the numbers are that note's). (6) A loop landing on a sealed test day is left out and counted with the sealed landings, as stage C's window landings. (7) A commanded aircraft that ends `landed` is judged once, at the row after it, as over its threshold at its crossing state, so the wake minimum behind a commanded leader is judged. (19) A formal start of stage D refuses a smoke campaign and a campaign with rounds still to run (as `post_validation`). (21) The token part's ten features (`multi/tokens.py`, `multi-commanded-tokens-v1`): the flags commanded, silent and words in force; the heading word as the prior reads it (sine and cosine of its track less the course in force); the level over 1,000 m with a flag for no level-off; the angle word's nominal angle over 3°; the speed target over 100 m/s with a flag for unspecified; a recorded aircraft's part 0; the scales the edge features' (a mirror, pinned by a test). (22) With one commanded aircraft, stage D's rules give stage C's round only where no `records_kept` loss occurs (D145); D149 holds for stage C's own code. (25) The silent flag carries at most one bit of the records (a `records_kept` loss) once an aircraft: a stated limit of D144 × D145 × D152. (29) `Stage.start` and `Stage.start_model`; `Stage.read_batch` takes the draw (post-training D161). (30) The context records whether the campaign is formal (`Context.formal`); `round_start` refuses a smoke source under it, and `start_of` checks the source before anything is opened. (31) A landed leader past its plane is not established; at its threshold row only the wake minimum is judged (its state up to 2 s behind, up to about 140 m, 2.5 % of the least minimum): a stated limit. (32) A compressed window whose shifts all round to 0 is counted as real; a window without a later aircraft has no compressed form. (33) The select set: real windows of each span (§5 item 2). (34) The readout's numbers `[seed, 2^30, place, member]`, a later draw appended; the readouts of O18 read each aircraft's crossing (`WindowResult.crossing`). (35) A landed commanded aircraft counts as recorded in the readout's pairs; the census's baseline never judges one (a stated limit, below 0.1 % of steps). The names they give are in vocabulary §6 item 5, prior §7 items 2, 3, 7 and post-training §9 | Decided | User, 2026-10-07 ((6), (7), (21), (25) one by one; (19), (22), (29)–(35) together) |
```

**New text:**

```text
| D166 | **Stage D's implementer's readings of MC0–MC4, accepted** (`requests_from_d_to_designer.md`, 2026-10-07; the numbers are that note's). (6) Gone: the loops keep no landings (post-training D208). (7) A commanded aircraft that ends `landed` is judged once, at the row after it, as over its threshold at its crossing state, so the wake minimum behind a commanded leader is judged. (19) A formal start of stage D refuses a smoke campaign and a campaign with rounds still to run (as `post_validation`). (21) The token part's ten features (`multi/tokens.py`, `multi-commanded-tokens-v1`): the flags commanded, silent and words in force; the heading word as the prior reads it (sine and cosine of its track less the course in force); the level over 1,000 m with a flag for no level-off; the angle word's nominal angle over 3°; the speed target over 100 m/s with a flag for unspecified; a recorded aircraft's part 0; the scales the edge features' (a mirror, pinned by a test). (22) With one commanded aircraft, stage D's rules give stage C's round only where no `records_kept` loss occurs (D145); D149 holds for stage C's own code. (25) The silent flag carries at most one bit of the records (a `records_kept` loss) once an aircraft: a stated limit of D144 × D145 × D152. (29) `Stage.start` and `Stage.start_model`; `Stage.read_batch` takes the draw (post-training D161). (30) The context records whether the campaign is formal (`Context.formal`); `round_start` refuses a smoke source under it, and `start_of` checks the source before anything is opened. (31) A landed leader past its plane is not established; at its threshold row only the wake minimum is judged (its state up to 2 s behind, up to about 140 m, 2.5 % of the least minimum): a stated limit. (32) A compressed window whose shifts all round to 0 is counted as real; a window without a later aircraft has no compressed form. (33) The select set: real windows of each span (§5 item 2). (34) The readout's numbers `[seed, 2^30, place, member]`, a later draw appended; the readouts of O18 read each aircraft's crossing (`WindowResult.crossing`). (35) A landed commanded aircraft counts as recorded in the readout's pairs; the census's baseline never judges one (a stated limit, below 0.1 % of steps). The names they give are in vocabulary §6 item 5, prior §7 items 2, 3, 7 and post-training §9 | Decided | User, 2026-10-07 ((6), (7), (21), (25) one by one; (19), (22), (29)–(35) together) |
```

### 119. `multi_control.md`: D186 (1)

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| D186 | **The window loop's own work, cut** (MC11; stage D's item 70, the profile of the continuations; the evaluation: Claude, 2026-10-09). The continuations spend more time in the loop's own Python than in the model and the executor together (stage D's log). Three changes, each in its stage's code by stage D's implementer (outline D184), each before use checked on fixed inputs: (1) a landing added to a loop (`SpeakingLoop.add_landing`, `LandingIndex.with_landing`, stage B) checks and inserts the new landing only instead of building and checking the airport's whole roster again — the same index, so the same numbers, bit for bit; (2) the grammar's masks of a row computed over the batch (`instructions/grammar.column_mask`, stage A), which replaces the readable mask after its check (outline D187; the labeller reads `apply`) — boolean logic, so equal bit for bit; it rests on a property of rules 1–6 that vocabulary §6 item 2 states (a heading or a speed word enters the rules only through rule 1), which a change of the rules must keep or change the mask with; (3) the inputs of a row computed over the batch (`prior/inputs.state_inputs` and `instructions/airport.relative_to_runway`, stages B and A), which replaces the computation one aircraft at a time after its check (outline D187), equal bit for bit as D138 asks; if a vectorised function differs in its last bits (a library's own sine or cosine), the differences are reported and the user decides whether a bound (words and outcomes the same but for draws near a boundary, counted; D181 (57)) may stand in for equality. Each with its stage's checks (the prior's behaviour check, prior D108; the labeller's conformance where the grammar is read; D149's check on stage C's and stage D's windows), once before its commit (outline D187). Item 56's runner measures each before and after. Not now: the traffic module's tokens, the model's step and the executor | Decided | User, 2026-10-09 (the request; the evaluation: Claude) |
```

**New text:**

```text
| D186 | **The window loop's own work, cut** (MC11; stage D's item 70, the profile of the continuations; the evaluation: Claude, 2026-10-09). The continuations spend more time in the loop's own Python than in the model and the executor together (stage D's log). Three changes, each in its stage's code by stage D's implementer (outline D184), each before use checked on fixed inputs: (1) gone with the loops' landings (post-training D208); (2) the grammar's masks of a row computed over the batch (`instructions/grammar.column_mask`, stage A), which replaces the readable mask after its check (outline D187; the labeller reads `apply`) — boolean logic, so equal bit for bit; it rests on a property of rules 1–6 that vocabulary §6 item 2 states (a heading or a speed word enters the rules only through rule 1), which a change of the rules must keep or change the mask with; (3) the inputs of a row computed over the batch (`prior/inputs.state_inputs` and `instructions/airport.relative_to_runway`, stages B and A), which replaces the computation one aircraft at a time after its check (outline D187), equal bit for bit as D138 asks; if a vectorised function differs in its last bits (a library's own sine or cosine), the differences are reported and the user decides whether a bound (words and outcomes the same but for draws near a boundary, counted; D181 (57)) may stand in for equality. Each with its stage's checks (the prior's behaviour check, prior D108; the labeller's conformance where the grammar is read; D149's check on stage C's and stage D's windows), once before its commit (outline D187). Item 56's runner measures each before and after. Not now: the traffic module's tokens, the model's step and the executor | Decided | User, 2026-10-09 (the request; the evaluation: Claude) |
```

### 120. `multi_control.md`: D189 (0)

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| D189 | **A step's own work cut where it is computed again for nothing** (MC11; the user, 2026-10-09: no measure first). Each a fast form that replaces the code before it after one check on fixed inputs before its commit, bit for bit, and written to be read (outline D187): (0) a commanded aircraft's landings (post-training §3, D105) hold only what its rows can count — the landings from 30 min before the window's first row to the window's end, and its own — so that a landing added in the loop (D186 (1)) copies tens of landings, not the airport's roster; every count stays the same, since a row counts only the 30 min before it; (1) each step, each window's per-aircraft values are computed once for all its aircraft, commanded and recorded — the motion, the scene's fields (the approach clock, the track less the course, the distance right of the centreline, established) and the speeds along the course — and each commanded aircraft's own view (itself first, then the recorded aircraft, then the other commanded ones, the order of today) is that computation reordered, not computed again; the judged set after a landing computes the landed aircraft only; the traffic tokens read the motions and the approach clock from it, and a pair's runway relation and required distance from a table by runway pair and wake category pair; the tokens' other features stay one call for each commanded aircraft, over all its others, so that every token keeps its order and its bits; (2) the speaker brings back from the device, for each column, only the word drawn, and the row's records (the probability forbidden, the words permitted, the probability drawn) once a row; the masks of the later columns other than the grammar's (the procedure's and the caller's), fixed once the runway word is drawn, are computed once a row and read again for the records; (3) then the tokens of all pairs of a window in one computation: a product that changes its kernel can change a last bit, and a difference found on fixed inputs goes to the user before use, as D186 (3). Why: each commanded aircraft held its airport's whole roster of landings, copied whole for each landing added in the loop; in stage D every aircraft of a window is commanded, and the scene, the speeds along the course and the tokens' motions and clocks were computed again for each commanded aircraft (n + 1 scenes a window a step for n commanded aircraft), though each value depends on its aircraft only; the speaker brought back about four results a column, twenty a row | Decided | User, 2026-10-09 (cut at once, no measure; the form: Claude) |
```

**New text:**

```text
| D189 | **A step's own work cut where it is computed again for nothing** (MC11; the user, 2026-10-09: no measure first). Each a fast form that replaces the code before it after one check on fixed inputs before its commit, bit for bit, and written to be read (outline D187): (0) gone with the loops' landings (post-training D208); (1) each step, each window's per-aircraft values are computed once for all its aircraft, commanded and recorded — the motion, the scene's fields (the approach clock, the track less the course, the distance right of the centreline, established) and the speeds along the course — and each commanded aircraft's own view (itself first, then the recorded aircraft, then the other commanded ones, the order of today) is that computation reordered, not computed again; the judged set after a landing computes the landed aircraft only; the traffic tokens read the motions and the approach clock from it, and a pair's runway relation and required distance from a table by runway pair and wake category pair; the tokens' other features stay one call for each commanded aircraft, over all its others, so that every token keeps its order and its bits; (2) the speaker brings back from the device, for each column, only the word drawn, and the row's records (the probability forbidden, the words permitted, the probability drawn) once a row; the masks of the later columns other than the grammar's (the procedure's and the caller's), fixed once the runway word is drawn, are computed once a row and read again for the records; (3) then the tokens of all pairs of a window in one computation: a product that changes its kernel can change a last bit, and a difference found on fixed inputs goes to the user before use, as D186 (3). Why: in stage D every aircraft of a window is commanded, and the scene, the speeds along the course and the tokens' motions and clocks were computed again for each commanded aircraft (n + 1 scenes a window a step for n commanded aircraft), though each value depends on its aircraft only; the speaker brought back about four results a column, twenty a row | Decided | User, 2026-10-09 (cut at once, no measure; the form: Claude) |
```

### 121. `multi_control.md`: §1 Scope

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
  aircraft and their branch points, the landings of a loop, their runners (`experiments/multi_*.py`), the package
```

**New text:**

```text
  aircraft and their branch points, their runners (`experiments/multi_*.py`), the package
```

### 122. `multi_control.md`: §2.6 Compressed windows

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
  and words do not change (the executor and the inputs read no absolute time; the reward's landings are counted at its
  new time).
```

**New text:**

```text
  and words do not change (the executor and the inputs read no absolute time).
```

### 123. `multi_control.md`: §3 The step of a window, items 5 and 6

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
5. A commanded aircraft whose judged outcome is `landed` adds its landing on its landed runway to the other
   aircraft's landings from its crossing time (D147).
6. The window ends when every commanded aircraft is done or silent (D144).
```

**New text:**

```text
5. The window ends when every commanded aircraft is done or silent (D144).
```

### 124. `multi_control.md`: §6.1

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
the landings (item 2), the losses that an aircraft answers for (item 3),
```

**New text:**

```text
the losses that an aircraft answers for (item 3),
```

### 125. `multi_control.md`: §6.2 row 3

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| 3 | Prior §7 item 7, the landings of a loop | A landing added to chosen aircraft's landings while the loop runs; copied with the loop (§6.3 item 5) | A landing goes to the chosen aircraft still held, and changes no input (prior D206) |
```

**New text:**

(deleted)

### 126. `multi_control.md`: §6.3 item 5

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
5. **The landings of a loop** (stage B, `LoopRows` and `SpeakingLoop`). A landing (its flight, runway and time) added
   to chosen aircraft's landings: each aircraft's `LandingIndex` replaced by one that holds it (the index keeps its
   checks: time order, the sealed days, one landing a flight). The counts of a row read the landings before its time,
   so a landing added at its crossing time changes only later rows. Copied with the loop.
```

**New text:**

(deleted)

### 127. `multi_control.md`: §9.3

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| A commanded aircraft's recorded landing is not counted; its loop landing is (D147) | The order of the aircraft inside a step (D148) |
```

**New text:**

```text
| | The order of the aircraft inside a step (D148) |
```

### 128. `prior.md`: D63

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| D63 | The identity of a prior's data (§8, item 1) also holds the landings that the candidate vectors count: for each airport, the sha256 of each landing's flight, runway and time, in time order, and of the number of landings left out on the sealed test days. The landings come from the airport's tracks roster, outside the artefact: without them, a changed roster would change the inputs and leave the identity the same. By their flights, never by the bytes of the roster (D21). A run that reads the landings again computes the digest again and refuses a difference (§2, §8) | Decided | User, 2026-10-04, on the reading of the stage B agent |
```

**New text:**

```text
| D63 | The identity of a prior's data (§8, item 1) also holds the airport's landings (no input reads them since D206, and no loop since post-training D208; post-training O26): for each airport, the sha256 of each landing's flight, runway and time, in time order, and of the number of landings left out on the sealed test days. The landings come from the airport's tracks roster, outside the artefact: without them, a changed roster would have changed the inputs (before D206) and left the identity the same. By their flights, never by the bytes of the roster (D21). A run that reads the landings again computes the digest again and refuses a difference (§2, §8) | Decided | User, 2026-10-04, on the reading of the stage B agent |
```

### 129. `prior.md`: D105

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| D105 | The landings that a loop's row counts (§2, §7 item 2) are given for each aircraft, not one index for each airport: the function of a loop's row takes each aircraft's landings. A caller builds them from the roster's landings (`Landing`, `LandingIndex`, in §7 item 2); the index refuses a landing on a sealed test day by itself (C32), not only the reader of the roster. Free generation gives each flight its airport's roster landings. (The window's landings: D105 in the post-training) | Decided | User, 2026-10-05, on Claude's check of stage B |
```

**New text:**

```text
| D105 | A caller builds landings from the roster's (`Landing`, `LandingIndex`, in §7 item 2; the window's landings: D105 in the post-training); the index refuses a landing on a sealed test day by itself (C32), not only the reader of the roster. No input (D206) and no loop (post-training D208) reads landings | Decided | User, 2026-10-05, on Claude's check of stage B |
```

### 130. `prior.md`: D206

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| D206 | The prior reads no landings: the candidate vector loses the number of landings on the candidate in the 30 min before the step (`prior/batch.py` `CANDIDATE_FEATURES`: 6 in `full`, 8 in `constants`). The loops keep each aircraft's landings for their caller (`SpeakingLoop.landings`, `add_landing`; D105); since post-training D207 the reward does not read them either (post-training O26). The identity of a prior's data still holds their digest (D63). The checkpoint format is `ts-prior-checkpoint-v10`; a v9 checkpoint read the landings and is refused by name. Why: the user's reason, a better generalisation ("理论上为了增加泛化能力，最好能去掉它"). The input counted every landing of the roster, light aircraft too. At KRDU, runway 32 is a runway of light aircraft: on the train days 1,540 landings, 918 of them P28A, 72 with a closed-loop sentence (all C172). B20's fold without KRDU (configuration A, seed 1337) said 32 as its first runway in 60 of 400 sentences (B19's: 17), and of the 59 sentences with 32 last in force 21 landed. A trial without the input (the variant `no_landings`, branch `trial-no-landings` at `54dc46aa`; `outputs/POOLED/prior/trial_no_landings_20261014`, read-only), the folds KRDU and KMSY of A seed 1337 against B20's: KRDU landed 84.8 → 90.8 % of its held-out sentences (B19 85.2 %), 32 first in 19 sentences; KMSY 92.0 → 91.5 %; the first-step runway right KRDU 69.1 → 70.1 %, KMSY 92.6 → 86.5 % | Decided | User, 2026-10-14 ("直接把这个输入去掉", then after the trial "直接采用，重跑整个活动") |
```

**New text:**

```text
| D206 | The prior reads no landings: the candidate vector loses the number of landings on the candidate in the 30 min before the step (`prior/batch.py` `CANDIDATE_FEATURES`: 6 in `full`, 8 in `constants`). The reward does not read them (post-training D207), and the loops keep none (post-training D208). The identity of a prior's data still holds their digest (D63). The checkpoint format is `ts-prior-checkpoint-v10`; a v9 checkpoint read the landings and is refused by name. Why: the user's reason, a better generalisation ("理论上为了增加泛化能力，最好能去掉它"). The input counted every landing of the roster, light aircraft too. At KRDU, runway 32 is a runway of light aircraft: on the train days 1,540 landings, 918 of them P28A, 72 with a closed-loop sentence (all C172). B20's fold without KRDU (configuration A, seed 1337) said 32 as its first runway in 60 of 400 sentences (B19's: 17), and of the 59 sentences with 32 last in force 21 landed. A trial without the input (the variant `no_landings`, branch `trial-no-landings` at `54dc46aa`; `outputs/POOLED/prior/trial_no_landings_20261014`, read-only), the folds KRDU and KMSY of A seed 1337 against B20's: KRDU landed 84.8 → 90.8 % of its held-out sentences (B19 85.2 %), 32 first in 19 sentences; KMSY 92.0 → 91.5 %; the first-step runway right KRDU 69.1 → 70.1 %, KMSY 92.6 → 86.5 % | Decided | User, 2026-10-14 ("直接把这个输入去掉", then after the trial "直接采用，重跑整个活动") |
```

### 131. `prior.md`: §2 Inputs, No landings

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
- **No landings (D206).** The prior does not read the landings before the step. The loops keep each aircraft's
  landings for their caller (D105); since post-training D207 the reward does not read them either (post-training
  O26). Offline, the landings come from the airport's tracks roster: every record that the harvest assigned to a
  candidate, with a sentence or without one, less the landings on the sealed test days (C32). The index of landings
  refuses a landing on a test day or on a day outside the day split by itself, and counts the landings in
  [t − 30 min, t) without a given flight's own landing. The roster is outside the artefact, so the identity of a
  prior's data holds a digest of the landings (§8, item 1; D63).
```

**New text:**

```text
- **No landings (D206).** The prior does not read the landings before the step, and no loop keeps them
  (post-training D208). An airport's landings come from its tracks roster: every record that the harvest
  assigned to a candidate, with a sentence or without one, less the landings on the sealed test days (C32). The
  index of landings refuses a landing on a test day or on a day outside the day split by itself. The identity of
  a prior's data holds a digest of them (§8, item 1; D63; post-training O26).
```

### 132. `prior.md`: §7 Public interface, item 2

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| 2 | The inputs of a row | One function from the states on the data's 2 s rows (observed before the first predicted step, flown from it), the words said and the candidates to the inputs of a row (§2; no landings, D206). Training, free generation and a loop of several aircraft use the same function: for the sentences of the artefact; and in a loop, from each aircraft's states and the speaker's words in force, the inputs of the next Δ row and the position that the masks read (D96). The landings, which the loops keep for their caller (D105, D206): from the tracks roster, less the sealed test days (the index refuses them), counted before a time without a given flight's own landing (D63). The motion of rows: the 2 s displacement, with the velocities east and north (D106) Built (multi-aircraft control D150, MC0): `LoopRows` takes each aircraft's join tick in place of one first predicted step for the batch (aircraft b's inputs at tick t are those of its own row t − j_b; an absent aircraft's row is not present); with every join tick 0, today's inputs, bit for bit | `prior/inputs.py` `state_inputs`, `sentence_rows`, `own_flight_key`, `motion` (the motion of rows); `prior/loop.py` `LoopRows` (`select`; one first predicted step for the batch, an int); `prior/landings.py` `Landing`, `LandingIndex`; `prior/source.py` `airport_landings`; stage D's: `LoopRows(…, join_ticks=…)`, `LoopRows.join_ticks`, `LandingIndex.with_landing`, `batch.row_tensors(…, present=…)`; `LandingIndex` refuses landings out of time-then-flight-key order (multi-aircraft control D186 (1): a landing added is checked and inserted alone); `LoopRows.select(indices, device=)` (multi-aircraft control D182: a copy on another device) |
```

**New text:**

```text
| 2 | The inputs of a row | One function from the states on the data's 2 s rows (observed before the first predicted step, flown from it), the words said and the candidates to the inputs of a row (§2; no landings, D206). Training, free generation and a loop of several aircraft use the same function: for the sentences of the artefact; and in a loop, from each aircraft's states and the speaker's words in force, the inputs of the next Δ row and the position that the masks read (D96). The landings of an airport, which no input and no loop reads (D206; post-training D208): from the tracks roster, less the sealed test days (the index refuses them; D63, D105). The motion of rows: the 2 s displacement, with the velocities east and north (D106) Built (multi-aircraft control D150, MC0): `LoopRows` takes each aircraft's join tick in place of one first predicted step for the batch (aircraft b's inputs at tick t are those of its own row t − j_b; an absent aircraft's row is not present); with every join tick 0, today's inputs, bit for bit | `prior/inputs.py` `state_inputs`, `sentence_rows`, `own_flight_key`, `motion` (the motion of rows); `prior/loop.py` `LoopRows` (`select`; one first predicted step for the batch, an int); `prior/landings.py` `Landing`, `LandingIndex`; `prior/source.py` `airport_landings`; stage D's: `LoopRows(…, join_ticks=…)`, `LoopRows.join_ticks`, `batch.row_tensors(…, present=…)`; `LandingIndex` refuses landings out of time-then-flight-key order; `LoopRows.select(indices, device=)` (multi-aircraft control D182: a copy on another device) |
```

### 133. `prior.md`: §7 Public interface, item 7

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| 7 | The step of a speaker's closed loop | The closed loop of a speaker and the executor, one Δ row at a time (D106): the observed rows that the start of the closed loop gives back (vocabulary §6 item 5: `start_moved`'s, a moved start's moved rows; never read from the stored sentence), then the inputs of each row, the speaker with the caller's numbers, masks and input of the added modules, the executor's step through the start of a closed loop, the flights done halted, a flight ended by the caller; a copy of its state for chosen flights; free generation's random numbers of a flight. Free generation and the post-training's window loop use it. Built (multi-aircraft control D150, MC0): each aircraft joins at its own tick — `step` advances one tick for every aircraft, absent, observed or said by its own clock; each aircraft's records, words, states and sentence are of its own rows from its row 0, as today; a landing added to chosen aircraft's landings, which the loop keeps for its caller (D105, D206); the copy keeps the join ticks. With every join tick 0, today's loop, bit for bit | `experiments/prior_speaking_loop.py` `SpeakingLoop` (`SpeakingLoop(model, loop, order, sentences, observed, flights, geometries, landings, finals, words, *, interval_s, variant, device, temperature)`: `observed` the start's rows; the executor flies a row inside the speaker's `accept`, so a refused row leaves the loop as it was; `copy(flights)` on the loop's own `Loop.copy`; `said(b)`, `states(b)`; `generated(flights)`, which refuses a flight the caller ended, D118), `Generated`; free generation's numbers `flight_numbers`; stage D's: `SpeakingLoop.join_ticks`, `roles`, `own_row`, `joined`, `said_now`, `add_landing`; `held`, `left` (each left flight's frozen record), `inside`, `outcome(flight)`, `landings(flight)`, `in_force(flight)`, `permitted(flights)` (one record a flight; multi-aircraft control D193 (3)); `SpeakingLoop.copy(flights, device=)` (multi-aircraft control D182); `SpeakingLoop(…, executor_device=)`, the executor's device, kept by a copy (multi-aircraft control D193 (1)) |
```

**New text:**

```text
| 7 | The step of a speaker's closed loop | The closed loop of a speaker and the executor, one Δ row at a time (D106): the observed rows that the start of the closed loop gives back (vocabulary §6 item 5: `start_moved`'s, a moved start's moved rows; never read from the stored sentence), then the inputs of each row, the speaker with the caller's numbers, masks and input of the added modules, the executor's step through the start of a closed loop, the flights done halted, a flight ended by the caller; a copy of its state for chosen flights; free generation's random numbers of a flight. Free generation and the post-training's window loop use it. Built (multi-aircraft control D150, MC0): each aircraft joins at its own tick — `step` advances one tick for every aircraft, absent, observed or said by its own clock; each aircraft's records, words, states and sentence are of its own rows from its row 0, as today; the copy keeps the join ticks. With every join tick 0, today's loop, bit for bit | `experiments/prior_speaking_loop.py` `SpeakingLoop` (`SpeakingLoop(model, loop, order, sentences, observed, flights, geometries, finals, words, *, interval_s, variant, device, temperature)`: `observed` the start's rows; the executor flies a row inside the speaker's `accept`, so a refused row leaves the loop as it was; `copy(flights)` on the loop's own `Loop.copy`; `said(b)`, `states(b)`; `generated(flights)`, which refuses a flight the caller ended, D118), `Generated`; free generation's numbers `flight_numbers`; stage D's: `SpeakingLoop.join_ticks`, `roles`, `own_row`, `joined`, `said_now`; `held`, `left` (each left flight's frozen record), `inside`, `outcome(flight)`, `in_force(flight)`, `permitted(flights)` (one record a flight; multi-aircraft control D193 (3)); `SpeakingLoop.copy(flights, device=)` (multi-aircraft control D182); `SpeakingLoop(…, executor_device=)`, the executor's device, kept by a copy (multi-aircraft control D193 (1)) |
```

### 134. `prior.md`: §9 Values, the landings that the loops keep

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| The landings that the loops keep (no input, D206) | Counted in [t − 30 min, t), the flight's own landing not counted | §2, D63 |
```

**New text:**

(deleted)

### 135. `prior.md`: §11 Code index, the landings

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| The landings that the loops keep (D206): the index, its digest, the count in [t − 30 min, t) without the own landing, the roster | `prior/landings.py:40` `LandingIndex`, `:67` `digest`, `:121` `counts_before`, `:140` `roster_landings` |
```

**New text:**

```text
| The landings of an airport (D63; post-training D105): the index, its digest, the roster | `prior/landings.py:30` `LandingIndex`, `:53` `digest`, `:65` `roster_landings` |
```

### 136. `vocabulary.md`: O23

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| O23 | Stage D's landing context: a commanded aircraft adds its landing to the others' context only when the judge says `landed`; D196 passes fewer crossings than the judge before it. No input (prior D206) and no reward (post-training D207) reads that context now (post-training O26) | Closes with post-training O26 | 5.8 |
```

**New text:**

(deleted)

### 137. `outline.md`: §3.2 Index

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| Vocabulary | D1–D4, D6, D8–D12, D14, D15, D18, D19, D22, D25–D28, D32–D35, D38, D42–D54, D56–D59, D61, D62, D66, D67, D69–D71, D73, D74, D77–D84, D86–D90, D97, D102, D111, D126, D196, D197, D199, D201, D203 | O8, O22, O23 |
```

**New text:**

```text
| Vocabulary | D1–D4, D6, D8–D12, D14, D15, D18, D19, D22, D25–D28, D32–D35, D38, D42–D54, D56–D59, D61, D62, D66, D67, D69–D71, D73, D74, D77–D84, D86–D90, D97, D102, D111, D126, D196, D197, D199, D201, D203 | O8, O22 |
```

### 138. `outline.md`: §3.2 Index

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| Post-training | D29–D31, D36, D37, D76, D91–D94, D98–D101, D103–D105, D107, D110, D113–D117, D123–D125, D129, D130, D132, D137, D157, D161, D162, D165, D167–D171, D173–D176, D185, D207 | O15, O26 |
```

**New text:**

```text
| Post-training | D29–D31, D36, D37, D76, D91–D94, D98–D101, D103–D105, D107, D110, D113–D117, D123–D125, D129, D130, D132, D137, D157, D161, D162, D165, D167–D171, D173–D176, D185, D207, D208 | O15, O26 |
```

### 139. `outline.md`: §3.2 Index

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| Multi-aircraft control | D140–D153, D164, D166, D172, D179–D183, D186, D188–D195, D200 | O18–O21 |
```

**New text:**

```text
| Multi-aircraft control | D140–D146, D148–D153, D164, D166, D172, D179–D183, D186, D188–D195, D200 | O18–O21 |
```

### 140. `multi_control.md`: D150

**What changed:** the loops keep no landings (post-training D208)

**Old text:**

```text
| D150 | **The aircraft of one closed loop join it at their own Δ rows, in one batch** (§6.3). A loop has a row clock (its ticks, one Δ each); each aircraft has a join tick j: before it the aircraft is absent (its rows not present, not spoken, not flown, its executor waiting), from it the aircraft's own row 0, then its observed rows to its own first predicted step, then its said rows. Built in stages A's and B's code (the start and `Loop`; `LoopRows`, the speaker, `SpeakingLoop`), each written first in its stage's public interface (vocabulary §6 item 5, prior §7 items 2, 3, 7). The prior's network does not change: its time attention already reads only present rows (`Past.present`) and time differences (RoPE on each aircraft's own seconds). With every join tick 0 the loop is today's loop, bit for bit (a test); free generation and stage C call it so. Also: a landing added to chosen aircraft's landings while the loop runs (D147). Why: the commanded aircraft of a window enter at different times; one loop for each entry row would speak a row in up to L / Δ small batches (150 at L = 10 min), each with its fixed cost in Python, and the speaking is bound by the CPU (outline D138) | Decided | User, 2026-10-06 (one batch with each aircraft's own first row, Claude's option 1); the design of §6.3: Claude |
```

**New text:**

```text
| D150 | **The aircraft of one closed loop join it at their own Δ rows, in one batch** (§6.3). A loop has a row clock (its ticks, one Δ each); each aircraft has a join tick j: before it the aircraft is absent (its rows not present, not spoken, not flown, its executor waiting), from it the aircraft's own row 0, then its observed rows to its own first predicted step, then its said rows. Built in stages A's and B's code (the start and `Loop`; `LoopRows`, the speaker, `SpeakingLoop`), each written first in its stage's public interface (vocabulary §6 item 5, prior §7 items 2, 3, 7). The prior's network does not change: its time attention already reads only present rows (`Past.present`) and time differences (RoPE on each aircraft's own seconds). With every join tick 0 the loop is today's loop, bit for bit (a test); free generation and stage C call it so. Why: the commanded aircraft of a window enter at different times; one loop for each entry row would speak a row in up to L / Δ small batches (150 at L = 10 min), each with its fixed cost in Python, and the speaking is bound by the CPU (outline D138) | Decided | User, 2026-10-06 (one batch with each aircraft's own first row, Claude's option 1); the design of §6.3: Claude |
```

### 141. `prior.md`: D63

**What changed:** the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed)

**Old text:**

```text
| D63 | The identity of a prior's data (§8, item 1) also holds the airport's landings (no input reads them since D206, and no loop since post-training D208; post-training O26): for each airport, the sha256 of each landing's flight, runway and time, in time order, and of the number of landings left out on the sealed test days. The landings come from the airport's tracks roster, outside the artefact: without them, a changed roster would have changed the inputs (before D206) and left the identity the same. By their flights, never by the bytes of the roster (D21). A run that reads the landings again computes the digest again and refuses a difference (§2, §8) | Decided | User, 2026-10-04, on the reading of the stage B agent |
```

**New text:**

```text
| D63 | The identity of a prior's data (§8, item 1) holds no landings: no input (D206) and no loop (post-training D208) reads them. A checkpoint written before 2026-10-14 (B21's) holds their digest; the entry is kept as written and not compared (`checkpoint.NOT_COMPARED`). Opening a prior reads no tracks roster; only the diagnostic readout of stage C reads an airport's landings (post-training D175) | Decided | User, 2026-10-04, on the reading of the stage B agent; the digest removed: user, 2026-10-14 ("Drop it: not written, not compared") |
```

### 142. `prior.md`: §2, No landings

**What changed:** the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed)

**Old text:**

```text
  index of landings refuses a landing on a test day or on a day outside the day split by itself. The identity of
  a prior's data holds a digest of them (§8, item 1; D63; post-training O26).
```

**New text:**

```text
  index of landings refuses a landing on a test day or on a day outside the day split by itself. The identity of
  a prior's data does not hold them (§8, item 1; D63); only the diagnostic readout of stage C reads them
  (post-training D175).
```

### 143. `prior.md`: §8 item 1

**What changed:** the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed)

**Old text:**

```text
| 1 | The artefact of a prior | The spec sha, the day split, the candidate table (`candidates.json`), the row interval Δ, the sha256 of each split's closed-loop sentence file at Δ and of its stored signals (D118: they bind the faulty-track marks, D111), the selection of its sentences (D75: the rule and, for each split, airport, stratum and outcome, the sentences kept and left out for each reason, D111; val's counts held and never shown, D120), and the landings that the candidate vectors count (D63): for each airport, the sha256 of each landing's flight, runway and time, in time order, and of the number left out on the sealed test days. A checkpoint opens only for an equal identity, compared whole; a reader before the val read compares val's counts by the files' sha256, and the val readers count them again after their claim (D118). A run that reads the landings again computes the digest again and refuses a difference |
```

**New text:**

```text
| 1 | The artefact of a prior | The spec sha, the day split, the candidate table (`candidates.json`), the row interval Δ, the sha256 of each split's closed-loop sentence file at Δ and of its stored signals (D118: they bind the faulty-track marks, D111), the selection of its sentences (D75: the rule and, for each split, airport, stratum and outcome, the sentences kept and left out for each reason, D111; val's counts held and never shown, D120). No landings (D63). A checkpoint opens only for an equal identity, compared whole (a stored `landings` entry, written before 2026-10-14, is not compared: `checkpoint.NOT_COMPARED`); a reader before the val read compares val's counts by the files' sha256, and the val readers count them again after their claim (D118). |
```

### 144. `prior.md`: code index, the landings

**What changed:** the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed)

**Old text:**

```text
| The landings of an airport (D63; post-training D105): the index, its digest, the roster | `prior/landings.py:30` `LandingIndex`, `:53` `digest`, `:65` `roster_landings` |
```

**New text:**

```text
| The landings of an airport (post-training D105, D175): the index, the roster, every airport's | `prior/landings.py:29` `LandingIndex`, `:57` `roster_landings`; `prior/source.py:79` `airport_landings` |
```

### 145. `prior.md`: code index, the identity

**What changed:** the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed)

**Old text:**

```text
| The identity of the data (§8 item 1); the val recount after the claim | `prior/source.py:45` `artefact_identity`, `:68` `require_selection_of` |
```

**New text:**

```text
| The identity of the data (§8 item 1); the val recount after the claim | `prior/source.py:45` `artefact_identity`, `:66` `require_selection_of` |
```

### 146. `prior.md`: code index, the artefact

**What changed:** the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed)

**Old text:**

```text
| The artefact as sentences of a selection | `prior/source.py:87` `ArtefactSource` |
```

**New text:**

```text
| The artefact as sentences of a selection | `prior/source.py:85` `ArtefactSource` |
```

### 147. `prior.md`: code index, the checkpoint

**What changed:** the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed)

**Old text:**

```text
| The checkpoint; the val read: its lock, its claim with the reader's options, a written readout under an unspent claim, the spent mark, a written claim (D119, D128); the identity a readout shows; opening a prior run | `prior/checkpoint.py:67` `load_checkpoint`, `:89` `CLAIM_SPENT_BY`, `:99` `lock_val_read`, `:113` `claim_validation_read`, `:152` `settle_written_claim`, `:164` `spend_validation_claim`, `:174` `holds_written_claim`, `:180` `written_claim`, `:190` `readable_identity`, `:208` `open_prior` |
```

**New text:**

```text
| The checkpoint; the val read: its lock, its claim with the reader's options, a written readout under an unspent claim, the spent mark, a written claim (D119, D128); the identity a readout shows; opening a prior run | `prior/checkpoint.py:46` `NOT_COMPARED`, `:72` `load_checkpoint`, `:94` `CLAIM_SPENT_BY`, `:104` `lock_val_read`, `:118` `claim_validation_read`, `:157` `settle_written_claim`, `:169` `spend_validation_claim`, `:179` `holds_written_claim`, `:185` `written_claim`, `:195` `readable_identity`, `:212` `open_prior` |
```

### 148. `post_training.md`: D208

**What changed:** the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed)

**Old text:**

```text
| D208 | **The loops keep no landings** (O26): no input of the prior (prior D206) and no reward (D207) reads them. Deleted, with their tests: the landings that a speaker's loop keeps for each aircraft (prior §7 item 7: `SpeakingLoop`'s argument `landings`, `landings(flight)`, `add_landing`); a commanded aircraft's landing added to the others' in the loop (multi-aircraft control D147, D186 (1), D189 (0): `post.landings.commanded_landings`, `LandingIndex.with_landing`); the count of the 30 min before a step (`LandingIndex.counts_before`, `LANDINGS_WINDOW_S`); the rosters that the loops took (`WindowLoop`, `speak_and_fly`, `branch_round`, `backward_round`). Kept: the landings of a window (D105, `window_landings`), which the diagnostic readout's field 1 reads (D175, `other_class`), and the landings' digest in the identity of a prior's data (prior D63): without it the identity changes, and every prior checkpoint written before, B21's base too, is refused. Both stay in O26. No word, state, outcome, reward or record of free generation or of a window loop changes (their tests on fixed inputs) | Decided | User, 2026-10-14 (O26: "现在就删"); what is kept: Claude, for the user to confirm |
```

**New text:**

```text
| D208 | **The loops keep no landings** (O26): no input of the prior (prior D206) and no reward (D207) reads them. Deleted, with their tests: the landings that a speaker's loop keeps for each aircraft (prior §7 item 7: `SpeakingLoop`'s argument `landings`, `landings(flight)`, `add_landing`); a commanded aircraft's landing added to the others' in the loop (multi-aircraft control D147, D186 (1), D189 (0): `post.landings.commanded_landings`, `LandingIndex.with_landing`); the count of the 30 min before a step (`LandingIndex.counts_before`, `LANDINGS_WINDOW_S`); the rosters that the loops took (`WindowLoop`, `speak_and_fly`, `branch_round`, `backward_round`). Kept: the landings of a window (D105, `window_landings`), which the diagnostic readout's field 1 reads (D175, `other_class`); the readout reads each airport's roster itself (`prior.source.airport_landings`), and no context holds them (`Context.rosters` deleted). The identity of a prior's data holds no landings (prior D63). No word, state, outcome, reward or record of free generation or of a window loop changes (their tests on fixed inputs) | Decided | User, 2026-10-14 (O26: "现在就删"; the window's landings for D175: "Keep"; the digest: prior D63) |
```

### 149. `post_training.md`: O26

**What changed:** the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed)

**Old text:**

```text
| O26 | What D208 kept of the landings that no input, no reward and no loop reads: the landings of a window (D105, `post/landings.py` `window_landings`), read only by the diagnostic readout's field 1 (D175, `other_class`), and their digest in the identity of a prior's data (prior D63, `prior/source.py` `artefact_identity`) | The digest goes with the next prior campaign (the identity then changes its format, and the checkpoints before it are refused); D175's field 1 without the window's landings, or the landings kept for it alone. The user decides | §3 |
```

**New text:**

(deleted)

### 150. `post_training.md`: §3, Landings

**What changed:** the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed)

**Old text:**

```text
  readout reads them (D175; O26).
```

**New text:**

```text
  readout reads them (D175).
```

### 151. `outline.md`: index, post-training

**What changed:** the identity of a prior's data holds no landings (prior D63); D175 keeps the window's landings (O26 closed)

**Old text:**

```text
| Post-training | D29–D31, D36, D37, D76, D91–D94, D98–D101, D103–D105, D107, D110, D113–D117, D123–D125, D129, D130, D132, D137, D157, D161, D162, D165, D167–D171, D173–D176, D185, D207, D208 | O15, O26 |
```

**New text:**

```text
| Post-training | D29–D31, D36, D37, D76, D91–D94, D98–D101, D103–D105, D107, D110, D113–D117, D123–D125, D129, D130, D132, D137, D157, D161, D162, D165, D167–D171, D173–D176, D185, D207, D208 | O15 |
```

### 152. `prior.md`: D63

**What changed:** a checkpoint opens on the meaning and the day split; the files' sha256 and the selection's counts describe the data (prior D209)

**Old text:**

```text
| D63 | The identity of a prior's data (§8, item 1) holds no landings: no input (D206) and no loop (post-training D208) reads them. A checkpoint written before 2026-10-14 (B21's) holds their digest; the entry is kept as written and not compared (`checkpoint.NOT_COMPARED`). Opening a prior reads no tracks roster; only the diagnostic readout of stage C reads an airport's landings (post-training D175) | Decided | User, 2026-10-04, on the reading of the stage B agent; the digest removed: user, 2026-10-14 ("Drop it: not written, not compared") |
```

**New text:**

```text
| D63 | The identity of a prior's data (§8, item 1) holds no landings: no input (D206) and no loop (post-training D208) reads them. A checkpoint written before 2026-10-14 (B21's) holds their digest; the entry is kept as written and not compared (D209: not an opened entry). Opening a prior reads no tracks roster; only the diagnostic readout of stage C reads an airport's landings (post-training D175) | Decided | User, 2026-10-04, on the reading of the stage B agent; the digest removed: user, 2026-10-14 ("Drop it: not written, not compared") |
```

### 153. `prior.md`: D118

**What changed:** a checkpoint opens on the meaning and the day split; the files' sha256 and the selection's counts describe the data (prior D209)

**Old text:**

```text
| D118 | Stage B's readings of B9–B11, accepted by the user: (1) a faulty flight that also did not land is left out for its fault (the mark is asked first); (2) the identity of a prior's data also holds the sha256 of each split's stored signals (§8 item 1): the faulty-track marks are read from the signals, so a reader before the claim of the val read (`open_prior`) checks the val marks by the files' sha256 without reading them (D85) — the training run itself reads val's outcomes and marks to count its identity, held and never shown (D120); it binds data, not code — it changes only when the artefact is built again, when the sentence files' sha256 changes too, so no change of code refuses a prior unless it changes which flights the selection keeps; (3) the claim of the val read names its readout relative to the repository; the base's one validation readout is two readers, each with its own claim (`prior_validation`: the teacher-forced loss and the masks on the labelled words; `prior_free_generation`: the free generation, whose set the Training view opens, D109); (4) the claimed validation set flies on a second service of stage A's live segment that permits val alone (frontend D109), so stage A's code is not changed; (5) after the claim, the val readers count val's selection again and refuse a difference before they read (with (2), only a change of stage A's fault rule that moves no train or select mark can cause it; nothing is written, so the claim may be run again, D119); (6) a flight that the caller ends (post-training D93) gets the caller's outcome: the closed loop's step gives its words and states and refuses to give it an outcome of the judge; (7) D108's behaviour check takes the artefact's vocabulary spec and its first airport's finals as its fixed inputs, and covers two steps of training, the speaker, the input functions, the variant `constants`, the first-step runway and the closed loop's step on a straight-flying stand-in; the campaign also compares its own settings (the seeds, the selection, the configurations, free generation, the temperature, D68's bound); (8) a flight's random numbers in free generation come from numpy `default_rng([seed, sample, index])`; the log-probability under records is 0 at a row not asked, and an asked row reads the record said at its time, its own-state inputs checked against the record's. Free generation and the validation readout write the identity without the val counts (D85) | Decided | User, 2026-10-05 |
```

**New text:**

```text
| D118 | Stage B's readings of B9–B11, accepted by the user: (1) a faulty flight that also did not land is left out for its fault (the mark is asked first); (2) the identity of a prior's data also holds the sha256 of each split's stored signals (§8 item 1), from which the faulty-track marks are read: a description of the data, not compared when a prior opens (D209) — the training run itself reads val's outcomes and marks to count its identity, held and never shown (D120); (3) the claim of the val read names its readout relative to the repository; the base's one validation readout is two readers, each with its own claim (`prior_validation`: the teacher-forced loss and the masks on the labelled words; `prior_free_generation`: the free generation, whose set the Training view opens, D109); (4) the claimed validation set flies on a second service of stage A's live segment that permits val alone (frontend D109), so stage A's code is not changed; (5) after the claim, the val readers count val's selection again and refuse a difference before they read (nothing is written, so the claim may be run again, D119); (6) a flight that the caller ends (post-training D93) gets the caller's outcome: the closed loop's step gives its words and states and refuses to give it an outcome of the judge; (7) D108's behaviour check takes the artefact's vocabulary spec and its first airport's finals as its fixed inputs, and covers two steps of training, the speaker, the input functions, the variant `constants`, the first-step runway and the closed loop's step on a straight-flying stand-in; the campaign also compares its own settings (the seeds, the selection, the configurations, free generation, the temperature, D68's bound); (8) a flight's random numbers in free generation come from numpy `default_rng([seed, sample, index])`; the log-probability under records is 0 at a row not asked, and an asked row reads the record said at its time, its own-state inputs checked against the record's. Free generation and the validation readout write the identity without the val counts (D85) | Decided | User, 2026-10-05 |
```

### 154. `prior.md`: §8 item 1

**What changed:** a checkpoint opens on the meaning and the day split; the files' sha256 and the selection's counts describe the data (prior D209)

**Old text:**

```text
| 1 | The artefact of a prior | The spec sha, the day split, the candidate table (`candidates.json`), the row interval Δ, the sha256 of each split's closed-loop sentence file at Δ and of its stored signals (D118: they bind the faulty-track marks, D111), the selection of its sentences (D75: the rule and, for each split, airport, stratum and outcome, the sentences kept and left out for each reason, D111; val's counts held and never shown, D120). No landings (D63). A checkpoint opens only for an equal identity, compared whole (a stored `landings` entry, written before 2026-10-14, is not compared: `checkpoint.NOT_COMPARED`); a reader before the val read compares val's counts by the files' sha256, and the val readers count them again after their claim (D118). |
```

**New text:**

```text
| 1 | The artefact of a prior | The spec sha, the day split, the candidate table (`candidates.json`), the row interval Δ, the sha256 of each split's closed-loop sentence file at Δ and of its stored signals (the faulty-track marks are read from the signals, D111), the selection of its sentences (D75: the rule and, for each split, airport, stratum and outcome, the sentences kept and left out for each reason, D111; val's counts held and never shown, D120). No landings (D63). A checkpoint opens only on an equal spec sha, candidate table, Δ, day split and selection rule (D209); the files' sha256 and the selection's counts describe the data and are not compared; only the val readers count a selection again, val's, after their claim (D118 (5)). |
```
