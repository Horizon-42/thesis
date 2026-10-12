# Stage D of the two-tier model: the multi-aircraft control

Experiment report, 2026-10-12. Branch `dev-two-tier`. This report uses saved files only. No new experiment ran for it.

The LaTeX file `2026-10-12_stage_d_report_tables.tex` holds every table of this report with the same numbers.

**Terms.** A *commanded* aircraft is an aircraft for which the model speaks. A *recorded* aircraft flies its record and does not react. The *landed share* is the number of landed commanded aircraft divided by the number of commanded aircraft. KL is the Kullback-Leibler distance of the model from the base model. ADE and FDE are the average and the final displacement error against the observed track (Table 9). Every share in the tables has the commanded aircraft as its denominator, unless the caption says otherwise.

## 1 Summary

Stage D trains the two-tier model to command every arrival of a traffic window at the same time. The formal campaign ran 12 rounds from stage C round 5, and the user chose round 7. On the select days round 7 lands 83.7 % of the commanded aircraft, and the start model lands 81.5 % on the same windows. On the val days, which the user read once, round 7 lands 83.9 % of 4,534 commanded aircraft. After round 5 the select landed share stays between 83.0 % and 83.7 %, so more rounds of the same method give no measurable gain.

The evaluation package passes 63.8 % of the val aircraft, and it passes 90.2 % of the observed tracks of the same flights. The largest gap is the lateral alignment at the threshold, and the next gap is loss of separation (606 of the 731 val failures). Nobody read the start model on the val days, so the val gain of stage D is not known.

## 2 Purpose and question

Stage C trains the model to command one aircraft in a window of recorded traffic. The other aircraft fly their records and do not react. Stage D removes this limit. The model commands every arrival of a time span, and each aircraft flies its own words. The model must keep the separation between all aircraft of the window.

This report answers three questions.

1. Can the model land all commanded aircraft of a window without a loss of separation?
2. Do the landed share, the lost separation, the reward and the KL change from round to round?
3. How good are the landed tracks of the chosen round when the evaluation package grades them?

Each campaign in Table 6 also asks a narrower question on the method.

## 3 Method

### 3.1 The windows and the commanded aircraft

A window has an anchor flight and a span L of 300 s, 600 s or 1,200 s. The model commands each arrival of the same airport and split that starts in the span. The span begins at the first state of the anchor and lasts L. Every other flight in the air flies its record.

The model speaks for all commanded aircraft at each row of 4 s. Each aircraft flies its own words through the executor. A window ends when each commanded aircraft ends its flight or becomes silent. The model reads no span, so it can fly any span.

A real window keeps the recorded times. A compressed window moves each commanded aircraft closer to the anchor in time, by a factor between 0.8 and 1. Compressed windows exist for the update passes only. The select and val reads use real windows only.

### 3.2 The reward

The reward of one aircraft is 1 if it lands with no go-around. It is 0.9 to the power n after n go-arounds. It is 0 for every other outcome. It is also 0 when the aircraft answers for a loss of separation. The reward of a window is the sum of the rewards of its commanded aircraft. The sum is not divided by the number of aircraft.

### 3.3 Loss of separation

The separation judge applies the rules of FAA Order 7110.65 in the mode `VISUAL` (`inference/separation.py`). After each row it judges every pair of aircraft in the window. The rules name the responsible aircraft: the aircraft behind in trail, the aircraft that joins the final, or both. A commanded aircraft answers for a loss with a recorded aircraft where the rules make it responsible. It also answers where only the recorded aircraft is responsible but the two records kept their separation at that step.

An aircraft that answers for a loss becomes silent. Its reward is 0 from that step, and its outcome is `lost_separation`. It does not leave the scene. It flies on its last words, and the other aircraft must keep their separation from it. A loss that only the recorded aircraft cause, and that the records also have, earns nothing and costs nothing.

### 3.4 The judge: outcome `landed`

The executor flies the words of the model. The two-tier judge (`autopilot/judge.py`) ends the flight and names the outcome. The outcome `landed` needs that the aircraft cross the threshold plane in line with the runway. At the cross, the height must be 100 m or less, and the lateral offset must be 106.7 m or less.

The aircraft must also pass the decision-altitude check. There the aircraft must be inside the cone of the final approach segment (FAS) and within 22 m of the glidepath. The judge has no speed gate. Table 11 compares this judge with the evaluation package.

### 3.5 One round

A round has three steps: speak, update and read.

In the speak step, the model speaks for the 2,000 windows of the round from the train days. Each commanded aircraft with a reward below 1, and each aircraft in a loss, is a varied aircraft. For each varied aircraft the search goes back from its event in steps of 32 s, at most 8 points. At each point the model speaks 8 new continuations for that aircraft only. A continuation avoids the event if the event does not happen up to 60 s after it. The aircraft must also not answer for a loss.

The option `majority` keeps the first point back where at least 5 of the 8 continuations avoid the event. The search also keeps the first point where one continuation avoids the event, and a point 16 s after it.

Each kept point gives a group: the first sentence and its 8 continuations. The advantage of a sentence is the reward of its window less the mean reward of the group. The advantage applies only to the words of the varied aircraft after the point.

In the update step, the loss has three terms. The first term is the clipped-ratio surrogate (clip 0.2) on the advantages. The second term pulls the model toward the base model (weight 0.04). The third term is the cross-entropy of 64 single-aircraft sentences of the train days (weight 1). In the read step, the model speaks the select windows once. Table 1 gives the settings of the formal campaign.

**Table 1.** Settings of the formal campaign (stage D).

| Setting | Value |
|---|---|
| Start model | stage C, campaign post_seg60_20261007, round 5 (checkpoint SHA-256 0436cb79...) |
| Seed | 2035 |
| Rounds | 12 (8 planned at the start; the user extended the campaign to 12) |
| Windows of each round (train days) | 2,000: 1,000 real and 1,000 compressed |
| Spans of a window | 300 s, 600 s and 1,200 s, in equal parts |
| Compression factor of a compressed window | between 0.8 and 1 |
| Continuations of a branch point (K) | 8 |
| Branch points | majority: the first point back from the event where at least 5 of the 8 continuations avoid the event |
| Learning rate: prior / traffic module | 1e-5 / 1e-4 |
| Weight decay | 0.01 |
| Groups and single-aircraft sentences of an update | 4 groups and 64 sentences |
| Weight of the data term / of the pull to the base model | 1 / 0.04 |
| Select read | 100 anchors for each airport and span: 1,500 real windows |
| Speaking workers | 3 on the GPU and 4 on the CPU; 8 of every 15 batches on the CPU |
| Code at the first launch | 226ed303 |

Source: `4dTrajectory/outputs/POOLED/multi/multi_train_majority_formal_20261011/campaign.json`; the pull weight 0.04 is in `docs/experiments/intents.json` (key multi_train_majority_formal_20261011).

### 3.6 Splits and selection

The data split is by operating day. The file `data/day_split_20260924.json` holds 53 train days, 9 select days, 14 val days and 14 test days. The test days stay sealed. No read of stage D opens them.

The select read uses 100 anchors for each airport and span. It has 1,500 real windows and 4,492 commanded aircraft. It uses the same windows and the same random numbers in every round. The campaigns read the select split after every round, so its numbers read high for the chosen round.

The val read uses the same rule on the val days. It has 1,500 windows and 4,534 commanded aircraft. The user read the val days once, for round 7.

The user chose the criterion before the val read: the highest select reward sum over the rounds of the campaigns. The reward sum is the sum of the rewards of the 4,492 select aircraft.

## 4 Results

### 4.1 The formal campaign by round

Table 2 shows the select read of every round. Round r is the model after the update step of round r. The row `start` is the stage C model before any stage D update. The first campaign read it once on the same select windows, and no second read checked it. The landed share rises from 81.5 % at the start to 83.3 % in round 5. From round 5 to round 11 it stays between 83.0 % and 83.7 %.

**Table 2.** The formal campaign by round on the select days (4,492 commanded aircraft in 1,500 windows). Round 7 is the chosen round.

| Round | Landed | Lost separation | Other failures | Reward sum | Reward per aircraft | KL | Groups |
|---|---|---|---|---|---|---|---|
| start | 3,660 (81.5 %) | 737 (16.4 %) | 95 (2.1 %) | 3633.7 | 0.8089 | - | - |
| 0 | 3,677 (81.9 %) | 720 (16.0 %) | 95 (2.1 %) | 3652.9 | 0.8132 | 0.0437 | 1,734 |
| 1 | 3,667 (81.6 %) | 730 (16.3 %) | 95 (2.1 %) | 3637.6 | 0.8098 | 0.0511 | 1,768 |
| 2 | 3,706 (82.5 %) | 690 (15.4 %) | 96 (2.1 %) | 3678.8 | 0.8190 | 0.0512 | 1,681 |
| 3 | 3,733 (83.1 %) | 648 (14.4 %) | 111 (2.5 %) | 3703.7 | 0.8245 | 0.0535 | 1,816 |
| 4 | 3,719 (82.8 %) | 663 (14.8 %) | 110 (2.4 %) | 3690.9 | 0.8217 | 0.0589 | 1,615 |
| 5 | 3,741 (83.3 %) | 625 (13.9 %) | 126 (2.8 %) | 3711.8 | 0.8263 | 0.0684 | 1,704 |
| 6 | 3,750 (83.5 %) | 625 (13.9 %) | 117 (2.6 %) | 3718.8 | 0.8279 | 0.0660 | 1,646 |
| 7 | 3,759 (83.7 %) | 604 (13.4 %) | 129 (2.9 %) | 3727.2 | 0.8297 | 0.0704 | 1,659 |
| 8 | 3,756 (83.6 %) | 613 (13.6 %) | 123 (2.7 %) | 3726.9 | 0.8297 | 0.0745 | 1,824 |
| 9 | 3,746 (83.4 %) | 610 (13.6 %) | 136 (3.0 %) | 3714.9 | 0.8270 | 0.0700 | 1,544 |
| 10 | 3,727 (83.0 %) | 638 (14.2 %) | 127 (2.8 %) | 3703.8 | 0.8245 | 0.0733 | 1,787 |
| 11 | 3,762 (83.7 %) | 593 (13.2 %) | 137 (3.0 %) | 3734.5 | 0.8314 | 0.0746 | 1,652 |

Note: Round r is the model after the update pass of round r. The start row is the stage C model before any stage D update.

Source: `4dTrajectory/outputs/POOLED/multi/multi_train_majority_formal_20261011/round_<r>/round.json` (selection_readout, pass.kl, speaking.groups); the start row: `4dTrajectory/outputs/POOLED/multi/handover_20261010/start_read/start_readout.json`.

Lost separation falls from 737 aircraft (16.4 %) at the start to 604 (13.4 %) in round 7. The other failures rise from 95 aircraft (2.1 %) to 129 (2.9 %) in round 7, and to 137 (3.0 %) in round 11. The KL grows from 0.044 in round 0 to 0.075 in round 11. The data term stays between 1.06 and 1.07 in every round.

Round 7 has the highest reward sum of rounds 0 to 10 (3727.2). Round 8 has 3726.9. Round 11 has 3734.5, which is 7.3 more than round 7. The reward sums of rounds 5 to 11 lie between 3703.8 and 3734.5. The user judged the difference of 7.3 to be inside the spread between rounds, and the user kept round 7. At that time the val read of round 7 existed, and it used the val claim.

The stop rule of the extension was: two rounds in a row not above 3727.2, a KL above 0.10, or more than 150 other failures. Rounds 8 and 9 did not beat 3727.2, so the rule stopped the extension. The user then ordered rounds 10 and 11, to complete the 12 rounds.

### 4.2 Round 7 by airport and by span

Table 3 gives round 7 by airport. On the select days the landed share rises most at KSTL (5.2 points) and KRDU (3.5 points). It rises by 2.1 points at KMSY. It does not change at KSJC and KSMF. On the val days KSJC lands 90.3 % and KSMF lands 77.5 %.

The val share of lost separation is highest at KSMF (18.7 %) and KRDU (16.8 %). It is lowest at KMSY (7.7 %) and KSJC (8.4 %).

**Table 3.** Round 7 by airport. Select: start model and round 7. Val: round 7.

| Airport | Select aircraft | Start landed | Round 7 landed | Change (points) | Round 7 lost separation | Val aircraft | Val landed | Val lost separation |
|---|---|---|---|---|---|---|---|---|
| KMSY | 748 | 86.8 % | 88.9 % | +2.1 | 7.9 % | 744 | 87.8 % | 7.7 % |
| KRDU | 1,060 | 78.5 % | 82.0 % | +3.5 | 16.0 % | 1,135 | 81.2 % | 16.8 % |
| KSJC | 925 | 83.2 % | 82.9 % | -0.3 | 13.6 % | 884 | 90.3 % | 8.4 % |
| KSMF | 806 | 81.6 % | 81.5 % | -0.1 | 15.0 % | 877 | 77.5 % | 18.7 % |
| KSTL | 953 | 78.8 % | 84.1 % | +5.2 | 13.4 % | 894 | 83.9 % | 13.4 % |
| All | 4,492 | 81.5 % | 83.7 % | +2.2 | 13.4 % | 4,534 | 83.9 % | 13.4 % |

Note: The select and val days hold different flights, so the airport shares of the two splits do not compare one to one.

Source: `4dTrajectory/outputs/POOLED/multi/multi_train_majority_formal_20261011/round_7/round.json`; `handover_20261010/start_read/start_readout.json`; `val_majority_formal_r7_20261012/readout.json`. The short paths are under `4dTrajectory/outputs/POOLED/multi/`.

Table 4 gives round 7 by span. On the select days the landed share rises at every span, and it rises most at 600 s (from 80.8 % to 83.9 %). On the val days the shares are 84.3 %, 84.9 % and 83.1 %. The binomial standard error of one span lies between 0.8 and 1.2 points, so the spans do not differ. The aircraft of one window are not independent, so the true standard error is larger.

**Table 4.** Round 7 by window span (s). Select: start model and round 7. Val: round 7.

| Span (s) | Select aircraft | Start landed | Round 7 landed | Start lost separation | Round 7 lost separation | Val aircraft | Val landed | Val lost separation |
|---|---|---|---|---|---|---|---|---|
| 300 | 935 | 83.0 % | 83.4 % | 14.8 % | 12.6 % | 930 | 84.3 % | 13.7 % |
| 600 | 1,372 | 80.8 % | 83.9 % | 17.0 % | 13.8 % | 1,374 | 84.9 % | 12.3 % |
| 1,200 | 2,185 | 81.3 % | 83.7 % | 16.8 % | 13.6 % | 2,230 | 83.1 % | 13.9 % |
| All | 4,492 | 81.5 % | 83.7 % | 16.4 % | 13.4 % | 4,534 | 83.9 % | 13.4 % |

Source: Same files as Table 3.

Table 5 gives the val landed share by airport and span.

**Table 5.** Round 7 on the val days: landed share and, in brackets, the number of aircraft that lost separation, by airport and span.

| Airport | 300 s | 600 s | 1,200 s | All spans |
|---|---|---|---|---|
| KMSY | 88.5 % (15) | 89.1 % (13) | 86.6 % (29) | 87.8 % (57) |
| KRDU | 82.1 % (34) | 83.0 % (55) | 79.9 % (102) | 81.2 % (191) |
| KSJC | 89.4 % (17) | 90.6 % (22) | 90.5 % (35) | 90.3 % (74) |
| KSMF | 78.4 % (36) | 75.6 % (50) | 78.4 % (78) | 77.5 % (164) |
| KSTL | 84.1 % (25) | 87.2 % (29) | 81.8 % (66) | 83.9 % (120) |
| All | 84.3 % (127) | 84.9 % (169) | 83.1 % (310) | 83.9 % (606) |

Source: `4dTrajectory/outputs/POOLED/multi/val_majority_formal_r7_20261012/readout.json`.

### 4.3 The other campaigns

Table 6 lists the eight campaigns. All of them start from stage C round 5 and use seed 2035. All except the first launch read the same 4,492 select aircraft. The best select landed share of a campaign other than the formal campaign is 83.3 % (arm A, round 1). The formal campaign reaches 83.7 % in round 7. The user stopped two campaigns early: `multi_train_20261009` in round 0 and `multi_train_20261010` in round 5.

**Table 6.** The campaigns of stage D. The best select result is the round with the highest reward sum (4,492 commanded aircraft in every campaign).

| Campaign | Question | Rounds | Best select result | Stopped early? |
|---|---|---|---|---|
| multi_train_20261009 | First campaign, first launch. Can the model land all commanded aircraft of a window without a loss of separation? | 8 planned; 0 finished | none | Yes: the user stopped it in round 0 because one round took hours |
| multi_train_20261010 | First campaign, second launch. Same question on faster code, with branch points found backward from each event. | 8 planned; 5 finished | round 4: 82.8 % landed, 662 lost separation, reward 3694.3 | Yes: the user stopped it at 2026-10-10 16:31 in round 5 (92 of 217 batches) |
| multi_train_lr3_20261010 | Is the small step the reason that the landed share stops at 82.8 %? Learning rates 3 times larger. | 2 planned; 2 finished | round 0: 82.0 % landed, 702 lost separation, reward 3659.9 | No |
| multi_train_traffic_only_20261010 | Does the traffic module alone give the gain? Prior learning rate 0. | 2 planned; 2 finished | round 1: 82.5 % landed, 682 lost separation, reward 3676.9 | No |
| multi_train_majority_20261010 | Arm A. Do majority branch points give a better landed share and less loss of separation than backward points? | 2 planned; 2 finished | round 1: 83.3 % landed, 659 lost separation, reward 3715.8 | No |
| multi_train_majority_data025_20261011 | Arm B. Does a data weight of 0.25 on top of arm A change the result? Does the speech of the model stay normal? | 2 planned; 2 finished | round 0: 82.8 % landed, 691 lost separation, reward 3690.8 | No |
| multi_train_majority_formal_20261011 | The formal campaign: the settings of arm A for 8 rounds. How far do majority branch points bring the landed share? | 8 planned; 12 finished | round 11: 83.7 % landed, 593 lost separation, reward 3734.5 | No; it paused twice (a memory stop in round 5; the val read of round 7), and the user extended it to 12 rounds |
| multi_train_minority_side_20261011 | Control arm. Does a group weight by the minority side lift the plateau? From round 5 the learning rates are 3 times larger. | 3 planned, then 5, then 8; 8 finished | round 5: 83.1 % landed, 631 lost separation, reward 3707.8 | No |

Note: The start model lands 3,660 of 4,492 aircraft (81.5 %) on the same select windows. Its reward sum is 3633.7.

Source: `4dTrajectory/outputs/POOLED/multi/<campaign>/round_*/round.json`; the questions: `4dTrajectory/ts_transformer/docs/experiments/intents.json` and `docs/two_tier/readouts/2026-10-06_stage_d_implementation_log.md` section 22.

The arms of two rounds do not rank the methods. The second launch of the first campaign gives the noise: five rounds of one recipe landed between 82.1 % and 82.9 %. Arm A (83.3 %) and arm B (82.8 % and 82.3 %) lie close to this range. The learning rate arm falls to 80.4 % in round 1. This is below the start (81.5 %).

The control arm with the minority-side group weight reached 83.1 % in rounds 4 and 5. Its learning rates were 3 times larger in rounds 5 to 7. The landed share in these rounds was 83.1 %, 82.5 % and 83.0 %, and the KL rose from 0.074 to 0.100. The larger rates did not lift the plateau.

### 4.4 Round 7 on the val days

Table 7 shows the outcomes. Round 7 lands 3,803 of 4,534 val aircraft (83.9 %) in 1,500 windows. In 606 cases (13.4 %) the aircraft loses separation. The other 125 aircraft (2.8 %) fail in other ways. Of these, 43 are `unstable_at_minimums`, 37 are `ground_contact`, 31 are `timeout`, and 14 cross the threshold at a wrong height or place. The select and val shares agree to 0.3 points (83.7 % and 83.9 %), but the two sets hold different flights.

**Table 7.** Outcomes of the commanded aircraft of round 7 on the select days and on the val days.

| Outcome | Select | Select share | Val | Val share |
|---|---|---|---|---|
| landed | 3,759 | 83.7 % | 3,803 | 83.9 % |
| lost_separation | 604 | 13.4 % | 606 | 13.4 % |
| unstable_at_minimums | 36 | 0.8 % | 43 | 0.9 % |
| ground_contact | 34 | 0.8 % | 37 | 0.8 % |
| timeout | 42 | 0.9 % | 31 | 0.7 % |
| crossed_too_high | 5 | 0.1 % | 6 | 0.1 % |
| crossed_other_runway | 3 | 0.1 % | 4 | 0.1 % |
| crossed_off_runway | 4 | 0.1 % | 4 | 0.1 % |
| dynamics_failure | 5 | 0.1 % | 0 | 0.0 % |
| Total | 4,492 | 100 % | 4,534 | 100 % |

Source: `4dTrajectory/outputs/POOLED/multi/multi_train_majority_formal_20261011/round_7/round.json`; `val_majority_formal_r7_20261012/readout.json`.

Table 8 gives the kinds of pair in the loss windows. On the val days, 190 windows have a loss between two commanded aircraft. In 176 windows a commanded aircraft answers for a loss with a recorded aircraft by the rules. In 82 windows a commanded aircraft answers because the records kept their separation. Only 17 windows have a loss that nobody answers for.

The number of windows with a loss between two commanded aircraft grows with the span. It is 30, 55 and 105 at 300 s, 600 s and 1,200 s. The number of windows with a loss between a commanded and a recorded aircraft grows less: 75, 84 and 99. A window has 1.9, 2.7 and 4.5 commanded aircraft on average at these spans. The select split shows the same pattern.

**Table 8.** Round 7: windows with at least one loss step, by kind of pair. One window can appear in more than one column.

| Split | Span (s) | Windows | Aircraft lost separation | Two commanded | Commanded answers, rules | Commanded answers, records kept | Recorded only |
|---|---|---|---|---|---|---|---|
| Val | 300 | 500 | 127 | 30 | 54 | 21 | 3 |
| Val | 600 | 500 | 169 | 55 | 59 | 25 | 6 |
| Val | 1,200 | 500 | 310 | 105 | 63 | 36 | 8 |
| Val | All | 1,500 | 606 | 190 | 176 | 82 | 17 |
| Select | 300 | 500 | 118 | 29 | 61 | 18 | 4 |
| Select | 600 | 500 | 189 | 62 | 67 | 30 | 8 |
| Select | 1,200 | 500 | 297 | 102 | 76 | 44 | 10 |
| Select | All | 1,500 | 604 | 193 | 204 | 92 | 22 |

Note: Two commanded: both aircraft are commanded. Commanded answers, rules: one commanded and one recorded aircraft, and the rules make the commanded aircraft responsible. Commanded answers, records kept: only the recorded aircraft is responsible by the rules, but the two records keep their separation, so the commanded aircraft answers. Recorded only: the records also lose separation, so nobody answers.

Source: `4dTrajectory/outputs/POOLED/multi/val_majority_formal_r7_20261012/readout.json` (loss_windows, loss_steps); `round_7/round.json` of the formal campaign.

### 4.5 Single-aircraft closed loop on the val days

Table 9 repeats table 1 of the paper materials. Stage D round 7 lands 96.9 % in the single-aircraft closed loop. This is 1.0 point less than stage C round 5 (97.9 %). The mean arc ADE does not change (1,248 m and 1,242 m). The mean arc FDE falls from 708 m to 674 m.

**Table 9.** Single-aircraft closed loop on the val days: one commanded aircraft, no other commanded aircraft. Distances in m, mean / median.

| Model | Trajectories | Landing rate | Arc ADE | Arc FDE |
|---|---|---|---|---|
| Labelled words flown by the executor (reference) | 947 | 100.0 % | 83 / 69 | 132 / 103 |
| Base model | 1,894 | 94.6 % | 1,444 / 315 | 775 / 192 |
| Stage C round 5 (the start of stage D) | 1,894 | 97.9 % | 1,242 / 298 | 708 / 190 |
| Stage D round 7 | 1,894 | 96.9 % | 1,248 / 295 | 674 / 195 |

Note: 200 val flights for each airport, 2 samples for each flight, seed 1337, the same flights for the three models. Only flights inside the prior's selection count (947 flights). Arc ADE is the mean distance between the generated and the observed track at 64 equal fractions of the horizontal arc length. Arc FDE is the distance at the last fraction.

Source: `docs/paper_materials/results/measured/table1_val_free_generation.json` (key inside); `docs/paper_materials/results/two_tier_results.md` table 1; stage D round 7: `4dTrajectory/outputs/POOLED/multi/single_val_majority_formal_r7_20261012/`.

Table 10 shows that the decrease is in the vectored approaches. They land 93.6 % against 96.6 % for stage C round 5. The difference is 3.0 points on 652 trajectories, and one binomial standard error of one rate is 0.9 points. The straight-in approaches land 98.7 % against 98.6 %. Stage D round 7 does not keep the single-aircraft landing rate of stage C in full.

**Table 10.** Single-aircraft val by approach type: landing rate and arc ADE mean (m).

| Approach type | Trajectories | Base model | Stage C round 5 | Stage D round 7 |
|---|---|---|---|---|
| straight-in | 1,242 | 97.6 % / 417 | 98.6 % / 370 | 98.7 % / 363 |
| vectored | 652 | 89.0 % / 3,402 | 96.6 % / 2,904 | 93.6 % / 2,934 |

Source: `docs/paper_materials/results/measured/table1_val_free_generation.json` (key strata).

### 4.6 The val flights graded by the evaluation package

The runner `multi_val_evaluation` flew the 1,500 val windows of round 7 again. It stops unless the summed readout equals the val read, and the check passed. It wrote one evaluation record for each commanded aircraft. The 4,534 aircraft are 2,005 different flights, and a flight in two windows counts twice. The runner graded the observed tracks of the same flights with the same code. Table 11 compares the two judges.

**Table 11.** The two judges side by side.

| Item | Two-tier judge: outcome `landed` | Evaluation package: verdict `pass` |
|---|---|---|
| Where it looks | The crossing of the threshold plane, lined up with the runway, and the decision altitude before it | The state at the threshold plane (LTP plus TCH) |
| Lateral bound | 106.7 m at the threshold (half-width of the FAS course); at the decision altitude inside the FAS cone | Half-width of the runway at the threshold: 22.86 m for 4,529 aircraft and 15.24 m for 5 aircraft |
| Vertical bound | At most 100 m high at the crossing; at the decision altitude within 22 m of the glidepath | Within 22 m of LTP plus TCH at the threshold |
| Speed bound | None | From the published approach speed at the crossing mass to that speed plus 10.3 m/s (20 kt in the criterion name) |
| Loss of separation | Overrides a later landing in stage D (outcome `lost_separation`) | Not judged; the package grades one track |
| Code | ts_transformer/autopilot/judge.py and ends.py | evaluation package, schema terminal-approach-evaluation-v9 |

Source: `4dTrajectory/ts_transformer/autopilot/judge.py` (module text); the bounds in `4dTrajectory/outputs/POOLED/multi/val_evaluation_majority_formal_r7_flown_mass_20261012/records/<ICAO>/evaluation_report.json` (field bounds).

Table 12 gives the evaluation result. 2,894 of 4,534 aircraft pass (63.8 %). The observed tracks of the same flights pass in 4,088 cases (90.2 %). Of the 4,088 aircraft with an observed track that passes, 2,608 (63.8 %) pass when the model flies them.

The pass share is 59.1 % at KSMF and 68.3 % at KSJC. It is 63.2 %, 63.8 % and 64.1 % at the spans 300 s, 600 s and 1,200 s. The span does not change it.

**Table 12.** Round 7 on the val days graded by the evaluation package: flown tracks and the observed tracks of the same flights.

| Group | Aircraft | Two-tier landed | Evaluation pass | Landed and pass | Observed pass |
|---|---|---|---|---|---|
| All | 4,534 | 3,803 | 2,894 (63.8 %) | 2,858 (75.2 %) | 4,088 (90.2 %) |
| KMSY | 744 | 653 | 492 (66.1 %) | 488 (74.7 %) | 631 (84.8 %) |
| KRDU | 1,135 | 922 | 710 (62.6 %) | 701 (76.0 %) | 999 (88.0 %) |
| KSJC | 884 | 798 | 604 (68.3 %) | 594 (74.4 %) | 796 (90.0 %) |
| KSMF | 877 | 680 | 518 (59.1 %) | 512 (75.3 %) | 868 (99.0 %) |
| KSTL | 894 | 750 | 570 (63.8 %) | 563 (75.1 %) | 794 (88.8 %) |
| All airports, span 300 s | 930 | 784 | 588 (63.2 %) | 583 (74.4 %) | 848 (91.2 %) |
| All airports, span 600 s | 1,374 | 1,166 | 877 (63.8 %) | 871 (74.7 %) | 1,236 (90.0 %) |
| All airports, span 1,200 s | 2,230 | 1,853 | 1,429 (64.1 %) | 1,404 (75.8 %) | 2,004 (89.9 %) |

Note: Every aircraft has a verdict `pass` or `fail`. The observed tracks also have 221 verdicts `indeterminate`.

Source: `4dTrajectory/outputs/POOLED/multi/val_evaluation_majority_formal_r7_flown_mass_20261012/readout.json`.

Table 13 gives the outcome and the verdict. Of the 3,803 aircraft that the two-tier judge lands, 2,858 (75.2 %) pass and 945 fail. Of the 606 aircraft that lose separation, 24 pass the evaluation package. The package does not judge separation, so these 24 aircraft are still failures of the two-tier judge. Of the 125 aircraft with another failure, 12 pass.

**Table 13.** Round 7 on the val days: two-tier outcome and evaluation verdict.

| Two-tier outcome | Evaluation pass | Evaluation fail | Aircraft |
|---|---|---|---|
| landed | 2,858 | 945 | 3,803 |
| lost_separation | 24 | 582 | 606 |
| unstable_at_minimums | 9 | 34 | 43 |
| ground_contact | 3 | 34 | 37 |
| timeout | 0 | 31 | 31 |
| crossed_too_high | 0 | 6 | 6 |
| crossed_other_runway | 0 | 4 | 4 |
| crossed_off_runway | 0 | 4 | 4 |
| Total | 2,894 | 1,640 | 4,534 |

Source: `4dTrajectory/outputs/POOLED/multi/val_evaluation_majority_formal_r7_flown_mass_20261012/readout.json` (verdict_by_outcome).

Table 14 gives the gates of the landed aircraft. 899 of the 945 landed failures fail the lateral gate, and 878 of them fail the lateral gate alone. 59 fail the speed gate and 8 fail the vertical gate.

**Table 14.** Gates of the 3,803 two-tier landed aircraft (val, round 7).

| Result | Aircraft | Share of landed |
|---|---|---|
| Evaluation pass | 2,858 | 75.2 % |
| Fail: lateral gate only | 878 | 23.1 % |
| Fail: speed gate only | 40 | 1.1 % |
| Fail: vertical gate only | 6 | 0.2 % |
| Fail: lateral and speed gates | 19 | 0.5 % |
| Fail: lateral and vertical gates | 2 | 0.1 % |
| Fail: total | 945 | 24.8 % |
| Landed aircraft | 3,803 | 100 % |
| Gate failures: lateral / vertical / speed | 899 / 8 / 59 | - |

Source: `4dTrajectory/outputs/POOLED/multi/val_evaluation_majority_formal_r7_flown_mass_20261012/aircraft.jsonl` and `readout.json` (landed_gates).

Table 15 shows the cross-track distance at the threshold plane. The landed aircraft cross at a median of 14.0 m from the centreline. The observed tracks of the same flights cross at 1.9 m. The 95th percentile of the model is 35.9 m and the maximum is 68.6 m. 867 of the 899 lateral failures lie between one and two half-widths of the runway (22.86 m to 45.7 m).

All landed aircraft are inside the 106.7 m bound of the two-tier judge. The evaluation package rejects 23.6 % of them (899 of 3,803), because its bound is the runway half-width. All 3,803 observed tracks of the same flights are inside this bound.

**Table 15.** Absolute cross-track distance at the threshold plane (m): median, 75th, 90th and 95th percentile, maximum, and the share inside the lateral bound of the evaluation package.

| Tracks | Count | Median | 75th | 90th | 95th | Maximum | Lateral gate pass |
|---|---|---|---|---|---|---|---|
| Model, round 7, val (landed aircraft) | 3,803 | 14.0 | 22.3 | 30.8 | 35.9 | 68.6 | 2,904 (76.4 %) |
| Observed tracks of the same flights | 3,803 | 1.9 | 3.2 | 4.9 | 6.6 | 15.4 | 3,803 (100.0 %) |
| Executor alone, labelled words, select days, all flights with a crossing | 6,151 | 12.6 | 19.0 | 24.1 | 26.7 | 281.0 | 5,328 of 6,193 (86.0 %) |
| Observed tracks of those flights | 6,151 | 1.9 | 3.4 | 5.4 | 7.1 | 20.7 | - |

Note: Rows 3 and 4 come from a stored executor replay of 2026-10-05 on other flights (select days, row interval 4 s). They do not replace the measurement on the val flights.

Source: Rows 1-2: `4dTrajectory/outputs/POOLED/multi/val_evaluation_majority_formal_r7_flown_mass_20261012/records/<ICAO>/evaluation_report.json` and `observed/<ICAO>/evaluation_report.json` (field deviation.cross_track_m). Rows 3-4: `4dTrajectory/outputs/POOLED/executor/v17_20261005/replay-closed-select-4s/` (records and observed, same field).

## 5 Discussion

### 5.1 What the numbers show

Stage D lifts the select landed share by 2.2 points from the start to round 7. It cuts the loss of separation by 133 aircraft (737 to 604). The gain is largest at KSTL and KRDU. The val landed share of round 7 (83.9 %) agrees with the select share (83.7 %).

The gain stops after round 5. The rounds 5 to 11 lie within 0.8 points. The control arm with 3 times larger learning rates did not lift the plateau. The gradient read of the second launch of the first campaign shows no common direction of the reward gradient of the prior between updates. The pairwise cosine is -0.001, and random directions give 0. The data term sets the direction of the update.

Loss of separation is the largest failure class. It has 606 of the 731 val failures. Its share is stable over the spans (13.7 %, 12.3 % and 13.9 %). The pairs between two commanded aircraft grow with the span. KSMF and KRDU have the highest shares.

The lateral alignment is the largest gap to the evaluation package. The error is small in metres but systematic: the median is 14.0 m and the bound is 22.86 m. The observed tracks of the same flights have a median of 1.9 m.

The span does not change the pass share, and this agrees with a cause that does not depend on traffic. The stored executor replay on the select days gives a median of 12.6 m. This is close to the 14.0 m of the model. It suggests that the executor causes much of the error. This is a hypothesis, not a result, because the replay uses other flights.

Stage D has two costs. The other failures rise from 95 to 129 aircraft on the select days. The single-aircraft landing rate of the vectored approaches falls from 96.6 % to 93.6 %.

### 5.2 What the numbers do not show

The campaigns read the select split after each of 33 rounds in seven campaigns. Its numbers read high for the chosen round. The val read is the number to quote, and the user read it once.

Nobody read the start model or the base model on the val windows. The run `multi_val_models_20261012.run` was to do this. The user stopped it on 2026-10-12 at 11:50 (local time) to run it later in parallel. It wrote no result. The select and val sets hold different flights, and the airport shares differ by up to 7.4 points. The select gain of 2.2 points does not transfer to the val days.

The seed noise is large. Five rounds of one recipe landed between 82.1 % and 82.9 %. The binomial standard error of one read is 0.56 points, and the true error is larger because the aircraft of one window are not independent. The formal campaign used one seed (2035). A difference below 1 point is not evidence.

The two judges measure different things (Table 11). Do not subtract the pass share (63.8 %) from the landed share (83.9 %) as a loss of quality. The comparison with the observed tracks is also not equal. 221 observed verdicts are `indeterminate` and count as no pass. Of the 4,313 observed verdicts with a result, 94.8 % pass.

The cause of the lateral error is not known. The model or the executor can cause it. The test days stay sealed, so no number in this report is a test result.

## 6 Data locations and protection

The outputs of stage D are under `4dTrajectory/outputs/POOLED/multi/`. Two other inputs are the executor replay `4dTrajectory/outputs/POOLED/executor/v17_20261005/replay-closed-select-4s` and the table file `docs/paper_materials/results/measured/table1_val_free_generation.json`. This report reads all of them and writes none of them. Table 16 lists the directories that this report uses. Five campaign directories and four read directories have a `SHA256SUMS` file and are read-only. The check of the sums of the four read directories found no difference. This session did not check the sums of the campaign directories.

The directories `multi_train_20261010`, `multi_train_lr3_20261010` and `multi_train_20261009` have no `SHA256SUMS` and are not read-only. The directory `handover_20261010/start_read` is also not sealed. It is the only source of the start numbers in Tables 2, 3, 4 and 6. The directory `multi_val_models_20261012.run` holds scripts and a log, and no result.

**Table 16.** Data locations, SHA256SUMS and write protection (checked on 2026-10-12).

| Directory | Content | SHA256SUMS | Directory read-only |
|---|---|---|---|
| 4dTrajectory/outputs/POOLED/multi/multi_train_majority_formal_20261011 | Formal campaign: 12 rounds, checkpoints, run logs | 2,650 lines | yes |
| 4dTrajectory/outputs/POOLED/multi/multi_train_minority_side_20261011 | Control arm (group weight) | 1,764 lines | yes |
| 4dTrajectory/outputs/POOLED/multi/multi_train_majority_20261010 | Arm A | 452 lines | yes |
| 4dTrajectory/outputs/POOLED/multi/multi_train_majority_data025_20261011 | Arm B | 445 lines | yes |
| 4dTrajectory/outputs/POOLED/multi/multi_train_traffic_only_20261010 | Traffic-only arm | 445 lines | yes |
| 4dTrajectory/outputs/POOLED/multi/multi_train_lr3_20261010 | Learning rate arm | no | no |
| 4dTrajectory/outputs/POOLED/multi/multi_train_20261010 | First campaign, second launch | no | no |
| 4dTrajectory/outputs/POOLED/multi/multi_train_20261009 | First campaign, first launch (no round finished) | no | no |
| 4dTrajectory/outputs/POOLED/multi/val_majority_formal_r7_20261012 | Val read of round 7 | 4 lines | yes |
| 4dTrajectory/outputs/POOLED/multi/single_val_majority_formal_r7_20261012 | Single-aircraft val read of round 7 | 8 lines | yes |
| 4dTrajectory/outputs/POOLED/multi/val_evaluation_majority_formal_r7_flown_mass_20261012 | Evaluation grading of the val flights | 6,562 lines | yes |
| 4dTrajectory/outputs/POOLED/multi/majority_formal_r7_window_read_20261012 | Round 7 read window by window (select) | 2 lines | yes |
| 4dTrajectory/outputs/POOLED/multi/handover_20261010/start_read | Read of the start model on select | no | no |

Source: Directory listing and file modes on disk.

## 7 Recommendations for further optimisation

The recommendations follow the order of the expected gain. The mark "decision of the user" means that the user must decide. The mark **own** means that this report adds the recommendation without a request.

**R1. Measure the executor alone on the val flights.** Fly the observed (labelled) sentences of the 2,005 val flights with the executor. Grade the tracks with the evaluation package. Compare the cross-track at the threshold with Table 15. This separates the executor's own cross-track error from the model's choice of words.

Evidence: 899 of the 945 landed failures fail the lateral gate (Table 14). A stored replay of the labelled words on other flights gives a median of 12.6 m. Its lateral gate pass share is 86.0 % (Table 15). The expected gain is large. If the lateral error is zero, the 878 aircraft that fail the lateral gate alone pass. The pass share then is 83.2 % at most, against 63.8 % now.

**R2. Fix the lateral error where the measure of R1 finds the cause. This is a decision of the user.** If the executor causes the error, correct the final-approach lateral law of the executor (stage A). If the model causes the error, put a lateral term at the threshold into the reward. Or make the bound of the two-tier judge closer to the bound of the evaluation package. The judge keeps its bound of 106.7 m until the user decides.

**R3. Analyse the causes of the loss of separation before the next change of the method.** Loss of separation is the largest failure class: 606 of the 731 val failures. Analyse the loss steps by pair geometry (runway pair, wake category, distance at the loss) and by span. Use the pair kinds of Table 8 as the first split. Start with KSMF (18.7 %) and KRDU (16.8 %). Expected gain: at most 13.4 points of the landed share, and a part of it only.

**R4. Change the method or the data. More rounds of the same method are not enough.** The select landed share stays between 83.0 % and 83.7 % after round 5. Four variants did not lift it: traffic only, larger rates, a data weight of 0.25 and a group weight. The reward gradient of the prior has no common direction between updates. **Own:** The credit of a group runs over a median of 33 rows. The first rows after the branch point do not show a larger effect. A credit that is closer to the decisive word, or more groups for each update, can lower the noise of the reward term.

**R5. Read the start model (round 0) and the base model on the same val windows.** Use the paused run `multi_val_models_20261012.run`. This gives the gain of stage D on the val days. The val read of round 7 took 28 minutes in one process. The reads of the start model and of the base model need no new update passes.

**R6. Decide the final test. This is a decision of the user.** Two options exist. The first option is the 14 sealed test days at the five airports. The second option is the held-out airport KAUS (28,879 arrivals in `trajectory_data_process/outputs/harvest-heldout/KAUS`). Each read spends a claim, so decide before more changes of the method.

**R7. Check the single-aircraft landing rate of the vectored approaches. Own.** Stage D round 7 lands 93.6 % of them, and stage C round 5 lands 96.6 % (Table 10). Read the single-aircraft val again with more samples for each flight. This shows whether the decrease is real. The data term of the update must keep this ability.

**R8. Analyse the other failures. Own.** They rise from 95 to 129 aircraft on the select days (Table 2). The val read has 125. The classes are `unstable_at_minimums` (43), `ground_contact` (37) and `timeout` (31). If the analysis finds a cause, add a cost for it to the reward.

**R9. Measure the seed noise of the formal recipe with a second seed. Own.** Five rounds of one recipe spread by 0.8 points. A second seed gives the spread between runs. Do this before the user quotes a difference below 1 point.

**R10. Software work: let several processes speak in the evaluation runner.** The runner `multi_val_evaluation` flies all windows in one process. The runner `multi_window_read` has the option `--speak-workers`. Add the same option to `multi_val_evaluation` and to `multi_validation`.

**R11. Seal the unsealed directories. Own.** Run `sha256sum` on `handover_20261010/start_read`, `multi_train_20261010` and `multi_train_lr3_20261010`, and make them read-only. The start read is the only source of the start numbers.
