# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。上一条（跟进 B 的这一轮、B12 之后的第 6 步）已做完（实现日志 §17、§18，`963ee427`）。

```
一处小修正（B12 复审里看到的，readouts/2026-10-06_stage_b_b12_check.zh.md 末尾"给 C 的"）。
B5 在跑（主树 dev-two-tier）：不用 GPU；主树上只改你的实现日志，改完立刻用显式路径提交（树一脏 B5 就停）。

1. experiments/post_window_loop.py：_traffic 在 speaking.step 之前就往 self._tokens 追加这一行，
   _speed_masks 在之前就累加 self.speed_mask_rows。执行器拒绝这一行时（vocabulary D80），
   共用闭环步和说话器都会回到原样（prior §7 第 3、7 项），但你的这两份记录多了一行。
   改成这一行被留下之后才记（先算在局部，speaking.step 返回后再追加/累加）。
2. 测试：执行器拒绝一行后，_tokens 和 speed_mask_rows 与没说这一行时相同。
3. 单文件测试 → 独立审查（只审代码）→ 用显式路径提交。报告提交号。
4. 之后 B13 提交到 dev-two-tier-v4-prior 时（B 会报告），合进本分支；B13 若改了 prior §7 的名字，按 §7 跟进。
不改 instructions/、autopilot/、prior/，也不改 B 的共用模块。
```
