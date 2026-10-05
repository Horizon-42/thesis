# 阶段 B：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_b_implementation_log.md`）。
2026-10-05，Claude 写，用户转发。

```
B10：执行 Claude 对 B 阶段复审的修正。
依据：prior.md §12 B10（规格）、§0.1 D105–D108、§7（代码列）；outline D109；
复审记录 readouts/2026-10-05_stage_b_check.zh.md §2。

1. 正在跑的 prior_campaign 冒烟跑完即停。B10 完成前不再启动任何战役。
2. 把 dev-two-tier（含 4d64c1cd、9e44a687）合进 dev-two-tier-v4-prior。
3. 按 B10 逐条实现，在合成产物上测试。
   D109 那一条等 vocabulary A39 合进本分支后再做。
4. 复审记录 §2 第 12 条列的测试，可从审查员的脚本改写（先复制出来，那是临时目录）：
   /tmp/claude-1000/-home-supercomputing-studys-thesis/c7f0f026-4de2-41cc-8e66-0c768e1bb49c/scratchpad/review_b/
   r1_inputs/{test_d90_loop.py, check_select.py, test_loop_rows_8s.py}
   r2_training/test_smoke_reads_val.py
   r3_speaker/{test_leaks.py, test_paths.py}
5. 删掉实现日志 §1 里两处 val 计数（"val 9,540 (210)"、"val 9,545 (205)"），依据 D85。
6. 每一步：跑单文件测试 → 独立审查（只审代码）→ 修正 → 用显式路径提交。
   全部完成后跑全量 ts 套件（先按 outline §5 规则 13 看资源）。
7. 在 A34 产物上重跑 B3 冒烟和自由生成，包括 B10 要求的正式规模显存测量。
8. 停在 B5 之前，然后报告：
   - 提交号和测试数；
   - §7 中标 "new, B10" 的各项名字（交给 Claude 写进 §7）；
   - 设计没写到的地方你的读法，作为提议。
不改 instructions/、autopilot/，也不改 A 阶段的运行器。
```
