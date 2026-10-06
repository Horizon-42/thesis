# 阶段 B：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_b_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。上一条（B13）已做完（`142272f0`）。

```
B13 复审完：没有要修的（readouts/2026-10-06_stage_b_b13_check.zh.md）；行为检查的覆盖不再加（outline D131）。
依据：prior.md §0.3、§0.4，outline D131（复审标准）。

1. 现在什么代码都不改。B5 在跑（主树 dev-two-tier，prior_base_20261006，B12 的代码）：不碰主树代码，不用 GPU。
2. B5 结束后（campaign.json 里没有 running、最后一步 base/free_generation 有 readout.json）：
   把 dev-two-tier-v4-prior（142272f0 或更新）合进 dev-two-tier，然后按 outline §5 规则 13 看资源，跑全量套件（./run_all_tests.sh）。
3. 报告：合并的提交号、全量结果；实现日志写一行。之后等 B6 发布的命令。
不改 instructions/、autopilot/，也不改 A 阶段的代码和运行器。
```
