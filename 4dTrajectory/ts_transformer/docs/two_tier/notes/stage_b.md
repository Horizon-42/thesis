# 阶段 B：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_b_implementation_log.md`）。
2026-10-05，Claude 写，用户转发。本条取代 B10 那一条；B10 的规格仍在 prior.md §12 B10。

```
跟进 A 阶段已合并的线，并做 B11。
依据：prior.md §0.4 计划表（"Now: stage A's line is merged" 那一行）、§12 B10、§12 B11、§0.1 D111；
design/requests_from_a_to_designer.md §2；vocabulary.md D111、§6。

1. B10（e4e7ba42）的审查、修正和测试，如果还没做完，先做完（按 §12 B10）。
2. 合进 dev-two-tier（cf6bf26a）和 dev-two-tier-v4（ed2530ae：A37–A40）。
3. 跟进 A 阶段的改动（§0.4 那一行）：
   - A37：航班为什么结束，改从判定结果读（judge.TIMEOUT）；
     导出里跑道的高度差改用 training_export.candidate_hae_minus_msl_m；
   - A38：共用闭环步骤里的复制改用 Loop.copy；被 halt 的航班不再听词；
   - A39：做 B10 里 D109 那一条，把允许的划分交给后端服务和前端读取函数。
4. B11（§12 B11）：landed 选择额外剔除 A 阶段标记了观测航迹有故障的航班。
   - 选择记录把"因故障剔除"和"因结果剔除"分开计数；
   - 检查点 schema 和 B10 的改动合用一个新名字；
   - val 的标记只记在身份里，不打印（D85）；
   - 自由生成的读数把这两种剔除原因分开报告；
   - 测试按 B11 的规格写。
5. 每一步：跑单文件测试 → 独立审查（只审代码）→ 修正 → 用显式路径提交。
   全部完成后跑全量 ts 套件（先按 outline §5 规则 13 看资源）。
6. 然后在 A34 产物上重跑一次 B3 冒烟和自由生成，包括 B10 要求的正式规模显存测量。
7. 停在 B5 之前，然后报告：
   - 提交号和测试数；
   - §7 中标 "new, B10" 的各项名字，以及 B11 若改了选择的名字也一并给出（交给 Claude 写进 §7）；
   - 设计没写到的地方你的读法，作为提议。
不改 instructions/、autopilot/，也不改 A 阶段的运行器。
```
