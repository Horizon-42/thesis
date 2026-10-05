# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-05，Claude 写，用户转发。上一条（跟进 D105–D107、D110）已做完（`3725565c`、`0168d457`）。

```
普查：回放的飞机里带故障观测航迹的有多少（vocabulary D111）。
依据：post_training.md §8 C1 最后一条普查、§0.4 计划表对应的那一行；vocabulary.md D111、§6 第 3 项
（instructions/faults.py 的 faulty_flights / track_faults）。

1. 等 B 阶段把 A 阶段合并后的线（ed2530ae，含 A40）合进 dev-two-tier-v4-prior，
   再把 dev-two-tier-v4-prior 和 dev-two-tier（e258e7b7）合进本分支。
2. 在 post_windows 的普查里加上 C1 这一条。按机场和划分（train、select）分别数：
   - 有被标记的回放飞机在空中的窗口数；
   - 回放飞机读到故障点的场景步数（该行本身是故障点，或它的 2 s 运动读到故障点），以及这些步上的 token 数；
   - 所有飞机都按记录飞时，间隔丧失的总数；其中，在事件步或事件前 2 Δ 内有一方读到故障点的有多少。
3. 在 A34 产物的 train 和 select 上，在 scratch 目录里跑（outline §5 规则 7、12）；不读 val，不写 outputs/。
4. 改代码的步骤：单文件测试 → 独立审查（只审代码）→ 修正 → 用显式路径提交。
5. 报告：提交号和上面的计数表。不设判据，由用户决定要不要为这些步或窗口定规则。
不改 instructions/、autopilot/、prior/，也不改 B 的共用模块。
```
