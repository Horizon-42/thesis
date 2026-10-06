# 阶段 B：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_b_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。上一条（统一布局第二部分、模型速度、A44 / C13 / B14）已做完，复审见下。

```
复审完（readouts/2026-10-06_training_view_and_speed_check.zh.md）：没有 S1。
你的 9 条读法用户逐条看过，全部接受（outline D139），第 9 条要顺手修正高估；都已写进设计。
依据：outline D131（复审标准）、D139；prior §12 B14；post_training §8 C13。
仍在 dev-two-tier-v4 上做（工作树 .claude/worktrees/two-tier-v4），开工前先和 dev-two-tier 对齐。

现在就做（不用等 C10）：
1. 复审第 2 条（S2）：给 experiments/training_export.py「包络超出飞行航迹就拒绝」加 Python 测试：
   替换 envelopes，让它返回 endRow = rows + 1，断言 flown_sentence 拒绝。
2. 复审第 3 条（S2）：tests/test_prior_training_export.py 里，对 prior 目录的字节做导出前后比较。
3. 读法 9 的修正（O15）：
   - worker 的峰值扣掉和父进程共享的页（用 SwapPss，或减去 worker 刚启动时的读数）；
   - 把 worker 留着的读取模型和一轮里保留的 series 算进去。
4. 每步：单文件测试 → 独立审查（只审代码）→ 显式路径提交 → 日志一行。
5. requests 文件重写：§1 的 9 条和 §3 的 3 条已进设计（D139、prior §7、vocabulary §6），删掉；
   各文档 §0.3 的状态行写上 A44、C13、B14 已建成并审查。
复审第 4–11 条是 S3，不改。

C10 结束后（主机和 GPU 上没有别的任务时），按这个顺序：
1. 先跑 B14 的真实数据检查：用 BATCH 把 B5 的 KRDU 折在 select 日重说一遍（写在 scratch 目录），
   句子必须和它的读数逐词相同；不同就停下来报告。在这一步之前，不要用 BATCH 跑任何真实数据。
2. C13 在 GPU 上的检查：跑一轮，单进程和 N 个 worker 的结果必须相同。
3. 对 base 测速（model_speed），写进 4dTrajectory/outputs/POOLED/speed/ 下的新目录。
4. 跑 sha256sum -c 校验 base 的 SHA256SUMS；重导 A 和 B 的集合（B 的带 source.speed）；
   再跑一遍 sha256sum -c，结果必须一致。
5. 浏览器检查交给一次性子代理。
6. 报告 dev-two-tier-v4 能否快进，由用户合并。

C10 在 two-tier-v4-post 工作树里跑：不合并进 dev-two-tier，不碰那个工作树。
```
