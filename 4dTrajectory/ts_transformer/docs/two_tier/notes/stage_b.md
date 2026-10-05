# 阶段 B：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_b_implementation_log.md`）。
2026-10-05，Claude 写，用户转发。上一条（B10、跟进 A37–A41、B11）已做完（`89845fa6`）。

```
收尾 requests_from_b_to_designer.md 的这一轮；B5 先不开始，等用户另行下令。
依据：prior.md §0.1 D118、§7（本轮的名字已写入）、§8 第 1 项；notes/stage_a.md（A 线上的一处小改动）。

1. 把 dev-two-tier（249aacfb 或更新）合进 dev-two-tier-v4-prior。
2. 你的 8 条读法用户全部接受，已写成 D118；§7 的名字、§8 第 1 项的信号文件 sha256 都已写入设计。
   核对 §7"代码"列和代码里的名字一致；不一致的写进 requests 文件，不要自己改设计。
3. A 的改动（support.py 的 runway_ends_from，dev-two-tier-v4-a42）合进 dev-two-tier-v4 后，把它合进本分支，
   删掉 tests/test_prior_training_export.py 里的替身 with_runway_ends。
   那条陈旧的注释是你自己的钩子行（backend.py 第 107 行附近 self.prior = PriorSegments(self)），只在你的分支上，
   A 的线上没有这一行，A 改不了：你自己改，不再写 index_prior_v1.json，改为引用 prior.training_files.INDEX_FILE，
   不写文件版本号。
   测试照旧要过：单文件测试 → 独立审查 → 用显式路径提交。
4. 把 requests_from_b_to_designer.md 重写为当前状态（已处理的项删掉）。
5. 停住，等 B5 的命令。报告：提交号、测试数。
不改 instructions/、autopilot/，也不改 A 阶段的运行器。
```
