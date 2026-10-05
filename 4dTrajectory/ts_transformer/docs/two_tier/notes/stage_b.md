# 阶段 B：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_b_implementation_log.md`）。
2026-10-05，Claude 写，用户转发。上一条的第 1–5 步已做完（`5818fbbf`、`b24c11be`）；它的 A43 部分并入下面第 1 步。

```
B5 在跑（主树 dev-two-tier，prior_base_20261005）。本条全部在 dev-two-tier-v4-prior 上做，B5 结束前不合进 dev-two-tier。
依据：prior.md §0.1 D119–D122，§12 B12（逐条照做），§7 第 3、7 项；复审报告 readouts/2026-10-05_stage_b_check_2.zh.md。

B5 期间的规矩：
- 不碰主树的代码，不用 GPU；只跑单文件测试和 B 的测试，全量套件等 B5 结束。
- 主树上只改你的实现日志，改完立刻用显式路径提交（B5 每步之前查树是否干净，树一脏它就停）。

1. 先做上一条剩下的 A43 部分：把 dev-two-tier-v4（60bbc901 或更新）合进本分支，然后改你自己的钩子
   aeroviz_backend/autopilot_segment/prior.py：
   - 第 204 行的注释改成 "# before the set is opened (A37)"；
   - 第 158 行预热外面那层 with self.backend._lock: 去掉（A 的预热已不拿请求锁，每个集合用自己的锁）；
   - apart_from_exported 算完距离后调 fly.refuse_past_bound(horizontal_m, vertical_m, flight, against)，
     against 写 "the readout's flown states"，测试照 A 的写法补一条。
2. B12（prior §12）：D119 的重跑和 --airports 先查；D108 的守卫（设置由行为检查的进程给出，行为检查加上
   sentence_rows、选择规则、prior_select 的规则，战役格式换新名）；说话器的两处拒绝；SpeakingLoop 接收起点
   给回的观测行（自由生成用 start_moved 的 NO_MOVE，结果逐位不变）、执行器拒绝的行不留痕迹；LoopRows 一个首步；
   prior_select 的拒绝和核对；D122 的测试；inputs.py 的说明文字；§12 列的各项测试。
3. 每步：单文件测试 → 独立审查（只审代码）→ 用显式路径提交。
4. 重写 requests_from_b_to_designer.md；报告：提交号、测试数，§7 第 3、7 项改了的名字和签名。
不改 instructions/、autopilot/，也不改 A 阶段的运行器。
```
