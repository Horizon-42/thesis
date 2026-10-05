# 阶段 A：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_a_implementation_log.md`）。
2026-10-05，Claude 写，用户转发。上一条（A41、删除被取代的产物）已做完。

```
两处小改动（B 阶段的请求，requests_from_b_to_designer.md §2；用户 2026-10-05 同意由 A 改）。
只改代码注释和测试辅助；不建产物，不重跑任何东西。

1. 把 dev-two-tier（249aacfb 或更新）合进 dev-two-tier-v4，在新分支上做。
2. aeroviz_backend/autopilot_segment/backend.py 第 107 行附近，B 的钩子（self.prior = PriorSegments(self)）的注释：
   不再写 index_prior_v1.json，改为引用 prior.training_files.INDEX_FILE，不写文件版本号。
3. tests/support.py 的 labelled_instruction_artefact：在合成的 signals.json 里写上 runway_ends_from，
   字段和真实产物的一致（training_export.candidate_hae_minus_msl_m 读的就是它）。
4. A 自己的测试和夹具照旧要通过（A23 的集合和夹具逐字节不变）。
5. 审查（只审代码）→ 用显式路径提交 → 报告提交号。之后 B 会删掉它测试里的替身。
```
