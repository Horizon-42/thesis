# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。上一条（D130、C10 的验证读数、说明修正）已做完（`02a85049`、`0c95a821`）。

```
Claude 第二轮复审（readouts/2026-10-06_stage_c_check.zh.md §5）之后的修正。
依据：post_training.md D132、§8 C10；复审指南 §3 第 6 步。

1. 把 dev-two-tier 合进本分支（含 D132）。
2. D132 / P48：post_train 正式启动、post_validation 正式读时，检查 --prior 是 B 阶段的正式 base，
   冒烟 prior 或某个折的模型按名字拒绝（冒烟运行不检查）。测试：传入折或冒烟 prior 时被拒绝，正式 base 能通过。
3. P46、P47 已由用户接受（D132），代码不用改。
4. 更新 post_training.md §0.3 状态表（验证读数已实现；C8 的状态）和 requests 文件（只写当前未决的请求）。
5. 每步：单文件测试 → 独立审查（只审代码）→ 用显式路径提交。报告提交号。
不改 instructions/、autopilot/、prior/，也不改 B 的共用模块。
6. 不改 outline §6.2 第 9 条列出的文件：前端 Training 视图的文件、aeroviz_backend/http_server.py，
   以及 C 的 Training 导出（experiments/post_training_export.py、post/training_files.py）。
   三个阶段 Training 视图的统一布局、结果页和包络（outline §6.2、D133–D135）由 B 的实现者在 dev-training-layout 上做，
   窗口视图和窗口导出跟着它改。
```
