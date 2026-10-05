# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-05，Claude 写，用户转发。上一条（跟进 D114–D117、C6、C9）已做完（见实现日志 `17017a55` 一行）。

```
跟进 B 阶段这一轮（B10 收尾、跟进 A37–A41、B11，D118）。
依据：prior.md §0.1 D111、D118，§7（"代码"列，本轮的名字已写入），§8 第 1 项；post_training.md C0。

1. 把 dev-two-tier-v4-prior（89845fa6 或更新）和 dev-two-tier（249aacfb 或更新）合进本分支。
2. tests/test_architecture.py 的 PRIOR_INTERFACE 按 prior §7"代码"列补齐本轮的名字：
   - checkpoint：readable_identity、validation_claim、holds_claim；source：require_selection_of；
   - selection：kept(rule, outcome, faulty)、left_out、side、SIDES、REASONS、CELL；
   - §7 第 7 项：SpeakingLoop 的 copy、said、states、generated。
3. 跟进改名和签名：
   - 格式新名：检查点 ts-prior-checkpoint-v8、自由生成 v4、验证读数 v3、procedure-masks-v5；
   - 用到 kept 的地方改成新签名；数据项经 ArtefactSource 读，D111 的故障航班自动剔除；
   - 被调用方结束的航班（D93 的间隔丧失）：用 said/states 取它的句子，不要找 generated 要判定结果（D118 第 6 条）。
4. C 的测试和全量 ts 套件照旧要过（全量前按 outline §5 规则 13 看资源）。
   每步：单文件测试 → 独立审查（只审代码）→ 用显式路径提交。
5. 报告：提交号、测试结果，以及你的读法（作为提议）。
不改 instructions/、autopilot/、prior/，也不改 B 的共用模块。
```
