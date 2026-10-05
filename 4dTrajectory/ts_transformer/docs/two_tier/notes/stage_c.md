# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-05，Claude 写，用户转发。上一条（跟进 B 阶段这一轮，D118）还没做，原样保留在第 1–5 步；第 6 步是新加的。

```
跟进 B 阶段这一轮（B10 收尾、跟进 A37–A41、B11，D118），以及 B12 落地之后的接口变化。
依据：prior.md §0.1 D111、D118–D122，§7（"代码"列），§8 第 1 项，§12 B12；post_training.md C0。
B5 在跑（主树 dev-two-tier）：不用 GPU；主树上只改你的实现日志，改完立刻用显式路径提交（树一脏 B5 就停）。

1. 把 dev-two-tier-v4-prior（89845fa6 或更新）和 dev-two-tier 合进本分支（你已合到 44f6000f）。
2. tests/test_architecture.py 的 PRIOR_INTERFACE 按 prior §7"代码"列补齐本轮的名字：
   - checkpoint：readable_identity、validation_claim、holds_claim；source：require_selection_of；
   - selection：kept(rule, outcome, faulty)、left_out、side、SIDES、REASONS、CELL；
   - §7 第 7 项：SpeakingLoop 的 copy、said、states、generated。
3. 跟进改名和签名：
   - 格式新名：检查点 ts-prior-checkpoint-v8、自由生成 v4、验证读数 v3、procedure-masks-v5；
   - 用到 kept 的地方改成新签名；数据项经 ArtefactSource 读，D111 的故障航班自动剔除；
   - 被调用方结束的航班（D93 的间隔丧失）：用 said/states 取它的句子，不要找 generated 要判定结果（D118 第 6 条）。
4. C 的测试要过；全量 ts 套件等 B5 结束后再跑（outline §5 规则 13）。
   每步：单文件测试 → 独立审查（只审代码）→ 用显式路径提交。
5. 报告：提交号、测试结果，以及你的读法（作为提议）。
6. B12 在 dev-two-tier-v4-prior 上提交之后（B 会报告提交号）：合进本分支，然后
   - SpeakingLoop 直接接收 start_moved 给回的观测行（§7 第 7 项）：删掉 post_window_loop.moved_sentences，
     窗口循环把 observed 交给 SpeakingLoop；
   - 说话器现在拒绝"首步标志与自己的状态不符"的行（§7 第 3 项）：窗口循环里你自己构造的首步标志要对；
   - 温度（D121）：B 阶段的运行都用 1；后训练若要别的值，写成你的读法报给设计者，不要自己定。
不改 instructions/、autopilot/、prior/，也不改 B 的共用模块。
```
