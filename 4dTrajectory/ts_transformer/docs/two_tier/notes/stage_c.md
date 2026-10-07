# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-07，Claude 写，用户转发。

```
你的 requests 1–5 都处理了：
- 1（D167 的读法）写进了 D167；
- 2（settings_type）写进了 §9 item 11；
- 3（D 合并 C18 时要保留 CPU worker 的检查）转给了 D；
- 4（P55）写成 post_training D170；
- 5（P54）写成 D171，细则在 §2 item 10。
用户补定（2026-10-07）：V 除了模型看到的东西，只多读两样——录制飞机的未来（+30、60、120 s 时相对被指挥机此刻状态的
边特征、那时是否还在空中、离它花名册落地还有多久），和离 judge 时限还剩多少秒；不读被指挥机自己的录制航迹。
这是 outline 原则 7 / vocabulary D90 唯一的例外：V 不说话，任何读数都不读 V，模型的输入里不能有 V 的特征。
路径相对 4dTrajectory/ts_transformer/。设计文档不记实验信息（目录、轮次、数字写在 intents.json 和你的日志）。

现在 post_k16lr3_20261007 在 run-post-k16lr3 运行工作树里跑：只写代码，只跑改动文件的测试，少开进程（rule 13），
不动那个运行工作树。

一、代码（在 dev-two-tier-v4-post 上，先把 dev-two-tier 合进来；每项单独提交）
1. C21（D170）：Settings.branch_every_s（默认 120，必须是 Δ 的整数倍）和 Settings.segment_only（默认 False）。
   - branch_points 加参数 every_s，默认 BRANCH_EVERY_S；stage_c_rules 读设置里的间隔。
   - post.branches.samples 加参数 segment_rows，默认 None，只有阶段 C 的 pass 传它。
   - 阶段 D 的 multi/credit.py 调 branch_points 的那一行、D 的 pass 都不改；D 的测试原样通过。
   - 两个设置不是默认值时，在 landed 和 value 方法下按名字拒绝。
   - 测试按 §8 C21。D 的相关测试（multi/ 下用到 branch_points、samples 的）也算改动文件的测试，要跑。
2. C22（D171，§2 item 10）：新方法 Settings.method = value。要点：
   - 每个窗口只说一遍（first_numbers），没有第二遍、没有分支组；每句都是样本，落地的也是；
     计数的行从第一预测步到事件。
   - V 的结构：campaign 起点模型（先验 + 交通注意力）的一份拷贝，加 V 自己的 token part（投影从零开始，§9 item 7），
     再加一个小头：读 Prior.encode 最后一层每行的输出和剩余时间，输出一个数。
     拷贝之后 V 和模型不再共享权重。V 读输入时关 dropout。
     V 有自己的 AdamW（Settings.value_lr），用 campaign 的 weight decay 和 clip_norm。
   - 优势：每轮说完以后，用本轮开始时的 V 读一遍所有样本，算 GAE（γ = 1，λ = 0.95，奖励在事件那一行，事件之后 V = 0），
     得到每行优势 A_t 和目标 R_t = A_t + v_t；本轮的 E 遍 pass 都用这一套，不做归一化。
     A_t 填进 Samples.advantage，现有的 loss 不用改。
   - 每次更新：模型的三项（§2 item 5，分块）走一步；V 的损失 = 计数行上 (V − R_t)² 的平均，V 的优化器走一步。
   - 热身：前 Settings.value_warmup 轮只训 V，模型和它的优化器不动；这几轮的 selection 读数照常读，
     应该和起点那一轮的读数一样（像 D161 的 draw 0），把这个比较记进 round.json。
   - 文件：round_r/value.pt（ts-post-value-v1：V 的权重、优化器、形状、本轮 identity），写在 checkpoint.pt 之前；
     checkpoint 格式和 identity 一点不改；续跑从最后一轮的 value.pt 读 V；任何起点（D162）都不读 V；
     任何读数都不打开 value.pt。
   - 设置：value_lr、value_warmup 在 value 方法下必填，其他方法下按名字拒绝。
   - 还要做：value 方法自己的 pass 内存测量（O15、D167 要用）。
   - 测试按 §8 C22，包括：改 V 的特征时模型的 log 概率不变；去掉 value.pt 后 selection 读数不变。
   - 设计没说的地方（每次更新多少个样本、value 方法下 continuations 怎么记等）写成读法，放进 requests，
     按读法先做，报告里说明。
   C21 做完再做 C22。
每项：改动文件的测试和 test_architecture → 独立审查（只审代码，审查者不能是作者，审查和测试同时开始）
→ 显式路径提交（提交前看 git diff --cached --stat）→ 日志写一行 → 报告 dev-two-tier-v4-post 能否快进
dev-two-tier，由用户合并。

二、实验
post_k16lr3 跑完、C21 合并以后，按用户已经定的跑 P55 的第一个 campaign（你 requests 第 4 条里的设置）。
C22 合并以后，你提出价值训练第一个 campaign 的设置（value_lr、epochs、每轮窗口数、起点），由用户定。
每个 campaign 都从用户合并后的提交建运行工作树（outline D163），主机和 GPU 上没有别的任务时跑，用 systemd 单元；
intent 在发布前写进 intents.json；跑完数据设为只读并写 SHA256SUMS，结果写进实验日志。

你只能写这些设计文本：post_training §0.3 的状态行和你的日志。requests_from_c_to_designer.md 整份重写：
1–5 条已处理，删掉；新的读法另起。日志和 requests 文件提交到 dev-two-tier：用一个临时工作树加快进，只放这些文件。
```
