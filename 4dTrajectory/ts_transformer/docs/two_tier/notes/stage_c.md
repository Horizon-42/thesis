# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-08，Claude 写，用户转发。

```
你的 requests 1–5 都处理了：
- 1（C22 的六条读法）用户全部接受，写成 post_training D174，§2 item 10 各点也改了；
- 2（Stage.companion、Stage.close_round、Stage.train 收到轮次和 companion）写进 §9 item 11；
- 3（WindowLoop 的 value_reader、WindowLoop.values）写进 §9 item 8（窗口循环是 item 8，不是 9）；
- 4 是实验设置，留在你的实验日志里，设计不记；
- 5（P56）写成 D173，里程碑 C23（§8）。
D 不受影响：D 的 Stage 用默认值（什么都不留、不加），D 的分支已经全部并入 dev-two-tier。
路径相对 4dTrajectory/ts_transformer/。

一、代码：C23（D173），在 dev-two-tier-v4-post 上，先把 dev-two-tier 合进来
1. Settings.advantage_centering（默认 False）：本轮开始时的 V 读完所有样本、算出 GAE 以后，每个计数行的优势减去
   本轮所有计数行优势的均值；只减均值，不除标准差；目标 R_t 仍用减均值之前的 A_t + v_t；本轮几遍 pass 都用这一套。
2. Settings.value_epochs（默认 None = 每遍都训，即旧行为）：V 只在本轮前 value_epochs 遍里走步，和模型用同样的更新；
   不能大于 epochs，大于就按名字拒绝；热身轮只训 V 的这几遍。
3. 两个设置在别的方法下不是默认值时按名字拒绝；round.json 记下减掉的均值和 V 训了几遍。
4. 测试按 §8 C23：默认值下 value 轮和改之前逐位相同；减均值后计数行优势均值为 0、目标不变；value_epochs 1、epochs 4 时
   V 的权重只在第一遍变、模型四遍都变；没有这两个字段的旧记录按默认值读；续跑会比较它们。
5. 改动文件的测试和 test_architecture → 独立审查（只审代码，审查者不能是作者，审查和测试同时开始）→ 显式路径提交
   （提交前看 git diff --cached --stat）→ 日志写一行 → 报告 dev-two-tier-v4-post 能否快进 dev-two-tier，由用户合并。

二、实验
C23 合并以后，跑用户已经定的那个 campaign（你 requests 第 5 条里的设置：同 post_value1_20261007，加
--advantage-centering --value-epochs 1，种子 2031）。从用户合并后的提交建运行工作树（outline D163），用 systemd 单元，
主机和 GPU 上没有别的任务时跑；intent 在发布前写进 intents.json；跑完数据设为只读并写 SHA256SUMS，结果写进实验日志。

你只能写这些设计文本：post_training §0.3 的状态行和你的日志。requests_from_c_to_designer.md 整份重写：1–5 条已处理，
删掉；新的读法另起。日志和 requests 文件提交到 dev-two-tier：用一个临时工作树加快进，只放这些文件。
```
