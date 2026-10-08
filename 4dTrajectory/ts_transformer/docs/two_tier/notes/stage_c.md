# 阶段 C：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_c_implementation_log.md`）。
2026-10-08，Claude 写，用户转发。

```
你的诊断读数请求写成了 post_training D175，里程碑 C24（§8）。只读不训练，和 A、B、D 都没有耦合。
和你的请求不同的地方（设计者定的）：
1. 起点模型不再读：它的逐窗口结果直接用 ceiling_20261007 里起点的 draw 0（同样的窗口、同样的随机数）。
   这样也不用为它找检查对象。
2. 不加 --per-window 选项：每次读都写 per_window.jsonl；每个丢间隔的窗口都算五个字段，关交通的读数也算。
3. 字段的定义按 D175：
   (1) 邻机分四类：no_landing / leader / follower / other_runway。
       跑道用被指挥机在丢间隔那一刻生效的跑道，没有就用它记录里的跑道；
       "同一跑道"包括按一条跑道间隔的跑道（airport_separation）；
       前后用 approach_clock_m 判。
   (2) 距离分档按你的。
   (3) 时间分档按你的。
   (4) 开局就冲突：只让被指挥机按第一预测步的速度走直线，邻机照它的记录飞（闭环里它本来就这样飞）。
       用 VISUAL 判官在每个 Δ 行判，判到丢间隔那一行和第一预测步后 120 s 两者中较早的那个为止。
   (5) 只对 ceiling 读过的轮次：32 次抽样里是不是每次都丢间隔。
4. 检查：同一进程里先读开交通，必须等于该轮 round.json 的读数，不等就按名字拒绝；再读关交通。
   比较的几个模型必须读同一批窗口（按航班和起始时间比），不同就拒绝。
   只接受 ts-post-train-v1 的 campaign。
5. "关交通"等于起点的初始化（§2 item 1、prior §7 item 5），不是 D116。
路径相对 4dTrajectory/ts_transformer/。

一、代码：C24（在 dev-two-tier-v4-post 上，先把 dev-two-tier 合进来）
experiments/post_diagnose.py 和 runners.md 的 R66。测试按 §8 C24。
改动文件的测试和 test_architecture → 独立审查（只审代码，审查者不能是作者，审查和测试同时开始）
→ 显式路径提交（提交前看 git diff --cached --stat）→ 日志写一行
→ 报告 dev-two-tier-v4-post 能否快进 dev-two-tier，由用户合并。

二、读数（C24 合并以后）
1. 读的模型按用户定的：post_seg60_20261007 第 5 轮，开交通和关交通各读一次；
   post_train_20261006 第 8 轮，同样开、关各一次；起点用 ceiling 的 draw 0。
2. 判读规则写进每个目录的 intent.json，要在读之前写好。规则按你 request 里的数值，用户要改就在读之前改。
3. 从用户合并后的提交建运行工作树（outline D163），用 systemd 单元，主机和 GPU 上没有别的任务时跑。
4. 跑完数据设为只读，写 SHA256SUMS。
5. 结果和你的读法写进实验日志，按判读规则给用户报告：G2 先报，G1 后报。下一步由用户定。

你只能写这些设计文本：post_training §0.3 的状态行和你的日志。requests_from_c_to_designer.md 整份重写：第 1 条已处理，
删掉。日志和 requests 文件提交到 dev-two-tier：用一个临时工作树加快进，只放这些文件。
```
