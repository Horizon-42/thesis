# 阶段 D：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_stage_d_implementation_log.md`）。
2026-10-10 晚，Claude 写，用户转发。上一条（D191、CPU worker 的 profile、混合 worker）你已做完（日志 §21），这条整份替换它。

```
你 requests 的 86、90–98 处理完了（multi_control D179、D191 改写，新 D192、D193；prior §7 item 7、
post_training §9 item 8 和接口表第 4 行补了名字）：
- 91、94：写进 D191（重批放最后；续跑可以改 CPU 批的规则；选择读数只在说话设备的 worker 上读；记录里的
  cpu_workers、cpu_measured、deferred）。D179 改成：1.3 的余量只管主机和训练那一遍，GPU worker 按批与批之间
  占的显存算。92、93 是测量，留在日志里。90 已经启动。这几条从 requests 里删掉。
- 95（R1）：写成 D192，新取值 branching = majority；grid、backward 都不动、逐位相同。细节我定了：往回的上限不放宽
  （8 个点里没有 5 条避开的点，就只有决策点那几组）；不加"没说复飞"的条件；16 s 的点保留；决策点自己就有 5 条
  避开时只成一组；批记录里每组加上类别（决策点、16 s、多数点）和找到多数点的搜索数。用户说"先不实现"：不建。
  计划清单 R1 那行写上 D192。
- 96、97（S1、S2）：写进 D193 (1)(2)，按你建的接受。S1 是进程选项，不是活动设置；正式用之前要在真实窗口上
  按 D181 (57) 的界核对。
- 98（S6）：接受，D193 (3)。
- S4：接受，D193 (4)。
- 86（S5）：不做（D193 末尾写了原因），从 requests 里删掉。
- R2、R3、R4、R7、S3 是实验，用户定，不改设计。R5 只读盘上的数据，你自己排，在第 5 轮的组被删之前读完。

名字：设计里 MC7 是阶段 D 的 Training view（前端）的里程碑。multi_train_20261010 是 MC6 第二次启动
（multi_train_20261009 是第一次）。日志、计划清单、交接、intents.json 里一律叫 MC6，计划清单里的"MC7"改掉。

顺序（代码在 dev-multi-speed 上接着做，先把 dev-two-tier 并进来，只有文档，小改动）：
1. D193 (3)：联合循环重建时，已经不飞的行（降落了，或失去间隔后结束，窗口里还有别的飞机在飞）拿掉，不再说、
   不再飞、不再算 token。核对按 D188 (1) 的方式：CPU 单线程，固定输入，stage D 和 stage C 的窗口；决策点、16 s
   点、B 点、组、深度都和改之前相同，边界附近的抽样计数，状态在 STATE_BOUND_M 内；逐位相同就写明。新写法要
   好读，旧写法删掉。审核按项目 CLAUDE.md 的 Code review。
2. D193 (4)：CPU worker 设主机内存上限，超了在 worker 里报错、接住；CPU 批超份额就在其他批说完后、少几个 CPU
   worker 同时的时候再说（还在 CPU 上）；profiled_fit 按 CPU worker 批与批之间占的主机内存算，最后一级放不下
   profile 的主机峰值 × 1.3 才拒绝。测试照 GPU 那边的层级测试写。审核同上。
3. GPU 空出来以后（只训交通模块的对照跑完）：D193 (1) 的核对（MC6 第 0 轮第 0、23、86 批的真实窗口，
   --executor-device cpu 对 speak，D181 (57) 的界），第 86 批两种执行器设备各计时一次；D193 (3) 前后同一批的
   时间顺手记下。都是报告，不是放行的门槛。
4. 完整 ts 套件，然后报告 dev-multi-speed 能不能 ff 进 dev-two-tier。
5. 文档：交接 §8 欠的照做（日志 §22、multi_control 和 post_training 的 §0.3 行、intents.json 里 MC6 第二次
   启动、学习率对照、只训交通模块对照的条目）；requests 整份重写，只留还没定的。
```
