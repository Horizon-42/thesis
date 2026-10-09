# 阶段 D：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_stage_d_implementation_log.md`）。
2026-10-09 晚，Claude 写，用户转发。上一条（A–D、MC12）你已做完，这条整份替换它；它的"二、之后"原样留在下面。

```
你 requests 的 71–77 条处理完了（用户 2026-10-09）：
- 71–76 的名字写进了接口：post_training §9 item 8、11（D185 的名字、window_behaviour R68），
  prior §7 items 1、2、3、7（CACHE_STEP_ROWS、Past.taken、LandingIndex 的排序、复制链的 device），
  vocabulary §6 item 5（Loop.copy、Executor.take、Spoken.take 的 device）。
- 77 的 (a)(c)–(j) 按你建的接受，写进 multi_control D182。
- 77 (b) 改了（D182 (3) 已改写）：所有事件的"避开"都要求 v 到事件后 60 s 不承担任何丢间隔。
  - 丢间隔的：同一对飞机保持间隔，且 v 不承担任何丢间隔（和原来一样）；
  - 复飞的：v 没说复飞，且不承担任何丢间隔；
  - 其他失败的：v 没有同样的结局，且不承担任何丢间隔。
  用更差的结局（v 承担的丢间隔）换掉原来的事件，不算避开。B 点的"没说复飞"照旧另加。

一、代码（dev-multi-control）
1. multi/backward 的 avoids 按上面改。
   测试补上：复飞事件和其他失败事件，各一条"事件没发生、但 v 承担了丢间隔"的续句，判为没避开；
   原来判为避开的情形照旧。grid 不经过它，不用逐位核对。30 行内算小改动，不用审核 agent。
2. 日志写一行；requests_from_d_to_designer.md 整份重写：71–77 已处理，删掉。
3. 合并前跑一次完整 ts 测试套件（现在没有 campaign 在跑）；报告 dev-multi-control 能否快进 dev-two-tier，由用户合并。

二、之后（等用户的话）
GPU 现在空着（MC6 已停），代码合并后、主机上没有别的任务时：
- 先用 multi_speed 量一批：第 69 条和 D183 之前、之后各一次，看保留峰值；再用 multi_profile 重新量内存（D179），定 worker 数。
  缓存按段扩容会产生碎片，所以看保留的，不只看已分配的；碎片明显就提议给说话 worker 开 expandable_segments。
- 再各量一批 grid 和 backward，比较三项：
  - 出样本的组数；
  - 飞到窗口结束的续句数；
  - 时间。
  backward 的时间里看一次只算一个窗口的一个点（77 (e)）是不是瓶颈；是的话，提出把各窗口同一搜索步的续句合成一批。
报给用户。哪个 campaign 用 backward，由用户定。

你只能写各文档 §0.3 的状态行、multi_control §0.3、你的日志、requests 文件和 code-health-followups。
```
