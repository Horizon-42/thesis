# 阶段 D：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_stage_d_implementation_log.md`）。
2026-10-10，Claude 写，用户转发。上一条（D190、profile、完整套件）你已做完（日志 §20），这条整份替换它。

```
你 requests 的 85、87、88 处理完了（写进 multi_control D190 和 prior §7）：
- 85、88 按你建的接受；88：句子结束之后的行在界外，不算。
- 87：两种方式都保留是用户的话，D190 已改写；prior §7 item 1、3 的名字补上了（Past.taken 换成 Front、Past.together、
  COPIED、MERGED、FRONT_ATTENTIONS；Speaker(…, attention=)）。runners.md 里 post_train（R65）和 multi_train 的
  --front-attention 你自己补上。
- 86（前缀分两块存）和 90（下一个 campaign）等用户定，先不做。

新的：multi_control D191（O20 (61)，用户 2026-10-10）——说话 worker 同时用 GPU 和 CPU。

一、先量（一次）
1. multi_profile --speak-device cpu，设置同 §18（MC6 的设置、--branching backward、front_attention merged），
   从 run worktree 跑，主机和 GPU 上不跑别的：一个 CPU worker 的主机内存峰值和 batch 0、23、86 的时间。
2. 算一下：3 个 GPU worker 旁边，主机放得下几个 CPU worker（主机这份要算上 GPU worker 的主机部分和主进程）。
   少于 2 个就停下，把数报给用户，不往下建。

二、代码（dev-multi-control，放得下 2 个以上才做）
3. 混合 worker（D191）：
   - stage C 和 stage D 的 runner：GPU worker 照旧，另加 CPU worker 的个数（参数名你定，按 L25 的规矩）。
   - 每批在哪个设备上说，由固定规则按它在轮里的位置定（按批号，每一段里固定几批放 CPU）；这个规则是 campaign 设置，
     记录下来、续跑时比较；默认全部放在 campaign 的说话设备上（等于现在的行为，没有这个字段的记录按默认读）。
     规则里 CPU 的份额按两边每批的实测时间定，让两边同时说完。选择读数的批用同一规则。
   - 每种设备的 worker 按顺序取自己设备的批，所以一轮的结果和 worker 谁快谁慢无关。
   - D185 的续跑：每个已完成的批按规则核对它的设备类型（不再是"剩下的批和已完成的用同一类设备"）。
   - 内存估算（D179、O15）分两本账：GPU 那本是 GPU worker 和训练那一遍；主机那本是所有 worker 的主机部分
     （CPU worker 的整个峰值、GPU worker 的主机部分）和主进程。
   - CPU worker 一个线程。
   - 核对：默认规则（全在一种设备上）时，一轮和改之前逐位相同（window_behaviour 和 C27 的续跑测试）；
     测试：规则按批号分配、同一规则两次分得一样、续跑时设备不符按名拒绝、两本账各自算。
     CPU 和 GPU 说同一批最后几位可能不同（post-training §6.4），不做跨设备的比较。
   - 审核走 agent（范围是 diff）。
4. 用混合 worker 说一轮的几批（GPU 3 个 + 你算出的 CPU 个数），记下两边各说了几批、各用了多久，
   估一轮说话的时间，写进日志。
5. 合并前跑完整 ts 测试套件，提交号和通过数写进日志；报告 dev-multi-control 能否快进 dev-two-tier，由用户合并。
   然后按新的数重提下一个 campaign（requests 90），由用户定。

日志写一行；requests 整份重写，85、87、88 删掉（86、90 留着等用户）。
你只能写各文档 §0.3 的状态行、multi_control §0.3、你的日志、requests 文件、code-health-followups 和 runners.md。
```
