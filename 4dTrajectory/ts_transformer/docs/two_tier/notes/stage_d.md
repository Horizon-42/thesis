# 阶段 D：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_stage_d_implementation_log.md`）。
2026-10-12，Claude 写，用户转发。上一条（D194 (3) 推广、真实窗口核对、A/B 臂）你已做完（日志 §22），这条整份替换它。

```
你 requests 的 108–120 处理完了：
- 109：写进 D194 (3)：CPU 上逐位相同，GPU 上在 GPU 舍入内（一次搜索可能选别的点）。
- 112：写成 D195（multi_control 新行），里程碑叫 MC14（MC13 已经是 D192 的多数点）。
- 113、114、115：写进 D191（续跑保留所有已完成批，不管哪种设备说的；CPU 最后一级还超就去说话设备再说一次；
  --gpu-squares 属于规则，续跑可改；--batch-squares 改分批，是 campaign 设置、进 profile、续跑不能改；所有 worker
  做 malloc_trim）。D193 (4) 的"批留在规则给的设备上"改了。轮的可重复性：CPU 逐位，GPU 在舍入内。
- 113、116：写进 post_training D185（续跑可以改轮数、CPU 规则和两个学习率，每次改动记前后值和起作用的轮；
  改了学习率之后的轮，比较时算另一个臂）。§7 第 4 行同步。
- 117：写成 multi_control §5 第 5 条（multi_window_read；stage D 的窗口列表：任一架受控飞机的结局在列表里就列入）。
- 118 (a)：写进 frontend §5.1。(b) 是读取器的修复，不用写进设计。
- 119：写进 §0.4 和 §5：选轮规则（正式 campaign 和对照臂所有轮里 select reward 最高的一轮）、原地延长到 12 轮、
  三条提前停止规则；选中的轮在 select 上偏高，报的数用 val。
- 120：写进 outline D163（run worktree 也要链接 aeroviz-4d/public/data/airports）。
- 108：不采用。(a) 平方和预测内存太粗，选不出峰值批；(b) 批是在上限处被打断的，达到的峰值总在下一级预算内，
  跳级省不了什么。重批已经由 114（CPU 超了去 GPU）和 115（--gpu-squares）处理。
- 110：信息，下次新 campaign 按测得的速度定规则（D194 (4)）。
这些都从 requests 里删掉。

顺序：
1. §0.3：D195 那一行的里程碑名改成 MC14；更新状态行。
2. 小改动，不用 agent 审核：multi_training_export 里窗口身份的 lambda 删掉，改为 import
   post.window_lists.identity_of（code-health-followups 里 96b5b87f 那条），翻掉它的状态行。核对一次：用第 7 轮
   的一个列表，改前改后选出的窗口相同。在分支上做；正式 campaign 在跑，只跑改动模块的测试。
3. 正式 campaign 结束后（停止规则触发或第 11 轮跑完）：
   a. 按规则选轮：正式 campaign 各轮和对照臂 8 轮里 select reward 最高的那一轮。
   b. 如果不是第 7 轮：先写 intents，再用 multi_window_read 读这一轮，照第 7 轮的方式发布两套
      （抽取的窗口、没落地的窗口）。
   c. 对选中的轮读一次 val（multi_control §5 第 3 条：读之前先 claim，只读这一次）。
   d. 报告给用户：每轮的 select 读数（落地、间隔丢失、其他失败、reward、KL），选中的轮，val 读数按机场和跨度。
      其他失败（最低高度不稳、超时）在涨，单独列出来。
4. 文档：日志 §22 补 115–120 和上面这些；requests 整份重写，只留还没定的。
```
