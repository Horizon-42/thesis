# 阶段 D：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_stage_d_implementation_log.md`）。
2026-10-08，Claude 写，用户转发。

```
你 requests 的 49–53 条用户都定了（2026-10-08），写进 multi_control：
- D179（49、50）：MC5 只量内存。每种跨度的第一批和行数最多的一批各量一个 worker，再量 pass；
  记下主机和 GPU 的峰值和每批的时间；不说整轮、不读读数、不量离散度。
  估算时把测得的峰值乘 1.3，系数记进 profile.json；C 的估算不变。
  select 集大小（select_per_airport）是开跑前定的设置，D166 里"由第 0 轮离散度定"作废。
- D180（52、53）：multi_train 和 multi_profile 加 --speak-device，照 C 的做法：
  - 默认就是 --device（旧行为）；只有一个 worker 时拒绝，campaign 在 CPU 上时也不许把 worker 放到 cuda；
  - profile 在 worker 所在的设备上量，并记下设备；profiled_fit 拒绝另一种设备的测量；
  - CPU 上的 worker 算主机内存，不算 GPU。
  - 1200 s 跨度的批行数（53）是设置，在测量里一起量。
- 第 51 条：起点、种子、设置、轮数是实验信息，不进设计；测量完和 worker 数一起报给用户定。
路径相对 4dTrajectory/ts_transformer/。

一、代码（在 dev-multi-control 上，先把 dev-two-tier 合进来）
1. 上一条的求位置 helper（frontend D178 (6)）如果还没做，先做。这是小改动，不用审核 agent。
2. D179：multi_profile 只留内存测量（第 1–2 部分），删掉整轮、读数和离散度；profiled_fit 乘 1.3。
3. D180：两个 runner 的 --speak-device。拒绝规则和 C 的一样，能复用共用代码就复用。
4. 测试：
   - 默认设备时行为和改之前一样；
   - 拒绝的几种情况；
   - CPU 的测量只算主机内存；
   - 另一种设备的测量被拒绝；
   - 余量系数写进记录并参与计算；
   - profile 不再说整轮。
5. 改动文件的测试和审核同时开始。审核按项目 CLAUDE.md 的 "Code review"：范围是 diff，不搜仓库，S3 不改。
   → 显式路径提交 → 日志写一行 → 报告 dev-multi-control 能否快进 dev-two-tier，由用户合并。

二、测量（用户合并后；主机和 GPU 上没有别的任务时，用 systemd 单元，从运行工作树 D163）
1. CPU 和 GPU 上各量一次（几分钟）。
2. 1200 s 跨度量 24 行和 48 行两种（另两个跨度用 64、48）。
3. 报告给用户：
   - 每个设备、每种跨度一批的时间和峰值，以及按 1.3 余量各能放几个 worker；
   - 你建议的 MC6 设置：worker 数和设备、批行数、第 51 条的起点、种子、轮数和停止规则，以及估算的一轮时间。

三、MC6
用户定了设置以后：intent 写进 intents.json，从运行工作树启动（outline D163），用 systemd 单元。

你只能写各文档 §0.3 的状态行、multi_control §0.3 和你的日志。requests_from_d_to_designer.md 整份重写：
49–53 条已定，删掉；新的读法另起。
```
