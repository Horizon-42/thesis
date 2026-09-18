# 机动 token 方案：读数记录（2026-09-18 起）

设计与计划在 `2026-09-18_manoeuvre_token_plan.zh.md`；本文只放 campaign 的读数，**每个 campaign 一节，最新在前**，
每节固定四段：配置（臂文件、intents、commit、cohort）→ 表 → 读法（只读数字，不超出门）→ 门的判定与对状态表的一行结论。
门的判据不在这里重述，引用计划 §3.3。

---

## P1.4 · `manoeuvre_tok_20260918`（分词器 + 执行器联合；段长 × K × 种子）— 进行中

**配置。** 臂文件 `docs/experiments/manoeuvre_tok_20260918_s{20,30,60,90,120}_arms.json`（每段长 14 臂：学习码 K ∈ {16, 32, 64, 128, 256} × 两种子、
无 token 对照 × 两种子、指令词表 B × 两种子；文件内顺序 = 队列优先级：nt、K32、K128、cv 各两种子，再 K16 / K64 / K256）；intents 条目 `manoeuvre_tok_20260918`；
代码 commit **c391265**（执行器接线 9002cc0，分词器 31b9c24，段 81ca7b1）；runs worktree `.claude/worktrees/two-tier-runs` @ c391265；
cohort `development_cohort.json`：120 s 契约（锚点 59）下 6798 train + 1392 val（v2 的 L1 cohort 是 6848 / 1401：多丢 3 个 train、9 个 val，
因为整个 campaign 用最严格的 120 s 契约一个 cohort，段长轴在同一批航班上比较）。执行器配方 = v2 L1c pa（§2.6）。
队列：`<campaign>/queue_p14.sh`，段长顺序 60 → 30 → 90 → 20 → 120，每臂约 11 min（20 s 段的冒烟：3.3 s/epoch × 180 + 1 min 建数据；120 s 段的 rollout 更长）。
读数 runner：`python run_ts.py manoeuvre_readout --campaign <campaign> --segment-s <S> --out <campaign>/readout_s<S>`（协议 C，固定锚点 59，val 1392 航班，
ADE[0,Δ] 1 s 网格，与同种子无 token 臂逐航班配对；输出目录存在即拒绝，重读用新目录名）。

**早读（2026-09-18 07:25，单种子 1337、60 s 段，训练时的数字——不是门的读数，等 readout 覆盖）。** 前四臂的 `history.json`：
固定锚点 59 的 val ADE[0,60]（选 epoch 的指标，均值）无 token 116.5 m、K32 98.0 m、K128 99.8 m、指令词表 110.3 m——码给 L1 约 17–19 m（均值，
非配对 p50；门 T(i) 要两种子配对 p50 ≥ 30 m）。联合训练的梯度到达了编码器（`manoeuvre_tokenizer` 梯度组均值 0.0075 / 0.0045，骨干 0.013 / 0.011）。
**码用量偏斜**（训练随机锚点上的直方图）：K32 用了 15/32、最大份额 0.52；K128 用了 29/128、最大份额 0.57；指令词表 42/63、最大份额 0.66——
远超 T(iii) 的 25 %。读法：进近的大多数段是"直飞、保持"，一个码就装下了；执行器从码里取到的是粗的"转不转"信息。待 readout 的 val 固定锚点直方图与两种子配对差再判。

**表。** （待各段长训完后由 readout 填；每段一张：臂 × {ADE[0,Δ] p50 全/直/雷/建立，对无 token 的增益 p50 与"臂更好"份额，码用量 used/K、最大份额、熵}，两种子并列。）

**读法。** 只读数字：增益 = 无 token 臂 ADE − 带真值码臂 ADE（正 = 码有用）；两种子都 ≥ 30 m 才算过 T(i)；码用量最大份额 ≤ 25 %、未用 ≤ 10 % 才过 T(iii)；
选 K = 过门的 K 里增益（两种子均值）与最好者差 ≤ 10 m 的最小 K。T(ii) 在 P2.3 补。

**门的判定与结论。** （待填。）

---


