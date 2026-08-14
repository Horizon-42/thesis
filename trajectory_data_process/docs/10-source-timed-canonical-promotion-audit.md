# Source-timed 轨迹新旧差异与 canonical 切换审计

审计与切换日期：2026-08-15（Europe/Berlin）

## 结论

审计通过。`harvest-source-timed-v1` 已替换 canonical
`trajectory_data_process/outputs/harvest`。旧数据没有删除，完整保存在：

```text
trajectory_data_process/outputs/harvest-pre-source-timed-20260815
```

新数据使用 `lastposupdate` 作为水平位置时间基准，保留原始轨迹中未经修改的
经度、纬度和 HAE 高度样本，并通过 producer-side robust fit 生成
`observed-threshold-event-v6`。Evaluation 只消费该事件，不重新拟合。

## 审计范围与判定条件

审计逐条读取了旧、新两个数据集的所有 track JSON，并检查：

1. 旧 manifest 的 SHA-256 和 roster stat fingerprint 与新 manifest 保存的来源指纹一致；
2. `source_total = output_total + excluded_total`；
3. 新轨迹的每个 `[lon, lat, alt_hae_m]` 都是对应旧轨迹的同序子序列，坐标和高度没有被改写；
4. 所有数值有限，高度不缺失，时间严格递增，最终连续块的最大位置更新时间间隔不超过 15 s；
5. manifest 没有重复 identity、重复路径、缺失文件或未登记 track JSON；
6. 所有 estimated observed event 都是 `observed-threshold-event-v6`、
   `final_segment_robust_fit` method version 6；
7. arrivals、observed summary、evaluation report 和逐航迹 evaluation JSON 均可按严格 JSON 读取，并满足各自 roster/denominator 契约；
8. 从 canonical 路径实际加载全部模型 arrivals，验证 source SHA、样本 slice 和 runway target。

## 总体差异

| 指标 | 旧数据 | 新数据 | 说明 |
|---|---:|---:|---|
| Track 总数 | 269,652 | 267,194 | 2,458（0.91%）因最终 fresh block 少于 2 点而明确排除 |
| Sample 总数 | 116,183,567 | 94,100,827 | 保留 80.99%；删除 held、stale 和被大间隔隔开的非最终块 |
| Assigned | 46,378 | 44,693 | 清理后重新运行同一分类器 |
| Estimated threshold events | — | 44,555 | 44,693 个 assigned 中 138 个无法可靠拟合 |
| 有 source-valid bracket 的 estimated events | — | 28,479 | bracket 只选择物理 pass；事件值仍来自同一 robust 3D fit |
| Model-ready arrivals | — | 42,855 | canonical loader 全部成功 |
| Observed evaluations | — | 42,874 | pass 32,904；fail 9,843；indeterminate 127 |

在旧、新数据中同时保持 assigned 的 44,682 条轨迹里，跑道变化为 **0**。

## 分机场结果

| 机场 | Tracks 旧→新 | Samples 旧→新 | 新 assigned | Estimated event | Model arrivals | Evaluation pass / fail / indeterminate |
|---|---:|---:|---:|---:|---:|---:|
| KMSY | 19,786→19,581 | 7,484,421→5,977,365 | 4,159 | 4,147 | 4,159 | 2,972 / 1,175 / 12 |
| KRDU | 67,528→66,942 | 27,989,793→22,327,371 | 15,945 | 15,925 | 14,385 | 10,798 / 3,580 / 9 |
| KSJC | 106,468→106,250 | 47,766,343→43,518,355 | 11,204 | 11,123 | 11,193 | 8,554 / 2,569 / 81 |
| KSMF | 25,378→24,984 | 13,099,700→8,694,956 | 4,567 | 4,549 | 4,302 | 3,694 / 594 / 18 |
| KSTL | 50,492→49,437 | 19,843,310→13,582,780 | 8,818 | 8,811 | 8,816 | 6,886 / 1,925 / 7 |

## Robust fit 差异解释

有 bracket 的主群体对 freshness rebuild 不敏感。新旧 threshold altitude 的绝对差值
p95 为：KMSY 2.43 m、KRDU 2.95 m、KSJC 0.00 m、KSMF 0.94 m、KSTL 1.90 m。

较大变化集中在没有 bracket、旧记录含长 stale tail 或 coverage gap 的轨迹。典型情况是旧拟合使用了
陈旧保持段；新数据删除该段后，只用最终 fresh continuous block 拟合。KRDU 无 bracket、但有已发布
垂直目标的轨迹，其 threshold altitude 新旧绝对差值 p95 为 16.13 m；KSTL 对应值为 7.80 m。
这些记录没有被隐藏或强制修成 pass，而是保留 robust fit、窗口敏感性和 uncertainty 诊断，并由现行
标准产生真实 pass/fail。该变化是本次时间源修正的预期影响，不是坐标或高度被改写。

## Arrivals 与 observed evaluation 的不同 roster

初次审计曾错误假设 arrivals 必须包含每个 assigned track。代码契约实际为：

- `tracks` 保存全部 assigned；
- 模型 arrivals 排除没有已发布 LPV TCH/glidepath 的跑道和 local circuit；
- observed evaluation 仍评估 local circuit，但对没有已发布垂直目标的跑道记为 skipped。

按此契约重新验证后，五个机场的 include/exclude roster 均构成 assigned roster 的无重叠完整分区，
observed results、skipped 和 evaluation report 也逐 flight identity 完全一致。

## 代码与下游验证

相关测试结果：

```text
433 passed, 2 skipped
```

测试范围：`final_approach/tests`、`trajectory_data_process/harvest/tests`、
`evaluation/tests` 和 `4dTrajectory/ts_transformer/tests/test_ts_transformer.py`。

canonical arrivals 实际加载结果：

```text
KMSY   4,159
KRDU  14,385
KSJC  11,193
KSMF   4,302
KSTL   8,816
合计  42,855
```

## 切换与回滚状态

切换使用同一文件系统内的目录 rename，没有复制或删除数据。当前状态：

```text
canonical new: trajectory_data_process/outputs/harvest                         (8.6 GB)
rollback old:   trajectory_data_process/outputs/harvest-pre-source-timed-20260815 (5.0 GB)
```

五个新 manifest 保存的旧 source manifest SHA-256 均与回滚目录中的对应 manifest 一致。
旧目录在用户明确确认不再需要回滚前不得删除。
