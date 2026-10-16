# docs/literature 中的 decoder-only 序列预测论文

日期：2026-10-16
问题：`docs/literature/` 里有没有讲 decoder-only 模型应用到时序预测的论文？

方法：对各 README 和 PDF 正文做关键词检索（decoder-only、GPT、LLM、TimesFM、Lag-Llama、Moirai、Chronos、Timer 等）。
"decoder-only" 字样在 PDF 正文里核实过的，标 **[已核]**；其余是我的判断，标 **[判断]**。

## 结论

有，但都是**轨迹/运动序列**预测，没有通用时序预测（forecasting benchmark）一类的论文。
库里没有 TimesFM、Lag-Llama、Moirai、Chronos、Timer 的任何命中。

## 一、明确 decoder-only，用于轨迹/运动序列

| 论文 | 位置 | 做法 | 状态 |
|---|---|---|---|
| **BehaviorGPT** (Zhou 等, NeurIPS 2024) | `trajectory_as_language/papers/` | 原文："fully autoregressive Transformer… similar to decoder-only LLMs"。对 1 s 的 patch 做 next-patch 预测，不离散化 | **[已核]** |
| **SMART** (Wu 等, NeurIPS 2024) | `manoeuvre_tokens/papers/` | 轨迹离散成 0.5 s 的 token，经 "decoder-only transformer" 做 next-token 预测；另有 road token 预测，GPT 风格 | **[已核]** |
| **InfGen** (Yang 等, 2025) | `trajectory_as_language/papers/` | 在 decoder-only 架构上交错生成运动和场景；文中把 CAT-K 也归为 decoder-only | **[已核]** |
| **CAT-K** (Zhang 等, 2025) | `trajectory_as_language/papers/` | 在 SMART 类模型上做闭环监督微调；该文件夹的核心论文，效果超过 14 倍规模扩大 | 基于 InfGen 的描述 |
| **GUMP** (Hu 等, 2024) | `trajectory_as_language/papers/` | 自称 "GPT-like autoregressive predictive model"，未写 "decoder-only" | 原文表述已核；归类为 decoder-only 是 **[判断]** |

## 二、航空领域

**FTP-LLM / LLM-FTP**（Luo & Zhou 2025，arXiv:2501.17459）
- 位置：`manoeuvre_tokens/papers/`，笔记见 `manoeuvre_tokens/README.md` §2.13。
- 用 8 个开源 LLM（Gemma-2-9B、GLM-4-9B、LLaMA2-7B、LLaMA-3.1-8B、Mistral-7B-v0.2、Qwen2.5-7B 等）做 ADS-B 单步和多步预测。
- 这些 LLM 本身是 decoder-only，但论文没有使用该词。**[判断]**
- 做法：航迹点按文本打印，PEFT 微调；token 是数字的子词（如 "103.25" → "103"、"."、"25"），不是机动 token。
- 数据：PEK 和 PVG 的 ADS-B，窗口间隔 1 分钟，单步窗口长度 17。
- 结果：LLaMA-3.1-8B 综合最好，Mistral-7B 单步最好；推理延迟被论文列为实时使用的障碍。

## 三、不算 decoder-only

- **Trajeglish**（Philion 2023）：原文 "GPT-like **encoder-decoder**"。
- **MotionLM**（Seff 2023）：正文检索没有命中 "decoder-only"，架构未核实，不归类。

## 四、缺口与可补的候选

库里缺"通用时序预测里的 decoder-only"，若论文需要这一类，可补：
- TimesFM
- Timer / Timer-XL
- Lag-Llama

库里现有的时序论文（PatchTST、Crossformer、Pathformer、iTransformer 等，见 `hierarchical_prediction/`）都是 encoder 式。
