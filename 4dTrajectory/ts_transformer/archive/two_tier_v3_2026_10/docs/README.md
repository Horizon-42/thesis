# 两层模型 `instruction-v3` 及以前的设计文档（已取代）

这些是指令词设计第一代到 `instruction-v3`（2026-09-23 至 2026-10-03）的中文设计和阶段笔记。现行设计（`instruction-v4`）是
`docs/two_tier/design/` 的四份英文文档（大纲、词表、先验、后训练），2026-10-04 起取代这里的全部内容；它们的代码由阶段 A 的 A0
归档到上一级目录（分支 `dev-two-tier-v4`）。文件原样移来，只改了相对链接，内容停在移走那天。

| 文件 | 内容 | 现在由哪份取代 |
|---|---|---|
| `two_tier_framework.zh.md` | 分层、包、产物、门 | `design/outline.md` |
| `instruction_vocabulary_design.zh.md` | 词、包络、标注器、取值 | `design/vocabulary.md` §3、§4 |
| `executor_design.zh.md` | 执行器（到 v11） | `design/vocabulary.md` §5 |
| `prior_design.zh.md` | 先验 | `design/prior.md` |
| `post_training_design.zh.md` | 后训练 | `design/post_training.md` §2 |
| `multi_aircraft_design.zh.md` | 多机（情境、扩充、间隔、排序；到第 9 步） | `design/post_training.md` §3 |
| `two_tier_stage_notes.zh.md` | 各阶段的状态、产物、决定 | 各份设计的 §0 |
| `continuous_segment_ablation_design.zh.md` | 对照实验：连续片段代替指令词，不用标注器（草稿，决定项未定） | 没有取代；按旧设计写的（进近列、许可词），要做时按现行设计重写 |

代码注释里的 "executor design §…"、"multi-aircraft design §…" 等指的是这里的文件。
