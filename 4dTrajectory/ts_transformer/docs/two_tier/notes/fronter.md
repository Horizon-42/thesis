# Fronter：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-06_fronter_implementation_log.md`）。
2026-10-08，Claude 写，用户转发。

```
你 requests 的 1–8 条用户全部定了，写成 frontend D178：
- 1–6、8 按你建的接受。
- 第 7 条：速度和集合解绑；格式名不改，这是用户点头的一条例外：
  已发布的旧集合留着没人读的 source.speed，新集合没有，不重导。
- D136 和 §3 item 10 已经按解绑和抽样改写。
路径相对仓库根；前端路径相对 aeroviz-4d/src/。

一、还没导的集合
loss_windows_c10_20261008（post_train_20261006 的 start 和 8，按清单 windows_lost_separation_select_20261008）
如果还没导：
1. 现在不用 --speed，意图已在 intents.json；
2. 从用户合并后的提交建运行工作树（D163），开跑前看主机；
3. 导完浏览器检查（一次性的 sonnet 子代理，只回文字结论）；
4. 日志写一行，删掉运行工作树。

二、第 6 条的两处镜像
等 C 的 identity_of（post/window_lists.py）和 D 的求位置 helper 合进 dev-two-tier 以后：
- 导出改调它们；
- 删掉两处镜像和钉它们的测试。
这是小改动（项目 CLAUDE.md "Code review"）：自己对着 diff 看一遍、跑改动文件的测试就提交，提交信息写明
"small change, no agent review"。日志写一行，报告能否快进，由用户合并。

requests_from_fronter_to_designer.md 整份重写：1–8 条已定，删掉；新的读法另起。
审核按项目 CLAUDE.md 的 "Code review" 一节：范围是 diff，不搜仓库，S3 不改，小改动不用 agent。
```
