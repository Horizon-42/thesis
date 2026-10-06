# 阶段 B：当前命令

只放最新一条命令，新命令整份覆盖，不是日志（过程记在 `readouts/2026-10-05_stage_b_implementation_log.md`）。
2026-10-06，Claude 写，用户转发。上一条（B5 后合并 B13、全量套件）、B6 发布和 B7 已做完（`d14a2f76`、`d1386a77`）。

```
三个阶段 Training 视图的统一布局。依据：outline §6.2、D133（用户 2026-10-06 定由你来做）。
用户特许：这次可以改 A 和 C 的前端视图文件，以及 aeroviz_backend/http_server.py 的路由。
不改任何 Python 导出、集合 / 样本 / 索引格式，不重新导出，不改后端的实时航段。

1. 从 dev-two-tier（含 D133 的提交）新开分支 dev-training-layout，worktree .claude/worktrees/training-layout
   （数据目录按 outline §5 规则 1 链接到 live 数据）。
2. 按 §6.2 第 1–5 条做：
   - 左栏三个阶段一样：阶段切换；选集合，下面一行是意图里这个集合自己的那一行，加 SMOKE 标记；四列条目表；一行一个读数；Draw。左栏不放表格。
   - 句子栏的标签页选句子：A 是 Labelled 和各 Δ；B 是 Labelled、Closed loop、Sample 0、Sample 1；C 是 Labelled、Start (base) 和各轮。
     标签上有结局颜色的小点和提示，方向键可切换；左栏不再有第二个选择器。
   - chip 和句子栏 ⓘ 的说明按句子种类写；ⓘ 的第一行写集合和意图那一行，并链到详情页。
   - 详情页三个阶段都有，第一节是「The set and the experiment」。
   - B 的「At the cursor」一行和概率条留在左栏；行检查器的完整内容放进详情页。
   - 后端 GET /experiments/intent?run=<set id>：每次请求都读 intents.json；找到一个、没找到、找到多个，各自按名字回答。
   - 共用部件：一个加载集合的 hook、选集合、条目表、标签页、详情页状态、DetailsLink、Draw。
3. 测试按 §6.2 第 6 条：
   - 共用部件的 Vitest，夹具由导出写；
   - 三个阶段现有的测试全部保留；
   - 后端 intent 的测试：找到一个、没找到、找到多个。
4. 每一步：单文件测试 → 独立审查（只审代码）→ 用显式路径提交。
   全部做完后：
   - 先看资源（§5 规则 13），再跑全量测试；
   - 从 worktree 起测试栈，后端和 vite 用自己的端口；
   - 浏览器检查交给一次性子代理，三个阶段都查：标签页能选句子；每个 ⓘ 和读数行都能打开详情页；意图显示出来；一个词能实时飞；
   - 报告地址和停止命令。
5. 实现日志在 B 的日志里新开一节。设计没说的地方写成提议，放进 requests 文件，不改设计文本。
6. requests 文件重写：§3 第 1、3、4 条已进设计（§7、outline §6.2、§11）。
   第 5 条 C 已经跟上（post_window_loop.py 已按新签名传 start 的观测行），一并删掉。
7. 分支不合并：报告能否快进，由用户合并。
```
