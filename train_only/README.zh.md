# 只跑训练的分支：two-tier2-train-only

这个分支 = `two-tier2` + 本目录 `train_only/`。代码没有删、没有改：只加了把两层模型的训练流程（B → C → D）搬到另一台机器（A40
服务器）上跑所需的脚本和清单。以后 `two-tier2` 有更新，直接 `git merge two-tier2` 进来即可（本目录只新增文件，不会冲突）。

| 文件 | 作用 |
|---|---|
| `data_paths.txt` | 训练要读的数据（相对仓库根目录），由追踪实测得出（§2） |
| `track_paths.py` | 列出要带的航迹文件：标注数据集里每架航班一个（train、select、val；不含 test 日） |
| `pack_data.sh` | `size` 统计大小；`list` 列出全部路径；`rsync <目标>` 直接拷到服务器；`tar <文件>` 打成一个包 |
| `setup_env.sh` | 在服务器上建 conda 环境（Python 3.12、torch 2.11.0 cu128、numpy、geokit） |
| `requirements-train.txt` | torch 之外的包和版本 |
| `smoke.sh` | 小规模把 B、C、D 的 runner 跑一遍，D 按 `chain.sh` 的方式（profile + 多个说话进程）；正式训练之前先跑它 |
| `chain.sh` | 正式训练链：B（prior 交叉验证 + base + 一次 val 读数）→ C（post_train）→ D（multi_profile + multi_train），可断点续跑 |
| `run_tests.sh` | 只跑训练相关代码的测试（48 个测试文件，925 个测试），其余测试不跑 |

## 1 服务器上的步骤

```bash
# 1) 代码：分支没有推到 origin。任选其一：
#    a) 推送后在服务器上 git clone，然后 git checkout two-tier2-train-only
#    b) 在本机打包：git bundle create train-only.bundle two-tier2-train-only
#       拷到服务器后：git clone -b two-tier2-train-only train-only.bundle thesis
# 2) 数据：在本机（数据所在的 checkout 或其 worktree）执行，直接拷进服务器上的仓库目录
train_only/pack_data.sh size                                   # 先看大小
train_only/pack_data.sh rsync user@a40-host:/path/to/thesis    # 或: train_only/pack_data.sh tar /path/bundle.tar
#    用 tar 时，在服务器的仓库根目录解开：tar -xf /path/bundle.tar -C /path/to/thesis
# 服务器上核对（可选）：只核对拷过去的文件
cd 4dTrajectory/outputs/POOLED/instruction_language/v19_20261014 && sha256sum -c --ignore-missing --quiet SHA256SUMS
cd ../../executor/v24_20261014 && sha256sum -c --ignore-missing --quiet SHA256SUMS
# 3) 环境（setup_env.sh 本机没有实跑过：本机的验证都用现有的 aeroviz 环境；在服务器上它是第一次检查，必须先跑）
train_only/setup_env.sh aeroviz && conda activate aeroviz
# 4) 测试（可选）与小规模试跑
train_only/run_tests.sh
train_only/smoke.sh cuda                                       # 输出在 4dTrajectory/outputs/POOLED/smoke_<UTC>
# 5) 正式训练链（git 工作区必须干净；建议放在 tmux / systemd-run / nohup 里）
train_only/chain.sh 20261020                                   # 中断后同一条命令再跑一次即续跑
```

结果在 `4dTrajectory/outputs/POOLED/{prior/prior_<tag>, post/post_<tag>, multi/multi_<tag>}`，日志在同名 `.log`，链的总日志
`4dTrajectory/outputs/POOLED/train_only_<tag>.log`。拷回本机时放到同样的相对路径下（记录里的路径都是相对仓库根目录的，任何
checkout 都能读）。

## 2 训练需要的数据有多大（2026-10-15 实测）

方法：在本机用 Python 审计钩子（audit hook）记录 B、C、D 每个 runner 打开的每一个文件（`smoke.sh` 的全部步骤：prior_train、
prior_free_generation、prior_validation、post_train、两种起点的 multi_train），得出上面的清单；然后只用清单里的文件（硬链接
副本）把 `smoke.sh` 再跑一遍，确认没有缺文件：退出码 0，每一轮的更新数和 select 读数与用全部数据跑的结果完全相同；在同一个
副本上 `run_tests.sh`：922 个通过，3 个因缺 CIFP 原文失败；把它加进清单后这 3 个单独重跑通过（4 个 worker，41 min，旁边
有 D200 在跑）。

| 部分 | 大小 |
|---|---|
| 标注数据集 `instruction_language/v19_20261014`（只带 4 s 行间隔的闭环句子 + 各自参考） | 1.07 GB（整个目录 2.4 GB：2 s、8 s 的闭环句子和 labels.json 训练不读） |
| 执行器规格 `executor/v24_20261014`（spec + conformance 参考） | 19 MB（整个目录 2.2 GB：replay 目录是读数，训练不读） |
| 窗口普查 `post/windows_20261013` | 92 KB |
| B21 的 base `prior/prior_b21_20261014/base/run`（可选：不重跑 B 也能直接做 C、D） | 8.5 MB |
| harvest 的两个名册（每个机场 arrivals + tracks manifest） | 186 MB |
| 航迹文件：62,362 架航班（train 44,703 + select 6,913 + val 10,746） | 1.6 GB |
| 程序细节 `aeroviz-4d/public/data/airports/<机场>/procedure-details` | 1.4 MB |
| FAA CIFP 原文 `data/CIFP/CIFP_260806/FAACIFP18`（训练不读，只有 3 个测试拿它核对候选跑道和最终进近） | 51 MB |
| **合计** | **2.9 GB**（训练本身 2.8 GB） |

仓库自带的数据（`aircraft/*.json`、`docs/literature/arrival_separation/*.csv`）随代码一起 clone，不在清单里。

## 3 注意

- **test 日的航迹和 KAUS 不出本机**：清单不含 test 日的航迹文件，也不含 KAUS（留出机场）。两个名册（arrivals、tracks
  manifest）是整份拷的，里面有 test 日航班的名册行（只有元数据：航班号、跑道、时间等，没有轨迹）。
- **val 只读一次**：`chain.sh` 的 B 是正式活动，它的 base 会读一次 val（由 base 目录里的 claim 文件记账）。同一个 base 不能再读。
- **C、D 的设置要在 D200 选定后再定**：`chain.sh` 里的设置是目前最近的正式设置（B = B21 的活动；C = post_seg60 的设置，从
  base 出发；D = D200 的 (b) 臂：每 60 s 分支、功劳记到下一个分支点）。用户选定 D200 的方法后改 `chain.sh` 的 SETTINGS 段；
  `D_START=base` 让 D 直接从 B 的 base 出发（D200 的做法），默认 `c` 从 C 的最后一轮出发。改了设置就换一个新 tag：同一个
  tag 下，最后一轮已在的阶段会被跳过，profile 也会沿用。
- **工作进程数**：`SPEAK_WORKERS`（默认 3）、`CPU_WORKERS`（默认 4），各至少 2（multi_profile 要测一个进程旁边另一个）。runner 会先测内存，放不下就按名拒绝；A40 显存 48 GB，
  可以加大，主机内存和 CPU 核数决定 CPU 工作进程数。
- **没有为 A40 改代码**：例如几个 fold 同时训练、TF32/BF16，都需要改代码，属于用户的决定，本分支没有做。
- **检查都保留**：每个 runner 启动时的一致性检查（标注器、执行器、闭环的参考航班）照常运行；它们读的文件都在清单里。换了
  机器，执行器的参考轨迹若超出 1e-6 m 的界限会按名拒绝——那时先报告，不要关检查。
- 以后换了标注数据集或执行器版本：改 `data_paths.txt`、`pack_data.sh` 的 `ARTEFACT`、`smoke.sh` 和 `chain.sh` 的输入路径。
