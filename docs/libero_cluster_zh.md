# 个人 fork 的 LIBERO 复现记录

仓库：https://github.com/PanBikang/GalaxeaVLA 。`origin` 指向个人 fork，`upstream` 指向 OpenGalaxea。
上游基线：`89f2322b4ad016e192437adc1a2c253b05bab246`。

## 目标与证据边界

论文：https://arxiv.org/html/2608.11739v1 ，§5.2.3 / Table 3。
标准 LIBERO 四个 suite，每个 10 个任务，每任务 50 次 benchmark 初始状态 rollout，共 2,000 次。

| Suite | 论文成功率 |
| --- | ---: |
| Spatial | 98.4% |
| Object | 100.0% |
| Goal | 98.6% |
| Long (`libero_10`) | 98.6% |
| 平均 | 98.9% |

本次计划先验证三步真实数据微调、保存及加载推理，再复测官方 `g05-libero`。
官方 checkpoint 评测属于发布权重的结果复测；三步训练不等于从头复现论文的 100K-step 微调。
论文的 100K steps、学习率 1e-5、weight decay 1e-2 可在后续完整训练中使用，但还需核对全局 batch 和数据筛选。
§5.7 的 AR/FM GRPO 比较需要另外的 RL 实验；本仓库未发现对应 GRPO 训练入口，不能由成功率评测推出该结论。

## 环境与本地资源

```bash
cd /public/node03/users/panbk/data/GalaxeaVLA
source scripts/cluster/env.sh
```

- `.venv`：Python 3.10.16、PyTorch 2.7.1+cu128、Transformers 4.57.1；`uv.lock` 已改为可访问的官方 PyPI，并补齐 open3d 的 ipywidgets 依赖。
- TorchCodec 固定从 PyPI 安装 CPU 版 0.4.0；训练视频走 CPU 解码，避免误选 CUDA wheel 后缺少 NPP 动态库。
- 标准 LIBERO：`/public/node03/users/panbk/data/third_party/LIBERO`，commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`。不使用 LIBERO-plus 任务替代标准评测。
- 现有数据：`/public/node03/users/panbk/data/FasterWAM/data/libero_mujoco3.3.2`。
- 数据为 LeRobot v2.1、双相机 512×512，直接使用代码已有的 v2.1 读取器。
- 四个 suite 共 1,712 episodes / 277,713 frames；这份经过筛选的数据适合流水线验证，不能直接宣称与论文每任务 50 demonstrations 完全相同。
- `configs/task/libero_local.yaml` 对齐 base 权重的 AR / 27D 配置；`configs/data/libero_local.yaml` 通过 `LIBERO_DATA_ROOT` 引用现有数据。
- Local task 另外对齐发布 base 配置的 `tokenizer.vq_config.block_size=8` 与 `vision.temporal_freq=4`；单帧输入仍由模型的单帧分支处理。
- 修复标准 LIBERO NumPy 初始状态与 PyTorch 2.6+ 的加载兼容；robosuite 私有配置关闭 `/tmp` 文件日志，输出由 Slurm 收集。
- 修复预先分词的变长请求在推理批处理中未 padding 的问题，使用仓库已有的批处理一致性测试验证。
- 通用 `env.sh` 默认 `MUJOCO_GL=osmesa`，供 CPU 诊断使用。GPU 评测 / smoke 脚本启用已验证的 EGL：`graphics.sh` 复用用户已有的 NVIDIA 580.178.04 图形库并检查内核驱动版本。最初作业 3494 因节点未配置 NVIDIA EGL vendor 失败，随后用正确的用户态库解决。

## 作业入口

每个脚本显式声明 CPU、内存、时间及共享日志目录。提交前检查 `squeue -u "$USER"` 和 account/QoS 剩余额度；不要取消其他项目的作业。

```bash
mkdir -p runs/slurm
sbatch --parsable scripts/cluster/setup.sbatch
sbatch --parsable scripts/cluster/check_environment.sbatch
sbatch --parsable scripts/cluster/check_libero.sbatch
sbatch --parsable scripts/cluster/validate.sbatch
```

安装和检查按依赖顺序执行；`setup` 的 frozen lock 不会自动升级依赖。
GPU 排队时，可单独验证软件渲染，不能把它写成 EGL GPU 渲染通过：

```bash
sbatch --parsable --partition=cpu --qos=cpu --gres=none \
  --export=ALL,MUJOCO_GL=osmesa scripts/cluster/check_libero.sbatch
```

## 官方权重与下载

`OpenGalaxea/G05` 是 Hugging Face gated repository。初次下载返回 401；用户登录并取得授权后，已验证 ActionCodec 和 LIBERO 文件可访问。
新机器需由账号持有人在 https://huggingface.co/OpenGalaxea/G05 确认访问权限，然后在终端登录（不要将 token 写入仓库或聊天）：

```bash
HF_HOME=/public/node03/users/panbk/data/hf-cache \
  /public/node03/users/panbk/data/GalaxeaVLA/.venv/bin/hf auth login
```

之后依序提交并检查每个作业的实际退出码：

```bash
sbatch --parsable scripts/cluster/assets.sbatch
# 上一步成功，确认权重、processor、ActionCodec 和配置 sidecar 完整后：
sbatch --parsable scripts/cluster/train_smoke.sbatch
# 官方发布权重的少量 rollout；先检查服务端和仿真通信：
sbatch --parsable scripts/cluster/eval.sbatch 1
# smoke 通过且额度允许时，每节点一个 suite，跨分区并行（先核对节点空余 GPU）：
sbatch --parsable --partition=h20 --nodelist=zp-nc06 --export=ALL,G05_EVAL_SUITE=libero_spatial,G05_EVAL_RUN_ID=official_repro scripts/cluster/eval.sbatch 50
sbatch --parsable --partition=h20 --nodelist=zp-nc12 --export=ALL,G05_EVAL_SUITE=libero_object,G05_EVAL_RUN_ID=official_repro scripts/cluster/eval.sbatch 50
sbatch --parsable --partition=h200 --nodelist=zp-nc69 --export=ALL,G05_EVAL_SUITE=libero_goal,G05_EVAL_RUN_ID=official_repro scripts/cluster/eval.sbatch 50
sbatch --parsable --partition=h200 --nodelist=zp-nc118 --export=ALL,G05_EVAL_SUITE=libero_10,G05_EVAL_RUN_ID=official_repro scripts/cluster/eval.sbatch 50
```

`assets.sbatch` 下载全部五个模型及公共 sidecar，固定并记录 HF revision。
当前 snapshot 共 29 个文件，五个模型各约 11.44 GB。配置检查如下；这不是性能测量：

| 模型 | 动作维数 | 离散动作 | 连续动作 | CoT |
| --- | ---: | --- | --- | --- |
| base | 27 | 开 | 开 | 开 |
| DROID | 20 | 开 | 开 | 开 |
| LIBERO | 20 | 关 | 开 | 关 |
| RoboTwin 2.0 | 20 | 关 | 开 | 关 |
| SO-101 | 20 | 开 | 开 | 开 |

公开 LIBERO / RoboTwin bundle 默认走连续动作头，因此其成功率复测不能当作 AR/CoT 消融实验。
下载脚本使用四个已验证可联网的 CPU 节点并行处理五个模型，每个文件进行 HTTP 分段续传和 LFS SHA-256 校验。
`.ranges` 是保留的下载分段缓存；只有完成 SHA-256 校验后才将文件放到正式 checkpoint 路径。
`runs/provenance/checkpoint_inventory.json` 记录文件清单；`inspect_checkpoints.sbatch` 检查各文件的张量数量、dtype 和配置，写 `checkpoint_inspection.json`。
下载作业 **3558** 已完成，退出码 `0:0`；总计 **57,742,756,737 bytes**，全部五个模型 SHA-256 匹配。检查作业 **3545** 退出码 `0:0`，五个 bundle 均可解析，均包含 946 个 FP32 state-dict 张量（张量元素总和包含权重共享别名，不能直接当作独立参数数量）。
`train_smoke.sbatch` 输出 `runs/libero_local/smoke_<jobid>/`，保存 `last.pt` 后加载并从真实仿真观测推理，写 `inference_check.json` 和动作数组。
评测固定 simulator seed=7。结果分别写入 `runs/libero_reproduction/<array-jobid>/<suite>/summary.json`；汇总时必须核对四个 suite 均为 500 episodes，且所有 Slurm 作业 exit code 为 0。
可用 `G05_EVAL_RUN_ID` 指定共同的结果目录，用 `G05_EVAL_SUITE` 指定单个 suite。不要在同一个结果目录重复提交同一 suite。
本集群 `zp-nc17` 出现多环境 EGL 初始化时的驱动锁等待；本次将其它 suite 分散到不同节点执行，保留已有成功结果。节点空闲情况变化时，应调整上述节点列表，而不是挤到该节点重复启动。单个 Slurm 多节点作业不能把 `h20,h200` 两个分区合并为一个分区。
`scripts/cluster/summarize_libero.py <run_dir>` 检查完整的 2,000 次结果并生成论文对照 JSON。
另提供 `train.sbatch` 全数据训练入口，可通过 `G05_MAX_STEPS` / `G05_BATCH_SIZE` 调整训练规模；该入口尚不代表已经完成 100K-step 训练。Smoke 使用独立统计文件，避免其单 suite 统计污染全数据训练。

```bash
squeue -j <jobid>
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -n 60 runs/slurm/<对应日志>
```

## 已验证与未验证

环境安装作业 3484、H20 CUDA 前向/反向检查 3485 均完成，退出码 `0:0`。
四套数据字段、40 个任务的初始状态数量，以及每 suite 一个场景的 OSMesa 渲染/20 步仿真已通过，结构化证据在 `runs/provenance/libero_check.json`。
作业 3496 的仿真部分通过，但随后批处理一致性测试暴露变长 token 问题，因此该作业整体退出码为 1。
修复后，作业 **3498**（配置、路径、语法、批处理一致性）和 **3499**（四 suite 场景、数据与批处理一致性）均完成，退出码 **`0:0`**。
作业 **3518** 进一步在 H20 上验证实际 FA4 vision 和 FLA 算子的前向/反向，退出码 `0:0`。
作业 **3534** 验证新增下载/训练脚本的语法、Hydra 配置及推理批处理测试，退出码 `0:0`。
作业 **3539** 修复 CPU TorchCodec 依赖并通过 `uv pip check`；**3542** 实际解码四套数据双相机视频，退出码均为 `0:0`。
作业 **3544** 使用 NVIDIA EGL 在 H20 上通过四套场景/初始状态/批处理检查，退出码 `0:0`。

真实数据训练已完成三次更新，loss 为 **4.0019 → 2.5461 → 1.7877**，首批 ActionCodec token roundtrip 为 **LOSSLESS**，base 权重加载为 **946/946，missing=0，unexpected=0**。
训练输出：`runs/libero_local/smoke_3568/checkpoints/step_3.pt`。该组合任务初次在保存后的配置恢复阶段失败；修复 OmegaConf tokenizer 引用恢复后，作业 **3574** 使用已有权重通过推理检查，退出码 `0:0`，没有重跑已完成的训练。
推理检查确认 vision patch 权重发生非零更新（样本最大绝对差 `2.5474466383457184e-05`），返回有限的 `[1,32,6]` 右臂动作和 `[1,32,1]` 夹爪字段。AR 结果中的 absent-group 标志仍记录在报告中；三步 smoke 不代表控制策略已收敛。
新增修复包括 `logger.type=null` 的禁用日志器处理、tokenizer sidecar 不再破坏 OmegaConf 引用，以及并行环境数不超过请求 trial 数的计数保护。作业 **3573** 通过 13 项相关测试，退出码 `0:0`。
官方 LIBERO smoke 作业 **3566** 为 Spatial **10/10**，退出码 `0:0`。完整评测仍在进行，已有 Spatial **494/500 = 98.8%**（作业 `3569_0`，退出码 `0:0`）；其余 suite 未完成前不报告总体复现成功率。
