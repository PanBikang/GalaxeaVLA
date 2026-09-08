# RoboTwin 2.0 复测

目标为论文 §5.2.2 / Table 2：Clean 93.7%、Randomized 92.8%、平均 93.3%。使用已下载并校验的 `g05-robotwin20` 发布权重，不重新训练论文的四个 epoch。

## 固定协议

- 论文 Table 7 的全部 50 个任务，已与模拟器 `_eval_step_limit.yml` 逐项核对。
- 每任务每场景 100 个通过标准专家可解性检查的种子：共 10,000 次策略 rollout。
- 论文正文写 100 次；仓库默认 50 次，Table 7 的数值均为 2% 的整数倍。因此预先规定：主结果使用 100 次，同时独立报告前 50 次前缀统计。
- seed=7（RoboTwin 起始候选 seed=800000），instruction_type=seen。
- action horizon=32，replan_steps=24，FM steps=10，bf16。24 是 `configs/sim_robotwin.yaml` 的有效默认值，覆盖策略 YAML 中的 8。
- 使用 PyTorch SDPA 视觉注意力，以处理 FA4 与 SAPIEN/cuRobo 同进程中的 CUDA library 加载错误。GPU 数值对照最大绝对差为 0.0009765625。
- 启用动作缓存期间跳过重复相机读回，但仍执行 `_update_render()`，保留相机位姿和随机光照/RNG 更新。关闭视频输出。
- 每回合记录 trial ID、实际候选 seed、success、action steps、instruction；检查精确回合数和结果文件的一致性。中断后从已完成回合继续，不重复计入。

冻结协议与任务分片：`runs/robotwin/robotwin_20260908/manifest.json`。

## 环境

使用独立 `.venv-robotwin`，原 LIBERO `.venv` 不做安装更改：

```bash
source scripts/cluster/robotwin_env.sh
```

Python 3.10.16、PyTorch 2.7.1+cu128、SAPIEN 3.0.0b1、Warp 0.11.0、cuRobo v0.7.5。cuRobo 使用已有的用户 CUDA 12.8 工具链编译，未修改系统软件。

RoboTwin 固定为官方 `bf44be51cf5717a5595ce59447f2cf5263d2aa95`；最新版主分支已移除本模型依赖的旧评测入口。cuRobo commit 为 `36ea382dabbde2a2431951fc9048f7e5ffd8106b`。

资产复用 `/public/node03/users/panbk/data/RoboTwin/assets` 的 objects 和 background_texture。机器人资产从已有压缩包解压到本项目，配置路径仅在该副本中更新。设备补丁显式使用分配到的 `cuda:0`；runner 保留 Slurm 的 GPU 可见性映射。Warp 编译缓存按进程隔离。

`robotwin_setup.sbatch` 配置环境，`robotwin_render.sbatch` 验证真实光线追踪图像，`robotwin_validate.sbatch` 验证 GPU 映射、回合计数与续跑。环境通过 `uv pip check`。

## 已完成的验证

| 检查 | 作业 | 结果 |
| --- | --- | --- |
| 光线追踪真实图像 | 3594 | exit 0 |
| 环境安装 | 3598 | exit 0 |
| Runner 9 项测试 | 3599 / 3606 | exit 0 |
| FA4/SDPA 数值对照 | 3607 | exit 0 |
| Click Alarmclock clean/random | 3608 / 3609 | 各 2/2，exit 0 |
| Handover Block clean/random | 3610 / 3611 | 各 2/2，exit 0 |
| Adjust Bottle + 严格权重加载 | 3612 | 1/1；946/946，无缺失/部分加载，exit 0 |
| 50 任务协议核对与环境/脚本检查 | 3613 | exit 0 |

以上 smoke 只验证链路，不能代替完整成功率。完整结果将在 100 个 task/phase case 全部完成后汇总。
