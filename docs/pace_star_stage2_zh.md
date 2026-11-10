# PACE + STAR 适配版：高难动作第二阶段训练

这套流程以已有 `ckpt_012000.pt` 为起点，不续接原训练的迭代号和优化器；第二阶段保存在新的 run 中。**STAR 是适配本项目世界模型训练器的片段重采样，不是论文中的 PPO/GAE 更新，也不保证未经实测的提升。**论文参考：`/home/kevin/下载/ex-rgmt.pdf`。

## 服务器操作

以下命令在仓库根目录执行。使用有 NVIDIA 驱动的服务器并确保动作 CSV 与基础 run 的 `run_config.json` 可访问。下例的三个自定义 CSV 路径按服务器实际位置修改；基础 checkpoint 应选与动作库一致的 run。依赖安装后可以使用 `.venv/bin/python -m ...`，无需重新生成命令入口。

```bash
BASE="$PWD/runs/supermimic_mjwarp_001/ckpt_012000.pt"
CUSTOM="$HOME/GMR/retargeted_motions/csv"
MANIFEST="$PWD/outputs/pace_stage2/grouping.json"

.venv/bin/python -m unisupermimic.scripts.pace prepare \
  --ckpt "$BASE" \
  --motion "$CUSTOM/freekick_g1.csv" \
  --motion "$CUSTOM/marseille_turn.csv" \
  --motion "$CUSTOM/tennis_g1.csv" \
  --output "$MANIFEST" --backend mjwarp --seed 20261110
```

这会在标准库 40 段和指定的三个动作上，每段做 5 次**从首帧开始、带随机初始状态**的完整复测；每次分别用种子 `20261110` 至 `20261114`。未失败且运行到动作末尾才计成功。成功 ≥4/5 的动作进巩固集，其他进困难集。输出清单保存绝对路径、CSV SHA-256、分组、完成次数、归一化回报、种子与一次不带随机扰动的确定性结果。`--motion` 可以重复更多次以追加其他动作，但 CSV 文件名不得重名，必须是至少两帧、36 列且数值有限；清单生成后修改 CSV 会被拒绝。若自动分组只有一个角色，需检查动作和基线，不要强行开始第二阶段。

```bash
.venv/bin/unisupermimic-train --config-name stage2 \
  task=g1_super_mimic/mjwarp \
  "algo.stage2_base_checkpoint=$BASE" \
  "algo.stage2_manifest=$MANIFEST" \
  algo.num_envs=4096 algo.iterations=6000 \
  "training.log_dir=$PWD/runs/pace_stage2_001"
```

`stage2.yaml` 是独立配置；请指定新的 `training.log_dir`，不要覆盖基础 run。配置快照和动作库身份写入该 run，恢复时检查清单身份和基础 checkpoint 路径。编号 checkpoint 只保存世界模型和策略；若同目录 `last.pt` **迭代号、世界模型与策略全部逐张量相同**，会取其 critic，否则从新 critic 初始化，等首次 critic 更新后才启用 STAR。不会混用其他迭代的 critic。世界模型可用两组数据；策略从困难环境的回放起点优化，在巩固环境起点施加相对于冻结基础策略的动作一致性损失。默认 80% 环境固定为困难角色、20% 为巩固角色；困难角色只在困难 bin 内使用自适应采样，巩固角色在自身 bin 内均匀采样。有效困难样本比例的 EMA (`β=0.99`) 调整巩固损失权重 (`λbase=0.3, κ=5, ρref=0.6`)。

STAR 适配版以 critic 对真实转移的单步 TD 代理分数排序；在**每个难度权重超过均匀基线的困难 bin**中取分数最高的 5% 连续片段（窗口或终止处截断）。默认 25% 的策略起点来自入选片段、按困难权重抽取，其余是困难角色普通回放。没有候选、critic 未准备好或回放条目过期时退回普通抽样。`algo.stage2_fraction`、`algo.stage2_star_fraction`、`algo.stage2_star_top`、`algo.stage2_star_refresh` 可调。`stage2_metrics.jsonl` 记录实际 STAR/普通采样比例、候选数、巩固损失/权重、困难有效样本 EMA，以及两类片段中观测到终止的完成比例（仅为截断窗口诊断指标，不能代替 20 次完整动作复测）；`train_log.csv`、`last.pt` 与周期 checkpoint 在新 run 下。若续训，使用该 run 的 `last.pt` 和原清单，并设置 `algo.resume=true`、`algo.load_run=<新 run 路径>`；不是重新初始化基础 checkpoint。

## 前后同条件比较

训练前后使用**同一清单、后端、种子和每动作 20 次**复测；两次都另外保存一次首帧、无初始扰动的确定性评估。

```bash
.venv/bin/python -m unisupermimic.scripts.pace evaluate \
  --ckpt "$BASE" --manifest "$MANIFEST" --backend mjwarp \
  --seed 20261110 --rollouts 20 --output "$PWD/outputs/pace_stage2/baseline20.json"

.venv/bin/python -m unisupermimic.scripts.pace evaluate \
  --ckpt "$PWD/runs/pace_stage2_001/last.pt" --manifest "$MANIFEST" \
  --backend mjwarp --seed 20261110 --rollouts 20 \
  --baseline "$PWD/outputs/pace_stage2/baseline20.json" \
  --output "$PWD/outputs/pace_stage2/trained20.json"
```

结果中 `per_clip` 给出逐动作完成率和归一化回报，`groups` 给出困难/巩固集的**逐动作平均完成率**，`group_norm_return` 给出两组的平均归一化回报，`deterministic` 保留逐动作首帧确定性评估。带 `--baseline` 会计算 `target_met`：仅当困难集完成率严格高于基线且巩固集下降不超过 **2 个百分点** 时为 `true`。应连同两组归一化回报一起报告；否则保留数据调参，不宣称提升。

由于本机 NVIDIA 驱动不可用，本轮仅交付实现、测试与操作流程；上述训练和 20 次复测需在服务器实际运行。
