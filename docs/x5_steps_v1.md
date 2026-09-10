# X5 Steps v1：对完成的有效迈步给予有限奖励

Steps v1 在 Rewards v1 的 16 项奖励基础上，增加一个只在轮子重新触地时结算的 `completed_step` 奖励。目标是为有侧移命令或处于台阶、障碍地形中的有效抬腿迈步提供额外信号。它是一项待训练验证的设计，不代表已经获得侧移、越障或上下楼梯能力。

| 对照层级 | 任务 ID | 实验目录名 | 奖励项数 |
| --- | --- | --- | ---: |
| 原 X5 | `RobotLab-X5-MoECTS-v0` | `x5_moe_cts` | 14 |
| Rewards v1 | `RobotLab-X5-MoECTS-Rewards-v1` | `x5_moe_cts_rewards_v1` | 16 |
| Steps v1 | `RobotLab-X5-MoECTS-Steps-v1` | `x5_moe_cts_steps_v1` | 17 |

Steps v1 的配置位于 `source/robot_lab/robot_lab/tasks/x5_rewards_v1/env_cfg.py` 中的 `X5StepsV1EnvCfg`。原 X5 和 Rewards v1 保持独立。Rewards v1 的姿态舒适区、零命令轮速、离地空转及其他奖励保持原定义与权重，详见 [x5_rewards_v1.md](x5_rewards_v1.md)。算法、控制、观测、命令采样、地形、课程、随机化和终止条件继续继承，不为本实验增加固定步态或关闭轮驱动。

## 1. 参考项目实际提供了什么

本次同时检查了用户下载的目录：

```text
/home/xgy/重要文档/Isaac-RL-Two-wheel-Legged-Bot-flamingo_isaac_lab_envs
```

该目录不是 Git checkout。62 个 Python/README 文件的 Git blob 哈希与远端 `flamingo_isaac_lab_envs` 分支以下提交的树记录一致，没有缺失或不匹配：

```text
61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02
```

需要区分奖励函数、最终任务配置与视频展示。参考工程的当前代码不能直接作为“一套已验证侧移和上下楼梯的训练设置”使用：

- `Rough-Flamingo-v3` 注册到 stand_walk，但最终配置将地形改成 plane，关闭地形课程，并把 `vy`、yaw 命令设为零。类开头的 `vy=(-1,1)` 随后被覆盖。[地形覆盖](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/rough_env/stand_walk/rough_env_stand_walk_cfg.py#L177-L189)、[最终命令](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/rough_env/stand_walk/rough_env_stand_walk_cfg.py#L264-L269)
- stand_walk 的活动设计包括双足异步步态 `+4`、轮心高度奖励 `+0.5` 和轮动作惩罚 `-10`。这些是组合目标，不能只取其中一个函数并假定能复现展示动作。[配置](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/rough_env/stand_walk/rough_env_stand_walk_cfg.py#L66-L138)
- stand_walk 的轮电机仍然使用 `DelayedPDActuatorCfg`，并未禁用；轮动作归零只是较强的奖励约束。本实验不移植这一约束。[轮执行器](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/lab/flamingo/assets/flamingo/flamingo_walk.py#L93-L110)
- Rough v1 有“卡住时在落地结算离地时间”的想法；但其卡住判断把 yaw 命令与垂直线速度混合比较，计数器也缺少单独的环境重置处理。因此这里重新定义触发条件和结算依据。[原函数](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/lab/flamingo/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L116-L187)
- 参考的 `foot_clearance_reward` 使用固定世界高度，且轮心水平速度为零时可以取得最大值；它并不测量障碍物上沿或轮下地面。本实验不直接复制该公式。[原函数](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/lab/flamingo/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L599-L621)

本地历史 Hydra 快照中也存在 generator 地形和楼梯配置，但它们与当前 stand_walk 配置不同，不能据此将视频、checkpoint、奖励和任务版本对应起来。关于参考工程独立侧移、越障和上下楼梯的训练效果，本次仍没有建立这种对应证据。Steps v1 是借鉴设计思路后的 X5 实现，没有复制外部代码。

## 2. 什么情况下记录一次候选迈步

`completed_step` 必须先观察到轮子从接触变为离地，再观察到它重新触地。环境重置后的落地不能直接作为完成一步，未落地的持续腾空也不会按帧领取奖励。

起步时必须满足以下条件：

1. 平移命令大小至少为 `0.1 m/s`。
2. 有明确侧移命令 `abs(vy_cmd) >= 0.1 m/s`，**或者**机器人所在的地形类别为 `stairs_up`、`stairs_down`、`obstacles`。

这里使用工程已有的地形类别，不根据整片高度扫描的最大最小差判定障碍。后一种判断会将普通坡面也判为需要迈步的地形。地形类别本身较粗：同一台阶或障碍 tile 内的平坦出生平台也属于该类别，并不意味着轮子前方此刻存在障碍。

在候选迈步期间，平移命令必须继续达到阈值，当前机身坐标系下的平移命令方向与起步时的方向余弦必须至少为 `0.866`，约允许 30° 的方向变化。方向反转或明显改变会使该候选失效。

四轮中必须始终有至少两轮满足承重接触条件：接触传感器报告接触，且世界 Z 方向的净接触力大于 `5 N`。重新落脚的轮子也必须满足这个条件，单纯撞到台阶立面不能结算为有效落地。这是向上的受力门槛，不是完整的支撑多边形或动态稳定性判定；不规定哪两轮支撑。

初次仿真与每次环境重置都留出 0.5 s 宽限，使用 episode 与全局仿真时间的较小值，避免训练启动时随机 episode 计数绕过宽限。

## 3. 落地结算的条件与公式

令 \(t_0\) 为轮子离地时刻，\(t_1\) 为其重新触地时刻。记录起步时的机身朝向，将起步平移命令转到世界 XY 平面，得到固定单位方向 \(\hat d_0\)。以下位移都投影到这个起步方向；这是对最长 0.6 s 的短步使用的局部近似，不是沿弯曲轨迹累计距离。

定义：

\[
T_i=t_1-t_0,
\]

\[
L_i^{world}=\max_{t\in[t_0,t_1]}
\left(z_i(t)-z_i(t_0)\right),
\]

\[
L_i^{rel}=\max_{t\in[t_0,t_1]}
\left[(z_i(t)-z_b(t))-(z_i(t_0)-z_b(t_0))\right],
\]

\[
D_i=\left(p_{i,xy}(t_1)-p_{i,xy}(t_0)\right)\cdot\hat d_0,
\qquad
D_b=\left(p_{b,xy}(t_1)-p_{b,xy}(t_0)\right)\cdot\hat d_0.
\]

其中 \(i\) 表示轮子，\(b\) 表示机身。\(L_i^{rel}\) 使用轮心与机身的世界竖直高度差，不是完整机身坐标变换，也不是地形法向净空。

只有在落地时满足全部条件，才接受该步：

| 检查 | 初始阈值 | 作用 |
| --- | ---: | --- |
| 完整离地时间 \(T_i\) | `0.1 ≤ T_i ≤ 0.6 s` | 排除很短接触抖动和长时间腾空 |
| 世界竖直抬升峰值 \(L_i^{world}\) | `≥ 0.04 m` | 轮子确实向上抬过 |
| 相对机身抬升峰值 \(L_i^{rel}\) | `≥ 0.02 m` | 排除仅由全身升高造成的抬升 |
| 轮子沿指令的净位移 \(D_i\) | `≥ 0.04 m` | 排除只上下摆腿 |
| 机身沿指令的净位移 \(D_b\) | `≥ 0.01 m` | 要求这一步同时具有任务进展 |
| 支撑轮数 | 全程 `≥ 2` | 排除不满足支撑条件的候选 |
| 承重接触 | 接触有效且世界 Z 净力 `> 5 N` | 支撑计数及落地轮均要求，排除只有立面碰撞 |
| 命令连续性 | 平移大小 `≥ 0.1 m/s`，方向余弦 `≥ 0.866` | 不奖励指令变化前后拼接出的位移 |

对接受的轮子计算有界分数：

\[
S_i=\operatorname{clip}\left(\frac{L_i^{rel}}{0.06},0,1\right)
\operatorname{clip}\left(\frac{D_b}{0.04},0,1\right).
\]

总项对四轮平均，仅在该轮完成有效落地的那一帧计入：

\[
r_{step}=5.0\times0.02\times
\frac14\sum_{i=1}^{4}\mathbf1_{\text{有效落地}_i}S_i.
\]

权重为 `+5.0`，策略步长为 `0.02 s`。**单个轮子完成一步的最大新增奖励是 `0.025`**。离地越久不会持续增加奖励，也不会逐帧累加高度奖励；超过 0.6 s 的候选不获此奖励。初始阈值和权重均需通过训练与消融确定是否合适。

同时检查世界抬升与相对机身抬升，是为了降低跳起整台机器人或只抬高机身获得分数的机会。机身下沉可能增加相对高度，因此仍要求轮子具有真实的世界向上位移。

## 4. 如何理解侧移、越障与下楼的适用性

侧移时，轮子的滚动方向通常无法直接产生所需侧向运动，候选条件允许在有侧移命令时对实际抬轮、侧向落脚且带来机身进展的动作给分。该设计不强制左右对称、固定对角步态或某个步频，也不禁止正常轮驱动。

在台阶和障碍类别中，前进或后退命令也可启用候选。所用高度和位置都相对于本次起步，允许上楼和下楼有不同落脚高度，不使用固定世界高度目标。但是，**该项不测量轮子是否越过障碍物上沿，也不证明存在足够碰撞净空**。有向上抬轮再降低落脚位置的下楼步可以获奖；单纯控制腿伸展、平稳向下落脚而没有满足抬升条件时，没有这项加分，也不会由这项额外扣分。

这不是已经解决卡住脱困的奖励。如果机器人必须先在机身几乎不移动时重新安放第一只脚，由于 \(D_b\) 不足，那一步不会获奖。这样的门槛能减少原地摆腿刷分，也可能降低困难情形下的学习信号，需要通过具体失败轨迹审核。

位移和方向条件仍无法排除所有取巧行为。例如，在较长时间内反复前后移动，仍可能多次产生满足局部条件的正向步。较小的事件奖励、原有速度跟踪项和分项运动评估用于降低并发现这种问题，不能视为形式保证。稀疏落地奖励也不保证策略会自主发现所需步态。

默认 X5 已有侧移命令、台阶和障碍地形，本实验不改变其采样比例。应首先观察这些类别实际占比、候选数量和通过比例；没有足够侧移或困难地形样本时，不应直接放宽奖励条件来制造更多奖励。

## 5. 对照训练与单项消融

首先比较 Rewards v1 与 Steps v1，以隔离新增 `completed_step` 的影响。以下命令使用同样的环境数、迭代数与种子，显式指向本地源码；分别执行，避免同时占用同一 GPU。

```bash
cd /home/xgy/重要文档/legbot_lab
X5_STEPS_PY=/home/xgy/miniforge3/envs/x3w_isaaclab/bin/python

PYTHONPATH="$PWD/source/rsl_rl:$PWD/source/robot_lab" "$X5_STEPS_PY" scripts/rsl_rl/train.py \
  --task RobotLab-X5-MoECTS-Rewards-v1 --num_envs 256 --max_iterations 5000 \
  --seed 42 --run_name steps_control_s42 --headless agent.resume=false

PYTHONPATH="$PWD/source/rsl_rl:$PWD/source/robot_lab" "$X5_STEPS_PY" scripts/rsl_rl/train.py \
  --task RobotLab-X5-MoECTS-Steps-v1 --num_envs 256 --max_iterations 5000 \
  --seed 42 --run_name completed_steps_s42 --headless agent.resume=false

# 关闭唯一新增项，在奖励数学定义上恢复 Rewards v1。
PYTHONPATH="$PWD/source/rsl_rl:$PWD/source/robot_lab" "$X5_STEPS_PY" scripts/rsl_rl/train.py \
  --task RobotLab-X5-MoECTS-Steps-v1 --num_envs 256 --max_iterations 5000 \
  --seed 42 --run_name completed_steps_off_s42 --headless agent.resume=false \
  env.rewards.completed_step.weight=0.0
```

256 环境、5000 轮用于初步对照，不是最终训练规模。不同奖励实验从新训练开始，不直接跨奖励定义 resume；网络尺寸一致不能保证价值函数和优化器历史仍适用于另一奖励目标。单一种子结果还需通过多个种子复查。

如果要进一步评估相对于最初 X5 的总体变化，再加入原始 `RobotLab-X5-MoECTS-v0` 对照，命令见 [Rewards v1 文档](x5_rewards_v1.md)。不能将原 X5 与 Steps v1 的全部差异都归因于 `completed_step`。

## 6. 验证与观察指标

必须区分奖励公式检查、Isaac 短步运行和真正的训练运动表现。运行检查命令：

```bash
cd /home/xgy/重要文档/legbot_lab
PYTHONPATH="$PWD/source/rsl_rl:$PWD/source/robot_lab" \
  /home/xgy/miniforge3/envs/x3w_isaaclab/bin/python \
  scripts/tools/check_x5_rewards_v1.py \
  --profile steps --headless --num-envs 4 --steps 64 \
  --output outputs/x5_rewards_v1/steps_smoke.json
```

2026-09-10：新增 Rewards/Steps 纯张量与合成轨迹测试 **63 passed**，包括横移、上/下台阶、坐标平移、方向改变、重复读取、部分 reset、0.6 s 浮点边界、随机初始 episode 计数与立面接触。扩大到原有资产/控制/奖励契约的检查为 **96 passed、1 failed**；唯一失败是本轮开始前已存在的力矩权重旧断言，详见 Rewards 文档。

两种任务均已完成 4 环境、64 步零动作 Isaac 检查，实际物理步长 0.0025 s、策略周期 0.02 s，奖励和观测有限，没有终止。零动作检查中 `completed_step` 没有触发；正奖励轨迹来自 CPU 合成样本，不能冒充仿真迈步成功。未执行优化器更新，未验证侧移、楼梯或实机表现。完整报告写入上面的输出路径。

脚本中的 `sampled_completed_step_events` 为返回时的诊断计数；恰好发生在 timeout 自动 reset 的事件可能已被清空。它不适合作为长期训练的完整事件统计，训练评估需要独立累计事件并保留 reset 帧。

后续至少观察以下内容：

1. **奖励门控**：侧移/台阶/障碍候选数量、触地次数、接受数量，以及命令变化、时间、抬升、位移、支撑条件各自的拒绝比例。
2. **实际运动**：侧移速度误差与净位移，上台阶、下台阶和障碍通过率，跌倒率、episode 长度，以及纯前进滚动和原地转向是否退化。
3. **迈步质量**：轮心与机身轨迹、抬升峰值、离地时长、支撑轮数、落脚位移和机身进展，结合碰撞画面判断是否真正越过障碍。
4. **奖励量级**：事件频率乘单次奖励后的平均贡献，与原速度跟踪和轮速惩罚比较。不要只比较 total reward 或单步最高值。
5. **取巧与稀疏性**：原地摆腿、短周期接触抖动、重复前后移动、平台上无必要迈步，以及长期没有有效事件的情况。
6. **消融结果**：关闭 `completed_step` 后的运动差异；训练条件一致时，再对具体阈值做单因素调整。

如果统计上没有有效事件，应先检查任务采样、接触状态和轨迹是否满足定义，再判断奖励是否过于稀疏。人工构造事件获得正确分数只说明实现符合公式，不说明策略已经学会该动作。
