# X5 Rewards v1：参考 Flamingo 的独立奖励实验

这项实验为 X5 增加零命令轮速与离地空转约束，并把原有的腿部默认姿态惩罚改为允许小范围自由运动的舒适区间。所有新增权重都是待验证的初值，不代表已经提升了跟踪、爬阶或实机表现。

| 配置 | 原 X5 基线 | 本实验 |
| --- | --- | --- |
| 任务 ID | `RobotLab-X5-MoECTS-v0` | `RobotLab-X5-MoECTS-Rewards-v1` |
| 配置目录 | `source/robot_lab/robot_lab/tasks/x5/` | `source/robot_lab/robot_lab/tasks/x5_rewards_v1/` |
| 实验目录名 | `x5_moe_cts` | `x5_moe_cts_rewards_v1` |
| 活动奖励数量 | 14 | 16 |

实验继承当前 X5 基线；算法、网络、控制、观测、地形、速度命令、课程、随机化和终止条件保持一致。原 14 项奖励权重保持一致，其中两项姿态奖励更换公式，另外新增两项轮速惩罚。原 X5、Legbot 和 Go2 的配置不因该实验改变。

## 1. 参考来源及适用范围

参考项目为 [jaykorea/Isaac-RL-Two-wheel-Legged-Bot](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/tree/flamingo_isaac_lab_envs)，本次审查固定在 `flamingo_isaac_lab_envs` 分支的提交：

```text
61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02
```

[README 的训练命令](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/README.md#L85-L94) 使用 `Isaac-Velocity-Flat-Flamingo-v1`。[任务注册](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/__init__.py#L18-L110) 区分 v1 的 stand_drive、v2 的 track_z 和 Rough v3 的 stand_walk，不能将它们视为同一套训练奖励。

本实验借鉴两种设计思路，并针对 X5 重新实现，没有复制或导入外部项目代码：

- **关节舒适区间**：Flamingo 的 Flat v1 确实启用了肩、腿关节的区间奖励，区间内没有惩罚，区间外惩罚到最近边界的距离。本实验用 X5 自己的默认关节角定义区间。[活动配置](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/flat_env/stand_drive/flat_env_stand_drive_cfg.py#L35-L59)
- **零命令时约束轮速**：参考项目中存在 `stand_still` 函数，但审查的活动训练配置没有启用它。本实验采用阈值与持续时间判断，不能将其称为参考项目已经验证的有效奖励。[函数实现](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/lab/flamingo/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L263-L290)

**离地空转项是针对 X5 新增的设计**，参考项目没有对应的轮子离地旋转奖励。

## 2. 四项具体变化

以下公式先给出未乘权重和策略步长的非负代价。奖励管理器最终累计：

\[
r_k = w_k c_k\Delta t,\qquad \Delta t=0.02\ \mathrm{s}.
\]

其中 \([x]_+=\max(x,0)\)，轮半径 \(R=0.1005\ \mathrm{m}\)，\(q_j^0\) 是当前 X5 配置的默认关节角，\(\dot q_i\) 是实际轮关节速度。轮速换算 \(R|\dot q_i|\) 的单位是 m/s，它不是按电机能力上限归一化。

| 奖励项 | 改动 | 权重 | 初始参数 |
| --- | --- | ---: | --- |
| `hip_pos_penalty_l1` | HAA 偏差增加舒适区间 | `-0.05`，沿用 | 每关节 ±0.08 rad |
| `joint_pos_penalty_l1` | HFE/KFE 偏差增加舒适区间 | `-0.01`，沿用 | 每关节 ±0.15 rad |
| `stand_still_wheel_l2` | 新增零命令轮速惩罚 | `-0.1` | 轮缘速度容差 0.05 m/s |
| `wheel_air_spin_l2` | 新增持续离地空转惩罚 | `-0.02` | 轮缘速度容差 0.5 m/s |

### 2.1 腿部舒适区间

\[
c_{\mathrm{HAA}}=\sum_{j\in\mathrm{HAA}}
\left[|q_j-q_j^0|-0.08\right]_+,
\]

\[
c_{\mathrm{HFE/KFE}}=\sum_{j\in\mathrm{HFE,KFE}}
\left[|q_j-q_j^0|-0.15\right]_+.
\]

原来偏离默认姿态即受罚；现在约 ±4.6° 的 HAA 偏差和 ±8.6° 的 HFE/KFE 偏差不受这两项惩罚。超过容差后，仍保持原来的 L1 斜率和权重。关节限位惩罚及控制目标限位继续生效。

该变化意在减少小幅调整支撑姿态的成本，为侧向移动和不平地形上的腿部调整留下空间。它不指定步态，也不要求左右腿角度相同。舒适区内的梯度变为零，可能使站姿更自由；是否增加无效摆动要用实际轨迹判断。

### 2.2 持续零命令时的轮速

定义静止命令条件：

\[
z(t)=\mathbf 1\{\sqrt{v_{x,cmd}^2+v_{y,cmd}^2}\le0.05
\ \land\ |\omega_{z,cmd}|\le0.1\}.
\]

仅当条件连续满足至少 0.3 s，且 episode 与全局仿真计数均已经过至少 0.5 s 时，静止门控 \(s(t)\) 才有效。全局计数条件防止 RSL-RL 启动时随机化 `episode_length_buf` 绕过首次宽限；后续局部 reset 由 episode 计数保护。条件失效或环境重置时重新计时。

\[
c_{\mathrm{stop}}=s(t)\frac14\sum_{i=1}^{4}
\left[R|\dot q_i|-0.05\right]_+^2.
\]

该项约束的是零命令下多余的轮子旋转，不要求车体回到出生位置，也不比较左右轮速。纯转向命令超过 yaw 阈值时不会触发。0.3 s 的持续时间给指令切换后的制动留下时间，0.5 s 的重置宽限避免直接评价初始化瞬态。

本工程中，部分“站立”命令处理只把 XY 命令归零而保留 yaw。因此，XY 为零并不必然满足这里的静止条件，实际启用比例可能很低。应记录门控启用比例和满足条件时的轮速；若长期没有样本，应先解释命令分布，再单独讨论是否调整命令采样，不能由零奖励值推断策略已经稳定。

### 2.3 持续离地时的空转

对每个轮子独立判断，\(t_i^{air}\) 为接触传感器的当前连续离地时间，\(\mathbf F_{i,h}\) 为其接触力历史：

\[
a_i(t)=\mathbf1\{t_i^{air}>0.15\ \mathrm{s}\}
\mathbf1\{\max_h\|\mathbf F_{i,h}\|\le1\ \mathrm{N}\}
\mathbf1\{\min(t_{episode},t_{global})\ge0.5\ \mathrm{s}\}.
\]

\[
c_{\mathrm{air}}=\frac14\sum_{i=1}^{4}a_i(t)
\left[R|\dot q_i|-0.5\right]_+^2.
\]

这里对四个轮子取平均，不按当时离地轮数重新归一化。没有接触的轮子可以保持一定转速，持续离地且超过轮缘速度容差后才受罚；正常接触滚动和短暂离地不会触发该项。接触历史尚有力反馈时也不触发，避免接触状态切换时立即施加约束。

该项没有奖励“四轮都接触”，没有规定单腿或对角步态。它仍可能使长时间摆腿时的轮速预同步变慢，所以权重较小，并提供单独关闭它的消融实验。当前公式使用实际关节速度，不使用策略输出，也不需要假设左右轮关节正方向一致。

## 3. 为什么本轮不加入其他项

| 候选设计 | 本轮处理及原因 |
| --- | --- |
| `feet_slide` | 不移植。参考函数惩罚接触部件中心的 XY 速度，直接用于轮子会惩罚正常滚动；真正滑移需要接触点运动学、轮轴和地形法向等信息。该函数也没有在参考活动配置中启用。[实现](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/lab/flamingo/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L106-L113) |
| 强制四轮持续接触 | 不加入。爬阶和侧向步进需要局部离地，不能用接触数量代替任务进展。 |
| 左右关节或轮速对称 | 不加入。参考 `joint_align_l1` 只比较两个关节；X5 的转弯、侧移和不同高度支撑需要独立关节运动。[实现](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/lab/flamingo/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L293-L320) |
| 强姿态水平约束 | 不加入。参考 v1 的 `flat_orientation_l2` 权重为 -5；X5 的斜坡和越障可能需要姿态适应，本轮先减少同时变化的目标。[配置](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/flat_env/stand_drive/flat_env_stand_drive_cfg.py#L96-L116) |
| 卡住时奖励抬腿 | 不照抄。参考实现将含 yaw 的三维命令与三维线速度相减，而且持续计数没有单独的环境重置处理；需要先重定义沿指令方向的进度和按秒计时。[实现](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/lab/flamingo/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L116-L187) |
| 轮心相对高度区间 | 不照抄。参考高度区间函数在区间外选反上下边界；平均轮心高度也会随摆腿改变。本轮沿用现有 X5 高度项。[实现](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/lab/flamingo/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L378-L429) |
| 力矩能力归一化、饱和、真终止惩罚 | 本轮不加入。它们各自会改变奖励平衡，适合单独对照；参考饱和函数实际衡量 `applied_torque` 与 `computed_torque` 的差值。[实现](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02/modified_source/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/mdp/rewards.py#L264-L283) |

## 4. 对照训练与消融命令

在项目根目录执行，显式使用本地 `robot_lab`、`rsl_rl` 源码和指定 Python。以下是实验命令，文档本身不表示已经启动训练。

```bash
cd /home/xgy/重要文档/legbot_lab
X5_REWARD_PY=/home/xgy/miniforge3/envs/x3w_isaaclab/bin/python

PYTHONPATH="$PWD/source/rsl_rl:$PWD/source/robot_lab" "$X5_REWARD_PY" scripts/rsl_rl/train.py \
  --task RobotLab-X5-MoECTS-v0 --num_envs 256 --max_iterations 5000 \
  --seed 42 --run_name rewards_baseline_s42 --headless agent.resume=false

PYTHONPATH="$PWD/source/rsl_rl:$PWD/source/robot_lab" "$X5_REWARD_PY" scripts/rsl_rl/train.py \
  --task RobotLab-X5-MoECTS-Rewards-v1 --num_envs 256 --max_iterations 5000 \
  --seed 42 --run_name rewards_v1_s42 --headless agent.resume=false
```

上面两次训练应分别启动、完成后比较，避免在同一 GPU 上相互争抢资源。256 环境和 5000 轮是较小规模的初步对照，不代表最终训练规模。同种子可减少一部分比较噪声，后续仍需多个种子验证。

单项消融均从新训练开始，沿用相同环境数、迭代数与种子：

```bash
cd /home/xgy/重要文档/legbot_lab
X5_REWARD_PY=/home/xgy/miniforge3/envs/x3w_isaaclab/bin/python

# 关闭零命令轮速项。
PYTHONPATH="$PWD/source/rsl_rl:$PWD/source/robot_lab" "$X5_REWARD_PY" scripts/rsl_rl/train.py \
  --task RobotLab-X5-MoECTS-Rewards-v1 --num_envs 256 --max_iterations 5000 \
  --seed 42 --run_name rewards_v1_no_stop_s42 --headless agent.resume=false \
  env.rewards.stand_still_wheel_l2.weight=0.0

# 关闭离地空转项。
PYTHONPATH="$PWD/source/rsl_rl:$PWD/source/robot_lab" "$X5_REWARD_PY" scripts/rsl_rl/train.py \
  --task RobotLab-X5-MoECTS-Rewards-v1 --num_envs 256 --max_iterations 5000 \
  --seed 42 --run_name rewards_v1_no_air_spin_s42 --headless agent.resume=false \
  env.rewards.wheel_air_spin_l2.weight=0.0

# 将两个姿态舒适区宽度设为零，数学上恢复原来的默认姿态 L1 代价。
PYTHONPATH="$PWD/source/rsl_rl:$PWD/source/robot_lab" "$X5_REWARD_PY" scripts/rsl_rl/train.py \
  --task RobotLab-X5-MoECTS-Rewards-v1 --num_envs 256 --max_iterations 5000 \
  --seed 42 --run_name rewards_v1_original_pose_s42 --headless agent.resume=false \
  env.rewards.hip_pos_penalty_l1.params.deadband=0.0 \
  env.rewards.joint_pos_penalty_l1.params.deadband=0.0
```

不同奖励定义之间不直接 resume。基线与新奖励应从各自的新实验开始；已有 checkpoint 的兼容尺寸不意味着其价值函数、优化器状态和训练历史适合新的奖励目标。若后续需要微调旧模型，应另立实验，明确加载范围。

## 5. 观察哪些证据

不能只比较总 reward：舒适区本身就减少了负奖励，即使运动表现不变，总 reward 也可能上升。至少检查：

1. 相同指令分布下的 XY/yaw 跟踪误差、episode 长度、跌倒率，以及平地、侧移、转弯、台阶、斜坡的分项表现。
2. 两项新惩罚的原始代价、加权贡献和门控启用比例；分别观察零命令样本和持续离地样本。均值很小可能来自很少启用。
3. 满足静止门控时的轮缘速度、车体漂移；满足离地门控时的轮速，以及触地前后的轮速变化。
4. 姿态舒适区占比、腿部摆动幅度、动作变化、接触情况，确认减少姿态约束后没有形成无效动作。
5. 对比两项新增惩罚与速度跟踪奖励的量级。若明显压制跟踪或越障，先通过单项关闭定位原因，再调权重。

本轮不改变原始力矩、功率和关节加速度惩罚的能力归一化方式，X5 当前力矩权重仍为 `-4e-5`。因此这些项的旧有量级问题需要独立统计，不能认为新增轮速惩罚解决了它们。

## 6. 验证状态

运行检查命令为：

```bash
cd /home/xgy/重要文档/legbot_lab
PYTHONPATH="$PWD/source/rsl_rl:$PWD/source/robot_lab" \
  /home/xgy/miniforge3/envs/x3w_isaaclab/bin/python \
  scripts/tools/check_x5_rewards_v1.py \
  --headless --num-envs 4 --steps 64 \
  --output outputs/x5_rewards_v1/smoke.json
```

2026-09-10 检查结果：新增 Rewards/Steps 公式与轨迹测试 **63 passed**；两种任务均完成 4 环境、64 步零动作 Isaac 检查，奖励与 530/53/291 维观测均有限，原 X5 配置不受新配置实例化影响，算法配置仅实验名不同。启动宽限、承重接触修订后的复核报告见 `outputs/x5_rewards_v1/`，迈步细节见 [Steps 文档](x5_steps_v1.md)。

没有执行优化器更新或训练对照，尚未证明跟踪、行走或实机效果。零动作 smoke 中真正的零指令门控启用率为 0，因此另用真实 manager 配合合成命令检查了停止与纯转向分支；这不是策略静止效果验证。

扩大检查包含原有资产/控制/奖励契约时，仍有一项本轮开始前已存在的失败：`tests/test_x5_reward_contract.py` 要求 X5 力矩权重与 Legbot 同为 `-1e-4`，当前基线实际为 `-4e-5`。本实验保留该基线权重，未通过修改基线或旧断言隐藏这项不一致。
