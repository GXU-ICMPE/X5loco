# RobotLab-X5-MoECTS-v0 迁移检查点

## 范围与当前状态

目标是在 `WF-CTS-MOE` 分支新增独立的 X5 MoE-CTS 任务，保留现有
`RobotLab-Legbot-v0`、Go2 任务及算法行为。每一步重大修改前先与用户确认。

2026-09-09：用户已选定已有 X5 迁移目录中的 115.780333 kg 模型。
第一阶段已导入 `resources/x5/`：URDF、实际引用的 13 个网格、来源校验清单。
此后用户确认继续下一阶段，现已注册 `RobotLab-X5-MoECTS-v0` 并接入执行器。
用户随后指定以 `RobotLab-Legbot-v0` 为正式环境基准。当前 X5 配置直接继承
其地形、命令、课程、随机化、终止条件和 MoE-CTS 算法，仅适配机器人及物理量纲。
平地、零命令的机械检查已独立到 `tasks/x5/validation_cfg.py`，不是注册任务默认值。
**本轮及前序检查均未进行 PPO/Student 优化器更新，也没有启动小规模训练。**

本阶段验证：`scripts/tools/validate_x5_asset.py` 返回
`PASS_OFFLINE_ASSET_ONLY`，`tests/test_x5_asset.py` 的 7 项测试全部通过。
使用 `/home/xgy/miniforge3/envs/x3w_isaaclab/bin/python` 执行，未改动其环境。

模型基准：

`/home/xgy/重要文档/LocoWheeledLegged_X5_migration/LocoWheeledLegged/locowheeledlegged/assets/x5/urdf/x5.urdf`

不要与下载目录中的另一个 X5 混用：那版不仅总质量为 113.796477 kg，
腿关节限速为 4.5/4.5/6 rad/s，大腿上限为 2.09 rad，而且小腿惯量、
部分关节原点和碰撞体位置也不同。当前选定版对应 6/6/8 rad/s、
大腿上限 2.618 rad；模型物理参数不从 Go2W 借用。

## Go2W 可以参考什么

仓库确实有 `resources/go2w_description/urdf/go2w_description.urdf`，
其引用的 8 个独立网格文件也齐全，现已核实。
但当前源码没有连接这个资产的 Go2W 配置和训练任务注册；不能把“资产存在”
当成“已有可直接运行的 Go2W MoE-CTS 训练配置”。

Go2W 资产可用于对照 12 个腿关节加 4 个主动轮的结构；腿位置/轮速度控制、
轮角位置不进入本体观测则沿用本分支的轮足任务设计。
Go2W 质量约 19.523 kg，X5 约 115.78 kg，
关节名称、几何、限位与控制参数不同；Go2W 的滚轮碰撞体是网格，X5 是圆柱。
不把 Go2W 的质量、增益、限速或训练权重直接用于 X5。

## 当前 8 个专家到底学什么

**没有预定义的八项任务，也没有“某编号专家负责某种地形”的配置。**

当前数据流是：

```text
本体观测历史 h（530 维）
    ├─ 共享前部网络 → 8 个专家输出 e_i(h)（每个 32 维）
    └─ 门控网络 → softmax 权重 w_i(h)（8 个权重，总和为 1）
                         ↓
             z_student = L2Normalize(Σ w_i e_i)
                         ↓ 与当前本体观测（53 维）拼接
                  一个共享 Actor → 16 维动作

特权观测 → Teacher Encoder → z_teacher
                             ↓
             用隐变量均方误差训练 Student MoE
```

依据：`source/rsl_rl/rsl_rl/networks/moe.py` 中的 `Experts` / `MoE`，
`modules/actor_critic_moe_cts.py` 中的 `StudentMoEEncoder` / `act`，
`algorithms/moe_cts.py` 中的学生更新循环。

- 每个专家都是隐变量分支，不是独立 Actor；所有分支都计算并参与软加权，
  不是每次只选一个专家。前部网络是共享的。
- Actor、Teacher 和 Critic 主要通过 PPO 学习运动任务；Student 的专家和
  门控通过蒸馏学习从本体历史近似 Teacher 的特权表示，学生编码器不直接
  接收这里的 PPO 梯度。
- 学生损失为隐变量 MSE 加负载均衡项。均衡项约束的是批次平均门控权重
  接近 `1/8`，不是专家任务标签，也不保证八个专家有不同技能。
  例如每个样本都均匀使用八个专家也能使这个均衡项为零。
- 当前活动地形恰好有八类，不代表一对一绑定。专家编号在不同训练种子之间
  没有固定语义；没有模型分析结果时，不能命名为“滚动专家”“上楼专家”等。

后续可增加每个专家的使用率、门控熵、按地形/速度/接触状态分组的统计，
再结合屏蔽某专家后的性能变化判断分工。仅凭使用率高低不能证明该专家
负责某个技能。若要人为指定八项技能，需要另行设计任务分配或监督机制，
属于算法改造，不在此次默认的机器人接入范围内。

## 第二阶段：控制与任务接口（已接入）

已新增 `source/robot_lab/robot_lab/assets/x5.py` 和独立
`source/robot_lab/robot_lab/tasks/x5/`，注册 `RobotLab-X5-MoECTS-v0`，
实验目录用 `x5_moe_cts`。不加载 Go2W/Legbot checkpoint，不覆盖原任务。

下表记录机械接入阶段；第三阶段的正式任务差异见后文。

| 项目 | 机械接入值/行为 | 依据与边界 |
|---|---|---|
| 动作顺序 | RF、LF、RH、LH，每腿 HAA/HFE/KFE 共 12 维，最后同序 4 个 WHEEL | 沿用选定 X5 项目的显式顺序；不依赖 USD 自动排序 |
| 腿控制 | `q_target = q_default + 0.2 × action`；PD 800/20 | 选定迁移项目的仿真起点；不是实测硬件参数 |
| 腿目标限幅 | 右 HAA [-0.79, 0.64]；左 HAA [-0.64, 0.79]；HFE [-1.60, 2.39]；KFE [-2.32, -0.52] rad | 位于所选 URDF 的软限位内 |
| 轮控制 | `ω_target = action / 0.1005`；限幅 ±24 rad/s；速度增益 3 | 即动作按轮缘速度 m/s 解释；最终上限约 2.412 m/s，不等于车体保证速度 |
| 执行器上限 | HAA/HFE/KFE：300/250/250 Nm、6/6/8 rad/s；轮：30 Nm、30 rad/s | 来自选定 URDF；执行器与 PhysX 两层都显式设置 |
| 初始姿态 | 每腿 [0, 0.75, -1.5] rad；生成高度 0.5875 m | 几何站高约 0.56753 m，预留约 2 cm |
| 仿真/策略频率 | 400 Hz / 50 Hz（dt 0.0025、decimation 8） | 沿用 X5 起点，策略频率与本分支一致 |
| 初始验证 | 平地、零动作站立、单关节和轮速符号响应；关闭随机推力、动作延迟和观测噪声 | 验证时关闭；粗糙地形及训练随机化参数另行确认 |
| 观测/网络 | 保留 53 维单帧、10 帧历史共 530 维、16 维动作、8 专家 CTS | 不照搬另一训练框架的历史帧数；Critic 运行时确认 291 维 |

第一轮机械验证使用无延迟的显式 PD，不引入原 X5 项目的低通动作类和
0–75 ms 随机执行延迟；在确认基础机械响应之后，再单独讨论这些机制。
第二阶段默认环境数为 4，便于匹配 CTS 的 3 Teacher / 1 Student 分组；纯机械验证
脚本可以运行 1 个环境，因为它不创建 CTS 优化器或采样分组。
当时 Runner 设为 1 次迭代且从未执行。第三阶段已恢复 Legbot 的配置值：
默认 16384 环境、300000 迭代、每 500 迭代保存；恢复配置不代表启动训练。

### 导入器兼容处理

本机 Isaac Lab 的 URDF 导入器会将碰撞子树实例化，导致普通
`collision_props` 覆盖无法写入；设置通用 `make_instanceable=False` 也未生效。
X5 使用局部 `spawn_x5_from_urdf` 包装器：先生成资产，再仅将各刚体的
`collisions` 子树解除实例化并写入接触属性，最后供环境克隆使用。
视觉网格继续保留实例化，不改 URDF、网格、缓存 USD 或 Isaac Lab 公共源码。

运行时检查所有 29 个碰撞形状的 `contact_offset=0.002`、`rest_offset=0`，
确认 4 个半径 0.1005 m 的滚轮圆柱、自碰撞开启、隐式 PhysX 驱动增益为零。
显式 PD 和 PhysX 两层关节限速/限力矩均对齐选定模型。

### 已获得的仿真证据

验证环境：Isaac Sim 5.1.0.0、Isaac Lab 0.54.4（本机 `/home/xgy/IsaacLab`），
PyTorch 2.7.0+cu128、RTX 5050 Laptop 8 GB；使用本仓库 `source/rsl_rl`，
没有修改现有 Python 环境的安装。

4 个环境均完成 1000 个策略步、20 秒零动作站立：

- 无终止/超时/重置，观测、奖励、力矩保持有限值，力矩不越界。
- 最低机身高度约 0.54301 m，最终约 0.55021 m。
- 最大倾角约 0.01211 rad（0.69°），最大腿角误差约 0.12976 rad。
- 最大水平位移约 0.05743 m，最后 2 秒四轮接触比例均为 100%。
- 导入后 17 个刚体质量、质心和完整惯量逐项对照 URDF 固定关节合并结果；
  最大质量误差约 3.82e-6 kg、质心误差 1.27e-8 m、惯量误差 8.90e-7 kg·m²。
- 运行时 `policy=(4,530)`、`single_obs=(4,53)`、`critic=(4,291)`；
  Teacher/Student/Actor/Critic 前向计算通过，未将随机网络动作施加到机器人。

原始 JSON：`outputs/x5_bringup/stance_4env.json`，其中记录资产/环境配置校验和。
这些证明机械与网络接口可用，不证明策略会走路或增益已适合真机。

第二阶段配置的单环境 20 秒站立也通过，记录在 `outputs/x5_bringup/stance_1env.json`。
16 个关节的正负方向检查全部通过，记录在 `outputs/x5_bringup/axes_1env.json`：
腿目标偏移 ±0.05 rad，0.4 秒后响应约 ±0.0503–0.0511 rad；轮目标 ±2 rad/s，
0.4 秒后采样响应约 +1.428 / -1.432 rad/s。这里仅验收方向，不据此声称轮速
已经准确稳态跟踪；后续训练前仍可单独检查轮速动态响应。

离线回归：14 项 X5 资产/控制接口测试加 18 项现有 RSL-RL 工具测试，
共 **32 passed**。公共任务、共享 MoE-CTS 算法及安装环境均未改动。

### 复现命令

从仓库根目录运行，无需安装或替换其他项目的包：

```bash
X5_PY=/home/xgy/miniforge3/envs/x3w_isaaclab/bin/python
"$X5_PY" scripts/tools/validate_x5_asset.py
PYTHONPATH="$PWD/source/rsl_rl:$PWD/source/robot_lab" "$X5_PY" -m pytest tests source/rsl_rl/tests -q
"$X5_PY" scripts/tools/check_x5_sim.py --headless --config-only
"$X5_PY" scripts/tools/check_x5_sim.py --headless --mode stance --num-envs 4 --steps 1000 --output outputs/x5_bringup/stance_4env.json
"$X5_PY" scripts/tools/check_x5_sim.py --headless --mode axes --output outputs/x5_bringup/axes_1env.json
```

轴向检查将底座固定在空中并关闭重力，依次对 16 个关节施加正负小目标；
它只验证控制方向，不是地面行走或轮速稳态精度验收。输出必须包含明确的
`PASS_*` 状态；仅看到 Isaac 进程退出或环境构造完成不算通过。

第二阶段发现四轮站立接触力约 261–308 N，高于继承的 Critic/Teacher
接触观测截断值 100 N。这一缺口及身高、力矩量纲问题在下面的第三阶段处理。

## 第三阶段：严格按 Legbot 基准适配（不训练）

### 保持不变

`X5EnvCfg(LegbotEnvCfg)`、`X5SceneCfg(LegbotSceneCfg)` 和
`X5MoECTSRunnerCfg(LegbotMoECTSRunnerCfg)` 直接继承，不另写一套训练任务。
`config_audit.py` 对实例化后的完整配置做差异白名单检查，越界即失败。

- 同一套 10 行 × 20 列地形、八类活动地形比例、几何难度、初始等级与地形课程。
- 同一个 `Go2RLGymCommand`：重采样、命令范围、按地形限速、命令课程均不修改。
- 同样的 9 项事件：质量、质心、材质、增益、零位、初始关节/根状态和周期推力。
  包括 base 质量加减 1 kg、其他刚体乘 0.9–1.1、COM ±3 cm、增益乘 0.9–1.1、
  位置零位 ±0.035 rad 等。没有用机器人总质量倍数扩大这些随机化。
- 本体噪声与原幅度恢复，53D 单帧、530D 历史、291D 特权观测、16D 动作不变。
  本体量的缩放保持原值；物理范围变小并不自动意味着必须放大传感器噪声。
- 14 个奖励项、原标量权重及权重课程不变；力矩、功率、加速度惩罚已按用户要求
  恢复原版函数，不再做上限比例缩放（见下文撤回记录）。
- 25 秒 episode、base 接触终止 10 N、腿部非期望接触 5 N 不变。
  后两个数值是“检测是否碰地”的阈值，不是承载上限，不能与 Teacher 截断混为一谈。
- MoE 8 专家、32D latent、共享 Actor/Critic、CTS 比例、PPO/蒸馏全部超参数不变。
  Runner 唯一差异是实验名 `x5_moe_cts`；不加载其他机器人的 checkpoint。

### X5 差异与依据

| 项目 | Legbot → X5 | 原因 |
|---|---|---|
| 资产、实体选择、默认姿态、PD、力矩/速度与动作限幅 | 使用第一、二阶段已核验的 X5 参数 | 真实结构、能力边界与关节命名不同 |
| 物理步进/decimation | 0.005 × 4 → 0.0025 × 8 | 保留已经检查过的 X5 PD 步进，策略周期仍为 20 ms |
| 接触传感器采样 | 保持 5 ms、3 帧历史 | 不因物理步进变快而改变 Teacher 接触历史的时间尺度 |
| Teacher 接触力 clip | ±100 → [0, 3000] N，scale 仍为 0.001 | 单轮静载约 284 N；正常四轮载荷不再全部变成 0.1 |
| Teacher 力矩 clip | ±100 → ±300 N·m，scale 仍为 0.01 | 覆盖 X5 最大关节力矩配置上限 |
| Teacher 加速度 clip | ±100 → ±30000 rad/s²，scale 仍为 0.0001 | ±30 rad/s 在 2.5 ms 反向的离散参考为 24000 rad/s²，预留余量 |
| 受载站高奖励目标 | 0.42 → 0.55 m | X5 先前实测受载站高约 0.5502 m；几何站高 0.5675 m 不是平衡点 |
| 高度扫描 offset | 默认 0.5 → 0.63 m | 补偿站高差 0.13 m，保持名义平地特征中心；扫描网格、clip、scale 不变 |

Teacher clip 是**特征边界，不是力/加速度的物理约束或真机安全限值**。
3000 N 约为名义单轮静载的十倍，是工程初值；剧烈碰撞可能仍截断。
不是为了让任何异常冲量都进入网络而无限提高上限。

### 当前奖励：撤回三项比例缩放（2026-09-09）

按用户明确要求，X5 的三项物理惩罚直接继承 Legbot 原函数和原权重，
仅将实体选择改为 X5 的 12 个腿关节和 4 个轮关节：

| 项目 | 当前加权公式（未乘共同的 0.02 s 策略步长） | 原函数 |
|---|---|---|
| 力矩 | `-1e-4 × Στ²` | `isaaclab.envs.mdp.rewards:joint_torques_l2` |
| 功率 | `-2e-5 × Σabs(τ × ω)` | `robot_lab.tasks.go2.mdp.rewards:joint_power` |
| 加速度 | `-1e-7 × Σα²` | `isaaclab.envs.mdp.rewards:joint_acc_l2` |

撤回原因：按配置能力上限归一化只是一种可选择的奖励设计，不是机器人迁移必需的
物理定律，尤其速度上限不能直接代表可实现的加速度。在缺少实际奖励贡献分解与
动力学依据时，将其作为 X5 必要适配实施并不充分。此前保留标量权重，并不代表
保持了有效惩罚强度；相同物理状态下，加速度实际被放大约 2.96–12.18 倍。

已删除运行目录中的 `scaled_rewards.py` 和三组比例常量，收紧 `config_audit.py`：
这三项现在只允许变更关节选择，不允许变更函数、权重或添加缩放参数。
旧缩放函数和本轮修改前文件备份在 `/tmp/legbot-x5-reward-rollback.5Rd5Kx/`，
它是本机临时备份，不是训练依赖；下节保留历史公式以便追溯旧日志。

本次验证：

- CPU 回归 **58 passed**。检查三项继承不改函数/权重，仅替换关节；
  对三项分别注入函数替换、权重修改和比例参数，共 9 个负例均被审计拒绝。
- `outputs/x5_alignment/config_original_penalties.json`：`PASS_CONFIG_ONLY`。
  在实际 Isaac Lab 配置中逐项确认与 Legbot 是同一个函数对象、同一权重、
  没有比例参数；真实奖励函数在 CPU 合成正/负功率与加速度样本上的公式检查通过。
- 本轮没有执行物理仿真步、采样训练、反向传播或优化器更新。
  观测截断/尺度、站高、资产/PD/限位、其余奖励、课程、地形、命令、随机化、
  重置与 MoECTS 训练逻辑均不在此次撤回范围。

运行中的进程不会自动加载源码修改，必须重新启动才会使用原版惩罚。
不要把旧缩放奖励日志与新运行的总奖励直接作学习效果对比。若需要干净的原版奖励
基线，应另开 run 且不使用 `--resume`；旧权重和价值估计是在旧奖励下学到的。
本次没有停止、重启训练，没有删除旧日志或 checkpoint。

### 已撤回的奖励量级换算规则（仅历史记录，不再生效）

原参数曾集中在 `tasks/x5/parameters.py`，函数曾在 `scaled_rewards.py`；未改共享奖励源码。
参照 `assets/legbot.py` 的执行器配置上限，而非 Go2W 的总质量。
当前仓库没有 Legbot 的私有 URDF，不能推测其质量再套统一倍数。

令每关节 `rτ = τ_limit_Legbot / τ_limit_X5`，
`rω = ω_limit_Legbot / ω_limit_X5`，维持原奖励权重：

- 力矩：`-1e-4 × Σ(rτ × τ)²`。
- 功率：`-2e-5 × Σ(rτ × rω × |τ × ω|)`。
- 加速度：`-1e-7 × Σ(rω × dω/dt)²`。

| 系数 | HAA | HFE | KFE | WHEEL |
|---|---:|---:|---:|---:|
| `rτ` | 0.4 | 0.48 | 0.70152 | 0.956 |
| `rω` | 3.35 | 3.35 | 1.72 | 3.490667 |

这套旧公式在占用相同配置力矩/速度比例，或相同归一化速度变化率时，代价等于 Legbot。
旧测试仅证明这个数学性质，没有证明该缩放适合实际训练，也不代表两台机器人
能耗/效率相同。相关测试已替换为原版函数继承及禁止缩放回归测试。
角度偏离、动作变化率、跟踪和接触奖励没有因 X5 较重而统一乘系数。

### 无训练验证与复现

正式任务配置和诊断配置分开：`--mode interface` 使用完整任务，只覆盖环境数、seed、
device；`--mode stance/axes` 显式使用平地、零命令、关闭噪声/随机化的诊断副本。
复制使用 `deepcopy`：本机 `configclass.copy()` 会重新执行父类初始化，可能短暂共享并
改写输入的嵌套 sim 配置。检查同时断言实际 `physics_dt=0.0025`、`step_dt=0.02`。
首次发现此问题的记录已标记 `INVALID_TIMING_DIAGNOSTIC_NOT_ACCEPTANCE`，不用于验收。

```bash
X5_PY=/home/xgy/miniforge3/envs/x3w_isaaclab/bin/python
PYTHONPATH="$PWD/source/rsl_rl:$PWD/source/robot_lab" "$X5_PY" -m pytest tests source/rsl_rl/tests -q
"$X5_PY" scripts/tools/check_x5_sim.py --headless --config-only --output outputs/x5_alignment/config.json
"$X5_PY" scripts/tools/check_x5_sim.py --headless --mode interface --num-envs 20 --steps 256 --output outputs/x5_alignment/interface_20env.json
"$X5_PY" scripts/tools/check_x5_sim.py --headless --mode stance --num-envs 4 --steps 1000 --output outputs/x5_alignment/stance_4env.json
```

上面的仿真检查命令没有训练入口、优化器、反向传播或 PPO 更新。20 个环境用于覆盖全部 20 个地形列，
不是训练规模选择。注册配置继承的 16384 环境也不意味着本机 8 GB 显存能够训练该规模。
检查输出的奖励项统计来自零动作、随机初始姿态和扰动，不能用于判断已学习策略的奖励平衡。

以下为撤回比例缩放前的历史结果（2026-09-09），其中的奖励统计不代表当前版本：

- 离线回归 **43 passed**：原 32 项加 11 项执行器上限来源、奖励量纲、关节重排、
  观测范围及差异报告测试。资产清单仍为 14 个文件/13 个网格校验通过。
- `outputs/x5_alignment/config.json`：完整配置差异检查通过；修改随机推力周期的
  负例会被拒绝；诊断副本不再改动正式配置的物理步长。
- `outputs/x5_alignment/interface_20env.json`：实际 400 Hz 物理/50 Hz 策略，
  20 环境覆盖全部八类活动地形，零动作执行 256 步（5.12 秒），无终止/超时重置，
  9 项事件和 3 项课程注册，非零命令与全部奖励/观测数值有限。
  实际 Teacher 的接触力分量逐步对照 `clip(raw_force) × 0.001` 通过；
  Teacher/Student/Actor/Critic 仅前向，不执行网络动作、不创建优化器。
- `outputs/x5_alignment/stance_4env.json`：独立机械诊断 4 环境 20 秒零动作站立通过。
  最低站高约 0.54246 m，最终约 0.55025 m；最大倾角约 0.01256 rad，
  最后 2 秒四轮接触比例均为 100%，末帧每轮约 260–308 N。

**未消除的初始化瞬态：** 完整任务 20 环境检查的接触力峰值约 36736 N，
发生在从 0 计数的第 1 个策略步；16 维接触特征（包含重复的历史最大值）
总截断比例约 0.07324%，初始化 0.5 秒之后的采样截断比例为 0。
这证明正常载荷已不再恒定饱和，但不证明随机初始化落地没有穿插/冲量问题。
尚未定位峰值的具体碰撞来源；下一项宜检查随机初始关节姿态、落地高度与接触几何。
本轮没有据此提高上限到 37 kN，也没有擅自缩小随机化分布或调整重置逻辑。
同理，零动作测试的加速度代价均值受瞬态影响，不能宣称量纲换算已得到最优权重。

后续单环境排查已复现随机初始姿态的轮体穿地及首个物理步的大冲量，详见
[初始化诊断报告](x5_reset_diagnosis.md)。默认姿态也确实存在约 2 cm 自由落地，
但不能用它解释全部极端峰值。该次仅诊断，尚未修改训练重置逻辑。

## MoECTS 构造参数兼容修复（2026-09-09）

本机新版 Isaac Lab 的 `RslRlPpoAlgorithmCfg` 会把继承字段 `optimizer="adam"` 和
`share_cnn_encoders=False` 一起传给 Runner。当前分支原版 `MoECTS.__init__` 不接收
这两个字段，因而在创建算法时抛出 `unexpected keyword argument 'optimizer'`。
此前的环境/网络前向检查没有创建算法，不能覆盖这一错误。

本次仅在 `source/rsl_rl/rsl_rl/algorithms/moe_cts.py` 添加 7 行：

- 将两个兼容参数追加到构造签名末尾，不改变原有位置参数顺序。
- 仅接受 `optimizer="adam"` 与 `share_cnn_encoders=False`；其他取值明确报错。
- 不添加通用 `**kwargs` 吞掉未知字段，不改两处原有 Adam 构造和任何训练语句。

`share_cnn_encoders` 是新版通用配置的 Actor/Critic CNN 共享选项，
不是 Teacher/Student 共享 Actor 的选项。当前策略未接入这种 CNN 共享机制；
Student MoE 内部的分组 `Conv1d` 专家计算也不受该字段影响，保持原实现。

一致性核对以当前分支提交 `e36128d438dccabb38763636db21459df13b5594` 为基准：
去掉新增的两个形参及两个前置检查后，整个 MoECTS 文件的 AST 与原版相等。
`actor_critic_moe_cts.py`、`networks/moe.py`、`rollout_storage_cts.py`、
`on_policy_runner_cts.py` 和 `train.py` 与该提交逐字节一致。
本次没有改地形、奖励、命令、课程、随机化、重置或 X5 物理参数。

验证结果：

- CPU 回归 **49 passed**（此前 43 项 + 6 项兼容测试）。相同种子下，不传新字段
  与显式传入默认字段的模型权重、随机数状态、优化器配置和参数组、师生划分一致；
  其他优化器、CNN 共享及拼错字段均明确拒绝。不执行反向传播或优化器更新。
- `outputs/x5_alignment/runner_init_4env.json`：`PASS_RUNNER_INIT_ONLY`。
  在本机 `x3w_isaaclab` 环境、`cuda:0` 上使用完整注册 X5 任务、真实向量环境包装器、
  注册算法配置和 `OnPolicyRunnerCTS`；成功创建两个 Adam。
  Teacher 索引 `[1, 2, 3]`、Student 索引 `[0]`，动作缓存 `[24, 4, 16]`；
  优化器状态均为空、采样数及学习迭代均为 0，未写 checkpoint。
- Legbot 与 X5 的完整算法配置（包括 Isaac Lab 继承字段）均通过构造签名绑定检查。

仅初始化复现命令（不调用 `learn`）：

```bash
/home/xgy/miniforge3/envs/x3w_isaaclab/bin/python scripts/tools/check_x5_sim.py \
  --headless --runner-init-only --num-envs 4 --device cuda:0 \
  --output outputs/x5_alignment/runner_init_4env.json
```

此结果证明当前构造报错已消除，不代表已经验证 PPO 更新、训练收敛，
也不承诺不同硬件/依赖版本上的训练轨迹逐位一致。训练入口及原命令不需调整；
在其他服务器运行时，需要同步修复后的 `moe_cts.py` 并重新启动进程。

## 后续检查点与验收边界

1. **资产基准：已确认并导入。** 离线校验质量、惯量正定性及三角不等式、
   关节类型/轴/上限、网格完整性、参考站姿轮底共面性。
2. **控制/任务接线：已接入并进行仿真验证。** 独立资产和任务、显式动作/观测
   顺序、物理模型及接触属性验证完成；检查脚本与证据见上文。
3. **训练配置：按用户指定的 Legbot 基准接入。** 保留任务和算法，X5 观测边界
   调整保留；三项物理惩罚已撤回比例缩放并恢复原版，最新验证见撤回记录。
4. **训练与导出：等待用户另行授权。** 尚未验证优化器更新、学习收敛、地形通过率、
   Teacher/Student 性能差距或真实机器人部署。完整配置接入不等于训练效果验收。

共有算法里的恢复课程状态、导出历史初始化和分布式学生梯度同步等问题，
需另列修改范围，不在资产导入阶段顺手改动。离线通过不能替代 USD/PhysX
检查，仿真通过不能替代真机电机顺序、方向、限位与安全验收。
