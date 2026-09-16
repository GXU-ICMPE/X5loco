"""A fresh V8 experiment, keeping the established MoE-CTS learner."""

from isaaclab.utils import configclass
from robot_lab.tasks.x5_v7.rsl_rl_cfg import X5V7RunnerCfg


@configclass
class X5V8RunnerCfg(X5V7RunnerCfg):
    experiment_name = "x5_moe_cts_v8"
    save_interval = 250
    resume = False
    # Console only; keep all TensorBoard scalars for later diagnosis.
    console_rewards_curriculum_only: bool = True
