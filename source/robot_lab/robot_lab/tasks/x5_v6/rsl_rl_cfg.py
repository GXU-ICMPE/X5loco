"""Original PPO-CTS-MOE settings with independent X5 v6 logs/checkpoints."""

from isaaclab.utils import configclass

from robot_lab.tasks.x5.rsl_rl_cfg import X5MoECTSRunnerCfg


@configclass
class X5V6RunnerCfg(X5MoECTSRunnerCfg):
    experiment_name = "x5_moe_cts_v6"
    max_iterations = 30000
    resume = False
