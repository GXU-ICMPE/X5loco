"""V6 learner in an independent V7 experiment, training from scratch."""

from isaaclab.utils import configclass

from robot_lab.tasks.x5_v6.rsl_rl_cfg import X5V6RunnerCfg


@configclass
class X5V7RunnerCfg(X5V6RunnerCfg):
    experiment_name = "x5_moe_cts_v7"
    resume = False
