"""Same learner and network as X5 v0, with a separate experiment directory."""

from isaaclab.utils import configclass
from robot_lab.tasks.x5.rsl_rl_cfg import X5MoECTSRunnerCfg


@configclass
class X5RewardsV1RunnerCfg(X5MoECTSRunnerCfg):
    experiment_name = "x5_moe_cts_rewards_v1"


@configclass
class X5StepsV1RunnerCfg(X5MoECTSRunnerCfg):
    experiment_name = "x5_moe_cts_steps_v1"
