"""Isolated reward experiment; the original X5 task remains the comparison baseline."""

import gymnasium as gym

gym.register(
    id="RobotLab-X5-MoECTS-Rewards-v1",
    entry_point="robot_lab.tasks.x5.env:X5Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:X5RewardsV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:X5RewardsV1RunnerCfg",
    },
)

gym.register(
    id="RobotLab-X5-MoECTS-Steps-v1",
    entry_point="robot_lab.tasks.x5.env:X5Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:X5StepsV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:X5StepsV1RunnerCfg",
    },
)
