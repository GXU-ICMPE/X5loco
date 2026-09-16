"""X5 V6 with additional command coverage and terrain-conditioned posture."""

import gymnasium as gym


gym.register(
    id="RobotLab-X5-MoECTS-v7",
    entry_point="robot_lab.tasks.x5_v7.env:X5V7Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:X5V7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:X5V7RunnerCfg",
    },
)
