"""X5 v0 reward and domain-randomization tuning in an independent task."""

import gymnasium as gym


gym.register(
    id="RobotLab-X5-MoECTS-v6",
    entry_point="robot_lab.tasks.x5.env:X5Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:X5V6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:X5V6RunnerCfg",
    },
)
