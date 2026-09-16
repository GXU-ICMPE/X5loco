"""X5 V8 training task."""

import gymnasium as gym

gym.register(
    id="RobotLab-X5-MoECTS-v8",
    entry_point="robot_lab.tasks.x5_v8.env:X5V8Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:X5V8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:X5V8RunnerCfg",
    },
)
