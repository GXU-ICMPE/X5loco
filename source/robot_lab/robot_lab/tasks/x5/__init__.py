"""Independent X5 task; currently a flat, deterministic bring-up baseline."""

import gymnasium as gym


gym.register(
    id="RobotLab-X5-MoECTS-v0",
    entry_point=f"{__name__}.env:X5Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:X5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:X5MoECTSRunnerCfg",
    },
)
