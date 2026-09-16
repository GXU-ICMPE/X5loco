"""Small V8 reward adapters; physical costs inherited from V7 remain separate."""

import torch


def motion_cost(env, name):
    return env.v8_motion.update().costs[name]


def lateral_progress(env, std=0.25):
    command = env.command_manager.get_command("base_velocity")
    actual = env.scene["robot"].data.root_lin_vel_b[:, :2]
    desired_y = command[:, 1]
    progress = (actual[:, 1] * desired_y.sign() / desired_y.abs().clamp_min(0.08)).clamp(0, 1)
    tracking = torch.exp(-((actual - command[:, :2]) / std).square().sum(-1))
    yaw_error = env.scene["robot"].data.root_ang_vel_b[:, 2] - command[:, 2]
    # Pure lateral motion cannot substitute an uncontrolled body spin.
    return (desired_y.abs() > 0.08) * progress * tracking * torch.exp(-(yaw_error / 0.5).square())
