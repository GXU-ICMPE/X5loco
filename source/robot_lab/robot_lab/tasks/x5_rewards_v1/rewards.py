"""Torch-only reward formulas, independently testable without starting Isaac Sim.

Comfort intervals and stopped-wheel regularization are adapted conceptually from
Flamingo commit 61805bbf7fb5bdf14a3ba46adfe2d26e63ca8c02; see docs/x5_rewards_v1.md.
Sustained-air wheel spin is an X5-specific experimental extension, not wheel slip.
All functions return nonnegative costs; RewardManager supplies negative weights
and the policy time step. Wheel radius converts rad/s to m/s, not motor capacity.
"""

import torch


def joint_pose_deadband_l1(env, asset_cfg, deadband: float) -> torch.Tensor:
    """Distance outside a symmetric comfort interval about each default angle."""
    if deadband < 0.0:
        raise ValueError("deadband must be nonnegative")
    data = env.scene[asset_cfg.name].data
    error = data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids]
    return (error.abs() - deadband).clamp_min(0.0).sum(dim=-1)


def wheel_air_spin_l2(
    env, asset_cfg, sensor_cfg, wheel_radius: float, speed_deadband: float,
    air_time_grace: float, contact_threshold: float, reset_grace: float,
) -> torch.Tensor:
    """Weak excess-rim-speed cost after sustained loss of contact.

    asset_cfg.joint_ids and sensor_cfg.body_ids must describe the same wheels in
    the same order. Neither brief flight nor flight with a stationary wheel costs
    anything. Recent contact history suppresses chatter around contact switches.
    """
    if wheel_radius <= 0.0 or min(speed_deadband, air_time_grace, contact_threshold, reset_grace) < 0.0:
        raise ValueError("wheel radius must be positive and thresholds nonnegative")
    rim_speed = wheel_radius * env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids].abs()
    contact = env.scene[sensor_cfg.name].data
    air_time = contact.current_air_time
    if air_time is None:
        raise ValueError("wheel_air_spin_l2 requires contact sensor track_air_time=True")
    recent_force = contact.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).amax(dim=1)
    air_time = air_time[:, sensor_cfg.body_ids]
    if rim_speed.shape != air_time.shape or rim_speed.shape != recent_force.shape:
        raise ValueError("wheel joints and contact bodies must have matching shapes and order")
    airborne = (air_time > air_time_grace) & (recent_force <= contact_threshold)
    excess = (rim_speed - speed_deadband).clamp_min(0.0)
    cost = (excess.square() * airborne).mean(dim=-1)
    # RSL-RL randomizes initial episode_length_buf; the global counter prevents
    # that artificial age from bypassing the very first physical settling period.
    settled = ((env.episode_length_buf * env.step_dt >= reset_grace)
               & (env.common_step_counter * env.step_dt >= reset_grace))
    return cost * settled


class QuietCommandGate:
    """Per-environment command dwell time, with explicit reset and idempotent reads."""

    def __init__(self, num_envs: int, device):
        self.elapsed = torch.zeros(num_envs, device=device)
        self.last_step = None

    def reset(self, env_ids=None):
        self.elapsed[slice(None) if env_ids is None else env_ids] = 0.0

    def update(
        self, commands: torch.Tensor, step: int, dt: float,
        linear_threshold: float, yaw_threshold: float, settle_time: float,
    ) -> torch.Tensor:
        if dt <= 0.0 or min(linear_threshold, yaw_threshold, settle_time) < 0.0:
            raise ValueError("dt must be positive and command thresholds nonnegative")
        quiet = (commands[:, :2].norm(dim=-1) <= linear_threshold) & (commands[:, 2].abs() <= yaw_threshold)
        # Also clear on a same-step command change; do not keep a stale ready mask.
        self.elapsed[~quiet] = 0.0
        if self.last_step != step:
            # If the term was disabled or steps were skipped, don't assume that
            # unobserved commands were quiet. Rebuild the dwell time from here.
            if self.last_step is not None and step != self.last_step + 1:
                self.elapsed.zero_()
            self.elapsed += quiet.to(self.elapsed.dtype) * dt
            self.last_step = step
        return quiet & (self.elapsed + 1.0e-6 >= settle_time)
