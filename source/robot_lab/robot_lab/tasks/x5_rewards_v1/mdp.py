"""Isaac Lab adapters for stateful wheel and completed-step rewards."""

import torch
from isaaclab.managers import ManagerTermBase

from .rewards import QuietCommandGate
from .stepping import CompletedStepTracker


class StandStillWheelVelocity(ManagerTermBase):
    """Penalize wheel motion only after a genuinely quiet three-axis command."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.gate = QuietCommandGate(env.num_envs, env.device)
        self.active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.gate.reset(env_ids)
        self.active[slice(None) if env_ids is None else env_ids] = False

    def __call__(
        self, env, asset_cfg, command_name: str, wheel_radius: float,
        speed_deadband: float, linear_threshold: float, yaw_threshold: float,
        settle_time: float, reset_grace: float,
    ) -> torch.Tensor:
        if wheel_radius <= 0.0 or min(speed_deadband, reset_grace) < 0.0:
            raise ValueError("wheel radius must be positive and thresholds nonnegative")
        commands = env.command_manager.get_command(command_name)
        ready = self.gate.update(commands, env.common_step_counter, env.step_dt,
                                 linear_threshold, yaw_threshold, settle_time)
        self.active[:] = (ready & (env.episode_length_buf * env.step_dt >= reset_grace)
                          & (env.common_step_counter * env.step_dt >= reset_grace))
        rim_speed = wheel_radius * env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids].abs()
        return (rim_speed - speed_deadband).clamp_min(0.0).square().mean(dim=-1) * self.active


class CompletedWheelStep(ManagerTermBase):
    """A sparse bonus for progressing steps in lateral or stair/obstacle contexts."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.tracker = CompletedStepTracker(env.num_envs, 4, env.device)

    def reset(self, env_ids=None):
        self.tracker.reset(env_ids)

    def __call__(
        self, env, asset_cfg, sensor_cfg, command_name: str, terrain_names: tuple[str, ...],
        min_command: float, lateral_threshold: float, min_air_time: float, max_air_time: float,
        min_lift: float, min_relative_lift: float, target_relative_lift: float,
        min_foot_progress: float, min_base_progress: float, target_base_progress: float,
        min_support: int, reset_grace: float, command_cos_min: float, support_force_threshold: float,
    ) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_yaw

        data = env.scene[asset_cfg.name].data
        sensor = env.scene[sensor_cfg.name].data
        if sensor.current_air_time is None:
            raise ValueError("CompletedWheelStep requires track_air_time=True")
        command = env.command_manager.get_term(command_name)
        commands = command.command
        planar_command = torch.cat((commands[:, :2], torch.zeros_like(commands[:, :1])), dim=-1)
        world_xy = quat_apply_yaw(data.root_quat_w, planar_command)[:, :2]
        terrain_eligible = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # Reuse the X5 command manager's existing terrain-column classification.
        # This is a coarse context, not local obstacle or contact-plane detection.
        for name in terrain_names:
            idx = command.terrain_type2idx.get(name)
            if idx is not None:
                terrain_eligible |= command.terrain_idxs == idx
        if support_force_threshold < 0.0:
            raise ValueError("support_force_threshold must be nonnegative")
        contacts = sensor.current_contact_time[:, sensor_cfg.body_ids] > 0.0
        support_contacts = contacts & (sensor.net_forces_w[:, sensor_cfg.body_ids, 2] > support_force_threshold)
        episode_time = (env.episode_length_buf * env.step_dt).clamp_max(env.common_step_counter * env.step_dt)
        value = self.tracker.update(
            env.common_step_counter, env.step_dt, data.body_link_pos_w[:, asset_cfg.body_ids],
            data.root_pos_w, commands, world_xy, contacts, terrain_eligible,
            episode_time, min_command, lateral_threshold,
            min_air_time, max_air_time, min_lift, min_relative_lift, target_relative_lift,
            min_foot_progress, min_base_progress, target_base_progress,
            min_support, reset_grace, command_cos_min, support_contacts,
        )
        return value * ~env.termination_manager.terminated
