"""V7 sampling coverage with independent performance-driven x/y/yaw ranges.

The EMA / up-hold-down mechanism follows local LocoWheeledLegged's
locowheeledlegged/mdp/curriculums.py::command_axis_levels_vel. Scores here
exclude inactive axes and require actual progress, appropriate for wheel legs.
"""

import torch
from isaaclab.utils import configclass

from robot_lab.tasks.x5_v7.commands import X5V7Command, X5V7CommandCfg
from robot_lab.tasks.go2.mdp.commands import Go2RLGymCommandCfg


@configclass
class X5V8VelocityCurriculumCfg:
    max_ranges: dict = {"lin_vel_x": (-1.5, 1.5), "lin_vel_y": (-0.8, 0.8), "ang_vel_yaw": (-1.5, 1.5)}
    level_step: float = 0.05
    upper_threshold: float = 0.8
    lower_threshold: float = 0.5
    ema_alpha: float = 0.3
    tracking_std: tuple[float, float, float] = (0.35, 0.20, 0.40)
    min_command: tuple[float, float, float] = (0.08, 0.06, 0.10)
    min_active_time: float = 2.0


class X5V8Command(X5V7Command):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.command_range_curriculum:
            raise ValueError("V8 axis curriculum replaces iteration-based range jumps")
        ordinary_share = 1 - cfg.full_stop_probability - cfg.hold_x_probability - cfg.hold_y_probability
        if not 0 <= cfg.hold_yaw_probability < ordinary_share:
            raise ValueError("V8 must retain a positive share of the original sampler")
        # Extend V7's sampling counters and keep their callbacks/reset ordering.
        self._mode_names = (*self._mode_names, "hold_yaw")
        self._mode_time = torch.zeros(self.num_envs, 5, device=self.device)
        self._mode_samples = torch.zeros_like(self._mode_time)
        self.metrics.clear()
        for name in self._mode_names:
            for suffix in ("time_fraction", "sample_fraction"):
                self.metrics[f"v8_{name}_{suffix}"] = torch.zeros(self.num_envs, device=self.device)

        ccfg = cfg.velocity_curriculum
        if not (0 <= ccfg.lower_threshold < ccfg.upper_threshold <= 1 and 0 < ccfg.ema_alpha <= 1
                and 0 < ccfg.level_step <= 1 and ccfg.min_active_time > 0):
            raise ValueError("Invalid V8 axis curriculum thresholds")
        self._keys = ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
        initial = torch.tensor([self.command_ranges[key] for key in self._keys], device=self.device)
        self._maximum_ranges = torch.tensor([ccfg.max_ranges[key] for key in self._keys], device=self.device)
        if (not torch.isfinite(self._maximum_ranges).all() or not torch.isfinite(initial).all()
                or not torch.allclose(initial[:, 0], -initial[:, 1])
                or not torch.allclose(self._maximum_ranges[:, 0], -self._maximum_ranges[:, 1])
                or (initial[:, 1] <= 0).any() or (self._maximum_ranges[:, 1] < initial[:, 1]).any()):
            raise ValueError("V8 curriculum requires finite symmetric ranges: 0 < initial <= maximum")
        self._tracking_std = torch.tensor(ccfg.tracking_std, device=self.device)
        self._min_command = torch.tensor(ccfg.min_command, device=self.device)
        if (self._tracking_std.shape != (3,) or self._min_command.shape != (3,)
                or not torch.isfinite(self._tracking_std).all() or (self._tracking_std <= 0).any()
                or not torch.isfinite(self._min_command).all() or (self._min_command <= 0).any()):
            raise ValueError("V8 curriculum requires three positive finite std/min_command values")
        self._minimum_level = initial[:, 1] / self._maximum_ranges[:, 1]
        self.command_levels = self._minimum_level.repeat(self.num_envs, 1)
        self.tracking_ema = torch.zeros_like(self.command_levels)
        self._has_score = torch.zeros_like(self.command_levels, dtype=torch.bool)
        self._tracking_sum = torch.zeros_like(self.command_levels)
        self._error_square_sum = torch.zeros_like(self.command_levels)
        self._active_time = torch.zeros_like(self.command_levels)
        self._elapsed_time = torch.zeros(self.num_envs, device=self.device)
        self._last_tracking_step = torch.full_like(self.mode, env.common_step_counter)
        self._last_curriculum_step = torch.full_like(self.command_levels, -1, dtype=torch.long)
        # Cache terrain-specific caps, then apply per-environment axis levels.
        self.command_ranges = {key: tuple(value) for key, value in ccfg.max_ranges.items()}
        self._update_env_command_ranges()
        self._terrain_caps = torch.stack([self.env_command_ranges[key].clone() for key in self._keys], dim=1)
        self.max_lin_vel = self._maximum_ranges[:2, 1].max().item()
        self._apply_curriculum_ranges(self._all_ids)
        for axis in "xyz":
            for name in ("level", "max", "tracking_ema", "active_error_rms", "active_seconds"):
                self.metrics[f"v8_{axis}_{name}"] = torch.zeros(self.num_envs, device=self.device)

    def _apply_curriculum_ranges(self, ids):
        target = self.command_levels[ids, :, None] * self._maximum_ranges
        target[:, :, 0] = torch.maximum(target[:, :, 0], self._terrain_caps[ids, :, 0])
        target[:, :, 1] = torch.minimum(target[:, :, 1], self._terrain_caps[ids, :, 1])
        for axis, key in enumerate(self._keys):
            self.env_command_ranges[key][ids] = target[:, axis]

    def _accumulate_metrics(self, env_ids):
        ids = env_ids[self._last_metric_step[env_ids] < self._env.common_step_counter]
        self._mode_time[ids, self.mode[ids]] += self._env.step_dt
        self._last_metric_step[ids] = self._env.common_step_counter
        for index, name in enumerate(self._mode_names):
            self.metrics[f"v8_{name}_time_fraction"][:] = (
                self._mode_time[:, index] / self._mode_time.sum(-1).clamp_min(self._env.step_dt))
            self.metrics[f"v8_{name}_sample_fraction"][:] = (
                self._mode_samples[:, index] / self._mode_samples.sum(-1).clamp_min(1))

    def _accumulate_tracking(self, env_ids):
        ids = env_ids[self._last_tracking_step[env_ids] < self._env.common_step_counter]
        if not len(ids):
            return
        command = self.commands[ids]
        actual = torch.cat((self.robot.data.root_lin_vel_b[ids, :2],
                            self.robot.data.root_ang_vel_b[ids, 2:3]), dim=-1)
        active = command.abs() > self._min_command
        error2 = (actual - command).square()
        progress = (actual * command.sign() / command.abs().clamp_min(self._min_command)).clamp(0, 1)
        score = torch.exp(-error2 / self._tracking_std.square()) * progress
        dt = self._env.step_dt
        self._tracking_sum[ids] += torch.nan_to_num(score, nan=0.0) * active * dt
        self._error_square_sum[ids] += torch.nan_to_num(error2, nan=1.0, posinf=1.0) * active * dt
        self._active_time[ids] += active * dt
        self._elapsed_time[ids] += dt
        self._last_tracking_step[ids] = self._env.common_step_counter

    def update_axis_curriculum(self, env_ids, axis):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._accumulate_tracking(ids)  # includes terminal state BEFORE scene reset
        cfg = self.cfg.velocity_curriculum
        eligible = ((self._active_time[ids, axis] >= cfg.min_active_time)
                    & (self._last_curriculum_step[ids, axis] != self._env.common_step_counter))
        ids = ids[eligible]
        score = self._tracking_sum[ids, axis] / self._active_time[ids, axis].clamp_min(self._env.step_dt)
        # Early falls cannot upgrade a brief, well-tracked command segment.
        score *= (self._elapsed_time[ids] / self._env.max_episode_length_s).clamp(0, 1)
        ema = (1 - cfg.ema_alpha) * self.tracking_ema[ids, axis] + cfg.ema_alpha * score
        ema = torch.where(self._has_score[ids, axis], ema, score)
        self.tracking_ema[ids, axis] = ema
        self._has_score[ids, axis] = True
        delta = (ema > cfg.upper_threshold).float() - (ema < cfg.lower_threshold).float()
        self.command_levels[ids, axis] = (self.command_levels[ids, axis] + cfg.level_step * delta).clamp(
            min=self._minimum_level[axis], max=1.0)
        self._last_curriculum_step[ids, axis] = self._env.common_step_counter
        self._apply_curriculum_ranges(ids)
        self._refresh_metrics()
        return {"level": self.command_levels[:, axis].mean(),
                "tracking_ema": self.tracking_ema[:, axis].mean(),
                "max_velocity": self.env_command_ranges[self._keys[axis]][:, 1].mean()}

    def _refresh_metrics(self):
        for axis, key in enumerate(self._keys):
            prefix = f"v8_{'xyz'[axis]}_"
            self.metrics[prefix + "level"][:] = self.command_levels[:, axis]
            self.metrics[prefix + "max"][:] = self.env_command_ranges[key][:, 1]
            self.metrics[prefix + "tracking_ema"][:] = self.tracking_ema[:, axis]
            self.metrics[prefix + "active_error_rms"][:] = (
                self._error_square_sum[:, axis] / self._active_time[:, axis].clamp_min(self._env.step_dt)).sqrt()
            self.metrics[prefix + "active_seconds"][:] = self._active_time[:, axis]

    def _update_metrics(self):
        self._accumulate_tracking(self._all_ids)
        self._accumulate_metrics(self._all_ids)
        self._refresh_metrics()

    def reset(self, env_ids=None):
        ids = self._all_ids if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._accumulate_tracking(ids)
        self._refresh_metrics()
        result = super().reset(ids)
        for value in (self._tracking_sum, self._error_square_sum, self._active_time, self._elapsed_time):
            value[ids] = 0
        self._last_tracking_step[ids] = self._env.common_step_counter
        return result

    def _resample(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not len(ids):
            return
        before = self.commands_xy_accumulation[ids].clone()
        super()._resample(ids)
        original_share = 1 - self.cfg.full_stop_probability - self.cfg.hold_x_probability - self.cfg.hold_y_probability
        choose_yaw = ((self.mode[ids] == 0)
                      & (torch.rand(len(ids), device=self.device) < self.cfg.hold_yaw_probability / original_share))
        for bounds in self.env_command_ranges.values():
            choose_yaw &= (bounds[ids, 0] <= 0) & (bounds[ids, 1] >= 0)
        bounds = self.env_command_ranges["ang_vel_yaw"][ids]
        choose_yaw &= bounds.abs().amax(-1) > 0
        yaw_ids = ids[choose_yaw]
        lower, upper = self.env_command_ranges["ang_vel_yaw"][yaw_ids].unbind(-1)
        endpoint = torch.where(torch.rand(len(yaw_ids), device=self.device) < 0.5, lower, upper)
        fraction = torch.empty(len(yaw_ids), device=self.device).uniform_(*self.cfg.axis_speed_fraction_range)
        self.commands[yaw_ids] = 0
        self.commands[yaw_ids, 2] = endpoint * fraction
        self.time_left[yaw_ids] = torch.empty(len(yaw_ids), device=self.device).uniform_(*self.cfg.yaw_hold_time_range)
        self.commands_xy_accumulation[yaw_ids] = before[choose_yaw]
        self.last_is_limit_vel[yaw_ids] = False
        self.mode[yaw_ids] = 4
        self._mode_samples[yaw_ids, 0] -= 1
        self._mode_samples[yaw_ids, 4] += 1


@configclass
class X5V8CommandCfg(X5V7CommandCfg):
    class_type: type = X5V8Command
    ranges: Go2RLGymCommandCfg.Ranges = Go2RLGymCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.2, 0.2), ang_vel_yaw=(-0.5, 0.5))
    command_range_curriculum: list[dict] = []
    hold_y_probability: float = 0.25
    hold_yaw_probability: float = 0.10
    yaw_hold_time_range: tuple[float, float] = (4.0, 6.0)
    velocity_curriculum: X5V8VelocityCurriculumCfg = X5V8VelocityCurriculumCfg()


def command_axis_levels_vel(env, env_ids, command_axis, command_name="base_velocity"):
    if command_axis not in ("x", "y", "z"):
        raise ValueError("command_axis must be x, y, or z (yaw)")
    return env.command_manager.get_term(command_name).update_axis_curriculum(env_ids, "xyz".index(command_axis))
