"""A mixture retaining the entire V6 sampler, plus stop and held-axis segments."""

import math

import torch
from isaaclab.utils import configclass

from robot_lab.tasks.go2.mdp.commands import Go2RLGymCommand, Go2RLGymCommandCfg


class X5V7Command(Go2RLGymCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        probabilities = (cfg.full_stop_probability, cfg.hold_x_probability, cfg.hold_y_probability)
        if any(not math.isfinite(p) or p < 0 for p in probabilities) or sum(probabilities) >= 1:
            raise ValueError("V7 command probabilities must leave a positive share for the original sampler")
        for limits in (cfg.full_stop_time_range, cfg.axis_hold_time_range):
            if not all(math.isfinite(v) for v in limits) or not 0 < limits[0] <= limits[1]:
                raise ValueError("V7 command durations must be finite, positive and ordered")
        if not 0 < cfg.axis_speed_fraction_range[0] <= cfg.axis_speed_fraction_range[1] <= 1:
            raise ValueError("V7 axis speed fractions must lie in (0, 1]")
        self._all_ids = torch.arange(self.num_envs, device=self.device)
        # Codes describe the selected sampling branch, not measured robot motion.
        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._mode_names = ("original", "full_stop", "hold_x", "hold_y")
        self._mode_time = torch.zeros(self.num_envs, 4, device=self.device)
        self._mode_samples = torch.zeros_like(self._mode_time)
        self._last_metric_step = torch.full_like(self.mode, env.common_step_counter)
        for name in self._mode_names:
            self.metrics[f"v7_{name}_time_fraction"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"v7_{name}_sample_fraction"] = torch.zeros(self.num_envs, device=self.device)

    def _accumulate_metrics(self, env_ids):
        ids = env_ids[self._last_metric_step[env_ids] < self._env.common_step_counter]
        self._mode_time[ids, self.mode[ids]] += self._env.step_dt
        self._last_metric_step[ids] = self._env.common_step_counter
        time_total = self._mode_time.sum(-1).clamp_min(self._env.step_dt)
        samples_total = self._mode_samples.sum(-1).clamp_min(1)
        for index, name in enumerate(self._mode_names):
            self.metrics[f"v7_{name}_time_fraction"][:] = self._mode_time[:, index] / time_total
            self.metrics[f"v7_{name}_sample_fraction"][:] = self._mode_samples[:, index] / samples_total

    def _update_metrics(self):
        super()._update_metrics()
        self._accumulate_metrics(self._all_ids)

    def reset(self, env_ids=None):
        ids = self._all_ids if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # Include the terminal step before CommandTerm.reset logs/clears metrics.
        self._accumulate_metrics(ids)
        self._mode_time[ids] = 0
        self._mode_samples[ids] = 0
        self._last_metric_step[ids] = self._env.common_step_counter
        return super().reset(ids)

    def _resample(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(ids) == 0:
            return
        previous_accumulation = self.commands_xy_accumulation[ids].clone()
        # Keep V6's ordinary/dynamic, boundary, continuous-boundary inversion,
        # zero-XY/pure-turn, terrain caps and iteration schedules intact. Calling
        # it first also updates the ranges used by the added branches.
        super()._resample(ids)
        draw = torch.rand(len(ids), device=self.device)
        stop_end = self.cfg.full_stop_probability
        x_end = stop_end + self.cfg.hold_x_probability
        y_end = x_end + self.cfg.hold_y_probability
        selected = torch.zeros(len(ids), dtype=torch.long, device=self.device)
        selected[draw < stop_end] = 1
        selected[(draw >= stop_end) & (draw < x_end)] = 2
        selected[(draw >= x_end) & (draw < y_end)] = 3

        # A user-supplied range excluding zero cannot support a pure-axis/stop
        # command. Such samples safely retain their original V6 command.
        zero_allowed = torch.ones(len(ids), dtype=torch.bool, device=self.device)
        for bounds in self.env_command_ranges.values():
            zero_allowed &= (bounds[ids, 0] <= 0) & (bounds[ids, 1] >= 0)
        selected[~zero_allowed] = 0
        for code, key in ((2, "lin_vel_x"), (3, "lin_vel_y")):
            bounds = self.env_command_ranges[key][ids]
            selected[(selected == code) & (bounds.abs().amax(-1) == 0)] = 0

        self.mode[ids] = selected
        stop_ids = ids[selected == 1]
        self.commands[stop_ids] = 0
        self.time_left[stop_ids] = torch.empty(len(stop_ids), device=self.device).uniform_(
            *self.cfg.full_stop_time_range
        )
        for axis, code, key in ((0, 2, "lin_vel_x"), (1, 3, "lin_vel_y")):
            axis_ids = ids[selected == code]
            lower, upper = self.env_command_ranges[key][axis_ids].unbind(-1)
            positive = (lower >= 0) | ((upper > 0) & (torch.rand(len(axis_ids), device=self.device) < 0.5))
            endpoint = torch.where(positive, upper, lower)
            fraction = torch.empty(len(axis_ids), device=self.device).uniform_(*self.cfg.axis_speed_fraction_range)
            self.commands[axis_ids] = 0
            self.commands[axis_ids, axis] = endpoint * fraction
            self.time_left[axis_ids] = torch.empty(len(axis_ids), device=self.device).uniform_(
                *self.cfg.axis_hold_time_range
            )

        added = selected != 0
        added_ids = ids[added]
        self.last_is_limit_vel[added_ids] = False
        # Replace the discarded parent sample exactly once. The inherited
        # distance accounting uses a 5 s reference; a 9 s segment contributes
        # v_xy * 9 / 5, and a stop contributes zero. The upgrade rule stays V6.
        self.commands_xy_accumulation[added_ids] = previous_accumulation[added] + (
            self.commands[added_ids, :2] * (self.time_left[added_ids] / self.cfg.resampling_time).unsqueeze(-1)
        )
        self.command_counter[ids] += 1
        self._mode_samples[ids, selected] += 1


@configclass
class X5V7CommandCfg(Go2RLGymCommandCfg):
    class_type: type = X5V7Command
    # Probabilities per segment selection, not fractions of elapsed time.
    # The remaining 60% uses the full original V6 sampling distribution.
    full_stop_probability: float = 0.10
    hold_x_probability: float = 0.15
    hold_y_probability: float = 0.15
    full_stop_time_range: tuple[float, float] = (3.0, 5.0)
    axis_hold_time_range: tuple[float, float] = (8.0, 10.0)
    axis_speed_fraction_range: tuple[float, float] = (0.6, 1.0)
