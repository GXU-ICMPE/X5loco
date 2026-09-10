"""Completed, progressing wheel-foot steps; a bounded X5-specific shaping reward.

Inspired by Flamingo's touchdown/air-time design, without biped phase constraints
or fixed world-height targets. This is not an obstacle-clearance measurement.
"""

import torch


class CompletedStepTracker:
    """Track each swing independently and pay only at a valid touchdown."""

    def __init__(self, num_envs: int, num_feet: int, device):
        shape = (num_envs, num_feet)
        self.prev_contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self.tracking = torch.zeros_like(self.prev_contact)
        self.duration = torch.zeros(shape, device=device)
        self.start_foot = torch.zeros((*shape, 3), device=device)
        self.start_base = torch.zeros_like(self.start_foot)
        self.direction_w = torch.zeros((*shape, 2), device=device)
        self.direction_b = torch.zeros_like(self.direction_w)
        self.peak_z = torch.zeros(shape, device=device)
        self.start_relative_z = torch.zeros(shape, device=device)
        self.peak_relative_z = torch.zeros(shape, device=device)
        self.accepted_events = torch.zeros_like(self.prev_contact)
        self.last_reward = torch.zeros(num_envs, device=device)
        self.last_step = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for tensor in (self.prev_contact, self.tracking, self.duration, self.start_foot,
                       self.start_base, self.direction_w, self.direction_b, self.peak_z,
                       self.start_relative_z, self.peak_relative_z, self.accepted_events, self.last_reward):
            tensor[ids] = 0

    def update(
        self, step: int, dt: float, wheel_pos_w: torch.Tensor, base_pos_w: torch.Tensor,
        commands_b: torch.Tensor, command_xy_w: torch.Tensor, contacts: torch.Tensor,
        terrain_eligible: torch.Tensor, episode_time: torch.Tensor,
        min_command: float = 0.1, lateral_threshold: float = 0.1,
        min_air_time: float = 0.1, max_air_time: float = 0.6,
        min_lift: float = 0.04, min_relative_lift: float = 0.02,
        target_relative_lift: float = 0.06, min_foot_progress: float = 0.04,
        min_base_progress: float = 0.01, target_base_progress: float = 0.04,
        min_support: int = 2, reset_grace: float = 0.5, command_cos_min: float = 0.866,
        support_contacts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dt <= 0 or not 0 <= min_air_time < max_air_time:
            raise ValueError("Invalid physics interval or air-time window")
        if min(target_relative_lift, target_base_progress, min_command) <= 0:
            raise ValueError("Reward scales and minimum command must be positive")
        if not 1 <= min_support < contacts.shape[1] or not -1 <= command_cos_min <= 1:
            raise ValueError("Invalid support count or command cosine threshold")
        if min(lateral_threshold, min_lift, min_relative_lift, min_foot_progress,
               min_base_progress, reset_grace) < 0:
            raise ValueError("Step thresholds must be nonnegative")
        if self.last_step == step:
            return self.last_reward
        if self.last_step is not None and step != self.last_step + 1:
            self.reset()
        self.last_step = step

        speed = commands_b[:, :2].norm(dim=-1)
        direction_b = commands_b[:, :2] / speed.clamp_min(1.0e-6).unsqueeze(-1)
        direction_w = command_xy_w / command_xy_w.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        moving = speed >= min_command
        context = (commands_b[:, 1].abs() >= lateral_threshold) | terrain_eligible
        # The adapter distinguishes upward support from contact with a riser.
        # Pure tensor callers can use contacts for both if no force data exists.
        if support_contacts is None:
            support_contacts = contacts
        else:
            support_contacts = support_contacts & contacts
        enough_support = support_contacts.sum(dim=-1) >= min_support
        ready = (episode_time >= reset_grace) & moving & enough_support
        relative_z = wheel_pos_w[:, :, 2] - base_pos_w[:, None, 2]

        takeoff = self.prev_contact & ~contacts & (ready & context).unsqueeze(-1)
        self.tracking |= takeoff
        self.duration = torch.where(takeoff, 0.0, self.duration)
        self.start_foot = torch.where(takeoff.unsqueeze(-1), wheel_pos_w, self.start_foot)
        self.start_base = torch.where(takeoff.unsqueeze(-1), base_pos_w[:, None, :], self.start_base)
        self.direction_w = torch.where(takeoff.unsqueeze(-1), direction_w[:, None, :], self.direction_w)
        self.direction_b = torch.where(takeoff.unsqueeze(-1), direction_b[:, None, :], self.direction_b)
        self.peak_z = torch.where(takeoff, wheel_pos_w[:, :, 2], self.peak_z)
        self.start_relative_z = torch.where(takeoff, relative_z, self.start_relative_z)
        self.peak_relative_z = torch.where(takeoff, relative_z, self.peak_relative_z)

        # Compare command direction in body coordinates: a commanded turn can
        # rotate the world frame during a swing without changing the task intent.
        stable_command = (self.direction_b * direction_b[:, None, :]).sum(dim=-1) >= command_cos_min
        self.tracking &= ready.unsqueeze(-1) & stable_command
        self.duration += (self.tracking & ~contacts).to(self.duration.dtype) * dt
        self.peak_z = torch.where(self.tracking, torch.maximum(self.peak_z, wheel_pos_w[:, :, 2]), self.peak_z)
        self.peak_relative_z = torch.where(
            self.tracking, torch.maximum(self.peak_relative_z, relative_z), self.peak_relative_z)

        foot_progress = ((wheel_pos_w[:, :, :2] - self.start_foot[:, :, :2]) * self.direction_w).sum(dim=-1)
        base_progress = ((base_pos_w[:, None, :2] - self.start_base[:, :, :2]) * self.direction_w).sum(dim=-1)
        lift = self.peak_z - self.start_foot[:, :, 2]
        relative_lift = self.peak_relative_z - self.start_relative_z
        touchdown = self.tracking & contacts & ~self.prev_contact
        accepted = (touchdown & support_contacts & (self.duration + 1.0e-6 >= min_air_time)
                    & (self.duration <= max_air_time + 1.0e-6)
                    & (lift + 1.0e-6 >= min_lift) & (relative_lift + 1.0e-6 >= min_relative_lift)
                    & (foot_progress + 1.0e-6 >= min_foot_progress)
                    & (base_progress + 1.0e-6 >= min_base_progress))
        score = (relative_lift / target_relative_lift).clamp(0.0, 1.0)
        score *= (base_progress / target_base_progress).clamp(0.0, 1.0)
        self.last_reward[:] = (score * accepted).mean(dim=-1)
        self.accepted_events[:] = accepted
        self.tracking &= ~contacts & (self.duration <= max_air_time + 1.0e-6)
        self.prev_contact[:] = contacts
        return self.last_reward
