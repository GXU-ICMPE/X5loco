"""Soft wheel-center geometry costs on locally flat ground while stopping/rolling."""

import torch
from isaaclab.utils.math import quat_apply_inverse

from robot_lab.assets.x5 import X5_FOOT_BODY_NAMES


class X5V7WheelSupportState:
    def __init__(self, env):
        self.env, self.cfg = env, env.cfg.v7_wheel_support
        self.cfg.validate()
        self.robot = env.scene["robot"]
        self.wheel_ids, names = self.robot.find_bodies(X5_FOOT_BODY_NAMES, preserve_order=True)
        if list(names) != X5_FOOT_BODY_NAMES:
            raise ValueError("V7 wheel support requires RF/LF/RH/LH wheel link frames in that order")
        self.reference_xy = torch.tensor(
            self.cfg.reference_wheel_xy_m, device=env.device, dtype=self.robot.data.default_joint_pos.dtype
        )
        self.xy_tolerance = self.reference_xy.new_tensor(
            (self.cfg.longitudinal_tolerance_m, self.cfg.lateral_tolerance_m)
        )
        # Signed distances preserve left/right and front/rear order: crossed
        # wheels cannot look like a wide support polygon by taking abs(distance).
        self.min_wheelbase = (self.reference_xy[:2, 0] - self.reference_xy[2:, 0]
                              - self.cfg.wheelbase_shortening_tolerance_m)
        self.min_track_width = (self.reference_xy[[1, 3], 1] - self.reference_xy[[0, 2], 1]
                                - self.cfg.track_width_shortening_tolerance_m)
        self.position_cost = torch.zeros(env.num_envs, device=env.device)
        self.wheelbase_cost = torch.zeros_like(self.position_cost)
        self.track_width_cost = torch.zeros_like(self.position_cost)
        self.samples = torch.zeros_like(self.position_cost)
        self.gate_sum = torch.zeros_like(self.position_cost)
        self.sums = {name: torch.zeros_like(self.position_cost) for name in (
            "rf_x_m", "lf_x_m", "front_retraction_m", "right_wheelbase_m", "left_wheelbase_m",
            "front_track_width_m", "rear_track_width_m", "xy_error_rms_m",
        )}
        self.last_step = -1

    def update(self):
        env, cfg = self.env, self.cfg
        if self.last_step == env.common_step_counter:
            return self
        self.last_step = env.common_step_counter
        gate = env.v7_posture.update().flat_straight_gate
        data = self.robot.data
        offsets_w = data.body_link_pos_w[:, self.wheel_ids] - data.root_link_pos_w[:, None, :]
        base_quat = data.root_link_quat_w[:, None, :].expand(-1, 4, -1)
        centers_b = quat_apply_inverse(base_quat.reshape(-1, 4), offsets_w.reshape(-1, 3)).reshape(-1, 4, 3)
        xy = centers_b[..., :2]
        error_xy = xy - self.reference_xy
        excess_xy = (error_xy.abs() - self.xy_tolerance).clamp_min(0)
        self.position_cost = gate * (excess_xy / cfg.position_error_scale_m).square().sum(-1).mean(-1)

        wheelbase = xy[:, :2, 0] - xy[:, 2:, 0]  # right, left
        track_width = xy[:, [1, 3], 1] - xy[:, [0, 2], 1]  # front, rear
        short_wheelbase = (self.min_wheelbase - wheelbase).clamp_min(0)
        narrow_track = (self.min_track_width - track_width).clamp_min(0)
        self.wheelbase_cost = gate * (short_wheelbase / cfg.wheelbase_error_scale_m).square().mean(-1)
        self.track_width_cost = gate * (narrow_track / cfg.track_width_error_scale_m).square().mean(-1)

        # Geometry metrics are weighted by the same gate as the rewards, so
        # stair swing poses do not contaminate flat-ground posture diagnostics.
        values = (xy[:, 0, 0], xy[:, 1, 0], (-error_xy[:, :2, 0]).clamp_min(0).mean(-1),
                  wheelbase[:, 0], wheelbase[:, 1], track_width[:, 0], track_width[:, 1],
                  error_xy.square().sum(-1).mean(-1).sqrt())
        for total, value in zip(self.sums.values(), values):
            total += gate * value
        self.samples += 1
        self.gate_sum += gate
        return self

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return {}
        # Pool active samples over resetting environments. Environments with
        # no flat/straight samples do not drag the reported distances to zero.
        denominator = self.gate_sum[env_ids].sum().clamp_min(1.0e-6)
        log = {f"X5V7/wheel_support/{name}": total[env_ids].sum() / denominator
               for name, total in self.sums.items()}
        log["X5V7/wheel_support/active_fraction"] = (
            self.gate_sum[env_ids].sum() / self.samples[env_ids].sum().clamp_min(1)
        )
        for total in self.sums.values():
            total[env_ids] = 0
        for value in (self.position_cost, self.wheelbase_cost, self.track_width_cost, self.samples, self.gate_sum):
            value[env_ids] = 0
        return log
