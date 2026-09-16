"""Local plane/residual geometry and smooth posture coefficients for X5 V7."""

import math

import torch


def smooth_gate(value, limits):
    t = ((value - limits[0]) / (limits[1] - limits[0])).clamp(0, 1)
    return t.square() * (3 - 2 * t)


class X5V7PostureState:
    def __init__(self, env):
        self.env, self.cfg = env, env.cfg.v7_posture
        self.cfg.validate()
        self.robot = env.scene["robot"]
        self.scanner = env.scene[self.cfg.scanner_name]
        if self.scanner.cfg.ray_alignment != "yaw":
            raise ValueError("V7 posture requires a yaw-aligned, downward grid scanner")
        pattern = self.scanner.cfg.pattern_cfg
        # This fixed geometry is tiny. Prepare it on CPU to avoid loading a
        # CUDA linear-algebra workspace just for the one-time pseudoinverse.
        starts, directions = pattern.func(pattern, "cpu")
        if not torch.allclose(directions, directions.new_tensor((0.0, 0.0, -1.0)).expand_as(directions)):
            raise ValueError("V7 posture requires vertical downward rays")
        xy = starts[:, :2]
        design = torch.cat((xy, torch.ones_like(xy[:, :1])), dim=-1)
        # Small constant solve at initialization; subsequent fits are matmuls.
        self.design = design.to(env.device)
        self.inverse = torch.linalg.pinv(design).to(env.device)
        pairs = ((torch.cdist(xy, xy) - pattern.resolution).abs() < 1.0e-4).triu(1).nonzero()
        if len(pairs) == 0 or len(starts) < 4:
            raise ValueError("V7 posture requires a two-dimensional height grid with adjacent rays")
        self.pair_a, self.pair_b = pairs.to(env.device).unbind(-1)
        self.quiet_time = torch.zeros(env.num_envs, device=env.device)
        self.hip_scale = torch.ones_like(self.quiet_time)
        self.leg_scale = torch.ones_like(self.quiet_time)
        self.tilt_cost = torch.zeros_like(self.quiet_time)
        self.flat_straight_gate = torch.zeros_like(self.quiet_time)
        self.last_step = -1
        self.samples = torch.zeros_like(self.quiet_time)
        self.sums = {name: torch.zeros_like(self.quiet_time) for name in (
            "flat_fraction", "slope_fraction", "rough_fraction", "geometry_valid_fraction",
            "hip_multiplier", "thigh_calf_multiplier", "stand_gate", "tilt_deg", "tilt_allowance_deg",
        )}

    def update(self):
        env, cfg = self.env, self.cfg
        if self.last_step == env.common_step_counter:
            return self
        if self.last_step != env.common_step_counter - 1:
            self.quiet_time.zero_()
        self.last_step = env.common_step_counter
        hits = self.scanner.data.ray_hits_w
        if hits.shape[1] != self.design.shape[0]:
            raise ValueError("V7 height scan shape differs from its configured grid")
        valid = torch.isfinite(hits).all(dim=-1).all(dim=-1) & (hits[..., 2].abs().amax(-1) < 1.0e6)
        heights = torch.where(valid[:, None], hits[..., 2], 0.0)
        heights = heights - heights.mean(-1, keepdim=True)
        plane = heights @ self.inverse.T
        residual = heights - plane @ self.design.T
        slope_deg = torch.rad2deg(torch.atan(torch.linalg.vector_norm(plane[:, :2], dim=-1)))
        residual_rms = residual.square().mean(-1).sqrt()
        # Subtracting the fitted plane removes a ramp's expected height change.
        # Stair edges retain a discontinuity even when their overall trend is a slope.
        residual_jump = (residual[:, self.pair_a] - residual[:, self.pair_b]).abs().amax(-1)
        rough = torch.maximum(smooth_gate(residual_rms, cfg.residual_transition_m),
                              smooth_gate(residual_jump, cfg.jump_transition_m))
        slope = (1 - rough) * smooth_gate(slope_deg, cfg.slope_transition_deg)
        flat = 1 - rough - slope
        # Invalid scans retain weak nonzero posture constraints. They never
        # masquerade as flat ground or turn all orientation costs off.
        flat = torch.where(valid, flat, 0.0)
        rough = torch.where(valid, rough, 0.0)
        slope = torch.where(valid, slope, 1.0)

        data = self.robot.data
        commands = env.command_manager.get_command("base_velocity")
        stopping = (commands[:, :2].norm(dim=-1) < cfg.stop_linear_threshold) & (
            commands[:, 2].abs() < cfg.stop_yaw_threshold
        )
        quiet = stopping & (data.root_lin_vel_b[:, :2].norm(dim=-1) < cfg.quiet_linear_threshold) & (
            data.root_ang_vel_b[:, 2].abs() < cfg.quiet_yaw_threshold
        )
        self.quiet_time = torch.where(quiet, self.quiet_time + env.step_dt, 0.0)
        stand = smooth_gate(self.quiet_time, (cfg.stand_dwell_s, cfg.stand_dwell_s + cfg.stand_ramp_s))
        maneuver = torch.maximum(smooth_gate(commands[:, 1].abs(), cfg.lateral_transition),
                                 smooth_gate(commands[:, 2].abs(), cfg.yaw_transition))
        # Command-based: drifting while commanded to stop must not disable the
        # wheel-position costs. No all-wheels-contact condition that the policy
        # could evade by slightly lifting a wheel. Z motion itself is unpenalized.
        self.flat_straight_gate = flat * (1 - maneuver)
        flat_hip = 1 + maneuver * (cfg.hip_maneuver_scale - 1)
        flat_leg = 1 + maneuver * (cfg.leg_maneuver_scale - 1)
        flat_hip = flat_hip + stand * (cfg.hip_stand_scale - flat_hip)
        flat_leg = flat_leg + stand * (cfg.leg_stand_scale - flat_leg)
        self.hip_scale = flat * flat_hip + slope * cfg.hip_slope_scale + rough * cfg.hip_rough_scale
        self.leg_scale = flat * flat_leg + slope * cfg.leg_slope_scale + rough * cfg.leg_rough_scale

        # World-vertical tilt; yaw is unconstrained. Permit more lean on slopes
        # without requiring the body to match an estimated stair/ramp plane.
        slope_allowance = (slope_deg + cfg.slope_tilt_margin_deg).clamp(max=cfg.slope_tilt_cap_deg)
        allowance_deg = (flat * cfg.flat_tilt_allowance_deg + slope * slope_allowance
                         + rough * cfg.rough_tilt_allowance_deg)
        allowance_deg = torch.where(valid, allowance_deg, cfg.invalid_tilt_allowance_deg)
        gravity = data.projected_gravity_b
        tilt = torch.atan2(gravity[:, :2].norm(dim=-1), -gravity[:, 2])
        tilt_scale = flat + (1 - flat) * cfg.nonflat_tilt_scale
        self.tilt_cost = tilt_scale * (tilt - allowance_deg * math.pi / 180.0).clamp_min(0).square()

        values = (flat, slope, rough, valid.float(), self.hip_scale, self.leg_scale,
                  stand, torch.rad2deg(tilt), allowance_deg)
        for total, value in zip(self.sums.values(), values):
            total += value
        self.samples += 1
        return self

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return {}
        count = self.samples[env_ids].clamp_min(1)
        log = {f"X5V7/pose/{name}": (total[env_ids] / count).mean() for name, total in self.sums.items()}
        for total in self.sums.values():
            total[env_ids] = 0
        self.samples[env_ids] = 0
        self.quiet_time[env_ids] = 0
        self.hip_scale[env_ids] = 1
        self.leg_scale[env_ids] = 1
        self.tilt_cost[env_ids] = 0
        self.flat_straight_gate[env_ids] = 0
        return log
