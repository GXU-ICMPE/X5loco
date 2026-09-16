"""Wheel P control during motion, zero-reference PI only during commanded stops."""

from dataclasses import dataclass
from math import isfinite

import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.envs.mdp.actions import JointVelocityAction, JointVelocityActionCfg
from isaaclab.utils import configclass

from robot_lab.assets.x5 import X5_FOOT_BODY_NAMES, X5_WHEEL_JOINT_NAMES


@dataclass
class X5StopPIParameters:
    enabled: bool = True
    enter_linear_mps: float = 0.02
    exit_linear_mps: float = 0.03
    enter_yaw_rps: float = 0.04
    exit_yaw_rps: float = 0.05
    dwell_s: float = 0.20
    ramp_s: float = 0.20
    contact_dwell_s: float = 0.02
    contact_force_n: float = 5.0
    ki: float = 3.0  # Nm/rad; the inherited wheel damping is velocity P gain.
    integral_limit_nm: float = 12.0
    torque_limit_nm: float = 30.0
    max_dt_s: float = 0.05

    def validate(self):
        for name, value in vars(self).items():
            if name != "enabled" and (not isfinite(value) or value <= 0):
                raise ValueError(f"Stop PI {name} must be finite and positive")
        if not (self.enter_linear_mps < self.exit_linear_mps
                and self.enter_yaw_rps < self.exit_yaw_rps
                and self.integral_limit_nm <= self.torque_limit_nm):
            raise ValueError("Stop PI needs ordered hysteresis thresholds and bounded integral torque")


class X5StopWheelActuator(IdealPDActuator):
    """Integrates once per physics substep, not once per scene buffer write."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        cfg.stop_pi.validate()
        if len(self.joint_names) != 4 or set(self.joint_names) != set(X5_WHEEL_JOINT_NAMES):
            raise ValueError("X5 stop PI requires exactly the four X5 wheel joints")
        self.integral = torch.zeros_like(self.damping)
        self.contact_time = torch.zeros_like(self.integral)
        self.stop_requested = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self.holding = torch.zeros_like(self.stop_requested)
        self.stop_time = torch.zeros(self._num_envs, device=self._device)
        self.commands = torch.zeros(self._num_envs, 3, device=self._device)
        self.contacts = torch.zeros_like(self.integral, dtype=torch.bool)
        self.reference = torch.zeros_like(self.integral)
        self._tick = self._last_tick = -1
        self._dt = 0.0
        self.metric_sums = {name: torch.zeros_like(self.stop_time) for name in (
            "elapsed_s", "holding_s", "integral_sq", "speed_sq", "saturated_s",
        )}

    def set_stop_context(self, commands, contacts, dt, tick):
        if not isfinite(dt) or not 0 < dt <= self.cfg.stop_pi.max_dt_s:
            raise ValueError("X5 stop PI requires a finite, positive physics timestep")
        if tick < self._last_tick:
            self.reset(slice(None))
            self._last_tick = -1
        self.commands = commands
        self.contacts = contacts
        self._dt, self._tick = dt, tick

    def reset(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.integral, self.contact_time, self.stop_requested, self.holding,
                      self.stop_time, self.reference):
            value[ids] = 0
        for value in self.metric_sums.values():
            value[ids] = 0

    def compute(self, control_action, joint_pos, joint_vel):
        cfg = self.cfg.stop_pi
        advance = self._tick >= 0 and self._tick != self._last_tick
        if advance:
            self._last_tick = self._tick
            linear = self.commands[:, :2].norm(dim=-1)
            yaw = self.commands[:, 2].abs()
            enter = (linear < cfg.enter_linear_mps) & (yaw < cfg.enter_yaw_rps)
            leave = (linear >= cfg.exit_linear_mps) | (yaw >= cfg.exit_yaw_rps)
            self.stop_requested = (enter | (self.stop_requested & ~leave)) & cfg.enabled
            self.stop_time = torch.where(self.stop_requested, self.stop_time + self._dt, 0.0)
            self.contact_time = torch.where(self.contacts, self.contact_time + self._dt, 0.0)
            self.holding = self.stop_requested & (self.stop_time >= cfg.dwell_s + cfg.ramp_s)

        blend = ((self.stop_time - cfg.dwell_s) / cfg.ramp_s).clamp(0, 1)
        blend = blend.square() * (3 - 2 * blend)
        self.reference = control_action.joint_velocities * (1 - blend[:, None])
        error = self.reference - joint_vel
        p_effort = (self.stiffness * (control_action.joint_positions - joint_pos)
                    + self.damping * error + control_action.joint_efforts)
        limit = self.effort_limit.clamp(max=cfg.torque_limit_nm)
        if advance:
            eligible = (self.holding[:, None] & self.contacts
                        & (self.contact_time >= cfg.contact_dwell_s))
            # Both output and state are zeroed on motion, reset, or loss of
            # contact. Freezing a nonzero I on restart would retain braking.
            self.integral = torch.where(eligible, self.integral, 0.0)
            raw = p_effort + self.integral
            can_integrate = (raw.abs() < limit) | (error * raw <= 0)
            increment = torch.where(eligible & can_integrate, cfg.ki * error * self._dt, 0.0)
            self.integral = (self.integral + increment).clamp(-cfg.integral_limit_nm, cfg.integral_limit_nm)

        self.computed_effort = p_effort + self.integral
        self.applied_effort = torch.clamp(self.computed_effort, -limit, limit)
        if advance:
            held_dt = self.holding.float() * self._dt
            self.metric_sums["elapsed_s"] += self._dt
            self.metric_sums["holding_s"] += held_dt
            self.metric_sums["integral_sq"] += self.integral.square().mean(-1) * held_dt
            self.metric_sums["speed_sq"] += joint_vel.square().mean(-1) * held_dt
            self.metric_sums["saturated_s"] += (self.computed_effort.abs() >= limit).float().mean(-1) * held_dt
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def episode_metrics(self, env_ids):
        sums = {name: value[env_ids].sum() for name, value in self.metric_sums.items()}
        held = sums["holding_s"].clamp_min(1.0e-6)
        return {
            "X5V7/stop_pi/seconds": self.metric_sums["holding_s"][env_ids].mean(),
            "X5V7/stop_pi/holding_fraction": sums["holding_s"] / sums["elapsed_s"].clamp_min(1.0e-6),
            "X5V7/stop_pi/integral_torque_rms_nm": (sums["integral_sq"] / held).sqrt(),
            "X5V7/stop_pi/wheel_speed_rms_rps": (sums["speed_sq"] / held).sqrt(),
            "X5V7/stop_pi/saturation_fraction": sums["saturated_s"] / held,
        }


@configclass
class X5StopWheelActuatorCfg(IdealPDActuatorCfg):
    class_type: type = X5StopWheelActuator
    stop_pi: X5StopPIParameters = X5StopPIParameters()


class X5StopWheelVelocityAction(JointVelocityAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.actuator = self._asset.actuators["wheels"]
        if not isinstance(self.actuator, X5StopWheelActuator):
            raise ValueError("X5 stop action requires X5StopWheelActuator")
        self.sensor = env.scene["contact_forces"]
        # Isaac actuator groups follow articulation order, while policy actions
        # use preserve_order=True. Match contacts by name in ACTUATOR order.
        contact_names = [name.replace("_WHEEL", "_FOOT") for name in self.actuator.joint_names]
        self.contact_ids, names = self.sensor.find_bodies(contact_names, preserve_order=True)
        if list(names) != contact_names or self._joint_names != X5_WHEEL_JOINT_NAMES:
            raise ValueError("X5 stop action contact mapping or policy joint order is invalid")

    def apply_actions(self):
        # Sensor update period remains the existing 5 ms; use its current force,
        # not the max of the history, which would keep an airborne wheel active.
        forces = self.sensor.data.net_forces_w[:, self.contact_ids]
        contacts = forces.norm(dim=-1) > self.actuator.cfg.stop_pi.contact_force_n
        self.actuator.set_stop_context(
            self._env.command_manager.get_command("base_velocity"), contacts,
            self._env.physics_dt, self._env._sim_step_counter,
        )
        super().apply_actions()


@configclass
class X5StopWheelVelocityActionCfg(JointVelocityActionCfg):
    class_type: type = X5StopWheelVelocityAction
