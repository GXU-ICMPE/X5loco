"""X5 reward-only experiment; physical penalties and the complete task are inherited."""

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.assets.x5 import X5_FOOT_BODY_NAMES, X5_WHEEL_JOINT_NAMES, X5_WHEEL_RADIUS
from robot_lab.tasks.x5.env_cfg import X5EnvCfg, X5RewardsCfg
from .mdp import StandStillWheelVelocity, CompletedWheelStep
from .rewards import joint_pose_deadband_l1, wheel_air_spin_l2


@configclass
class X5RewardsV1Cfg(X5RewardsCfg):
    stand_still_wheel_l2 = RewTerm(
        func=StandStillWheelVelocity, weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=X5_WHEEL_JOINT_NAMES, preserve_order=True),
            "command_name": "base_velocity", "wheel_radius": X5_WHEEL_RADIUS,
            "speed_deadband": 0.05, "linear_threshold": 0.05, "yaw_threshold": 0.1,
            "settle_time": 0.3, "reset_grace": 0.5,
        },
    )
    wheel_air_spin_l2 = RewTerm(
        func=wheel_air_spin_l2, weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=X5_WHEEL_JOINT_NAMES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=X5_FOOT_BODY_NAMES, preserve_order=True),
            "wheel_radius": X5_WHEEL_RADIUS, "speed_deadband": 0.5,
            "air_time_grace": 0.15, "contact_threshold": 1.0, "reset_grace": 0.5,
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # Replace, rather than stack onto, the existing point-pose penalties.
        # Their original negative weights and X5 joint selections are retained.
        for name, deadband in (("hip_pos_penalty_l1", 0.08), ("joint_pos_penalty_l1", 0.15)):
            term = getattr(self, name)
            term.func = joint_pose_deadband_l1
            term.params = {"asset_cfg": term.params["asset_cfg"], "deadband": deadband}


@configclass
class X5RewardsV1EnvCfg(X5EnvCfg):
    rewards: X5RewardsV1Cfg = X5RewardsV1Cfg()


@configclass
class X5StepsV1RewardsCfg(X5RewardsV1Cfg):
    completed_step = RewTerm(
        func=CompletedWheelStep, weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=X5_FOOT_BODY_NAMES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=X5_FOOT_BODY_NAMES, preserve_order=True),
            "command_name": "base_velocity", "terrain_names": ("stairs_up", "stairs_down", "obstacles"),
            "min_command": 0.1, "lateral_threshold": 0.1,
            "min_air_time": 0.1, "max_air_time": 0.6,
            "min_lift": 0.04, "min_relative_lift": 0.02, "target_relative_lift": 0.06,
            "min_foot_progress": 0.04, "min_base_progress": 0.01, "target_base_progress": 0.04,
            "min_support": 2, "reset_grace": 0.5, "command_cos_min": 0.866,
            "support_force_threshold": 5.0,
        },
    )


@configclass
class X5StepsV1EnvCfg(X5RewardsV1EnvCfg):
    rewards: X5StepsV1RewardsCfg = X5StepsV1RewardsCfg()
