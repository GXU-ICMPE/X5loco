"""V8: deployment standPose, recoverable maneuver posture, and axis curricula."""

from isaaclab.managers import RewardTermCfg as RewTerm, CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from robot_lab.tasks.x5_v7.env_cfg import X5V7EnvCfg, X5V7RewardsCfg, X5V7CurriculumCfg, X5V7CommandsCfg
from .commands import X5V8CommandCfg, command_axis_levels_vel
from .parameters import X5V8PostureParameters
from . import rewards


@configclass
class X5V8CommandsCfg(X5V7CommandsCfg):
    base_velocity: X5V8CommandCfg = X5V8CommandCfg()


@configclass
class X5V8RewardsCfg(X5V7RewardsCfg):
    joint_pose_corridor = RewTerm(func=rewards.motion_cost, weight=-0.15, params={"name": "joint_pose"})
    wheel_position_corridor = RewTerm(func=rewards.motion_cost, weight=-0.25, params={"name": "wheel_position"})
    wheelbase_corridor = RewTerm(func=rewards.motion_cost, weight=-0.15, params={"name": "wheelbase"})
    track_width_corridor = RewTerm(func=rewards.motion_cost, weight=-0.15, params={"name": "track_width"})
    height_corridor = RewTerm(func=rewards.motion_cost, weight=-0.35, params={"name": "height"})
    lateral_progress = RewTerm(func=rewards.lateral_progress, weight=0.75, params={"std": 0.25})
    maneuver_completed_step = RewTerm(func=rewards.motion_cost, weight=0.30, params={"name": "maneuver_step"})

    def __post_init__(self):
        super().__post_init__()
        # Disable after the parent has resolved X5 selections, rather than
        # passing None through parent __post_init__ implementations.
        for name in ("hip_pos_penalty_l1", "joint_pos_penalty_l1", "base_height_l2",
                     "flat_wheel_position_l2", "flat_wheelbase_l2", "flat_track_width_l2"):
            setattr(self, name, None)
        # Keep a fixed, moderate kernel while the command range itself adapts.
        self.track_lin_vel_xy_exp.params["std"] = 0.4
        self.track_ang_vel_z_exp.params["std"] = 0.4


@configclass
class X5V8CurriculumCfg(X5V7CurriculumCfg):
    base_height_l2 = None
    tracking_precision = None
    command_x = CurrTerm(func=command_axis_levels_vel, params={"command_axis": "x"})
    command_y = CurrTerm(func=command_axis_levels_vel, params={"command_axis": "y"})
    command_z = CurrTerm(func=command_axis_levels_vel, params={"command_axis": "z"})


@configclass
class X5V8EnvCfg(X5V7EnvCfg):
    commands: X5V8CommandsCfg = X5V8CommandsCfg()
    rewards: X5V8RewardsCfg = X5V8RewardsCfg()
    curriculum: X5V8CurriculumCfg = X5V8CurriculumCfg()
    v8_posture: X5V8PostureParameters = X5V8PostureParameters()

    def __post_init__(self):
        super().__post_init__()
        haa, hfe, kfe = self.v8_posture.stand_joint_angles
        self.scene.robot.init_state.joint_pos = {
            ".*_HAA": haa, ".*_HFE": hfe, ".*_KFE": kfe, ".*_WHEEL": 0.0,
        }
        # Geometric stance height plus a small spawn clearance, not the MPC
        # defaultBodyHeight parameter (which does not define standPose FK).
        self.scene.robot.init_state.pos = (0.0, 0.0, self.v8_posture.reference_base_height_m + 0.02)
