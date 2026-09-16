"""V7 posture, stop PI, stop-only wheel costs, and moderate tracking precision."""

from isaaclab.managers import RewardTermCfg as RewTerm, CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from robot_lab.tasks.legbot.env_cfg import CommandsCfg, CurriculumCfg
from robot_lab.tasks.x5.env_cfg import X5ActionsCfg
from robot_lab.assets.x5 import X5_WHEEL_JOINT_NAMES, X5_WHEEL_RADIUS
from robot_lab.tasks.x5_v6.env_cfg import X5V6EnvCfg, X5V6RewardsCfg

from .commands import X5V7CommandCfg
from .parameters import X5V7PostureParameters, X5V7WheelSupportParameters
from . import rewards
from .curriculums import tracking_precision
from .stop_pi import X5StopWheelActuatorCfg, X5StopWheelVelocityActionCfg


@configclass
class X5V7CommandsCfg(CommandsCfg):
    base_velocity: X5V7CommandCfg = X5V7CommandCfg()


@configclass
class X5V7RewardsCfg(X5V6RewardsCfg):
    terrain_roll_pitch_l2 = RewTerm(func=rewards.terrain_roll_pitch_l2, weight=-1.0)
    flat_wheel_position_l2 = RewTerm(func=rewards.flat_wheel_position_l2, weight=-0.2)
    flat_wheelbase_l2 = RewTerm(func=rewards.flat_wheelbase_l2, weight=-0.1)
    flat_track_width_l2 = RewTerm(func=rewards.flat_track_width_l2, weight=-0.1)
    stop_wheel_speed_l2 = RewTerm(func=rewards.stop_wheel_speed_l2, weight=-0.1,
                                params={"rim_speed_scale": 0.2})
    stop_wheel_target_l2 = RewTerm(func=rewards.stop_wheel_target_l2, weight=-0.05,
                                 params={"rim_speed_scale": 0.5})

    def __post_init__(self):
        super().__post_init__()
        for name, function in (("hip_pos_penalty_l1", rewards.terrain_hip_pose_l1),
                               ("joint_pos_penalty_l1", rewards.terrain_leg_pose_l1)):
            term = getattr(self, name)
            term.func = function
            # Keep V6's resolved X5 joint selection; remove the old standstill
            # function parameters, which do not belong to the new signatures.
            term.params = {"asset_cfg": term.params["asset_cfg"]}


@configclass
class X5V7ActionsCfg(X5ActionsCfg):
    wheel_vel = X5StopWheelVelocityActionCfg(
        asset_name="robot", joint_names=X5_WHEEL_JOINT_NAMES,
        scale=1.0 / X5_WHEEL_RADIUS, use_default_offset=False,
        clip={".*": (-24.0, 24.0)}, preserve_order=True,
    )


@configclass
class X5V7CurriculumCfg(CurriculumCfg):
    tracking_precision = CurrTerm(func=tracking_precision, params={
        "start_it": 1000, "end_it": 5000, "initial_std": 0.5, "final_std": 0.4,
        "steps_per_iteration": 24,
    })


@configclass
class X5V7EnvCfg(X5V6EnvCfg):
    commands: X5V7CommandsCfg = X5V7CommandsCfg()
    rewards: X5V7RewardsCfg = X5V7RewardsCfg()
    actions: X5V7ActionsCfg = X5V7ActionsCfg()
    curriculum: X5V7CurriculumCfg = X5V7CurriculumCfg()
    v7_posture: X5V7PostureParameters = X5V7PostureParameters()
    v7_wheel_support: X5V7WheelSupportParameters = X5V7WheelSupportParameters()

    def __post_init__(self):
        super().__post_init__()
        # Copy the existing X5 wheel actuator's physical parameters into the
        # V7-only class; shared V0/V6 assets and leg PD gains stay independent.
        wheel_cfg = self.scene.robot.actuators["wheels"]
        if not isinstance(wheel_cfg, X5StopWheelActuatorCfg):
            settings = {key: value for key, value in vars(wheel_cfg).items() if key != "class_type"}
            self.scene.robot.actuators["wheels"] = X5StopWheelActuatorCfg(**settings)
