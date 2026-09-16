"""X5 v0 with separately tuned physical costs and narrower randomization."""

import isaaclab.envs.mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.assets.x5 import X5_LEG_JOINT_NAMES, X5_WHEEL_JOINT_NAMES
from robot_lab.tasks.legbot.env_cfg import EventCfg
from robot_lab.tasks.x5.env_cfg import X5EnvCfg, X5ObservationsCfg, X5RewardsCfg, X5SceneCfg


@configclass
class X5V6RewardsCfg(X5RewardsCfg):
    # Replace the two combined costs with disjoint leg/wheel sums. Keep the
    # original physical-value formulas; there is no additional normalization.
    joint_acc_l2 = None
    joint_torques_l2 = None

    leg_joint_acc_l2 = RewTerm(
        func=isaac_mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=X5_LEG_JOINT_NAMES, preserve_order=True),
        },
    )
    wheel_joint_acc_l2 = RewTerm(
        func=isaac_mdp.joint_acc_l2,
        weight=-2.0e-8,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=X5_WHEEL_JOINT_NAMES, preserve_order=True),
        },
    )
    leg_joint_torques_l2 = RewTerm(
        func=isaac_mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=X5_LEG_JOINT_NAMES, preserve_order=True),
        },
    )
    wheel_joint_torques_l2 = RewTerm(
        func=isaac_mdp.joint_torques_l2,
        weight=-4.0e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=X5_WHEEL_JOINT_NAMES, preserve_order=True),
        },
    )


@configclass
class X5V6ObservationsCfg(X5ObservationsCfg):
    def __post_init__(self):
        super().__post_init__()
        # Both actor observation groups need the same physical noise ranges.
        # Keep the privileged critic clean and preserve the 530/53/291 layout.
        for group in (self.policy, self.single_obs):
            group.joint_pos.noise.n_min = -0.01
            group.joint_pos.noise.n_max = 0.01
            group.joint_vel.noise.n_min = -0.5
            group.joint_vel.noise.n_max = 0.5


@configclass
class X5V6EventCfg(EventCfg):
    # Additive offsets also randomize HAA joints whose default angle is zero.
    # Select only the legs so wheel state is reset separately and exactly once.
    reset_robot_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=X5_LEG_JOINT_NAMES, preserve_order=True),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_wheel_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=X5_WHEEL_JOINT_NAMES, preserve_order=True),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # Retain the base mass perturbation of +/-1 kg. The remaining mass,
        # COM and material terms are still sampled at startup, not on a course.
        self.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.95, 1.05)
        self.randomize_com_positions.params["com_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (-0.01, 0.01),
        }

        self.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.95, 1.05)
        self.randomize_actuator_gains.params["damping_distribution_params"] = (0.95, 1.05)
        self.randomize_motor_zero_offset.params["offset_range"] = (-0.005, 0.005)

        self.randomize_rigid_body_material.params["static_friction_range"] = (0.6, 1.2)
        self.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.6, 1.2)
        self.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.05)
        self.randomize_rigid_body_material.params["make_consistent"] = True

        self.randomize_push_robot.interval_range_s = (10.0, 15.0)
        # The installed push function ADDS these increments to root velocity.
        self.randomize_push_robot.params["velocity_range"] = {
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-0.15, 0.15),
        }

        # Preserve the v0 spawn pose distribution and its pit-terrain special
        # case. Only narrow the initial velocity on non-pit environments.
        self.reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (0.0, 0.0),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        }


@configclass
class X5V6EnvCfg(X5EnvCfg):
    scene: X5SceneCfg = X5SceneCfg(num_envs=512, env_spacing=0.5)
    rewards: X5V6RewardsCfg = X5V6RewardsCfg()
    observations: X5V6ObservationsCfg = X5V6ObservationsCfg()
    events: X5V6EventCfg = X5V6EventCfg()
