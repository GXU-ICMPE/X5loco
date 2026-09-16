"""X5 on RobotLab-Legbot-v0, with robot-specific unit calibration only.

Terrain, commands, curricula, randomization, termination and reward weights are
inherited. Mechanical-test overrides live separately in validation_cfg.py.
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.assets.x5 import (
    X5_CFG, X5_FOOT_BODY_NAMES, X5_JOINT_NAMES, X5_LEG_JOINT_NAMES,
    X5_LEG_TARGET_LIMITS, X5_WHEEL_JOINT_NAMES, X5_WHEEL_RADIUS,
)
import robot_lab.tasks.go2.mdp as mdp
from robot_lab.tasks.legbot.env_cfg import (
    LegbotEnvCfg, LegbotSceneCfg, ObservationsCfg, RewardsCfg,
)
from .parameters import (
    CONTACT_FORCE_CLIP, JOINT_ACCELERATION_CLIP,
    JOINT_TORQUE_CLIP, BASE_HEIGHT_TARGET,
    HEIGHT_SCAN_OFFSET,
)


@configclass
class X5SceneCfg(LegbotSceneCfg):
    robot: ArticulationCfg = X5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class X5ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=X5_LEG_JOINT_NAMES,
        scale=0.2, use_default_offset=True, clip=X5_LEG_TARGET_LIMITS,
        preserve_order=True,
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=X5_WHEEL_JOINT_NAMES,
        scale=1.0 / X5_WHEEL_RADIUS, use_default_offset=False,
        clip={".*": (-24.0, 24.0)}, preserve_order=True,
    )


@configclass
class X5ObservationsCfg(ObservationsCfg):
    def __post_init__(self):
        for group in (self.policy, self.single_obs, self.critic):
            group.joint_pos.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=X5_LEG_JOINT_NAMES, preserve_order=True,
            )
            group.joint_vel.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=X5_JOINT_NAMES, preserve_order=True,
            )
        for term in (self.critic.joint_acc, self.critic.joint_torque):
            term.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=X5_JOINT_NAMES, preserve_order=True,
            )
        self.critic.contact_force.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=X5_FOOT_BODY_NAMES, preserve_order=True,
        )
        # ObservationManager clips physical values BEFORE applying their scale.
        self.critic.contact_force.clip = CONTACT_FORCE_CLIP
        self.critic.joint_torque.clip = JOINT_TORQUE_CLIP
        self.critic.joint_acc.clip = JOINT_ACCELERATION_CLIP
        self.critic.height_scan.params["offset"] = HEIGHT_SCAN_OFFSET


@configclass
class X5RewardsCfg(RewardsCfg):
    def __post_init__(self):
        # Inherit Legbot's raw physical-value penalty functions unchanged.
        # Only select the corresponding X5 joints (12 legs + 4 wheels).
        for term in (self.joint_acc_l2, self.joint_power, self.joint_torques_l2):
            # Derived reward experiments may replace a combined term with
            # separate leg/wheel terms and disable the original using None.
            if term is None:
                continue
            term.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=X5_JOINT_NAMES, preserve_order=True,
            )
        if self.joint_torques_l2 is not None:
            self.joint_torques_l2.weight = -1.0e-4 / 2.5
        self.base_height_l2.params["target_height"] = BASE_HEIGHT_TARGET
        self.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=".*_(thigh|calf)",
        )
        self.joint_pos_limits.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=X5_LEG_JOINT_NAMES)
        self.hip_pos_penalty_l1.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=".*_HAA")
        self.joint_pos_penalty_l1.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=".*_(HFE|KFE)")


@configclass
class X5EnvCfg(LegbotEnvCfg):
    scene: X5SceneCfg = X5SceneCfg(num_envs=16384, env_spacing=0.5)
    observations: X5ObservationsCfg = X5ObservationsCfg()
    actions: X5ActionsCfg = X5ActionsCfg()
    rewards: X5RewardsCfg = X5RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Previously checked X5 explicit-PD physics step. Policy period, ray
        # updates and contact history still sample at 20 ms, 20 ms and 5 ms.
        self.sim.dt = 0.0025
        self.decimation = 8
        self.sim.render_interval = self.decimation
