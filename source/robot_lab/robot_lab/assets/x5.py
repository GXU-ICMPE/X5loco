"""Selected X5 model and explicit PD bring-up configuration.

The gains are simulation starting points, not hardware-accepted parameters.
See resources/x5/manifest.json for the user-selected model provenance.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.utils import clone, make_uninstanceable

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


X5_LEG_JOINT_NAMES = [
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
]
X5_WHEEL_JOINT_NAMES = ["RF_WHEEL", "LF_WHEEL", "RH_WHEEL", "LH_WHEEL"]
X5_JOINT_NAMES = X5_LEG_JOINT_NAMES + X5_WHEEL_JOINT_NAMES
X5_FOOT_BODY_NAMES = ["RF_FOOT", "LF_FOOT", "RH_FOOT", "LH_FOOT"]
X5_WHEEL_RADIUS = 0.1005
X5_BASE_HEIGHT = 0.5675
X5_LEG_TARGET_LIMITS = {
    "R[FH]_HAA": (-0.79, 0.64),
    "L[FH]_HAA": (-0.64, 0.79),
    ".*_HFE": (-1.60, 2.39),
    ".*_KFE": (-2.32, -0.52),
}


@clone
def spawn_x5_from_urdf(prim_path, cfg, translation=None, orientation=None):
    """Apply contact overrides to editable colliders; keep visuals instanced.

    The installed URDF importer instances collision subtrees even when the
    generic converter's make_instanceable flag is False. Modify only the local
    scene, before environment cloning; never edit the URDF or cached USD files.
    """
    prim = sim_utils.spawn_from_urdf(
        prim_path, cfg.replace(collision_props=None), translation=translation, orientation=orientation,
    )
    for body in prim.GetChildren():
        collisions = body.GetChild("collisions")
        if collisions.IsValid():
            make_uninstanceable(collisions.GetPath())
    if cfg.collision_props is not None:
        sim_utils.modify_collision_properties(prim_path, cfg.collision_props)
    return prim


X5_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=spawn_x5_from_urdf,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/x5/urdf/x5.urdf",
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=True,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.002, rest_offset=0.0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5875),
        joint_pos={".*_HAA": 0.0, ".*_HFE": 0.75, ".*_KFE": -1.5, ".*_WHEEL": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_(HAA|HFE|KFE)"],
            stiffness=800.0,
            damping=20.0,
            effort_limit={".*_HAA": 300.0, ".*_HFE": 250.0, ".*_KFE": 250.0},
            velocity_limit={".*_HAA": 6.0, ".*_HFE": 6.0, ".*_KFE": 8.0},
            effort_limit_sim={".*_HAA": 300.0, ".*_HFE": 250.0, ".*_KFE": 250.0},
            velocity_limit_sim={".*_HAA": 6.0, ".*_HFE": 6.0, ".*_KFE": 8.0},
            friction=0.0,
        ),
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=X5_WHEEL_JOINT_NAMES,
            stiffness=0.0,
            damping=3.0,
            effort_limit=30.0,
            velocity_limit=30.0,
            effort_limit_sim=30.0,
            velocity_limit_sim=30.0,
            friction=0.0,
        ),
    },
)
