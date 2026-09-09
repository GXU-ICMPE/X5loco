"""Explicit diagnostic overrides, NEVER registered as the training task default."""

from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import robot_lab.tasks.go2.mdp as mdp


@configclass
class ZeroCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(5.0, 5.0),
        rel_standing_envs=1.0, heading_command=False, debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class DeterministicEventsCfg:
    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")


def mechanical_check_cfg(training_cfg):
    """Return an isolated copy; no silent changes to the formal task."""
    # configclass.copy() calls dataclasses.replace(), re-running post_init with
    # temporarily shared nested fields. Parent post_init would mutate the input
    # sim.dt before the child config is deep-copied. Diagnostic copies must not.
    cfg = deepcopy(training_cfg)
    cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=deepcopy(training_cfg.scene.terrain.physics_material), debug_vis=False,
    )
    cfg.scene.sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )
    cfg.scene.env_spacing = 3.0
    cfg.sim.physics_material = cfg.scene.terrain.physics_material
    cfg.commands = ZeroCommandsCfg()
    cfg.events = DeterministicEventsCfg()
    cfg.curriculum = None
    for group in (cfg.observations.policy, cfg.observations.single_obs, cfg.observations.critic):
        group.enable_corruption = False
    return cfg
