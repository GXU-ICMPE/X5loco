"""X5 baseline fidelity, full-task interface and mechanical checks (no RL updates).

Examples:
    python scripts/tools/check_x5_sim.py --headless --config-only
    python scripts/tools/check_x5_sim.py --headless --runner-init-only --num-envs 4
    python scripts/tools/check_x5_sim.py --headless --mode interface --num-envs 4 --steps 256
    python scripts/tools/check_x5_sim.py --headless --mode stance --steps 1000
    python scripts/tools/check_x5_sim.py --headless --mode axes

Axis mode explicitly fixes the base in the air and disables gravity. It tests
joint/control direction, not ground locomotion or whole-body stability.
"""

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

# Do not depend on which editable project / RSL-RL was installed in this Python.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/robot_lab"))
sys.path.insert(0, str(ROOT / "source/rsl_rl"))

import h5py  # noqa: F401 - preload native modules before Kit
import tensordict  # noqa: F401
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config-only", action="store_true")
parser.add_argument("--runner-init-only", action="store_true", help="Construct the real CTS Runner and optimizers, but never train.")
parser.add_argument("--mode", choices=("interface", "stance", "axes"), default="stance")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000, help="Environment-only steps at 50 Hz; no optimizer.")
parser.add_argument("--output", type=Path, help="Optional JSON evidence file.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 1:
    parser.error("--num-envs and --steps must be positive")
if args.runner_init_only and (args.config_only or args.num_envs % 4 != 0):
    parser.error("--runner-init-only requires a multiple of 4 environments and cannot use --config-only")
launcher = AppLauncher(args)
app = launcher.app


def check_config(cfg):
    from rsl_rl.algorithms import MoECTS
    from robot_lab.assets.x5 import X5_FOOT_BODY_NAMES, X5_JOINT_NAMES, X5_LEG_JOINT_NAMES
    from robot_lab.tasks.legbot.env_cfg import LegbotEnvCfg
    from robot_lab.tasks.legbot.rsl_rl_cfg import LegbotMoECTSRunnerCfg
    from robot_lab.tasks.x5.config_audit import audit_legbot_equivalence
    from robot_lab.tasks.x5.env_cfg import X5EnvCfg
    from robot_lab.tasks.x5.rsl_rl_cfg import X5MoECTSRunnerCfg
    from robot_lab.tasks.x5.validation_cfg import mechanical_check_cfg

    original = LegbotEnvCfg()
    another = X5EnvCfg()
    assert cfg.sim.dt == 0.0025 and cfg.decimation == 8
    audit = audit_legbot_equivalence(original, cfg, LegbotMoECTSRunnerCfg(), X5MoECTSRunnerCfg())
    assert cfg.scene.terrain.terrain_type == "generator" and cfg.curriculum is not None
    assert cfg.scene.num_envs == 16384 and cfg.scene.contact_forces.update_period == 0.005
    assert cfg.actions.joint_pos.joint_names == X5_LEG_JOINT_NAMES
    assert cfg.actions.joint_pos.preserve_order and cfg.actions.wheel_vel.preserve_order
    for group in (cfg.observations.policy, cfg.observations.single_obs, cfg.observations.critic):
        assert group.joint_pos.params["asset_cfg"].joint_names == X5_LEG_JOINT_NAMES
        assert group.joint_vel.params["asset_cfg"].joint_names == X5_JOINT_NAMES
    assert cfg.observations.policy.history_length == 10
    assert cfg.observations.single_obs.history_length == 1
    assert cfg.observations.critic.contact_force.params["sensor_cfg"].body_names == X5_FOOT_BODY_NAMES
    assert cfg.observations.policy.enable_corruption and cfg.observations.single_obs.enable_corruption
    assert not cfg.observations.critic.enable_corruption
    penalty_checks = {}
    for name in ("joint_acc_l2", "joint_torques_l2", "joint_power"):
        reference = getattr(original.rewards, name)
        actual = getattr(cfg.rewards, name)
        assert actual.func is reference.func, f"{name}: must use the original Legbot function"
        assert actual.weight == reference.weight
        assert set(actual.params) == set(reference.params) == {"asset_cfg"}
        assert actual.params["asset_cfg"].joint_names == X5_JOINT_NAMES
        assert actual.params["asset_cfg"].preserve_order
        # Same physical samples, including both power signs; no clipping/scaling.
        from types import SimpleNamespace

        values = torch.arange(-8.0, 8.0).repeat(2, 1)
        data = SimpleNamespace(applied_torque=values * 10,
                               joint_vel=torch.stack((values[0], -values[0])), joint_acc=values * 1000)
        sample_env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
        selection = SimpleNamespace(name="robot", joint_ids=slice(None))
        expected = {"joint_acc_l2": data.joint_acc.square().sum(dim=1),
                    "joint_torques_l2": data.applied_torque.square().sum(dim=1),
                    "joint_power": (data.applied_torque * data.joint_vel).abs().sum(dim=1)}[name]
        torch.testing.assert_close(actual.func(sample_env, asset_cfg=selection), expected)
        penalty_checks[name] = {"function": f"{actual.func.__module__}:{actual.func.__name__}",
                               "weight": actual.weight, "physical_formula": "PASS_UNSCALED"}
    assert cfg.rewards.base_height_l2.params["target_height"] == 0.55
    assert cfg.observations.critic.contact_force.clip == (0.0, 3000.0)
    assert cfg.observations.critic.joint_torque.clip == (-300.0, 300.0)
    assert cfg.observations.critic.joint_acc.clip == (-30000.0, 30000.0)
    assert X5MoECTSRunnerCfg().experiment_name == "x5_moe_cts"
    assert X5MoECTSRunnerCfg().policy.expert_num == 8
    assert X5MoECTSRunnerCfg().max_iterations == 300000 and X5MoECTSRunnerCfg().save_interval == 500
    # Both the old task and separate X5 instances must stay independent.
    assert original.observations.policy.enable_corruption
    assert original.observations.policy.joint_pos.params["asset_cfg"].joint_names[0] == "fl_hip_joint"
    assert original.rewards.base_height_l2.params["target_height"] == 0.42
    assert cfg.actions.joint_pos.clip is not another.actions.joint_pos.clip
    assert cfg.observations.policy.joint_pos.params is not another.observations.policy.joint_pos.params
    diagnostic = mechanical_check_cfg(cfg)
    assert diagnostic.scene.terrain.terrain_type == "plane" and diagnostic.curriculum is None
    assert cfg.scene.terrain.terrain_type == "generator" and cfg.observations.policy.enable_corruption
    # Detect drift in protected fields, even if the task is still instantiable.
    changed = deepcopy(cfg)
    changed.events.randomize_push_robot.interval_range_s = (8.0, 8.0)
    try:
        audit_legbot_equivalence(original, changed, LegbotMoECTSRunnerCfg(), X5MoECTSRunnerCfg())
    except AssertionError:
        pass
    else:
        raise AssertionError("Audit failed to detect randomization drift")
    audit_legbot_equivalence(original, cfg, LegbotMoECTSRunnerCfg(), X5MoECTSRunnerCfg())
    assert cfg.sim.dt == 0.0025 and cfg.sim.render_interval == 8
    # Include inherited Isaac Lab fields, not just this project's explicit overrides.
    for runner_cfg in (LegbotMoECTSRunnerCfg(), X5MoECTSRunnerCfg()):
        algorithm_cfg = runner_cfg.algorithm.to_dict()
        assert algorithm_cfg.pop("class_name") == "MoECTS"
        inspect.signature(MoECTS).bind(None, None, 4, device="cpu", multi_gpu_cfg=None, **algorithm_cfg)
    return {"config_isolation": "PASS", "task": "RobotLab-X5-MoECTS-v0", "baseline_audit": audit,
            "original_penalties": penalty_checks,
            "algorithm_constructor_signature": "PASS_LEGBOT_AND_X5"}


def check_observations(obs, num_envs):
    for group, size in (("policy", 530), ("single_obs", 53), ("critic", 291)):
        assert obs[group].shape == (num_envs, size), (group, obs[group].shape)
        assert torch.isfinite(obs[group]).all(), f"Non-finite {group} observation"


def check_imported_physics(robot):
    """Compare each merged body's mass/COM/inertia with the selected URDF."""
    import numpy as np
    import xml.etree.ElementTree as ET
    from validate_x5_asset import origin

    model = ET.parse(ROOT / "resources/x5/urdf/x5.urdf").getroot()
    fixed = {joint.find("child").get("link"): joint for joint in model.findall("joint") if joint.get("type") == "fixed"}
    parts = {name: [] for name in robot.body_names}
    for link in model.findall("link"):
        body = link.get("name")
        transform = np.eye(4)
        while body in fixed:
            joint = fixed[body]
            transform = origin(joint.find("origin")) @ transform
            body = joint.find("parent").get("link")
        pose = transform @ origin(link.find("inertial/origin"))
        m = float(link.find("inertial/mass").get("value"))
        v = {key: float(value) for key, value in link.find("inertial/inertia").attrib.items()}
        inertia = np.array([[v["ixx"], v["ixy"], v["ixz"]], [v["ixy"], v["iyy"], v["iyz"]], [v["ixz"], v["iyz"], v["izz"]]])
        rotation = pose[:3, :3]
        parts[body].append((m, pose[:3, 3], rotation @ inertia @ rotation.T))
    masses = robot.root_physx_view.get_masses().cpu().numpy()
    coms = robot.root_physx_view.get_coms().cpu().numpy()[..., :3]
    inertias = robot.root_physx_view.get_inertias().cpu().numpy().reshape(robot.num_instances, robot.num_bodies, 3, 3)
    max_mass_error = max_com_error = max_inertia_error = 0.0
    for i, name in enumerate(robot.body_names):
        members = parts[name]
        mass = sum(m for m, _, _ in members)
        center = sum(m * c for m, c, _ in members) / mass
        inertia = np.zeros((3, 3))
        for m, c, tensor in members:
            d = c - center
            inertia += tensor + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        np.testing.assert_allclose(masses[:, i], mass, atol=1e-4, rtol=1e-5, err_msg=name)
        np.testing.assert_allclose(coms[:, i], np.broadcast_to(center, coms[:, i].shape), atol=1e-5, rtol=1e-4, err_msg=name)
        np.testing.assert_allclose(inertias[:, i], np.broadcast_to(inertia, inertias[:, i].shape), atol=1e-5, rtol=1e-3, err_msg=name)
        max_mass_error = max(max_mass_error, float(np.max(np.abs(masses[:, i] - mass))))
        max_com_error = max(max_com_error, float(np.max(np.abs(coms[:, i] - center))))
        max_inertia_error = max(max_inertia_error, float(np.max(np.abs(inertias[:, i] - inertia))))
    # Explicit PD must not be combined with nonzero implicit PhysX drive gains.
    assert torch.count_nonzero(robot.root_physx_view.get_dof_stiffnesses()) == 0
    assert torch.count_nonzero(robot.root_physx_view.get_dof_dampings()) == 0
    return {"merged_bodies_checked": len(parts), "max_mass_error_kg": max_mass_error,
            "max_com_error_m": max_com_error, "max_inertia_error_kg_m2": max_inertia_error,
            "implicit_drive_gains": "zero"}


def check_collisions():
    from isaaclab.sim.utils import get_current_stage
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
    from robot_lab.assets.x5 import X5_FOOT_BODY_NAMES, X5_WHEEL_RADIUS

    stage = get_current_stage()
    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    assert robot_prim.IsValid(), "X5 prim not found on the active physics stage"
    collision_count = 0
    wheel_cylinders = []
    articulation_found = False
    # Physics cloning may instance the already-configured geometry afterwards.
    # Traverse proxies too: inspect their effective values, not only authored prims.
    for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_found = True
            assert PhysxSchema.PhysxArticulationAPI(prim).GetEnabledSelfCollisionsAttr().Get()
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_count += 1
        collision = PhysxSchema.PhysxCollisionAPI(prim)
        assert abs(collision.GetContactOffsetAttr().Get() - 0.002) < 1e-7
        assert abs(collision.GetRestOffsetAttr().Get()) < 1e-7
        path = str(prim.GetPath())
        if prim.IsA(UsdGeom.Cylinder) and any(f"/{name}/" in path for name in X5_FOOT_BODY_NAMES):
            if abs(UsdGeom.Cylinder(prim).GetRadiusAttr().Get() - X5_WHEEL_RADIUS) < 1e-6:
                wheel_cylinders.append(path)
    assert articulation_found and collision_count > 0, (articulation_found, collision_count)
    assert len(wheel_cylinders) == 4, wheel_cylinders
    return {"collision_shapes_checked": collision_count, "rolling_cylinders": wheel_cylinders,
            "contact_offset_m": 0.002, "rest_offset_m": 0.0, "self_collision": True}


def check_policy(obs, device):
    from rsl_rl.modules import ActorCriticMoECTS
    from robot_lab.tasks.x5.rsl_rl_cfg import X5MoECTSRunnerCfg

    cfg = X5MoECTSRunnerCfg()
    kwargs = cfg.policy.to_dict()
    kwargs.pop("class_name")
    batch = tensordict.TensorDict(obs, batch_size=[obs["policy"].shape[0]])
    model = ActorCriticMoECTS(batch, cfg.obs_groups, 16, **kwargs).to(device)
    for teacher in (True, False):
        actions = model.act(batch, is_teacher=teacher)
        values = model.evaluate(batch, is_teacher=teacher)
        assert actions.shape == (batch.batch_size[0], 16) and torch.isfinite(actions).all()
        assert values.shape == (batch.batch_size[0], 1) and torch.isfinite(values).all()
    assert model.act_inference(batch).shape == (batch.batch_size[0], 16)
    latent, weights = model.student_moe_encoder(batch["policy"])
    assert latent.shape == (batch.batch_size[0], 32)
    assert weights.shape == (batch.batch_size[0], 8)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(batch.batch_size[0], device=device))
    return "PASS_FORWARD_ONLY_NO_OPTIMIZER_OR_POLICY_ACTIONS_APPLIED"


def check_runtime(env, obs, randomized=False):
    from robot_lab.assets.x5 import (
        X5_FOOT_BODY_NAMES, X5_JOINT_NAMES, X5_LEG_JOINT_NAMES, X5_LEG_TARGET_LIMITS, X5_WHEEL_JOINT_NAMES,
    )
    import re

    robot = env.scene["robot"]
    assert env.physics_dt == 0.0025 and env.step_dt == 0.02
    assert robot.num_joints == 16 and robot.num_bodies == 17
    ids, names = robot.find_joints(X5_JOINT_NAMES, preserve_order=True)
    assert names == X5_JOINT_NAMES
    contact = env.scene["contact_forces"]
    feet, names = contact.find_bodies(X5_FOOT_BODY_NAMES, preserve_order=True)
    assert names == X5_FOOT_BODY_NAMES
    observations_cfg = env.observation_manager.cfg
    for group in (observations_cfg.policy, observations_cfg.single_obs, observations_cfg.critic):
        assert group.joint_pos.params["asset_cfg"].joint_ids == ids[:12]
        assert group.joint_vel.params["asset_cfg"].joint_ids == ids
    assert observations_cfg.critic.contact_force.params["sensor_cfg"].body_ids == feet
    for term_name in ("joint_acc_l2", "joint_power", "joint_torques_l2"):
        assert env.reward_manager.get_term_cfg(term_name).params["asset_cfg"].joint_ids == ids
    assert list(env.action_manager.active_terms) == ["joint_pos", "wheel_vel"]
    for term_name, wanted in (("joint_pos", X5_LEG_JOINT_NAMES), ("wheel_vel", X5_WHEEL_JOINT_NAMES)):
        term = env.action_manager.get_term(term_name)
        assert [robot.joint_names[i] for i in term._joint_ids] == wanted
    for name, group in (("legs", X5_LEG_JOINT_NAMES), ("wheels", X5_WHEEL_JOINT_NAMES)):
        actuator = robot.actuators[name]
        assert set(actuator.joint_names) == set(group)
    expected_effort = robot.data.joint_pos.new_tensor([300, 250, 250] * 4 + [30] * 4)
    expected_speed = robot.data.joint_pos.new_tensor([6, 6, 8] * 4 + [30] * 4)
    torch.testing.assert_close(robot.data.joint_effort_limits[:, ids], expected_effort.expand(env.num_envs, -1))
    torch.testing.assert_close(robot.data.joint_vel_limits[:, ids], expected_speed.expand(env.num_envs, -1))
    masses = robot.root_physx_view.get_masses().sum(dim=1)
    if randomized:
        # Startup mass randomization must not be mistaken for an importer error.
        actual = robot.root_physx_view.get_masses()
        nominal = robot.data.default_mass.cpu()
        base = robot.body_names.index("base")
        other = [i for i in range(robot.num_bodies) if i != base]
        assert ((actual[:, base] - nominal[:, base]).abs() <= 1.0001).all()
        ratio = actual[:, other] / nominal[:, other]
        assert ((ratio >= 0.8999) & (ratio <= 1.1001)).all()
        assert not torch.allclose(actual, nominal), "Startup mass randomization did not take effect"
    else:
        torch.testing.assert_close(masses, torch.full_like(masses, 115.780333), atol=1e-4, rtol=0)
    check_observations(obs, env.num_envs)
    actions = torch.zeros(env.num_envs, 16, device=env.device)
    env.action_manager.process_action(actions)
    torch.testing.assert_close(
        env.action_manager.get_term("joint_pos").processed_actions,
        env.action_manager.get_term("joint_pos")._offset if randomized else robot.data.default_joint_pos[:, ids[:12]],
    )
    assert torch.count_nonzero(env.action_manager.get_term("wheel_vel").processed_actions) == 0
    # Check target clipping without stepping physics at these extreme raw actions.
    for sign in (-1, 1):
        env.action_manager.process_action(torch.full_like(actions, sign * 1e6))
        leg_target = env.action_manager.get_term("joint_pos").processed_actions
        for i, name in enumerate(X5_LEG_JOINT_NAMES):
            bounds = [value for pattern, value in X5_LEG_TARGET_LIMITS.items() if re.fullmatch(pattern, name)]
            assert len(bounds) == 1
            assert torch.allclose(leg_target[:, i], torch.full_like(leg_target[:, i], bounds[0][sign > 0]))
        torch.testing.assert_close(
            env.action_manager.get_term("wheel_vel").processed_actions,
            torch.full((env.num_envs, 4), sign * 24.0, device=env.device),
        )
    env.reset()
    return {
        "runtime_joint_order": robot.joint_names,
        "action_joint_order": X5_JOINT_NAMES,
        "action_joint_ids": ids,
        "body_count": robot.num_bodies,
        "physics_dt": env.physics_dt,
        "policy_dt": env.step_dt,
        "total_mass_kg": masses.tolist(),
        "imported_physics": "Startup randomized masses checked within baseline ranges; nominal inertia check is mechanical-only."
        if randomized else check_imported_physics(robot),
        "collision_properties": check_collisions(),
        "observation_shapes": {key: list(value.shape) for key, value in obs.items()},
        "target_clipping": "PASS",
    }, ids, feet


def stance(env, ids, feet):
    robot = env.scene["robot"]
    action = torch.zeros(env.num_envs, 16, device=env.device)
    min_height, max_tilt, max_leg_error, max_drift = float("inf"), 0.0, 0.0, 0.0
    contact_samples = []
    for step in range(args.steps):
        obs, reward, terminated, truncated, _ = env.step(action)
        check_observations(obs, env.num_envs)
        assert torch.isfinite(reward).all()
        assert not (terminated | truncated).any(), f"Unexpected reset at step {step}"
        assert torch.count_nonzero(env.command_manager.get_command("base_velocity")) == 0
        assert torch.isfinite(robot.data.applied_torque).all()
        assert (robot.data.applied_torque.abs() <= robot.data.joint_effort_limits + 1e-4).all()
        height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
        error = (robot.data.joint_pos[:, ids[:12]] - robot.data.default_joint_pos[:, ids[:12]]).abs()
        drift = (robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]).norm(dim=-1)
        min_height = min(min_height, height.min().item())
        max_tilt = max(max_tilt, tilt.max().item())
        max_leg_error = max(max_leg_error, error.max().item())
        max_drift = max(max_drift, drift.max().item())
        if step >= args.steps - min(100, args.steps):
            forces = env.scene["contact_forces"].data.net_forces_w[:, feet].norm(dim=-1)
            contact_samples.append((forces > 1.0).float())
        if (step + 1) % 250 == 0:
            print(f"[X5 stance] {step + 1}/{args.steps}, height_min={min_height:.4f}, tilt_max={max_tilt:.4f}", flush=True)
    report = {
        "simulated_seconds": args.steps * env.step_dt,
        "min_base_height_m": min_height,
        "max_tilt_rad": max_tilt,
        "max_leg_error_rad": max_leg_error,
        "max_xy_drift_m": max_drift,
        "last_window_wheel_contact_fraction": torch.stack(contact_samples).mean(dim=(0, 1)).tolist(),
        "final_base_height_m": height.tolist(),
        "final_wheel_contact_force_N": forces.tolist(),
        "critic_contact_clip_N": list(env.cfg.observations.critic.contact_force.clip),
    }
    print("[X5 stance metrics] " + json.dumps(report), flush=True)
    assert min_height > 0.45, "Excessive stance collapse"
    assert max_tilt < 0.262, "Stance tilt exceeded 15 degrees"
    assert max_leg_error < 0.3, "Excessive leg position error"
    assert min(report["last_window_wheel_contact_fraction"]) > 0.9, "Not all four wheels maintain contact"
    return report


def interface(env, feet):
    """Exercise actual terrain/events/commands; resets are expected with zero actions."""
    from robot_lab.tasks.go2.mdp.observations import foot_contact_force_norm

    robot = env.scene["robot"]
    action = torch.zeros(env.num_envs, 16, device=env.device)
    resets = 0
    commands_nonzero = False
    peak_force = 0.0
    clipped_samples = total_samples = 0
    settled_clipped = settled_samples = 0
    peak_step = -1
    command = env.command_manager.get_term("base_velocity")
    covered_terrains = [name for name, index in command.terrain_type2idx.items()
                        if (command.terrain_idxs == index).any()]
    active_terrains = [name for name, terrain in env.scene.terrain.cfg.terrain_generator.sub_terrains.items()
                       if terrain.proportion > 0]
    if env.num_envs >= env.scene.terrain.cfg.terrain_generator.num_cols:
        assert set(covered_terrains) == set(active_terrains)
    critic_terms = env.observation_manager.active_terms["critic"]
    contact_index = critic_terms.index("contact_force")
    contact_start = sum(math.prod(dim) for dim in env.observation_manager.group_obs_term_dim["critic"][:contact_index])
    term_names = list(env.reward_manager.active_terms)
    reward_sum = torch.zeros(len(term_names), device=env.device)
    for step in range(args.steps):
        obs, reward, terminated, truncated, _ = env.step(action)
        check_observations(obs, env.num_envs)
        assert torch.isfinite(reward).all() and torch.isfinite(robot.data.applied_torque).all()
        assert (robot.data.applied_torque.abs() <= robot.data.joint_effort_limits + 1e-4).all()
        resets += int((terminated | truncated).sum().item())
        commands_nonzero |= bool(env.command_manager.get_command("base_velocity").abs().max() > 0)
        term = env.observation_manager.cfg.critic.contact_force
        raw_force = foot_contact_force_norm(env, **term.params)
        assert raw_force.shape == (env.num_envs, 16)
        assert torch.isfinite(raw_force).all()
        torch.testing.assert_close(obs["critic"][:, contact_start:contact_start + 16],
                                   raw_force.clamp(*term.clip) * term.scale)
        if raw_force.max().item() > peak_force:
            peak_force = raw_force.max().item()
            peak_step = step
        clipped_samples += int((raw_force >= term.clip[1]).sum().item())
        total_samples += raw_force.numel()
        settled = env.episode_length_buf * env.step_dt > 0.5
        settled_clipped += int((raw_force[settled] >= term.clip[1]).sum().item())
        settled_samples += raw_force[settled].numel()
        # Recorded diagnostics only; do not tune weights to this untrained behavior.
        assert torch.isfinite(env.reward_manager._step_reward).all()
        reward_sum += env.reward_manager._step_reward.abs().mean(dim=0)
        if (step + 1) % 64 == 0:
            print(f"[X5 interface] {step + 1}/{args.steps}, resets={resets}, peak_force={peak_force:.1f} N", flush=True)
    assert commands_nonzero, "Full-task command sampler remained zero"
    assert env.scene.terrain.cfg.terrain_type == "generator"
    return {"simulated_seconds": args.steps * env.step_dt, "zero_action_resets": resets,
            "nonzero_commands": commands_nonzero, "peak_raw_teacher_contact_force_N": peak_force,
            "peak_force_step_zero_based": peak_step,
            "contact_clip_fraction": clipped_samples / total_samples,
            "contact_clip_fraction_after_reset_0_5s": settled_clipped / settled_samples if settled_samples else None,
            "teacher_contact_pipeline": "PASS_RAW_CLIP_SCALE_MATCHES_291D_OBSERVATION",
            "terrain_types_with_robots": covered_terrains,
            "mean_abs_weighted_reward_terms_per_second": dict(zip(term_names, (reward_sum / args.steps).tolist())),
            "event_terms": env.event_manager.active_terms,
            "curriculum_terms": env.curriculum_manager.active_terms,
            "terrain_rows_columns": [env.scene.terrain.cfg.terrain_generator.num_rows,
                                      env.scene.terrain.cfg.terrain_generator.num_cols],
            "limitation": "Zero-action interface test, not terrain traversal or training."}


def axes(env, ids):
    from robot_lab.assets.x5 import X5_JOINT_NAMES, X5_WHEEL_RADIUS

    robot = env.scene["robot"]
    results = {}
    for index, name in enumerate(X5_JOINT_NAMES):
        responses = []
        for sign in (1, -1):
            env.reset()
            q_start = robot.data.joint_pos[:, ids[index]].clone()
            action = torch.zeros(env.num_envs, 16, device=env.device)
            action[:, index] = sign * (0.05 / 0.2 if index < 12 else 2.0 * X5_WHEEL_RADIUS)
            for _ in range(20):
                obs, reward, terminated, truncated, _ = env.step(action)
                check_observations(obs, env.num_envs)
                assert torch.isfinite(reward).all()
                assert not (terminated | truncated).any()
            response = robot.data.joint_pos[:, ids[index]] - q_start if index < 12 else robot.data.joint_vel[:, ids[index]]
            responses.append(response.tolist())
            assert (sign * response > 1e-3).all(), f"Wrong/zero signed response: {name}, {sign}, {response}"
        results[name] = {"positive": responses[0], "negative": responses[1], "unit": "rad" if index < 12 else "rad/s"}
        print(f"[X5 axis] {name}: {results[name]}", flush=True)
    return results


def check_runner_init(wrapped):
    """Same wrapper/config/Runner as train.py; stop before learn or any update."""
    import os
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunnerCTS

    assert int(os.environ.get("WORLD_SIZE", "1")) == 1, "Run this initialization check without torchrun."
    runner_cfg = load_cfg_from_registry("RobotLab-X5-MoECTS-v0", "rsl_rl_cfg_entry_point")
    env = RslRlVecEnvWrapper(wrapped, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunnerCTS(env, runner_cfg.to_dict(), log_dir=None, device=env.device)
    algorithm = runner.alg
    check_observations(env.get_observations(), env.num_envs)
    assert env.num_actions == 16
    assert type(algorithm.optimizer) is type(algorithm.optimizer_stu_enc) is torch.optim.Adam
    assert len(algorithm.optimizer.state) == len(algorithm.optimizer_stu_enc.state) == 0
    assert algorithm.storage.step == runner.current_learning_iteration == 0
    assert algorithm.storage.actions.shape == (24, env.num_envs, 16)
    assert not algorithm.storage.actions.is_inference()
    assert all(param.requires_grad and not param.is_inference() and param.grad is None
               for param in algorithm.policy.parameters())
    return {"runner": type(runner).__name__, "algorithm": type(algorithm).__name__,
            "teacher_env_indices": algorithm.teacher_env_idxs.tolist(),
            "student_env_indices": algorithm.student_env_idxs.tolist(),
            "optimizers": [type(algorithm.optimizer).__name__, type(algorithm.optimizer_stu_enc).__name__],
            "optimizer_state_entries": [len(algorithm.optimizer.state), len(algorithm.optimizer_stu_enc.state)],
            "storage_shape": list(algorithm.storage.actions.shape),
            "collected_transitions": algorithm.storage.step,
            "learning_iteration": runner.current_learning_iteration, "checkpoint_written": False}


def main():
    import gymnasium as gym
    import robot_lab.tasks  # noqa: F401
    import rsl_rl
    from isaaclab_tasks.utils import load_cfg_from_registry

    cfg = load_cfg_from_registry("RobotLab-X5-MoECTS-v0", "env_cfg_entry_point")
    report = check_config(cfg)
    report["rsl_rl_path"] = rsl_rl.__file__
    report["source_sha256"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in ("source/robot_lab/robot_lab/assets/x5.py", "source/robot_lab/robot_lab/tasks/x5/env_cfg.py",
                     "source/robot_lab/robot_lab/tasks/x5/parameters.py",
                     "source/robot_lab/robot_lab/tasks/x5/validation_cfg.py", "source/robot_lab/robot_lab/tasks/x5/config_audit.py",
                     "source/robot_lab/robot_lab/tasks/x5/rsl_rl_cfg.py", "scripts/tools/check_x5_sim.py",
                     "source/rsl_rl/rsl_rl/algorithms/moe_cts.py",
                     "source/robot_lab/robot_lab/tasks/legbot/env_cfg.py", "source/robot_lab/robot_lab/tasks/legbot/rsl_rl_cfg.py")
    }
    if args.config_only:
        report["status"] = "PASS_CONFIG_ONLY"
        return report
    if args.runner_init_only:
        report["test_profile"] = "Full registered task; only num_envs, seed and device overridden. Runner initialization only."
    elif args.mode != "interface":
        from robot_lab.tasks.x5.validation_cfg import mechanical_check_cfg

        cfg = mechanical_check_cfg(cfg)
        report["test_profile"] = "Mechanical-only: plane, zero commands, deterministic reset, no curriculum/noise."
        cfg.episode_length_s = max(25.0, args.steps * 0.02 + 2.0)
    else:
        report["test_profile"] = "Full registered task; only num_envs, seed and device overridden. No learner."
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    if args.mode == "axes" and not args.runner_init_only:
        cfg.scene.robot.spawn.fix_base = True
        cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.5)
    wrapped = gym.make("RobotLab-X5-MoECTS-v0", cfg=cfg)
    env = wrapped.unwrapped
    try:
        if args.runner_init_only:
            report["runner_init"] = check_runner_init(wrapped)
            report["status"] = "PASS_RUNNER_INIT_ONLY"
            report["limitations"] = "No rollout collection, backward pass, optimizer step, checkpoint, or learned locomotion validation."
            return report
        obs, _ = env.reset()
        runtime, ids, feet = check_runtime(env, obs, randomized=args.mode == "interface")
        report.update(runtime)
        report["policy_forward"] = check_policy(obs, env.device)
        report[args.mode] = interface(env, feet) if args.mode == "interface" else (
            stance(env, ids, feet) if args.mode == "stance" else axes(env, ids)
        )
        report["status"] = f"PASS_SIM_{args.mode.upper()}_ONLY"
        report["limitations"] = "No PPO training, learned locomotion, or hardware validation."
        return report
    finally:
        wrapped.close()


if __name__ == "__main__":
    try:
        with torch.inference_mode(not args.runner_init_only):
            result = main()
        text = json.dumps(result, indent=2, ensure_ascii=False)
        print(text, flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
    except BaseException as error:
        import os
        import traceback

        # Kit fast shutdown can replace a Python failure with exit code zero.
        # Flush evidence and exit nonzero on failure instead of losing the error
        # in framework shutdown. The process exit releases its GPU resources.
        traceback.print_exc()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"status": "FAIL", "error": repr(error)}) + "\n", encoding="utf-8")
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
    finally:
        app.close()
