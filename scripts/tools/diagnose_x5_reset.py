"""Single-environment X5 reset diagnostics; no learner or training-config edits.

Examples (run with x3w_isaaclab Python):
  python scripts/tools/diagnose_x5_reset.py --headless --profile nominal --trials 1 --output outputs/x5_reset/nominal.json
  python scripts/tools/diagnose_x5_reset.py --headless --profile task --trials 20 --output outputs/x5_reset/task.json

Task trials stratify terrain columns, keeping the registered geometry and reset
distributions. --raise-clearance is a diagnostic state-only counterfactual.
Contact forces are read directly every 2.5 ms from PhysX, not a 3-frame maximum.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
sys.path.insert(0, str(ROOT / "source/robot_lab"))
sys.path.insert(0, str(ROOT / "source/rsl_rl"))

import h5py  # noqa: F401 - native module preload before Kit
import tensordict  # noqa: F401
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", choices=("nominal", "task"), default="task")
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--steps", type=int, default=50, help="Policy steps per trial (50 Hz).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--trial-start", type=int, default=0)
parser.add_argument("--raise-clearance", type=float, default=None,
                    help="Diagnostic only: lift sampled pose until sampled collision clearance reaches this many meters.")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.trials < 1 or args.steps < 1 or args.trial_start < 0:
    parser.error("trials/steps must be positive; trial-start must be nonnegative")
if args.raise_clearance is not None and args.raise_clearance < 0:
    parser.error("raise-clearance must be nonnegative")
launcher = AppLauncher(args)
app = launcher.app


class Geometry:
    """URDF kinematics plus sampled primitive surfaces, independent of USD pose caches."""

    def __init__(self):
        from validate_x5_asset import origin

        model = ET.parse(ROOT / "resources/x5/urdf/x5.urdf").getroot()
        self.joints = model.findall("joint")
        self.colliders = []
        for link in model.findall("link"):
            for index, collision in enumerate(link.findall("collision")):
                geom = collision.find("geometry")
                if geom.find("box") is not None:
                    size = np.fromstring(geom.find("box").get("size"), sep=" ")
                    # 5x5x5 lattice includes all vertices/edges and face/interior points.
                    points = np.stack(np.meshgrid(*[np.linspace(-v / 2, v / 2, 5) for v in size]), axis=-1).reshape(-1, 3)
                elif geom.find("cylinder") is not None:
                    cylinder = geom.find("cylinder")
                    radius, length = float(cylinder.get("radius")), float(cylinder.get("length"))
                    theta, z = np.meshgrid(np.linspace(0, 2 * np.pi, 72, endpoint=False), np.linspace(-length / 2, length / 2, 5))
                    points = np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=-1).reshape(-1, 3)
                else:
                    raise ValueError(f"Unsupported collision geometry on {link.get('name')}")
                self.colliders.append((link.get("name"), index, origin(collision.find("origin")), points))

    def poses(self, root_state, joint_positions):
        from validate_x5_asset import origin, rotation

        w, x, y, z = np.asarray(root_state[3:7]) / np.linalg.norm(root_state[3:7])
        base = np.eye(4)
        base[:3, :3] = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ])
        base[:3, 3] = root_state[:3]
        poses = {"base": base}
        pending = list(self.joints)
        while pending:
            ready = [j for j in pending if j.find("parent").get("link") in poses]
            assert ready, "Invalid URDF tree"
            for joint in ready:
                motion = np.eye(4)
                if joint.get("type") != "fixed":
                    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                    motion[:3, :3] = rotation(axis, joint_positions[joint.get("name")])
                poses[joint.find("child").get("link")] = poses[joint.find("parent").get("link")] @ origin(joint.find("origin")) @ motion
                pending.remove(joint)
        return poses

    def inspect(self, env):
        from isaaclab.utils.warp import raycast_mesh

        robot = env.scene["robot"]
        root = robot.data.root_state_w[0].cpu().numpy()
        q = dict(zip(robot.joint_names, robot.data.joint_pos[0].cpu().tolist()))
        poses = self.poses(root, q)
        all_points, labels = [], []
        for name, index, offset, points in self.colliders:
            transform = poses[name] @ offset
            all_points.append(points @ transform[:3, :3].T + transform[:3, 3])
            labels.append((name, index, len(points)))
        points = torch.tensor(np.concatenate(all_points), dtype=torch.float32, device=env.device)
        starts = points.clone()
        starts[:, 2] += 10
        directions = torch.zeros_like(starts)
        directions[:, 2] = -1
        mesh = env.scene["height_scanner"].meshes["/World/ground"]
        hits, _, _, _ = raycast_mesh(starts, directions, mesh, max_dist=30)
        assert torch.isfinite(hits).all(), "Terrain ray missed during clearance audit"
        gaps = points[:, 2] - hits[:, 2]
        details, offset = [], 0
        for name, index, count in labels:
            section = gaps[offset:offset + count]
            minimum, where = section.min(dim=0)
            worst = offset + int(where)
            details.append({"body": name, "collision_index": index,
                            "min_sampled_clearance_m": minimum.item(),
                            "surface_point_w": points[worst].tolist(), "ground_hit_w": hits[worst].tolist()})
            offset += count
        # Compare FK with PhysX as an independent guard against wrong frames/joint order.
        actual_positions = robot.root_physx_view.get_link_transforms()[0, :, :3].cpu().numpy()
        expected_positions = np.stack([poses[name][:3, 3] for name in robot.body_names])
        error = float(np.max(np.abs(actual_positions - expected_positions)))
        assert error < 1e-4, f"URDF FK vs PhysX link-frame mismatch: {error}"
        return {"root_state_w": root.tolist(), "q": q,
                "joint_vel": robot.data.joint_vel[0].tolist(),
                "physics_parameters": {
                    "masses_kg": robot.root_physx_view.get_masses()[0].tolist(),
                    "materials": robot.root_physx_view.get_material_properties()[0].tolist(),
                    "coms": robot.root_physx_view.get_coms()[0].tolist(),
                    "inertias": robot.root_physx_view.get_inertias()[0].tolist(),
                    "gains": {name: {"stiffness": actuator.stiffness.tolist(), "damping": actuator.damping.tolist()}
                              for name, actuator in robot.actuators.items()},
                    "position_action_offset": env.action_manager.get_term("joint_pos")._offset.tolist(),
                },
                "root_height_above_tile_origin_m": float(root[2] - env.scene.env_origins[0, 2]),
                "min_sampled_clearance_m": gaps.min().item(),
                "wheel_clearance_m": {d["body"]: d["min_sampled_clearance_m"] for d in details
                                      if d["body"].endswith("_FOOT") and d["collision_index"] == 0},
                "penetrating_colliders": [d for d in details if d["min_sampled_clearance_m"] < -0.002],
                "colliders": details, "max_fk_vs_physx_link_position_error_m": error,
                "clearance_method": "Vertical terrain rays under sampled URDF primitive surface points; not a full mesh intersection proof."}


def main():
    import gymnasium as gym
    import robot_lab.tasks  # noqa: F401
    from robot_lab.tasks.x5.env_cfg import X5EnvCfg
    from robot_lab.tasks.x5.validation_cfg import mechanical_check_cfg

    cfg = X5EnvCfg()
    if args.profile == "nominal":
        cfg = mechanical_check_cfg(cfg)
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = args.seed
    before = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (ROOT / "source/robot_lab/robot_lab/tasks/x5").glob("*.py")}
    wrapped = gym.make("RobotLab-X5-MoECTS-v0", cfg=cfg)
    env = wrapped.unwrapped
    assert env.num_envs == 1 and env.physics_dt == 0.0025 and env.step_dt == 0.02
    robot = env.scene["robot"]
    contact = env.scene["contact_forces"]
    feet = [contact.body_names.index(f"{leg}_FOOT") for leg in ("RF", "LF", "RH", "LH")]
    geometry = Geometry()
    results = []
    action = torch.zeros(1, 16, device=env.device)
    original_update = env.scene.update
    capture = None

    def update_and_capture(dt):
        original_update(dt)
        if capture is None:
            return
        forces = contact.contact_physx_view.get_net_contact_forces(dt=env.physics_dt).reshape(1, -1, 3)[0]
        assert torch.isfinite(forces).all() and torch.isfinite(robot.data.joint_acc).all()
        magnitude = forces.norm(dim=-1)
        capture.append({"t_s": (len(capture) + 1) * dt,
                        "root_pos_w": robot.data.root_pos_w[0].tolist(),
                        "root_lin_vel_w": robot.data.root_lin_vel_w[0].tolist(),
                        "body_force_N": magnitude.tolist(), "wheel_force_xyz_N": forces[feet].tolist(),
                        "max_joint_acc_rad_s2": robot.data.joint_acc[0].abs().max().item(),
                        "joint_torque_Nm": robot.data.applied_torque[0].tolist()})

    env.scene.update = update_and_capture
    try:
        for trial in range(args.trial_start, args.trial_start + args.trials):
            column = level = None
            if args.profile == "task":
                terrain = env.scene.terrain
                command = env.command_manager.get_term("base_velocity")
                # Stratify this ONE robot across terrain columns between trials.
                # Clear old episode progress so reset curriculum does not move the requested tile.
                command.max_move_distance.zero_()
                command.commands_xy_accumulation.zero_()
                column = trial % terrain.cfg.terrain_generator.num_cols
                level = (trial // terrain.cfg.terrain_generator.num_cols + 2) % 6
                terrain.terrain_types[0] = column
                terrain.terrain_levels[0] = level
                terrain.env_origins[0] = terrain.terrain_origins[level, column]
                command._init_terrain_infos()
            env.reset(seed=args.seed + trial)
            initial = geometry.inspect(env)
            if args.profile == "task":
                command = env.command_manager.get_term("base_velocity")
                terrain_name = command.terrain_types[int(command.terrain_idxs[0])]
            else:
                terrain_name = "plane"
            lift = 0.0
            if args.raise_clearance is not None:
                lift = max(0.0, args.raise_clearance - initial["min_sampled_clearance_m"])
                pose = robot.data.root_state_w[:, :7].clone()
                pose[:, 2] += lift
                robot.write_root_pose_to_sim(pose)
                env.scene.write_data_to_sim()
                env.sim.forward()
            after_lift = geometry.inspect(env) if lift else initial
            if not args.headless:
                position = np.asarray(after_lift["root_state_w"][:3])
                env.sim.set_camera_view(eye=position + [2.0, 2.0, 1.3], target=position)
            capture = []
            reward_samples = []
            reset = False
            for _ in range(args.steps):
                obs, reward, terminated, truncated, _ = env.step(action)
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(reward).all()
                reward_samples.append(env.reward_manager._step_reward[0].tolist())
                if bool((terminated | truncated).any()):
                    reset = True
                    break
            trace, capture = capture, None
            wheel_peaks = [max(np.linalg.norm(row["wheel_force_xyz_N"], axis=1)) for row in trace]
            peak_index = int(np.argmax(wheel_peaks))
            first_contact = next((row for row, peak in zip(trace, wheel_peaks) if peak > 10), None)
            summary = {"trial": trial, "seed": args.seed + trial, "terrain_column": column, "terrain_level": level,
                       "terrain_name": terrain_name,
                       "min_initial_clearance_m": initial["min_sampled_clearance_m"],
                       "diagnostic_lift_m": lift, "initial_vz_m_s": initial["root_state_w"][9],
                       "first_contact_s": first_contact["t_s"] if first_contact else None,
                       "peak_wheel_force_N": wheel_peaks[peak_index], "peak_time_s": trace[peak_index]["t_s"],
                       "peak_root_vz_m_s": trace[peak_index]["root_lin_vel_w"][2],
                       "first_physics_step_peak_wheel_force_N": wheel_peaks[0],
                       "root_drop_at_peak_m": after_lift["root_state_w"][2] - trace[peak_index]["root_pos_w"][2],
                       "peak_body": contact.body_names[int(np.argmax(trace[peak_index]["body_force_N"]))],
                       "terminated_or_truncated": reset}
            print("[X5 reset] " + json.dumps(summary), flush=True)
            results.append({"summary": summary, "sampled_initial": initial, "stepped_initial": after_lift,
                            "physics_trace": trace, "reward_names": env.reward_manager.active_terms,
                            "weighted_rewards_per_second": reward_samples})
    finally:
        env.scene.update = original_update
        wrapped.close()
    after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in (ROOT / "source/robot_lab/robot_lab/tasks/x5").glob("*.py")}
    assert before == after, "Training task source changed during diagnosis"
    return {"status": "PASS_DIAGNOSTIC_EXECUTION_NOT_TRAINING_ACCEPTANCE", "profile": args.profile,
            "num_envs": 1, "physics_dt": 0.0025, "policy_dt": 0.02, "training_config_sha256": before,
            "diagnostic_script_sha256": SCRIPT_SHA256,
            "body_names": contact.body_names, "trials": results,
            "limitations": "No training or parameter fix. One robot, sequential stratified tiles. Sampled clearance is not exhaustive collision detection."}


if __name__ == "__main__":
    try:
        with torch.inference_mode():
            report = main()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[X5 reset] {report['status']}: {args.output}", flush=True)
    except BaseException as error:
        import os
        import traceback

        traceback.print_exc()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"status": "FAIL", "error": repr(error)}) + "\n", encoding="utf-8")
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
    finally:
        app.close()
