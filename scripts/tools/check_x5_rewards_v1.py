"""Validate the isolated X5 reward task and optionally step it, without training.

Checks the full inherited config, actual manager registration and entity order.
The physics smoke uses zero actions; it is not a learned locomotion evaluation.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/robot_lab"))
sys.path.insert(0, str(ROOT / "source/rsl_rl"))

import h5py  # noqa: F401
import tensordict  # noqa: F401
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config-only", action="store_true")
parser.add_argument("--profile", choices=("rewards", "steps"), default="rewards")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--output", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.steps < 1:
    parser.error("--num-envs and --steps must be positive")
app = AppLauncher(args).app


def main():
    import gymnasium as gym
    import robot_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry
    from robot_lab.assets.x5 import X5_FOOT_BODY_NAMES, X5_WHEEL_JOINT_NAMES
    from robot_lab.tasks.x5.config_audit import differences
    from robot_lab.tasks.x5_rewards_v1.rewards import joint_pose_deadband_l1

    baseline_id = "RobotLab-X5-MoECTS-v0"
    task_id = "RobotLab-X5-MoECTS-Steps-v1" if args.profile == "steps" else "RobotLab-X5-MoECTS-Rewards-v1"
    baseline = load_cfg_from_registry(baseline_id, "env_cfg_entry_point")
    before = deepcopy(baseline.to_dict())
    cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    assert not differences(before, baseline.to_dict()), "Experimental construction changed the baseline"
    again = load_cfg_from_registry(baseline_id, "env_cfg_entry_point")
    assert not differences(before, again.to_dict()), "A new baseline instance changed"

    base_dict, experiment_dict = baseline.to_dict(), cfg.to_dict()
    base_rewards = base_dict.pop("rewards")
    rewards = experiment_dict.pop("rewards")
    assert not differences(base_dict, experiment_dict), "Non-reward environment configuration changed"
    added = {"stand_still_wheel_l2", "wheel_air_spin_l2"}
    if args.profile == "steps":
        added.add("completed_step")
    assert set(rewards) == set(base_rewards) | added
    pose_bands = {"hip_pos_penalty_l1": 0.08, "joint_pos_penalty_l1": 0.15}
    for name, reference in base_rewards.items():
        if name in pose_bands:
            term = getattr(cfg.rewards, name)
            assert term.func is joint_pose_deadband_l1
            assert term.weight == getattr(baseline.rewards, name).weight
            assert set(term.params) == {"asset_cfg", "deadband"}
            assert term.params["deadband"] == pose_bands[name]
            assert not differences(reference["params"]["asset_cfg"], rewards[name]["params"]["asset_cfg"])
        else:
            assert not differences(reference, rewards[name]), name
    base_runner = load_cfg_from_registry(baseline_id, "rsl_rl_cfg_entry_point").to_dict()
    runner = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point").to_dict()
    assert set(differences(base_runner, runner)) == {"experiment_name"}
    assert runner["experiment_name"] == f"x5_moe_cts_{args.profile}_v1"
    assert cfg.sim.dt == 0.0025 and cfg.sim.dt * cfg.decimation == 0.02
    report = {
        "status": "PASS_CONFIG_ONLY", "task": task_id, "baseline": baseline_id,
        "baseline_unchanged": True, "shared_reward_weights_unchanged": True,
        "reward_terms": len(rewards), "added_terms": sorted(added), "pose_deadbands_rad": pose_bands,
        "learner_difference": "experiment_name_only", "physics_dt": cfg.sim.dt, "policy_dt": 0.02,
    }
    if args.config_only:
        return report

    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    if args.device is not None:
        cfg.sim.device = args.device
    env = gym.make(task_id, cfg=cfg)
    try:
        env = env.unwrapped
        obs, _ = env.reset()
        robot = env.scene["robot"]
        wheel_ids, _ = robot.find_joints(X5_WHEEL_JOINT_NAMES, preserve_order=True)
        sensor = env.scene["contact_forces"]
        foot_ids, _ = sensor.find_bodies(X5_FOOT_BODY_NAMES, preserve_order=True)
        stand_cfg = env.reward_manager.get_term_cfg("stand_still_wheel_l2")
        air_cfg = env.reward_manager.get_term_cfg("wheel_air_spin_l2")
        assert stand_cfg.params["asset_cfg"].joint_ids == wheel_ids
        assert air_cfg.params["asset_cfg"].joint_ids == wheel_ids
        assert air_cfg.params["sensor_cfg"].body_ids == foot_ids
        term_count = 17 if args.profile == "steps" else 16
        assert len(env.reward_manager.active_terms) == term_count
        step_cfg = None
        if args.profile == "steps":
            step_cfg = env.reward_manager.get_term_cfg("completed_step")
            body_ids, _ = robot.find_bodies(X5_FOOT_BODY_NAMES, preserve_order=True)
            assert step_cfg.params["asset_cfg"].body_ids == body_ids
            assert step_cfg.params["sensor_cfg"].body_ids == foot_ids
        assert env.physics_dt == 0.0025 and env.step_dt == 0.02
        cost_sum = torch.zeros(term_count, device=env.device)
        active_count = torch.zeros((), device=env.device)
        completed_step_count = 0
        terminated_count = timeout_count = 0
        actions = torch.zeros((args.num_envs, 16), device=env.device)
        for _ in range(args.steps):
            obs, reward, terminated, timeout, _ = env.step(actions)
            assert torch.isfinite(reward).all()
            for name, size in (("policy", 530), ("single_obs", 53), ("critic", 291)):
                assert obs[name].shape == (args.num_envs, size) and torch.isfinite(obs[name]).all()
            assert torch.isfinite(env.reward_manager._step_reward).all()
            cost_sum += env.reward_manager._step_reward.abs().mean(dim=0)
            active_count += stand_cfg.func.active.float().mean()
            if step_cfg is not None:
                completed_step_count += step_cfg.func.tracker.accepted_events.sum().item()
            terminated_count += terminated.sum().item()
            timeout_count += timeout.sum().item()

        # Exercise quiet-command state with actual manager objects, independently
        # of whether the random command sample includes a true stop in 64 steps.
        command = env.command_manager.get_command("base_velocity")
        saved_command = command.clone()
        saved_step = env.common_step_counter
        saved_episode = env.episode_length_buf.clone()
        try:
            command.zero_()
            env.episode_length_buf.fill_(100)
            stand_cfg.func.reset()
            for i in range(20):
                env.common_step_counter = saved_step + i + 1
                value = stand_cfg.func(env, **stand_cfg.params)
                assert value.shape == (args.num_envs,) and torch.isfinite(value).all()
            assert stand_cfg.func.active.all()
            command[:, 2] = 0.5
            assert torch.count_nonzero(stand_cfg.func(env, **stand_cfg.params)) == 0
            assert not stand_cfg.func.active.any()
            stand_cfg.func.reset([0])
            assert stand_cfg.func.gate.elapsed[0] == 0
        finally:
            command[:] = saved_command
            env.common_step_counter = saved_step
            env.episode_length_buf[:] = saved_episode
            stand_cfg.func.reset()
        report.update({
            "status": "PASS_CONFIG_AND_PHYSICS_SMOKE", "num_envs": args.num_envs, "steps": args.steps,
            "actions": "zero", "observations": {"policy": 530, "single_obs": 53, "critic": 291},
            "mean_abs_weighted_reward_terms_per_second": dict(zip(
                env.reward_manager.active_terms, (cost_sum / args.steps).tolist())),
            "sampled_standstill_activation_fraction": (active_count / args.steps).item(),
            "sampled_completed_step_events": completed_step_count,
            "quiet_and_pure_yaw_manager_check": "PASS_SYNTHETIC_COMMANDS_NO_PHYSICS",
            "terminated_count": terminated_count, "timeout_count": timeout_count,
            "optimizer_updates": 0, "learned_locomotion_evaluated": False,
        })
    finally:
        env.close()
    return report


try:
    result = main()
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
finally:
    app.close()
