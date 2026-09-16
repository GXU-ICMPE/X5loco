#!/usr/bin/env python3
"""Export a saved X5 MoE-CTS Student to stateless ONNX without launching Isaac."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/rsl_rl"))
from rsl_rl.modules import ActorCriticMoECTS


class StudentPolicy(torch.nn.Module):
    """Input contains term-wise history; latest-frame extraction is in the graph."""

    def __init__(self, policy):
        super().__init__()
        self.encoder = copy.deepcopy(policy.student_moe_encoder)
        self.actor = copy.deepcopy(policy.actor)
        self.history_normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.single_normalizer = copy.deepcopy(policy.single_obs_normalizer)
        self.feature_dims = (3, 3, 3, 12, 16, 16)

    def single_frame(self, obs):
        offset = 0
        frames = []
        for dim in self.feature_dims:
            end = offset + 10 * dim
            frames.append(obs[:, end - dim:end])
            offset = end
        return torch.cat(frames, dim=-1)

    def forward(self, obs):
        single = self.single_normalizer(self.single_frame(obs))
        latent, _ = self.encoder(self.history_normalizer(obs))
        return self.actor(torch.cat((latent, single), dim=-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", help="Task recorded in the manifest; inferred from known X5 experiments by default.")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip ONNX checking and numerical parity; export only.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = args.checkpoint.resolve()
    agent_path = checkpoint.parent / "params/agent.yaml"
    env_path = checkpoint.parent / "params/env.yaml"
    agent = yaml.load(agent_path.read_text(), Loader=yaml.FullLoader)
    experiment_tasks = {
        "x5_moe_cts": "RobotLab-X5-MoECTS-v0",
        "x5_moe_cts_v4": "RobotLab-X5-MoECTS-v4",
        "x5_moe_cts_v5": "RobotLab-X5-MoECTS-v5",
        "x5_moe_cts_v6": "RobotLab-X5-MoECTS-v6",
        "x5_moe_cts_v7": "RobotLab-X5-MoECTS-v7",
        "x5_moe_cts_v8": "RobotLab-X5-MoECTS-v8",
    }
    inferred_task = experiment_tasks.get(agent.get("experiment_name"))
    if args.task and inferred_task and args.task != inferred_task:
        raise ValueError(f"Task {args.task!r} disagrees with saved experiment task {inferred_task!r}")
    task = args.task or inferred_task
    if task is None:
        raise ValueError("Unknown experiment name; specify --task for the export manifest")
    # Read Python-tagged environment YAML as data; never instantiate Isaac objects.
    saved_env = yaml.load(env_path.read_text(), Loader=yaml.BaseLoader)
    expected_terms = ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"]
    for name, history in (("policy", 10), ("single_obs", 1)):
        group = saved_env["observations"][name]
        terms = [k for k, v in group.items() if isinstance(v, dict) and "func" in v]
        if terms != expected_terms or int(group["history_length"]) != history:
            raise ValueError("Saved observation ordering/history does not match the X5 MoE-CTS adapter")
        for term, scale in zip(expected_terms, (0.25, 1., 1., 1., 0.05, 1.)):
            if float(group[term]["scale"]) != scale or list(map(float, group[term]["clip"])) != [-100., 100.]:
                raise ValueError("Saved observation scaling/clipping does not match the adapter")
    leg_names = [f"{leg}_{joint}" for leg in ("RF", "LF", "RH", "LH") for joint in ("HAA", "HFE", "KFE")]
    wheel_names = [f"{leg}_WHEEL" for leg in ("RF", "LF", "RH", "LH")]
    actions = saved_env["actions"]
    if (list(actions) != ["joint_pos", "wheel_vel"]
            or actions["joint_pos"]["joint_names"] != leg_names
            or actions["wheel_vel"]["joint_names"] != wheel_names
            or float(actions["joint_pos"]["scale"]) != 0.2
            or abs(float(actions["wheel_vel"]["scale"]) - 1. / 0.1005) > 1e-8):
        raise ValueError("Saved action contract does not match the adapter")
    if str(actions["joint_pos"]["use_default_offset"]).lower() != "true":
        raise ValueError("Expected leg actions relative to the saved default joint pose")
    nominal_patterns = saved_env["scene"]["robot"]["init_state"]["joint_pos"]
    nominal_leg_positions = []
    for name in leg_names:
        matches = [float(value) for pattern, value in nominal_patterns.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely resolve the nominal position for {name}")
        nominal_leg_positions.append(matches[0])
    policy_cfg = dict(agent["policy"])
    if policy_cfg.pop("class_name") != "ActorCriticMoECTS" or policy_cfg.get("state_dependent_std", False):
        raise ValueError("Expected a state-independent X5 MoE-CTS policy")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    checkpoint_sha = sha(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dummy = TensorDict({"policy": torch.zeros(1, 530), "single_obs": torch.zeros(1, 53),
                        "critic": torch.zeros(1, 291)}, batch_size=[1])
    policy = ActorCriticMoECTS(dummy, agent["obs_groups"], 16, **policy_cfg).cpu().eval()
    policy.load_state_dict(state["model_state_dict"], strict=True)
    exported = StudentPolicy(policy).cpu().eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"policy_model{state['iter']}.onnx"
    torch.onnx.export(exported, dummy["policy"], str(output), input_names=["obs"],
                      output_names=["actions"], opset_version=17, dynamo=False)
    errors = []
    if not args.skip_validation:
        onnx.checker.check_model(onnx.load(str(output)))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
        assert session.get_inputs()[0].shape == [1, 530] and session.get_outputs()[0].shape == [1, 16]
        # Numerical export verification only; this does not execute a simulator.
        rng = np.random.default_rng(42)
        for i in range(32):
            obs = torch.from_numpy((rng.standard_normal((1, 530)) * (0.1, 1., 3.)[i % 3]).astype(np.float32))
            if i == 0:
                obs.zero_()
            native_obs = TensorDict({"policy": obs, "single_obs": exported.single_frame(obs),
                                     "critic": dummy["critic"]}, batch_size=[1])
            with torch.inference_mode():
                expected = policy.act_inference(native_obs).numpy()
            actual = session.run(None, {"obs": obs.numpy()})[0]
            if not np.isfinite(actual).all():
                raise ValueError("Non-finite exported action")
            np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=2e-5)
            errors.append(float(np.max(np.abs(actual - expected))))
        assert sha(checkpoint) == checkpoint_sha, "Checkpoint changed during export"
    manifest = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha, "iteration": state["iter"],
        "task": task, "run_name": agent["run_name"],
        "model_name": output.name, "onnx_sha256": sha(output), "onnxruntime": ort.__version__,
        "input": {"obs": [1, 530]}, "output": {"actions": [1, 16]},
        "deployed_networks": ["student_moe_encoder", "actor"],
        "history": {"frames": 10, "term_dims": [3, 3, 3, 12, 16, 16],
                    "order": "term-major, oldest-to-newest", "reset": "repeat first observed frame"},
        "single_frame_dim": 53, "policy_dt": 0.02,
        "leg_joint_names": leg_names, "wheel_joint_names": wheel_names,
        "nominal_leg_joint_positions": nominal_leg_positions,
        "joint_position_observation": "q - nominal_leg_joint_positions, in leg_joint_names order",
        "leg_position_target": "nominal_leg_joint_positions + leg_action_scale * raw_action, then joint clipping",
        "leg_action_scale": 0.2, "wheel_action_scale": 1. / 0.1005,
        "action_filter": "none", "phase_input": False, "mixed_commands": True,
        "validation": "skipped" if args.skip_validation else "passed",
        "parity_cases": len(errors), "max_abs_error": max(errors) if errors else None,
        "agent_config_sha256": sha(agent_path), "env_config_sha256": sha(env_path),
    }
    if task == "RobotLab-X5-MoECTS-v8":
        manifest["deployment_note"] = (
            "V8 standPose changes the nominal HFE from 0.75 to 0.80 rad. "
            "Both deployment joint-position observations and action targets must use this manifest's nominal pose; "
            "the existing V7 adapter with a hardcoded 0.75 rad action offset is not compatible."
        )
    stop_pi = saved_env["scene"]["robot"]["actuators"]["wheels"].get("stop_pi")
    if stop_pi is not None:
        # The ONNX graph returns raw policy actions. PI is a stateful control
        # layer outside the graph and must travel with the exported policy.
        pi_contract = {key: (str(value).lower() == "true" if key == "enabled" else float(value))
                       for key, value in stop_pi.items()}
        manifest["wheel_stop_pi"] = pi_contract
        manifest["wheel_stop_pi"]["reference"] = "zero during stop hold; policy target during motion"
        manifest["wheel_stop_pi"]["reset"] = "motion exit, liftoff, episode reset, controller exit, clock reset"
        deployment_pi = {key: value for key, value in pi_contract.items() if key not in ("reference", "reset")}
        deployment_pi["simulation_only"] = True
        (args.output_dir / "stop_pi.yaml").write_text(yaml.safe_dump({"stop_pi": deployment_pi}, sort_keys=False))
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
