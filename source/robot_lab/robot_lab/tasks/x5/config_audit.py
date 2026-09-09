"""Fail closed if X5 drifts outside the documented Legbot adaptation whitelist."""


def differences(reference, actual, path=""):
    if isinstance(reference, dict) and isinstance(actual, dict):
        result = {}
        for key in reference.keys() | actual.keys():
            child = f"{path}.{key}" if path else key
            if key not in reference or key not in actual:
                result[child] = (reference.get(key, "<missing>"), actual.get(key, "<missing>"))
            else:
                result.update(differences(reference[key], actual[key], child))
        return result
    equal = reference == actual
    if hasattr(equal, "all"):
        equal = equal.all()
    return {} if equal else {path: (reference, actual)}


def audit_legbot_equivalence(reference_cfg, x5_cfg, reference_runner, x5_runner):
    """Compare instantiated configs, not just inheritance names or source text."""
    delta = differences(reference_cfg.to_dict(), x5_cfg.to_dict())
    allowed = ["scene.robot.", "actions.", "sim.dt", "sim.render_interval", "decimation"]
    for group in ("policy", "single_obs", "critic"):
        for term in ("joint_pos", "joint_vel"):
            allowed.append(f"observations.{group}.{term}.params.asset_cfg.joint_names")
    for term in ("joint_acc", "joint_torque"):
        allowed.extend((f"observations.critic.{term}.params.asset_cfg.joint_names",
                        f"observations.critic.{term}.clip"))
    allowed.extend(("observations.critic.contact_force.params.sensor_cfg.body_names",
                    "observations.critic.contact_force.params.sensor_cfg.preserve_order",
                    "observations.critic.contact_force.clip", "observations.critic.height_scan.params.offset"))
    for term in ("joint_acc_l2", "joint_power", "joint_torques_l2"):
        allowed.extend((f"rewards.{term}.params.asset_cfg.joint_names",
                        f"rewards.{term}.params.asset_cfg.preserve_order"))
    allowed.extend(("rewards.base_height_l2.params.target_height",
                    "rewards.undesired_contacts.params.sensor_cfg.body_names"))
    for term in ("joint_pos_limits", "hip_pos_penalty_l1", "joint_pos_penalty_l1"):
        allowed.append(f"rewards.{term}.params.asset_cfg.joint_names")
    unexpected = {key: value for key, value in delta.items()
                  if not any(key.startswith(prefix) if prefix.endswith(".") else key == prefix for prefix in allowed)}
    assert not unexpected, f"Undocumented deviation from Legbot: {unexpected}"
    runner_delta = differences(reference_runner.to_dict(), x5_runner.to_dict())
    assert set(runner_delta) == {"experiment_name"}, runner_delta
    assert x5_cfg.sim.dt * x5_cfg.decimation == reference_cfg.sim.dt * reference_cfg.decimation
    return {
        "baseline": "RobotLab-Legbot-v0",
        "unchanged": ["terrain", "commands", "curriculum", "events", "terminations",
                      "observation_noise", "reward_weights", "MoE_CTS_algorithm", "policy_period", "sensor_periods"],
        "environment_differences": {key: [repr(a), repr(b)] for key, (a, b) in sorted(delta.items())},
        "runner_differences": {key: list(value) for key, value in runner_delta.items()},
    }
