"""CPU-only X5 reward inheritance and observation-bound tests; no simulation."""

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
X5 = ROOT / "source/robot_lab/robot_lab/tasks/x5"


def load(name):
    spec = importlib.util.spec_from_file_location(name, X5 / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p = load("parameters")
audit = load("config_audit")


PENALTIES = ("joint_acc_l2", "joint_torques_l2", "joint_power")
JOINT_NAMES = [f"{leg}_{joint}" for leg in ("RF", "LF", "RH", "LH") for joint in ("HAA", "HFE", "KFE")]
JOINT_NAMES += [f"{leg}_WHEEL" for leg in ("RF", "LF", "RH", "LH")]


def run_reward_override():
    """Execute the real override body on a minimal config, without importing Kit.

    The simulator config check separately verifies actual inherited callables.
    """
    original = SimpleNamespace()
    for name in (*PENALTIES, "base_height_l2", "undesired_contacts", "joint_pos_limits",
                 "hip_pos_penalty_l1", "joint_pos_penalty_l1"):
        setattr(original, name, SimpleNamespace(func=lambda: None, weight=-1.0,
                                               params={"asset_cfg": SimpleNamespace(name="robot")}))
    original.joint_acc_l2.weight = -1e-7
    original.joint_torques_l2.weight = -1e-4
    original.joint_power.weight = -2e-5
    actual = deepcopy(original)
    tree = ast.parse((X5 / "env_cfg.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "X5RewardsCfg")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
    scope = {"SceneEntityCfg": lambda name, **kwargs: SimpleNamespace(name=name, **kwargs),
             "X5_JOINT_NAMES": JOINT_NAMES, "X5_LEG_JOINT_NAMES": JOINT_NAMES[:12],
             "BASE_HEIGHT_TARGET": p.BASE_HEIGHT_TARGET}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(X5 / "env_cfg.py"), "exec"), scope)
    scope["__post_init__"](actual)
    return original, actual


@pytest.mark.parametrize("name,field", [("EFFORT", "effort"), ("SPEED", "velocity")])
def test_x5_ratings_match_selected_urdf(name, field):
    model = ET.parse(ROOT / "resources/x5/urdf/x5.urdf").getroot()
    limits = {j.get("name"): j.find("limit") for j in model.findall("joint") if j.get("type") != "fixed"}
    names = [f"{leg}_{joint}" for leg in ("RF", "LF", "RH", "LH") for joint in ("HAA", "HFE", "KFE")]
    names += [f"{leg}_WHEEL" for leg in ("RF", "LF", "RH", "LH")]
    assert tuple(float(limits[j].get(field)) for j in names) == getattr(p, f"X5_{name}")


@pytest.mark.parametrize("name", PENALTIES)
def test_reward_override_only_changes_joint_selection(name):
    original, actual = run_reward_override()
    before, after = getattr(original, name), getattr(actual, name)
    assert after.func is before.func
    assert after.weight == before.weight
    assert set(after.params) == {"asset_cfg"}
    assert after.params["asset_cfg"].joint_names == JOINT_NAMES
    assert after.params["asset_cfg"].preserve_order
    assert not hasattr(before.params["asset_cfg"], "joint_names")


@pytest.mark.parametrize("name", ["TORQUE_SCALE", "ACCELERATION_SCALE", "POWER_SCALE"])
def test_penalty_scaling_constants_are_removed(name):
    assert not hasattr(p, name)


@pytest.mark.parametrize("term", PENALTIES)
@pytest.mark.parametrize("field", ["func", "weight", "scale_factors"])
def test_audit_rejects_penalty_function_weight_or_scaling_changes(term, field):
    reference = {"rewards": {term: {"func": "original", "weight": -1e-7,
                                   "params": {"asset_cfg": {"joint_names": ["old"]}}}},
                 "sim": {"dt": 0.005}, "decimation": 4}
    actual = deepcopy(reference)
    actual["rewards"][term]["params"]["asset_cfg"] = {"joint_names": JOINT_NAMES, "preserve_order": True}
    actual["sim"]["dt"] = 0.0025
    actual["decimation"] = 8

    def config(value):
        return SimpleNamespace(to_dict=lambda: value, sim=SimpleNamespace(dt=value.get("sim", {}).get("dt")),
                               decimation=value.get("decimation"))

    args = (config(reference), config(actual), config({"experiment_name": "legbot"}),
            config({"experiment_name": "x5"}))
    audit.audit_legbot_equivalence(*args)
    if field == "scale_factors":
        actual["rewards"][term]["params"][field] = [2.0] * 16
    else:
        actual["rewards"][term][field] = "scaled" if field == "func" else -1.0
    with pytest.raises(AssertionError, match="Undocumented deviation"):
        audit.audit_legbot_equivalence(*args)


def test_teacher_does_not_saturate_loaded_stance():
    forces = torch.tensor([261.08, 267.67, 299.56, 307.50])
    scaled = forces.clamp(*p.CONTACT_FORCE_CLIP) * 1e-3
    torch.testing.assert_close(scaled, forces * 1e-3)
    assert len(scaled.unique()) == 4
    assert len(forces.clamp(-100, 100).unique()) == 1
    assert p.JOINT_TORQUE_CLIP[1] >= max(p.X5_EFFORT)
    assert p.JOINT_ACCELERATION_CLIP[1] >= 2 * max(p.X5_SPEED) / 0.0025


def test_height_feature_nominal_center_is_preserved():
    assert p.BASE_HEIGHT_TARGET - p.HEIGHT_SCAN_OFFSET == pytest.approx(0.42 - 0.5)


def test_difference_report_catches_added_and_removed_fields():
    result = audit.differences({"terrain": {"rows": 10}, "noise": True}, {"terrain": {"rows": 2}, "extra": 4})
    assert set(result) == {"terrain.rows", "noise", "extra"}
    assert result["terrain.rows"] == (10, 2)
