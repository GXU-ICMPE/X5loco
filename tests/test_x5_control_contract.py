"""Check the declared X5 actuator contract against the URDF without Isaac Sim.

This is a static contract test, not a substitute for check_x5_sim.py.
"""

import ast
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "source/robot_lab/robot_lab/assets/x5.py"


def literal(node, constants):
    if isinstance(node, ast.Name):
        return constants[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal(node.left, constants) + literal(node.right, constants)
    return ast.literal_eval(node)


class X5ControlContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(ASSET.read_text())
        cls.constants = {}
        for node in cls.tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name.startswith("X5_") and name != "X5_CFG":
                    cls.constants[name] = literal(node.value, cls.constants)
        model = ET.parse(ROOT / "resources/x5/urdf/x5.urdf").getroot()
        cls.joints = {joint.get("name"): joint for joint in model.findall("joint") if joint.get("type") != "fixed"}
        cls.actuators = []
        for node in ast.walk(cls.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "IdealPDActuatorCfg":
                cls.actuators.append({kw.arg: literal(kw.value, cls.constants) for kw in node.keywords})

    def test_explicit_joint_order(self):
        legs = ("RF", "LF", "RH", "LH")
        expected = [f"{leg}_{joint}" for leg in legs for joint in ("HAA", "HFE", "KFE")]
        expected += [f"{leg}_WHEEL" for leg in legs]
        self.assertEqual(self.constants["X5_JOINT_NAMES"], expected)
        self.assertEqual(set(expected), set(self.joints))

    def test_every_joint_has_exactly_one_actuator(self):
        selected = []
        for cfg in self.actuators:
            selected.extend(name for name in self.joints if any(re.fullmatch(pat, name) for pat in cfg["joint_names_expr"]))
        self.assertCountEqual(selected, list(self.joints))
        self.assertEqual(len(self.actuators), 2)

    def test_both_limit_layers_match_selected_urdf(self):
        for cfg in self.actuators:
            for name, joint in self.joints.items():
                if not any(re.fullmatch(pat, name) for pat in cfg["joint_names_expr"]):
                    continue
                for key, urdf_key in (("effort_limit", "effort"), ("velocity_limit", "velocity")):
                    for field in (key, key + "_sim"):
                        value = cfg[field]
                        if isinstance(value, dict):
                            matches = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
                            self.assertEqual(len(matches), 1)
                            value = matches[0]
                        self.assertEqual(value, float(joint.find("limit").get(urdf_key)), (name, field))

    def test_target_clips_are_inside_soft_joint_limits(self):
        for name in self.constants["X5_LEG_JOINT_NAMES"]:
            bounds = [v for pat, v in self.constants["X5_LEG_TARGET_LIMITS"].items() if re.fullmatch(pat, name)]
            self.assertEqual(len(bounds), 1)
            limit = self.joints[name].find("limit")
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            middle = (lower + upper) / 2
            self.assertGreaterEqual(bounds[0][0], middle + (lower - middle) * 0.9)
            self.assertLessEqual(bounds[0][1], middle + (upper - middle) * 0.9)
            self.assertLess(bounds[0][0], bounds[0][1])

    def test_gains_and_no_delay(self):
        self.assertEqual([(cfg["stiffness"], cfg["damping"]) for cfg in self.actuators], [(800.0, 20.0), (0.0, 3.0)])
        self.assertNotIn("DelayedPDActuatorCfg", ASSET.read_text())

    def test_preserve_rolling_cylinders_and_zero_importer_drive(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            values = {kw.arg: kw.value for kw in node.keywords}
            if node.func.attr == "UrdfFileCfg":
                self.assertFalse(ast.literal_eval(values["replace_cylinders_with_capsules"]))
                self.assertFalse(ast.literal_eval(values["fix_base"]))
                self.assertTrue(ast.literal_eval(values["merge_fixed_joints"]))
            elif node.func.attr == "PDGainsCfg":
                self.assertEqual(ast.literal_eval(values["stiffness"]), 0)
                self.assertEqual(ast.literal_eval(values["damping"]), 0)

    def test_task_registration_is_isolated(self):
        path = ROOT / "source/robot_lab/robot_lab/tasks/x5/__init__.py"
        tree = ast.parse(path.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "register"]
        self.assertEqual(len(calls), 1)
        task_id = next(kw.value for kw in calls[0].keywords if kw.arg == "id")
        self.assertEqual(ast.literal_eval(task_id), "RobotLab-X5-MoECTS-v0")


if __name__ == "__main__":
    unittest.main()
