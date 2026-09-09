"""Asset-package regression tests; runnable without Isaac Sim."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validate_x5_asset", ROOT / "scripts/tools/validate_x5_asset.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class X5AssetTests(unittest.TestCase):
    def test_selected_asset_passes(self):
        result = AUDIT.validate()
        self.assertEqual(result["status"], "PASS_OFFLINE_ASSET_ONLY")
        self.assertEqual(result["meshes_verified"], 13)
        self.assertEqual(result["actuated_joints"], 16)
        self.assertAlmostEqual(result["mass_kg"], 115.780333, places=6)
        self.assertAlmostEqual(result["reference_base_height_m"], 0.5675, places=3)

    def test_joint_inventory(self):
        self.assertEqual(len(set(AUDIT.LEG_JOINTS + AUDIT.WHEEL_JOINTS)), 16)
        self.assertEqual(AUDIT.LEG_JOINTS[:3], ("RF_HAA", "RF_HFE", "RF_KFE"))
        self.assertEqual(AUDIT.WHEEL_JOINTS, ("RF_WHEEL", "LF_WHEEL", "RH_WHEEL", "LH_WHEEL"))

    def test_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Missing asset file"):
                AUDIT.asset_file(Path(directory).resolve(), "missing.STL")

    def test_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "escapes package"):
                AUDIT.asset_file(Path(directory).resolve(), "../outside.STL")

    def test_corrupted_urdf_rejected(self):
        # No meshes are needed: integrity must fail at the first file, the URDF.
        manifest = json.loads((AUDIT.ASSET_ROOT / "manifest.json").read_text())
        original = (AUDIT.ASSET_ROOT / manifest["urdf"]).read_bytes()
        corrupted = original.replace(b'46.707585', b'46.707586', 1)
        self.assertNotEqual(original, corrupted)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "urdf").mkdir()
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / manifest["urdf"]).write_bytes(corrupted)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch: urdf/x5.urdf"):
                AUDIT.validate(root)

    def test_truncated_urdf_rejected(self):
        manifest = json.loads((AUDIT.ASSET_ROOT / "manifest.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "urdf").mkdir()
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / manifest["urdf"]).write_text("<robot />")
            with self.assertRaisesRegex(ValueError, "Size mismatch: urdf/x5.urdf"):
                AUDIT.validate(root)

    def test_forward_kinematics_rotation(self):
        np.testing.assert_allclose(AUDIT.rotation([0, 1, 0], np.pi / 2) @ [1, 0, 0], [0, 0, -1], atol=1e-12)
        np.testing.assert_array_equal(AUDIT.origin(None), np.eye(4))


if __name__ == "__main__":
    unittest.main()
