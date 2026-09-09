"""Offline checks for the selected X5 model; no Isaac Sim launch or file writes.

Run from any directory with a Python environment containing NumPy.
These checks do not establish PhysX import correctness or controller stability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ASSET_ROOT = Path(__file__).resolve().parents[2] / "resources/x5"
LEGS = ("RF", "LF", "RH", "LH")
LEG_JOINTS = tuple(f"{leg}_{joint}" for leg in LEGS for joint in ("HAA", "HFE", "KFE"))
WHEEL_JOINTS = tuple(f"{leg}_WHEEL" for leg in LEGS)
# A reference pose from the selected migration source, not a validated controller.
REFERENCE_POSE = {"HAA": 0.0, "HFE": 0.75, "KFE": -1.5, "WHEEL": 0.0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def asset_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"Asset path escapes package: {relative}")
    require(path.is_file(), f"Missing asset file: {path}")
    return path


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        roll, pitch, yaw = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        transform[:3, :3] = (
            rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)
        )
    return transform


def validate(asset_root: Path = ASSET_ROOT) -> dict:
    root = asset_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["schema_version"] == 1, "Unsupported manifest schema")
    expected = manifest["expected"]
    for relative, info in manifest["files"].items():
        path = asset_file(root, relative)
        require(path.stat().st_size == info["size_bytes"], f"Size mismatch: {relative}")
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        require(checksum == info["sha256"], f"SHA-256 mismatch: {relative}")

    urdf = asset_file(root, manifest["urdf"])
    model = ET.parse(urdf).getroot()
    meshes = set()
    for mesh in model.iter("mesh"):
        path = asset_file(root, str(urdf.parent / mesh.attrib["filename"]))
        meshes.add(str(path.relative_to(root)))
        require(mesh.get("scale", "1 1 1") == "1 1 1", "Unexpected mesh scale")
    require(meshes == set(manifest["files"]) - {manifest["urdf"]}, "Mesh inventory mismatch")

    links = {link.attrib["name"]: link for link in model.findall("link")}
    joints = model.findall("joint")
    require(len(links) == len(model.findall("link")) == expected["links"], "Link count/name mismatch")
    require(len({joint.attrib["name"] for joint in joints}) == len(joints), "Duplicate joint name")
    children = {joint.find("child").attrib["link"] for joint in joints}
    require(set(links) - children == {expected["root_link"]}, "Expected single root: base")
    require(len(children) == len(joints) == len(links) - 1, "Invalid kinematic tree")
    for kind in ("fixed", "revolute", "continuous"):
        count = sum(joint.attrib["type"] == kind for joint in joints)
        require(count == expected[f"{kind}_joints"], f"Wrong {kind} joint count")
    movable = {joint.attrib["name"] for joint in joints if joint.attrib["type"] != "fixed"}
    require(movable == set(LEG_JOINTS + WHEEL_JOINTS), "Unexpected actuated joints")

    mass = 0.0
    for name, link in links.items():
        value = float(link.find("inertial/mass").attrib["value"])
        require(np.isfinite(value) and value > 0, f"Invalid mass: {name}")
        mass += value
        inertia = {key: float(value) for key, value in link.find("inertial/inertia").attrib.items()}
        tensor = np.array([
            [inertia["ixx"], inertia["ixy"], inertia["ixz"]],
            [inertia["ixy"], inertia["iyy"], inertia["iyz"]],
            [inertia["ixz"], inertia["iyz"], inertia["izz"]],
        ])
        require(np.isfinite(tensor).all(), f"Non-finite inertia: {name}")
        eigenvalues = np.linalg.eigvalsh(tensor)
        require(eigenvalues[0] > 0, f"Non-positive inertia: {name}")
        require(eigenvalues[2] <= sum(eigenvalues[:2]) + 1e-9, f"Inertia triangle inequality: {name}")
    require(abs(mass - expected["mass_kg"]) < 1e-6, "Total mass mismatch")

    transforms = {expected["root_link"]: np.eye(4)}
    pending = list(joints)
    while pending:
        ready = [joint for joint in pending if joint.find("parent").attrib["link"] in transforms]
        require(ready, "Disconnected or cyclic kinematic tree")
        for joint in ready:
            name = joint.attrib["name"]
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            require(child in links, f"Unknown child link: {child}")
            motion = np.eye(4)
            if joint.attrib["type"] != "fixed":
                suffix = name.rsplit("_", 1)[1]
                axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
                expected_axis = [1, 0, 0] if suffix == "HAA" else [0, 1, 0]
                require(np.array_equal(axis, expected_axis), f"Axis mismatch: {name}")
                limit = joint.find("limit")
                effort = {"HAA": 300, "HFE": 250, "KFE": 250, "WHEEL": 30}[suffix]
                speed = {"HAA": 6, "HFE": 6, "KFE": 8, "WHEEL": 30}[suffix]
                require(float(limit.attrib["effort"]) == effort, f"Effort mismatch: {name}")
                require(float(limit.attrib["velocity"]) == speed, f"Velocity mismatch: {name}")
                if suffix == "WHEEL":
                    require(joint.attrib["type"] == "continuous", f"Wheel must be continuous: {name}")
                else:
                    require(float(limit.attrib["lower"]) < REFERENCE_POSE[suffix] < float(limit.attrib["upper"]),
                            f"Reference pose outside limits: {name}")
                motion[:3, :3] = rotation(axis, REFERENCE_POSE[suffix])
            transforms[child] = transforms[parent] @ origin(joint.find("origin")) @ motion
            pending.remove(joint)
    require(set(transforms) == set(links), "Unreachable links")

    bottoms = []
    for leg in LEGS:
        name = f"{leg}_FOOT"
        # The selected URDF lists the rolling wheel cylinder before the motor cylinder.
        collision = links[name].find("collision")
        cylinder = collision.find("geometry/cylinder")
        require(cylinder is not None, f"Missing rolling cylinder: {name}")
        radius = float(cylinder.attrib["radius"])
        require(abs(radius - expected["wheel_radius_m"]) < 1e-9, f"Wheel radius mismatch: {name}")
        transform = transforms[name] @ origin(collision.find("origin"))
        axis_z = np.clip(transform[2, 2], -1.0, 1.0)
        half_height = radius * np.sqrt(1 - axis_z**2) + float(cylinder.attrib["length"]) / 2 * abs(axis_z)
        bottoms.append(float(transform[2, 3] - half_height))
    require(np.ptp(bottoms) < 0.0005, "Reference wheel bottoms not coplanar")
    nominal_height = float(-np.mean(bottoms))
    require(abs(nominal_height - 0.5675) < 0.001, "Unexpected reference stance geometry")
    return {
        "status": "PASS_OFFLINE_ASSET_ONLY",
        "asset_root": str(root),
        "files_verified": len(manifest["files"]),
        "meshes_verified": len(meshes),
        "links_before_fixed_joint_merge": len(links),
        "actuated_joints": len(movable),
        "mass_kg": round(mass, 6),
        "reference_leg_pose_rad": [REFERENCE_POSE[key] for key in ("HAA", "HFE", "KFE")],
        "reference_base_height_m": nominal_height,
        "wheel_bottom_spread_m": float(np.ptp(bottoms)),
        "wheel_radius_m": expected["wheel_radius_m"],
        "limitations": ["No PhysX import, collision interaction, PD stability, or locomotion validation."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT)
    args = parser.parse_args()
    try:
        result = validate(args.asset_root)
    except (ValueError, OSError, KeyError, ET.ParseError) as error:
        parser.exit(1, f"X5 asset validation failed: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
