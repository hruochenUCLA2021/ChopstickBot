#!/usr/bin/env python3
"""
Programmatically generate a symmetric lower-body humanoid ("PhoneBot") URDF + MJCF.

Structure (config-driven):
- Base motor (root)
- Trunk box (one branch)
- Two legs (two branches), symmetric across the YZ-plane:
  l_hip_pitch -> l_hip_roll -> l_hip_thigh -> l_hip_calf -> l_ankle_pitch -> l_ankle_roll -> l_foot
  r_* is mirrored from the left.

The generator is intentionally parameterized via YAML so you can manually tune all xyz/rpy offsets.

MJCF generation:
- Tries to convert URDF->MJCF via `urdf2mjcf` (python import or CLI).
- If conversion isn't available, the fallback MJCF in this file is NOT guaranteed to match PhoneBot.
  (In practice you should use the converter.)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path(__file__).resolve().parent

DEFAULT_MOTOR_STL = (
    REPO_ROOT
    / "reference"
    / "xm430_motor"
    / "good_conversion_stl"
    / "XL-430_new_coordinate_on_axis.STL"
)


def _indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print helper for ElementTree."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def _xyz(x: float, y: float, z: float) -> str:
    return f"{x:.6f} {y:.6f} {z:.6f}"


def _rpy(r: float, p: float, y: float) -> str:
    return f"{r:.6f} {p:.6f} {y:.6f}"


def _matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ],
        [
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ],
    ]


def _rotx(r: float) -> list[list[float]]:
    import math

    c = math.cos(r)
    s = math.sin(r)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _roty(p: float) -> list[list[float]]:
    import math

    c = math.cos(p)
    s = math.sin(p)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rotz(y: float) -> list[list[float]]:
    import math

    c = math.cos(y)
    s = math.sin(y)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _rpy_from_R(R: list[list[float]]) -> tuple[float, float, float]:
    """
    Extract URDF RPY such that:
      R == Rz(yaw) * Ry(pitch) * Rx(roll)
    (same convention as tf2::Quaternion::setRPY used by ROS tools).
    """
    import math

    # Guard for numerical drift
    r20 = max(-1.0, min(1.0, R[2][0]))
    pitch = math.asin(-r20)
    cp = math.cos(pitch)
    if abs(cp) < 1e-9:
        # Gimbal lock: roll and yaw are coupled
        roll = 0.0
        yaw = math.atan2(-R[0][1], R[1][1])
    else:
        roll = math.atan2(R[2][1], R[2][2])
        yaw = math.atan2(R[1][0], R[0][0])
    return roll, pitch, yaw


def _R_from_rpy(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    """Build rotation matrix for URDF rpy convention: R = Rz(yaw)*Ry(pitch)*Rx(roll)."""
    return _matmul3(_matmul3(_rotz(yaw), _roty(pitch)), _rotx(roll))


def _transpose3(R: list[list[float]]) -> list[list[float]]:
    return [[R[0][0], R[1][0], R[2][0]], [R[0][1], R[1][1], R[2][1]], [R[0][2], R[1][2], R[2][2]]]


def _matvec3(R: list[list[float]], v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    )


def _normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    import math

    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)

@dataclass(frozen=True)
class Params:
    # Geometry
    trunk_size_m: float = 0.05
    leg_bar_length_m: float = 0.15
    leg_bar_radius_m: float = 0.005  # diameter 1cm

    # Placement / "just make it visible" offsets
    base_to_trunk_z_m: float = 0.05
    hip_y_offset_m: float = 0.05
    hip_z_offset_m: float = 0.00
    motor_to_joint_z_m: float = 0.05  # simple spacing between motor body and downstream joint
    knee_to_leg_z_m: float = 0.05

    # Mass (rough placeholders)
    motor_mass_kg: float = 0.17
    trunk_mass_kg: float = 0.20
    small_link_mass_kg: float = 0.01
    leg_bar_mass_kg: float = 0.03

    # Mesh scale (STL likely in mm -> meters)
    motor_mesh_scale: float = 0.001


def _as_vec3(x) -> tuple[float, float, float]:
    if not (isinstance(x, (list, tuple)) and len(x) == 3):
        raise ValueError(f"Expected 3-vector, got: {x!r}")
    return (float(x[0]), float(x[1]), float(x[2]))


def _as_rpy(x) -> tuple[float, float, float]:
    return _as_vec3(x)


def _mirror_R_yz(R: list[list[float]]) -> list[list[float]]:
    """
    Mirror rotation across YZ plane (x -> -x) for a body orientation.
    Using reflection matrix M = diag(-1, 1, 1):
      R_mirror = M * R * M
    """
    M = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return _matmul3(_matmul3(M, R), M)


def _mirror_xyz_yz(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    return (-xyz[0], xyz[1], xyz[2])


def _R_from_sequence(seq: list[dict]) -> list[list[float]]:
    """
    Build a rotation matrix from an INTRINSIC rotation sequence expressed in the current local frame:
      R = R_axis1(angle1) * R_axis2(angle2) * ...
    """
    R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    for step in seq:
        axis = str(step["axis"]).lower()
        ang = float(step["angle"])
        if axis == "x":
            Ri = _rotx(ang)
        elif axis == "y":
            Ri = _roty(ang)
        elif axis == "z":
            Ri = _rotz(ang)
        else:
            raise ValueError(f"Unknown axis in rotation_sequence: {axis!r}")
        R = _matmul3(R, Ri)
    return R


def load_config(config_path: Path) -> dict:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def _add_inertial_box(
    link: ET.Element,
    mass: float,
    x: float,
    y: float,
    z: float,
    com_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=_xyz(*com_xyz), rpy=_rpy(0, 0, 0))
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    # very rough inertia for a box about its center: Ixx = 1/12 m (y^2 + z^2), ...
    ixx = (1 / 12) * mass * (y * y + z * z)
    iyy = (1 / 12) * mass * (x * x + z * z)
    izz = (1 / 12) * mass * (x * x + y * y)
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{ixx:.8f}",
        ixy="0",
        ixz="0",
        iyy=f"{iyy:.8f}",
        iyz="0",
        izz=f"{izz:.8f}",
    )


def _add_inertial_cylinder_z(link: ET.Element, mass: float, radius: float, length: float) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=_xyz(0, 0, 0), rpy=_rpy(0, 0, 0))
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    # cylinder about its center, axis along Z:
    ixx = (1 / 12) * mass * (3 * radius * radius + length * length)
    iyy = ixx
    izz = 0.5 * mass * radius * radius
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{ixx:.8f}",
        ixy="0",
        ixz="0",
        iyy=f"{iyy:.8f}",
        iyz="0",
        izz=f"{izz:.8f}",
    )


def _add_visual_mesh(
    link: ET.Element,
    mesh_path: Path,
    scale: float,
    xyz=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
) -> None:
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", xyz=_xyz(*xyz), rpy=_rpy(*rpy))
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(
        geom,
        "mesh",
        filename=str(mesh_path),
        scale=_xyz(scale, scale, scale),
    )
    mat = ET.SubElement(vis, "material", name="motor_gray")
    ET.SubElement(mat, "color", rgba="0.89804 0.91765 0.92941 1.0")


def _add_collision_mesh(
    link: ET.Element,
    mesh_path: Path,
    scale: float,
    xyz=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
) -> None:
    col = ET.SubElement(link, "collision")
    ET.SubElement(col, "origin", xyz=_xyz(*xyz), rpy=_rpy(*rpy))
    geom = ET.SubElement(col, "geometry")
    ET.SubElement(
        geom,
        "mesh",
        filename=str(mesh_path),
        scale=_xyz(scale, scale, scale),
    )


def _add_visual_box(
    link: ET.Element,
    size_xyz,
    rgba="0.89804 0.91765 0.92941 1.0",
    xyz=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
    material_name="trunk_gray",
) -> None:
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", xyz=_xyz(*xyz), rpy=_rpy(*rpy))
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "box", size=_xyz(*size_xyz))
    mat = ET.SubElement(vis, "material", name=material_name)
    ET.SubElement(mat, "color", rgba=rgba)


def _add_collision_box(link: ET.Element, size_xyz, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> None:
    col = ET.SubElement(link, "collision")
    ET.SubElement(col, "origin", xyz=_xyz(*xyz), rpy=_rpy(*rpy))
    geom = ET.SubElement(col, "geometry")
    ET.SubElement(geom, "box", size=_xyz(*size_xyz))


def _add_visual_cylinder_z(
    link: ET.Element,
    radius: float,
    length: float,
    rgba="0.89804 0.91765 0.92941 1.0",
    xyz=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
) -> None:
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", xyz=_xyz(*xyz), rpy=_rpy(*rpy))
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "cylinder", radius=f"{radius:.6f}", length=f"{length:.6f}")
    mat = ET.SubElement(vis, "material", name="chopstick_gray")
    ET.SubElement(mat, "color", rgba=rgba)


def _add_collision_cylinder_z(
    link: ET.Element,
    radius: float,
    length: float,
    xyz=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
) -> None:
    col = ET.SubElement(link, "collision")
    ET.SubElement(col, "origin", xyz=_xyz(*xyz), rpy=_rpy(*rpy))
    geom = ET.SubElement(col, "geometry")
    ET.SubElement(geom, "cylinder", radius=f"{radius:.6f}", length=f"{length:.6f}")


def _add_motor_link(
    link: ET.Element,
    motor_stl: Path,
    motor_mesh_scale: float,
    motor_mass_kg: float,
    motor_box_size_xyz: tuple[float, float, float],
    motor_com_offset_xyz: tuple[float, float, float],
) -> None:
    _add_visual_mesh(link, motor_stl, motor_mesh_scale)
    _add_collision_box(link, motor_box_size_xyz, xyz=motor_com_offset_xyz)
    _add_inertial_box(
        link,
        motor_mass_kg,
        motor_box_size_xyz[0],
        motor_box_size_xyz[1],
        motor_box_size_xyz[2],
        com_xyz=motor_com_offset_xyz,
    )


def _add_joint(
    robot: ET.Element,
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    origin_xyz=(0.0, 0.0, 0.0),
    origin_rpy=(0.0, 0.0, 0.0),
    axis_xyz: Optional[Sequence[float]] = None,
    limit: Optional[dict] = None,
) -> None:
    joint = ET.SubElement(robot, "joint", name=name, type=joint_type)
    ET.SubElement(joint, "origin", xyz=_xyz(*origin_xyz), rpy=_rpy(*origin_rpy))
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    if axis_xyz is not None:
        ET.SubElement(joint, "axis", xyz=_xyz(*axis_xyz))
    if limit is not None:
        ET.SubElement(
            joint,
            "limit",
            lower=str(limit.get("lower", -3.14159)),
            upper=str(limit.get("upper", 3.14159)),
            effort=str(limit.get("effort", 2.0)),
            velocity=str(limit.get("velocity", 5.0)),
        )


def build_urdf_from_config(cfg: dict) -> ET.ElementTree:
    """
    Build URDF structure:
    - Base motor (first link)
    - From base motor, three branches:
      1. Left leg (6 motors + foot)
      2. Right leg (mirrored)
      3. Trunk box
    """
    robot = ET.Element("robot", name=str(cfg.get("robot", {}).get("name", "phonebot")))

    assets = cfg["assets"]
    motor_stl = Path(str(assets["motor_stl"])).resolve()
    motor_mesh_scale = float(assets.get("motor_mesh_scale", 0.001))

    components = cfg.get("components", {}) or {}
    const = cfg.get("constants", {}) or {}

    motor_cfg = components.get("motor", {}) or {}
    motor_mass_kg = float(motor_cfg.get("mass_kg", const.get("motor_mass_kg", 0.17)))
    motor_box_size_xyz = _as_vec3(motor_cfg.get("size_xyz", (0.0285, 0.0465, 0.034)))
    motor_com_offset_xyz = _as_vec3(motor_cfg.get("com_offset_xyz", (0.0, 0.012, 0.0)))

    frame_cfg = components.get("frame", {}) or {}
    small_frame_mass_kg = float(frame_cfg.get("mass_kg", const.get("small_frame_mass_kg", 1e-3)))

    trunk_comp = components.get("trunk", {}) or {}
    foot_comp = components.get("foot", {}) or {}

    options = cfg.get("options", {}) or {}
    # Backward-compat:
    # - old: options.hip_pitch_motor_rotates_with_joint / hip_roll_motor_rotates_with_joint
    # - new: options.motor_rotates_with_joint: {hip_pitch: bool, hip_roll: bool, ...}
    default_rotates = bool(options.get("motor_rotates_with_joint", False)) if not isinstance(options.get("motor_rotates_with_joint", None), dict) else False
    old_hip_pitch = bool(options.get("hip_pitch_motor_rotates_with_joint", default_rotates))
    old_hip_roll = bool(options.get("hip_roll_motor_rotates_with_joint", default_rotates))
    rotates_map = options.get("motor_rotates_with_joint", None)
    if isinstance(rotates_map, dict):
        _rot_map: dict[str, bool] = {str(k): bool(v) for k, v in rotates_map.items()}
    else:
        _rot_map = {
            "hip_pitch": old_hip_pitch,
            "hip_roll": old_hip_roll,
            "hip_thigh": old_hip_roll,
            "hip_calf": old_hip_roll,
            "ankle_pitch": old_hip_roll,
            "ankle_roll": old_hip_roll,
        }

    def _motor_rotates_with_joint(stage: str) -> bool:
        return bool(_rot_map.get(stage, False))

    # Semantics:
    # - options.motor_rotates_with_joint decides whether a given motor link is placed BEFORE or AFTER its own joint.
    # - options.use_joint_frames ONLY decides whether intermediate frame links (created only when needed) get small inertia.
    #
    # Note: We only need an intermediate frame when two joints must exist on the same boundary
    # (a "false -> true" transition between consecutive motors).
    USE_JOINT_FRAMES = bool((cfg.get("options", {}) or {}).get("use_joint_frames", True))
    ADD_FRAME_INERTIA = USE_JOINT_FRAMES

    # -------------------------
    # Base motor (first body, root of robot)
    # -------------------------
    base_motor = ET.SubElement(robot, "link", name="base_motor_link")
    _add_motor_link(
        base_motor,
        motor_stl,
        motor_mesh_scale,
        motor_mass_kg,
        motor_box_size_xyz,
        motor_com_offset_xyz,
    )

    # -------------------------
    # Legs
    # -------------------------
    def _add_massless_frame_link(name: str) -> None:
        link = ET.SubElement(robot, "link", name=name)
        # NOTE: This link may become a *moving body* in MuJoCo (after URDF->MJCF conversion).
        # MuJoCo requires mass/inertia of moving bodies to be > mjMINVAL, so when requested we add a
        # small-but-safe inertial. When not requested, we leave it without inertial.
        if ADD_FRAME_INERTIA:
            _add_inertial_box(link, small_frame_mass_kg, 0.01, 0.01, 0.01)

    def _axis_in_joint_frame_for_parent_z(origin_rpy: tuple[float, float, float]) -> tuple[float, float, float]:
        """
        If a joint has origin_rpy != 0, then the joint frame is rotated by origin_rpy relative to parent.
        MuJoCo/URDF axis is expressed in the joint frame.

        We often want the axis to be parent's +Z (0,0,1) even if origin_rpy is used to mount the child.
        That requires axis_joint = R(origin_rpy)^T * z_parent.
        """
        R = _R_from_rpy(*origin_rpy)
        Rt = _transpose3(R)
        return _normalize3(_matvec3(Rt, (0.0, 0.0, 1.0)))

    def _mirror_rpy_yz(rpy_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        R = _R_from_rpy(*rpy_xyz)
        return _rpy_from_R(_mirror_R_yz(R))

    def add_leg(prefix: str, leg_cfg: dict, mirror_from_left: bool) -> None:
        """
        Build one leg:
          base_motor_link -> <prefix>_hip_pitch_mount (fixed) -> <prefix>_hip_pitch (fixed)
          then for each chain edge:
            parent_motor --(revolute about +Z)--> joint_frame --(fixed)--> child_motor
          finally:
            ankle_roll --(revolute about +Z)--> ankle_roll_joint_frame --(fixed)--> foot_link
        """
        # --- base -> hip_pitch mount frame
        hp_mount_xyz = _as_vec3(leg_cfg["hip_pitch_mount"]["xyz"])
        hp_mount_rpy = _as_rpy(leg_cfg["hip_pitch_mount"]["rpy"])
        if mirror_from_left:
            hp_mount_xyz = _mirror_xyz_yz(hp_mount_xyz)
            hp_mount_rpy = _mirror_rpy_yz(hp_mount_rpy)

        hip_pitch_mount_link = f"{prefix}_hip_pitch_mount"
        _add_massless_frame_link(hip_pitch_mount_link)
        _add_joint(
            robot,
            name=f"base_to_{prefix}_hip_pitch_mount_fixed",
            joint_type="fixed",
            parent="base_motor_link",
            child=hip_pitch_mount_link,
            origin_xyz=hp_mount_xyz,
            origin_rpy=hp_mount_rpy,
        )

        # --- hip_pitch motor body (fixed on mount)
        hip_pitch_motor_link = f"{prefix}_hip_pitch"
        hip_pitch_motor = ET.SubElement(robot, "link", name=hip_pitch_motor_link)
        _add_motor_link(
            hip_pitch_motor,
            motor_stl,
            motor_mesh_scale,
            motor_mass_kg,
            motor_box_size_xyz,
            motor_com_offset_xyz,
        )
        # --- hip_pitch stage:
        # If hip_pitch rotates with its joint -> joint at hip_pitch_mount boundary.
        # If not -> joint will be placed at hip_pitch->hip_roll boundary later.
        motors: dict[str, str] = {}
        if _motor_rotates_with_joint("hip_pitch"):
            _add_joint(
                robot,
                name=f"{prefix}_hip_pitch_joint",
                joint_type="revolute",
                parent=hip_pitch_mount_link,
                child=hip_pitch_motor_link,
                origin_xyz=(0.0, 0.0, 0.0),
                origin_rpy=(0.0, 0.0, 0.0),
                axis_xyz=(0.0, 0.0, 1.0),
                limit={"lower": -3.14159, "upper": 3.14159, "effort": 5.0, "velocity": 8.0},
            )
            motors["hip_pitch"] = hip_pitch_motor_link
        else:
            _add_joint(
                robot,
                name=f"{prefix}_hip_pitch_mount_to_motor_fixed",
                joint_type="fixed",
                parent=hip_pitch_mount_link,
                child=hip_pitch_motor_link,
                origin_xyz=(0.0, 0.0, 0.0),
                origin_rpy=(0.0, 0.0, 0.0),
            )
            motors["hip_pitch"] = hip_pitch_motor_link

        # --- hip_roll boundary definition
        # Preferred (single-style config): take hip_pitch->hip_roll from chain[0].
        # Backward-compat: if absent, use legacy left_leg.hip_roll_mount.
        chain_entries = list(leg_cfg.get("chain", []) or [])
        hr_joint_origin_xyz: tuple[float, float, float]
        hr_mount_rpy: tuple[float, float, float]
        if chain_entries and str(chain_entries[0].get("parent")) == "hip_pitch" and str(chain_entries[0].get("child")) == "hip_roll":
            edge0 = chain_entries.pop(0)
            hr_joint_origin_xyz = _as_vec3(edge0["joint_origin_xyz"])
            hr_mount_rpy = _as_rpy(edge0.get("mount_rpy", (0.0, 0.0, 0.0)))
        else:
            hr_mount = leg_cfg.get("hip_roll_mount", None)
            if hr_mount is None:
                raise ValueError(
                    "Need hip_pitch->hip_roll definition: either chain[0] = {parent: hip_pitch, child: hip_roll, ...} "
                    "or left_leg.hip_roll_mount (legacy)."
                )
            hr_joint_origin_xyz = _as_vec3(hr_mount["joint_origin_xyz"])
            if "rpy" in hr_mount:
                hr_mount_rpy = _as_rpy(hr_mount["rpy"])
            elif "rotation_sequence" in hr_mount:
                hr_mount_rpy = _rpy_from_R(_R_from_sequence(hr_mount["rotation_sequence"]))
            else:
                raise ValueError("left_leg.hip_roll_mount must have either `rpy` or `rotation_sequence`")

        if mirror_from_left:
            hr_joint_origin_xyz = _mirror_xyz_yz(hr_joint_origin_xyz)
            hr_mount_rpy = _mirror_rpy_yz(hr_mount_rpy)

        hip_pitch_out = motors["hip_pitch"]
        hip_roll_motor_link = f"{prefix}_hip_roll"
        hip_roll_motor = ET.SubElement(robot, "link", name=hip_roll_motor_link)
        _add_motor_link(
            hip_roll_motor,
            motor_stl,
            motor_mesh_scale,
            motor_mass_kg,
            motor_box_size_xyz,
            motor_com_offset_xyz,
        )

        # --- hip_pitch <-> hip_roll boundary
        hip_pitch_false = (not _motor_rotates_with_joint("hip_pitch"))
        hip_roll_true = _motor_rotates_with_joint("hip_roll")
        need_two = hip_pitch_false and hip_roll_true

        if need_two:
            bframe = f"{prefix}_hip_pitch_hip_roll_boundary_frame"
            _add_massless_frame_link(bframe)
            _add_joint(
                robot,
                name=f"{prefix}_hip_pitch_joint",
                joint_type="revolute",
                parent=hip_pitch_motor_link,
                child=bframe,
                origin_xyz=hr_joint_origin_xyz,
                origin_rpy=(0.0, 0.0, 0.0),
                axis_xyz=(0.0, 0.0, 1.0),
                limit={"lower": -3.14159, "upper": 3.14159, "effort": 5.0, "velocity": 8.0},
            )
            _add_joint(
                robot,
                name=f"{prefix}_hip_roll_joint",
                joint_type="revolute",
                parent=bframe,
                child=hip_roll_motor_link,
                origin_xyz=(0.0, 0.0, 0.0),
                origin_rpy=hr_mount_rpy,
                axis_xyz=(0.0, 0.0, 1.0),
                limit={"lower": -3.14159, "upper": 3.14159, "effort": 5.0, "velocity": 8.0},
            )
            motors["hip_roll"] = hip_roll_motor_link
        else:
            if hip_pitch_false:
                axis = _axis_in_joint_frame_for_parent_z(hr_mount_rpy)
                _add_joint(
                    robot,
                    name=f"{prefix}_hip_pitch_joint",
                    joint_type="revolute",
                    parent=hip_pitch_motor_link,
                    child=hip_roll_motor_link,
                    origin_xyz=hr_joint_origin_xyz,
                    origin_rpy=hr_mount_rpy,
                    axis_xyz=axis,
                    limit={"lower": -3.14159, "upper": 3.14159, "effort": 5.0, "velocity": 8.0},
                )
            else:
                if hip_roll_true:
                    _add_joint(
                        robot,
                        name=f"{prefix}_hip_roll_joint",
                        joint_type="revolute",
                        parent=hip_pitch_motor_link,
                        child=hip_roll_motor_link,
                        origin_xyz=hr_joint_origin_xyz,
                        origin_rpy=hr_mount_rpy,
                        axis_xyz=(0.0, 0.0, 1.0),
                        limit={"lower": -3.14159, "upper": 3.14159, "effort": 5.0, "velocity": 8.0},
                    )
                else:
                    _add_joint(
                        robot,
                        name=f"{prefix}_hip_roll_mount_fixed",
                        joint_type="fixed",
                        parent=hip_pitch_motor_link,
                        child=hip_roll_motor_link,
                        origin_xyz=hr_joint_origin_xyz,
                        origin_rpy=hr_mount_rpy,
                    )

            if not hip_roll_true:
                # hip_roll is joint-after-motor: add its own joint at hip_roll output.
                hip_roll_joint_frame = f"{prefix}_hip_roll_joint_frame"
                _add_massless_frame_link(hip_roll_joint_frame)
                _add_joint(
                    robot,
                    name=f"{prefix}_hip_roll_joint",
                    joint_type="revolute",
                    parent=hip_roll_motor_link,
                    child=hip_roll_joint_frame,
                    origin_xyz=(0.0, 0.0, 0.0),
                    origin_rpy=(0.0, 0.0, 0.0),
                    axis_xyz=(0.0, 0.0, 1.0),
                    limit={"lower": -3.14159, "upper": 3.14159, "effort": 5.0, "velocity": 8.0},
                )
                motors["hip_roll"] = hip_roll_joint_frame
            else:
                motors["hip_roll"] = hip_roll_motor_link

        # --- motor chain
        for edge in chain_entries:
            parent = str(edge["parent"])
            child = str(edge["child"])
            if parent not in motors:
                raise ValueError(f"left_leg.chain references unknown parent motor: {parent!r}")

            joint_origin_xyz = _as_vec3(edge["joint_origin_xyz"])
            mount_xyz = _as_vec3(edge.get("mount_xyz", (0.0, 0.0, 0.0)))
            mount_rpy = _as_rpy(edge.get("mount_rpy", (0.0, 0.0, 0.0)))

            if mirror_from_left:
                joint_origin_xyz = _mirror_xyz_yz(joint_origin_xyz)
                mount_xyz = _mirror_xyz_yz(mount_xyz)
                mount_rpy = _mirror_rpy_yz(mount_rpy)

            parent_link = motors[parent]  # "output frame" of the parent stage

            child_motor_link = f"{prefix}_{child}"
            child_motor = ET.SubElement(robot, "link", name=child_motor_link)
            _add_motor_link(
                child_motor,
                motor_stl,
                motor_mesh_scale,
                motor_mass_kg,
                motor_box_size_xyz,
                motor_com_offset_xyz,
            )

            if USE_JOINT_FRAMES:
                child_joint_frame = f"{prefix}_{child}_joint_frame"
                _add_massless_frame_link(child_joint_frame)

                if _motor_rotates_with_joint(child):
                    # joint first -> joint_frame, then fixed -> motor
                    _add_joint(
                        robot,
                        name=f"{prefix}_{child}_joint",
                        joint_type="revolute",
                        parent=parent_link,
                        child=child_joint_frame,
                        origin_xyz=joint_origin_xyz,
                        origin_rpy=(0.0, 0.0, 0.0),
                        axis_xyz=(0.0, 0.0, 1.0),
                        limit={"lower": -3.14159, "upper": 3.14159, "effort": 5.0, "velocity": 8.0},
                    )
                    _add_joint(
                        robot,
                        name=f"{prefix}_{child}_joint_frame_to_motor_fixed",
                        joint_type="fixed",
                        parent=child_joint_frame,
                        child=child_motor_link,
                        origin_xyz=mount_xyz,
                        origin_rpy=mount_rpy,
                    )
                    motors[child] = child_motor_link
                else:
                    # motor fixed first, joint after motor
                    _add_joint(
                        robot,
                        name=f"{prefix}_{child}_mount_fixed",
                        joint_type="fixed",
                        parent=parent_link,
                        child=child_motor_link,
                        origin_xyz=(
                            joint_origin_xyz[0] + mount_xyz[0],
                            joint_origin_xyz[1] + mount_xyz[1],
                            joint_origin_xyz[2] + mount_xyz[2],
                        ),
                        origin_rpy=mount_rpy,
                    )

                    # For the terminal ankle_roll stage with a foot block, ankle_roll's own joint
                    # is created at the foot boundary below. Avoid duplicate joint names here.
                    defer_child_joint_to_foot = (child == "ankle_roll") and (leg_cfg.get("foot", None) is not None)
                    if defer_child_joint_to_foot:
                        motors[child] = child_motor_link
                    else:
                        # motor -> joint_frame translation is -R^T * mount_xyz
                        R = _R_from_rpy(*mount_rpy)
                        Rt = _transpose3(R)
                        inv_t_vec = _matvec3(Rt, tuple(float(x) for x in mount_xyz))
                        inv_t = (-inv_t_vec[0], -inv_t_vec[1], -inv_t_vec[2])

                        _add_joint(
                            robot,
                            name=f"{prefix}_{child}_joint",
                            joint_type="revolute",
                            parent=child_motor_link,
                            child=child_joint_frame,
                            origin_xyz=inv_t,
                            origin_rpy=(0.0, 0.0, 0.0),
                            axis_xyz=(0.0, 0.0, 1.0),
                            limit={"lower": -3.14159, "upper": 3.14159, "effort": 5.0, "velocity": 8.0},
                        )
                        motors[child] = child_joint_frame
            else:
                # No joint-frame links in this mode.
                joint_xyz = (
                    joint_origin_xyz[0] + mount_xyz[0],
                    joint_origin_xyz[1] + mount_xyz[1],
                    joint_origin_xyz[2] + mount_xyz[2],
                )
                if _motor_rotates_with_joint(child):
                    # Joint belongs to child motor (joint before motor body).
                    _add_joint(
                        robot,
                        name=f"{prefix}_{child}_joint",
                        joint_type="revolute",
                        parent=parent_link,
                        child=child_motor_link,
                        origin_xyz=joint_xyz,
                        origin_rpy=mount_rpy,
                        axis_xyz=(0.0, 0.0, 1.0),
                        limit={"lower": -3.14159, "upper": 3.14159, "effort": 5.0, "velocity": 8.0},
                    )
                else:
                    # Joint-after-motor for `child`: place joint at child output using a frame.
                    _add_joint(
                        robot,
                        name=f"{prefix}_{child}_mount_fixed",
                        joint_type="fixed",
                        parent=parent_link,
                        child=child_motor_link,
                        origin_xyz=joint_xyz,
                        origin_rpy=mount_rpy,
                    )
                    defer_child_joint_to_foot = (child == "ankle_roll") and (leg_cfg.get("foot", None) is not None)
                    if defer_child_joint_to_foot:
                        motors[child] = child_motor_link
                        continue
                    child_joint_frame = f"{prefix}_{child}_joint_frame"
                    _add_massless_frame_link(child_joint_frame)
                    _add_joint(
                        robot,
                        name=f"{prefix}_{child}_joint",
                        joint_type="revolute",
                        parent=child_motor_link,
                        child=child_joint_frame,
                        origin_xyz=(0.0, 0.0, 0.0),
                        origin_rpy=(0.0, 0.0, 0.0),
                        axis_xyz=(0.0, 0.0, 1.0),
                        limit={"lower": -3.14159, "upper": 3.14159, "effort": 5.0, "velocity": 8.0},
                    )
                    motors[child] = child_joint_frame
                    continue
                motors[child] = child_motor_link

        # --- foot (optional, NO extra DOF)
        # The foot is attached with a FIXED joint to the ankle_roll output frame.
        # For debugging, you can omit ankle/foot stages from the YAML; we will just stop the chain.
        foot_cfg = leg_cfg.get("foot", None)
        if foot_cfg is not None:
            if "ankle_roll" not in motors:
                raise ValueError(
                    "left_leg.foot is provided, but chain does not contain ankle_roll stage. "
                    "Either add ankle_roll back, or comment out `left_leg.foot` for partial-chain debugging."
                )

            foot_cfg = foot_cfg or {}
            foot_joint_origin_xyz = _as_vec3(foot_cfg.get("joint_origin_xyz", (0.0, 0.0, 0.0)))
            foot_mount_xyz = _as_vec3(foot_cfg.get("mount_xyz", (0.0, 0.0, 0.0)))
            foot_mount_rpy = _as_rpy(foot_cfg.get("mount_rpy", (0.0, 0.0, 0.0)))
            if mirror_from_left:
                foot_joint_origin_xyz = _mirror_xyz_yz(foot_joint_origin_xyz)
                foot_mount_xyz = _mirror_xyz_yz(foot_mount_xyz)
                foot_mount_rpy = _mirror_rpy_yz(foot_mount_rpy)

            foot_size = tuple(
                float(x)
                for x in foot_cfg.get("box_size_xyz", foot_comp.get("size_xyz", (0.10, 0.06, 0.01)))
            )
            foot_mass = float(foot_cfg.get("mass_kg", foot_comp.get("mass_kg", 0.05)))
            foot_link = f"{prefix}_foot"
            foot = ET.SubElement(robot, "link", name=foot_link)
            _add_visual_box(foot, foot_size)
            _add_collision_box(foot, foot_size)
            _add_inertial_box(foot, foot_mass, *foot_size)

            ankle_roll_out = motors["ankle_roll"]
            if _motor_rotates_with_joint("ankle_roll"):
                _add_joint(
                    robot,
                    name=f"{prefix}_ankle_roll_to_{prefix}_foot_fixed",
                    joint_type="fixed",
                    parent=ankle_roll_out,
                    child=foot_link,
                    origin_xyz=(
                        foot_joint_origin_xyz[0] + foot_mount_xyz[0],
                        foot_joint_origin_xyz[1] + foot_mount_xyz[1],
                        foot_joint_origin_xyz[2] + foot_mount_xyz[2],
                    ),
                    origin_rpy=foot_mount_rpy,
                )
            else:
                # ankle_roll is joint-after-motor: place ankle_roll joint at the foot boundary.
                axis = _axis_in_joint_frame_for_parent_z(foot_mount_rpy)
                _add_joint(
                    robot,
                    name=f"{prefix}_ankle_roll_joint",
                    joint_type="revolute",
                    parent=ankle_roll_out,
                    child=foot_link,
                    origin_xyz=(
                        foot_joint_origin_xyz[0] + foot_mount_xyz[0],
                        foot_joint_origin_xyz[1] + foot_mount_xyz[1],
                        foot_joint_origin_xyz[2] + foot_mount_xyz[2],
                    ),
                    origin_rpy=foot_mount_rpy,
                    axis_xyz=axis,
                    limit={"lower": -3.14159, "upper": 3.14159, "effort": 5.0, "velocity": 8.0},
                )

    left = cfg["left_leg"]
    add_leg("l", left, mirror_from_left=False)
    add_leg("r", left, mirror_from_left=True)

    # Add legs first (left, then right)
    # (already added above)

    # -------------------------
    # Trunk box (last branch from base motor)
    # -------------------------
    trunk_cfg = cfg["trunk"]
    trunk = ET.SubElement(robot, "link", name="trunk_link")
    trunk_size = tuple(float(x) for x in trunk_cfg.get("box_size_xyz", trunk_comp.get("size_xyz", (0.15, 0.07, 0.01))))
    _add_visual_box(trunk, trunk_size)
    _add_collision_box(trunk, trunk_size)
    trunk_mass_kg = float(trunk_cfg.get("mass_kg", trunk_comp.get("mass_kg", 0.30)))
    _add_inertial_box(trunk, trunk_mass_kg, *trunk_size)

    # Joint after base motor: trunk rotates about Z (up)
    trunk_joint = trunk_cfg["joint"]
    _add_joint(
        robot,
        name="base_to_trunk",
        joint_type="revolute",
        parent="base_motor_link",
        child="trunk_link",
        origin_xyz=_as_vec3(trunk_joint["origin_xyz"]),
        origin_rpy=_as_rpy(trunk_joint.get("origin_rpy", (0.0, 0.0, 0.0))),
        axis_xyz=_as_vec3(trunk_joint.get("axis_xyz", (0.0, 0.0, 1.0))),
        limit={"lower": -3.14159, "upper": 3.14159, "effort": 5.0, "velocity": 8.0},
    )

    _indent(robot)
    return ET.ElementTree(robot)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_urdf_tree(path: Path, tree: ET.ElementTree) -> None:
    """Write URDF (XML format) to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def postprocess_mjcf_fix_colors(mjcf_path: Path, collision_rgba: str = "0.0 0.8 0.1 1.0") -> bool:
    """Set collision material color in MJCF (green by default)."""
    try:
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
    except Exception:
        return False

    changed = False

    # Update collision_material
    for material in root.findall(".//material[@name='collision_material']"):
        if material.get("rgba") != collision_rgba:
            material.set("rgba", collision_rgba)
            changed = True
    
    # Also update any geoms that have explicit collision-ish rgba
    for geom in root.findall(".//geom"):
        rgba = geom.get("rgba")
        if rgba and ("1.0 0.28 0.1" in rgba or "0.89804 0.91765 0.92941" in rgba):
            geom.set("rgba", collision_rgba)
            changed = True
    
    if changed:
        _indent(root)
        tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)
    return changed


def postprocess_mjcf_remove_geom_scale(mjcf_path: Path) -> bool:
    """
    MuJoCo (>=3.x) schema does not accept `scale` attribute on <geom>.
    Mesh scaling should be specified on the referenced <asset><mesh ... scale="..."/>.

    This postprocess:
    - Finds <geom ... type="mesh" ... scale="sx sy sz"> (or any geom w/ "mesh" attr + "scale")
    - Moves scale to the referenced <mesh name="..."> in <asset>, if present
    - Removes the `scale` attribute from the <geom>
    """
    try:
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
    except Exception:
        return False

    asset = root.find("asset")
    if not isinstance(asset, ET.Element):
        return False

    mesh_elems: dict[str, ET.Element] = {}
    for mesh in asset.findall("mesh"):
        name = mesh.attrib.get("name")
        if name:
            mesh_elems[name] = mesh

    changed = False
    for geom in root.findall(".//geom"):
        scale = geom.attrib.get("scale")
        mesh_name = geom.attrib.get("mesh")
        gtype = geom.attrib.get("type")
        if scale is None or mesh_name is None:
            continue
        if gtype is not None and gtype != "mesh":
            continue

        mesh_elem = mesh_elems.get(mesh_name)
        if mesh_elem is not None:
            # Only set asset scale if not already present; if it differs, keep the existing one.
            if "scale" not in mesh_elem.attrib:
                mesh_elem.set("scale", scale)
                changed = True
            # Remove geom-scale either way (MuJoCo schema violation)
            del geom.attrib["scale"]
            changed = True
        else:
            # If mesh asset not found, still remove geom-scale to avoid schema violation.
            del geom.attrib["scale"]
            changed = True

    if changed:
        _indent(root)
        tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)
    return changed


def postprocess_mjcf_prune_invalid_references(mjcf_path: Path) -> bool:
    """
    Remove MJCF elements that reference missing bodies/joints/sites.
    This makes converted files robust when upstream conversion emits stale excludes/actuators/sensors.
    """
    try:
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
    except Exception:
        return False

    changed = False

    # Collect existing names
    body_names = {b.get("name") for b in root.findall(".//body") if b.get("name")}
    joint_names = {j.get("name") for j in root.findall(".//joint") if j.get("name")}
    site_names = {s.get("name") for s in root.findall(".//site") if s.get("name")}

    # 1) contact/exclude with missing bodies
    contact = root.find("contact")
    if isinstance(contact, ET.Element):
        for ex in list(contact.findall("exclude")):
            b1 = ex.get("body1")
            b2 = ex.get("body2")
            if (b1 and b1 not in body_names) or (b2 and b2 not in body_names):
                contact.remove(ex)
                changed = True

    # 2) actuator entries referencing missing joints
    actuator = root.find("actuator")
    if isinstance(actuator, ET.Element):
        for act in list(actuator):
            jn = act.get("joint")
            if jn and jn not in joint_names:
                actuator.remove(act)
                changed = True

    # 3) sensor entries referencing missing sites
    sensor = root.find("sensor")
    if isinstance(sensor, ET.Element):
        for sen in list(sensor):
            # Patterns used by converters:
            # - site="..."
            # - objtype="site" objname="..."
            site_ref = sen.get("site")
            objtype = sen.get("objtype")
            objname = sen.get("objname")
            bad = False
            if site_ref and site_ref not in site_names:
                bad = True
            if objtype == "site" and objname and objname not in site_names:
                bad = True
            if bad:
                sensor.remove(sen)
                changed = True

    if changed:
        _indent(root)
        tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)
    return changed


def _is_mjcf_full_robot(
    mjcf_path: Path,
    required_joint_names: Sequence[str] = ("l_hip_pitch_joint", "l_hip_roll_joint", "r_hip_pitch_joint", "r_hip_roll_joint"),
    min_body_count: int = 8,
) -> bool:
    """
    Guard against "successful" conversions that output a collapsed one-body XML.
    """
    try:
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
    except Exception:
        return False

    body_count = len(root.findall(".//body"))
    if body_count < min_body_count:
        return False

    joint_names = {j.get("name", "") for j in root.findall(".//joint")}
    return all(name in joint_names for name in required_joint_names)


def _try_convert_with_mujoco_compiler(urdf_path: Path, mjcf_path: Path) -> bool:
    """
    Robust fallback: let MuJoCo load URDF directly and re-save as MJCF.
    """
    try:
        import mujoco  # type: ignore
    except Exception:
        return False

    try:
        if mjcf_path.exists():
            mjcf_path.unlink()
        model = mujoco.MjModel.from_xml_path(str(urdf_path))
        mujoco.mj_saveLastXML(str(mjcf_path), model)
        return mjcf_path.exists() and mjcf_path.stat().st_size > 0 and _is_mjcf_full_robot(mjcf_path)
    except Exception:
        return False


def try_convert_urdf_to_mjcf(urdf_path: Path, mjcf_path: Path, collision_rgba: str = "0.0 0.8 0.1 1.0") -> bool:
    """
    Best-effort URDF->MJCF conversion.

    Tries a few likely options:
    - Python API: `urdf2mjcf.convert.convert_urdf_to_mjcf` (preferred)
    - A CLI named `urdf2mjcf` (if installed)
    - A python module `urdf2mjcf` with a `__main__` supporting `python -m urdf2mjcf ...`
    """
    # 1) Preferred: python API (works with urdf2mjcf==0.2.x)
    try:
        try:
            from urdf2mjcf.convert import convert_urdf_to_mjcf
        except ModuleNotFoundError:
            # Fallback: allow running directly from the cloned reference repo without pip installing,
            # e.g. AnyMorphologyLocomotion_AnyMorL/reference_repos/urdf2mjcf
            ref_repo_pkg_root = (
                REPO_ROOT / "AnyMorphologyLocomotion_AnyMorL" / "reference_repos" / "urdf2mjcf"
            )
            if ref_repo_pkg_root.exists():
                sys.path.insert(0, str(ref_repo_pkg_root))
            from urdf2mjcf.convert import convert_urdf_to_mjcf  # type: ignore

        if mjcf_path.exists():
            mjcf_path.unlink()
        convert_urdf_to_mjcf(
            urdf_path=str(urdf_path),
            mjcf_path=str(mjcf_path),
            copy_meshes=False,
        )
        postprocess_mjcf_remove_geom_scale(mjcf_path)
        postprocess_mjcf_fix_colors(mjcf_path, collision_rgba=collision_rgba)
        postprocess_mjcf_prune_invalid_references(mjcf_path)
        if mjcf_path.exists() and mjcf_path.stat().st_size > 0 and _is_mjcf_full_robot(mjcf_path):
            return True
    except Exception:
        # If the installed urdf2mjcf has missing deps in this environment, fall through to CLI attempts.
        pass

    # 2) CLI / module entrypoint attempts
    candidates = [
        # CLI styles
        (["urdf2mjcf", str(urdf_path), str(mjcf_path)], "urdf2mjcf CLI"),
        (["urdf2mjcf", "--input", str(urdf_path), "--output", str(mjcf_path)], "urdf2mjcf CLI (flags)"),
        # python -m styles
        ([sys.executable, "-m", "urdf2mjcf", str(urdf_path), str(mjcf_path)], "python -m urdf2mjcf"),
        ([sys.executable, "-m", "urdf2mjcf", "--input", str(urdf_path), "--output", str(mjcf_path)], "python -m urdf2mjcf (flags)"),
    ]
    for cmd, _label in candidates:
        try:
            if mjcf_path.exists():
                mjcf_path.unlink()
            proc = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode == 0 and mjcf_path.exists() and mjcf_path.stat().st_size > 0:
                postprocess_mjcf_remove_geom_scale(mjcf_path)
                postprocess_mjcf_fix_colors(mjcf_path, collision_rgba=collision_rgba)
                postprocess_mjcf_prune_invalid_references(mjcf_path)
                if _is_mjcf_full_robot(mjcf_path):
                    return True
        except FileNotFoundError:
            continue
        except Exception:
            continue

    # 3) Robust fallback conversion: MuJoCo's native URDF compiler.
    if _try_convert_with_mujoco_compiler(urdf_path, mjcf_path):
        postprocess_mjcf_fix_colors(mjcf_path, collision_rgba=collision_rgba)
        postprocess_mjcf_prune_invalid_references(mjcf_path)
        return True
    return False


def build_fallback_mjcf(params: Params, motor_stl: Path) -> str:
    """
    Minimal MJCF (MuJoCo XML) that matches the URDF kinematics approximately.

    Notes:
    - Uses primitive geoms for trunk + leg.
    - Motor mesh is not embedded here (to avoid mesh path + scaling headaches in MJCF);
      you can extend later if you want mesh visuals in MuJoCo as well.
    """
    # MuJoCo uses: z up, x forward, y left (same as common robotics convention).
    trunk = params.trunk_size_m
    bar_len = params.leg_bar_length_m
    bar_rad = params.leg_bar_radius_m

    # positions are simplistic and chosen to be readable/visible
    base_to_trunk = params.base_to_trunk_z_m
    hip_y = params.hip_y_offset_m

    xml = f"""\
<mujoco model="chopstick_bot">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <joint limited="true" damping="0.2"/>
    <geom contype="1" conaffinity="1" friction="0.8 0.1 0.1" density="800"/>
  </default>

  <worldbody>
    <!-- base motor body (visual placeholder) -->
    <body name="base_motor" pos="0 0 0">
      <geom type="box" size="0.02 0.02 0.02" rgba="0.6 0.6 0.6 1"/>

      <!-- trunk rotates about Z after motor -->
      <body name="trunk" pos="0 0 {base_to_trunk:.6f}">
        <joint name="base_to_trunk" type="hinge" axis="0 0 1" range="-3.14159 3.14159"/>
        <geom type="box" size="{trunk/2:.6f} {trunk/2:.6f} {trunk/2:.6f}" rgba="0.1 0.2 0.8 1"/>

        <!-- left leg -->
        <body name="left_hip_motor" pos="0 {hip_y:.6f} 0">
          <geom type="box" size="0.02 0.02 0.02" rgba="0.6 0.6 0.6 1"/>
          <body name="left_hip_out" pos="0 0 {params.motor_to_joint_z_m:.6f}">
            <joint name="left_hip_joint" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
            <body name="left_knee_motor" pos="0 0 {params.motor_to_joint_z_m:.6f}">
              <geom type="box" size="0.02 0.02 0.02" rgba="0.6 0.6 0.6 1"/>
              <body name="left_knee_out" pos="0 0 {params.knee_to_leg_z_m:.6f}">
                <joint name="left_knee_joint" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
                <body name="left_leg" pos="0 0 {-bar_len/2:.6f}">
                  <geom type="cylinder" size="{bar_rad:.6f} {bar_len/2:.6f}" rgba="0.8 0.7 0.2 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>

        <!-- right leg -->
        <body name="right_hip_motor" pos="0 {-hip_y:.6f} 0">
          <geom type="box" size="0.02 0.02 0.02" rgba="0.6 0.6 0.6 1"/>
          <body name="right_hip_out" pos="0 0 {params.motor_to_joint_z_m:.6f}">
            <joint name="right_hip_joint" type="hinge" axis="0 -1 0" range="-1.57 1.57"/>
            <body name="right_knee_motor" pos="0 0 {params.motor_to_joint_z_m:.6f}">
              <geom type="box" size="0.02 0.02 0.02" rgba="0.6 0.6 0.6 1"/>
              <body name="right_knee_out" pos="0 0 {params.knee_to_leg_z_m:.6f}">
                <joint name="right_knee_joint" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
                <body name="right_leg" pos="0 0 {-bar_len/2:.6f}">
                  <geom type="cylinder" size="{bar_rad:.6f} {bar_len/2:.6f}" rgba="0.8 0.7 0.2 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>

      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="act_base_to_trunk" joint="base_to_trunk" kp="50"/>
    <position name="act_left_hip" joint="left_hip_joint" kp="30"/>
    <position name="act_left_knee" joint="left_knee_joint" kp="30"/>
    <position name="act_right_hip" joint="right_hip_joint" kp="30"/>
    <position name="act_right_knee" joint="right_knee_joint" kp="30"/>
  </actuator>
</mujoco>
"""
    # mention mesh path so user can wire in mesh later if desired
    return (
        "<!-- NOTE: motor mesh (not embedded in fallback MJCF): "
        + str(motor_stl)
        + " -->\n"
        + xml
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a symmetric lower-body humanoid URDF + MJCF for PhoneBot.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    ap.add_argument("--urdf-name", type=str, default="phonebot.urdf", help="URDF filename")
    ap.add_argument("--mjcf-name", type=str, default="phonebot.xml", help="MJCF filename")
    ap.add_argument(
        "--motor-stl",
        type=str,
        default=None,
        help="Optional override for motor STL mesh (otherwise uses config)",
    )
    ap.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "model_config.yaml"),
        help="Path to model_config.yaml",
    )
    ap.add_argument("--no-convert", action="store_true", help="Skip URDF->MJCF conversion attempt")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    urdf_path = out_dir / args.urdf_name
    mjcf_path = out_dir / args.mjcf_name
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[ERROR] config not found: {config_path}", file=sys.stderr)
        return 2
    cfg = load_config(config_path)
    # Optional CLI override
    if args.motor_stl is not None:
        cfg.setdefault("assets", {})
        cfg["assets"]["motor_stl"] = str(Path(args.motor_stl).resolve())

    # Validate motor STL exists (needed both for URDF + fallback MJCF)
    motor_stl = Path(str(cfg["assets"]["motor_stl"])).resolve()
    if not motor_stl.exists():
        print(f"[ERROR] motor STL not found: {motor_stl}", file=sys.stderr)
        return 2

    print(f"[INFO] python: {sys.executable}")
    print(f"[INFO] version: {sys.version.split()[0]}")

    # URDF generation from YAML config
    urdf_tree = build_urdf_from_config(cfg)
    _write_urdf_tree(urdf_path, urdf_tree)
    print(f"[OK] wrote URDF: {urdf_path}")

    converted = False
    if not args.no_convert:
        motor_cfg = (cfg.get("components", {}) or {}).get("motor", {}) or {}
        collision_rgba_cfg = motor_cfg.get("collision_rgba", [0.0, 0.8, 0.1, 1.0])
        collision_rgba = " ".join(str(float(x)) for x in collision_rgba_cfg)
        converted = try_convert_urdf_to_mjcf(urdf_path, mjcf_path, collision_rgba=collision_rgba)
        if converted:
            print(f"[OK] wrote MJCF (converted): {mjcf_path}")

    if not converted:
        const = cfg.get("constants", {})
        # Use an explicit joint origin from config (required) to populate fallback params.
        # Even though this is only for the fallback MJCF builder, we keep it consistent with "no fallbacks".
        try:
            mjcf_motor_to_joint_z = float(cfg["left_leg"]["hip_roll_mount"]["joint_origin_xyz"][2])
        except Exception as e:
            raise ValueError("left_leg.hip_roll_mount.joint_origin_xyz[2] is required") from e
        params = Params(
            trunk_size_m=float(cfg.get("trunk", {}).get("box_size_xyz", [0.05, 0.05, 0.05])[0]),
            leg_bar_length_m=float(const.get("leg_bar_length_m", 0.15)),
            leg_bar_radius_m=float(const.get("leg_bar_radius_m", 0.005)),
            motor_to_joint_z_m=mjcf_motor_to_joint_z,
            motor_mass_kg=float(const.get("motor_mass_kg", 0.17)),
            trunk_mass_kg=float(const.get("trunk_mass_kg", 0.20)),
            leg_bar_mass_kg=float(const.get("leg_bar_mass_kg", 0.03)),
            motor_mesh_scale=float(cfg.get("assets", {}).get("motor_mesh_scale", 0.001)),
        )
        mjcf_text = build_fallback_mjcf(params, motor_stl)
        _write_text(mjcf_path, mjcf_text)
        print(f"[OK] wrote MJCF (fallback): {mjcf_path}")
        if not args.no_convert:
            print(
                textwrap.dedent(
                    """\
                    [NOTE] URDF->MJCF conversion tool not detected/working in this environment.
                           The MJCF was generated via a simple fallback builder.
                           If you want, tell me which `urdf2mjcf` repo you installed (package name / CLI),
                           and I can wire up a direct python import path for a cleaner conversion.
                    """
                ).rstrip()
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


