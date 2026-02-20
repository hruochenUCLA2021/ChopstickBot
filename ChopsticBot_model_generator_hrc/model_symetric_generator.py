#!/usr/bin/env python3
"""
Programmatically generate a simple symmetric biped ("chopstick bot") URDF + MJCF.

Design goals (initial version):
- Minimal structure: base motor -> trunk box -> 2 legs (2 motors each) -> 15cm bar.
- Motor bodies are modeled as meshes on links that do NOT rotate with the joint.
  (i.e., motor link is the parent of the revolute joint; the child link rotates.)
- Uses the XL-430 motor STL mesh (binary) from:
  reference/xm430_motor/good_conversion_stl/XL-430_new_coordinate_on_axis.STL
- Outputs:
  - chopstick_bot.urdf
  - chopstick_bot.xml   (MJCF)

MJCF generation:
- Tries URDF->MJCF conversion via `urdf2mjcf` (python import or CLI).
- If conversion fails, this script exits with an error (no fallback MJCF builder).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
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
    # Inertia tensor below is about the box COM principal axes.
    # If the link frame is not at COM, put COM in inertial origin.
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
    # Visual uses detailed mesh.
    _add_visual_mesh(link, motor_stl, motor_mesh_scale)
    # Collision uses simple box proxy per motor datasheet dimensions.
    _add_collision_box(link, motor_box_size_xyz, xyz=motor_com_offset_xyz)
    # Inertial uses box model with COM offset.
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
    - From base motor, three branches (in order):
      1. Left leg (l_hip_pitch -> l_hip_roll -> bar)
      2. Right leg (r_hip_pitch -> r_hip_roll -> bar)
      3. Trunk box
    """
    robot = ET.Element("robot", name=str(cfg.get("robot", {}).get("name", "chopstick_bot")))

    assets = cfg["assets"]
    motor_stl = Path(str(assets["motor_stl"])).resolve()
    motor_mesh_scale = float(assets.get("motor_mesh_scale", 0.001))

    components = cfg.get("components", {}) or {}
    const = cfg.get("constants", {}) or {}

    motor_cfg = components.get("motor", {}) or {}
    motor_mass_kg = float(motor_cfg.get("mass_kg", const.get("motor_mass_kg", 0.17)))
    motor_box_size_xyz = _as_vec3(motor_cfg.get("size_xyz", (0.0285, 0.0465, 0.034)))
    motor_com_offset_xyz = _as_vec3(motor_cfg.get("com_offset_xyz", (0.0, 0.012, 0.0)))

    trunk_comp = components.get("trunk", {}) or {}
    trunk_mass_kg = float(trunk_comp.get("mass_kg", const.get("trunk_mass_kg", 0.20)))

    leg_bar_comp = components.get("leg_bar", {}) or {}
    leg_bar_length_m = float(leg_bar_comp.get("length_m", const.get("leg_bar_length_m", 0.15)))
    leg_bar_radius_m = float(leg_bar_comp.get("radius_m", const.get("leg_bar_radius_m", 0.005)))
    leg_bar_mass_kg = float(leg_bar_comp.get("mass_kg", const.get("leg_bar_mass_kg", 0.03)))

    options = cfg.get("options", {}) or {}
    # Backward-compat: old single flag sets the default for both
    default_rotates = bool(options.get("motor_rotates_with_joint", False))
    hip_pitch_rotates = bool(options.get("hip_pitch_motor_rotates_with_joint", default_rotates))
    hip_roll_rotates = bool(options.get("hip_roll_motor_rotates_with_joint", default_rotates))

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
    # Left leg (first branch from base motor)
    # Simplified structure: base -> l_hip_pitch (motor) -> l_hip_roll (motor) -> bar
    # No intermediate *_out links - motors attach directly via revolute joints
    # -------------------------
    def _add_massless_frame_link(name: str) -> None:
        link = ET.SubElement(robot, "link", name=name)
        # NOTE: This link may become a *moving body* in MuJoCo (after URDF->MJCF conversion).
        # MuJoCo requires mass and inertia of moving bodies to be > mjMINVAL.
        #
        # Also, our `_add_inertial_box` formats inertia with limited decimals; if the numbers are
        # too small they can round to 0.0 and MuJoCo will error.
        #
        # So we use a small-but-safe inertial here. This link has no visuals/collisions, so it
        # won't affect rendering; dynamically it adds negligible inertia relative to the motors.
        _add_inertial_box(link, 1e-3, 0.01, 0.01, 0.01)


    def add_leg_from_left_config(
        prefix: str,
        hip_pitch_mount_xyz,
        hip_pitch_mount_rpy,
        hip_roll_mount_R,
        hip_roll_joint_origin_xyz: tuple[float, float, float],
        bar_cfg: dict,
    ) -> None:
        """
        Build one leg given explicit mounts in the parent frames.
        `prefix` should be "l" or "r".
        """
        hip_pitch_mount_xyz = _as_vec3(hip_pitch_mount_xyz)
        hip_pitch_mount_rpy = _as_rpy(hip_pitch_mount_rpy)
        R_base_hip_pitch = _R_from_rpy(*hip_pitch_mount_rpy)

        # A fixed mount frame for the hip_pitch assembly (helps keep symmetry/mirroring clean).
        hip_pitch_mount_link = f"{prefix}_hip_pitch_mount"
        _add_massless_frame_link(hip_pitch_mount_link)

        _add_joint(
            robot,
            name=f"base_to_{prefix}_hip_pitch_mount_fixed",
            joint_type="fixed",
            parent="base_motor_link",
            child=hip_pitch_mount_link,
            origin_xyz=hip_pitch_mount_xyz,
            origin_rpy=hip_pitch_mount_rpy,
        )

        # First motor body link
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

        if hip_pitch_rotates:
            # Joint first, then motor body: hip_pitch motor rotates with its own joint.
            _add_joint(
                robot,
                name=f"{prefix}_hip_pitch_joint",
                joint_type="revolute",
                parent=hip_pitch_mount_link,
                child=hip_pitch_motor_link,
                origin_xyz=(0.0, 0.0, 0.0),
                origin_rpy=(0.0, 0.0, 0.0),
                axis_xyz=(0.0, 0.0, 1.0),
                limit={"lower": -1.57, "upper": 1.57, "effort": 3.0, "velocity": 6.0},
            )
            hip_pitch_output_parent = hip_pitch_motor_link
        else:
            # Motor body first, then joint: motor does NOT rotate with its own joint.
            _add_joint(
                robot,
                name=f"{prefix}_hip_pitch_mount_to_motor_fixed",
                joint_type="fixed",
                parent=hip_pitch_mount_link,
                child=hip_pitch_motor_link,
                origin_xyz=(0.0, 0.0, 0.0),
                origin_rpy=(0.0, 0.0, 0.0),
            )
            hip_pitch_joint_frame = f"{prefix}_hip_pitch_joint_frame"
            _add_massless_frame_link(hip_pitch_joint_frame)
            _add_joint(
                robot,
                name=f"{prefix}_hip_pitch_joint",
                joint_type="revolute",
                parent=hip_pitch_motor_link,
                child=hip_pitch_joint_frame,
                origin_xyz=hip_roll_joint_origin_xyz,
                origin_rpy=(0.0, 0.0, 0.0),
                axis_xyz=(0.0, 0.0, 1.0),
                limit={"lower": -1.57, "upper": 1.57, "effort": 3.0, "velocity": 6.0},
            )
            hip_pitch_output_parent = hip_pitch_joint_frame

        # Second motor: hip_roll
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

        hip_roll_rpy = _rpy_from_R(hip_roll_mount_R)
        if hip_roll_rotates:
            # hip_roll motor rotates with its own joint: joint before motor body
            _add_joint(
                robot,
                name=f"{prefix}_hip_roll_joint",
                joint_type="revolute",
                parent=hip_pitch_output_parent,
                child=hip_roll_motor_link,
                origin_xyz=hip_roll_joint_origin_xyz,
                origin_rpy=hip_roll_rpy,
                axis_xyz=(0.0, 0.0, 1.0),
                limit={"lower": -1.57, "upper": 1.57, "effort": 3.0, "velocity": 6.0},
            )
        else:
            # hip_roll motor does NOT rotate with its own joint:
            # mount motor with fixed joint, then put joint after the motor.
            _add_joint(
                robot,
                name=f"{prefix}_hip_pitch_to_{prefix}_hip_roll_fixed",
                joint_type="fixed",
                parent=hip_pitch_output_parent,
                child=hip_roll_motor_link,
                origin_xyz=(0.0, 0.0, 0.0),
                origin_rpy=hip_roll_rpy,
            )

        # Final leg bar ("chopstick") - attached directly to hip_roll joint
        # Bar should be along Y direction of motor (so it swings back/forth)
        # Cylinder default is along Z, so rotate around X by 90° to make it along Y
        leg_bar_link = f"{prefix}_leg_bar"
        leg_bar = ET.SubElement(robot, "link", name=leg_bar_link)
        bar_geom = bar_cfg["geom"]
        bar_geom_rpy = _as_rpy(bar_geom["rpy"])
        bar_geom_xyz = _as_vec3(bar_geom["xyz"])
        _add_visual_cylinder_z(
            leg_bar,
            leg_bar_radius_m,
            leg_bar_length_m,
            xyz=bar_geom_xyz,
            rpy=bar_geom_rpy,
        )
        _add_collision_cylinder_z(
            leg_bar,
            leg_bar_radius_m,
            leg_bar_length_m,
            xyz=bar_geom_xyz,
            rpy=bar_geom_rpy,
        )
        _add_inertial_cylinder_z(leg_bar, leg_bar_mass_kg, leg_bar_radius_m, leg_bar_length_m)

        bar_joint_xyz = _as_vec3(bar_cfg["joint_origin_xyz"])
        if hip_roll_rotates:
            # hip_roll joint already rotates the hip_roll motor; attach the bar rigidly to the motor.
            _add_joint(
                robot,
                name=f"{prefix}_hip_roll_to_{prefix}_leg_bar_fixed",
                joint_type="fixed",
                parent=hip_roll_motor_link,
                child=leg_bar_link,
                origin_xyz=bar_joint_xyz,
                origin_rpy=(0.0, 0.0, 0.0),
            )
        else:
            # Joint after hip_roll motor: motor does not rotate, bar rotates.
            _add_joint(
                robot,
                name=f"{prefix}_hip_roll_joint",
                joint_type="revolute",
                parent=hip_roll_motor_link,
                child=leg_bar_link,
                origin_xyz=bar_joint_xyz,
                origin_rpy=(0.0, 0.0, 0.0),
                axis_xyz=(0.0, 0.0, 1.0),
                limit={"lower": -1.57, "upper": 1.57, "effort": 3.0, "velocity": 6.0},
            )

    # Left leg from config
    left = cfg["left_leg"]
    l_hp_xyz = left["hip_pitch_mount"]["xyz"]
    l_hp_rpy = left["hip_pitch_mount"]["rpy"]

    # hip_roll mount can be given as URDF rpy or as a sequence
    hr_mount = left["hip_roll_mount"]
    if "rpy" in hr_mount:
        l_hr_R = _R_from_rpy(*_as_rpy(hr_mount["rpy"]))
    elif "rotation_sequence" in hr_mount:
        l_hr_R = _R_from_sequence(hr_mount["rotation_sequence"])
    else:
        raise ValueError("left_leg.hip_roll_mount must have either `rpy` or `rotation_sequence`")

    if "joint_origin_xyz" not in hr_mount:
        raise ValueError("left_leg.hip_roll_mount.joint_origin_xyz is required (no fallback)")
    l_hr_joint_origin_xyz = _as_vec3(hr_mount["joint_origin_xyz"])
    add_leg_from_left_config("l", l_hp_xyz, l_hp_rpy, l_hr_R, l_hr_joint_origin_xyz, left["bar"])

    # Right leg is mirrored across YZ plane from the left leg (position/orientation symmetry)
    r_hp_xyz = _mirror_xyz_yz(_as_vec3(l_hp_xyz))
    l_hp_R = _R_from_rpy(*_as_rpy(l_hp_rpy))
    r_hp_R = _mirror_R_yz(l_hp_R)
    r_hp_rpy = _rpy_from_R(r_hp_R)

    r_hr_R = _mirror_R_yz(l_hr_R)

    # bar config is mirrored: link-local geom xyz mirrors x only (since it's in the bar link frame),
    # but since our bar geom uses Y offsets, mirroring across YZ does not change it; we keep as-is.
    # If you later add nonzero X offsets in bar geom, the mirror will handle it.
    r_bar_cfg = dict(left["bar"])
    r_bar_cfg["geom"] = dict(left["bar"]["geom"])
    r_bar_cfg["geom"]["xyz"] = list(_mirror_xyz_yz(_as_vec3(left["bar"]["geom"]["xyz"])))
    r_bar_cfg["joint_origin_xyz"] = list(_mirror_xyz_yz(_as_vec3(left["bar"]["joint_origin_xyz"])))

    r_hr_joint_origin_xyz = _mirror_xyz_yz(l_hr_joint_origin_xyz)
    add_leg_from_left_config("r", r_hp_xyz, r_hp_rpy, r_hr_R, r_hr_joint_origin_xyz, r_bar_cfg)

    # Add legs first (left, then right)
    # (already added above)

    # -------------------------
    # Trunk box (last branch from base motor)
    # -------------------------
    trunk_cfg = cfg["trunk"]
    trunk = ET.SubElement(robot, "link", name="trunk_link")
    trunk_comp = components.get("trunk", {}) or {}
    trunk_size = tuple(
        float(x)
        for x in trunk_cfg.get("box_size_xyz", trunk_comp.get("size_xyz", (0.05, 0.05, 0.05)))
    )
    _add_visual_box(trunk, trunk_size)
    _add_collision_box(trunk, trunk_size)
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

        convert_urdf_to_mjcf(
            urdf_path=str(urdf_path),
            mjcf_path=str(mjcf_path),
            copy_meshes=False,
        )
        postprocess_mjcf_remove_geom_scale(mjcf_path)
        postprocess_mjcf_fix_colors(mjcf_path, collision_rgba=collision_rgba)
        return mjcf_path.exists() and mjcf_path.stat().st_size > 0
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
            proc = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode == 0 and mjcf_path.exists() and mjcf_path.stat().st_size > 0:
                return True
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a symmetric biped URDF + MJCF for ChopstickBot.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    ap.add_argument("--urdf-name", type=str, default="chopstick_bot.urdf", help="URDF filename")
    ap.add_argument("--mjcf-name", type=str, default="chopstick_bot.xml", help="MJCF filename")
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

    # Validate motor STL exists (needed for URDF generation)
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

    if not converted and not args.no_convert:
        print("[ERROR] URDF->MJCF conversion failed. No fallback MJCF builder is used.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


