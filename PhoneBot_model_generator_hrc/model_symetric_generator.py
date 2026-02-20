#!/usr/bin/env python3
"""
PhoneBot model generator (from scratch):
- Generates BOTH URDF and MJCF directly from YAML config.
- MJCF includes visual MOTOR MESH geoms (so `simulate` shows the STL), and collision boxes.
- MJCF can include a base `freejoint` so robot is not fixed to world.

Important:
- We do NOT rely on any URDF->MJCF converter here, because those were dropping meshes.
- `options.motor_rotates_with_joint` and `options.use_joint_frames` are independent.
  If config requests a physically-contradictory combination (e.g. joint-after-motor with no joint frames),
  we still generate and print warnings; the config is responsible for physical correctness.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path(__file__).resolve().parent


def _indent(elem: ET.Element, level: int = 0) -> None:
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


def _as_vec3(x) -> tuple[float, float, float]:
    if not (isinstance(x, (list, tuple)) and len(x) == 3):
        raise ValueError(f"Expected 3-vector, got: {x!r}")
    return (float(x[0]), float(x[1]), float(x[2]))


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
    c = math.cos(r)
    s = math.sin(r)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _roty(p: float) -> list[list[float]]:
    c = math.cos(p)
    s = math.sin(p)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rotz(y: float) -> list[list[float]]:
    c = math.cos(y)
    s = math.sin(y)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _R_from_rpy(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    # URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
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
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _mirror_R_yz(R: list[list[float]]) -> list[list[float]]:
    # Mirror across YZ plane with M=diag(-1,1,1): Rm = M R M
    M = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return _matmul3(_matmul3(M, R), M)


def _mirror_xyz_yz(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    return (-xyz[0], xyz[1], xyz[2])


def _quat_from_R(R: list[list[float]]) -> tuple[float, float, float, float]:
    # Returns MuJoCo quat (w, x, y, z)
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2][1] - R[1][2]) / S
        y = (R[0][2] - R[2][0]) / S
        z = (R[1][0] - R[0][1]) / S
    elif (R[0][0] > R[1][1]) and (R[0][0] > R[2][2]):
        S = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0
        w = (R[2][1] - R[1][2]) / S
        x = 0.25 * S
        y = (R[0][1] + R[1][0]) / S
        z = (R[0][2] + R[2][0]) / S
    elif R[1][1] > R[2][2]:
        S = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0
        w = (R[0][2] - R[2][0]) / S
        x = (R[0][1] + R[1][0]) / S
        y = 0.25 * S
        z = (R[1][2] + R[2][1]) / S
    else:
        S = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0
        w = (R[1][0] - R[0][1]) / S
        x = (R[0][2] + R[2][0]) / S
        y = (R[1][2] + R[2][1]) / S
        z = 0.25 * S
    # normalize
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / n, x / n, y / n, z / n)


def _quat_str(q: tuple[float, float, float, float]) -> str:
    return f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"


def load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


# --------------------------- URDF helpers ---------------------------

def _urdf_add_inertial_box(
    link: ET.Element,
    mass: float,
    size_xyz: tuple[float, float, float],
    com_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    x, y, z = size_xyz
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=_xyz(*com_xyz), rpy=_rpy(0.0, 0.0, 0.0))
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    ixx = (1.0 / 12.0) * mass * (y * y + z * z)
    iyy = (1.0 / 12.0) * mass * (x * x + z * z)
    izz = (1.0 / 12.0) * mass * (x * x + y * y)
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


def _urdf_add_visual_mesh(link: ET.Element, mesh_path: Path, scale: float, rgba: list[float]) -> None:
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", xyz=_xyz(0, 0, 0), rpy=_rpy(0, 0, 0))
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "mesh", filename=str(mesh_path), scale=_xyz(scale, scale, scale))
    mat = ET.SubElement(vis, "material", name="motor_gray")
    ET.SubElement(mat, "color", rgba=" ".join(str(float(x)) for x in rgba))


def _urdf_add_collision_box(link: ET.Element, size_xyz: tuple[float, float, float], xyz=(0.0, 0.0, 0.0)) -> None:
    col = ET.SubElement(link, "collision")
    ET.SubElement(col, "origin", xyz=_xyz(*xyz), rpy=_rpy(0, 0, 0))
    geom = ET.SubElement(col, "geometry")
    ET.SubElement(geom, "box", size=_xyz(*size_xyz))


def _urdf_add_visual_box(link: ET.Element, size_xyz: tuple[float, float, float], rgba: list[float]) -> None:
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", xyz=_xyz(0, 0, 0), rpy=_rpy(0, 0, 0))
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "box", size=_xyz(*size_xyz))
    mat = ET.SubElement(vis, "material", name="box_gray")
    ET.SubElement(mat, "color", rgba=" ".join(str(float(x)) for x in rgba))


def _urdf_add_joint(
    robot: ET.Element,
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    origin_xyz=(0.0, 0.0, 0.0),
    origin_rpy=(0.0, 0.0, 0.0),
    axis_xyz: Optional[tuple[float, float, float]] = None,
) -> None:
    j = ET.SubElement(robot, "joint", name=name, type=joint_type)
    ET.SubElement(j, "origin", xyz=_xyz(*origin_xyz), rpy=_rpy(*origin_rpy))
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)
    if axis_xyz is not None:
        ET.SubElement(j, "axis", xyz=_xyz(*axis_xyz))
    if joint_type == "revolute":
        ET.SubElement(j, "limit", lower="-3.14159", upper="3.14159", effort="5.0", velocity="8.0")


def build_urdf(cfg: dict) -> ET.ElementTree:
    robot = ET.Element("robot", name=str(cfg.get("robot", {}).get("name", "phonebot")))
    assets = cfg["assets"]
    comp = cfg["components"]
    motor_stl = Path(str(assets["motor_stl"])).resolve()
    scale = float(assets.get("motor_mesh_scale", 0.001))

    motor_c = comp["motor"]
    motor_size = _as_vec3(motor_c["size_xyz"])
    motor_com = _as_vec3(motor_c["com_offset_xyz"])
    motor_mass = float(motor_c["mass_kg"])
    motor_vis_rgba = list(motor_c.get("visual_rgba", [0.7, 0.7, 0.7, 1.0]))

    frame_c = comp.get("frame", {})
    frame_mass = float(frame_c.get("mass_kg", 0.001))
    frame_size = _as_vec3(frame_c.get("size_xyz", (0.01, 0.01, 0.01)))

    trunk_c = comp["trunk"]
    foot_c = comp["foot"]
    trunk_size = _as_vec3(trunk_c["size_xyz"])
    trunk_mass = float(trunk_c["mass_kg"])
    trunk_rgba = list(trunk_c.get("visual_rgba", [0.7, 0.7, 0.7, 1.0]))

    foot_size = _as_vec3(foot_c["size_xyz"])
    foot_mass = float(foot_c["mass_kg"])
    foot_rgba = list(foot_c.get("visual_rgba", [0.7, 0.7, 0.7, 1.0]))

    opts = cfg.get("options", {}) or {}
    use_joint_frames = bool(opts.get("use_joint_frames", True))
    rot_map = opts.get("motor_rotates_with_joint", {}) or {}

    def _rot(stage: str) -> bool:
        return bool(rot_map.get(stage, False))

    def _add_frame_link(name: str) -> None:
        link = ET.SubElement(robot, "link", name=name)
        if use_joint_frames:
            _urdf_add_inertial_box(link, frame_mass, frame_size, com_xyz=(0.0, 0.0, 0.0))

    # base motor
    base = ET.SubElement(robot, "link", name="base_motor_link")
    _urdf_add_visual_mesh(base, motor_stl, scale, motor_vis_rgba)
    _urdf_add_collision_box(base, motor_size, xyz=motor_com)
    _urdf_add_inertial_box(base, motor_mass, motor_size, com_xyz=motor_com)

    # legs
    leg = cfg["left_leg"]
    hp_mount_xyz = _as_vec3(leg["hip_pitch_mount"]["xyz"])
    hp_mount_rpy = _as_vec3(leg["hip_pitch_mount"]["rpy"])

    mount_keys = [
        ("hip_pitch", "hip_roll", "hip_roll_mount"),
        ("hip_roll", "hip_thigh", "hip_thigh_mount"),
        ("hip_thigh", "hip_calf", "hip_calf_mount"),
        ("hip_calf", "ankle_pitch", "ankle_pitch_mount"),
        ("ankle_pitch", "ankle_roll", "ankle_roll_mount"),
    ]

    def _mirror_rpy(rpy_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        R = _R_from_rpy(*rpy_xyz)
        Rm = _mirror_R_yz(R)
        # extract rpy from Rm (simple numeric)
        r20 = max(-1.0, min(1.0, Rm[2][0]))
        pitch = math.asin(-r20)
        cp = math.cos(pitch)
        if abs(cp) < 1e-9:
            roll = 0.0
            yaw = math.atan2(-Rm[0][1], Rm[1][1])
        else:
            roll = math.atan2(Rm[2][1], Rm[2][2])
            yaw = math.atan2(Rm[1][0], Rm[0][0])
        return (roll, pitch, yaw)

    def add_leg(prefix: str, mirror: bool) -> None:
        hp_xyz = _mirror_xyz_yz(hp_mount_xyz) if mirror else hp_mount_xyz
        hp_rpy = _mirror_rpy(hp_mount_rpy) if mirror else hp_mount_rpy

        hp_mount = f"{prefix}_hip_pitch_mount"
        _add_frame_link(hp_mount)
        _urdf_add_joint(robot, f"base_to_{prefix}_hip_pitch_mount_fixed", "fixed", "base_motor_link", hp_mount, hp_xyz, hp_rpy)

        # hip_pitch motor
        hp_link = f"{prefix}_hip_pitch"
        hp = ET.SubElement(robot, "link", name=hp_link)
        _urdf_add_visual_mesh(hp, motor_stl, scale, motor_vis_rgba)
        _urdf_add_collision_box(hp, motor_size, xyz=motor_com)
        _urdf_add_inertial_box(hp, motor_mass, motor_size, com_xyz=motor_com)

        if _rot("hip_pitch"):
            _urdf_add_joint(robot, f"{prefix}_hip_pitch_joint", "revolute", hp_mount, hp_link, (0, 0, 0), (0, 0, 0), (0, 0, 1))
            out = hp_link
        else:
            _urdf_add_joint(robot, f"{prefix}_hip_pitch_mount_to_motor_fixed", "fixed", hp_mount, hp_link)
            out = hp_link

        # chain mounts
        parent_stage_out = out
        for parent_stage, child_stage, key in mount_keys:
            m = leg[key]
            j_xyz = _as_vec3(m["joint_origin_xyz"])
            mount_xyz = _as_vec3(m.get("mount_xyz", (0.0, 0.0, 0.0)))
            mount_rpy = _as_vec3(m.get("mount_rpy", (0.0, 0.0, 0.0)))
            if mirror:
                j_xyz = _mirror_xyz_yz(j_xyz)
                mount_xyz = _mirror_xyz_yz(mount_xyz)
                mount_rpy = _mirror_rpy(mount_rpy)

            child_link = f"{prefix}_{child_stage}"
            child = ET.SubElement(robot, "link", name=child_link)
            _urdf_add_visual_mesh(child, motor_stl, scale, motor_vis_rgba)
            _urdf_add_collision_box(child, motor_size, xyz=motor_com)
            _urdf_add_inertial_box(child, motor_mass, motor_size, com_xyz=motor_com)

            if _rot(child_stage):
                # joint first
                if use_joint_frames:
                    jf = f"{prefix}_{child_stage}_joint_frame"
                    _add_frame_link(jf)
                    _urdf_add_joint(robot, f"{prefix}_{child_stage}_joint", "revolute", parent_stage_out, jf, j_xyz, (0, 0, 0), (0, 0, 1))
                    _urdf_add_joint(robot, f"{prefix}_{child_stage}_joint_frame_to_motor_fixed", "fixed", jf, child_link, mount_xyz, mount_rpy)
                else:
                    _urdf_add_joint(robot, f"{prefix}_{child_stage}_joint", "revolute", parent_stage_out, child_link, (j_xyz[0] + mount_xyz[0], j_xyz[1] + mount_xyz[1], j_xyz[2] + mount_xyz[2]), mount_rpy, (0, 0, 1))
                parent_stage_out = child_link
            else:
                # motor fixed first
                _urdf_add_joint(robot, f"{prefix}_{child_stage}_mount_fixed", "fixed", parent_stage_out, child_link, (j_xyz[0] + mount_xyz[0], j_xyz[1] + mount_xyz[1], j_xyz[2] + mount_xyz[2]), mount_rpy)
                if use_joint_frames:
                    jf = f"{prefix}_{child_stage}_joint_frame"
                    _add_frame_link(jf)
                    _urdf_add_joint(robot, f"{prefix}_{child_stage}_joint", "revolute", child_link, jf, (0, 0, 0), (0, 0, 0), (0, 0, 1))
                    parent_stage_out = jf
                else:
                    print(f"[WARN] {prefix}_{child_stage}: joint-after-motor with use_joint_frames=false; motor will rotate with downstream joint in MJCF/URDF interpretation.", file=sys.stderr)
                    parent_stage_out = child_link

        # foot
        foot_cfg = leg["foot"]
        foot_link = f"{prefix}_foot"
        foot_l = ET.SubElement(robot, "link", name=foot_link)
        _urdf_add_visual_box(foot_l, foot_size, foot_rgba)
        _urdf_add_collision_box(foot_l, foot_size, xyz=(0.0, 0.0, 0.0))
        _urdf_add_inertial_box(foot_l, foot_mass, foot_size, com_xyz=(0.0, 0.0, 0.0))

        j_xyz = _as_vec3(foot_cfg.get("joint_origin_xyz", (0.0, 0.0, 0.0)))
        mount_xyz = _as_vec3(foot_cfg.get("mount_xyz", (0.0, 0.0, 0.0)))
        mount_rpy = _as_vec3(foot_cfg.get("mount_rpy", (0.0, 0.0, 0.0)))
        if mirror:
            j_xyz = _mirror_xyz_yz(j_xyz)
            mount_xyz = _mirror_xyz_yz(mount_xyz)
            mount_rpy = _mirror_rpy(mount_rpy)

        if _rot("ankle_roll"):
            _urdf_add_joint(robot, f"{prefix}_ankle_roll_to_{prefix}_foot_fixed", "fixed", parent_stage_out, foot_link, (j_xyz[0] + mount_xyz[0], j_xyz[1] + mount_xyz[1], j_xyz[2] + mount_xyz[2]), mount_rpy)
        else:
            _urdf_add_joint(robot, f"{prefix}_ankle_roll_joint", "revolute", parent_stage_out, foot_link, (j_xyz[0] + mount_xyz[0], j_xyz[1] + mount_xyz[1], j_xyz[2] + mount_xyz[2]), mount_rpy, (0, 0, 1))

    add_leg("l", mirror=False)
    add_leg("r", mirror=True)

    # trunk
    trunk_link = ET.SubElement(robot, "link", name="trunk_link")
    _urdf_add_visual_box(trunk_link, trunk_size, trunk_rgba)
    _urdf_add_collision_box(trunk_link, trunk_size, xyz=(0.0, 0.0, 0.0))
    _urdf_add_inertial_box(trunk_link, trunk_mass, trunk_size, com_xyz=(0.0, 0.0, 0.0))
    tj = cfg["trunk"]["joint"]
    _urdf_add_joint(
        robot,
        "base_to_trunk",
        "revolute",
        "base_motor_link",
        "trunk_link",
        _as_vec3(tj["origin_xyz"]),
        _as_vec3(tj.get("origin_rpy", (0.0, 0.0, 0.0))),
        _as_vec3(tj.get("axis_xyz", (0.0, 0.0, 1.0))),
    )

    _indent(robot)
    return ET.ElementTree(robot)


# --------------------------- MJCF helpers ---------------------------

def _mj_inertial(body: ET.Element, mass: float, size_xyz: tuple[float, float, float], com_xyz: tuple[float, float, float]) -> None:
    x, y, z = size_xyz
    ixx = (1.0 / 12.0) * mass * (y * y + z * z)
    iyy = (1.0 / 12.0) * mass * (x * x + z * z)
    izz = (1.0 / 12.0) * mass * (x * x + y * y)
    # IMPORTANT: keep high precision for inertia terms.
    # With low precision (e.g. 6 decimals), MuJoCo can see triangle-inequality violations due to rounding,
    # especially for small inertias (e.g. foot box), and refuse to load the model.
    diaginertia = f"{ixx:.12f} {iyy:.12f} {izz:.12f}"
    ET.SubElement(
        body,
        "inertial",
        pos=_xyz(*com_xyz),
        mass=f"{mass:.6f}",
        diaginertia=diaginertia,
    )


def _mj_geom_mesh(body: ET.Element, mesh_name: str, rgba: list[float]) -> None:
    ET.SubElement(
        body,
        "geom",
        type="mesh",
        mesh=mesh_name,
        rgba=" ".join(str(float(x)) for x in rgba),
        contype="0",
        conaffinity="0",
        group="0",
    )


def _mj_geom_box(body: ET.Element, size_xyz: tuple[float, float, float], pos_xyz: tuple[float, float, float], rgba: list[float], group: str) -> None:
    # MuJoCo box uses half-sizes.
    sx, sy, sz = (0.5 * size_xyz[0], 0.5 * size_xyz[1], 0.5 * size_xyz[2])
    ET.SubElement(
        body,
        "geom",
        type="box",
        size=_xyz(sx, sy, sz),
        pos=_xyz(*pos_xyz),
        rgba=" ".join(str(float(x)) for x in rgba),
        group=group,
    )


def build_mjcf(cfg: dict) -> ET.ElementTree:
    model_name = str(cfg.get("robot", {}).get("name", "phonebot"))
    mj = ET.Element("mujoco", model=model_name)
    # balanceinertia: helps keep inertia physically valid if any rounding/modeling issues slip through.
    ET.SubElement(mj, "compiler", angle="radian", balanceinertia="true")

    assets = cfg["assets"]
    comp = cfg["components"]
    motor_stl = str(Path(str(assets["motor_stl"])).resolve())
    scale = float(assets.get("motor_mesh_scale", 0.001))

    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "mesh", name="motor_mesh", file=motor_stl, scale=_xyz(scale, scale, scale))

    opts = cfg.get("options", {}) or {}
    floating_base = bool(opts.get("floating_base", True))
    use_joint_frames = bool(opts.get("use_joint_frames", True))
    rot_map = opts.get("motor_rotates_with_joint", {}) or {}

    def _rot(stage: str) -> bool:
        return bool(rot_map.get(stage, False))

    motor_c = comp["motor"]
    motor_size = _as_vec3(motor_c["size_xyz"])
    motor_com = _as_vec3(motor_c["com_offset_xyz"])
    motor_mass = float(motor_c["mass_kg"])
    motor_vis_rgba = list(motor_c.get("visual_rgba", [0.7, 0.7, 0.7, 1.0]))
    motor_col_rgba = list(motor_c.get("collision_rgba", [0.0, 0.8, 0.1, 1.0]))

    frame_c = comp.get("frame", {})
    frame_mass = float(frame_c.get("mass_kg", 0.001))
    frame_size = _as_vec3(frame_c.get("size_xyz", (0.01, 0.01, 0.01)))

    trunk_c = comp["trunk"]
    trunk_size = _as_vec3(trunk_c["size_xyz"])
    trunk_mass = float(trunk_c["mass_kg"])
    trunk_rgba = list(trunk_c.get("visual_rgba", [0.7, 0.7, 0.7, 1.0]))

    foot_c = comp["foot"]
    foot_size = _as_vec3(foot_c["size_xyz"])
    foot_mass = float(foot_c["mass_kg"])
    foot_rgba = list(foot_c.get("visual_rgba", [0.7, 0.7, 0.7, 1.0]))

    world = ET.SubElement(mj, "worldbody")

    # Base body
    base_body = ET.SubElement(world, "body", name="base_motor_link", pos=_xyz(0, 0, 0), quat=_quat_str((1.0, 0.0, 0.0, 0.0)))
    if floating_base:
        ET.SubElement(base_body, "freejoint", name="floating_base")

    _mj_inertial(base_body, motor_mass, motor_size, motor_com)
    _mj_geom_mesh(base_body, "motor_mesh", motor_vis_rgba)
    _mj_geom_box(base_body, motor_size, motor_com, motor_col_rgba, group="1")

    # Legs
    leg = cfg["left_leg"]
    hp_mount_xyz = _as_vec3(leg["hip_pitch_mount"]["xyz"])
    hp_mount_rpy = _as_vec3(leg["hip_pitch_mount"]["rpy"])

    mount_order = [
        ("hip_pitch", "hip_roll", "hip_roll_mount"),
        ("hip_roll", "hip_thigh", "hip_thigh_mount"),
        ("hip_thigh", "hip_calf", "hip_calf_mount"),
        ("hip_calf", "ankle_pitch", "ankle_pitch_mount"),
        ("ankle_pitch", "ankle_roll", "ankle_roll_mount"),
    ]

    def _mirror_quat_from_rpy(rpy_xyz: tuple[float, float, float]) -> tuple[float, float, float, float]:
        R = _R_from_rpy(*rpy_xyz)
        Rm = _mirror_R_yz(R)
        return _quat_from_R(Rm)

    def _quat_from_rpy(rpy_xyz: tuple[float, float, float]) -> tuple[float, float, float, float]:
        return _quat_from_R(_R_from_rpy(*rpy_xyz))

    def _axis_in_joint_frame_for_parent_z(origin_rpy: tuple[float, float, float]) -> tuple[float, float, float]:
        # axis in joint frame to represent parent's +Z if joint frame is rotated by origin_rpy
        R = _R_from_rpy(*origin_rpy)
        Rt = _transpose3(R)
        return _normalize3(_matvec3(Rt, (0.0, 0.0, 1.0)))

    def _add_frame_body(parent: ET.Element, name: str) -> ET.Element:
        b = ET.SubElement(parent, "body", name=name, pos=_xyz(0, 0, 0), quat=_quat_str((1.0, 0.0, 0.0, 0.0)))
        if use_joint_frames:
            _mj_inertial(b, frame_mass, frame_size, (0.0, 0.0, 0.0))
        return b

    def add_leg(prefix: str, mirror: bool) -> None:
        # Build a single list of stage boundaries so every motor stage is built the same way.
        # Each boundary provides:
        # - j_xyz: parent stage output -> joint location (parent frame)
        # - mount_xyz/rpy: joint frame -> motor frame (fixed)
        stage_bounds: list[tuple[str, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []
        # Base -> hip_pitch comes from `hip_pitch_mount` (same semantics as other mounts).
        stage_bounds.append(("hip_pitch", hp_mount_xyz, (0.0, 0.0, 0.0), hp_mount_rpy))
        # Remaining chain
        for _parent_stage, child_stage, key in mount_order:
            m = leg[key]
            stage_bounds.append(
                (
                    str(child_stage),
                    _as_vec3(m["joint_origin_xyz"]),
                    _as_vec3(m.get("mount_xyz", (0.0, 0.0, 0.0))),
                    _as_vec3(m.get("mount_rpy", (0.0, 0.0, 0.0))),
                )
            )

        stage_out_body = base_body

        for stage, j_xyz0, mount_xyz0, mount_rpy0 in stage_bounds:
            j_xyz = _mirror_xyz_yz(j_xyz0) if mirror else j_xyz0
            mount_xyz = _mirror_xyz_yz(mount_xyz0) if mirror else mount_xyz0

            # Rotation for mounting the motor body
            if mirror:
                R_mount = _mirror_R_yz(_R_from_rpy(*mount_rpy0))
            else:
                R_mount = _R_from_rpy(*mount_rpy0)
            mount_q = _quat_from_R(R_mount)

            Rt = _transpose3(R_mount)
            inv_t_vec = _matvec3(Rt, tuple(float(x) for x in mount_xyz))
            inv_mount = (-inv_t_vec[0], -inv_t_vec[1], -inv_t_vec[2])
            axis_local_for_parent_z = _normalize3(_matvec3(Rt, (0.0, 0.0, 1.0)))

            motor_name = f"{prefix}_{stage}"
            joint_name = f"{prefix}_{stage}_joint"

            if _rot(stage):
                # joint-first: parent -> joint -> motor
                if use_joint_frames:
                    joint_body = ET.SubElement(
                        stage_out_body,
                        "body",
                        name=f"{motor_name}_joint_frame",
                        pos=_xyz(*j_xyz),
                        quat=_quat_str((1.0, 0.0, 0.0, 0.0)),
                    )
                    _mj_inertial(joint_body, frame_mass, frame_size, (0.0, 0.0, 0.0))
                    ET.SubElement(
                        joint_body,
                        "joint",
                        name=joint_name,
                        type="hinge",
                        axis=_xyz(0, 0, 1),
                        range="-3.14159 3.14159",
                    )
                    motor_body = ET.SubElement(
                        joint_body,
                        "body",
                        name=motor_name,
                        pos=_xyz(*mount_xyz),
                        quat=_quat_str(mount_q),
                    )
                else:
                    # No joint-frame bodies: put joint on motor body.
                    # Choose axis and pos so the effective axis is parent +Z and the joint is at the boundary.
                    motor_pos = (j_xyz[0] + mount_xyz[0], j_xyz[1] + mount_xyz[1], j_xyz[2] + mount_xyz[2])
                    motor_body = ET.SubElement(
                        stage_out_body,
                        "body",
                        name=motor_name,
                        pos=_xyz(*motor_pos),
                        quat=_quat_str(mount_q),
                    )
                    ET.SubElement(
                        motor_body,
                        "joint",
                        name=joint_name,
                        type="hinge",
                        axis=_xyz(*axis_local_for_parent_z),
                        pos=_xyz(*inv_mount),
                        range="-3.14159 3.14159",
                    )

                _mj_inertial(motor_body, motor_mass, motor_size, motor_com)
                _mj_geom_mesh(motor_body, "motor_mesh", motor_vis_rgba)
                _mj_geom_box(motor_body, motor_size, motor_com, motor_col_rgba, group="1")
                stage_out_body = motor_body

            else:
                # joint-after-motor: parent -> motor (fixed) -> joint
                motor_pos = (j_xyz[0] + mount_xyz[0], j_xyz[1] + mount_xyz[1], j_xyz[2] + mount_xyz[2])
                motor_body = ET.SubElement(
                    stage_out_body,
                    "body",
                    name=motor_name,
                    pos=_xyz(*motor_pos),
                    quat=_quat_str(mount_q),
                )
                _mj_inertial(motor_body, motor_mass, motor_size, motor_com)
                _mj_geom_mesh(motor_body, "motor_mesh", motor_vis_rgba)
                _mj_geom_box(motor_body, motor_size, motor_com, motor_col_rgba, group="1")

                if use_joint_frames:
                    out_body = _add_frame_body(motor_body, f"{motor_name}_out")
                    ET.SubElement(
                        out_body,
                        "joint",
                        name=joint_name,
                        type="hinge",
                        axis=_xyz(*axis_local_for_parent_z),
                        pos=_xyz(*inv_mount),
                        range="-3.14159 3.14159",
                    )
                    stage_out_body = out_body
                else:
                    print(
                        f"[WARN] {motor_name}: motor_rotates_with_joint=false and use_joint_frames=false; "
                        "placing joint on motor body (simplified).",
                        file=sys.stderr,
                    )
                    ET.SubElement(
                        motor_body,
                        "joint",
                        name=joint_name,
                        type="hinge",
                        axis=_xyz(*axis_local_for_parent_z),
                        pos=_xyz(*inv_mount),
                        range="-3.14159 3.14159",
                    )
                    stage_out_body = motor_body

        # foot at ankle_roll output
        foot_cfg = leg["foot"]
        fj = _as_vec3(foot_cfg.get("joint_origin_xyz", (0.0, 0.0, 0.0)))
        fmount_xyz = _as_vec3(foot_cfg.get("mount_xyz", (0.0, 0.0, 0.0)))
        fmount_rpy = _as_vec3(foot_cfg.get("mount_rpy", (0.0, 0.0, 0.0)))
        if mirror:
            fj = _mirror_xyz_yz(fj)
            fmount_xyz = _mirror_xyz_yz(fmount_xyz)
            fmount_q = _quat_from_R(_mirror_R_yz(_R_from_rpy(*fmount_rpy)))
        else:
            fmount_q = _quat_from_rpy(fmount_rpy)

        fpos = (fj[0] + fmount_xyz[0], fj[1] + fmount_xyz[1], fj[2] + fmount_xyz[2])
        foot_body = ET.SubElement(stage_out_body, "body", name=f"{prefix}_foot", pos=_xyz(*fpos), quat=_quat_str(fmount_q))
        # Foot is an end-effector (fixed). Do NOT add any additional joints here.
        _mj_inertial(foot_body, foot_mass, foot_size, (0.0, 0.0, 0.0))
        _mj_geom_box(foot_body, foot_size, (0.0, 0.0, 0.0), foot_rgba, group="0")

    add_leg("l", mirror=False)
    add_leg("r", mirror=True)

    # Trunk branch (emit after legs so XML order is: left leg, right leg, trunk)
    tj = cfg["trunk"]["joint"]
    tpos = _as_vec3(tj["origin_xyz"])
    tR = _R_from_rpy(*_as_vec3(tj.get("origin_rpy", (0.0, 0.0, 0.0))))
    tquat = _quat_from_R(tR)
    trunk_body = ET.SubElement(base_body, "body", name="trunk_link", pos=_xyz(*tpos), quat=_quat_str(tquat))
    ET.SubElement(
        trunk_body,
        "joint",
        name="base_to_trunk",
        type="hinge",
        axis=_xyz(*_as_vec3(tj.get("axis_xyz", (0.0, 0.0, 1.0)))),
        range="-3.14159 3.14159",
    )
    _mj_inertial(trunk_body, trunk_mass, trunk_size, (0.0, 0.0, 0.0))
    _mj_geom_box(trunk_body, trunk_size, (0.0, 0.0, 0.0), trunk_rgba, group="0")

    _indent(mj)
    return ET.ElementTree(mj)


def _write_xml(path: Path, tree: ET.ElementTree, declaration: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=declaration)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate PhoneBot URDF + MJCF from YAML.")
    ap.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent / "model_config.yaml"))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--urdf-name", type=str, default="phonebot.urdf")
    ap.add_argument("--mjcf-name", type=str, default="phonebot.xml")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    out_dir = Path(args.out_dir).resolve()
    urdf_path = out_dir / args.urdf_name
    mjcf_path = out_dir / args.mjcf_name

    if not cfg_path.exists():
        print(f"[ERROR] config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = load_config(cfg_path)

    motor_stl = Path(str(cfg["assets"]["motor_stl"])).resolve()
    if not motor_stl.exists():
        print(f"[ERROR] motor STL not found: {motor_stl}", file=sys.stderr)
        return 2

    urdf_tree = build_urdf(cfg)
    _write_xml(urdf_path, urdf_tree, declaration=True)
    print(f"[OK] wrote URDF: {urdf_path}")

    mjcf_tree = build_mjcf(cfg)
    _write_xml(mjcf_path, mjcf_tree, declaration=False)
    print(f"[OK] wrote MJCF: {mjcf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())