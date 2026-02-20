#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from model_utils import (
    R_from_rpy,
    as_vec3,
    box_diaginertia,
    cylinder_diaginertia_z,
    fmt_inertia_diag,
    fmt_quat_wxyz,
    fmt_xyz,
    mirror_R_yz,
    mirror_xyz_yz,
    normalize3,
    quat_from_R,
    rpy_from_R,
    require_keys,
)


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


def load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


# ---------------- URDF helpers ----------------


def _urdf_add_inertial_box(link: ET.Element, mass: float, size_xyz: tuple[float, float, float], com_xyz: tuple[float, float, float]) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=fmt_xyz(*com_xyz), rpy=fmt_xyz(0, 0, 0))
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    ixx, iyy, izz = box_diaginertia(mass, size_xyz)
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{ixx:.12f}",
        ixy="0",
        ixz="0",
        iyy=f"{iyy:.12f}",
        iyz="0",
        izz=f"{izz:.12f}",
    )


def _urdf_add_inertial_cylinder_z(link: ET.Element, mass: float, radius: float, length: float, com_xyz: tuple[float, float, float]) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=fmt_xyz(*com_xyz), rpy=fmt_xyz(0, 0, 0))
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    ixx, iyy, izz = cylinder_diaginertia_z(mass, radius, length)
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{ixx:.12f}",
        ixy="0",
        ixz="0",
        iyy=f"{iyy:.12f}",
        iyz="0",
        izz=f"{izz:.12f}",
    )


def _urdf_add_visual_mesh(link: ET.Element, mesh_path: Path, scale: float, rgba: list[float]) -> None:
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", xyz=fmt_xyz(0, 0, 0), rpy=fmt_xyz(0, 0, 0))
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "mesh", filename=str(mesh_path), scale=fmt_xyz(scale, scale, scale))
    mat = ET.SubElement(vis, "material", name="motor_mat")
    ET.SubElement(mat, "color", rgba=" ".join(str(float(x)) for x in rgba))


def _urdf_add_visual_box(link: ET.Element, size_xyz: tuple[float, float, float], rgba: list[float], xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> None:
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", xyz=fmt_xyz(*xyz), rpy=fmt_xyz(*rpy))
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "box", size=fmt_xyz(*size_xyz))
    mat = ET.SubElement(vis, "material", name="box_mat")
    ET.SubElement(mat, "color", rgba=" ".join(str(float(x)) for x in rgba))


def _urdf_add_collision_box(link: ET.Element, size_xyz: tuple[float, float, float], xyz: tuple[float, float, float], rpy=(0.0, 0.0, 0.0)) -> None:
    col = ET.SubElement(link, "collision")
    ET.SubElement(col, "origin", xyz=fmt_xyz(*xyz), rpy=fmt_xyz(*rpy))
    geom = ET.SubElement(col, "geometry")
    ET.SubElement(geom, "box", size=fmt_xyz(*size_xyz))


def _urdf_add_visual_cylinder_z(link: ET.Element, radius: float, length: float, rgba: list[float], xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> None:
    vis = ET.SubElement(link, "visual")
    ET.SubElement(vis, "origin", xyz=fmt_xyz(*xyz), rpy=fmt_xyz(*rpy))
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "cylinder", radius=f"{radius:.6f}", length=f"{length:.6f}")
    mat = ET.SubElement(vis, "material", name="cylinder_mat")
    ET.SubElement(mat, "color", rgba=" ".join(str(float(x)) for x in rgba))


def _urdf_add_collision_cylinder_z(link: ET.Element, radius: float, length: float, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> None:
    col = ET.SubElement(link, "collision")
    ET.SubElement(col, "origin", xyz=fmt_xyz(*xyz), rpy=fmt_xyz(*rpy))
    geom = ET.SubElement(col, "geometry")
    ET.SubElement(geom, "cylinder", radius=f"{radius:.6f}", length=f"{length:.6f}")


def _urdf_add_joint(
    robot: ET.Element,
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    origin_xyz: tuple[float, float, float],
    origin_rpy: tuple[float, float, float],
    axis_xyz: tuple[float, float, float] | None = None,
) -> None:
    j = ET.SubElement(robot, "joint", name=name, type=joint_type)
    ET.SubElement(j, "origin", xyz=fmt_xyz(*origin_xyz), rpy=fmt_xyz(*origin_rpy))
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)
    if axis_xyz is not None:
        ET.SubElement(j, "axis", xyz=fmt_xyz(*axis_xyz))
    if joint_type == "revolute":
        ET.SubElement(j, "limit", lower="-3.14159", upper="3.14159", effort="5.0", velocity="8.0")


def build_urdf(cfg: dict, *, motor_mesh_path: Path) -> ET.ElementTree:
    require_keys(cfg, ["robot", "assets", "components", "left_leg", "trunk"], "root")
    robot = ET.Element("robot", name=str(cfg["robot"]["name"]))

    assets = cfg["assets"]
    comp = cfg["components"]
    motor_stl = motor_mesh_path
    scale = float(assets.get("motor_mesh_scale", 0.001))

    motor_c = comp["motor"]
    motor_mass = float(motor_c["mass_kg"])
    motor_size = as_vec3(motor_c["size_xyz"])
    motor_com = as_vec3(motor_c["com_offset_xyz"])
    motor_vis_rgba = list(motor_c.get("visual_rgba", [0.7, 0.7, 0.7, 1.0]))
    motor_axis_default = as_vec3(motor_c.get("joint_axis_xyz", (0.0, 0.0, 1.0)))

    frame_c = comp.get("frame", {})
    frame_mass = float(frame_c.get("mass_kg", 0.001))
    frame_size = as_vec3(frame_c.get("size_xyz", (0.01, 0.01, 0.01)))

    ee_c = comp["end_effector"]
    ee_mass = float(ee_c["mass_kg"])
    ee_rgba = list(ee_c.get("visual_rgba", [0.7, 0.7, 0.7, 1.0]))
    ee_com = as_vec3(ee_c.get("com_offset_xyz", (0.0, 0.0, 0.0)))
    ee_geom = as_vec3(ee_c.get("geom_offset_xyz", (0.0, 0.0, 0.0)))
    ee_geom_rpy = as_vec3(ee_c.get("geom_offset_rpy", (0.0, 0.0, 0.0)))
    ee_shape = ee_c["shape"]
    ee_type = str(ee_shape["type"]).lower()
    if ee_type == "box":
        ee_size = as_vec3(ee_shape["size_xyz"])
        ee_radius = None
        ee_length = None
    elif ee_type in ("cylinder", "capsule"):
        ee_radius = float(ee_shape["radius"])
        ee_length = float(ee_shape["length"])
        ee_size = None
    else:
        raise ValueError(f"Unsupported end_effector.shape.type: {ee_type!r}")

    use_joint_frames = bool((cfg.get("options", {}) or {}).get("use_joint_frames", True))

    def add_motor_chain(prefix: str, mirror: bool) -> None:
        leg = cfg["left_leg"]
        base_link = "base_motor_link"

        parent_out = base_link
        # Iterate motors list
        for mi, m in enumerate(leg.get("motors", []) or []):
            mname = str(m["name"])
            mxyz = as_vec3(m.get("xyz", (0.0, 0.0, 0.0)))
            mrpy = as_vec3(m.get("rpy", (0.0, 0.0, 0.0)))
            if mirror:
                mxyz = mirror_xyz_yz(mxyz)
                mrpy = rpy_from_R(mirror_R_yz(R_from_rpy(*mrpy)))

            rotates = bool(m.get("rotates_with_joint", True))
            axis = as_vec3(m.get("joint_axis_xyz", motor_axis_default))
            axis = normalize3(axis)

            motor_link = f"{prefix}_{mname}"
            motor_out = f"{prefix}_{mname}_out"

            # links
            ml = ET.SubElement(robot, "link", name=motor_link)
            _urdf_add_visual_mesh(ml, motor_stl, scale, motor_vis_rgba)
            _urdf_add_collision_box(ml, motor_size, xyz=motor_com)
            _urdf_add_inertial_box(ml, motor_mass, motor_size, com_xyz=motor_com)

            outl = ET.SubElement(robot, "link", name=motor_out)
            if use_joint_frames:
                _urdf_add_inertial_box(outl, frame_mass, frame_size, com_xyz=(0.0, 0.0, 0.0))

            joint_name = f"{prefix}_{mname}_joint"

            if rotates:
                # joint first at motor pose
                _urdf_add_joint(robot, joint_name, "revolute", parent_out, motor_link, mxyz, mrpy, axis)
                _urdf_add_joint(robot, f"{prefix}_{mname}_to_out_fixed", "fixed", motor_link, motor_out, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            else:
                # motor fixed at pose, joint after motor
                _urdf_add_joint(robot, f"{prefix}_{mname}_mount_fixed", "fixed", parent_out, motor_link, mxyz, mrpy)
                _urdf_add_joint(robot, joint_name, "revolute", motor_link, motor_out, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), axis)

            parent_out = motor_out

        # end effector
        ee_cfg = leg["end_effector"]
        eexyz = as_vec3(ee_cfg.get("xyz", (0.0, 0.0, 0.0)))
        eerpy = as_vec3(ee_cfg.get("rpy", (0.0, 0.0, 0.0)))
        if mirror:
            eexyz = mirror_xyz_yz(eexyz)
            eerpy = rpy_from_R(mirror_R_yz(R_from_rpy(*eerpy)))

        ee_link = f"{prefix}_end_effector"
        eel = ET.SubElement(robot, "link", name=ee_link)
        if ee_type == "box":
            assert ee_size is not None
            _urdf_add_visual_box(eel, ee_size, ee_rgba, xyz=ee_geom, rpy=ee_geom_rpy)
            _urdf_add_collision_box(eel, ee_size, xyz=ee_geom, rpy=ee_geom_rpy)
            _urdf_add_inertial_box(eel, ee_mass, ee_size, com_xyz=ee_com)
        else:
            r = float(ee_radius)
            L = float(ee_length)
            _urdf_add_visual_cylinder_z(eel, r, L, ee_rgba, xyz=ee_geom, rpy=ee_geom_rpy)
            _urdf_add_collision_cylinder_z(eel, r, L, xyz=ee_geom, rpy=ee_geom_rpy)
            _urdf_add_inertial_cylinder_z(eel, ee_mass, r, L, com_xyz=ee_com)
        _urdf_add_joint(robot, f"{prefix}_end_effector_fixed", "fixed", parent_out, ee_link, eexyz, eerpy)

    # base motor (root)
    base = ET.SubElement(robot, "link", name="base_motor_link")
    _urdf_add_visual_mesh(base, motor_stl, scale, motor_vis_rgba)
    _urdf_add_collision_box(base, motor_size, xyz=motor_com)
    _urdf_add_inertial_box(base, motor_mass, motor_size, com_xyz=motor_com)

    # legs: left then right
    add_motor_chain("l", mirror=False)
    add_motor_chain("r", mirror=True)

    # trunk last
    trunk_c = comp["trunk"]
    trunk_mass = float(trunk_c["mass_kg"])
    trunk_size = as_vec3(trunk_c["size_xyz"])
    trunk_rgba = list(trunk_c.get("visual_rgba", [0.7, 0.7, 0.7, 1.0]))
    trunk_com = as_vec3(trunk_c.get("com_offset_xyz", (0.0, 0.0, 0.0)))
    trunk_geom = as_vec3(trunk_c.get("geom_offset_xyz", (0.0, 0.0, 0.0)))
    trunk_geom_rpy = as_vec3(trunk_c.get("geom_offset_rpy", (0.0, 0.0, 0.0)))
    trunk_link = "trunk_link"
    tlink = ET.SubElement(robot, "link", name=trunk_link)
    _urdf_add_visual_box(tlink, trunk_size, trunk_rgba, xyz=trunk_geom, rpy=trunk_geom_rpy)
    _urdf_add_collision_box(tlink, trunk_size, xyz=trunk_geom, rpy=trunk_geom_rpy)
    _urdf_add_inertial_box(tlink, trunk_mass, trunk_size, com_xyz=trunk_com)

    tm = cfg["trunk"]["mount"]
    txyz = as_vec3(tm["xyz"])
    trpy = as_vec3(tm["rpy"])
    tjoint = cfg["trunk"]["joint"]
    _urdf_add_joint(robot, str(tjoint.get("name", "base_to_trunk")), "revolute", "base_motor_link", trunk_link, txyz, trpy, as_vec3(tjoint.get("axis_xyz", (0.0, 0.0, 1.0))))

    _indent(robot)
    return ET.ElementTree(robot)


# ---------------- MJCF helpers ----------------


def _mj_inertial_box(body: ET.Element, mass: float, size_xyz: tuple[float, float, float], com_xyz: tuple[float, float, float]) -> None:
    ixx, iyy, izz = box_diaginertia(mass, size_xyz)
    ET.SubElement(
        body,
        "inertial",
        pos=fmt_xyz(*com_xyz),
        mass=f"{mass:.6f}",
        diaginertia=fmt_inertia_diag(ixx, iyy, izz),
    )


def _mj_inertial_cylinder_z(body: ET.Element, mass: float, radius: float, length: float, com_xyz: tuple[float, float, float]) -> None:
    ixx, iyy, izz = cylinder_diaginertia_z(mass, radius, length)
    ET.SubElement(
        body,
        "inertial",
        pos=fmt_xyz(*com_xyz),
        mass=f"{mass:.6f}",
        diaginertia=fmt_inertia_diag(ixx, iyy, izz),
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
        size=fmt_xyz(sx, sy, sz),
        pos=fmt_xyz(*pos_xyz),
        rgba=" ".join(str(float(x)) for x in rgba),
        group=group,
    )


def build_mjcf(cfg: dict, *, motor_mesh_file: str) -> ET.ElementTree:
    require_keys(cfg, ["robot", "assets", "components", "left_leg", "trunk"], "root")
    model_name = str(cfg["robot"]["name"])
    mj = ET.Element("mujoco", model=model_name)
    ET.SubElement(mj, "compiler", angle="radian", balanceinertia="true")

    assets = cfg["assets"]
    comp = cfg["components"]
    motor_stl = motor_mesh_file
    scale = float(assets.get("motor_mesh_scale", 0.001))

    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "mesh", name="motor_mesh", file=motor_stl, scale=fmt_xyz(scale, scale, scale))

    opts = cfg.get("options", {}) or {}
    floating_base = bool(opts.get("floating_base", True))
    use_joint_frames = bool(opts.get("use_joint_frames", True))
    add_sensors = bool(opts.get("add_sensors", True))

    motor_c = comp["motor"]
    motor_mass = float(motor_c["mass_kg"])
    motor_size = as_vec3(motor_c["size_xyz"])
    motor_com = as_vec3(motor_c["com_offset_xyz"])
    motor_vis_rgba = list(motor_c.get("visual_rgba", [0.7, 0.7, 0.7, 1.0]))
    motor_col_rgba = list(motor_c.get("collision_rgba", [0.0, 0.8, 0.1, 1.0]))
    motor_axis_default = normalize3(as_vec3(motor_c.get("joint_axis_xyz", (0.0, 0.0, 1.0))))

    frame_c = comp.get("frame", {})
    frame_mass = float(frame_c.get("mass_kg", 0.001))
    frame_size = as_vec3(frame_c.get("size_xyz", (0.01, 0.01, 0.01)))

    ee_c = comp["end_effector"]
    ee_mass = float(ee_c["mass_kg"])
    ee_rgba = list(ee_c.get("visual_rgba", [0.7, 0.7, 0.7, 1.0]))
    ee_com = as_vec3(ee_c.get("com_offset_xyz", (0.0, 0.0, 0.0)))
    ee_geom = as_vec3(ee_c.get("geom_offset_xyz", (0.0, 0.0, 0.0)))
    ee_geom_rpy = as_vec3(ee_c.get("geom_offset_rpy", (0.0, 0.0, 0.0)))
    ee_shape = ee_c["shape"]
    ee_type = str(ee_shape["type"]).lower()
    if ee_type == "box":
        ee_size = as_vec3(ee_shape["size_xyz"])
        ee_radius = None
        ee_length = None
    elif ee_type in ("cylinder", "capsule"):
        ee_radius = float(ee_shape["radius"])
        ee_length = float(ee_shape["length"])
        ee_size = None
    else:
        raise ValueError(f"Unsupported end_effector.shape.type: {ee_type!r}")

    trunk_c = comp["trunk"]
    sensor_cfg = cfg.get("sensors", {}) or {}
    imu_site_cfg = sensor_cfg.get("imu_site", {}) or {}
    ee_site_cfg = sensor_cfg.get("end_effector_sites", {}) or {}
    imu_site_name = str(imu_site_cfg.get("name", "imu"))
    imu_site_xyz = as_vec3(imu_site_cfg.get("xyz", (0.0, 0.0, 0.0)))
    imu_site_rpy = as_vec3(imu_site_cfg.get("rpy", (0.0, 0.0, math.pi / 2.0)))
    left_ee_site_name = str(ee_site_cfg.get("left_name", "left_foot"))
    right_ee_site_name = str(ee_site_cfg.get("right_name", "right_foot"))
    trunk_mass = float(trunk_c["mass_kg"])
    trunk_size = as_vec3(trunk_c["size_xyz"])
    trunk_rgba = list(trunk_c.get("visual_rgba", [0.7, 0.7, 0.7, 1.0]))
    trunk_com = as_vec3(trunk_c.get("com_offset_xyz", (0.0, 0.0, 0.0)))
    trunk_geom = as_vec3(trunk_c.get("geom_offset_xyz", (0.0, 0.0, 0.0)))
    trunk_geom_rpy = as_vec3(trunk_c.get("geom_offset_rpy", (0.0, 0.0, 0.0)))

    def add_frame_body(parent: ET.Element, name: str) -> ET.Element:
        b = ET.SubElement(parent, "body", name=name, pos=fmt_xyz(0, 0, 0), quat=fmt_quat_wxyz((1.0, 0.0, 0.0, 0.0)))
        if use_joint_frames:
            _mj_inertial_box(b, frame_mass, frame_size, (0.0, 0.0, 0.0))
        return b

    def add_motor(parent_out: ET.Element, *, prefix: str, motor_cfg: dict, mirror: bool) -> ET.Element:
        """
        Add a motor stage, *motor-centric*:
        - The motor body pose (xyz/rpy) is always applied first relative to parent_out.
        - If rotates_with_joint=true: joint is on motor body (motor rotates with its own joint).
        - If rotates_with_joint=false: motor is fixed; joint is on an output frame body after the motor.
        This guarantees the motor pose is the same regardless of rotates_with_joint.
        """
        require_keys(motor_cfg, ["name"], "left_leg.motors[]")
        name = str(motor_cfg["name"])
        xyz = as_vec3(motor_cfg.get("xyz", (0.0, 0.0, 0.0)))
        rpy = as_vec3(motor_cfg.get("rpy", (0.0, 0.0, 0.0)))
        rotates = bool(motor_cfg.get("rotates_with_joint", True))
        axis = normalize3(as_vec3(motor_cfg.get("joint_axis_xyz", motor_axis_default)))

        if mirror:
            xyz = mirror_xyz_yz(xyz)
            Rm = mirror_R_yz(R_from_rpy(*rpy))
        else:
            Rm = R_from_rpy(*rpy)
        quat = quat_from_R(Rm)

        motor_body = ET.SubElement(parent_out, "body", name=f"{prefix}_{name}", pos=fmt_xyz(*xyz), quat=fmt_quat_wxyz(quat))
        if rotates:
            ET.SubElement(motor_body, "joint", name=f"{prefix}_{name}_joint", type="hinge", axis=fmt_xyz(*axis), pos=fmt_xyz(0, 0, 0), range="-3.14159 3.14159")

        _mj_inertial_box(motor_body, motor_mass, motor_size, motor_com)
        _mj_geom_mesh(motor_body, "motor_mesh", motor_vis_rgba)
        _mj_geom_box(motor_body, motor_size, motor_com, motor_col_rgba, group="1")

        out_body = add_frame_body(motor_body, f"{prefix}_{name}_out")
        if not rotates:
            ET.SubElement(out_body, "joint", name=f"{prefix}_{name}_joint", type="hinge", axis=fmt_xyz(*axis), pos=fmt_xyz(0, 0, 0), range="-3.14159 3.14159")
        return out_body

    def add_leg(prefix: str, mirror: bool) -> None:
        leg = cfg["left_leg"]
        require_keys(leg, ["motors", "end_effector"], "left_leg")
        parent_out = base_body

        for m in leg["motors"]:
            parent_out = add_motor(parent_out, prefix=prefix, motor_cfg=m, mirror=mirror)

        # End effector (fixed)
        ee = leg["end_effector"]
        eexyz = as_vec3(ee.get("xyz", (0.0, 0.0, 0.0)))
        eerpy = as_vec3(ee.get("rpy", (0.0, 0.0, 0.0)))
        if mirror:
            eexyz = mirror_xyz_yz(eexyz)
            Ree = mirror_R_yz(R_from_rpy(*eerpy))
        else:
            Ree = R_from_rpy(*eerpy)
        eq = quat_from_R(Ree)

        ee_body = ET.SubElement(parent_out, "body", name=f"{prefix}_end_effector", pos=fmt_xyz(*eexyz), quat=fmt_quat_wxyz(eq))
        if ee_type == "box":
            assert ee_size is not None
            _mj_inertial_box(ee_body, ee_mass, ee_size, ee_com)
            _mj_geom_box(ee_body, ee_size, ee_geom, ee_rgba, group="0")
        else:
            r = float(ee_radius)
            L = float(ee_length)
            _mj_inertial_cylinder_z(ee_body, ee_mass, r, L, ee_com)
            gq = quat_from_R(R_from_rpy(*ee_geom_rpy))
            ET.SubElement(
                ee_body,
                "geom",
                type="cylinder" if ee_type == "cylinder" else "capsule",
                size=f"{r:.6f} {0.5*L:.6f}",
                pos=fmt_xyz(*ee_geom),
                quat=fmt_quat_wxyz(gq),
                rgba=" ".join(str(float(x)) for x in ee_rgba),
                group="0",
            )

        # Site on end-effector for foot sensors (name matches HERMES conventions)
        if add_sensors:
            site_name = left_ee_site_name if prefix == "l" else right_ee_site_name
            # Site frame matches the end-effector body frame.
            ET.SubElement(ee_body, "site", name=site_name, pos=fmt_xyz(0.0, 0.0, 0.0), quat=fmt_quat_wxyz((1.0, 0.0, 0.0, 0.0)))

    world = ET.SubElement(mj, "worldbody")
    base_body = ET.SubElement(world, "body", name="base_motor_link", pos=fmt_xyz(0, 0, 0), quat=fmt_quat_wxyz((1.0, 0.0, 0.0, 0.0)))
    if floating_base:
        ET.SubElement(base_body, "freejoint", name="floating_base")

    # IMU site on base body (rotate around Z by +90deg so +X faces robot front)
    if add_sensors:
        imu_q = quat_from_R(R_from_rpy(*imu_site_rpy))
        ET.SubElement(base_body, "site", name=imu_site_name, pos=fmt_xyz(*imu_site_xyz), quat=fmt_quat_wxyz(imu_q))
    _mj_inertial_box(base_body, motor_mass, motor_size, motor_com)
    _mj_geom_mesh(base_body, "motor_mesh", motor_vis_rgba)
    _mj_geom_box(base_body, motor_size, motor_com, motor_col_rgba, group="1")

    # Branch order: left leg, right leg, trunk
    add_leg("l", mirror=False)
    add_leg("r", mirror=True)

    # Trunk (last)
    tm = cfg["trunk"]["mount"]
    txyz = as_vec3(tm.get("xyz", (0.0, 0.0, 0.0)))
    trpy = as_vec3(tm.get("rpy", (0.0, 0.0, 0.0)))
    tR = R_from_rpy(*trpy)
    tq = quat_from_R(tR)
    trunk_body = ET.SubElement(base_body, "body", name="trunk_link", pos=fmt_xyz(*txyz), quat=fmt_quat_wxyz(tq))
    tjoint = cfg["trunk"]["joint"]
    ET.SubElement(
        trunk_body,
        "joint",
        name=str(tjoint.get("name", "base_to_trunk")),
        type=str(tjoint.get("type", "hinge")),
        axis=fmt_xyz(*as_vec3(tjoint.get("axis_xyz", (0.0, 0.0, 1.0)))),
        range="-3.14159 3.14159",
    )
    _mj_inertial_box(trunk_body, trunk_mass, trunk_size, trunk_com)
    _mj_geom_box(trunk_body, trunk_size, trunk_geom, trunk_rgba, group="0")

    # Sensors (IMU + feet + per-joint)
    if add_sensors:
        sensor = ET.SubElement(mj, "sensor")

        # IMU-like sensors (pattern from HERMES mjcf)
        ET.SubElement(sensor, "framepos", name="position", objtype="site", objname=imu_site_name, noise="0.001")
        ET.SubElement(sensor, "velocimeter", name="linear-velocity", site=imu_site_name, noise="0.001", cutoff="30")
        ET.SubElement(sensor, "framequat", name="orientation", objtype="site", objname=imu_site_name, noise="0.001")
        ET.SubElement(sensor, "gyro", name="angular-velocity", site=imu_site_name, noise="0.005", cutoff="34.9")
        ET.SubElement(sensor, "accelerometer", name="linear-acceleration", site=imu_site_name, noise="0.005", cutoff="157")
        ET.SubElement(sensor, "magnetometer", name="magnetometer", site=imu_site_name)

        ET.SubElement(sensor, "gyro", site=imu_site_name, name="gyro")
        ET.SubElement(sensor, "velocimeter", site=imu_site_name, name="local_linvel")
        ET.SubElement(sensor, "accelerometer", site=imu_site_name, name="accelerometer")
        ET.SubElement(sensor, "framezaxis", objtype="site", objname=imu_site_name, name="upvector")
        ET.SubElement(sensor, "framexaxis", objtype="site", objname=imu_site_name, name="forwardvector")
        ET.SubElement(sensor, "framelinvel", objtype="site", objname=imu_site_name, name="global_linvel")
        ET.SubElement(sensor, "frameangvel", objtype="site", objname=imu_site_name, name="global_angvel")

        # Foot-based sensors
        ET.SubElement(sensor, "framelinvel", objtype="site", objname=left_ee_site_name, name="left_foot_global_linvel")
        ET.SubElement(sensor, "framelinvel", objtype="site", objname=right_ee_site_name, name="right_foot_global_linvel")
        ET.SubElement(sensor, "framepos", objtype="site", objname=left_ee_site_name, name="left_foot_pos")
        ET.SubElement(sensor, "framepos", objtype="site", objname=right_ee_site_name, name="right_foot_pos")

        # Joint sensors (pos/vel/actuator force)
        def _add_joint_sensors(jname: str) -> None:
            ET.SubElement(sensor, "jointpos", name=f"{jname}_pos", joint=jname)
            ET.SubElement(sensor, "jointvel", name=f"{jname}_vel", joint=jname)
            ET.SubElement(sensor, "jointactuatorfrc", name=f"{jname}_frc", joint=jname)

        for m in cfg["left_leg"]["motors"]:
            stage = str(m["name"])
            _add_joint_sensors(f"l_{stage}_joint")
            _add_joint_sensors(f"r_{stage}_joint")

        trunk_joint_name = str(cfg["trunk"]["joint"].get("name", "base_to_trunk"))
        _add_joint_sensors(trunk_joint_name)

    _indent(mj)
    return ET.ElementTree(mj)


def _write_xml(path: Path, tree: ET.ElementTree, declaration: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=declaration)


def main() -> int:
    ap = argparse.ArgumentParser(description="General model generator (URDF + MJCF) driven by per-motor YAML lists.")
    ap.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent / "model_config.yaml"))
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--urdf-name", type=str, default=None)
    ap.add_argument("--mjcf-name", type=str, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"[ERROR] config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = load_config(cfg_path)

    out_cfg = cfg.get("output", {}) or {}
    out_dir = Path(args.out_dir or out_cfg.get("out_dir") or DEFAULT_OUT_DIR).resolve()
    urdf_name = str(args.urdf_name or out_cfg.get("urdf_name") or "robot.urdf")
    mjcf_name = str(args.mjcf_name or out_cfg.get("mjcf_name") or "robot.xml")

    assets = cfg.get("assets", {}) or {}
    motor_stl_cfg = Path(str(assets.get("motor_stl", "")))
    if not motor_stl_cfg.is_absolute():
        motor_stl_abs = (cfg_path.parent / motor_stl_cfg).resolve()
    else:
        motor_stl_abs = motor_stl_cfg.resolve()
    if not motor_stl_abs.exists():
        print(f"[ERROR] motor STL not found: {motor_stl_abs}", file=sys.stderr)
        return 2

    use_relative_paths = bool(assets.get("use_relative_paths", False))
    copy_assets_to_output = bool(assets.get("copy_assets_to_output", use_relative_paths))
    output_mesh_dir = str(assets.get("output_mesh_dir", "meshes"))

    if use_relative_paths:
        if copy_assets_to_output:
            dest_dir = out_dir / output_mesh_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / motor_stl_abs.name
            shutil.copy2(motor_stl_abs, dest_path)
            motor_mesh_path_for_urdf = Path(output_mesh_dir) / motor_stl_abs.name
            motor_mesh_file_for_mjcf = str(Path(output_mesh_dir) / motor_stl_abs.name)
        else:
            rel = os.path.relpath(str(motor_stl_abs), str(out_dir))
            motor_mesh_path_for_urdf = Path(rel)
            motor_mesh_file_for_mjcf = rel
    else:
        motor_mesh_path_for_urdf = motor_stl_abs
        motor_mesh_file_for_mjcf = str(motor_stl_abs)

    urdf_tree = build_urdf(cfg, motor_mesh_path=motor_mesh_path_for_urdf)
    mjcf_tree = build_mjcf(cfg, motor_mesh_file=motor_mesh_file_for_mjcf)

    urdf_path = out_dir / urdf_name
    mjcf_path = out_dir / mjcf_name
    _write_xml(urdf_path, urdf_tree, declaration=True)
    _write_xml(mjcf_path, mjcf_tree, declaration=False)
    print(f"[OK] wrote URDF: {urdf_path}")
    print(f"[OK] wrote MJCF: {mjcf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


