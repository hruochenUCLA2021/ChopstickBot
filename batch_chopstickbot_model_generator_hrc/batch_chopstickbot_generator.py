#!/usr/bin/env python3
"""
Batch Chopstickbot model generator (URDF + MJCF + scenes).

This script generates a sweep of Chopstickbot variants with different end-effector
bar lengths and physically-consistent mass/inertia computed from a constant density.

It reuses the existing generator:
  - `ChopstickBot/model_generator_hrc/model_generator.py`
  - `ChopstickBot/model_generator_hrc/model_utils.py`
  - `ChopstickBot/model_generator_hrc/model_config_chopstickbot.yaml`

Outputs are written into one subfolder per length.
"""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import sys
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping. Got: {type(data)}")
    return data


def _resolve_path(p: str, *, base_dir: Path) -> Path:
    pp = Path(str(p)).expanduser()
    return pp if pp.is_absolute() else (base_dir / pp).resolve()


def _frange_inclusive(a: float, b: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    if b < a:
        raise ValueError("max must be >= min")
    n = int(round((b - a) / step))
    out = [a + i * step for i in range(n + 1)]
    # guard rounding drift
    out = [float(f"{x:.10f}") for x in out if x <= b + 1e-9]
    return out


def _density_g_cm3_to_kg_m3(x: float) -> float:
    # 1 g/cm^3 = 1000 kg/m^3
    return float(x) * 1000.0


def _cylinder_mass_kg(*, density_kg_m3: float, radius_m: float, length_m: float) -> float:
    vol = math.pi * radius_m * radius_m * length_m
    return float(density_kg_m3) * float(vol)


def _safe_get(d: dict, path: str) -> dict:
    cur: object = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing key path: {path!r} at {k!r}")
        cur = cur[k]
    if not isinstance(cur, dict):
        raise TypeError(f"Expected dict at path {path!r}, got {type(cur)}")
    return cur


def _safe_get_any(d: dict, path: str) -> object:
    cur: object = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing key path: {path!r} at {k!r}")
        cur = cur[k]
    return cur


def _write_xml(path: Path, tree) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # keep formatting handled by the generator (it indents before returning)
    tree.write(path, encoding="utf-8", xml_declaration=path.suffix.lower() == ".urdf")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # The existing generator module uses non-package imports like `from model_utils import ...`,
    # so ensure that directory is importable as a top-level module path.
    gen_dir = repo_root / "ChopstickBot" / "model_generator_hrc"
    if gen_dir.is_dir() and str(gen_dir) not in sys.path:
        sys.path.insert(0, str(gen_dir))

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("batch_chopstickbot_generator_config.yaml")),
        help="Path to batch_chopstickbot_generator_config.yaml",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg_dir = cfg_path.parent
    cfg = _load_yaml(cfg_path)

    base_cfg_path = _resolve_path(str(_safe_get(cfg, "base").get("config_path")), base_dir=cfg_dir)
    base_cfg_dir = base_cfg_path.parent
    base_cfg = _load_yaml(base_cfg_path)

    out_base_dir = _resolve_path(str(_safe_get(cfg, "output").get("out_base_dir")), base_dir=cfg_dir)
    variant_dir_fmt = str(_safe_get(cfg, "output").get("variant_dir_fmt", "len_{length_m:.2f}m"))

    assets_cfg = cfg.get("assets", {}) or {}
    copy_scene_assets = bool(assets_cfg.get("copy_scene_assets", True))
    assets_src_dir = _resolve_path(str(assets_cfg.get("src_dir", "../model_generator_hrc/assets")), base_dir=cfg_dir)
    assets_dst_dir_name = str(assets_cfg.get("dst_dir_name", "assets"))
    assets_overwrite = bool(assets_cfg.get("overwrite", True))

    ee = _safe_get(cfg, "end_effector")
    radius_m = float(ee.get("radius_m", 0.005))
    length_min = float(ee.get("length_min_m", 0.1))
    length_max = float(ee.get("length_max_m", 5.0))
    length_step = float(ee.get("length_step_m", 0.1))
    density_g_cm3 = float(ee.get("density_g_cm3", 0.5))
    density_kg_m3 = _density_g_cm3_to_kg_m3(density_g_cm3)
    auto_xyz = bool(ee.get("auto_set_end_effector_xyz_to_minus_half_length", True))

    bp = cfg.get("base_pose", {}) or {}
    auto_base_z = bool(bp.get("auto_set_base_z_from_leg_length", True))
    extra_clearance_model = float(bp.get("extra_clearance_model_m", 0.0))
    extra_clearance_flat = float(bp.get("extra_clearance_flat_m", 0.0))
    extra_clearance_rough = float(bp.get("extra_clearance_rough_m", 0.0))
    length_ref_cfg = bp.get("length_ref_m", None)
    base_z_ref_cfg = bp.get("base_z_ref_m", None)

    # Infer reference length/base-z from the base YAML if not provided.
    length_ref_base = float(_safe_get_any(base_cfg, "components.end_effector.shape.length"))
    base_pose_xyz_base = _safe_get_any(base_cfg, "options.base_pose.xyz")
    if not (isinstance(base_pose_xyz_base, (list, tuple)) and len(base_pose_xyz_base) == 3):
        raise ValueError("base options.base_pose.xyz must be a 3-list in the base config")
    base_z_ref_base = float(base_pose_xyz_base[2])

    length_ref = float(length_ref_base if length_ref_cfg is None else length_ref_cfg)
    base_z_ref = float(base_z_ref_base if base_z_ref_cfg is None else base_z_ref_cfg)

    dry_run = bool((_safe_get(cfg, "run")).get("dry_run", False))

    lengths = _frange_inclusive(length_min, length_max, length_step)
    print(f"[INFO] base config: {base_cfg_path}")
    print(f"[INFO] out_base_dir: {out_base_dir}")
    print(f"[INFO] sweep lengths: {length_min}..{length_max} step {length_step} ({len(lengths)} variants)")
    print(f"[INFO] radius_m={radius_m} density={density_g_cm3} g/cm^3 ({density_kg_m3} kg/m^3)")
    print(f"[INFO] auto_set_end_effector_xyz_to_minus_half_length={auto_xyz}")
    print(
        "[INFO] auto_set_base_z_from_leg_length="
        f"{auto_base_z} (ref length={length_ref} m, ref base_z={base_z_ref} m, "
        f"extra_model={extra_clearance_model} m, extra_flat={extra_clearance_flat} m, extra_rough={extra_clearance_rough} m)"
    )
    print(
        "[INFO] copy_scene_assets="
        f"{copy_scene_assets} (src={assets_src_dir}, dst_dir_name={assets_dst_dir_name}, overwrite={assets_overwrite})"
    )
    print(f"[INFO] dry_run={dry_run}")

    # Import generator functions.
    from ChopstickBot.model_generator_hrc.model_generator import (  # noqa: E402
        build_mjcf,
        build_urdf,
        render_scene_flat_xml,
        render_scene_rough_xml,
    )

    for L in lengths:
        mass_kg = _cylinder_mass_kg(density_kg_m3=density_kg_m3, radius_m=radius_m, length_m=L)
        variant_dir = out_base_dir / variant_dir_fmt.format(length_m=L)

        # Clone base config and apply overrides.
        vcfg = copy.deepcopy(base_cfg)

        # Update end-effector geometry + mass.
        comp = _safe_get(vcfg, "components")
        ee_comp = _safe_get(comp, "end_effector")
        ee_shape = _safe_get(ee_comp, "shape")
        ee_comp["mass_kg"] = float(mass_kg)
        ee_shape["radius"] = float(radius_m)
        ee_shape["length"] = float(L)

        # Optionally update end-effector placement so the bar top stays at the joint.
        if auto_xyz:
            leg = _safe_get(vcfg, "left_leg")
            ee_mount = _safe_get(leg, "end_effector")
            ee_mount["xyz"] = [0.0, -0.5 * float(L), 0.0]

        # Optionally raise the base so longer bars do not start inside the floor.
        if auto_base_z:
            opts = vcfg.get("options", {}) or {}
            base_pose = opts.get("base_pose", {}) or {}
            xyz = base_pose.get("xyz", [0.0, 0.0, base_z_ref])
            if not (isinstance(xyz, (list, tuple)) and len(xyz) == 3):
                raise ValueError("options.base_pose.xyz must be a 3-list")
            xyz = [
                float(xyz[0]),
                float(xyz[1]),
                float(base_z_ref + (float(L) - float(length_ref)) + float(extra_clearance_model)),
            ]
            base_pose["xyz"] = xyz
            opts["base_pose"] = base_pose
            vcfg["options"] = opts

        # Update output folder.
        out_cfg = vcfg.get("output", {}) or {}
        out_cfg["out_dir"] = str(variant_dir)
        vcfg["output"] = out_cfg

        # Copy scene assets folder so scene references like `assets/hfield.png` work.
        if copy_scene_assets:
            if not assets_src_dir.exists():
                raise FileNotFoundError(f"assets.src_dir not found: {assets_src_dir}")
            dst_assets_dir = variant_dir / assets_dst_dir_name
            if dry_run:
                pass
            else:
                if dst_assets_dir.exists() and not assets_overwrite:
                    pass
                else:
                    # Copy entire folder; keep simple (assets are small).
                    shutil.copytree(assets_src_dir, dst_assets_dir, dirs_exist_ok=True)

        # Resolve motor STL similarly to the original generator.
        assets = vcfg.get("assets", {}) or {}
        motor_stl_cfg = Path(str(assets.get("motor_stl", "")))
        motor_stl_abs = (base_cfg_dir / motor_stl_cfg).resolve() if not motor_stl_cfg.is_absolute() else motor_stl_cfg.resolve()
        if not motor_stl_abs.exists():
            raise FileNotFoundError(f"motor STL not found: {motor_stl_abs}")

        use_relative_paths = bool(assets.get("use_relative_paths", False))
        copy_assets_to_output = bool(assets.get("copy_assets_to_output", use_relative_paths))
        output_mesh_dir = str(assets.get("output_mesh_dir", "meshes"))

        if use_relative_paths:
            if copy_assets_to_output:
                dest_dir = variant_dir / output_mesh_dir
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / motor_stl_abs.name
                if not dry_run:
                    shutil.copy2(motor_stl_abs, dest_path)
                motor_mesh_path_for_urdf = Path(output_mesh_dir) / motor_stl_abs.name
                motor_mesh_file_for_mjcf = str(Path(output_mesh_dir) / motor_stl_abs.name)
            else:
                rel = str(motor_stl_abs)
                motor_mesh_path_for_urdf = Path(rel)
                motor_mesh_file_for_mjcf = rel
        else:
            motor_mesh_path_for_urdf = motor_stl_abs
            motor_mesh_file_for_mjcf = str(motor_stl_abs)

        urdf_name = str(out_cfg.get("urdf_name", "chopstick_bot_general.urdf"))
        mjcf_name = str(out_cfg.get("mjcf_name", "chopstick_bot_general.xml"))
        scenes_cfg = vcfg.get("scenes", {}) or {}
        gen_scenes = bool(scenes_cfg.get("generate", True))
        scene_flat_name = str((scenes_cfg.get("flat", {}) or {}).get("filename", "scene_joystick_flat_terrain.xml"))
        scene_rough_name = str((scenes_cfg.get("rough", {}) or {}).get("filename", "scene_joystick_rough_terrain.xml"))

        base_z_model_now = (
            float(base_z_ref + (float(L) - float(length_ref)) + float(extra_clearance_model))
            if auto_base_z
            else base_z_ref_base
        )
        print(
            f"[VARIANT] L={L:.2f} m  mass={mass_kg:.6f} kg  base_z_model={base_z_model_now:.3f} m  "
            f"out={variant_dir}"
        )
        if dry_run:
            continue

        urdf_tree = build_urdf(vcfg, motor_mesh_path=motor_mesh_path_for_urdf)
        mjcf_tree = build_mjcf(vcfg, motor_mesh_file=motor_mesh_file_for_mjcf)
        _write_xml(variant_dir / urdf_name, urdf_tree)
        _write_xml(variant_dir / mjcf_name, mjcf_tree)

        if gen_scenes:
            # Scenes can require different initial clearances (flat vs rough/hfield).
            cfg_flat = copy.deepcopy(vcfg)
            cfg_rough = copy.deepcopy(vcfg)
            if auto_base_z:
                for c, extra in ((cfg_flat, extra_clearance_flat), (cfg_rough, extra_clearance_rough)):
                    opts = c.get("options", {}) or {}
                    bp2 = opts.get("base_pose", {}) or {}
                    xyz2 = bp2.get("xyz", [0.0, 0.0, base_z_model_now])
                    if not (isinstance(xyz2, (list, tuple)) and len(xyz2) == 3):
                        raise ValueError("options.base_pose.xyz must be a 3-list")
                    xyz2 = [float(xyz2[0]), float(xyz2[1]), float(base_z_model_now + float(extra))]
                    bp2["xyz"] = xyz2
                    opts["base_pose"] = bp2
                    c["options"] = opts

            flat_text = render_scene_flat_xml(cfg_flat, include_file=mjcf_name)
            rough_text = render_scene_rough_xml(cfg_rough, include_file=mjcf_name)
            _write_text(variant_dir / scene_flat_name, flat_text)
            _write_text(variant_dir / scene_rough_name, rough_text)

    print("[DONE] batch generation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

