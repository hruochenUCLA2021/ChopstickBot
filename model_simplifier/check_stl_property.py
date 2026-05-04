#!/usr/bin/env python3
"""
Print basic mesh complexity stats for STL/OBJ/PLY files.

This is useful for checking:
- vertex / face counts
- whether a mesh is watertight
- how many disconnected components the mesh contains

Config:
- See `check_stl_property_config.yaml`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    return cfg


def _format_bool(x: bool) -> str:
    return "yes" if x else "no"


def main() -> int:
    # Make imports work whether executed as script or module.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("check_stl_property_config.yaml")),
        help="Path to check_stl_property_config.yaml",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)

    from ChopstickBot.model_simplifier.common import iter_mesh_files, parse_io  # local import

    io = parse_io(cfg, cfg_path)
    check = cfg.get("check", {}) or {}
    show_component_sizes = bool(check.get("show_component_sizes", True))
    topk = int(check.get("component_sizes_topk", 5))

    if not io.input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {io.input_dir}")

    meshes = iter_mesh_files(io.input_dir, io.exts)
    print(f"[INFO] scanning: {io.input_dir}")
    print(f"[INFO] extensions: {list(io.exts)}")
    print(f"[INFO] found: {len(meshes)} mesh files")
    if not meshes:
        return 0

    try:
        import trimesh
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "trimesh is required for this tool. Install it in your current env, e.g. `pip install trimesh`."
        ) from e

    for p in meshes:
        m = trimesh.load_mesh(str(p), force="mesh")
        comps = m.split(only_watertight=False)
        comp_faces = sorted((len(c.faces) for c in comps), reverse=True)

        rel = p.relative_to(io.input_dir)
        print("-" * 80)
        print(f"file: {rel.as_posix()}")
        print(f"  vertices:   {len(m.vertices)}")
        print(f"  faces:      {len(m.faces)}")
        print(f"  components: {len(comps)}")
        print(f"  watertight: {_format_bool(bool(m.is_watertight))}")
        if show_component_sizes:
            print(f"  component_face_counts(top{topk}): {comp_faces[:topk]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

