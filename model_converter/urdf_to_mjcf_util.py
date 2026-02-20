#!/usr/bin/env python3
"""
URDF -> MJCF conversion utility for this repo.

Features:
- Reads a YAML config (urdf_to_mjcf_config.yaml)
- Preprocesses URDF to rewrite ROS `package://...` mesh URIs to relative filesystem paths
  (so urdf2mjcf can copy meshes and MuJoCo can load them)
- Converts using `urdf2mjcf.convert.convert_urdf_to_mjcf`
- Postprocesses MJCF for MuJoCo schema (geom-scale -> asset-mesh scale) and optional colors
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


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


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    return cfg


def resolve_package_uri(uri: str, package_map: dict[str, str], relative_to_dir: Path) -> str:
    """
    Convert `package://PKG/some/path` into a relative path from urdf_dir
    pointing to (package_map[PKG] / some/path).
    """
    if not uri.startswith("package://"):
        return uri
    rest = uri[len("package://") :]
    if "/" not in rest:
        raise ValueError(f"Invalid package URI: {uri}")
    pkg, rel = rest.split("/", 1)
    if pkg not in package_map:
        raise KeyError(f"Package {pkg!r} not found in package_map")
    pkg_root = Path(package_map[pkg]).resolve()
    abs_path = (pkg_root / rel).resolve()
    # Make relative to the URDF file directory (so urdf2mjcf copy_meshes works)
    rel_path = os.path.relpath(str(abs_path), start=str(relative_to_dir.resolve()))
    return rel_path


def preprocess_urdf_package_meshes(in_urdf: Path, out_urdf: Path, package_map: dict[str, str]) -> None:
    tree = ET.parse(in_urdf)
    root = tree.getroot()
    # IMPORTANT: mesh filenames must be correct relative to the *location of the rewritten URDF*,
    # because urdf2mjcf resolves meshes relative to urdf_path.parent when copy_meshes=True.
    relative_to_dir = out_urdf.parent.resolve()

    changed = 0
    for mesh in root.findall(".//mesh"):
        fname = mesh.attrib.get("filename")
        if not fname:
            continue
        if fname.startswith("package://"):
            mesh.attrib["filename"] = resolve_package_uri(fname, package_map, relative_to_dir)
            changed += 1

    _indent(root)
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)
    print(f"[OK] wrote resolved URDF: {out_urdf} (rewrote {changed} package:// mesh URIs)")


def write_local_urdf_with_meshdir(resolved_urdf: Path, local_urdf: Path, meshdir: str = "meshes") -> bool:
    """
    Create a URDF that references meshes from a local folder (e.g. out/meshes/<file>).
    This is just for convenience / portability; conversion uses the resolved URDF.
    """
    try:
        tree = ET.parse(resolved_urdf)
        root = tree.getroot()
    except Exception:
        return False

    changed = 0
    for mesh in root.findall(".//mesh"):
        fname = mesh.attrib.get("filename")
        if not fname:
            continue
        base = Path(fname).name
        mesh.attrib["filename"] = str(Path(meshdir) / base)
        changed += 1

    _indent(root)
    local_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(local_urdf, encoding="utf-8", xml_declaration=True)
    print(f"[OK] wrote local URDF: {local_urdf} (rewrote {changed} mesh filenames to {meshdir}/...)")
    return True


def postprocess_mjcf_use_local_copied_meshes(mjcf_path: Path, meshdir: str = "meshes") -> bool:
    """
    If meshes have been copied to <out_dir>/meshes, make MJCF reference them.
    - sets/updates <compiler meshdir="meshes/">
    - rewrites each <asset><mesh file="..."> to file="<basename>"
    """
    try:
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
    except Exception:
        return False

    asset = root.find("asset")
    if not isinstance(asset, ET.Element):
        return False

    changed = False

    # Ensure compiler has meshdir
    compiler = root.find("compiler")
    if compiler is None or not isinstance(compiler, ET.Element):
        compiler = ET.Element("compiler", attrib={"angle": "radian"})
        root.insert(0, compiler)
        changed = True
    want = meshdir.rstrip("/") + "/"
    if compiler.attrib.get("meshdir") != want:
        compiler.attrib["meshdir"] = want
        changed = True

    # Rewrite asset mesh files to basenames (resolved under meshdir)
    for mesh in asset.findall("mesh"):
        f = mesh.attrib.get("file")
        if not f:
            continue
        base = Path(f).name
        if mesh.attrib["file"] != base:
            mesh.attrib["file"] = base
            changed = True

    if changed:
        _indent(root)
        tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)
    return changed

def postprocess_mjcf_remove_geom_scale(mjcf_path: Path) -> bool:
    """
    MuJoCo 3.x schema rejects `scale` on <geom>. Move it to <asset><mesh>.
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
        if mesh_elem is not None and "scale" not in mesh_elem.attrib:
            mesh_elem.set("scale", scale)
            changed = True

        # Always remove geom scale (schema violation)
        del geom.attrib["scale"]
        changed = True

    if changed:
        _indent(root)
        tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)
    return changed


def postprocess_mjcf_fix_collision_colors(mjcf_path: Path) -> bool:
    """Make collision material not bright red (optional preference)."""
    try:
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
    except Exception:
        return False

    changed = False
    grey = "0.89804 0.91765 0.92941 1.0"
    for material in root.findall(".//material[@name='collision_material']"):
        if material.get("rgba") != grey:
            material.set("rgba", grey)
            changed = True

    if changed:
        _indent(root)
        tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)
    return changed


def postprocess_mjcf_fix_empty_material_names(mjcf_path: Path) -> bool:
    """
    MuJoCo requires non-empty names. Some URDF exporters emit <material name=""> in URDF,
    which becomes <material name=""> in MJCF and also geoms with material="".
    Fix by renaming the empty material to a valid name and updating references.

    We use the name `visualgeom` because urdf2mjcf's default visual class references it.
    """
    try:
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
    except Exception:
        return False

    asset = root.find("asset")
    if not isinstance(asset, ET.Element):
        return False

    changed = False

    # Ensure there's exactly one material named visualgeom
    has_visualgeom = asset.find("./material[@name='visualgeom']") is not None

    for mat in list(asset.findall("material")):
        if mat.attrib.get("name") == "":
            if has_visualgeom:
                # Drop the empty material; we'll re-point geoms to visualgeom
                asset.remove(mat)
                changed = True
            else:
                mat.attrib["name"] = "visualgeom"
                has_visualgeom = True
                changed = True

    # Update geom references
    for geom in root.findall(".//geom"):
        if geom.attrib.get("material") == "":
            geom.attrib["material"] = "visualgeom"
            changed = True

    if changed:
        _indent(root)
        tree.write(mjcf_path, encoding="utf-8", xml_declaration=False)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "urdf_to_mjcf_config.yaml"),
        help="Path to urdf_to_mjcf_config.yaml",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    in_urdf = Path(str(cfg["input_urdf"])).resolve()
    out_dir = Path(str(cfg["output"]["out_dir"])).resolve()
    mjcf_path = out_dir / str(cfg["output"]["mjcf_name"])
    resolved_urdf_path = out_dir / str(cfg["output"]["resolved_urdf_name"])
    local_urdf_path = out_dir / str(cfg["output"].get("local_urdf_name", "local.urdf"))

    if not in_urdf.exists():
        print(f"[ERROR] input URDF not found: {in_urdf}", file=sys.stderr)
        return 2

    package_map = cfg.get("package_map", {}) or {}
    if not isinstance(package_map, dict):
        raise ValueError("package_map must be a mapping")

    preprocess_urdf_package_meshes(in_urdf, resolved_urdf_path, {str(k): str(v) for k, v in package_map.items()})

    # Convert
    try:
        from urdf2mjcf.convert import convert_urdf_to_mjcf
    except ModuleNotFoundError as e:
        print("[ERROR] urdf2mjcf not importable in this python environment.", file=sys.stderr)
        print("        Activate your venv (RL_env_3_12) or install urdf2mjcf.", file=sys.stderr)
        raise e

    copy_meshes = bool(cfg.get("convert", {}).get("copy_meshes", True))
    out_dir.mkdir(parents=True, exist_ok=True)

    convert_urdf_to_mjcf(
        urdf_path=str(resolved_urdf_path),
        mjcf_path=str(mjcf_path),
        copy_meshes=copy_meshes,
    )
    print(f"[OK] wrote MJCF: {mjcf_path}")

    # Postprocess
    pp = cfg.get("postprocess", {}) or {}
    if postprocess_mjcf_fix_empty_material_names(mjcf_path):
        print("[OK] postprocess: fixed empty material names (material=\"\" -> visualgeom)")
    if bool(pp.get("fix_geom_mesh_scale", True)):
        if postprocess_mjcf_remove_geom_scale(mjcf_path):
            print("[OK] postprocess: moved geom scale -> asset mesh scale")
    if bool(pp.get("fix_collision_colors_to_grey", False)):
        if postprocess_mjcf_fix_collision_colors(mjcf_path):
            print("[OK] postprocess: adjusted collision material colors")
    if copy_meshes and bool(pp.get("use_local_copied_meshes", True)):
        if postprocess_mjcf_use_local_copied_meshes(mjcf_path, meshdir="meshes"):
            print("[OK] postprocess: rewrote MJCF to use local copied meshes (compiler meshdir=meshes/)")
        # Also emit a local URDF for convenience
        write_local_urdf_with_meshdir(resolved_urdf_path, local_urdf_path, meshdir="meshes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


