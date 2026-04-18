from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

from ...common import ensure_parent, iter_mesh_files, out_path_for, parse_io


from .utils import convert_mesh_to_obj, run_testvhacd, pick_decomp_output


def run(cfg: dict, cfg_path: Path) -> None:
    io = parse_io(cfg, cfg_path)
    tools = cfg.get("tools", {}) or {}
    vhacd_tool = tools.get("vhacd", {}) or {}
    vhacd_exe = str(vhacd_tool.get("exe") or "").strip()

    vcfg = (cfg.get("convex_decomposition", {}) or {}).get("vhacd", {}) or {}
    # If a key is missing or null, we do NOT pass that flag to TestVHACD (it will use its default).
    resolution = vcfg.get("resolution", None)
    resolution = None if resolution is None else int(resolution)
    max_hulls = vcfg.get("max_hulls", None)
    max_hulls = None if max_hulls is None else int(max_hulls)

    volume_error_percent = vcfg.get("volume_error_percent", None)
    volume_error_percent = None if volume_error_percent is None else float(volume_error_percent)
    recursion_depth = vcfg.get("recursion_depth", None)
    recursion_depth = None if recursion_depth is None else int(recursion_depth)
    shrinkwrap = vcfg.get("shrinkwrap", None)
    shrinkwrap = None if shrinkwrap is None else bool(shrinkwrap)
    fill_mode = vcfg.get("fill_mode", None)
    fill_mode = None if fill_mode is None else str(fill_mode)
    max_hull_verts = vcfg.get("max_hull_verts", None)
    max_hull_verts = None if max_hull_verts is None else int(max_hull_verts)
    async_acd = vcfg.get("async_acd", None)
    async_acd = None if async_acd is None else bool(async_acd)
    min_edge_length = vcfg.get("min_edge_length", None)
    min_edge_length = None if min_edge_length is None else float(min_edge_length)
    optimal_split_plane = vcfg.get("optimal_split_plane", None)
    optimal_split_plane = None if optimal_split_plane is None else bool(optimal_split_plane)
    show_logging = vcfg.get("show_logging", None)
    show_logging = None if show_logging is None else bool(show_logging)

    if not io.input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {io.input_dir}")
    io.output_dir.mkdir(parents=True, exist_ok=True)

    meshes = iter_mesh_files(io.input_dir, io.exts)
    if not meshes:
        print(f"[WARN] no meshes found under: {io.input_dir}")
        return

    if not vhacd_exe:
        raise RuntimeError(
            "VHACD method selected but tools.vhacd.exe is empty. "
            "Build/locate TestVHACD and set tools.vhacd.exe to its path."
        )

    print(f"[INFO] convex_decomposition.vhacd: {len(meshes)} mesh files")
    for src in meshes:
        dst = out_path_for(src, io)
        ensure_parent(dst)
        if dst.exists() and not io.overwrite:
            print(f"[SKIP] {dst}")
            continue

        # Match the previously-working folder layout:
        # - objects/: contains input OBJ and any per-hull STL outputs (cleaned after)
        # - outputs/: final renamed output
        work_root = io.output_dir / ".vhacd_work" / src.stem
        objects_dir = work_root / "objects"
        outputs_dir = work_root / "outputs"
        if work_root.exists():
            shutil.rmtree(work_root)
        objects_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Convert to OBJ into objects/ (or copy if already OBJ)
        obj_path = objects_dir / f"{src.stem}.obj"
        convert_mesh_to_obj(src, obj_path)

        # Run TestVHACD in work_root so decomp.stl lands there.
        # Use output format based on desired output extension.
        out_ext = dst.suffix.lower().lstrip(".")
        out_fmt = "stl" if out_ext in ("stl", "") else out_ext
        if out_fmt not in ("stl", "obj", "usda"):
            out_fmt = "stl"

        print(f"[RUN] TestVHACD ... {src.name} -> {dst.name} (format={out_fmt})")
        run_testvhacd(
            testvhacd_exe=vhacd_exe,
            obj_path=obj_path,
            output_format=out_fmt,
            max_hulls=max_hulls,
            resolution=resolution,
            volume_error_percent=volume_error_percent,
            recursion_depth=recursion_depth,
            shrinkwrap=shrinkwrap,
            fill_mode=fill_mode,
            max_hull_verts=max_hull_verts,
            async_acd=async_acd,
            min_edge_length=min_edge_length,
            optimal_split_plane=optimal_split_plane,
            show_logging=show_logging,
            cwd=work_root,
        )

        produced = pick_decomp_output(work_root=work_root, objects_dir=objects_dir, output_format=out_fmt)

        # Move produced result to final dst path.
        shutil.move(str(produced), str(dst))

        # Clean per-hull outputs under objects/ (matches old bash script intent).
        for p in objects_dir.glob("*.stl"):
            try:
                p.unlink()
            except Exception:
                pass

        print(f"[OK] wrote: {dst}")

