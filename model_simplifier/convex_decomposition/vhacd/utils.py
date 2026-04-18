from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def convert_mesh_to_obj(in_path: Path, out_obj: Path) -> None:
    """
    Convert mesh to OBJ.

    Matches the previously working flow in `toolbox/v-hacd_good_one_use_this/.../convert_stl_to_obj.py`,
    but falls back to trimesh if Open3D isn't available.
    """
    out_obj.parent.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".obj":
        shutil.copy2(in_path, out_obj)
        return

    # Try Open3D (same as user's old script)
    try:
        import open3d as o3d  # type: ignore

        mesh = o3d.io.read_triangle_mesh(str(in_path))
        if mesh.is_empty():
            raise RuntimeError(f"Open3D loaded empty mesh: {in_path}")
        ok = o3d.io.write_triangle_mesh(str(out_obj), mesh)
        if not ok:
            raise RuntimeError(f"Open3D failed to write OBJ: {out_obj}")
        return
    except ModuleNotFoundError:
        pass

    # Fallback: trimesh
    try:
        import trimesh  # type: ignore

        mesh = trimesh.load_mesh(str(in_path), force="mesh")
        if getattr(mesh, "is_empty", False):
            raise RuntimeError(f"trimesh loaded empty mesh: {in_path}")
        mesh.export(str(out_obj))
        return
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Need either `open3d` (preferred) or `trimesh` installed to convert STL/PLY to OBJ for TestVHACD."
        ) from e


def run_testvhacd(
    *,
    testvhacd_exe: str,
    obj_path: Path,
    output_format: str,
    max_hulls: int | None,
    resolution: int | None,
    volume_error_percent: float | None = None,
    recursion_depth: int | None = None,
    shrinkwrap: bool | None = None,
    fill_mode: str | None = None,
    max_hull_verts: int | None = None,
    async_acd: bool | None = None,
    min_edge_length: float | None = None,
    optimal_split_plane: bool | None = None,
    show_logging: bool | None = None,
    cwd: Path,
) -> None:
    """
    Run the built TestVHACD executable.

    Expected interface (from TestVHACD.cpp):
    - first arg: <wavefront.obj>
    - options are pairs: -o stl|obj|usda, -h <n>, -r <voxelresolution>, ...
    """
    if output_format not in ("stl", "obj", "usda"):
        raise ValueError(f"Unsupported output_format: {output_format}")

    def _tf(v: bool) -> str:
        return "true" if v else "false"

    cmd = [str(testvhacd_exe), str(obj_path)]
    cmd += ["-o", output_format]

    # Add flags only if the config value is provided (not missing/null).
    if max_hulls is not None:
        cmd += ["-h", str(int(max_hulls))]
    if resolution is not None:
        cmd += ["-r", str(int(resolution))]
    if volume_error_percent is not None:
        cmd += ["-e", str(float(volume_error_percent))]
    if recursion_depth is not None:
        cmd += ["-d", str(int(recursion_depth))]
    if shrinkwrap is not None:
        cmd += ["-s", _tf(bool(shrinkwrap))]
    if fill_mode is not None:
        fm = str(fill_mode).lower()
        if fm not in ("flood", "surface", "raycast"):
            raise ValueError(f"fill_mode must be one of flood|surface|raycast, got {fill_mode!r}")
        cmd += ["-f", fm]
    if max_hull_verts is not None:
        cmd += ["-v", str(int(max_hull_verts))]
    if async_acd is not None:
        cmd += ["-a", _tf(bool(async_acd))]
    if min_edge_length is not None:
        cmd += ["-l", str(float(min_edge_length))]
    if optimal_split_plane is not None:
        cmd += ["-p", _tf(bool(optimal_split_plane))]
    if show_logging is not None:
        cmd += ["-g", _tf(bool(show_logging))]
    subprocess.check_call(cmd, cwd=str(cwd))


def pick_decomp_output(*, work_root: Path, objects_dir: Path, output_format: str) -> Path:
    """
    Locate the main output file produced by TestVHACD.

    For the stl workflow you used before, this is typically `work_root/decomp.stl`.
    """
    if output_format == "stl":
        p = work_root / "decomp.stl"
        if p.exists():
            return p
        # Fallback: look for "*_decompose.stl" in work_root
        cands = sorted(work_root.glob("*_decompose.stl"))
        if len(cands) == 1:
            return cands[0]
        # As a last resort, if exactly one STL exists in work_root, use it.
        stls = sorted(work_root.glob("*.stl"))
        if len(stls) == 1:
            return stls[0]
        raise FileNotFoundError(
            f"TestVHACD did not produce decomp.stl in {work_root}. "
            f"Found: work_root STLs={len(stls)}, decompose candidates={len(cands)}"
        )

    if output_format == "usda":
        p = work_root / "decomp.usda"
        if p.exists():
            return p
        cands = sorted(work_root.glob("*.usda"))
        if len(cands) == 1:
            return cands[0]
        raise FileNotFoundError(f"TestVHACD did not produce a USDA in {work_root}")

    # output_format == "obj"
    # TestVHACD emits per-hull OBJ files; use the objects_dir content.
    objs = sorted(objects_dir.glob("*.obj"))
    if objs:
        # Return the first hull OBJ. (If you need a packed scene, use CoACD runner instead.)
        return objs[0]
    raise FileNotFoundError(f"TestVHACD did not produce OBJ hulls under {objects_dir}")

