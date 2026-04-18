"""
Blender helper script (run inside Blender) to voxel-remesh then decimate a mesh file.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import bpy


def _reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)

def _enable_addon(module: str) -> None:
    if module not in getattr(bpy.context.preferences, "addons", {}):
        return
    bpy.ops.preferences.addon_enable(module=module)


def _import_stl(path: Path) -> None:
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=str(path))
        return
    if hasattr(bpy.ops.import_mesh, "stl"):
        bpy.ops.import_mesh.stl(filepath=str(path))
        return
    raise AttributeError("No STL import operator found (wm.stl_import / import_mesh.stl).")


def _export_stl(path: Path) -> None:
    if hasattr(bpy.ops.wm, "stl_export"):
        bpy.ops.wm.stl_export(filepath=str(path), ascii_format=False)
        return
    if hasattr(bpy.ops.export_mesh, "stl"):
        bpy.ops.export_mesh.stl(filepath=str(path), ascii=False)
        return
    raise AttributeError("No STL export operator found (wm.stl_export / export_mesh.stl).")


def _import_mesh(path: Path) -> None:
    ext = path.suffix.lower()
    if ext == ".stl":
        _enable_addon("io_mesh_stl")
        _import_stl(path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=str(path))
    else:
        raise ValueError(f"Unsupported input extension: {ext}")


def _export_mesh(path: Path) -> None:
    ext = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if ext == ".stl":
        _enable_addon("io_mesh_stl")
        _export_stl(path)
    elif ext == ".obj":
        bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=False)
    elif ext == ".ply":
        bpy.ops.export_mesh.ply(filepath=str(path), ascii_format=False)
    else:
        raise ValueError(f"Unsupported output extension: {ext}")


def _join_all_mesh_objects() -> bpy.types.Object:
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects imported")
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def _tri_count(obj: bpy.types.Object) -> int:
    me = obj.data
    me.calc_loop_triangles()
    return len(me.loop_triangles)


def main() -> None:
    argv = []
    if "--" in os.sys.argv:
        argv = os.sys.argv[os.sys.argv.index("--") + 1 :]

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--voxel_size", type=float, required=True)
    ap.add_argument("--target_faces", type=int, required=True)
    args = ap.parse_args(argv)

    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()
    voxel_size = float(args.voxel_size)
    target_faces = int(args.target_faces)

    _reset_scene()
    _import_mesh(in_path)
    obj = _join_all_mesh_objects()

    before = _tri_count(obj)

    # Voxel remesh
    rem = obj.modifiers.new(name="Remesh", type="REMESH")
    rem.mode = "VOXEL"
    rem.voxel_size = voxel_size
    rem.use_remove_disconnected = False
    rem.use_smooth_shade = False

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=rem.name)
    after_remesh = _tri_count(obj)

    # Decimate to target faces
    ratio = 1.0
    if after_remesh > 0 and target_faces > 0 and after_remesh > target_faces:
        ratio = max(0.0001, min(1.0, target_faces / float(after_remesh)))
    dec = obj.modifiers.new(name="Decimate", type="DECIMATE")
    dec.decimate_type = "COLLAPSE"
    dec.ratio = ratio
    dec.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=dec.name)
    after = _tri_count(obj)

    print(
        f"[OK] remesh+decimate: {in_path.name}: {before} -> {after_remesh} -> {after} tris "
        f"(voxel_size={voxel_size}, target={target_faces})"
    )
    _export_mesh(out_path)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)

