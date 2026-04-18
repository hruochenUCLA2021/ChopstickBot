"""
Blender helper script (run inside Blender) to decimate a mesh file and export as STL/OBJ/PLY.

Called by: decimation/blender/run.py
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
    # Newer Blender versions often have built-in IO operators (no add-on needed).
    # If an add-on isn't shipped in this Blender bundle, enabling it just spams logs.
    if module not in getattr(bpy.context.preferences, "addons", {}):
        return
    bpy.ops.preferences.addon_enable(module=module)


def _import_stl(path: Path) -> None:
    # Blender operator names vary by version.
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
    # ensure evaluated
    me.calc_loop_triangles()
    return len(me.loop_triangles)


def main() -> None:
    # Blender passes its own args; parse after "--"
    argv = []
    if "--" in os.sys.argv:
        argv = os.sys.argv[os.sys.argv.index("--") + 1 :]

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--target_faces", type=int, required=True)
    args = ap.parse_args(argv)

    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()
    target_faces = int(args.target_faces)

    _reset_scene()
    _import_mesh(in_path)
    obj = _join_all_mesh_objects()

    # Apply a decimate modifier aiming for target_faces triangles.
    before = _tri_count(obj)
    ratio = 1.0
    if before > 0 and target_faces > 0 and before > target_faces:
        ratio = max(0.0001, min(1.0, target_faces / float(before)))

    mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    mod.use_collapse_triangulate = True

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    after = _tri_count(obj)
    print(f"[OK] decimate: {in_path.name}: {before} -> {after} tris (target={target_faces})")

    _export_mesh(out_path)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)

