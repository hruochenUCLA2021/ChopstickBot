from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ...common import ensure_parent, iter_mesh_files, out_path_for, parse_io


def run(cfg: dict, cfg_path: Path) -> None:
    io = parse_io(cfg, cfg_path)
    tool = (cfg.get("tools", {}) or {}).get("blender", {}) or {}
    blender_exe = str(tool.get("exe") or "blender")
    target_faces = int((cfg.get("decimation", {}) or {}).get("target_faces", 150000))

    script_py = Path(__file__).with_name("blender_decimate.py").resolve()
    if not io.input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {io.input_dir}")
    io.output_dir.mkdir(parents=True, exist_ok=True)

    meshes = iter_mesh_files(io.input_dir, io.exts)
    if not meshes:
        print(f"[WARN] no meshes found under: {io.input_dir}")
        return

    print(f"[INFO] decimation.blender: {len(meshes)} mesh files")
    for src in meshes:
        dst = out_path_for(src, io)
        ensure_parent(dst)
        if dst.exists() and not io.overwrite:
            print(f"[SKIP] {dst}")
            continue
        cmd = [
            blender_exe,
            "--background",
            "--factory-startup",
            "--python",
            str(script_py),
            "--",
            "--in",
            str(src),
            "--out",
            str(dst),
            "--target_faces",
            str(target_faces),
        ]
        print(f"[RUN] {' '.join(cmd[:4])} ... {src.name} -> {dst.name}")
        subprocess.check_call(cmd)

