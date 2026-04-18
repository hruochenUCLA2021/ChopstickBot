from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ...common import ensure_parent, iter_mesh_files, out_path_for, parse_io


def run(cfg: dict, cfg_path: Path) -> None:
    io = parse_io(cfg, cfg_path)
    tool = (cfg.get("tools", {}) or {}).get("meshlab", {}) or {}
    meshlab_exe = str(tool.get("exe") or "meshlabserver")

    rs = cfg.get("remesh_simplify", {}) or {}
    target_faces = int(rs.get("target_faces", 150000))

    mlx_template = Path(__file__).with_name("filters_remesh_and_decimate.mlx").resolve()
    rendered_dir = io.output_dir / ".meshlab_filters"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    mlx = rendered_dir / f"remesh_decimate_{target_faces}.mlx"
    txt = mlx_template.read_text(encoding="utf-8")
    txt = txt.replace("__TARGET_FACE_NUM__", str(target_faces))
    mlx.write_text(txt, encoding="utf-8")
    if not io.input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {io.input_dir}")
    io.output_dir.mkdir(parents=True, exist_ok=True)

    meshes = iter_mesh_files(io.input_dir, io.exts)
    if not meshes:
        print(f"[WARN] no meshes found under: {io.input_dir}")
        return

    print(f"[INFO] remesh_simplify.meshlab: {len(meshes)} mesh files")
    for src in meshes:
        dst = out_path_for(src, io)
        ensure_parent(dst)
        if dst.exists() and not io.overwrite:
            print(f"[SKIP] {dst}")
            continue
        cmd = [
            meshlab_exe,
            "-i",
            str(src),
            "-o",
            str(dst),
            "-m",
            "vn",
            "fn",
            "-s",
            str(mlx),
        ]
        print(f"[RUN] meshlabserver remesh+decimate ... {src.name} -> {dst.name}")
        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        subprocess.check_call(cmd, env=env)

