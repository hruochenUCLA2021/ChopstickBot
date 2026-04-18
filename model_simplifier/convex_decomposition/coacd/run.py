from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ...common import ensure_parent, iter_mesh_files, out_path_for, parse_io


def run(cfg: dict, cfg_path: Path) -> None:
    """
    Runs CoACD via the python package `coacd`.

    Output:
    - Writes a convex-decomposed mesh per input file.
      (Depending on CoACD version, output may be a single mesh containing multiple parts.)
    """
    io = parse_io(cfg, cfg_path)
    c = (cfg.get("convex_decomposition", {}) or {}).get("coacd", {}) or {}
    max_hulls = int(c.get("max_convex_hulls", 32))
    threshold = float(c.get("threshold", 0.05))

    if not io.input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {io.input_dir}")
    io.output_dir.mkdir(parents=True, exist_ok=True)

    meshes = iter_mesh_files(io.input_dir, io.exts)
    if not meshes:
        print(f"[WARN] no meshes found under: {io.input_dir}")
        return

    # We run in-process, but keep logic in a separate module so dependency errors are clearer.
    from .utils import decompose_one

    print(f"[INFO] convex_decomposition.coacd: {len(meshes)} mesh files")
    for src in meshes:
        dst = out_path_for(src, io)
        ensure_parent(dst)
        if dst.exists() and not io.overwrite:
            print(f"[SKIP] {dst}")
            continue
        print(f"[RUN] coacd ... {src.name} -> {dst.name}")
        decompose_one(src, dst, max_hulls=max_hulls, threshold=threshold)

