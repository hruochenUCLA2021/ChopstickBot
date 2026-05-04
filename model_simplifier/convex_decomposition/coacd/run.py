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
    # Optional CoACD knobs; if None/null we keep current runner behavior and/or
    # let CoACD fall back to its internal defaults.
    merge = c.get("merge", None)
    decimate = c.get("decimate", None)
    max_ch_vertex = c.get("max_ch_vertex", None)
    preprocess_resolution = c.get("preprocess_resolution", None)
    seed = c.get("seed", None)
    approximate_mode = c.get("approximate_mode", None)

    if merge is not None:
        merge = bool(merge)
    if decimate is not None:
        decimate = bool(decimate)
    if max_ch_vertex is not None:
        max_ch_vertex = int(max_ch_vertex)
    if preprocess_resolution is not None:
        preprocess_resolution = int(preprocess_resolution)
    if seed is not None:
        seed = int(seed)
    if approximate_mode is not None:
        approximate_mode = str(approximate_mode)

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
        decompose_one(
            src,
            dst,
            max_hulls=max_hulls,
            threshold=threshold,
            merge=merge,
            decimate=decimate,
            max_ch_vertex=max_ch_vertex,
            preprocess_resolution=preprocess_resolution,
            seed=seed,
            approximate_mode=approximate_mode,
        )

