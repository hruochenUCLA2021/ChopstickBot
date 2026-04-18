#!/usr/bin/env python3
"""
Inspect STL meshes and (optionally) render PNG previews.

This is for quick sanity checks on simplified outputs (triangle counts, bounds, watertightness)
and a basic visual look without opening a GUI mesh viewer.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RenderCfg:
    enabled: bool
    out_dir: Path
    max_plot_faces: int
    width: int
    height: int
    elev: float
    azim: float


def _iter_stl_files(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in items:
        p = Path(s).expanduser()
        if p.is_dir():
            for root, _dirs, files in os.walk(p):
                for fn in files:
                    if fn.lower().endswith(".stl"):
                        out.append((Path(root) / fn).resolve())
        else:
            out.append(p.resolve())
    # de-dupe
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    uniq.sort()
    return uniq


def _load_trimesh(path: Path):
    import trimesh

    mesh = trimesh.load_mesh(str(path), force="mesh")
    # Ensure we have a Trimesh (not Scene)
    if isinstance(mesh, trimesh.Scene):
        # Merge scene geometry if needed
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    return mesh


def _render_preview_png(mesh, out_png: Path, *, max_plot_faces: int, width: int, height: int, elev: float, azim: float) -> None:
    """
    Render using matplotlib (Agg). For large meshes, face-sample for preview only.
    """
    import numpy as np

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = mesh.vertices
    faces = mesh.faces
    if len(faces) > max_plot_faces:
        idx = np.random.default_rng(0).choice(len(faces), size=max_plot_faces, replace=False)
        faces = faces[idx]

    tris = vertices[faces]

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(tris, linewidths=0.05, alpha=1.0)
    poly.set_facecolor((0.75, 0.80, 0.90, 1.0))
    poly.set_edgecolor((0.15, 0.15, 0.15, 0.05))
    ax.add_collection3d(poly)

    # Fit axis bounds
    bounds = mesh.bounds
    mins = bounds[0]
    maxs = bounds[1]
    center = (mins + maxs) / 2.0
    extent = (maxs - mins)
    radius = float(max(extent) / 2.0) if len(extent) else 1.0
    if not math.isfinite(radius) or radius <= 0:
        radius = 1.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout(pad=0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("inspect_stl_config.yaml")),
        help="Path to inspect_stl_config.yaml",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")

    stl_paths = cfg.get("stl_paths", [])
    if not isinstance(stl_paths, list) or not all(isinstance(x, str) for x in stl_paths):
        raise ValueError("stl_paths must be a list of strings")

    inspect = cfg.get("inspect", {}) or {}
    verbose = bool(inspect.get("verbose", True))
    max_files = int(inspect.get("max_files", 0))

    render_cfg_raw = cfg.get("render", {}) or {}
    render_cfg = RenderCfg(
        enabled=bool(render_cfg_raw.get("enabled", False)),
        out_dir=Path(str(render_cfg_raw.get("out_dir", cfg_path.parent / "mesh_previews"))).expanduser().resolve(),
        max_plot_faces=int(render_cfg_raw.get("max_plot_faces", 50000)),
        width=int(render_cfg_raw.get("width", 900)),
        height=int(render_cfg_raw.get("height", 700)),
        elev=float(render_cfg_raw.get("elev", 20)),
        azim=float(render_cfg_raw.get("azim", 35)),
    )

    files = _iter_stl_files(stl_paths)
    if not files:
        print("[ERROR] No STL files found from stl_paths.")
        return 2

    if max_files > 0:
        files = files[:max_files]

    # Import here so missing dependency errors are clear and happen once.
    try:
        import trimesh  # noqa: F401
    except ModuleNotFoundError as e:
        print("[ERROR] Missing dependency: trimesh")
        print("Install: pip install trimesh numpy matplotlib")
        raise e

    print(f"[INFO] Inspecting {len(files)} STL file(s)")

    for p in files:
        if not p.exists():
            print(f"[MISSING] {p}")
            continue

        mesh = _load_trimesh(p)
        face_count = int(len(mesh.faces))
        vert_count = int(len(mesh.vertices))
        bounds = mesh.bounds
        extents = mesh.extents
        watertight = bool(getattr(mesh, "is_watertight", False))
        euler = getattr(mesh, "euler_number", None)

        if verbose:
            print("===")
            print(f"path: {p}")
            print(f"vertices: {vert_count}")
            print(f"faces: {face_count}")
            print(f"watertight: {watertight}")
            if euler is not None:
                print(f"euler_number: {euler}")
            print(f"bounds_min: {bounds[0].tolist()}")
            print(f"bounds_max: {bounds[1].tolist()}")
            print(f"extents: {extents.tolist()}")
        else:
            print(f"{face_count:>9d} faces  {vert_count:>9d} verts  watertight={watertight}  {p}")

        if render_cfg.enabled:
            out_png = render_cfg.out_dir / (p.stem + ".png")
            _render_preview_png(
                mesh,
                out_png,
                max_plot_faces=render_cfg.max_plot_faces,
                width=render_cfg.width,
                height=render_cfg.height,
                elev=render_cfg.elev,
                azim=render_cfg.azim,
            )
            if verbose:
                print(f"preview_png: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

