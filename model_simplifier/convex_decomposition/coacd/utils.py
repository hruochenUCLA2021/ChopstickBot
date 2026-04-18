from __future__ import annotations

from pathlib import Path


def decompose_one(in_path: Path, out_path: Path, *, max_hulls: int, threshold: float) -> None:
    """
    Decompose a mesh into convex parts using CoACD.

    Notes:
    - This uses `trimesh` to load/export.
    - CoACD returns multiple convex parts; we export them as a Scene (if possible) or merge.
    """
    import numpy as np
    import trimesh

    try:
        import coacd
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "coacd is not installed. Install it in your `model_convert` env (see Note_install_model_simplification_dependency.md)."
        ) from e

    mesh = trimesh.load_mesh(str(in_path), force="mesh")
    if mesh.is_empty:
        raise ValueError(f"empty mesh: {in_path}")

    # CoACD expects vertices/faces numpy arrays.
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)

    # CoACD python API differs by version.
    # - Some versions expose `coacd.coacd(vertices, faces, ...)`
    # - v1.0.10 exposes `coacd.run_coacd(mesh, ...)`
    parts = None
    if hasattr(coacd, "coacd"):
        parts = coacd.coacd(  # type: ignore[attr-defined]
            v,
            f,
            threshold=threshold,
            max_convex_hulls=max_hulls,
        )
    elif hasattr(coacd, "run_coacd"):
        # Build CoACD Mesh wrapper.
        mesh_in = coacd.Mesh(v, f)  # type: ignore[attr-defined]
        parts = coacd.run_coacd(  # type: ignore[attr-defined]
            mesh_in,
            threshold=float(threshold),
            max_convex_hull=int(max_hulls) if int(max_hulls) > 0 else -1,
            merge=False,  # keep parts separated so we can export multiple hulls
        )
    else:
        raise AttributeError(
            "Unsupported coacd API. Expected `coacd.coacd` or `coacd.run_coacd` to exist."
        )

    # Normalize output to list[(vertices, faces)]
    part_meshes = []
    if parts is None:
        raise RuntimeError("coacd returned no parts")

    # coacd.run_coacd may return a list of Mesh-like objects
    # or a list of (v, f). Handle both.
    for part in parts:
        if isinstance(part, (tuple, list)) and len(part) == 2:
            pv, pf = part
        elif hasattr(part, "vertices") and hasattr(part, "faces"):
            pv = part.vertices  # type: ignore[attr-defined]
            pf = part.faces  # type: ignore[attr-defined]
        else:
            raise TypeError(f"Unrecognized coacd part type: {type(part)}")
        pm = trimesh.Trimesh(vertices=pv, faces=pf, process=False)
        part_meshes.append(pm)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export strategy:
    # - If output is .obj/.ply: export a scene of parts if supported.
    # - If output is .stl: STL doesn't support multiple objects; we merge parts into one mesh.
    ext = out_path.suffix.lower()
    if ext == ".stl":
        merged = trimesh.util.concatenate(part_meshes)
        merged.export(str(out_path))
        return

    scene = trimesh.Scene()
    for i, pm in enumerate(part_meshes):
        scene.add_geometry(pm, node_name=f"part_{i}")
    scene.export(str(out_path))

