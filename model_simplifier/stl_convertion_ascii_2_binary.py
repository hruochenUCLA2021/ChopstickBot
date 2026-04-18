#!/usr/bin/env python3
"""
Batch convert ASCII STL -> binary STL.

Why:
- VHACD `TestVHACD -o stl` typically outputs ASCII STL ("solid ...").
- MuJoCo often errors on ASCII STL and prefers binary STL.

This script:
- detects STL encoding (binary vs ascii)
- loads mesh (Open3D preferred; falls back to trimesh; then numpy-stl)
- writes binary STL
"""

from __future__ import annotations

import argparse
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class IOConfig:
    input_dir: Path
    output_dir: Path
    in_place: bool
    recursive: bool
    overwrite: bool


def _stl_is_binary(path: Path) -> bool:
    """
    Determine if STL is binary by checking:
    - file size matches 84 + 50 * triangle_count (triangle_count is uint32 at byte offset 80)
    """
    try:
        size = path.stat().st_size
        if size < 84:
            return False
        with path.open("rb") as f:
            f.read(80)
            tri_bytes = f.read(4)
        if len(tri_bytes) != 4:
            return False
        tri = struct.unpack("<I", tri_bytes)[0]
        expected = 84 + 50 * tri
        return expected == size
    except Exception:
        return False


def _stl_looks_ascii(path: Path) -> bool:
    # Heuristic: ASCII STL usually starts with "solid" and contains "facet normal".
    try:
        with path.open("rb") as f:
            head = f.read(4096)
        return head.lstrip().lower().startswith(b"solid") and (b"facet normal" in head.lower())
    except Exception:
        return False


def _iter_stls(input_dir: Path, recursive: bool) -> list[Path]:
    out: list[Path] = []
    if recursive:
        for root, _dirs, files in os.walk(input_dir):
            for fn in files:
                fn_l = fn.lower()
                # Skip temp files created by this script (or prior runs).
                if ".tmp." in fn_l:
                    continue
                if fn_l.endswith(".stl"):
                    out.append((Path(root) / fn).resolve())
    else:
        for p in input_dir.iterdir():
            name_l = p.name.lower()
            if ".tmp." in name_l:
                continue
            if p.is_file() and name_l.endswith(".stl"):
                out.append(p.resolve())
    out.sort()
    return out


def _write_binary_stl_open3d(in_path: Path, out_path: Path) -> bool:
    try:
        import open3d as o3d  # type: ignore
    except ModuleNotFoundError:
        return False

    mesh = o3d.io.read_triangle_mesh(str(in_path))
    # Some files may be empty/invalid.
    if getattr(mesh, "is_empty", lambda: False)():
        raise RuntimeError(f"Open3D loaded empty mesh: {in_path}")

    # Open3D STL writer requires normals on some builds.
    try:
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
    except Exception:
        pass

    # Open3D's STL writer generally writes binary; prefer forcing write_ascii=False if supported.
    try:
        ok = o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False)
    except TypeError:
        ok = o3d.io.write_triangle_mesh(str(out_path), mesh)
    if not ok:
        raise RuntimeError(f"Open3D failed to write STL: {out_path}")
    return True


def _write_binary_stl_trimesh(in_path: Path, out_path: Path) -> bool:
    try:
        import trimesh  # type: ignore
    except ModuleNotFoundError:
        return False

    mesh = trimesh.load_mesh(str(in_path), force="mesh")
    if getattr(mesh, "is_empty", False):
        raise RuntimeError(f"trimesh loaded empty mesh: {in_path}")

    data = mesh.export(file_type="stl")
    # trimesh may return bytes or str depending on exporter/settings; ensure bytes.
    if isinstance(data, str):
        data_b = data.encode("utf-8")
    else:
        data_b = data
    out_path.write_bytes(data_b)
    return True


def _write_binary_stl_numpy_stl(in_path: Path, out_path: Path) -> bool:
    try:
        from stl import Mode, mesh  # type: ignore
    except ModuleNotFoundError:
        return False

    m = mesh.Mesh.from_file(str(in_path))
    m.save(str(out_path), mode=Mode.BINARY)
    return True


def _convert_one(in_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try backends in order. If a backend is installed but fails for this mesh,
    # fall back to the next one.
    tried_any = False
    try:
        tried_any = True
        if _write_binary_stl_open3d(in_path, out_path):
            return
    except Exception:
        pass
    try:
        tried_any = True
        if _write_binary_stl_trimesh(in_path, out_path):
            return
    except Exception:
        pass
    try:
        tried_any = True
        if _write_binary_stl_numpy_stl(in_path, out_path):
            return
    except Exception:
        pass

    raise ModuleNotFoundError(
        "No conversion backend available. Install one of:\n"
        "- open3d\n"
        "- trimesh\n"
        "- numpy-stl (pip package name: numpy-stl; import name: stl)\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("stl_convertion_ascii_2_binary_config.yaml")),
        help="Path to stl_convertion_ascii_2_binary_config.yaml",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")

    io_raw = cfg.get("io", {}) or {}
    conv_raw = cfg.get("convert", {}) or {}
    io = IOConfig(
        input_dir=Path(str(io_raw.get("input_dir", ""))).expanduser().resolve(),
        output_dir=Path(str(io_raw.get("output_dir", ""))).expanduser().resolve(),
        in_place=bool(io_raw.get("in_place", True)),
        recursive=bool(io_raw.get("recursive", True)),
        overwrite=bool(io_raw.get("overwrite", True)),
    )
    skip_if_binary = bool(conv_raw.get("skip_if_binary", True))

    if not io.input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {io.input_dir}")

    stls = _iter_stls(io.input_dir, io.recursive)
    if not stls:
        print(f"[WARN] no STL files found under: {io.input_dir}")
        return 0

    converted = 0
    skipped = 0
    failed = 0

    for src in stls:
        if skip_if_binary and _stl_is_binary(src):
            skipped += 1
            continue

        # If it doesn't look ascii but also doesn't match binary, we still try to rewrite it.
        out_path = src if io.in_place else (io.output_dir / src.relative_to(io.input_dir))
        if out_path.exists() and not io.overwrite and not io.in_place:
            skipped += 1
            continue

        # Write to temp then replace to avoid corrupting on partial write.
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=str(out_path.parent),
                prefix=out_path.stem + ".tmp.",
                suffix=out_path.suffix,
                delete=False,
            ) as tf:
                tmp = Path(tf.name)

            try:
                _convert_one(src, tmp)

                # Verify binary output
                if not _stl_is_binary(tmp):
                    # If Open3D wrote ASCII by default (rare), this will catch it.
                    raise RuntimeError(f"Converted file is not binary STL: {tmp}")

                tmp.replace(out_path)
            finally:
                # If anything failed before replace(), clean up temp file.
                if tmp.exists() and tmp != out_path:
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
            converted += 1
            print(f"[OK] binary STL: {src} -> {out_path}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] failed: {src} ({type(e).__name__}: {e})")

    print(f"[DONE] total={len(stls)} converted={converted} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

