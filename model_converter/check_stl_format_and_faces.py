#!/usr/bin/env python3
"""
Check STL format (ASCII vs binary) and triangle count.

Why:
- MuJoCo can error if STL is ASCII (or malformed) or if face count exceeds limits.
- This tool reports format + triangles so we can diagnose issues quickly.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class StlReport:
    path: Path
    exists: bool
    size_bytes: int | None
    fmt: str  # "binary" | "ascii" | "unknown" | "error"
    triangles: int | None
    expected_binary_size: int | None
    notes: list[str]


def _count_subseq_in_file(path: Path, needle: bytes, chunk_size: int = 1024 * 1024) -> int:
    # Count occurrences of a byte subsequence in a file, streaming.
    # Handles matches across chunk boundaries.
    if not needle:
        return 0
    count = 0
    tail = b""
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            data = tail + chunk
            count += data.count(needle)
            tail = data[-(len(needle) - 1) :] if len(needle) > 1 else b""
    return count


def analyze_stl(path: Path) -> StlReport:
    notes: list[str] = []
    if not path.exists():
        return StlReport(
            path=path,
            exists=False,
            size_bytes=None,
            fmt="error",
            triangles=None,
            expected_binary_size=None,
            notes=["file not found"],
        )

    size = path.stat().st_size
    if size < 84:
        return StlReport(
            path=path,
            exists=True,
            size_bytes=size,
            fmt="error",
            triangles=None,
            expected_binary_size=None,
            notes=[f"file too small for STL header (size={size})"],
        )

    with path.open("rb") as f:
        header = f.read(80)
        tri_bytes = f.read(4)
    tri = struct.unpack("<I", tri_bytes)[0]
    expected_size = 84 + 50 * tri

    # Strong binary check: exact file-size match to binary STL layout.
    if expected_size == size:
        fmt = "binary"
        triangles = tri
        if header[:5].lower() == b"solid":
            notes.append("header starts with 'solid' (binary STL can still do this)")
        return StlReport(
            path=path,
            exists=True,
            size_bytes=size,
            fmt=fmt,
            triangles=triangles,
            expected_binary_size=expected_size,
            notes=notes,
        )

    # Not an exact binary match. It could be ASCII STL or a malformed/extra-bytes binary STL.
    # Attempt ASCII detection and triangle count by scanning for "facet normal".
    # (Binary STL may contain this in the 80-byte header, but it won't appear thousands of times.)
    facet_token = b"facet normal"
    facet_count = _count_subseq_in_file(path, facet_token)
    if facet_count > 0:
        fmt = "ascii"
        triangles = facet_count
        notes.append(f"binary size mismatch: expected {expected_size}, got {size}")
        return StlReport(
            path=path,
            exists=True,
            size_bytes=size,
            fmt=fmt,
            triangles=triangles,
            expected_binary_size=expected_size,
            notes=notes,
        )

    # Unknown/malformed: doesn't look like ASCII and doesn't match binary layout.
    fmt = "unknown"
    triangles = tri  # still report the uint32 at offset 80 (useful clue)
    notes.append(f"binary size mismatch: expected {expected_size}, got {size}")
    if header[:5].lower() == b"solid":
        notes.append("starts with 'solid' but no 'facet normal' tokens found (maybe binary header or non-STL)")

    return StlReport(
        path=path,
        exists=True,
        size_bytes=size,
        fmt=fmt,
        triangles=triangles,
        expected_binary_size=expected_size,
        notes=notes,
    )


def _expand_paths(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in items:
        p = Path(s).expanduser()
        if p.is_dir():
            out.extend(sorted(p.glob("*.stl")))
            out.extend(sorted(p.glob("*.STL")))
        else:
            out.append(p)
    # de-dupe preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("check_stl_format_and_faces_config.yaml")),
        help="Path to check_stl_format_and_faces_config.yaml",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")

    stl_paths = cfg.get("stl_paths", [])
    if not isinstance(stl_paths, list) or not all(isinstance(x, str) for x in stl_paths):
        raise ValueError("stl_paths must be a list of strings")
    max_faces = int(cfg.get("max_faces", 200000))
    compact = bool(cfg.get("compact", False))

    paths = _expand_paths(stl_paths)
    if not paths:
        print("[ERROR] No STL paths provided (empty stl_paths and no directory matches).")
        return 2

    worst_over = 0
    any_error = False

    for p in paths:
        r = analyze_stl(p)
        if r.fmt == "error":
            any_error = True

        over = 0
        if r.triangles is not None:
            over = max(0, r.triangles - max_faces)
            worst_over = max(worst_over, over)

        if compact:
            tri_s = "?" if r.triangles is None else str(r.triangles)
            size_s = "?" if r.size_bytes is None else str(r.size_bytes)
            warn = ""
            if r.triangles is not None and r.triangles > max_faces:
                warn = f"  [OVER_LIMIT>{max_faces}]"
            print(f"{r.fmt:7s}  tris={tri_s:>8s}  bytes={size_s:>10s}{warn}  {r.path}")
            continue

        print("===")
        print(f"path: {r.path}")
        print(f"exists: {r.exists}")
        if r.size_bytes is not None:
            print(f"size_bytes: {r.size_bytes}")
        print(f"format: {r.fmt}")
        if r.triangles is not None:
            print(f"triangles: {r.triangles}")
            if r.triangles > max_faces:
                print(f"[WARN] triangles exceed MuJoCo limit ({max_faces}) by {r.triangles - max_faces}")
        if r.expected_binary_size is not None:
            print(f"expected_binary_size: {r.expected_binary_size}")
        if r.notes:
            print("notes:")
            for n in r.notes:
                print(f"- {n}")

    if any_error:
        return 2
    if worst_over > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

