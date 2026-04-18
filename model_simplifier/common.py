from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IOConfig:
    input_dir: Path
    output_dir: Path
    exts: tuple[str, ...]
    preserve_tree: bool
    overwrite: bool


def parse_io(cfg: dict, cfg_path: Path) -> IOConfig:
    io = cfg.get("io", {}) or {}
    common = cfg.get("common", {}) or {}

    input_dir = Path(str(io.get("input_dir", ""))).expanduser()
    output_dir = Path(str(io.get("output_dir", ""))).expanduser()
    if not input_dir.is_absolute():
        input_dir = (cfg_path.parent / input_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (cfg_path.parent / output_dir).resolve()

    exts = io.get("exts", [".stl", ".obj", ".ply"])
    if not isinstance(exts, list):
        raise ValueError("io.exts must be a list")
    exts_norm = tuple(str(e).lower() for e in exts)

    preserve_tree = bool(io.get("preserve_tree", False))
    overwrite = bool(common.get("overwrite", False))
    return IOConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        exts=exts_norm,
        preserve_tree=preserve_tree,
        overwrite=overwrite,
    )


def iter_mesh_files(input_dir: Path, exts: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for root, _dirs, files in os.walk(input_dir):
        for fn in files:
            p = Path(root) / fn
            if p.suffix.lower() in exts:
                out.append(p)
    out.sort()
    return out


def out_path_for(in_path: Path, io: IOConfig) -> Path:
    if io.preserve_tree:
        rel = in_path.relative_to(io.input_dir)
        return io.output_dir / rel
    return io.output_dir / in_path.name


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

