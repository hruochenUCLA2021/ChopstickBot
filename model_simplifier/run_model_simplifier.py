#!/usr/bin/env python3
"""
Dispatch mesh simplification based on a YAML config.

See: model_simplifier_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    return cfg


def main() -> int:
    # Make imports work whether this file is executed as:
    # - `python ChopstickBot/model_simplifier/run_model_simplifier.py` (script)
    # - `python -m ChopstickBot.model_simplifier.run_model_simplifier` (module)
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("model_simplifier_config.yaml")),
        help="Path to model_simplifier_config.yaml",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)
    method = str(cfg.get("method", "")).strip()

    # Map method -> module
    if method == "decimation.blender":
        from ChopstickBot.model_simplifier.decimation.blender.run import run as runner
    elif method == "decimation.meshlab":
        from ChopstickBot.model_simplifier.decimation.meshlab.run import run as runner
    elif method == "remesh_simplify.blender":
        from ChopstickBot.model_simplifier.remesh_simplify.blender.run import run as runner
    elif method == "remesh_simplify.meshlab":
        from ChopstickBot.model_simplifier.remesh_simplify.meshlab.run import run as runner
    elif method == "convex_decomposition.vhacd":
        from ChopstickBot.model_simplifier.convex_decomposition.vhacd.run import run as runner
    elif method == "convex_decomposition.coacd":
        from ChopstickBot.model_simplifier.convex_decomposition.coacd.run import run as runner
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Expected one of: "
            "decimation.blender, decimation.meshlab, remesh_simplify.blender, remesh_simplify.meshlab, "
            "convex_decomposition.vhacd, convex_decomposition.coacd"
        )

    runner(cfg, cfg_path=cfg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

