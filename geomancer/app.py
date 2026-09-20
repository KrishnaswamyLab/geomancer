"""Declare lab configs and caller identity, then hand the CLI to manyruns.

The temporary merge is only for manyruns 0.1.0. Replace it with path lists when
bumping the pin to the release containing the upstream config resolver.
"""
from __future__ import annotations

import atexit
from importlib import resources
import os
from pathlib import Path
import shutil
import tempfile

from manyruns import env

from . import __version__

_MERGED_ROOT: Path | None = None


def _load_cwd_dotenv() -> None:
    # Match manyruns.app._load_dotenv: cwd only, existing environment wins.
    try:
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env")
    except Exception:
        pass


def _cleanup() -> None:
    global _MERGED_ROOT
    if _MERGED_ROOT is not None:
        shutil.rmtree(_MERGED_ROOT, ignore_errors=True)
        _MERGED_ROOT = None


def _copy_tree(source: Path, destination: Path) -> None:
    """Copy recursively, including locks and adapters; never overwrite a file."""
    if not source.is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        target = destination / item.name
        if item.is_dir():
            _copy_tree(item, target)
        else:
            if target.exists():
                raise ValueError(f"lab config shadows a bundled file: {item.name}")
            shutil.copy2(item, target)


def _point_dir(name: str, lab: Path) -> None:
    global _MERGED_ROOT
    if env.is_set(name):
        return  # An explicit user value, including empty, wins outright.
    if _MERGED_ROOT is None:
        _MERGED_ROOT = Path(tempfile.mkdtemp(prefix="geomancer-"))
        atexit.register(_cleanup)
    kind = name.removesuffix("_DIR").lower()
    destination = _MERGED_ROOT / kind
    bundled = Path(str(resources.files("manyruns") / "configs" / kind))
    try:
        destination.mkdir(parents=True, exist_ok=True)
        _copy_tree(bundled, destination)
        _copy_tree(lab, destination)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    os.environ[f"MANYRUNS_{name}"] = str(destination)


def main(argv: list[str] | None = None) -> int:
    _load_cwd_dotenv()
    root = resources.files(__package__) / "configs"
    for kind in ("RECIPE", "DATASET", "METRICS", "TOOL"):
        _point_dir(f"{kind}_DIR", Path(str(root / kind.lower())))
    if not env.is_set("CALLER"):
        os.environ["MANYRUNS_CALLER"] = __package__
        os.environ["MANYRUNS_CALLER_VERSION"] = __version__
    from manyruns import app

    return app.main(argv)
