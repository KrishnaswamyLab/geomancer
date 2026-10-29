"""What a learner file or a tool result says about where it came from.

Plain functions over files and the environment. The lab tools under
geomancer/configs/tool import this in their own interpreter, so it must import
nothing from manyruns.
"""

import hashlib
import json
import os
import subprocess
from importlib import metadata
from pathlib import Path

CHUNK = 1 << 20


def sha256_file(path):
    """Hex sha256 of a file's bytes, read in chunks (real h5ad files are up to 1 GB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    """sha256 of a JSON value in canonical form (sorted keys, no spaces)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git(repo, *args):
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
    return proc.stdout.strip() if proc.returncode == 0 else None


def geomancer_revision():
    """The geomancer commit this code is, as ``{"commit", "dirty", "source"}``.

    In order: the git checkout this file is in, then ``$GEOMANCER_GIT_SHA`` (a job
    that knows its commit but runs from an install), then the install's
    ``direct_url.json``. ``commit`` is None when none of them says.

    ``dirty`` is true when a tracked file differs from the commit, or when an
    untracked file exists under ``geomancer/`` (a new module or recipe the run may
    import). It is None when no checkout says. ``$GEOMANCER_GIT_SHA`` is never
    trusted over a checkout: when both exist and name other commits, this raises.
    geomancer reports this value; nothing outside geomancer checks it.
    """
    env = os.environ.get("GEOMANCER_GIT_SHA")
    repo = Path(__file__).resolve().parents[2]
    if (repo / ".git").exists() and (commit := _git(repo, "rev-parse", "HEAD")):
        if env and env != commit:
            raise RuntimeError(f"GEOMANCER_GIT_SHA={env} but the checkout {repo} is at {commit}")
        tracked = _git(repo, "status", "--porcelain", "--untracked-files=no")
        untracked = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", "geomancer")
        dirty = None if tracked is None or untracked is None else bool(tracked or untracked)
        return {"commit": commit, "dirty": dirty, "source": "git"}
    if env:
        return {"commit": env, "dirty": None, "source": "GEOMANCER_GIT_SHA"}
    try:
        direct = json.loads(metadata.distribution("geomancer").read_text("direct_url.json") or "{}")
    except (metadata.PackageNotFoundError, json.JSONDecodeError):
        direct = {}
    commit = (direct.get("vcs_info") or {}).get("commit_id")
    return {"commit": commit, "dirty": None, "source": "direct_url" if commit else None}


def dist_revision(name):
    """``{"version", "commit", "editable"}`` of an installed distribution, from its direct_url.json.

    A git install names its commit (``vcs_info.commit_id``); a PyPI wheel has no
    direct_url.json, so ``commit`` is None; an editable install has ``editable`` true
    and no commit, because its code is whatever the directory holds now.
    """
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"version": None, "commit": None, "editable": None}
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
    except json.JSONDecodeError:
        direct = {}
    return {"version": dist.version, "commit": (direct.get("vcs_info") or {}).get("commit_id"),
            "editable": bool((direct.get("dir_info") or {}).get("editable"))}


# The code that loads the data and computes the embeddings: named in every row and
# manifest by version AND commit, because a version string cannot tell a PyPI wheel
# from a git pin (manylatents-omics 0.1.3 is both).
ENGINE_DISTS = ("manyruns", "manylatents", "manylatents-omics", "scshapebench")


def engine_revisions():
    """``{name: dist_revision(name)}`` for ENGINE_DISTS, plus the sha256 of geomancer's uv.lock."""
    out = {name: dist_revision(name) for name in ENGINE_DISTS}
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    out["uv_lock_sha256"] = sha256_file(lock) if lock.is_file() else None
    return out


def versions(names):
    """``{name: installed version or None}``."""
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


# What every lab tool reports about the code and environment it ran in.
TOOL_VERSIONS = ("numpy", "scipy", "scikit-learn", "anndata", "scanpy", "kmapper", "python-igraph")


def verified_points(path, sha256):
    """The array at ``path``, after its bytes are checked against ``sha256``."""
    import numpy as np

    actual = sha256_file(path)
    if actual != sha256:
        raise ValueError(f"points {Path(path).name} has sha256 {actual}, not the {sha256} the run names")
    return np.load(path, allow_pickle=False)


def grid_points(sample, sha256):
    """A sample's points, found through convert.py's folder and checked by sha256.

    A run names its points by ``sample`` and ``points_sha256`` only, never by a path,
    because manyruns hashes the step params into ``spec_id`` and a path would make it
    differ between machines. The folder is ``$GEOMANCER_GRID_INPUTS``, which the
    driver sets and the tool inherits. The manifest entry must be ok and name the
    same sha256, and the file's bytes must hash to it.
    """
    folder = os.environ.get("GEOMANCER_GRID_INPUTS")
    if not folder:
        raise ValueError("GEOMANCER_GRID_INPUTS is not set; the driver sets it to convert.py's folder")
    manifest = Path(folder) / "manifest.json"
    entry = json.loads(manifest.read_text())["samples"].get(sample) or {}
    if entry.get("status") != "ok" or entry.get("npy_sha256") != sha256:
        raise ValueError(f"{sample} is not an ok entry with sha256 {sha256} in the manifest {manifest}")
    return verified_points(Path(folder) / entry["npy"], sha256)


def tool_scalars(adapter):
    """The provenance scalars a lab tool adds to its result.

    ``caller`` and ``caller_version`` come from ``GEOMANCER_CALLER`` and
    ``GEOMANCER_CALLER_VERSION``, which the tool inherits from the process that runs
    manyruns (manyruns starts the adapter without its own ``env=``).
    """
    revision = geomancer_revision()
    return {
        "adapter_sha256": sha256_file(adapter),
        "caller": os.environ.get("GEOMANCER_CALLER"),
        "caller_version": os.environ.get("GEOMANCER_CALLER_VERSION"),
        "geomancer_revision": revision["commit"],
        "geomancer_dirty": revision["dirty"],
        **{f"version_{name.replace('-', '_')}": v for name, v in versions(TOOL_VERSIONS).items()},
    }
