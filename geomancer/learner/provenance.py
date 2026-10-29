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


def _git(repo, *args):
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
    return proc.stdout.strip() if proc.returncode == 0 else None


def geomancer_revision():
    """The geomancer commit this code is, as ``{"commit", "dirty", "source"}``.

    In order: ``$GEOMANCER_GIT_SHA`` (a job that knows its commit), the git checkout
    this file is in (``dirty`` is true when a tracked file differs from the commit),
    then the install's ``direct_url.json``. ``commit`` is None when none of them says.
    geomancer reports this value; nothing outside geomancer checks it.
    """
    if sha := os.environ.get("GEOMANCER_GIT_SHA"):
        return {"commit": sha, "dirty": None, "source": "GEOMANCER_GIT_SHA"}
    repo = Path(__file__).resolve().parents[2]
    if (repo / ".git").exists() and (commit := _git(repo, "rev-parse", "HEAD")):
        status = _git(repo, "status", "--porcelain", "--untracked-files=no")
        return {"commit": commit, "dirty": None if status is None else bool(status), "source": "git"}
    try:
        direct = json.loads(metadata.distribution("geomancer").read_text("direct_url.json") or "{}")
    except (metadata.PackageNotFoundError, json.JSONDecodeError):
        direct = {}
    commit = (direct.get("vcs_info") or {}).get("commit_id")
    return {"commit": commit, "dirty": None, "source": "direct_url" if commit else None}


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
