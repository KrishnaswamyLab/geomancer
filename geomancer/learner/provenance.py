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
