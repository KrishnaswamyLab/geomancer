"""Compile a lab tool's hashed lock, one per platform.

    python -m geomancer.learner.lock tool=ssb_graph
    python -m geomancer.learner.lock tool=ssb_state platforms=macos-arm64

manyruns v0.1.0 has no `tools lock` verb (its refusal text names one), so this
is the lab's: ``uv pip compile --generate-hashes`` of the manifest's ``requires``,
without the first line (the adapter and geomancer code, which is not a package).
Every version is held to geomancer's own uv.lock (``uv export``), so the tool
builds the same graphs as the learner. The lock goes to
``configs/tool/locks/<name>-<version>-<platform>.lock``, the path
manyruns.toolchain.lock_path reads; a run reports ``tool_reproducible: true`` only
when the environment built from it (``geomancer tools install <name>``) is used.
Run from a geomancer checkout; it needs uv and the network.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from geomancer.learner.table import override

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "geomancer" / "configs" / "tool"

config = {"tool": "ssb_graph", "platforms": "macos-arm64,linux-x86_64"}

# manyruns.toolchain.platform_tag -> uv's --python-platform. Mila's nodes have glibc >= 2.28.
PLATFORMS = {"macos-arm64": "aarch64-apple-darwin", "linux-x86_64": "x86_64-manylinux_2_28"}


def load(name):
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(TOOLS / f"{name}.yaml"), resolve=True)


def requirements(cfg):
    """The manifest's ``requires`` without its comments and without the tool's own line."""
    lines = [str(r).split("#")[0].strip() for r in cfg["requires"]]
    return [line for line in lines if line and not line.startswith(cfg["name"])]


def constraints():
    """geomancer's locked versions (base + learner extra), as pip constraints.

    Git and path requirements are dropped: a constraint cannot name a URL, and none
    of them is in a tool lock.
    """
    out = subprocess.run(
        ["uv", "export", "--frozen", "--extra", "learner", "--no-hashes", "--no-emit-project",
         "--no-header", "--no-annotate", "--project", str(ROOT)],
        capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines()
            if line and not line.startswith(("#", "-e", ".")) and " @ " not in line]


def compile_lock(name, tag):
    cfg = load(name)
    target = TOOLS / "locks" / f"{name}-{cfg['version']}-{tag}.lock"
    target.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        requires, held = Path(tmp) / f"{name}.in", Path(tmp) / "constraints.txt"
        requires.write_text("\n".join(requirements(cfg)) + "\n")
        held.write_text("\n".join(constraints()) + "\n")
        subprocess.run(
            ["uv", "pip", "compile", "--generate-hashes", "--quiet", "--no-annotate",
             "--python-version", str(cfg.get("python") or "3.12"),
             "--python-platform", PLATFORMS[tag], "--constraint", str(held),
             "--custom-compile-command", f"python -m geomancer.learner.lock tool={name}",
             "--output-file", str(target), str(requires)],
            check=True)
    return target


def main(argv=None):
    cfg = override(config, sys.argv[1:] if argv is None else argv)
    for tag in str(cfg["platforms"]).split(","):
        print(f"wrote {compile_lock(cfg['tool'], tag)}", flush=True)


if __name__ == "__main__":
    main()
