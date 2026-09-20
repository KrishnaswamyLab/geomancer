"""An immutable upstream ref on main, its lock, and the consumer install path."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/latent-reasoning-works/manyruns"


def target_branch(environ=None):
    environ = os.environ if environ is None else environ
    return (environ.get("GITHUB_BASE_REF") or environ.get("GITHUB_REF_NAME") or
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                    cwd=ROOT, text=True).strip())


def validate_declaration(project, target):
    dependencies = project["project"]["dependencies"]
    matches = [d for d in dependencies if re.match(r"(?i)^manyruns(?:\b|\[)", d.strip())]
    assert matches == ["manyruns"], "declare manyruns exactly once, as a bare name"
    source = project["tool"]["uv"]["sources"]["manyruns"]
    assert source["git"] == UPSTREAM
    refs = set(source) - {"git"}
    assert len(refs) == 1 and refs <= {"tag", "rev", "branch"}
    kind, = refs
    value = source[kind]
    assert isinstance(value, str) and value
    if kind == "tag":
        assert re.fullmatch(r"v\d+\.\d+\.\d+", value)
    elif target == "main":
        assert kind == "rev" and re.fullmatch(r"[0-9a-f]{40}", value), "main needs a tag or full SHA"
    else:
        assert kind == "rev" or value == "dev", "only the dev branch is allowed"
        print(f"co-development pin: {kind}={value}; cannot merge a dev ref to main")
    return source


def locked_revision(lock):
    packages = [p for p in lock["package"] if p["name"] == "manyruns"]
    assert len(packages) == 1
    parsed = urlsplit(packages[0]["source"]["git"])
    assert re.fullmatch(r"[0-9a-f]{40}", parsed.fragment)
    return parsed


def test_declaration():
    validate_declaration(tomllib.loads((ROOT / "pyproject.toml").read_text()), target_branch())


@pytest.mark.parametrize("base,ref,expected", [("main", "feature", "main"),
                                              ("", "dev", "dev")])
def test_target_uses_pr_base_before_push_ref(base, ref, expected):
    assert target_branch({"GITHUB_BASE_REF": base, "GITHUB_REF_NAME": ref}) == expected


def test_target_falls_back_to_checkout():
    expected = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    assert target_branch({}) == expected


def declaration(**ref):
    return {"project": {"dependencies": ["manyruns"]},
            "tool": {"uv": {"sources": {"manyruns": {"git": UPSTREAM, **ref}}}}}


@pytest.mark.parametrize("ref", [{"branch": "dev"}, {"rev": "abc123"},
                                {"tag": "main"}, {"tag": "v0.1.0", "branch": "dev"}])
def test_main_rejects_mutable_or_ambiguous_refs(ref):
    with pytest.raises(AssertionError):
        validate_declaration(declaration(**ref), "main")


@pytest.mark.parametrize("dependencies", [["manyruns", "manyruns"],
    ["manyruns>=0.1.0"], ["manyruns @ git+" + UPSTREAM + "@v0.1.0"], ["manyruns[real]"]])
def test_rejects_nonbare_or_duplicate_dependency(dependencies):
    project = declaration(tag="v0.1.0")
    project["project"]["dependencies"] = dependencies
    with pytest.raises(AssertionError):
        validate_declaration(project, "main")


@pytest.mark.parametrize("ref", [{"branch": "dev"}, {"rev": "abc123"}])
def test_development_branch_allows_ref_with_notice(ref, capsys):
    validate_declaration(declaration(**ref), "feature")
    assert "co-development pin" in capsys.readouterr().out


def test_main_accepts_full_sha():
    validate_declaration(declaration(rev="a" * 40), "main")


def test_lock_source_agrees(lock):
    source = validate_declaration(tomllib.loads((ROOT / "pyproject.toml").read_text()),
                                  target_branch())
    parsed = locked_revision(lock)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == UPSTREAM
    kind, = set(source) - {"git"}
    assert parse_qs(parsed.query) == {kind: [source[kind]]}
    if kind == "rev" and re.fullmatch(r"[0-9a-f]{40}", source[kind]):
        assert parsed.fragment == source[kind]


@pytest.mark.network
def test_uv_lock_check(lock):
    subprocess.run(["uv", "lock", "--check"], cwd=ROOT, check=True)


@pytest.mark.network
def test_exception_sha_is_on_upstream_main(tmp_path):
    source = validate_declaration(tomllib.loads((ROOT / "pyproject.toml").read_text()),
                                  target_branch())
    if target_branch() != "main" or "rev" not in source:
        return
    repo = tmp_path / "upstream.git"
    subprocess.run(["git", "init", "--bare", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "fetch", "--no-tags", UPSTREAM,
                    "main:refs/heads/main"], check=True)
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                    source["rev"], "refs/heads/main"], check=True)


@pytest.mark.network
def test_consumer_git_install_reports_locked_revision(lock, tmp_path):
    revision = locked_revision(lock).fragment
    environ = {k: v for k, v in os.environ.items()
               if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
               and not k.startswith("MANYRUNS_")}
    environ.update(UV_TOOL_DIR=str(tmp_path / "tools"),
                   UV_TOOL_BIN_DIR=str(tmp_path / "bin"))
    subprocess.run(["uv", "tool", "install", "--python", sys.executable,
                    "--torch-backend=cpu", "git+" + ROOT.as_uri()],
                   cwd=tmp_path, env=environ, check=True)
    script = tmp_path / "bin" / "geomancer"
    output = subprocess.check_output([str(script), "--version"], cwd=tmp_path,
                                     env=environ, text=True).strip()
    assert output.startswith("geomancer ")
    assert output.endswith(" · revision " + revision), output
    subprocess.run([str(script), "check"], cwd=tmp_path, env=environ, check=True)
