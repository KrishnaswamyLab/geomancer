import json
from pathlib import Path
import re
import sys
import tomllib

import manyruns
import pytest
import yaml

from test_pin import UPSTREAM, locked_revision

ROOT = Path(__file__).resolve().parents[1]


def test_readme_and_citation_credit_upstream():
    readme = (ROOT / "README.md").read_text()
    assert readme.splitlines()[0].startswith(f"geomancer is built on [manyruns]({UPSTREAM})")
    assert "Latent Reasoning Works" in readme.splitlines()[0]
    assert re.search(r"\[!\[Built on manyruns\]\(https://img\.shields\.io/[^)]+\)\]"
                     r"\(" + re.escape(UPSTREAM) + r"\)", readme)
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert any(ref.get("repository-code") == UPSTREAM and
               {"name": "Latent Reasoning Works"} in ref["authors"]
               for ref in citation["references"])
    assert {"name": "Krishnaswamy Lab"} in citation["authors"]


def test_entrypoint_and_package_metadata():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["name"] == "geomancer"
    assert project["project"]["scripts"] == {"geomancer": "geomancer.app:main"}
    assert project["project"]["requires-python"] == ">=3.11,<3.13"
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["include"] == ["/geomancer/"]


def version_line(app, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["geomancer"])
    with pytest.raises(SystemExit) as result:
        app.main(["--version"])
    assert result.value.code == 0
    return capsys.readouterr().out.strip()


def expected_line(app, version, revision):
    if manyruns.__version__ == "0.1.0":
        return f"geomancer {version} · revision {revision}"
    return f"geomancer {app.__version__} · built on manyruns {version} · revision {revision}"


def test_version_format_from_distribution_provenance(isolated_wrapper, monkeypatch, capsys):
    from importlib import metadata

    original = metadata.distribution
    revision = "a" * 40

    class Distribution:
        version = manyruns.__version__

        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps({"vcs_info": {"commit_id": revision}})

    monkeypatch.setattr(metadata, "distribution",
                        lambda name: Distribution() if name == "manyruns" else original(name))
    assert version_line(isolated_wrapper, monkeypatch, capsys) == expected_line(
        isolated_wrapper, manyruns.__version__, revision)


def test_runtime_version_matches_lane(lock, isolated_wrapper, monkeypatch, capsys, request):
    from importlib import metadata

    revision = locked_revision(lock).fragment
    dist = metadata.distribution("manyruns")
    actual = json.loads(dist.read_text("direct_url.json") or "{}")["vcs_info"]["commit_id"]
    assert re.fullmatch(r"[0-9a-f]{40}", actual)
    if request.config.getoption("--upstream"):
        assert actual != revision, "upstream lane still runs the locked revision"
    else:
        assert actual == revision
    assert version_line(isolated_wrapper, monkeypatch, capsys) == expected_line(
        isolated_wrapper, dist.version, actual)
