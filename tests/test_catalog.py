from importlib import resources
import json
from pathlib import Path

from manyruns import catalog, toolchain
import pytest
import yaml

ROOT = Path(str(resources.files("geomancer") / "configs"))


@pytest.mark.parametrize("kind", ["recipe", "dataset", "metrics"])
def test_config_roots_and_examples(kind):
    root = ROOT / kind
    assert root.is_dir(), f"{kind} config root moved"
    paths = list(root.glob("*.yaml"))
    assert len(paths) == 1
    assert "DELETABLE EXAMPLE" in paths[0].read_text()


def test_tool_directory_intentionally_omitted():
    assert not (ROOT / "tool").exists(), "G-1 has no lab tool; enable it with G-2"


def test_every_config_validates_and_no_lab_file_shadows_bundled_names():
    for kind in ("recipe", "dataset", "metrics", "tool"):
        lab = ROOT / kind
        bundled = Path(str(resources.files("manyruns") / "configs" / kind))
        assert bundled.is_dir()
        lab_files = {p.relative_to(lab) for p in lab.rglob("*") if p.is_file()}
        upstream_files = {p.relative_to(bundled) for p in bundled.rglob("*") if p.is_file()}
        assert not lab_files & upstream_files, f"{kind}: delete promoted or renamed lab files"
        for path in lab.glob("*.yaml"):
            cfg = yaml.safe_load(path.read_text())
            if kind == "metrics":
                suite = catalog.load_suite(path.stem, lab)
                assert suite == cfg["metrics"]
                assert catalog.check_suite(suite) == []
                from manyruns.pipeline.suite import check_ranges
                assert check_ranges(suite, path.stem, lab) == []
            else:
                checker = {"recipe": catalog.check_recipe, "dataset": catalog.check_dataset,
                           "tool": toolchain.check_tool}[kind]
                assert checker(cfg, path.stem) == []


def test_manyruns_check_accepts_merged_examples(isolated_wrapper, capsys):
    assert isolated_wrapper.main(["check"]) == 0
    output = capsys.readouterr().out
    assert "example_pca" in output and "example_blobs" in output
    assert "FAIL" not in output


def test_mock_run_writes_record(isolated_wrapper, tmp_path):
    import manyruns

    assert isolated_wrapper.main(["run", "example_blobs", "--recipe", "example_pca",
                                  "--engine", "mock", "--project", "g1-smoke"]) == 0
    row, = [json.loads(line) for line in
            (tmp_path / "outputs" / "index.jsonl").read_text().splitlines()]
    assert row["schema_version"] == 1
    assert row["run_id"]
    assert row["engine"] == "mock"
    assert row["ok"]
    if manyruns.__version__ != "0.1.0":
        assert row["caller"] == {"name": "geomancer", "version": isolated_wrapper.__version__}
