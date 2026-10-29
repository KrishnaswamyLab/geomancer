from importlib import resources
import json
from pathlib import Path

from manyruns import catalog, toolchain
import pytest
import yaml

ROOT = Path(str(resources.files("geomancer") / "configs"))


# Lab configs beside the deletable examples: the scShapeBench action grid's.
GRID = {"recipe": {"ssb_embed_pca", "ssb_embed_diffmap", "ssb_embed_phate"}, "dataset": set(),
        "metrics": set()}


@pytest.mark.parametrize("kind", ["recipe", "dataset", "metrics"])
def test_config_roots_and_examples(kind):
    root = ROOT / kind
    assert root.is_dir(), f"{kind} config root moved"
    paths = list(root.glob("*.yaml"))
    examples = [p for p in paths if p.stem not in GRID[kind]]
    assert {p.stem for p in paths} - {p.stem for p in examples} == GRID[kind]
    assert len(examples) == 1
    assert "DELETABLE EXAMPLE" in examples[0].read_text()


def test_lab_tools_have_their_adapter_and_a_lock_per_platform():
    root = ROOT / "tool"
    assert {p.stem for p in root.glob("*.yaml")} == {"ssb_graph", "ssb_state"}
    for name in ("ssb_graph", "ssb_state"):
        cfg = yaml.safe_load((root / f"{name}.yaml").read_text())
        assert (root / cfg["entrypoint"]).is_file()
        for tag in ("macos-arm64", "linux-x86_64"):
            assert (root / "locks" / toolchain.lock_name(cfg, tag)).is_file()


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
