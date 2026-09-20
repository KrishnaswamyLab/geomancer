import os
from pathlib import Path

from manyruns import app as upstream, catalog, toolchain
import pytest


KINDS = ("RECIPE", "DATASET", "METRICS", "TOOL")


def test_delegates_and_declares_caller_pair(isolated_wrapper, monkeypatch):
    app = isolated_wrapper
    argv = ["check"]
    calls = []

    def delegate(args):
        calls.append(args)
        assert os.environ["MANYRUNS_CALLER"] == "geomancer"
        assert os.environ["MANYRUNS_CALLER_VERSION"] == app.__version__
        return 17

    monkeypatch.setattr(upstream, "main", delegate)
    assert app.main(argv) == 17
    assert calls == [argv] and calls[0] is argv


@pytest.mark.parametrize("version", [None, "user-version"])
def test_inherited_caller_never_acquires_lab_version(isolated_wrapper, monkeypatch, version):
    monkeypatch.setenv("MANYRUNS_CALLER", "external-caller")
    if version is not None:
        monkeypatch.setenv("MANYRUNS_CALLER_VERSION", version)
    monkeypatch.setattr(upstream, "main", lambda argv: 0)
    isolated_wrapper.main([])
    assert os.environ["MANYRUNS_CALLER"] == "external-caller"
    assert os.environ.get("MANYRUNS_CALLER_VERSION") == version


def test_orphan_version_is_replaced_as_a_pair(isolated_wrapper, monkeypatch):
    monkeypatch.setenv("MANYRUNS_CALLER_VERSION", "orphan")
    monkeypatch.setattr(upstream, "main", lambda argv: 0)
    isolated_wrapper.main([])
    assert os.environ["MANYRUNS_CALLER"] == "geomancer"
    assert os.environ["MANYRUNS_CALLER_VERSION"] == isolated_wrapper.__version__


def test_dotenv_precedes_directories_and_caller(isolated_wrapper, monkeypatch, tmp_path):
    custom = tmp_path / "user-recipes"
    custom.mkdir()
    (tmp_path / ".env").write_text(
        f'MANYRUNS_RECIPE_DIR="{custom}"\nMANYRUNS_CALLER=dotenv-caller\n'
        'MANYRUNS_CALLER_VERSION=dotenv-version\n')

    def delegate(argv):
        assert catalog.recipe_dir() == custom
        assert os.environ["MANYRUNS_CALLER"] == "dotenv-caller"
        assert os.environ["MANYRUNS_CALLER_VERSION"] == "dotenv-version"
        return 0

    monkeypatch.setattr(upstream, "main", delegate)
    assert isolated_wrapper.main(["check"]) == 0


def test_dotenv_is_cwd_only_and_shell_wins(isolated_wrapper, monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("MANYRUNS_CALLER=parent-caller\n")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    isolated_wrapper._load_cwd_dotenv()
    assert "MANYRUNS_CALLER" not in os.environ
    (child / ".env").write_text("MANYRUNS_RECIPE_DIR=dotenv-dir\n")
    monkeypatch.setenv("MANYRUNS_RECIPE_DIR", "shell-dir")
    monkeypatch.setattr(upstream, "main", lambda argv: 0)
    isolated_wrapper.main([])
    assert os.environ["MANYRUNS_RECIPE_DIR"] == "shell-dir"


@pytest.mark.parametrize("value", ["user-dir", ""])
def test_explicit_directory_values_win_outright(isolated_wrapper, monkeypatch, value):
    for kind in KINDS:
        monkeypatch.setenv(f"MANYRUNS_{kind}_DIR", value)
    monkeypatch.setattr(upstream, "main", lambda argv: 0)
    isolated_wrapper.main([])
    assert isolated_wrapper._MERGED_ROOT is None
    for kind in KINDS:
        assert os.environ[f"MANYRUNS_{kind}_DIR"] == value


def test_merged_directories_keep_every_byte_and_use_one_temp_root(isolated_wrapper, monkeypatch):
    from importlib import resources

    app = isolated_wrapper
    monkeypatch.setattr(upstream, "main", lambda argv: 0)
    app.main([])
    root = app._MERGED_ROOT
    assert root.is_dir()
    for kind in KINDS:
        merged = Path(os.environ[f"MANYRUNS_{kind}_DIR"])
        assert merged.parent == root
        for package in ("manyruns", "geomancer"):
            source = Path(str(resources.files(package) / "configs" / kind.lower()))
            for path in source.rglob("*"):
                if path.is_file():
                    copy = merged / path.relative_to(source)
                    assert copy.is_file() and not copy.is_symlink()
                    assert copy.read_bytes() == path.read_bytes()
    assert "example_pca" in catalog.discover_recipes()
    assert "embed" in catalog.discover_recipes()
    assert "example_blobs" in catalog.discover_datasets()
    assert "gaussian_blob" in catalog.discover_datasets()
    assert catalog.load_suite("example_geometry")
    assert toolchain.discover_tools()
    assert list((toolchain.tool_dir() / "locks").glob("*.lock"))
    app.main([])
    assert app._MERGED_ROOT == root
    app._cleanup()
    assert not root.exists()


def test_recursive_copies_keep_lab_adapters_and_locks(isolated_wrapper, tmp_path):
    lab = tmp_path / "tool"
    (lab / "locks").mkdir(parents=True)
    (lab / "adapter.py").write_text("# synthetic adapter fixture\n")
    (lab / "locks" / "fixture.lock").write_text("# synthetic lock fixture\n")
    isolated_wrapper._point_dir("TOOL_DIR", lab)
    merged = Path(os.environ["MANYRUNS_TOOL_DIR"])
    assert (merged / "adapter.py").read_bytes() == (lab / "adapter.py").read_bytes()
    assert (merged / "locks" / "fixture.lock").is_file()
    assert len(list((merged / "locks").glob("*.lock"))) > 1


def test_shadow_refused_without_publishing_partial_directory(isolated_wrapper, tmp_path):
    lab = tmp_path / "recipe"
    lab.mkdir()
    (lab / "embed.yaml").write_text("name: embed\n")
    with pytest.raises(ValueError, match="shadows a bundled file"):
        isolated_wrapper._point_dir("RECIPE_DIR", lab)
    assert "MANYRUNS_RECIPE_DIR" not in os.environ
    assert not (isolated_wrapper._MERGED_ROOT / "recipe").exists()
