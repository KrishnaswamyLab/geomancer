import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "geomancer"
BANNED = ("manyruns.pipeline", "manyruns.tui", "manyruns.vocab")

#: The one exception: training mode loads data through manyruns' PUBLIC loaders, which live in
#: `manyruns.pipeline.__all__`. Only this file may import that package, and only these names.
#: Remove the exception once manyruns ships a loader outside `pipeline` (upstream follow-up).
LOADER_FILE = ROOT / "training" / "execute.py"
LOADERS = {"load_array", "load_labeled", "labels_of", "gene_axis", "as_matrix", "quiet"}


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            yield [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]


def test_package_does_not_import_compute_or_ui_internals():
    assert ROOT.is_dir(), "package root moved"
    modules = list(ROOT.rglob("*.py"))
    assert modules
    for path in modules:
        tree = ast.parse(path.read_text(), filename=str(path))
        for names in _imports(tree):
            if path == LOADER_FILE:
                names = [n for n in names if n not in ("manyruns", "manyruns.pipeline")]
            assert not any(name == banned or name.startswith(banned + ".")
                           for name in names for banned in BANNED), str(path)


def test_loader_exception_uses_only_public_loaders():
    assert LOADER_FILE.is_file(), "loader module moved"
    tree = ast.parse(LOADER_FILE.read_text())
    used = {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "pipeline"}
    assert used and used <= LOADERS, used - LOADERS
    import manyruns.pipeline as pipeline

    assert LOADERS <= set(pipeline.__all__), "an allowlisted loader is no longer public"
