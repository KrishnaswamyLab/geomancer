import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "geomancer"
BANNED = ("manyruns.pipeline", "manyruns.tui", "manyruns.vocab")


def test_package_does_not_import_compute_or_ui_internals():
    assert ROOT.is_dir(), "package root moved"
    modules = list(ROOT.rglob("*.py"))
    assert modules
    for path in modules:
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue
            assert not any(name == banned or name.startswith(banned + ".")
                           for name in names for banned in BANNED), str(path)
