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


def _manyruns_imports(path):
    names = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names += [node.module] + [f"{node.module}.{a.name}" for a in node.names]
    return {n for n in names if n == "manyruns" or n.startswith("manyruns.")}


def test_every_manyruns_call_of_the_learner_lives_in_drive_py():
    # Amendment 3: the move to manyruns.action_space touches one file. score.py (and
    # table.py through it) only reads ledgers. The lab tools import no manyruns at all.
    learner = ROOT / "learner"
    allowed = {"score.py": {"manyruns", "manyruns.store"}}
    for path in sorted(learner.rglob("*.py")) + sorted((ROOT / "configs" / "tool").glob("*.py")):
        found = _manyruns_imports(path)
        if path.parent == learner and path.name == "drive.py":
            continue
        limit = allowed.get(path.name, set()) if path.parent == learner else set()
        assert found <= limit, f"{path.name} imports {sorted(found - limit)}"
        assert not any(n.startswith("manyruns.experiment") for n in found), path.name
        assert "run_all" not in path.read_text(), path.name


def test_the_learner_writes_no_decision_rows():
    for path in sorted((ROOT / "learner").rglob("*.py")):
        assert not any(n.startswith("manyruns.decisions") for n in _manyruns_imports(path)), path.name
        assert "decisions.jsonl" not in path.read_text(), path.name
