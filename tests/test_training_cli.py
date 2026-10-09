"""`geomancer train` end to end. Real dataset resolution and loading; step execution is the
mocked boundary (`FakeEngine`), so nothing here is scientific evidence."""
import json

import pytest

from geomancer.training import execute
from training_fakes import FakeEngine


@pytest.fixture
def engine(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(execute, "execute_step", fake)
    return fake


@pytest.fixture
def train(isolated_wrapper, capsys):
    def run(*argv):
        code = isolated_wrapper.main(["train", *argv])
        return code, capsys.readouterr()
    return run


def test_cli_walk_select_stale_and_continue(train, engine):
    code, out = train("start", "synthetic_timecourse", "-s", "cli", "--time", "pseudotime",
                      "--expert", "jo")
    assert code == 0 and "time mode: pseudotime" in out.out
    for _ in range(4):  # filter_genes, normalize, transform, pca
        assert train("continue", "-s", "cli", "--select", "--no-plots")[0] == 0
    assert train("run", "-s", "cli", "phate", "--no-plots")[0] == 0
    assert train("run", "-s", "cli", "phate", "knn=10", "--note", "denser")[0] == 0
    code, out = train("run", "-s", "cli", "phate", "knn=many")
    assert code == 1 and "must be an integer" in out.err
    code, out = train("continue", "-s", "cli")
    assert code == 1 and "none is selected" in out.err
    assert train("select", "-s", "cli", "phate", "2")[0] == 0
    for _ in range(3):
        assert train("continue", "-s", "cli", "--select", "--no-plots")[0] == 0
    assert "nothing to continue" in train("continue", "-s", "cli")[1].out

    calls = len(engine.calls)
    code, out = train("select", "-s", "cli", "phate", "1")
    assert "now stale, not recomputed: dpt, discretize_time, mioflow" in out.out
    assert len(engine.calls) == calls
    code, out = train("continue", "-s", "cli", "--no-plots")
    assert "dpt#1 is stale" in out.out and len(engine.calls) == calls + 1

    code, out = train("run", "-s", "cli", "phate", "knn=10", "--no-plots")
    assert "same parameters, inputs and seed as phate#2" in out.out
    code, out = train("params", "-s", "cli", "dpt")
    assert "n_dcs" in out.out
    code, out = train("show", "-s", "cli", "phate", "2", "--json")
    assert json.loads(out.out)["params"]["changed"] == {"knn": [5, 10]}
    code, out = train("trace", "-s", "cli", "--json")
    rows = [json.loads(line) for line in out.out.splitlines()]
    assert rows[0]["event"] == "session_started" and rows[0]["expert"] == "jo"
    assert [r["event"] for r in rows].count("rejected") == 1
    assert "cli" in train("list")[1].out


def test_real_time_is_detected_and_k_is_required(train, engine):
    code, out = train("start", "synthetic_timecourse", "-s", "rt")
    assert code == 0 and "time mode: real (measured time found)" in out.out
    for _ in range(6):
        assert train("continue", "-s", "rt", "--select", "--no-plots")[0] == 0
    code, out = train("continue", "-s", "rt")
    assert code == 1 and "k_per_timepoint has no default and must be set" in out.err
    assert train("continue", "-s", "rt", "k_per_timepoint=3", "--select", "--no-plots")[0] == 0


def test_cli_refusals(train, engine):
    code, out = train("start", "synthetic_timecourse", "-s", "m", "--engine", "mock")
    assert code == 1 and "mock engine writes no arrays" in out.err
    code, out = train("start", "swissroll", "-s", "r", "--time", "real")
    assert code == 1 and "needs measured time labels" in out.err
    code, out = train("start", "pbmc3k_not_a_dataset", "-s", "x")
    assert code == 1 and "cannot resolve" in out.err


def test_cli_feedback_compare_and_loop(train, engine, monkeypatch):
    assert train("start", "synthetic_timecourse", "-s", "fb", "--time", "pseudotime")[0] == 0
    for _ in range(5):
        assert train("continue", "-s", "fb", "--select", "--no-plots")[0] == 0
    code, out = train("feedback", "-s", "fb", "phate", "1", "way too blurry")
    assert code == 0 and "1. t 15 → 10" in out.out and "--accept 1" in out.out
    code, out = train("feedback", "-s", "fb", "phate", "1", "way too blurry", "--accept", "1",
                      "--no-plots")
    assert code == 0 and "phate#2" in out.out
    code, out = train("compare", "-s", "fb", "phate", "1", "2")
    assert "t: 15 → 10" in out.out
    answers = iter(["knn=9", "", "keep", "done"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    code, out = train("loop", "-s", "fb", "phate", "--no-plots")
    assert code == 0 and "kept phate#3" in out.out


def test_model_tier_needs_its_extra(train, engine, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name in ("anthropic", "manyagents") else real(name, *a))
    code, out = train("start", "synthetic_timecourse", "-s", "m2", "--proposer", "rules+model")
    assert code == 1 and "geomancer[agents]" in out.err
