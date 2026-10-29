"""The action-grid driver: manyruns sessions, one run record per (sample, seed, action).

The engine tests run the real manylatents engine and the real ssb_graph adapter.
The tool interpreter is this venv's (MANYRUNS_TOOL_<NAME>_PYTHON), which has every
package the tool lock pins, so the tests need no built tool environment.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from geomancer.learner import actions, provenance

ROOT = Path(__file__).resolve().parents[1]


def _circle(n=160, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    points = np.c_[np.cos(theta), np.sin(theta), 0.1 * rng.normal(size=n)]
    return points + rng.normal(0, 0.03, points.shape)


def _inputs(tmp_path, names=("sample_00007",)):
    folder = tmp_path / "inputs" / "synthetic"
    folder.mkdir(parents=True)
    samples = {}
    for i, name in enumerate(names):
        np.save(folder / f"{name}.npy", _circle(seed=i))
        samples[name] = {"status": "ok", "npy": f"{name}.npy",
                         "npy_sha256": provenance.sha256_file(folder / f"{name}.npy")}
    samples["sample_09999"] = {"status": "error", "error": "BadZipFile: truncated"}
    (folder / "manifest.json").write_text(json.dumps({"kind": "synthetic", "samples": samples}))
    return folder


@pytest.fixture
def lab_env(monkeypatch):
    """A clean manyruns/geomancer environment; the tools run in this interpreter."""
    from geomancer import app

    for key in list(os.environ):
        if key.startswith(("MANYRUNS_", "GEOMANCER_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("MANYRUNS_TOOL_SSB_GRAPH_PYTHON", sys.executable)
    monkeypatch.setenv("MANYRUNS_TOOL_SSB_STATE_PYTHON", sys.executable)
    app._cleanup()
    yield
    for key in list(os.environ):
        if key.startswith(("MANYRUNS_", "GEOMANCER_")):
            monkeypatch.delenv(key)
    app._cleanup()


def _rows(out):
    from manyruns import store

    return [row for ledger in sorted((out / "ledgers").glob("*/index.jsonl"))
            for row in store.read(ledger.parent)]


# ── plain planning, no engine ─────────────────────────────────────────────────

def test_plan_is_every_action_per_seed_plus_one_state_cell_per_sample():
    from geomancer.learner import drive

    cells = drive.plan(["a", "b"], seeds=[0, 1, 2], names=sorted(actions.ACTIONS))

    graph = [c for c in cells if c["kind"] == "graph"]
    assert len(graph) == 2 * 3 * 24
    assert len({(c["sample"], c["seed"], c["action"]) for c in graph}) == len(graph)
    assert [c["sample"] for c in cells if c["kind"] == "state"] == ["a", "b"]
    pca = next(c for c in graph if c["action"] == "pca+mapper@fine")
    assert (pca["embedding"], pca["extractor"], pca["scale"]) == ("pca", "mapper", "fine")


def test_units_group_an_embedding_with_its_six_graph_cells():
    from geomancer.learner import drive

    units = drive.units(drive.plan(["a"], seeds=[0], names=sorted(actions.ACTIONS)))

    arms = sorted(u["arm"] for u in units)
    assert arms == ["diffmap", "pca", "phate", "raw", "state"]
    assert all(len(u["cells"]) == 6 for u in units if u["arm"] != "state")


def test_shards_are_contiguous_and_cover_every_sample_once():
    from geomancer.learner import drive

    names = [f"s{i}" for i in range(23)]
    shards = [drive.shard(names, i, 5) for i in range(5)]

    assert sum(shards, []) == names
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_seeds_parse_from_the_cli_forms():
    from geomancer.learner import drive

    assert drive.parse_seeds("0,1,2") == [0, 1, 2]
    assert drive.parse_seeds(3) == [3]
    assert drive.parse_seeds([0, 2]) == [0, 2]


def test_samples_load_from_the_manifest_and_errors_are_counted(tmp_path):
    from geomancer.learner import drive

    folder = _inputs(tmp_path)

    samples, errors = drive.load_samples(folder)

    assert [s["name"] for s in samples] == ["sample_00007"]
    assert samples[0]["sha256"] == provenance.sha256_file(folder / "sample_00007.npy")
    assert errors == {"sample_09999": "BadZipFile: truncated"}


def test_a_points_file_that_differs_from_its_manifest_is_refused(tmp_path):
    from geomancer.learner import drive

    folder = _inputs(tmp_path)
    np.save(folder / "sample_00007.npy", np.zeros((5, 3)))

    with pytest.raises(ValueError, match="sha256"):
        drive.load_samples(folder)


def test_setup_merges_the_lab_tools_and_sets_the_caller(lab_env):
    from geomancer.learner import drive

    drive.setup_env()

    tools = Path(os.environ["MANYRUNS_TOOL_DIR"])
    assert (tools / "ssb_graph.yaml").is_file() and (tools / "ssb_graph.py").is_file()
    assert (tools / "locks").is_dir()
    assert os.environ["GEOMANCER_CALLER"] == "geomancer.learner.drive"
    assert os.environ["GEOMANCER_CALLER_VERSION"]
    assert Path(os.environ["GEOMANCER_SRC"], "geomancer", "learner", "actions.py").is_file()
    # Rule: geomancer never invents MANYRUNS_* names for its own values.
    assert not any(k.startswith("MANYRUNS_") and "CALLER" in k for k in os.environ)


def test_setup_keeps_a_caller_that_is_already_set(lab_env, monkeypatch):
    from geomancer.learner import drive

    monkeypatch.setenv("GEOMANCER_CALLER", "outer")
    monkeypatch.setenv("GEOMANCER_CALLER_VERSION", "7")
    drive.setup_env()

    assert os.environ["GEOMANCER_CALLER"] == "outer" and os.environ["GEOMANCER_CALLER_VERSION"] == "7"


def test_done_keeps_only_ok_and_complete_graph_and_state_rows(tmp_path):
    from manyruns import store

    from geomancer.learner import drive

    ledger = tmp_path / "ledgers" / "x"

    def row(ok, complete, **geo):
        store.append({"run_id": "r", "ok": ok, "complete": complete}, out_dir=ledger,
                     extra={"extra": {"geomancer": {"geomancer_revision": {"commit": "abc"}, **geo}}})

    row(True, True, kind="graph", sample="s", seed=0, action="pca+paga@fine")
    row(True, False, kind="graph", sample="s", seed=0, action="pca+paga@coarse")
    row(False, False, kind="graph", sample="s", seed=1, action="pca+paga@fine")
    row(True, True, kind="embedding", sample="s", seed=0, embedding="pca")
    row(True, True, kind="state", sample="s", seed=0)

    assert drive.done(tmp_path) == {("s", 0, "pca+paga@fine", "abc"), ("s", None, "state", "abc")}


# ── the manylatents engine ────────────────────────────────────────────────────

pytest_engine = pytest.mark.skipif(
    not all(__import__("importlib").util.find_spec(m) for m in ("manylatents", "kmapper", "scanpy")),
    reason="needs the learner extra")


@pytest_engine
def test_session_branch_lineage(tmp_path, lab_env):
    """1 sample, 1 embedding, 2 branches: each branch row's parent is the embedding row."""
    from geomancer.learner import drive

    folder, out = _inputs(tmp_path), tmp_path / "out"
    cfg = {**drive.config, "inputs": str(folder), "out": str(out), "seeds": [0],
           "actions": "pca+paga@default,pca+mapper@coarse", "state": False}

    drive.run(cfg)

    rows = _rows(out)
    emb = [r for r in rows if r["extra"]["geomancer"]["kind"] == "embedding"]
    graph = [r for r in rows if r["extra"]["geomancer"]["kind"] == "graph"]
    assert len(emb) == 1 and len(graph) == 2
    emb = emb[0]
    assert emb["ok"] and emb["complete"] and emb["recipe"] == "ssb_embed_pca"
    complete = out / "cells" / "sample_00007" / "seed0" / "pca" / "state" / emb["run_id"] / "COMPLETE"
    assert any(Path(p).name.endswith("_emb.npy") and v["slot"] == "emb"
               for p, v in json.loads(complete.read_text()).items())
    for r in graph:
        geo = r["extra"]["geomancer"]
        assert r["ok"] and r["complete"], r["steps"]
        assert r["parent"] == {"run_id": emb["run_id"], "index": 0}
        assert geo["embedding_run_id"] == emb["run_id"]
        # The copied graph is the adapter's, tied by hash three ways.
        copied = out / geo["graph"]["path"]
        assert copied.name == f"{r['run_id']}.json"
        assert provenance.sha256_file(copied) == geo["graph"]["sha256"] == r["g_vector"]["ssb_graph.graph_sha256"]
        # Sample agreement: native dataset_name, the driver's record and the adapter's scalar.
        assert r["dataset_name"] == geo["sample"] == r["g_vector"]["ssb_graph.sample"] == "sample_00007"
        assert geo["points_sha256"] == r["g_vector"]["ssb_graph.points_sha256"]
        assert geo["caller"] == "geomancer.learner.drive" == r["g_vector"]["ssb_graph.caller"]
        assert geo["geomancer_revision"]["commit"] == r["g_vector"]["ssb_graph.geomancer_revision"]
        assert geo["status"] == "ok" and r["seed"] == 0
        assert not any("score" in k or "reward" in k for k in r["g_vector"] if k.startswith("ssb_graph."))
    assert {r["extra"]["geomancer"]["action"] for r in graph} == {"pca+paga@default", "pca+mapper@coarse"}
    # The two branches' graphs are different files, so the shared workdir did not leak.
    assert len({r["extra"]["geomancer"]["graph"]["sha256"] for r in graph}) == 2


@pytest_engine
def test_raw_arm_and_state_rows_have_no_parent(tmp_path, lab_env):
    from geomancer.learner import drive

    folder, out = _inputs(tmp_path), tmp_path / "out"
    cfg = {**drive.config, "inputs": str(folder), "out": str(out), "seeds": [0],
           "actions": "raw+paga@coarse"}

    drive.run(cfg)

    rows = _rows(out)
    kinds = sorted(r["extra"]["geomancer"]["kind"] for r in rows)
    assert kinds == ["graph", "state"]
    for r in rows:
        assert r["ok"] and r["complete"] and r["parent"] is None
    graph = next(r for r in rows if r["extra"]["geomancer"]["kind"] == "graph")
    assert graph["extra"]["geomancer"]["embedding_run_id"] is None
    stored = json.loads((out / graph["extra"]["geomancer"]["graph"]["path"]).read_text())
    X = np.load(folder / "sample_00007.npy")
    assert stored == json.loads(json.dumps(actions.graph("paga", "coarse", X, X)))
    state = next(r for r in rows if r["extra"]["geomancer"]["kind"] == "state")
    assert state["g_vector"]["ssb_state.sample"] == "sample_00007"


@pytest_engine
def test_resume_writes_no_row_for_a_finished_sample(tmp_path, lab_env):
    from geomancer.learner import drive

    folder, out = _inputs(tmp_path), tmp_path / "out"
    cfg = {**drive.config, "inputs": str(folder), "out": str(out), "seeds": [0],
           "actions": "raw+mapper@fine"}
    drive.run(cfg)
    before = len(_rows(out))

    drive.run(cfg)

    assert before == 2 and len(_rows(out)) == before


@pytest_engine
def test_a_failed_embedding_keeps_its_row_and_records_every_graph_cell_as_not_run(
        tmp_path, lab_env, monkeypatch):
    from geomancer.learner import drive

    real = drive.embedding_recipe

    def broken(name):
        recipe = real(name)
        recipe["steps"] = [{**recipe["steps"][0], "name": "no_such_embedding"}]
        return recipe

    monkeypatch.setattr(drive, "embedding_recipe", broken)
    folder, out = _inputs(tmp_path), tmp_path / "out"
    cfg = {**drive.config, "inputs": str(folder), "out": str(out), "seeds": [0],
           "actions": "pca+paga@fine,pca+mapper@fine", "state": False}

    drive.run(cfg)

    rows = _rows(out)
    emb = [r for r in rows if r["extra"]["geomancer"]["kind"] == "embedding"]
    graph = [r for r in rows if r["extra"]["geomancer"]["kind"] == "graph"]
    assert len(emb) == 1 and not emb[0]["complete"]
    assert len(graph) == 2
    for r in graph:
        geo = r["extra"]["geomancer"]
        assert r["complete"] is False and geo["status"] == "not_run"
        assert geo["embedding_run_id"] == emb[0]["run_id"] and geo["graph"] is None
        assert any("not run" in c for c in r["caveats"])
    # Not done, so a later run retries them.
    assert drive.done(out) == set()
