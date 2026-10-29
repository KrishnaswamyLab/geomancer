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
    (folder / "manifest.json").write_text(json.dumps(
        {"kind": "synthetic", "samples": samples,
         "code_revision": {"commit": "c0ffee", "dirty": False, "source": "git"}}))
    return folder


def _cfg(folder, out, **fields):
    # The test interpreter stands in for the built tool envs, and the checkout under
    # test may have uncommitted changes: both are allowed here, and only here.
    return {**drive_config(), "inputs": str(folder), "out": str(out), "seeds": [0],
            "allow_tool_override": True, "allow_dirty": True, **fields}


def drive_config():
    from geomancer.learner import drive

    return drive.config


@pytest.fixture
def lab_env(monkeypatch):
    """A clean manyruns/geomancer environment; the tools run in this interpreter."""
    from geomancer import app

    for key in list(os.environ):
        if key.startswith(("MANYRUNS_", "GEOMANCER_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("MANYRUNS_TOOL_SSB_GRAPH_PYTHON", sys.executable)
    monkeypatch.setenv("MANYRUNS_TOOL_SSB_STATE_PYTHON", sys.executable)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        monkeypatch.setenv(name, os.environ.get(name, "2"))
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
    assert len(graph) == 2 * (6 + 18 * 3)  # raw routes once per sample
    assert len({(c["sample"], c["seed"], c["action"]) for c in graph}) == len(graph)
    assert [c["sample"] for c in cells if c["kind"] == "state"] == ["a", "b"]
    pca = next(c for c in graph if c["action"] == "pca+mapper@fine")
    assert (pca["embedding"], pca["extractor"], pca["scale"]) == ("pca", "mapper", "fine")


def test_the_raw_routes_are_planned_once_per_sample_because_they_ignore_the_seed():
    from geomancer.learner import drive

    cells = drive.plan(["a"], seeds=[0, 1, 2], names=sorted(actions.ACTIONS))
    raw = [c for c in cells if c["kind"] == "graph" and c["embedding"] == "raw"]
    assert len(raw) == 6 and {c["seed"] for c in raw} == {0}
    assert sum(c["kind"] == "graph" for c in cells) == 6 + 18 * 3
    assert actions.seedless("raw+paga@fine") and not actions.seedless("pca+paga@fine")


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

    samples, errors, inputs = drive.load_samples(folder)

    assert [s["name"] for s in samples] == ["sample_00007"]
    assert samples[0]["sha256"] == provenance.sha256_file(folder / "sample_00007.npy")
    assert errors == {"sample_09999": "BadZipFile: truncated"}
    # Every row will name the manifest by sha256, and the code that wrote it.
    assert inputs["manifest_sha256"] == provenance.sha256_file(folder / "manifest.json")
    assert inputs["code_revision"]["commit"] == "c0ffee"


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


def _done_ledger(tmp_path, name="synthetic-000of001-local1-p1"):
    from manyruns import store

    ledger = tmp_path / "ledgers" / name

    def row(ok, complete, status, dirty=False, **geo):
        store.append({"run_id": "r", "ok": ok, "complete": complete}, out_dir=ledger,
                     extra={"extra": {"geomancer": {"geomancer_revision": {"commit": "abc", "dirty": dirty},
                                                    "status": status, **geo}}})

    return row


def test_done_counts_final_rows_and_reruns_aborted_and_unknown_ones(tmp_path):
    from geomancer.learner import drive

    row = _done_ledger(tmp_path)
    row(True, True, "ok", kind="graph", sample="s", seed=0, action="pca+paga@fine")
    row(True, False, "ok", kind="graph", sample="s", seed=0, action="pca+paga@coarse")
    # Deterministic failures are results: a failed embedding's cells and a declined adapter.
    row(True, False, "not_run", kind="graph", sample="s", seed=0, action="diffmap+paga@fine")
    row(False, False, "skipped", kind="graph", sample="s", seed=0, action="raw+paga@fine")
    # A cell left by an exception, and a sha256 mismatch, run again.
    row(True, False, "aborted", kind="graph", sample="s", seed=1, action="pca+paga@fine")
    row(True, True, "graph_sha256_mismatch", kind="graph", sample="s", seed=1, action="raw+paga@fine")
    row(True, True, "ok", kind="embedding", sample="s", seed=0, embedding="pca")
    row(True, True, "ok", kind="state", sample="s", seed=0)

    assert drive.done(tmp_path) == {("s", 0, "pca+paga@fine", "abc", False),
                                    ("s", 0, "diffmap+paga@fine", "abc", False),
                                    ("s", 0, "raw+paga@fine", "abc", False),
                                    ("s", None, "state", "abc", False)}
    # retry= reruns final statuses on request.
    assert ("s", 0, "raw+paga@fine", "abc", False) not in drive.done(tmp_path, retry="skipped")


def test_a_row_from_a_dirty_tree_is_never_done_for_a_clean_run(tmp_path):
    from geomancer.learner import drive

    row = _done_ledger(tmp_path)
    row(True, True, "ok", dirty=True, kind="graph", sample="s", seed=0, action="pca+paga@fine")
    cell = {"kind": "graph", "sample": "s", "seed": 0, "action": "pca+paga@fine"}

    finished = drive.done(tmp_path)

    assert drive._key(cell, {"commit": "abc", "dirty": False}) not in finished
    assert drive._key(cell, {"commit": "abc", "dirty": True}) in finished


def test_done_reads_only_this_shards_ledgers(tmp_path):
    from geomancer.learner import drive

    _done_ledger(tmp_path, "synthetic-001of002-local1-p1")(
        True, True, "ok", kind="graph", sample="s", seed=0, action="pca+paga@fine")

    assert drive.done(tmp_path, "synthetic-000of002-") == set()
    assert len(drive.done(tmp_path, "synthetic-001of002-")) == 1


def test_the_tool_steps_name_the_points_by_sample_and_sha256_and_no_path():
    from geomancer.learner import drive

    cell = {"kind": "graph", "embedding": "pca", "extractor": "paga", "scale": "fine"}
    sample = {"name": "sample_00001", "npy": "/somewhere/sample_00001.npy", "sha256": "a" * 64}

    for step in (drive.graph_step(cell, sample), drive.state_step(sample)):
        assert step["params"]["sample"] == "sample_00001" and step["params"]["points_sha256"] == "a" * 64
        assert not any("/" in str(v) for v in step["params"].values())


def test_action_spec_sha256_is_one_per_action_across_samples():
    from geomancer.learner import drive

    def sha(action, name):
        embedding, rest = action.split("+")
        extractor, scale = rest.split("@")
        cell = {"kind": "graph", "embedding": embedding, "extractor": extractor, "scale": scale}
        return drive.action_spec_sha256(cell, {"name": name, "sha256": name * 2})

    assert sha("pca+mapper@fine", "sample_00000") == sha("pca+mapper@fine", "sample_00002")
    # Unlike spec_id at v0.1.0, it tells the embeddings apart.
    shas = {sha(a, "sample_00000") for a in actions.ACTIONS}
    assert len(shas) == len(actions.ACTIONS)


def test_a_dirty_checkout_or_input_is_refused_unless_allowed():
    from geomancer.learner import drive

    with pytest.raises(RuntimeError, match="dirty=True"):
        drive.check_revision({"commit": "abc", "dirty": True}, "the checkout")
    with pytest.raises(RuntimeError, match="dirty=None"):
        drive.check_revision({"commit": "abc", "dirty": None}, "the checkout")
    drive.check_revision({"commit": "abc", "dirty": False}, "the checkout")
    drive.check_revision({"commit": "abc", "dirty": True}, "the checkout", allow_dirty=True)


def test_installs_with_a_commit_pass_and_an_editable_engine_is_refused(monkeypatch):
    from geomancer.learner import drive

    drive.check_installs()
    real = provenance.engine_revisions()
    monkeypatch.setattr(provenance, "engine_revisions",
                        lambda: {**real, "manyruns": {"version": "0.1.0", "commit": None, "editable": True}})

    with pytest.raises(RuntimeError, match="manyruns is an editable install"):
        drive.check_installs()


def test_a_tool_that_is_not_built_is_refused_before_any_cell_runs(lab_env, monkeypatch, tmp_path):
    from geomancer.learner import drive

    monkeypatch.delenv("MANYRUNS_TOOL_SSB_GRAPH_PYTHON")
    monkeypatch.setenv("MANYRUNS_TOOL_HOME", str(tmp_path / "empty"))
    drive.setup_env()

    with pytest.raises(RuntimeError, match="ssb_graph: not-built"):
        drive.check_tools(["ssb_graph"])
    # The test interpreter is an override: accepted only when allowed.
    monkeypatch.setenv("MANYRUNS_TOOL_SSB_GRAPH_PYTHON", sys.executable)
    with pytest.raises(RuntimeError, match="overridden"):
        drive.check_tools(["ssb_graph"])
    assert drive.check_tools(["ssb_graph"], allow_override=True) == {"ssb_graph": "overridden"}


# ── the manylatents engine ────────────────────────────────────────────────────

pytest_engine = pytest.mark.skipif(
    not all(__import__("importlib").util.find_spec(m) for m in ("manylatents", "kmapper", "scanpy")),
    reason="needs the learner extra")


@pytest_engine
def test_session_branch_lineage(tmp_path, lab_env):
    """1 sample, 1 embedding, 2 branches: each branch row's parent is the embedding row."""
    from geomancer.learner import drive

    folder, out = _inputs(tmp_path), tmp_path / "out"
    cfg = _cfg(folder, out, actions="pca+paga@default,pca+mapper@coarse", state=False)

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
        # No path in any step's params: spec_id must not depend on the machine.
        assert not any("/" in str(v) for step in r["steps"] for v in (step.get("params") or {}).values())
        assert geo["inputs"]["manifest_sha256"] == provenance.sha256_file(folder / "manifest.json")
        assert geo["engine_revisions"]["manylatents-omics"]["commit"]
        assert geo["actions_sha256"] == r["g_vector"]["ssb_graph.actions_sha256"]
    assert {r["extra"]["geomancer"]["action"] for r in graph} == {"pca+paga@default", "pca+mapper@coarse"}
    # The two branches' graphs are different files, so the shared workdir did not leak.
    assert len({r["extra"]["geomancer"]["graph"]["sha256"] for r in graph}) == 2


@pytest_engine
def test_raw_arm_and_state_rows_have_no_parent(tmp_path, lab_env):
    from geomancer.learner import drive

    folder, out = _inputs(tmp_path), tmp_path / "out"
    cfg = _cfg(folder, out, actions="raw+paga@coarse")

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
    cfg = _cfg(folder, out, actions="raw+mapper@fine")
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
    cfg = _cfg(folder, out, actions="pca+paga@fine,pca+mapper@fine", state=False)

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
    # A failed embedding fails the same way again: its not_run cells are final, so a
    # resubmission does not rerun the embedding.
    assert {k[2] for k in drive.done(out)} == {"pca+paga@fine", "pca+mapper@fine"}


@pytest_engine
def test_the_spec_id_of_a_cell_is_the_same_wherever_the_inputs_live(tmp_path, lab_env):
    import shutil

    from geomancer.learner import drive

    folder = _inputs(tmp_path)
    elsewhere = tmp_path / "elsewhere" / "synthetic"
    shutil.copytree(folder, elsewhere)
    specs = []
    for name, inputs in (("a", folder), ("b", elsewhere)):
        drive.run(_cfg(inputs, tmp_path / name, actions="raw+paga@coarse", state=False))
        specs.append([r["spec_id"] for r in _rows(tmp_path / name)])

    assert specs[0] == specs[1] and len(specs[0]) == 1


@pytest_engine
def test_an_exception_in_a_unit_leaves_an_aborted_row_for_each_remaining_cell(
        tmp_path, lab_env, monkeypatch):
    from geomancer.learner import drive

    def mismatch(res, out_dir, out):
        raise RuntimeError("graph.json does not hash to graph_sha256")

    monkeypatch.setattr(drive, "_keep_graph", mismatch)
    folder, out = _inputs(tmp_path), tmp_path / "out"

    summary = drive.run(_cfg(folder, out, actions="raw+paga@fine,raw+mapper@fine", state=False))

    rows = [r for r in _rows(out) if r["extra"]["geomancer"]["kind"] == "graph"]
    status = {r["extra"]["geomancer"]["action"]: r["extra"]["geomancer"]["status"] for r in rows}
    assert status == {"raw+paga@fine": "graph_sha256_mismatch", "raw+mapper@fine": "aborted"}
    aborted = next(r for r in rows if r["extra"]["geomancer"]["status"] == "aborted")
    assert "does not hash" in aborted["extra"]["geomancer"]["detail"] and not aborted["complete"]
    assert summary["errors"] and summary["status"] == {"graph_sha256_mismatch": 1, "aborted": 1}
    # Neither is final, so the next run retries both.
    assert drive.done(out) == set()
