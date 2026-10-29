"""score.py and table.py: reductions over the grid's ledgers, joined by run_id.

The ledger tests write rows with manyruns.store.append in the driver's shape
(drive._geo), so no engine runs. test_adapter_parity_raw runs the real ssb_graph
adapter in its own interpreter, as manyruns does. The last test runs the whole
chain, drive -> score -> table -> train, on the manylatents engine.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("kmapper")
pytest.importorskip("scshapebench")

from geomancer.learner import actions, convert, provenance, rewards, score, state, table, train

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "geomancer" / "configs" / "tool"
COMMIT = "c0ffee" * 6 + "c0ff"
CLASSES = ["archetypal", "multi_branch", "simple_traj", "clusters"]


# ── fixtures: a scShapeBench root, convert.py's manifest, driver-shaped rows ──

def _shapes():
    """Three truth graphs with their noisy points: a loop, a line and a Y."""
    rng = np.random.default_rng(0)

    def along(nodes, edges, n=180):
        pos = np.asarray(nodes, float)
        picks = rng.integers(len(edges), size=n)
        t = rng.uniform(size=(n, 1))
        clean = np.array([pos[u] + (pos[v] - pos[u]) * s for (u, v), s in zip(np.asarray(edges)[picks], t)])
        return pos, np.asarray(edges), clean, clean + rng.normal(0, 0.02, clean.shape)

    square = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
    line = [[0, 0, 0], [1, 0.2, 0], [2, 0, 0]]
    y = [[0, 0, 0], [1, 0, 0], [1.8, 0.8, 0], [1.8, -0.8, 0]]
    return [along(square, [[0, 1], [1, 2], [2, 3], [3, 0]]), along(line, [[0, 1], [1, 2]]),
            along(y, [[0, 1], [1, 2], [1, 3]])]


def _synthetic_root(tmp_path):
    folder = tmp_path / "root" / "data" / "synthetic"
    folder.mkdir(parents=True)
    for i, (pos, edges, clean, noisy) in enumerate(_shapes()):
        np.savez(folder / f"sample_{i:05d}.npz", nodes=np.arange(len(pos)), edges=edges,
                 positions=pos, clean_points=clean, noisy_points=noisy)
        (folder / f"sample_{i:05d}.json").write_text(json.dumps({"id": i}))
    manifest = convert.convert_synthetic(tmp_path / "root", tmp_path / "inputs")
    return tmp_path / "root", tmp_path / "inputs" / "synthetic", manifest


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _append(grid, row, geo, ledger="synthetic-000of001"):
    from manyruns import store

    store.append(row, out_dir=Path(grid) / "ledgers" / ledger,
                 extra={"modality": geo.get("modality", "synthetic"),
                        "extra": {"geomancer": {k: v for k, v in geo.items() if k != "modality"}}})


def _geo(kind, sample, seed, points_sha256, source="synthetic", **fields):
    return {"kind": kind, "source": source, "sample": sample, "seed": seed,
            "points_sha256": points_sha256, "geomancer_revision": {"commit": COMMIT, "dirty": False},
            "caller": "geomancer.learner.drive", "caller_version": "0.1.0", **fields}


def _graph(grid, run_id, sample, seed, action, payload, points_sha256, input_sha256,
           parent=None, status="ok", source="synthetic", at="2026-10-29T00:00:00+00:00", **extra):
    """One graph row, its graph file under graphs/, as the driver writes them."""
    embedding, rest = action.split("+")
    extractor, scale = rest.split("@")
    ok = status == "ok"
    graph = None
    g_vector = {}
    if ok:
        # The adapter's own bytes when given, so the hash is the one it emitted.
        data = extra.pop("data", None) or json.dumps(payload).encode()
        path = Path(grid) / "graphs" / f"{run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        graph = {"path": f"graphs/{run_id}.json", "sha256": _sha(data)}
        g_vector = {"ssb_graph.graph_sha256": _sha(data), "ssb_graph.sample": sample,
                    "ssb_graph.points_sha256": points_sha256, "ssb_graph.input_sha256": input_sha256}
    g_vector.update(extra.pop("g_vector", {}))
    row = {"run_id": run_id, "ok": ok, "complete": ok, "dataset_name": extra.pop("dataset_name", sample),
           "seed": seed, "parent": parent, "at": at, "g_vector": g_vector}
    geo = _geo("graph", sample, seed, points_sha256, source=source, action=action,
               embedding=embedding, extractor=extractor, scale=scale,
               embedding_run_id=parent["run_id"] if parent else None, graph=graph, status=status,
               detail=None if ok else "embedding step error", **extra)
    if source == "real":
        geo["modality"] = "scrna"
    _append(grid, row, geo)
    return run_id


def _embedding(grid, run_id, sample, seed, name, emb, points_sha256, ok=True):
    path = Path(grid) / "cells" / sample / f"seed{seed}" / name / "state" / run_id / f"00-{name}_emb.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    row = {"run_id": run_id, "ok": ok, "complete": ok, "dataset_name": sample, "seed": seed,
           "parent": None, "at": "2026-10-29T00:00:00+00:00", "recipe": f"ssb_embed_{name}",
           "steps": [{"index": 0, "name": name, "outcome": "ok" if ok else "error",
                      "artifacts": {"emb": str(path)}}]}
    _append(grid, row, _geo("embedding", sample, seed, points_sha256, embedding=name,
                            status="ok" if ok else "error", detail=None))
    return path


def _state(grid, run_id, sample, points, points_sha256, source="synthetic"):
    g_vector = {"ssb_state.sample": sample}
    for name, fn in state.STATES.items():
        vector = fn(points)
        g_vector[f"ssb_state.{name}_dim"] = int(vector.size)
        g_vector.update({f"ssb_state.{name}_{i}": float(v) for i, v in enumerate(vector)})
    row = {"run_id": run_id, "ok": True, "complete": True, "dataset_name": sample, "seed": 0,
           "parent": None, "at": "2026-10-29T00:00:00+00:00", "g_vector": g_vector}
    geo = _geo("state", sample, 0, points_sha256, source=source, action="state", status="ok",
               detail=None)
    if source == "real":
        geo["modality"] = "scrna"
    _append(grid, row, geo)


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _ledger_ids(grid):
    from manyruns import store

    return {r["run_id"] for d in (Path(grid) / "ledgers").iterdir() for r in store.read(d)}


# ── synthetic ─────────────────────────────────────────────────────────────────

def _adapter(tmp_path, name, points_path, extractor, scale):
    """The ssb_graph adapter on the raw points, run the way manyruns runs it."""
    work = tmp_path / "work" / name / "tool-ssb_graph"
    work.mkdir(parents=True)
    X = np.load(points_path)
    np.save(work / "input.npy", X)
    request = {"tool": "ssb_graph", "version": "0.1.0", "seed": 0,
               "params": {"extractor": extractor, "scale": scale, "points": str(points_path),
                          "points_sha256": provenance.sha256_file(points_path)},
               "input": str(work / "input.npy"), "output": str(work / "output.h5ad"),
               "result": str(work / "result.json"), "workdir": str(work)}
    (work / "request.json").write_text(json.dumps(request))
    proc = subprocess.run([sys.executable, str(TOOLS / "ssb_graph.py"), str(work / "request.json")],
                          capture_output=True, text=True,
                          env={**os.environ, "GEOMANCER_SRC": str(ROOT)})
    assert proc.returncode == 0, proc.stderr
    result = json.loads((work / "result.json").read_text())
    assert result["ok"], result
    return work, result["scalars"]


def test_adapter_parity_raw(tmp_path):
    """On 3 samples, score.py on the adapter's raw-arm graph equals rewards.synthetic_graph exactly."""
    from manylatents.singlecell.data import scshapebench

    root, inputs, manifest = _synthetic_root(tmp_path)
    grid = tmp_path / "grid"
    cells = {"sample_00000": "raw+paga@default", "sample_00001": "raw+mapper@default",
             "sample_00002": "raw+paga@coarse"}
    for i, (name, action) in enumerate(cells.items()):
        entry = manifest["samples"][name]
        extractor, scale = action.split("+")[1].split("@")
        work, scalars = _adapter(tmp_path, name, inputs / entry["npy"], extractor, scale)
        data = (work / "graph.json").read_bytes()
        _graph(grid, f"run{i}", name, 0, action, json.loads(data), entry["npy_sha256"],
               scalars["input_sha256"], data=data,
               g_vector={f"ssb_graph.{k}": v for k, v in scalars.items()})

    scores, meta = score.score_synthetic(grid, inputs)

    assert {s["run_id"] for s in scores} == {"run0", "run1", "run2"}
    for s in scores:
        name, action = s["sample"], s["action"]
        extractor, scale = action.split("+")[1].split("@")
        sample = scshapebench.load_synthetic(int(name.removeprefix("sample_")), root=root)
        expected = rewards.synthetic_graph(
            actions.graph(extractor, scale, sample["points"], sample["points"]), sample)
        assert s["status"] == "ok", s
        assert {k: s[k] for k in expected} == expected
    assert meta["run_ids"] == ["run0", "run1", "run2"]


def _synthetic_grid(tmp_path):
    """One sample, seed 0: a pca embedding with two branches, one raw cell, its state.

    Plus the cells that must not score: a branch whose parent is not in the ledger, a
    branch whose input is not its parent's embedding, a row whose sample disagrees and
    a not_run cell of a failed embedding.
    """
    root, inputs, manifest = _synthetic_root(tmp_path)
    grid = tmp_path / "grid"
    name = "sample_00000"
    entry = manifest["samples"][name]
    X = np.load(inputs / entry["npy"])
    emb = actions._pca(X)
    path = _embedding(grid, "emb0", name, 0, "pca", emb, entry["npy_sha256"])
    emb_sha = provenance.sha256_file(path)
    parent = {"run_id": "emb0", "index": 0}
    payload = actions.graph("paga", "default", emb, X)
    _graph(grid, "g_ok", name, 0, "pca+paga@default", payload, entry["npy_sha256"], emb_sha, parent)
    _graph(grid, "g_raw", name, 0, "raw+paga@default", actions.graph("paga", "default", X, X),
           entry["npy_sha256"], entry["npy_sha256"])
    _graph(grid, "g_orphan", name, 0, "pca+paga@fine", payload, entry["npy_sha256"], emb_sha,
           {"run_id": "nowhere", "index": 0})
    _graph(grid, "g_wrong_input", name, 0, "pca+mapper@default", payload, entry["npy_sha256"],
           "f" * 64, parent)
    _graph(grid, "g_other_sample", name, 0, "raw+mapper@default", payload, entry["npy_sha256"],
           entry["npy_sha256"], g_vector={"ssb_graph.sample": "sample_00001"})
    _graph(grid, "g_not_run", name, 0, "diffmap+paga@default", None, entry["npy_sha256"], None,
           status="not_run")
    _state(grid, "st0", name, X, entry["npy_sha256"])
    return grid, inputs, manifest


def test_join(tmp_path):
    """A graph row with an embedding scores only when its parent exists, is complete and fed it."""
    grid, inputs, _ = _synthetic_grid(tmp_path)

    scores, meta = score.score_synthetic(grid, inputs)

    by = {s["run_id"]: s for s in scores}
    assert by["g_ok"]["status"] == "ok" and by["g_ok"]["embedding_run_id"] == "emb0"
    assert by["g_raw"]["status"] == "ok" and by["g_raw"]["embedding_run_id"] is None
    assert by["g_orphan"]["status"] == "join_failed" and "nowhere" in by["g_orphan"]["detail"]
    assert by["g_wrong_input"]["status"] == "join_failed" and "sha256" in by["g_wrong_input"]["detail"]
    for run_id in ("g_orphan", "g_wrong_input", "g_not_run"):
        assert by[run_id]["reward"] is None
    assert by["g_not_run"]["status"] == "not_run"
    # Every row read is listed, the embedding and state rows too.
    assert set(meta["run_ids"]) == _ledger_ids(grid)


def test_sample_agreement(tmp_path):
    grid, inputs, manifest = _synthetic_grid(tmp_path)

    scores, meta = score.score_synthetic(grid, inputs)

    by = {s["run_id"]: s for s in scores}
    assert by["g_other_sample"]["status"] == "sample_disagrees"
    assert by["g_other_sample"]["reward"] is None
    for s in scores:
        if s["status"] == "ok":
            # The truth each score used is the .npz that convert.py's manifest names.
            assert s["truth_sha256"] == manifest["samples"][s["sample"]]["npz_sha256"]
    assert meta["truth"]["sample_00000"] == manifest["samples"]["sample_00000"]["npz_sha256"]


def test_points_that_differ_from_the_manifest_are_not_scored(tmp_path):
    root, inputs, manifest = _synthetic_root(tmp_path)
    grid = tmp_path / "grid"
    X = np.load(inputs / "sample_00001.npy")
    _graph(grid, "g", "sample_00001", 0, "raw+paga@default", actions.graph("paga", "default", X, X),
           "0" * 64, "0" * 64)

    scores, _ = score.score_synthetic(grid, inputs)

    assert scores[0]["status"] == "points_disagree" and scores[0]["reward"] is None


def test_a_graph_that_does_not_hash_to_its_row_is_refused(tmp_path):
    grid, inputs, _ = _synthetic_grid(tmp_path)
    (grid / "graphs" / "g_ok.json").write_text(json.dumps({"nodes": [0], "edges": []}))

    scores, _ = score.score_synthetic(grid, inputs)

    by = {s["run_id"]: s for s in scores}
    assert by["g_ok"]["status"] == "graph_sha256_mismatch" and by["g_ok"]["reward"] is None


def test_table_cells_cite_rows(tmp_path):
    grid, inputs, _ = _synthetic_grid(tmp_path)
    score.write(grid, "synthetic", *score.score_synthetic(grid, inputs))

    tables = table.build(grid, "synthetic")

    assert [p.name for p in tables] == ["seed0"]
    out = tables[0]
    rows, ids = _read(out / "rewards.jsonl"), _ledger_ids(grid)
    assert all(r["run_id"] in ids for r in rows)
    states = _read(out / "states.jsonl")
    assert [s["run_id"] for s in states] == ["st0"] and set(state.STATES) <= set(states[0])
    cfg = json.loads((out / "config.json").read_text())
    # Refused and failed cells are rows with their status, counted, not dropped.
    assert cfg["counts"] == {"ok": 2, "join_failed": 2, "sample_disagrees": 1, "not_run": 1}
    assert len(rows) == 6 and all(r["reward"] is None for r in rows if r["status"] != "ok")
    assert cfg["reward"] == "synthetic_graph" and cfg["seed"] == 0
    assert cfg["provenance"]["scores"]["sha256"] == provenance.sha256_file(grid / "scores" / "synthetic.jsonl")
    assert set(cfg["provenance"]["ledgers"]) == {"ledgers/synthetic-000of001/index.jsonl"}
    assert cfg["provenance"]["geomancer_commits"] == [COMMIT]
    loaded = train.load_table(out, "input_geometry")
    assert loaded["R"].shape == (1, len(cfg["actions"])) and loaded["excluded"] == []


def test_table_refuses_scores_older_than_the_ledger(tmp_path):
    grid, inputs, manifest = _synthetic_grid(tmp_path)
    score.write(grid, "synthetic", *score.score_synthetic(grid, inputs))
    X = np.load(inputs / "sample_00000.npy")
    sha = manifest["samples"]["sample_00000"]["npy_sha256"]
    _graph(grid, "late", "sample_00000", 0, "raw+mapper@coarse", actions.graph("mapper", "coarse", X, X),
           sha, sha)

    with pytest.raises(ValueError, match="rerun score"):
        table.build(grid, "synthetic")


def test_a_retried_cell_supersedes_its_failed_row_and_seeds_get_their_own_tables(tmp_path):
    root, inputs, manifest = _synthetic_root(tmp_path)
    grid = tmp_path / "grid"
    sha = manifest["samples"]["sample_00002"]["npy_sha256"]
    X = np.load(inputs / "sample_00002.npy")
    payload = actions.graph("paga", "default", X, X)
    _graph(grid, "first", "sample_00002", 0, "raw+paga@default", None, sha, None, status="error",
           at="2026-10-29T00:00:00+00:00")
    _graph(grid, "retry", "sample_00002", 0, "raw+paga@default", payload, sha, sha,
           at="2026-10-29T01:00:00+00:00")
    _graph(grid, "seed1", "sample_00002", 1, "raw+paga@default", payload, sha, sha)
    score.write(grid, "synthetic", *score.score_synthetic(grid, inputs))

    tables = table.build(grid, "synthetic")

    assert [p.name for p in tables] == ["seed0", "seed1"]
    seed0 = _read(tables[0] / "rewards.jsonl")
    assert [r["run_id"] for r in seed0] == ["retry"] and seed0[0]["status"] == "ok"
    cfg = json.loads((tables[0] / "config.json").read_text())
    assert cfg["superseded"] == ["first"]
    # No state run for this sample: the states row says so, and train leaves it out.
    assert _read(tables[0] / "states.jsonl") == [
        {"sample": "sample_00002", "run_id": None, "error": "no ok state run"}]
    assert [r["run_id"] for r in _read(tables[1] / "rewards.jsonl")] == ["seed1"]


def test_table_refuses_rows_from_two_geomancer_commits(tmp_path):
    root, inputs, manifest = _synthetic_root(tmp_path)
    grid = tmp_path / "grid"
    sha = manifest["samples"]["sample_00000"]["npy_sha256"]
    X = np.load(inputs / "sample_00000.npy")
    payload = actions.graph("paga", "default", X, X)
    _graph(grid, "a", "sample_00000", 0, "raw+paga@default", payload, sha, sha)
    _graph(grid, "b", "sample_00000", 0, "raw+paga@fine", payload, sha, sha,
           geomancer_revision={"commit": "other", "dirty": False})
    score.write(grid, "synthetic", *score.score_synthetic(grid, inputs))

    with pytest.raises(ValueError, match="commits"):
        table.build(grid, "synthetic")
    assert table.build(grid, "synthetic", allow_mixed_commits=True)


# ── real: a cross-dataset reduction per (action, seed), with nested folds ─────

def _graph_payload(kind, n, rng, dim=3):
    if kind == "simple_traj":
        edges = [[i, i + 1] for i in range(n - 1)]
    elif kind == "multi_branch":
        edges = [[0, i] for i in range(1, n)]
    elif kind == "archetypal":
        edges = [[i, (i + 1) % n] for i in range(n)] + [[0, n // 2]]
    else:
        edges = []
    positions = rng.normal(size=(n, dim))
    return {"graph_representation": "simple_graph", "nodes": list(range(n)), "edges": edges,
            "node_positions": {str(i): p.tolist() for i, p in enumerate(positions)}}


def _real_grid(grid, labels_flip=None, n=30):
    rng = np.random.default_rng(0)
    ids = [f"SCD-{i:04d}" for i in range(1, n + 1)]
    kinds = [CLASSES[i % 4] for i in range(n)]
    labels = {d: {c: int(c == k) for c in CLASSES} for d, k in zip(ids, kinds)}
    folds = [{"test": ids[f::5], "train": [d for d in ids if d not in ids[f::5]]} for f in range(5)]
    for d, k in zip(ids, kinds):
        payload = _graph_payload(k, int(rng.integers(6, 12)), rng)
        sha = _sha(d.encode())
        _graph(grid, f"r-{d}", d, 0, "raw+paga@default", payload, sha, sha, source="real")
        _state(grid, f"s-{d}", d, rng.normal(size=(40, 3)), sha, source="real")
    return ids, labels, folds


def test_real_scores_are_out_of_fold_and_cite_every_input_run(tmp_path):
    grid = tmp_path / "grid"
    ids, labels, folds = _real_grid(grid)

    scores, meta = score.score_real(grid, labels, folds=folds, models=("rf",))

    top = [s for s in scores if s["outer_fold"] is None]
    assert {s["run_id"] for s in top} == {f"r-{d}" for d in ids}
    assert all(s["status"] == "ok" and 0.0 <= s["reward"] <= 1.0 for s in top)
    assert np.mean([s["reward"] for s in top]) > 0.8
    reduction = next(r for r in meta["reductions"] if r["outer_fold"] is None)
    assert reduction["action"] == "raw+paga@default" and reduction["seed"] == 0
    assert reduction["run_ids"] == sorted(f"r-{d}" for d in ids)
    assert meta["outer_folds"] == [f["test"] for f in folds]
    assert meta["truth"]["labels_sha256"] == score.digest(labels)


def test_policy_training_rewards_never_see_the_labels_of_the_outer_test_fold(tmp_path):
    ids, labels, folds = _real_grid(tmp_path / "grid")
    d = folds[0]["test"][0]
    flipped = {**labels, d: {c: 1 - v for c, v in labels[d].items()}}

    outs = []
    for name, lab in (("before", labels), ("after", flipped)):
        grid = tmp_path / "grid"
        score.write(grid, "real", *score.score_real(grid, lab, folds=folds, models=("rf",)))
        outs.append(table.build(grid, "real", out=tmp_path / name)[0])
    before, after = outs

    nested = {k: _read(k / "rewards_nested.jsonl") for k in (before, after)}
    fold0 = [[r for r in nested[k] if r["outer_fold"] == 0] for k in (before, after)]
    # Rewards of the training samples of outer fold 0 do not move when a fold-0 label flips ...
    assert fold0[0] == fold0[1] and {r["sample"] for r in fold0[0]}.isdisjoint(folds[0]["test"])
    # ... but they do for outer folds whose training samples include d.
    fold1 = [[r for r in nested[k] if r["outer_fold"] == 1] for k in (before, after)]
    assert fold1[0] != fold1[1]
    assert all(r["run_id"] == f"r-{r['sample']}" for r in nested[before])
    cfg = json.loads((before / "config.json").read_text())
    assert cfg["outer_folds"] == [f["test"] for f in folds] and cfg["reward"] == "real_label"
    loaded = train.load_table(before, "none")
    assert [list(np.array(loaded["samples"])[f]) for f in loaded["folds"]] == [f["test"] for f in folds]
    assert loaded["R_fit"].shape == (5, 30, 1)


def test_a_failed_real_cell_leaves_training_and_keeps_its_status(tmp_path):
    grid = tmp_path / "grid"
    ids, labels, folds = _real_grid(grid)
    _graph(grid, "r-failed", "SCD-0031", 0, "raw+paga@default", None, "a" * 64, None,
           status="not_run", source="real")
    labels["SCD-0031"] = labels["SCD-0001"]
    folds[0]["test"].append("SCD-0031")

    scores, _ = score.score_real(grid, labels, folds=folds, models=("rf",))

    failed = [s for s in scores if s["run_id"] == "r-failed"]
    assert failed and all(s["status"] == "not_run" and s["reward"] is None for s in failed)


# ── the whole chain on the manylatents engine ─────────────────────────────────

@pytest.mark.skipif(not __import__("importlib").util.find_spec("manylatents"),
                    reason="needs the learner extra")
def test_drive_score_table_train_end_to_end(tmp_path, monkeypatch):
    from geomancer import app
    from geomancer.learner import drive

    for key in list(os.environ):
        if key.startswith(("MANYRUNS_", "GEOMANCER_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("MANYRUNS_TOOL_SSB_GRAPH_PYTHON", sys.executable)
    monkeypatch.setenv("MANYRUNS_TOOL_SSB_STATE_PYTHON", sys.executable)
    app._cleanup()
    try:
        root, inputs, _ = _synthetic_root(tmp_path)
        grid = tmp_path / "grid"
        drive.run({**drive.config, "inputs": str(inputs), "out": str(grid), "seeds": [0],
                   "samples": "sample_00000,sample_00001", "actions": "raw+paga@default"})
    finally:
        app._cleanup()

    assert score.main([f"grid={grid}", "source=synthetic", f"inputs={inputs}"]) == 0
    table.main([f"grid={grid}", "source=synthetic"])
    out = grid / "tables" / "synthetic" / "seed0"
    train.main([f"table={out}", "seeds=1", "folds=2", "policies=best_fixed,oracle"])

    rows = _read(out / "rewards.jsonl")
    assert len(rows) == 2 and all(r["status"] == "ok" for r in rows)
    assert {r["run_id"] for r in rows} <= _ledger_ids(grid)
    summary = json.loads((out / "train" / "input_geometry" / "summary.json").read_text())
    assert summary["table"]["rewards_sha256"] == provenance.sha256_file(out / "rewards.jsonl")
    assert summary["policies"]["oracle"]["n"] == 2
