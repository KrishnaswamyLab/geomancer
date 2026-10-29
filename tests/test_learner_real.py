import json
import types

import numpy as np
import pytest

pytest.importorskip("kmapper")
pytest.importorskip("scshapebench")
pytest.importorskip("torch_geometric")

from geomancer.learner import rewards, state, table, train

CLASSES = ["archetypal", "multi_branch", "simple_traj", "clusters"]


def _graph(kind, n, rng, dim=3):
    # A shape whose graph statistics differ by kind, so a classifier can learn it.
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


def _fake(n=30, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"SCD-{i:04d}" for i in range(1, n + 1)]
    kinds = [CLASSES[i % 4] for i in range(n)]
    payloads = {d: _graph(k, int(rng.integers(6, 12)), rng) for d, k in zip(ids, kinds)}
    labels = {d: {c: int(c == k) for c in CLASSES} for d, k in zip(ids, kinds)}
    folds = [{"test": ids[f::5], "train": [d for d in ids if d not in ids[f::5]]}
             for f in range(5)]
    return payloads, labels, folds


def test_payload_becomes_the_graph_the_scshapebench_loader_reads():
    payload = {"nodes": [0, 1, 2], "edges": [[0, 1], [1, 2]],
               "node_positions": {"0": [0.0, 0.0, 0.0], "1": [3.0, 4.0, 0.0],
                                  "2": [3.0, 4.0, 10.0]}}
    alone = {"nodes": [0], "edges": [], "node_positions": {"0": [1.0, 2.0, 3.0]}}

    graphs = rewards.payloads_to_graphs({"SCD-0001": payload, "SCD-0002": alone})

    g = graphs["SCD-0001"]
    assert g.x.tolist() == [[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]]
    assert g.edge_index.tolist() == [[0, 1, 1, 2], [1, 2, 0, 1]]
    # Lengths are Euclidean in the full point space (5 and 10), divided by the max.
    assert g.edge_weight.tolist() == pytest.approx([0.5, 1.0, 0.5, 1.0])
    assert tuple(graphs["SCD-0002"].edge_index.shape) == (2, 0)


def test_real_label_reward_is_the_mean_class_agreement_out_of_fold():
    payloads, labels, folds = _fake()

    rows = rewards.real_label(payloads, labels, folds=folds, models=("rf",))

    assert set(rows) == set(payloads)
    for d, row in rows.items():
        y = np.array([labels[d][c] for c in CLASSES])
        p = np.array([row["probs"][c] for c in CLASSES])
        assert row["status"] == "ok" and row["model"] == "rf"
        assert row["reward"] == pytest.approx(np.mean(y * p + (1 - y) * (1 - p)))
        assert row["reward_rf"] == row["reward"]
        assert d in folds[row["fold"]]["test"]
    # The shapes are separable from graph statistics, so the classifier learns them.
    assert np.mean([r["reward"] for r in rows.values()]) > 0.8


def test_a_dataset_label_never_reaches_its_own_prediction():
    payloads, labels, folds = _fake()
    d = folds[0]["test"][0]
    flipped = {**labels, d: {c: 1 - v for c, v in labels[d].items()}}

    before = rewards.real_label(payloads, labels, folds=folds, models=("rf",))
    after = rewards.real_label(payloads, flipped, folds=folds, models=("rf",))

    assert after[d]["probs"] == before[d]["probs"]
    assert any(after[o]["probs"] != before[o]["probs"] for o in folds[1]["test"] + folds[2]["test"])


def test_failed_graph_is_a_zero_reward_error_and_leaves_training():
    payloads, labels, folds = _fake()
    d = folds[1]["test"][0]

    rows = rewards.real_label({**payloads, d: None}, labels, folds=folds, models=("rf",))

    assert rows[d]["reward"] == 0.0 and rows[d]["status"] == "error"
    assert all(r["status"] == "ok" for o, r in rows.items() if o != d)


def test_extra_models_get_their_own_reward_columns_and_one_node_graphs_do_not_poison_gnn():
    payloads, labels, folds = _fake(n=20)
    d = folds[0]["test"][0]
    payloads[d] = {"nodes": [0], "edges": [], "node_positions": {"0": [0.0, 0.0, 0.0]}}

    rows = rewards.real_label(payloads, labels, folds=folds, models=("rf", "gnn"))

    assert all(r["model"] == "rf" for r in rows.values())
    # The one-node graph has no GNN probability; that is an error of gnn only.
    assert rows[d]["reward_gnn"] == 0.0 and rows[d]["probs_gnn"] is None
    assert rows[d]["status"] == "ok" and "gnn" in rows[d]["error_gnn"]
    others = [r for o, r in rows.items() if o != d]
    # Without the guard, NaN from the one-node graph spreads to most other predictions.
    assert all(r["probs_gnn"] is not None and "error_gnn" not in r for r in others)
    assert np.mean([r["reward_gnn"] for r in others]) > 0.6


def test_a_headline_model_without_a_probability_makes_the_row_an_error():
    payloads, labels, folds = _fake(n=20)
    d = folds[0]["test"][0]
    payloads[d] = {"nodes": [0], "edges": [], "node_positions": {"0": [0.0, 0.0, 0.0]}}

    rows = rewards.real_label(payloads, labels, folds=folds, models=("gnn",))

    assert rows[d]["status"] == "error" and rows[d]["reward"] == 0.0 and "gnn" in rows[d]["error"]
    assert all(r["status"] == "ok" for o, r in rows.items() if o != d)


def test_a_fold_with_nothing_to_train_on_is_an_error_not_a_zero_score():
    payloads, labels, folds = _fake()
    d = folds[0]["test"][0]
    folds = [{"train": [], "test": [d]}] + [
        {"train": f["train"], "test": [t for t in f["test"] if t != d]} for f in folds]

    rows = rewards.real_label(payloads, labels, folds=folds, models=("rf",))

    assert rows[d]["status"] == "error" and rows[d]["reward"] == 0.0
    assert "no training graphs" in rows[d]["error"]


def test_an_action_that_failed_everywhere_gives_error_rows_not_an_exception():
    payloads, labels, folds = _fake()

    rows = rewards.real_label(dict.fromkeys(payloads), labels, folds=folds, models=("rf",))

    assert all(r["status"] == "error" and r["reward"] == 0.0 for r in rows.values())


def test_a_graph_scshapebench_rejects_is_an_error_of_that_dataset_only():
    payloads, labels, folds = _fake()
    d = folds[2]["test"][0]
    payloads[d] = {**payloads[d], "node_positions": {**payloads[d]["node_positions"], "0": [np.nan] * 3}}

    rows = rewards.real_label(payloads, labels, folds=folds, models=("rf",))

    assert rows[d]["status"] == "error" and "NaN" in rows[d]["error"]
    assert all(r["status"] == "ok" for o, r in rows.items() if o != d)


def test_real_label_probabilities_are_the_ones_scshapebench_scores(tmp_path, monkeypatch):
    # Run scShapeBench's own runner on the same graphs, labels and folds, and record
    # the probabilities it passes to its metrics: they must equal real_label's.
    import torch
    from scshapebench.evaluator import runner
    from scshapebench.models import get_models

    payloads, labels, folds = _fake()
    order = sorted(labels)
    seen = []
    monkeypatch.setattr(runner, "load_graphs", lambda d: (rewards.payloads_to_graphs(payloads), {}))
    monkeypatch.setattr(runner, "load_labels", lambda path=None: (
        np.array([[labels[d][c] for c in CLASSES] for d in order], np.float32), CLASSES, order))
    monkeypatch.setattr(runner, "baseline_predictions", lambda **kwargs: {})
    monkeypatch.setattr(runner, "evaluate", lambda y, pred, p: seen.append(np.asarray(p)) or {})
    monkeypatch.setattr(runner, "aggregate", lambda results: results)
    monkeypatch.setattr(runner, "save_results", lambda *args: None)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({f"fold_{i}": f for i, f in enumerate(folds)}))
    names = [m.name for m in get_models(torch.device("cpu"))]

    runner.run_real_submission(tmp_path, label_type="high_support", output_dir=tmp_path,
                               device="cpu", splits_path=splits)
    models = ("rf", "stats_svm", "gnn")
    rows = rewards.real_label(payloads, labels, folds=folds, models=models)

    for fold_id, split in enumerate(folds):
        for m in models:
            ours = np.array([[rows[d][f"probs_{m}"][c] for c in CLASSES] for d in split["test"]])
            assert ours == pytest.approx(seen[fold_id * len(names) + names.index(m)])


def _fake_root(tmp_path, n_cells=120, n_genes=40):
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    folder = tmp_path / "data" / "scRNAseq"
    folder.mkdir(parents=True)
    for i, kind in enumerate(["X_pca", "no_pca", "counts"], 1):
        counts = rng.poisson(2.0, (n_cells, n_genes)).astype(np.float32)
        X = counts if kind == "counts" else np.log1p(counts)
        adata = ad.AnnData(sp.csr_matrix(X))
        if kind == "X_pca":
            adata.obsm["X_pca"] = rng.normal(size=(n_cells, 50)).astype(np.float32)
        adata.write_h5ad(folder / f"SCD-{i:04d}.h5ad")
    labels = {f"SCD-{i:04d}": {c: int(c == CLASSES[i]) for c in CLASSES} for i in (1, 2, 3)}
    return types.SimpleNamespace(
        list_real=lambda root=None: ["SCD-0001", "SCD-0002", "SCD-0003", "SCD-0999"],
        real_path=lambda d, root=None: folder / f"{d}.h5ad",
        load_shape_labels=lambda aggregation="high_support", root=None: labels,
    )


def test_real_points_subsample_with_a_fixed_seed_and_fall_back_to_computed_pca(tmp_path):
    loader = _fake_root(tmp_path)

    stored, meta = table.real_points(loader.real_path("SCD-0001"), max_cells=50, n_pcs=10)
    again, _ = table.real_points(loader.real_path("SCD-0001"), max_cells=50, n_pcs=10)
    computed, computed_meta = table.real_points(loader.real_path("SCD-0003"), max_cells=50, n_pcs=10)

    _, log1p_meta = table.real_points(loader.real_path("SCD-0002"), max_cells=50, n_pcs=10)
    _, forced_meta = table.real_points(loader.real_path("SCD-0001"), max_cells=50, n_pcs=10,
                                       points="computed")

    assert stored.shape == (50, 10) and np.array_equal(stored, again)
    # X is not read on the stored X_pca path, so whether it holds counts is unknown.
    assert meta == {"n_cells": 120, "n_points": 50, "points_source": "X_pca", "counts": None}
    assert computed.shape == (50, 10) and np.isfinite(computed).all()
    assert computed_meta["points_source"] == "computed" and computed_meta["counts"] is True
    assert log1p_meta["points_source"] == "computed" and log1p_meta["counts"] is False
    assert forced_meta["points_source"] == "computed" and forced_meta["counts"] is False


def _broken(points):
    raise RuntimeError("this action always fails")


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_real_table_runs_in_two_stages_into_the_files_train_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(table, "_loader", lambda: _fake_root(tmp_path / "root"))
    monkeypatch.setitem(table.actions_module.ACTIONS, "broken", _broken)
    out = tmp_path / "table"
    names = "pca+mapper@default,raw+paga@default,broken"

    table.main(["source=real", "stage=graphs", f"actions={names}", "max_cells=60", "n_pcs=10",
                "workers=1", f"out={out}"])

    graphs = _read(out / "graphs.jsonl")
    # SCD-0999 is on disk but has no label, so it is not a sample.
    assert sorted({g["sample"] for g in graphs}) == ["SCD-0001", "SCD-0002", "SCD-0003"]
    good = [g for g in graphs if g["action"] != "broken"]
    assert len(good) == 6 and all(g["status"] == "ok" and g["payload"]["nodes"] for g in good)
    states = _read(out / "states.jsonl")
    assert all(set(state.STATES) <= set(s) for s in states)
    assert {s["sample"]: s["meta"]["points_source"] for s in states} == {
        "SCD-0001": "X_pca", "SCD-0002": "computed", "SCD-0003": "computed"}
    assert not (out / "rewards.jsonl").exists()

    table.main(["source=real", "stage=rewards", "classifiers=rf", f"out={out}"])

    rows = _read(out / "rewards.jsonl")
    assert len(rows) == 9 and all(0.0 <= r["reward"] <= 1.0 for r in rows)
    broken = [r for r in rows if r["action"] == "broken"]
    assert all(r["status"] == "error" and r["reward"] == 0.0
               and r["error"] == "RuntimeError: this action always fails" for r in broken)
    loaded = train.load_table(out, "input_geometry")
    assert loaded["R"].shape == (3, 3) and loaded["actions"] == names.split(",")


def test_a_dataset_that_cannot_be_read_is_an_error_row_and_the_run_goes_on(tmp_path):
    loader = _fake_root(tmp_path / "root")
    bad = tmp_path / "bad.h5ad"
    bad.write_text("not an h5ad")
    jobs = [("SCD-0005", str(bad)), ("SCD-0001", str(loader.real_path("SCD-0001")))]
    names = {"raw+paga@default": table.actions_module.ACTIONS["raw+paga@default"]}

    table.build_real_graphs(jobs, {}, names, tmp_path / "t", max_cells=60, n_pcs=10)

    graphs = {g["sample"]: g for g in _read(tmp_path / "t" / "graphs.jsonl")}
    states = {s["sample"]: s for s in _read(tmp_path / "t" / "states.jsonl")}
    assert graphs["SCD-0005"]["status"] == "error" and "error" in states["SCD-0005"]
    assert graphs["SCD-0001"]["status"] == "ok" and "error" not in states["SCD-0001"]


def test_stage_one_resumes_from_the_datasets_already_written(tmp_path, monkeypatch):
    loader = _fake_root(tmp_path / "root")
    jobs = [(d, str(loader.real_path(d))) for d in ("SCD-0001", "SCD-0002", "SCD-0003")]
    names = {"raw+paga@default": table.actions_module.ACTIONS["raw+paga@default"]}
    out = tmp_path / "t"
    table.build_real_graphs(jobs[:2], {}, names, out, max_cells=60, n_pcs=10)
    # A crash in the middle of SCD-0003: its graph row was written, its states row was not.
    with open(out / "graphs.jsonl", "a") as f:
        f.write(json.dumps({"sample": "SCD-0003", "action": "raw+paga@default", "status": "ok"}) + "\n")
    computed = []
    real_points = table.real_points
    monkeypatch.setattr(table, "real_points", lambda path, *a: computed.append(path) or real_points(path, *a))

    table.build_real_graphs(jobs, {}, names, out, max_cells=60, n_pcs=10, resume=True)

    assert computed == [jobs[2][1]]
    assert [g["sample"] for g in _read(out / "graphs.jsonl")] == ["SCD-0001", "SCD-0002", "SCD-0003"]
    assert [s["sample"] for s in _read(out / "states.jsonl")] == ["SCD-0001", "SCD-0002", "SCD-0003"]
    with pytest.raises(ValueError, match="max_cells"):
        table.build_real_graphs(jobs, {}, names, out, max_cells=70, n_pcs=10, resume=True)


def _graphs_table(out, payloads, labels):
    out.mkdir()
    (out / "config.json").write_text(json.dumps({"actions": ["a"], "source": "real"}))
    (out / "labels.json").write_text(json.dumps(labels))
    (out / "graphs.jsonl").write_text("".join(
        json.dumps({"sample": d, "action": "a", "status": "ok", "payload": p}) + "\n"
        for d, p in payloads.items()))
    (out / "states.jsonl").write_text("".join(json.dumps({"sample": d, "none": []}) + "\n" for d in payloads))
    return out


def test_policy_training_rewards_never_see_the_labels_of_the_outer_test_fold(tmp_path):
    payloads, labels, folds = _fake()
    d = folds[0]["test"][0]
    flipped = {**labels, d: {c: 1 - v for c, v in labels[d].items()}}

    before = table.build_real_rewards(_graphs_table(tmp_path / "b", payloads, labels), folds=folds)
    after = table.build_real_rewards(_graphs_table(tmp_path / "a", payloads, flipped), folds=folds)

    nested = {k: _read(k / "rewards_nested.jsonl") for k in (before, after)}
    fold0 = [[r for r in nested[k] if r["outer_fold"] == 0] for k in (before, after)]
    # Rewards of the training samples of outer fold 0 do not move when a fold-0 label flips ...
    assert fold0[0] == fold0[1] and {r["sample"] for r in fold0[0]}.isdisjoint(folds[0]["test"])
    # ... but they do for outer folds whose training samples include d.
    fold1 = [[r for r in nested[k] if r["outer_fold"] == 1] for k in (before, after)]
    assert fold1[0] != fold1[1]
    cfg = json.loads((before / "config.json").read_text())
    assert cfg["outer_folds"] == [f["test"] for f in folds]
    loaded = train.load_table(before, "none")
    assert [list(np.array(loaded["samples"])[f]) for f in loaded["folds"]] == [f["test"] for f in folds]
    assert loaded["R_fit"].shape == (5, 30, 1)
