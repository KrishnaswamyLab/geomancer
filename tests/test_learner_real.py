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
    assert rows[d]["reward_gnn"] == 0.0 and rows[d]["probs_gnn"] is None
    others = [r["reward_gnn"] for o, r in rows.items() if o != d]
    assert all(np.isfinite(others)) and all(0.0 <= r <= 1.0 for r in others)


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

    assert stored.shape == (50, 10) and np.array_equal(stored, again)
    assert meta == {"n_cells": 120, "n_points": 50, "points_source": "X_pca", "counts": False}
    assert computed.shape == (50, 10) and np.isfinite(computed).all()
    assert computed_meta["points_source"] == "computed" and computed_meta["counts"] is True


def test_real_table_runs_in_two_stages_into_the_files_train_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(table, "_loader", lambda: _fake_root(tmp_path / "root"))
    out = tmp_path / "table"
    names = "pca+mapper@default,raw+paga@default"

    table.main(["source=real", "stage=graphs", f"actions={names}", "max_cells=60", "n_pcs=10",
                "workers=1", f"out={out}"])

    graphs = [json.loads(line) for line in (out / "graphs.jsonl").read_text().splitlines()]
    # SCD-0999 is on disk but has no label, so it is not a sample.
    assert sorted({g["sample"] for g in graphs}) == ["SCD-0001", "SCD-0002", "SCD-0003"]
    assert len(graphs) == 6 and all(g["status"] == "ok" and g["payload"]["nodes"] for g in graphs)
    states = [json.loads(line) for line in (out / "states.jsonl").read_text().splitlines()]
    assert all(set(state.STATES) <= set(s) for s in states)
    assert not (out / "rewards.jsonl").exists()

    table.main(["source=real", "stage=rewards", "classifiers=rf", f"out={out}"])

    rows = [json.loads(line) for line in (out / "rewards.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and all(0.0 <= r["reward"] <= 1.0 and r["model"] == "rf" for r in rows)
    loaded = train.load_table(out, "input_geometry")
    assert loaded["R"].shape == (3, 2) and loaded["actions"] == names.split(",")
