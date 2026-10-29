import json

import numpy as np
import pytest

pytest.importorskip("kmapper")
pytest.importorskip("scshapebench")
pytest.importorskip("torch_geometric")

from geomancer.learner import rewards

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
