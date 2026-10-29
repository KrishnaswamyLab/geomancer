"""Rewards: score a predicted graph payload against one sample's truth.

Each function returns a dict whose ``reward`` is in [0, 1], plus the scores it
was taken from, so the table keeps them for later analysis. `synthetic_graph`
scores one sample alone. `real_label` needs one action's graphs on every dataset,
because its classifier is trained across datasets.
"""

import contextlib
import csv
import io
import tempfile
from pathlib import Path

import numpy as np

# scShapeBench's fixed real-track splits: splits/folds_v1.json in
# KrishnaswamyLab/scShapeBench at d81963f1f950d86c16ee5171987994898029239c, the
# commit pyproject.toml pins. The wheel does not ship it, so it is copied here unchanged.
FOLDS = Path(__file__).with_name("folds_v1.json")


def synthetic_graph(pred, sample):
    """scShapeBench's full-graph-recovery evaluator; reward = its primary metric.

    The primary metric is the relaxed topology score. Persistence similarity and
    graph edit distance are kept alongside it.
    """
    # The evaluator filters sys.path entries for other Python versions and sets
    # PYTHONNOUSERSITE when it is imported; import it only when scoring.
    from scshapebench.synthetic import evaluation as ev

    truth_graph, truth_positions, truth_reduced = ev.build_graph_from_payload(
        sample["graph"], positions_required_for_persistence=True)
    pred_graph, pred_positions, pred_reduced = ev.build_graph_from_payload(
        pred, positions_required_for_persistence=True)
    result = ev.evaluate_full_graph_recovery(
        truth_graph=truth_graph,
        pred_graph=pred_graph,
        truth_positions=truth_positions,
        pred_positions=pred_positions,
        observed_points=sample["points"],
        truth_already_reduced_multigraph=truth_reduced,
        pred_already_reduced_multigraph=pred_reduced,
    )
    headline = result["headline_scores"]
    return {
        "reward": headline["topology_score"],
        **headline,
        "graph_edit_distance": result["graph_edit_distance"],
        "graph_edit_distance_exact": result["graph_edit_distance_exact"],
    }


def payloads_to_graphs(payloads):
    """``{dataset_id: payload}`` as scShapeBench's own loader reads them from a submission.

    The submission format has only x, y per node, so nodes.csv holds the first two
    coordinates of each node position. Edge lengths are Euclidean in the full point
    space; the loader adds both directions and divides by the longest edge.
    """
    from scshapebench.evaluator.loader import load_graphs

    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        for dataset_id, payload in payloads.items():
            folder = Path(tmp) / dataset_id
            folder.mkdir()
            index = {node: i for i, node in enumerate(payload["nodes"])}
            positions = np.array([payload["node_positions"][str(node)] for node in payload["nodes"]],
                                 dtype=float).reshape(len(index), -1)
            xy = np.pad(positions[:, :2], ((0, 0), (0, max(0, 2 - positions.shape[1]))))
            with open(folder / "nodes.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "x", "y"])
                writer.writerows([i, float(x), float(y)] for i, (x, y) in enumerate(xy))
            with open(folder / "edges.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["source", "target", "length"])
                for u, v in payload["edges"]:
                    length = float(np.linalg.norm(positions[index[u]] - positions[index[v]]))
                    writer.writerow([index[u], index[v], length])
        graphs, _ = load_graphs(Path(tmp))
    if set(graphs) != set(payloads):
        # The loader renames directories to SCD-%04d by their trailing digits.
        raise ValueError(f"dataset ids must be SCD-XXXX; got {sorted(set(payloads) - set(graphs))[:5]}")
    return graphs


def real_label(payloads, labels, models=("rf",), folds=None, device="cpu"):
    """Out-of-fold probability that scShapeBench's fixed classifier gives each dataset its labels.

    ``payloads`` maps a dataset id to one action's graph, or None when the action
    failed; ``labels`` maps it to ``{class: 0 or 1}``. On each fold of ``folds``
    (default folds_v1), each model in ``models`` is fit on that fold's train graphs
    and predicts its test graphs. This is the loop of scShapeBench's
    run_real_submission (evaluator/runner.py), with the same features, models and
    seeds, but it keeps the probabilities that the runner only scores.

    reward = mean over the 4 classes of y*p + (1-y)*(1-p). ``reward`` and ``probs``
    are the first model's; every model also has ``reward_<name>`` and ``probs_<name>``.
    Returns one row per dataset in ``payloads``.
    """
    import torch
    from scshapebench.datasets.labels import CLASS_ORDER
    from scshapebench.evaluator.runner import filter_split_ids, load_folds, set_model_seed
    from scshapebench.evaluator.validator import validate_submission
    from scshapebench.features.graph_stats import extract_features as graph_stats
    from scshapebench.features.persistence import extract_features as persistence
    from scshapebench.models import get_models

    models = list(models)
    known = [m.name for m in get_models(torch.device(device))]
    if unknown := sorted(set(models) - set(known)):
        raise ValueError(f"unknown models {unknown}; scShapeBench has {known}")
    folds = load_folds(FOLDS) if folds is None else folds
    graphs = payloads_to_graphs({d: p for d, p in payloads.items() if p is not None and d in labels})
    if graphs:
        with contextlib.redirect_stdout(io.StringIO()):
            validate_submission(graphs)
    Y = {d: np.array([labels[d][c] for c in CLASS_ORDER], dtype=np.float32) for d in graphs}
    extractors = {"graph_stats": graph_stats, "persistence": persistence}
    probs, fold_of = {m: {} for m in models}, {}
    for fold_id, split in enumerate(folds):
        with contextlib.redirect_stdout(io.StringIO()):
            train = filter_split_ids(split["train"], set(graphs), "train")
            test = filter_split_ids(split["test"], set(graphs), "test")
        fold_of.update(dict.fromkeys(test, fold_id))
        features = {}
        # model_idx counts every scShapeBench model, so the seeds match the benchmark's.
        for model_idx, model in enumerate(get_models(torch.device(device))):
            if model.name not in models or not test:
                continue
            fit = train
            if model.name == "gnn":
                # The GNN standardises node x per graph; one node gives std NaN, and
                # BatchNorm in training would spread it to the whole batch.
                fit = [d for d in train if graphs[d].num_nodes > 1]
            if not fit:
                continue
            src = model.features_source
            if src and src not in features:
                features[src] = extractors[src]([graphs[d] for d in train], [graphs[d] for d in test])
            # Only the GNN drops training graphs, and it reads no feature matrix.
            X_train, X_test = features[src] if src else (None, None)
            set_model_seed(10_000 + fold_id * 100 + model_idx)
            model.fit(graphs=[graphs[d] for d in fit], labels=np.stack([Y[d] for d in fit]),
                      features=X_train)
            _, p = model.predict(graphs=[graphs[d] for d in test], features=X_test)
            probs[model.name].update(zip(test, np.asarray(p, dtype=float)))

    rows = {}
    for d, payload in payloads.items():
        if payload is None or d not in labels or d not in fold_of:
            reason = ("the action failed" if payload is None else
                      "no label" if d not in labels else "in no test fold")
            rows[d] = {"reward": 0.0, "status": "error", "error": reason}
            continue
        y = Y[d]
        row = {"status": "ok", "model": models[0], "fold": fold_of[d],
               "label": dict(zip(CLASS_ORDER, y.astype(int).tolist()))}
        for m in models:
            p = probs[m].get(d)
            finite = p is not None and np.isfinite(p).all()
            row[f"reward_{m}"] = float(np.mean(y * p + (1 - y) * (1 - p))) if finite else 0.0
            row[f"probs_{m}"] = dict(zip(CLASS_ORDER, p.tolist())) if finite else None
        rows[d] = {"reward": row[f"reward_{models[0]}"], "probs": row[f"probs_{models[0]}"], **row}
    return rows


REWARDS = {"synthetic_graph": synthetic_graph}
