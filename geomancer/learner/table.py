"""Build the reward table: every action on every sample, once.

    python -m geomancer.learner.table limit=20 workers=4 out=outputs/tables/smoke

Writes ``rewards.jsonl`` (one row per sample x action), ``states.jsonl`` (one
row per sample, every state vector) and ``config.json`` under ``out``. A failed
action is a row with reward 0, status "error" and the error, never a gap.

    python -m geomancer.learner.table source=real stage=graphs out=outputs/tables/real_v1
    python -m geomancer.learner.table source=real stage=rewards out=outputs/tables/real_v1

The real reward trains a classifier across datasets, so it needs one action's
graphs on every dataset before any reward exists. The real table is built in two
stages: ``graphs`` writes ``graphs.jsonl`` (one row per sample x action: the
payload or the error), ``states.jsonl``, ``labels.json`` and ``config.json``;
``rewards`` reads them and writes ``rewards.jsonl``. ``stage=all`` runs both.
"""

import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from geomancer.learner import actions as actions_module
from geomancer.learner import rewards as rewards_module
from geomancer.learner import state as state_module

config = {
    "source": "synthetic",
    "root": None,  # scShapeBench directory; None means ${omics_data:}/scshapebench
    "limit": None,
    "actions": "all",  # or comma-separated action names
    "reward": "synthetic_graph",
    "workers": 4,
    "out": "outputs/tables/synthetic",
    # source=real only
    "stage": "all",  # graphs, rewards, or all
    "max_cells": 5000,  # seeded uniform subsample per dataset
    "n_pcs": 50,
    "points": "X_pca",  # stored obsm X_pca where it has n_pcs columns, else computed; or "computed"
    "classifiers": "rf",  # scShapeBench models; the first gives `reward`
}

SEED = 0
HVG = 2000


def override(config, argv):
    """nanoGPT-style ``key=value`` overrides; unknown keys are an error."""
    config = dict(config)
    for arg in argv:
        key, _, raw = arg.partition("=")
        if key not in config:
            raise KeyError(f"unknown config key {key!r}; known: {sorted(config)}")
        try:
            config[key] = json.loads(raw)
        except json.JSONDecodeError:
            config[key] = raw
    return config


def score_sample(sample, actions, reward):
    """One row per action for one sample."""
    rows = []
    for name, fn in actions.items():
        try:
            score = reward(fn(sample["points"]), sample)
            rows.append({"sample": sample["id"], "action": name, "status": "ok", **score})
        except Exception as exc:
            rows.append({"sample": sample["id"], "action": name, "reward": 0.0,
                         "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return rows


def sample_states(sample):
    return {"sample": sample["id"],
            **{name: fn(sample["points"]).tolist() for name, fn in state_module.STATES.items()}}


def _work(job):
    sample, names, reward_name = job
    chosen = {n: actions_module.ACTIONS[n] for n in names}
    try:
        return score_sample(sample, chosen, rewards_module.REWARDS[reward_name]), sample_states(sample)
    except Exception:
        traceback.print_exc()
        raise


def build_table(samples, actions, reward, out, workers=1):
    """Score every action on every sample; write rewards, states and config to ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reward_name = next(k for k, v in rewards_module.REWARDS.items() if v is reward)
    names = list(actions)
    (out / "config.json").write_text(json.dumps({"actions": names, "reward": reward_name}, indent=2))
    jobs = ((sample, names, reward_name) for sample in samples)
    started = time.time()
    with open(out / "rewards.jsonl", "w") as rewards_file, open(out / "states.jsonl", "w") as states_file:
        results = map(_work, jobs) if workers == 1 else ProcessPoolExecutor(workers).map(_work, jobs)
        for done, (rows, states) in enumerate(results, 1):
            rewards_file.writelines(json.dumps(row) + "\n" for row in rows)
            states_file.write(json.dumps(states) + "\n")
            errors = sum(row["status"] == "error" for row in rows)
            print(f"[{done}] sample={states['sample']} errors={errors} "
                  f"elapsed={time.time() - started:.0f}s", flush=True)
    return out


def _loader():
    try:
        from manylatents.singlecell.data import scshapebench
    except ImportError as exc:
        raise SystemExit(
            "the scShapeBench loader is in manylatents-omics after 0.1.3 "
            "(latent-reasoning-works/manylatents-omics#74); install that branch") from exc
    return scshapebench


def load_samples(source, root=None, limit=None):
    """Samples as ``{"id", "points", "graph", "meta"}`` from the manylatents-omics loader."""
    if source != "synthetic":
        raise ValueError(f"unknown source {source!r}; load_samples reads 'synthetic' only")
    scshapebench = _loader()
    ids = scshapebench.list_synthetic(root=root)[:limit]
    for sample_id in ids:
        sample = scshapebench.load_synthetic(sample_id, root=root)
        yield {"id": sample_id, "points": sample["points"], "graph": sample["graph"],
               "meta": sample["meta"]}


def real_points(path, max_cells, n_pcs, points="X_pca", seed=SEED):
    """One h5ad's cells, uniformly subsampled with a fixed seed, as principal components.

    The stored obsm X_pca is used when it has ``n_pcs`` columns. Otherwise (no
    X_pca, too few PCs, or points="computed") PCA is fit on the subsample: X is
    normalised and log1p'd if it holds integer counts, then cut to the most
    variable genes. Returns the points and how they were made.
    """
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        n = adata.n_obs
        idx = np.sort(np.random.default_rng(seed).choice(n, min(n, max_cells), replace=False))
        stored = adata.obsm.get("X_pca") if points == "X_pca" else None
        if stored is not None and stored.shape[1] >= n_pcs:
            return (np.asarray(stored[idx, :n_pcs], dtype=float),
                    {"n_cells": n, "n_points": len(idx), "points_source": "X_pca", "counts": False})
        X = adata[idx].to_memory().X
    finally:
        adata.file.close()
    return _computed_pca(X, n_pcs, seed), {"n_cells": n, "n_points": len(idx),
                                           "points_source": "computed", "counts": _is_counts(X)}


def _is_counts(X):
    import scipy.sparse as sp

    values = X.data if sp.issparse(X) else np.asarray(X).ravel()
    values = values[:10_000]
    return bool(values.size) and bool(np.all(values == np.round(values)))


def _computed_pca(X, n_pcs, seed):
    import anndata as ad
    import scanpy as sc
    from sklearn.decomposition import PCA

    adata = ad.AnnData(X.astype(np.float32))
    if _is_counts(X):
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    if adata.n_vars > HVG:
        sc.pp.highly_variable_genes(adata, n_top_genes=HVG)
        adata = adata[:, adata.var["highly_variable"]]
    dense = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    return PCA(n_components=min(n_pcs, *dense.shape), random_state=seed).fit_transform(dense)


def real_jobs(root=None, limit=None):
    """``(dataset_id, h5ad path)`` for the high_support datasets present locally, and their labels."""
    loader = _loader()
    labels = loader.load_shape_labels("high_support", root=root)
    ids = sorted(set(loader.list_real(root=root)) & set(labels))[:limit]
    return [(d, str(loader.real_path(d, root=root))) for d in ids], {d: labels[d] for d in ids}


def graph_rows(sample_id, points, actions):
    """One row per action: the payload, or the error."""
    rows = []
    for name, fn in actions.items():
        try:
            rows.append({"sample": sample_id, "action": name, "status": "ok", "payload": fn(points)})
        except Exception as exc:
            rows.append({"sample": sample_id, "action": name, "status": "error",
                         "error": f"{type(exc).__name__}: {exc}"})
    return rows


def _graph_work(job):
    sample_id, path, names, max_cells, n_pcs, points_kind = job
    try:
        points, meta = real_points(path, max_cells, n_pcs, points_kind)
        chosen = {n: actions_module.ACTIONS[n] for n in names}
        sample = {"id": sample_id, "points": points}
        return graph_rows(sample_id, points, chosen), {**sample_states(sample), "meta": meta}
    except Exception:
        traceback.print_exc()
        raise


def build_real_graphs(jobs, labels, actions, out, max_cells, n_pcs, points="X_pca", workers=1):
    """Stage 1: every action's graph on every dataset, plus states, labels and config."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(actions)
    (out / "config.json").write_text(json.dumps(
        {"actions": names, "source": "real", "max_cells": max_cells, "n_pcs": n_pcs,
         "points": points, "seed": SEED}, indent=2))
    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    work = [(d, path, names, max_cells, n_pcs, points) for d, path in jobs]
    started = time.time()
    with open(out / "graphs.jsonl", "w") as graphs_file, open(out / "states.jsonl", "w") as states_file:
        results = (map(_graph_work, work) if workers == 1
                   else ProcessPoolExecutor(workers).map(_graph_work, work))
        for done, (rows, states) in enumerate(results, 1):
            graphs_file.writelines(json.dumps(row) + "\n" for row in rows)
            states_file.write(json.dumps(states) + "\n")
            errors = sum(row["status"] == "error" for row in rows)
            print(f"[{done}/{len(work)}] sample={states['sample']} errors={errors} "
                  f"points={states['meta']['points_source']} elapsed={time.time() - started:.0f}s",
                  flush=True)
    return out


def build_real_rewards(out, classifiers="rf"):
    """Stage 2: rewards.jsonl from graphs.jsonl, one cross-dataset classifier per action."""
    out = Path(out)
    cfg = json.loads((out / "config.json").read_text())
    labels = json.loads((out / "labels.json").read_text())
    by_action = {name: {} for name in cfg["actions"]}
    with open(out / "graphs.jsonl") as f:
        for line in f:
            row = json.loads(line)
            by_action[row["action"]][row["sample"]] = row
    models = classifiers.split(",")
    started = time.time()
    with open(out / "rewards.jsonl", "w") as rewards_file:
        for done, (name, rows) in enumerate(by_action.items(), 1):
            scored = rewards_module.real_label(
                {s: r.get("payload") if r["status"] == "ok" else None for s, r in rows.items()},
                labels, models=models)
            for sample_id, row in rows.items():
                score = scored[sample_id]
                if row["status"] == "error":
                    score = {**score, "error": row["error"]}
                rewards_file.write(json.dumps({"sample": sample_id, "action": name, **score}) + "\n")
            mean = np.mean([scored[s]["reward"] for s in rows]) if rows else float("nan")
            print(f"[{done}/{len(by_action)}] action={name} mean_reward={mean:.3f} "
                  f"elapsed={time.time() - started:.0f}s", flush=True)
    (out / "config.json").write_text(json.dumps(
        {**cfg, "reward": "real_label", "classifiers": models}, indent=2))
    return out


def main(argv=None):
    cfg = override(config, sys.argv[1:] if argv is None else argv)
    names = (list(actions_module.ACTIONS) if cfg["actions"] == "all"
             else cfg["actions"].split(","))
    chosen = {n: actions_module.ACTIONS[n] for n in names}
    if cfg["source"] == "real":
        if cfg["stage"] not in ("graphs", "rewards", "all"):
            raise ValueError(f"unknown stage {cfg['stage']!r}; expected graphs, rewards or all")
        out = Path(cfg["out"])
        if cfg["stage"] in ("graphs", "all"):
            jobs, labels = real_jobs(cfg["root"], cfg["limit"])
            build_real_graphs(jobs, labels, chosen, out, cfg["max_cells"], cfg["n_pcs"],
                              cfg["points"], cfg["workers"])
        if cfg["stage"] in ("rewards", "all"):
            build_real_rewards(out, cfg["classifiers"])
    else:
        out = build_table(load_samples(cfg["source"], cfg["root"], cfg["limit"]), chosen,
                          rewards_module.REWARDS[cfg["reward"]], cfg["out"], cfg["workers"])
    run_file = f"run_{cfg['stage']}.json" if cfg["source"] == "real" else "run.json"
    (out / run_file).write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
