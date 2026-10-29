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
``resume=true`` keeps the datasets that stage 1 already wrote.

Each dataset's real reward comes from a classifier fit on the labels of the other
folds_v1 folds, so a policy fit on the rewards of some datasets has seen the labels
of the rest. ``rewards`` therefore also writes ``rewards_nested.jsonl``: for each
outer fold k, the rewards of the datasets outside k, recomputed with fold k left
out. train.py fits on those and scores on ``rewards.jsonl`` (see train.load_table).
"""

import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
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
    "resume": False,  # stage 1: keep datasets already in states.jsonl
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
            # X is not read here, so whether it holds counts is not known.
            return (np.asarray(stored[idx, :n_pcs], dtype=float),
                    {"n_cells": n, "n_points": len(idx), "points_source": "X_pca", "counts": None})
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
        states = {**sample_states({"id": sample_id, "points": points}), "meta": meta}
    except Exception as exc:
        # The dataset itself failed (unreadable file, PCA, states): every action is an error.
        traceback.print_exc()
        error = f"{type(exc).__name__}: {exc}"
        return ([{"sample": sample_id, "action": n, "status": "error", "error": error} for n in names],
                {"sample": sample_id, "error": error})
    chosen = {n: actions_module.ACTIONS[n] for n in names}
    return graph_rows(sample_id, points, chosen), states


def _graph_results(work, workers):
    """``_graph_work`` on each job, yielded as they finish; a dead worker loses only its own jobs."""
    if workers == 1:
        yield from map(_graph_work, work)
        return
    lost = []
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(_graph_work, job): job[0] for job in work}
        for future in as_completed(futures):
            try:
                result = future.result()
            except BrokenProcessPool:
                lost.append(futures[future])
                continue
            yield result
    if lost:
        raise BrokenProcessPool(
            f"a worker died (out of memory?); {len(lost)} datasets have no rows: {sorted(lost)[:10]}. "
            "The rows written are kept; rerun with resume=true.")


def _read_jsonl(path):
    # A run killed while writing can leave a partial last line; it is dropped.
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _resume(out, run_config):
    """The datasets a previous stage 1 finished; the files are cut back to exactly those."""
    old = json.loads((out / "config.json").read_text())
    if changed := sorted(k for k in run_config if old.get(k) != run_config[k]):
        raise ValueError(f"cannot resume {out}: {changed} differ from the run that wrote it")
    states = _read_jsonl(out / "states.jsonl")
    # A dataset is done when its states row is written, after its graph rows; failed
    # datasets are tried again.
    done = {s["sample"] for s in states if "error" not in s}
    graphs = [g for g in _read_jsonl(out / "graphs.jsonl") if g["sample"] in done]
    (out / "graphs.jsonl").write_text("".join(json.dumps(g) + "\n" for g in graphs))
    (out / "states.jsonl").write_text("".join(json.dumps(s) + "\n" for s in states if s["sample"] in done))
    return done


def build_real_graphs(jobs, labels, actions, out, max_cells, n_pcs, points="X_pca", workers=1,
                      resume=False):
    """Stage 1: every action's graph on every dataset, plus states, labels and config."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(actions)
    run_config = {"actions": names, "source": "real", "max_cells": max_cells, "n_pcs": n_pcs,
                  "points": points, "seed": SEED}
    resume = resume and (out / "states.jsonl").exists()
    done = _resume(out, run_config) if resume else set()
    (out / "config.json").write_text(json.dumps(run_config, indent=2))
    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    work = [(d, path, names, max_cells, n_pcs, points) for d, path in jobs if d not in done]
    if done:
        print(f"resuming: {len(done)} datasets done, {len(work)} to go", flush=True)
    started = time.time()
    mode = "a" if resume else "w"
    with open(out / "graphs.jsonl", mode) as graphs_file, open(out / "states.jsonl", mode) as states_file:
        for n, (rows, states) in enumerate(_graph_results(work, workers), 1):
            graphs_file.writelines(json.dumps(row) + "\n" for row in rows)
            graphs_file.flush()
            states_file.write(json.dumps(states) + "\n")
            states_file.flush()
            errors = sum(row["status"] == "error" for row in rows)
            source = states["meta"]["points_source"] if "meta" in states else "failed"
            print(f"[{n}/{len(work)}] sample={states['sample']} errors={errors} "
                  f"points={source} elapsed={time.time() - started:.0f}s", flush=True)
    return out


def build_real_rewards(out, classifiers="rf", folds=None):
    """Stage 2: rewards.jsonl and rewards_nested.jsonl from graphs.jsonl, per action.

    ``rewards.jsonl`` is out of fold on ``folds`` (default folds_v1). For each fold
    k, ``rewards_nested.jsonl`` holds the rewards of the datasets outside k, from
    classifiers fit without fold k (its other folds are the inner folds).
    """
    out = Path(out)
    cfg = json.loads((out / "config.json").read_text())
    labels = json.loads((out / "labels.json").read_text())
    folds = rewards_module.folds_v1() if folds is None else folds
    by_action = {name: {} for name in cfg["actions"]}
    for row in _read_jsonl(out / "graphs.jsonl"):
        by_action[row["action"]][row["sample"]] = row
    samples = {s for rows in by_action.values() for s in rows}
    models = classifiers.split(",") if isinstance(classifiers, str) else list(classifiers)
    started = time.time()
    with open(out / "rewards.jsonl", "w") as rewards_file, \
            open(out / "rewards_nested.jsonl", "w") as nested_file:
        for done, (name, rows) in enumerate(by_action.items(), 1):
            payloads = {s: r.get("payload") if r["status"] == "ok" else None for s, r in rows.items()}
            scored = rewards_module.real_label(payloads, labels, models=models, folds=folds)
            for sample_id, row in rows.items():
                score = scored[sample_id]
                if row["status"] == "error":
                    score = {**score, "error": row["error"]}
                rewards_file.write(json.dumps({"sample": sample_id, "action": name, **score}) + "\n")
            for k, outer in enumerate(folds):
                held_out = set(outer["test"])
                inner = rewards_module.real_label(
                    {s: p for s, p in payloads.items() if s not in held_out}, labels, models=models,
                    folds=[f for j, f in enumerate(folds) if j != k])
                nested_file.writelines(
                    json.dumps({"sample": s, "action": name, "outer_fold": k, **score}) + "\n"
                    for s, score in inner.items())
            mean = np.mean([scored[s]["reward"] for s in rows]) if rows else float("nan")
            print(f"[{done}/{len(by_action)}] action={name} mean_reward={mean:.3f} "
                  f"elapsed={time.time() - started:.0f}s", flush=True)
    outer_folds = [[s for s in f["test"] if s in samples] for f in folds]
    (out / "config.json").write_text(json.dumps(
        {**cfg, "reward": "real_label", "classifiers": models, "outer_folds": outer_folds}, indent=2))
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
                              cfg["points"], cfg["workers"], cfg["resume"])
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
