"""Score the grid's graphs after the runs: a reduction over the ledgers, by run_id.

    python -m geomancer.learner.score grid=outputs/ssb source=synthetic workers=8
    python -m geomancer.learner.score grid=outputs/ssb source=real classifiers=rf,stats_svm,gnn

The truth never enters a run: the ssb_graph tool extracts a graph and nothing
else. This file reads every row of every ledger under ``<grid>/ledgers`` with
``manyruns.store.read`` (its only manyruns call), checks each graph run, loads its
``graphs/<run_id>.json``, and scores it:

- synthetic: scShapeBench's full-graph-recovery evaluator (rewards.synthetic_graph)
  against the sample's truth graph, read from the ``.npz`` that convert.py's
  manifest names, after its bytes are checked against the manifest's sha256.
- real: rewards.real_label, whose classifier is trained across datasets, so it is
  one reduction per (action, seed) over the chosen row of every dataset. For each
  outer fold k it is run again with fold k left out: the nested rewards that
  train.py fits policies on, so no test-fold label reaches a policy.

A graph run is scored only when its row is ok and complete; its three sample names
agree (dataset_name, extra.geomancer.sample, ssb_graph.sample); its points are the
manifest's; its graph file hashes to the graph_sha256 the row carries (driver and
adapter); and it joins to what it was built on: a branched run's parent is an ok
and complete embedding run of the same sample, seed and embedding whose ``emb``
artifact hashes to the graph run's input_sha256, and a raw run's input is the
points. Any other row is a score row with that status and reward None. It is
counted, never imputed or dropped.

Writes ``<grid>/scores/<source>.jsonl`` (one row per graph run read; for real, one
more per outer fold) and ``<source>.json``: every run_id read, the sha256 of every
ledger, the truth's sha256, the code revision and the counts. table.py joins these
to the rows by run_id. Nothing here writes a ledger row.
"""

import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from geomancer.learner import provenance
from geomancer.learner.table import override

config = {
    "grid": "outputs/ssb",  # drive.py's out: ledgers/, graphs/, cells/
    "source": "synthetic",  # synthetic or real
    "inputs": None,  # convert.py's folder; None means <scShapeBench root>/grid_inputs/<source>
    "root": None,  # real: the scShapeBench root the labels are read from
    "labels": "high_support",
    "classifiers": "rf",  # real: scShapeBench models; the first gives `reward`
    "workers": 1,
    "out": None,  # None means <grid>/scores
}

VERSIONS = ("scshapebench", "manylatents-omics", "numpy", "networkx", "torch", "scikit-learn")


# ── reading the ledgers ──────────────────────────────────────────────────────

def ledger_files(grid):
    """Every shard ledger file under ``<grid>/ledgers``, sorted."""
    return sorted((Path(grid) / "ledgers").glob("*/index.jsonl"))


def read_ledgers(grid):
    """Every row of every ledger, in ledger then append order."""
    from manyruns import store

    return [row for path in ledger_files(grid) for row in store.read(path.parent)]


def ledger_hashes(grid):
    """``{ledger path relative to grid: sha256}``, so a score or table names the bytes it read."""
    return {str(p.relative_to(grid)): provenance.sha256_file(p) for p in ledger_files(grid)}


def geo(row):
    """The driver's fields of a row (``row.extra.geomancer``)."""
    return (row.get("extra") or {}).get("geomancer") or {}


def finished(row):
    return bool(row.get("ok") and row.get("complete"))


def cell_key(row):
    g = geo(row)
    return (g.get("sample"), g.get("seed"), g.get("action"))


def cells(rows):
    """``({(sample, seed, action): row}, superseded run_ids)`` for one kind of row.

    A cell can have several rows: a failed cell is retried by the next run of the
    driver. The cell's row is its latest ok and complete row, else its latest row;
    the others are superseded, and listed so that they are counted.
    """
    chosen, superseded = {}, []
    for order, row in enumerate(rows):
        key = cell_key(row)
        rank = (finished(row), row.get("at") or "", order)
        if key in chosen:
            if rank > chosen[key][0]:
                superseded.append(chosen[key][1]["run_id"])
                chosen[key] = (rank, row)
            else:
                superseded.append(row["run_id"])
        else:
            chosen[key] = (rank, row)
    return {key: row for key, (_, row) in chosen.items()}, superseded


def digest(value):
    """sha256 of a JSON value in canonical form (sorted keys, no spaces)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ── checking one graph run ───────────────────────────────────────────────────

def _artifact_sha(path, cache):
    if path not in cache:
        cache[path] = provenance.sha256_file(path) if path and Path(path).is_file() else None
    return cache[path]


def check(row, by_id, grid, points_sha256=None, cache=None):
    """``(status, detail, payload)`` for one graph row: "ok" and its graph, or why not.

    ``points_sha256`` is the manifest's sha256 of the sample's points, when known.
    """
    cache = {} if cache is None else cache
    g, vec = geo(row), row.get("g_vector") or {}
    if not finished(row):
        return g.get("status") or "failed", g.get("detail") or "the run is not ok and complete", None
    names = {"dataset_name": row.get("dataset_name"), "extra.geomancer.sample": g.get("sample"),
             "ssb_graph.sample": vec.get("ssb_graph.sample")}
    if len(set(names.values())) != 1:
        return "sample_disagrees", json.dumps(names), None
    if points_sha256 is not None and {g.get("points_sha256"), vec.get("ssb_graph.points_sha256")} != {points_sha256}:
        return "points_disagree", f"the row's points are not the manifest's {points_sha256}", None
    parent_id = (row.get("parent") or {}).get("run_id") or g.get("embedding_run_id")
    if g.get("embedding") == "raw":
        if parent_id is not None:
            return "join_failed", f"a raw-arm run has a parent {parent_id}", None
        if vec.get("ssb_graph.input_sha256") != g.get("points_sha256"):
            return "join_failed", "the raw-arm input sha256 is not the points' sha256", None
    else:
        parent = by_id.get(parent_id)
        if parent is None:
            return "join_failed", f"parent {parent_id} is not in the ledgers", None
        pg = geo(parent)
        if not finished(parent) or pg.get("kind") != "embedding":
            return "join_failed", f"parent {parent_id} is not an ok and complete embedding run", None
        if (pg.get("sample"), pg.get("seed"), pg.get("embedding")) != (g.get("sample"), g.get("seed"), g.get("embedding")):
            return "join_failed", f"parent {parent_id} is another sample, seed or embedding", None
        emb = next((s.get("artifacts", {}).get("emb") for s in parent.get("steps") or []
                    if (s.get("artifacts") or {}).get("emb")), None)
        emb_sha = _artifact_sha(emb, cache)
        if emb_sha is None:
            return "join_failed", f"parent {parent_id}'s emb artifact {emb} is not on disk", None
        if emb_sha != vec.get("ssb_graph.input_sha256"):
            return "join_failed", f"parent {parent_id}'s emb sha256 is not this run's input_sha256", None
    graph = g.get("graph") or {}
    path = Path(grid) / graph.get("path", "")
    if not graph.get("path") or not path.is_file():
        return "graph_missing", f"no graph file {graph.get('path')}", None
    actual = provenance.sha256_file(path)
    if not actual == graph.get("sha256") == vec.get("ssb_graph.graph_sha256"):
        return "graph_sha256_mismatch", f"{path.name} has sha256 {actual}", None
    return "ok", None, json.loads(path.read_text())


def _base(row, status, detail, kind, outer_fold=None):
    g = geo(row)
    return {"run_id": row["run_id"], "reward_kind": kind, "sample": g.get("sample"),
            "seed": g.get("seed"), "action": g.get("action"), "outer_fold": outer_fold,
            "embedding_run_id": (row.get("parent") or {}).get("run_id") or g.get("embedding_run_id"),
            "graph_sha256": (g.get("graph") or {}).get("sha256"), "status": status,
            "reward": None, "detail": detail}


def _manifest(grid, source, rows, scores, seconds, **fields):
    counts = {}
    for s in scores:
        counts[s["status"]] = counts.get(s["status"], 0) + 1
    return {"source": source, "reward": "synthetic_graph" if source == "synthetic" else "real_label",
            "ledgers": ledger_hashes(grid), "run_ids": [r["run_id"] for r in rows],
            "n_rows_read": len(rows), "counts": counts,
            "code_revision": provenance.geomancer_revision(), "versions": provenance.versions(VERSIONS),
            "seconds": round(seconds, 2), **fields}


def _graph_rows(rows, source):
    return [r for r in rows if geo(r).get("kind") == "graph" and geo(r).get("source") == source]


# ── synthetic ────────────────────────────────────────────────────────────────

def _score_sample(job):
    """Score one sample's checked graphs against its truth. Runs in a worker."""
    entry, items = job
    from manylatents.singlecell.data import scshapebench

    from geomancer.learner import rewards

    npz = Path(entry["npz"])
    sample = scshapebench.load_synthetic(int(npz.stem.removeprefix("sample_")), root=npz.parents[2])
    out = []
    for base, payload in items:
        try:
            out.append({**base, **rewards.synthetic_graph(payload, sample)})
        except Exception as exc:
            out.append({**base, "status": "score_error", "detail": f"{type(exc).__name__}: {exc}"})
    return out


def score_synthetic(grid, inputs, workers=1):
    """``(score rows, manifest)`` for every synthetic graph row in the ledgers."""
    started = time.time()
    rows = read_ledgers(grid)
    by_id = {r["run_id"]: r for r in rows}
    entries = json.loads((Path(inputs) / "manifest.json").read_text())["samples"]
    truth, cache, scores, jobs = {}, {}, [], {}
    for row in _graph_rows(rows, "synthetic"):
        name = geo(row).get("sample")
        entry = entries.get(name) or {}
        if name not in truth and entry.get("status") == "ok":
            actual = provenance.sha256_file(entry["npz"])
            if actual != entry["npz_sha256"]:
                raise ValueError(f"{entry['npz']} has sha256 {actual}, not the manifest's {entry['npz_sha256']}")
            truth[name] = actual
        if name not in truth:
            scores.append(_base(row, "no_truth", f"{name} is not an ok sample of {inputs}/manifest.json",
                                "synthetic_graph"))
            continue
        status, detail, payload = check(row, by_id, grid, entry["npy_sha256"], cache)
        base = {**_base(row, status, detail, "synthetic_graph"), "truth_sha256": truth[name]}
        if status != "ok":
            scores.append(base)
        else:
            jobs.setdefault(name, (entry, []))[1].append((base, payload))
    work = list(jobs.values())
    results = map(_score_sample, work) if workers == 1 else ProcessPoolExecutor(workers).map(_score_sample, work)
    for n, done in enumerate(results, 1):
        scores.extend(done)
        if n % 50 == 0:
            print(f"[{n}/{len(work)}] samples scored, elapsed={time.time() - started:.0f}s", flush=True)
    order = {r["run_id"]: i for i, r in enumerate(rows)}
    scores.sort(key=lambda s: order[s["run_id"]])
    meta = _manifest(grid, "synthetic", rows, scores, time.time() - started,
                     inputs=str(inputs), inputs_manifest_sha256=provenance.sha256_file(Path(inputs) / "manifest.json"),
                     truth=truth)
    return scores, meta


# ── real ─────────────────────────────────────────────────────────────────────

def _real_reduction(chosen, payloads, labels, models, folds, outer_fold, kind="real_label"):
    from geomancer.learner import rewards

    scored = rewards.real_label(payloads, labels, models=models, folds=folds)
    out = []
    for sample, (row, status, detail) in chosen.items():
        base = _base(row, status, detail, kind, outer_fold)
        base["truth_sha256"] = digest(labels[sample]) if sample in labels else None
        if status != "ok":
            out.append(base)
            continue
        s = scored[sample]
        row_out = {**base, **s, "run_id": row["run_id"], "outer_fold": outer_fold}
        if s["status"] != "ok":
            # real_label scores a failed dataset 0; here it is that status with no reward.
            row_out.update(status="score_error", reward=None, detail=s.get("error"))
        row_out.pop("error", None)
        out.append(row_out)
    return out


def score_real(grid, labels, folds=None, models=("rf",), labels_file=None, inputs=None):
    """``(score rows, manifest)``: real_label per (action, seed), and per outer fold left out.

    With ``inputs`` (convert.py's real folder), each row's points are checked against
    its manifest entry.
    """
    from geomancer.learner import rewards

    started = time.time()
    rows = read_ledgers(grid)
    by_id = {r["run_id"]: r for r in rows}
    folds_sha = provenance.sha256_file(rewards.FOLDS) if folds is None else digest(folds)
    folds = rewards.folds_v1() if folds is None else folds
    models = list(models)
    chosen, superseded = cells(_graph_rows(rows, "real"))
    scores = [_base(by_id[r], "superseded", "a later row of this cell is scored", "real_label")
              for r in superseded]
    entries = json.loads((Path(inputs) / "manifest.json").read_text())["samples"] if inputs else None
    groups, cache = {}, {}
    for (sample, seed, action), row in sorted(chosen.items(), key=lambda kv: tuple(map(str, kv[0]))):
        if entries is not None and (entries.get(sample) or {}).get("status") != "ok":
            status, detail, payload = "no_input", f"{sample} is not an ok dataset of {inputs}", None
        else:
            points = entries[sample]["npy_sha256"] if entries is not None else None
            status, detail, payload = check(row, by_id, grid, points, cache)
        groups.setdefault((action, seed), {})[sample] = (row, status, detail, payload)
    reductions = []
    for (action, seed), group in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        meta_rows = {s: (row, status, detail) for s, (row, status, detail, _) in group.items()}
        payloads = {s: p for s, (_, status, _, p) in group.items() if status == "ok"}
        # outer_fold None: out of fold on every fold. k: fold k left out, for the nested fit.
        plans = [(None, set(), folds)] + [
            (k, set(f["test"]), [g for j, g in enumerate(folds) if j != k]) for k, f in enumerate(folds)]
        for k, held_out, inner in plans:
            kept = {s: v for s, v in meta_rows.items() if s not in held_out}
            scores.extend(_real_reduction(kept, {s: p for s, p in payloads.items() if s not in held_out},
                                          labels, models, inner, k))
            reductions.append({"action": action, "seed": seed, "outer_fold": k,
                               "run_ids": sorted(row["run_id"] for row, _, _ in kept.values())})
        print(f"action={action} seed={seed}: {len(payloads)}/{len(group)} graphs scored, "
              f"elapsed={time.time() - started:.0f}s", flush=True)
    present = {s for group in groups.values() for s in group}
    truth = {"labels_sha256": digest(labels),
             "labels_file": str(labels_file) if labels_file else None,
             "labels_file_sha256": provenance.sha256_file(labels_file) if labels_file else None,
             "folds_sha256": folds_sha}
    meta = _manifest(grid, "real", rows, scores, time.time() - started, classifiers=models,
                     truth=truth, reductions=reductions, inputs=str(inputs) if inputs else None,
                     outer_folds=[[s for s in f["test"] if s in present] for f in folds])
    return scores, meta


# ── writing ──────────────────────────────────────────────────────────────────

def write(grid, source, scores, meta, out=None):
    """``<out>/<source>.jsonl`` and ``<out>/<source>.json``; returns the .jsonl path."""
    out = Path(out) if out else Path(grid) / "scores"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{source}.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in scores))
    (out / f"{source}.json").write_text(json.dumps({**meta, "scores_sha256": provenance.sha256_file(path)},
                                                   indent=2))
    return path


def main(argv=None):
    cfg = override(config, sys.argv[1:] if argv is None else argv)
    grid = Path(cfg["grid"])
    if cfg["source"] == "synthetic":
        inputs = cfg["inputs"]
        if inputs is None:
            from manylatents.singlecell.data import scshapebench

            inputs = Path(scshapebench.scshapebench_root(cfg["root"])) / "grid_inputs" / "synthetic"
        scores, meta = score_synthetic(grid, inputs, int(cfg["workers"]))
    elif cfg["source"] == "real":
        from manylatents.singlecell.data import scshapebench

        root = scshapebench.scshapebench_root(cfg["root"])
        labels = scshapebench.load_shape_labels(cfg["labels"], root=root)
        labels_file = root / "labels" / ("annotations.parquet" if cfg["labels"] == "high_support"
                                         else "labels.parquet")
        models = cfg["classifiers"].split(",") if isinstance(cfg["classifiers"], str) else cfg["classifiers"]
        inputs = Path(cfg["inputs"]) if cfg["inputs"] else root / "grid_inputs" / "real"
        scores, meta = score_real(grid, labels, models=models, labels_file=labels_file, inputs=inputs)
        meta["labels"] = cfg["labels"]
    else:
        raise ValueError(f"unknown source {cfg['source']!r}; expected synthetic or real")
    path = write(grid, cfg["source"], scores, {**meta, "config": cfg}, cfg["out"])
    print(f"{cfg['source']}: {meta['n_rows_read']} rows read, {len(scores)} scores {meta['counts']} "
          f"in {meta['seconds']}s -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
