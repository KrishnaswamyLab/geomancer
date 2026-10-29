"""Drive the scShapeBench action grid through manyruns: one run record per cell.

    python -m geomancer.learner.drive source=synthetic shard=0 of=100 seeds=0,1,2
    python -m geomancer.learner.drive source=real seeds=0 workers=4 out=outputs/ssb_real

A cell is one (sample, seed, action). For each (sample, seed, embedding) the
driver opens a manyruns Session with its own out_dir, applies the embedding's
latent step (configs/recipe/ssb_embed_<embedding>.yaml), closes it and appends
the embedding row. Then it applies the ssb_graph tool on six ``branch(0)``
children, one per (extractor, scale), so each graph row has the native
``parent = {run_id, index: 0}``. The raw arm is six single-step sessions on the
points, with no parent. Each sample also gets one ssb_state run.

The branches of one embedding share one tool workdir (Session.branch keeps the
parent's out_dir), so they run SERIALLY: after each, graph.json is copied to
``<out>/graphs/<run_id>.json`` and checked against the ``graph_sha256`` scalar
the adapter emitted, before the next branch starts. Parallelism is across
(sample, seed, arm) units only, each with its own out_dir.

Nothing here scores. The truth never enters a run; score.py reads the ledgers
afterwards. Rows are written only by ``manyruns.store.append``, with the
driver's own fields under ``extra.geomancer``. The grid is exhaustive, so it
writes no decision rows and makes no choices.

A cell that fails or is refused is a row with its status, never a gap: a
failed embedding is kept, and each of its graph cells is a ``not_run`` row
naming it. Resume skips cells that already have an ok and complete row for this
geomancer commit.

Every manyruns import and call of the learner lives in this file, so the move
to manyruns.action_space (0.1.1) touches only this file. manyruns.session loads
manyruns.pipeline transitively; geomancer never imports it directly.
"""

import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from importlib import metadata
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import yaml

from geomancer.learner import actions as actions_module
from geomancer.learner import provenance
from geomancer.learner.table import override

config = {
    "source": "synthetic",  # synthetic or real (convert.py's outputs)
    "inputs": None,  # None means <scShapeBench root>/grid_inputs/<source>
    "out": "outputs/ssb",
    "seeds": [0, 1, 2],  # or "0,1,2"
    "actions": "all",  # or comma-separated action names
    "samples": None,  # or comma-separated sample names
    "limit": None,
    "shard": 0,
    "of": 1,
    "workers": 1,
    "state": True,  # one ssb_state run per sample
}

RECIPES = Path(__file__).resolve().parents[1] / "configs" / "recipe"
MODALITY = {"synthetic": "synthetic", "real": "scrna"}
CALLER = "geomancer.learner.drive"
ENGINE = "manylatents"
TOOL_WORKDIR = "tool-ssb_graph"


# ── environment ──────────────────────────────────────────────────────────────

def setup_env():
    """Merge geomancer's lab configs into manyruns' and name the caller, before any tool step.

    The adapters inherit this process's environment (manyruns starts them without
    its own ``env=``), so ``GEOMANCER_CALLER``/``_VERSION`` and ``GEOMANCER_SRC``
    (where the adapter imports geomancer from) are set here. A value already set
    wins, as in geomancer.app.main.
    """
    from geomancer import __version__, app

    app.declare_lab_configs()
    os.environ.setdefault("GEOMANCER_CALLER", CALLER)
    os.environ.setdefault("GEOMANCER_CALLER_VERSION", __version__)
    os.environ.setdefault("GEOMANCER_SRC", str(Path(__file__).resolve().parents[2]))


def manyruns_revision():
    """manyruns' version and install commit, from its direct_url.json (as app.py reads it)."""
    try:
        dist = metadata.distribution("manyruns")
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
    except (metadata.PackageNotFoundError, json.JSONDecodeError):
        return {"version": None, "commit": None}
    return {"version": dist.version,
            "commit": (direct.get("vcs_info") or {}).get("commit_id")}


# ── planning (plain data) ────────────────────────────────────────────────────

def parse_seeds(value):
    """``"0,1,2"``, ``3`` or ``[0, 2]`` -> a list of ints."""
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, int):
        return [value]
    return [int(v) for v in str(value).split(",") if v.strip()]


def _names(value, known):
    if value in (None, "all"):
        return sorted(known)
    names = [v.strip() for v in str(value).split(",") if v.strip()]
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise KeyError(f"unknown {unknown}; known: {sorted(known)}")
    return names


def plan(samples, seeds, names, state=True):
    """One dict per cell: every action per (sample, seed), plus one state cell per sample."""
    cells = []
    for sample in samples:
        for seed in seeds:
            for name in names:
                embedding, rest = name.split("+")
                extractor, scale = rest.split("@")
                cells.append({"kind": "graph", "sample": sample, "seed": seed, "action": name,
                              "embedding": embedding, "extractor": extractor, "scale": scale})
        if state:
            cells.append({"kind": "state", "sample": sample, "seed": seeds[0], "action": "state"})
    return cells


def units(cells):
    """Cells grouped into (sample, seed, arm) units, each run in its own out_dir.

    An embedding arm's cells share one embedding run; the raw arm's cells are
    independent single-step sessions; the state unit is one run.
    """
    grouped = {}
    for cell in cells:
        arm = "state" if cell["kind"] == "state" else cell["embedding"]
        key = (cell["sample"], cell["seed"], arm)
        grouped.setdefault(key, {"sample": cell["sample"], "seed": cell["seed"], "arm": arm,
                                 "cells": []})["cells"].append(cell)
    return list(grouped.values())


def shard(names, index, count):
    """The ``index``-th of ``count`` contiguous, near-equal slices of ``names``."""
    bounds = np.linspace(0, len(names), count + 1).round().astype(int)
    return list(names[bounds[index]:bounds[index + 1]])


def load_samples(folder):
    """``([{name, npy, sha256}], {name: error})`` from convert.py's manifest.

    Each .npy is checked against the manifest's sha256 here, before any run cites it.
    """
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    samples, errors = [], {}
    for name, entry in sorted(manifest["samples"].items()):
        if entry.get("status") != "ok":
            errors[name] = entry.get("error") or entry.get("status")
            continue
        path = folder / entry["npy"]
        actual = provenance.sha256_file(path)
        if actual != entry["npy_sha256"]:
            raise ValueError(f"{path} has sha256 {actual}, not the manifest's {entry['npy_sha256']}")
        samples.append({"name": name, "npy": str(path), "sha256": actual})
    return samples, errors


def embedding_recipe(name):
    """The committed recipe for embedding ``name`` (pca, diffmap or phate)."""
    return yaml.safe_load((RECIPES / f"ssb_embed_{name}.yaml").read_text())


def graph_step(cell, sample):
    # `group: analysis`, or the runner records the step as unsupported; the step name
    # must be the tool name, because the runner resolves the step by name.
    return {"name": "ssb_graph", "group": "analysis", "tool": "ssb_graph",
            "params": {"extractor": cell["extractor"], "scale": cell["scale"],
                       "points": sample["npy"], "points_sha256": sample["sha256"]}}


def state_step(sample):
    return {"name": "ssb_state", "group": "analysis", "tool": "ssb_state",
            "params": {"points": sample["npy"], "points_sha256": sample["sha256"]}}


# ── ledger ───────────────────────────────────────────────────────────────────

def ledgers(out):
    """Every shard ledger directory under ``<out>/ledgers``."""
    return sorted(p.parent for p in (Path(out) / "ledgers").glob("*/index.jsonl"))


def read_rows(out):
    from manyruns import store

    return [row for ledger in ledgers(out) for row in store.read(ledger)]


def done(out):
    """``{(sample, seed, action, geomancer commit)}`` of cells with an ok and complete row.

    A state cell's seed is None (the state reads the raw points only). Embedding
    rows are not cells: a missing graph cell re-runs its embedding.
    """
    keys = set()
    for row in read_rows(out):
        geo = (row.get("extra") or {}).get("geomancer") or {}
        if not (row.get("ok") and row.get("complete")) or geo.get("kind") not in ("graph", "state"):
            continue
        commit = (geo.get("geomancer_revision") or {}).get("commit")
        if geo["kind"] == "state":
            keys.add((geo.get("sample"), None, "state", commit))
        else:
            keys.add((geo.get("sample"), geo.get("seed"), geo.get("action"), commit))
    return keys


def _key(cell, commit):
    seed = None if cell["kind"] == "state" else cell["seed"]
    return (cell["sample"], seed, cell["action"], commit)


# ── running one unit ─────────────────────────────────────────────────────────

def _status(res):
    if res.get("ok") and res.get("complete"):
        return "ok", None
    failed = [s for s in res.get("steps") or [] if s.get("outcome") != "ok"]
    if failed:
        return failed[0].get("outcome") or "failed", failed[0].get("detail")
    return "failed", "; ".join(map(str, res.get("caveats") or [])) or None


def _clear_workdir(out_dir):
    # A declined tool step leaves the last branch's files in the shared workdir; remove
    # them so a stale graph can never be copied as this run's.
    work = Path(out_dir) / TOOL_WORKDIR
    for name in ("graph.json", "result.json"):
        (work / name).unlink(missing_ok=True)


def _keep_graph(res, out_dir, out):
    """Copy this run's graph.json to ``<out>/graphs/<run_id>.json``, checked by sha256.

    Returns ``{"path", "sha256"}`` (path relative to ``out``), or None when the run made
    no graph. A graph that does not hash to the emitted scalar raises: the shared
    workdir would then be unsafe for the next branch.
    """
    scalar = (res.get("g_vector") or {}).get("ssb_graph.graph_sha256")
    source = Path(out_dir) / TOOL_WORKDIR / "graph.json"
    if not (res.get("ok") and scalar and source.is_file()):
        return None
    if provenance.sha256_file(source) != scalar:
        raise RuntimeError(f"{source} does not hash to run {res['run_id']}'s graph_sha256 {scalar}")
    relative = Path("graphs") / f"{res['run_id']}.json"
    target = Path(out) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, target)
    if provenance.sha256_file(target) != scalar:
        raise RuntimeError(f"copy {target} does not hash to {scalar}")
    return {"path": str(relative), "sha256": scalar}


def _append(res, ledger, geo, modality):
    from manyruns import store

    store.append(res, out_dir=ledger, extra={"modality": modality, "extra": {"geomancer": geo}})
    status, detail = geo.get("status"), geo.get("detail")
    return {"run_id": res.get("run_id"), "kind": geo["kind"], "action": geo.get("action"),
            "embedding": geo.get("embedding"), "status": status, "detail": detail}


def _geo(ctx, cell, kind, **fields):
    return {"kind": kind, "source": ctx["source"], "sample": cell["sample"], "seed": cell["seed"],
            "points_sha256": ctx["sample"]["sha256"], "points_npy": ctx["sample"]["npy"],
            "geomancer_revision": ctx["revision"], "manyruns_revision": ctx["manyruns"],
            "caller": os.environ.get("GEOMANCER_CALLER"),
            "caller_version": os.environ.get("GEOMANCER_CALLER_VERSION"), **fields}


def _session(ctx, out_dir, **kwargs):
    from manyruns import session

    return session.Session(project=ctx["sample"]["name"], engine=ENGINE, out_dir=out_dir,
                           seed=ctx["seed"], fast_dev_run=False, modality=ctx["modality"],
                           dataset_name=ctx["sample"]["name"], array=ctx["X"], **kwargs)


def _graph_fields(cell, **fields):
    return {"action": cell["action"], "embedding": cell["embedding"],
            "extractor": cell["extractor"], "scale": cell["scale"], **fields}


def _not_run(ctx, cell, out_dir, ledger, reason, embedding_run_id):
    """A graph cell whose embedding left nothing to branch from: an honest not_run row."""
    s = _session(ctx, out_dir)
    res = s.close(abandoned=True, not_run=[{"index": 0, "name": "ssb_graph", "reason": reason}])
    geo = _geo(ctx, cell, "graph", **_graph_fields(cell, embedding_run_id=embedding_run_id,
                                                      graph=None, status="not_run", detail=reason))
    return _append(res, ledger, geo, ctx["modality"])


def _graph_run(ctx, cell, s, out_dir, ledger, embedding_run_id):
    _clear_workdir(out_dir)
    s.apply(graph_step(cell, ctx["sample"]))
    res = s.close()
    status, detail = _status(res)
    try:
        graph = _keep_graph(res, out_dir, ctx["out"])
    except RuntimeError as exc:
        geo = _geo(ctx, cell, "graph", **_graph_fields(
            cell, embedding_run_id=embedding_run_id, graph=None,
            status="graph_sha256_mismatch", detail=str(exc)))
        _append(res, ledger, geo, ctx["modality"])
        raise
    geo = _geo(ctx, cell, "graph", **_graph_fields(cell, embedding_run_id=embedding_run_id,
                                                      graph=graph, status=status, detail=detail))
    return _append(res, ledger, geo, ctx["modality"])


def _embedding_unit(ctx, unit, out_dir, ledger):
    recipe = embedding_recipe(unit["arm"])
    root = _session(ctx, out_dir, recipe=recipe)
    rec = root.apply(dict(recipe["steps"][0]))
    res = root.close()
    status, detail = _status(res)
    geo = _geo(ctx, unit, "embedding", embedding=unit["arm"], recipe=recipe["name"],
               status=status, detail=detail)
    rows = [_append(res, ledger, geo, ctx["modality"])]
    for cell in unit["cells"]:
        # branch(0) reads the embedding back from disk, so it works on the closed root.
        reason = None if rec.get("outcome") == "ok" else (
            f"embedding step {rec.get('outcome')}: {rec.get('detail')}")
        if reason is None:
            try:
                child = root.branch(0)
            except (ValueError, IndexError) as exc:
                reason = f"branch(0) refused: {exc}"
        if reason is not None:
            rows.append(_not_run(ctx, cell, out_dir, ledger, reason, res["run_id"]))
            continue
        # Serial: the next branch starts only after this graph is copied and checked.
        rows.append(_graph_run(ctx, cell, child, out_dir, ledger, res["run_id"]))
    return rows


def _raw_unit(ctx, unit, ledger):
    rows = []
    for cell in unit["cells"]:
        out_dir = ctx["unit_dir"] / cell["action"]
        rows.append(_graph_run(ctx, cell, _session(ctx, out_dir), out_dir, ledger, None))
    return rows


def _state_unit(ctx, unit, out_dir, ledger):
    s = _session(ctx, out_dir)
    s.apply(state_step(ctx["sample"]))
    res = s.close()
    status, detail = _status(res)
    geo = _geo(ctx, unit, "state", action="state", status=status, detail=detail)
    return [_append(res, ledger, geo, ctx["modality"])]


def run_unit(unit, sample, source, out, ledger, revision):
    """Run one (sample, seed, arm) unit; returns one summary dict per row appended."""
    out = Path(out)
    if unit["arm"] == "state":
        unit_dir = out / "cells" / sample["name"] / "state"
    else:
        unit_dir = out / "cells" / sample["name"] / f"seed{unit['seed']}" / unit["arm"]
    ctx = {"source": source, "sample": sample, "seed": unit["seed"], "out": out,
           "X": np.load(sample["npy"], allow_pickle=False), "modality": MODALITY[source],
           "revision": revision, "manyruns": manyruns_revision(), "unit_dir": unit_dir}
    if unit["arm"] == "state":
        return _state_unit(ctx, unit, unit_dir, ledger)
    if unit["arm"] == "raw":
        return _raw_unit(ctx, unit, ledger)
    return _embedding_unit(ctx, unit, unit_dir, ledger)


def _work(job):
    unit, sample, source, out, ledger, revision, per_process = job
    if per_process:
        # One ledger per worker process: O_APPEND is not proven safe on NFS.
        ledger = f"{ledger}-p{os.getpid()}"
    started = time.time()
    try:
        rows = run_unit(unit, sample, source, out, ledger, revision)
        return unit, rows, None, time.time() - started
    except Exception as exc:
        traceback.print_exc()
        return unit, [], f"{type(exc).__name__}: {exc}", time.time() - started


def _ledger_dir(out, cfg):
    tag = f"{cfg['source']}-{int(cfg['shard']):03d}of{int(cfg['of']):03d}"
    return Path(out) / "ledgers" / tag


def run(cfg):
    """Run every pending cell of ``cfg``'s shard. Returns ``{"units", "rows", "errors", ...}``."""
    setup_env()
    source = cfg["source"]
    if source not in MODALITY:
        raise ValueError(f"unknown source {source!r}; expected synthetic or real")
    if cfg["inputs"]:
        inputs = Path(cfg["inputs"])
    else:
        from manylatents.singlecell.data import scshapebench

        inputs = Path(scshapebench.scshapebench_root()) / "grid_inputs" / source
    out = Path(cfg["out"])
    samples, input_errors = load_samples(inputs)
    by_name = {s["name"]: s for s in samples}
    names = sorted(by_name) if cfg["samples"] is None else _names(cfg["samples"], by_name)
    names = shard(names[:cfg["limit"]], int(cfg["shard"]), int(cfg["of"]))
    revision = provenance.geomancer_revision()
    finished = done(out)
    cells = [c for c in plan(names, parse_seeds(cfg["seeds"]),
                             _names(cfg["actions"], actions_module.ACTIONS), cfg["state"])
             if _key(c, revision["commit"]) not in finished]
    todo = units(cells)
    ledger = _ledger_dir(out, cfg)
    print(f"{source}: {len(names)} samples, {len(cells)} pending cells in {len(todo)} units, "
          f"{len(input_errors)} input errors; ledger {ledger}", flush=True)
    workers = int(cfg["workers"])
    jobs = [(u, by_name[u["sample"]], source, str(out), str(ledger), revision, workers > 1)
            for u in todo]
    if workers > 1:
        # spawn, not fork: the parent has already imported manyruns and its engine.
        pool = ProcessPoolExecutor(workers, mp_context=get_context("spawn"))
        results = pool.map(_work, jobs)
    else:
        pool, results = None, map(_work, jobs)
    summary = {"units": 0, "rows": 0, "errors": [], "input_errors": input_errors,
               "status": {}, "seconds": {}}
    started = time.time()
    try:
        for unit, rows, error, seconds in results:
            summary["units"] += 1
            summary["rows"] += len(rows)
            for row in rows:
                summary["status"][row["status"]] = summary["status"].get(row["status"], 0) + 1
            if error:
                summary["errors"].append({"sample": unit["sample"], "seed": unit["seed"],
                                          "arm": unit["arm"], "error": error})
            key = unit["sample"]
            summary["seconds"][key] = round(summary["seconds"].get(key, 0.0) + seconds, 2)
            print(f"[{summary['units']}/{len(jobs)}] {unit['sample']} seed={unit['seed']} "
                  f"{unit['arm']}: {len(rows)} rows {seconds:.1f}s"
                  + (f" ERROR {error}" if error else ""), flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    summary["wall_seconds"] = round(time.time() - started, 2)
    return summary


def main(argv=None):
    cfg = override(config, sys.argv[1:] if argv is None else argv)
    summary = run(cfg)
    print(json.dumps({k: v for k, v in summary.items() if k != "seconds"}, indent=1), flush=True)
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
