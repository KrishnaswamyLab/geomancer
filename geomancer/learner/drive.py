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
naming it. An exception inside a unit leaves an ``aborted`` row for every cell
of the unit that has no row yet.

Resume skips a cell that already has a row with a final status (ok, error,
not_run, skipped) for this geomancer commit and dirty flag, so a cell that fails
the same way every time is not rerun on each resubmission. ``retry`` names
final statuses to rerun anyway; aborted and other statuses always rerun. It
reads only this shard's ledgers, so a grid must be resumed with the same
``of``. Do not split the seeds of one shard across invocations that run at the
same time: both would run the sample's state unit in one workdir.

The grid runs only from a clean checkout (``allow_dirty`` for development), on
inputs that a clean checkout converted, with tools whose environments are the
locked ones (``allow_tool_override`` for the test interpreter), and with no
editable manyruns, manylatents, manylatents-omics or scShapeBench.

Identity. The step params name the points by sample and sha256, never by a path,
so a cell's ``spec_id`` is the same on every machine. At manyruns v0.1.0 a
``spec_id`` is NOT equal across samples: it hashes ``dataset_name``, which is the
sample. It is also the same for the six branches of every embedding, because it
does not hash ``parent``. So each row carries ``extra.geomancer.action_spec_sha256``,
the canonical hash of the action's own steps (the embedding step, and the tool
step without its per-sample params): equal across samples and seeds for one
action, and different between actions.

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
    "retry": "",  # final statuses to rerun anyway, e.g. "error,skipped"
    "threads": 2,  # BLAS/OpenMP threads per worker and per tool process
    "allow_dirty": False,  # run from a checkout with uncommitted changes (never evidence)
    "allow_tool_override": False,  # accept MANYRUNS_TOOL_<NAME>_PYTHON (tool_reproducible false)
    "allow_editable": False,  # accept an editable install of the engine or the loader
}

RECIPES = Path(__file__).resolve().parents[1] / "configs" / "recipe"
MODALITY = {"synthetic": "synthetic", "real": "scrna"}
CALLER = "geomancer.learner.drive"
ENGINE = "manylatents"
TOOL_WORKDIR = "tool-ssb_graph"
TOOLS = ("ssb_graph", "ssb_state")
# A row with one of these statuses is a result: resume does not rerun its cell.
FINAL = ("ok", "error", "not_run", "skipped")
# Per-sample params of the tool steps: left out of action_spec_sha256.
PER_SAMPLE = ("sample", "points_sha256")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")


# ── environment ──────────────────────────────────────────────────────────────

def setup_env(threads=2):
    """Merge geomancer's lab configs into manyruns' and name the caller, before any tool step.

    The adapters inherit this process's environment (manyruns starts them without
    its own ``env=``), so ``GEOMANCER_CALLER``/``_VERSION`` and ``GEOMANCER_SRC``
    (where the adapter imports geomancer from) are set here, and the BLAS/OpenMP
    thread counts, so that workers x threads does not oversubscribe the node. A
    value already set wins, as in geomancer.app.main.
    """
    from geomancer import __version__, app

    app.declare_lab_configs()
    os.environ.setdefault("GEOMANCER_CALLER", CALLER)
    os.environ.setdefault("GEOMANCER_CALLER_VERSION", __version__)
    os.environ.setdefault("GEOMANCER_SRC", str(Path(__file__).resolve().parents[2]))
    for name in THREAD_VARS:
        os.environ.setdefault(name, str(threads))


def manyruns_revision():
    """manyruns' version and install commit, from its direct_url.json (as app.py reads it)."""
    rev = provenance.dist_revision("manyruns")
    return {"version": rev["version"], "commit": rev["commit"]}


def check_installs(allow_editable=False):
    """Refuse an engine or loader whose code the rows could not name.

    manyruns, manylatents-omics and scShapeBench must be git installs with a commit;
    none of the four may be editable, because an editable install's code is whatever
    its directory holds now. manylatents is a PyPI wheel, named by its version.
    """
    problems = []
    for name, rev in provenance.engine_revisions().items():
        if name == "uv_lock_sha256":
            continue
        if rev["version"] is None:
            problems.append(f"{name} is not installed")
        elif rev["editable"] and not allow_editable:
            problems.append(f"{name} is an editable install")
        elif rev["commit"] is None and name != "manylatents" and not allow_editable:
            problems.append(f"{name} {rev['version']} has no git commit in its direct_url.json")
    if problems:
        raise RuntimeError("; ".join(problems) + ". Run `uv sync --locked --extra learner`; "
                           "allow_editable=true is for development only")


def check_tools(names, allow_override=False):
    """``{tool: status}``; refuses unless every tool is ready (built from its lock).

    A tool that is not built would make every one of its steps a skipped row while
    the process exits 0. An interpreter override (MANYRUNS_TOOL_<NAME>_PYTHON) runs,
    but its rows say tool_reproducible false, so it is accepted only when allowed.
    """
    from manyruns import toolchain

    statuses, problems = {}, []
    for name in names:
        resolved = toolchain.resolve(name)
        statuses[name] = resolved["status"]
        if resolved["status"] == "ready" or (resolved["status"] == "overridden" and allow_override):
            continue
        problems.append(f"{name}: {resolved['status']}: {resolved.get('reason')}"
                        + (f"; fix: {resolved['fix']}" if resolved.get("fix") else ""))
    if problems:
        raise RuntimeError("tools not ready: " + " | ".join(problems))
    return statuses


def check_revision(revision, what, allow_dirty=False):
    """Refuse code whose commit is unknown or whose tree had uncommitted changes."""
    if allow_dirty:
        return
    if revision.get("commit") is None or revision.get("dirty") is not False:
        raise RuntimeError(f"{what} is commit {revision.get('commit')} with dirty={revision.get('dirty')}; "
                           "a grid runs from a clean commit (allow_dirty=true for development)")


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
    """One dict per cell: every action per (sample, seed), plus one state cell per sample.

    A seedless action (`actions.seedless`, the raw routes) is planned for the first seed only:
    it gives the same graph for every seed, so table.py reuses that one run in every seed's
    table rather than this repeating it."""
    cells = []
    for sample in samples:
        for seed in seeds:
            for name in names:
                if actions_module.seedless(name) and seed != seeds[0]:
                    continue
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
    """``([{name, npy, sha256}], {name: error}, inputs)`` from convert.py's manifest.

    Each .npy is checked against the manifest's sha256 here, before any run cites it.
    ``inputs`` names the manifest by sha256 with the code revision that wrote it,
    for every row.
    """
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    inputs = {"path": str(folder), "manifest_sha256": provenance.sha256_file(folder / "manifest.json"),
              "code_revision": manifest.get("code_revision"),
              "engine_revisions": manifest.get("engine_revisions")}
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
    return samples, errors, inputs


def embedding_recipe(name):
    """The committed recipe for embedding ``name`` (pca, diffmap or phate)."""
    return yaml.safe_load((RECIPES / f"ssb_embed_{name}.yaml").read_text())


def graph_step(cell, sample):
    # `group: analysis`, or the runner records the step as unsupported; the step name
    # must be the tool name, because the runner resolves the step by name. The points
    # are named by sample and sha256 only: spec_id hashes these params, and a path
    # would make it differ between machines (the adapter finds the file in
    # $GEOMANCER_GRID_INPUTS).
    return {"name": "ssb_graph", "group": "analysis", "tool": "ssb_graph",
            "params": {"extractor": cell["extractor"], "scale": cell["scale"],
                       "sample": sample["name"], "points_sha256": sample["sha256"]}}


def state_step(sample):
    return {"name": "ssb_state", "group": "analysis", "tool": "ssb_state",
            "params": {"sample": sample["name"], "points_sha256": sample["sha256"]}}


def _spec(step):
    params = {k: v for k, v in (step.get("params") or {}).items() if k not in PER_SAMPLE}
    return {"name": step.get("name"), "group": step.get("group"), "tool": step.get("tool"),
            "params": params}


def action_spec_sha256(cell, sample=None):
    """The canonical hash of what an action does, the same for every sample and seed.

    A graph action is its embedding step (the committed recipe's, or "raw") and its
    ssb_graph step without the per-sample params; the state action is its step.
    """
    sample = sample or {"name": None, "sha256": None}
    if cell["kind"] == "state":
        return provenance.digest({"state": _spec(state_step(sample))})
    embedding = ("raw" if cell["embedding"] == "raw"
                 else _spec(embedding_recipe(cell["embedding"])["steps"][0]))
    return provenance.digest({"embedding": embedding, "graph": _spec(graph_step(cell, sample))})


# ── ledger ───────────────────────────────────────────────────────────────────

def ledgers(out, prefix=None):
    """The shard ledger directories under ``<out>/ledgers`` (those whose name starts with ``prefix``)."""
    return sorted(p.parent for p in (Path(out) / "ledgers").glob(f"{prefix or ''}*/index.jsonl"))


def _final(retry):
    retry = {v.strip() for v in (retry.split(",") if isinstance(retry, str) else retry or []) if v.strip()}
    return set(FINAL) - retry


def done(out, prefix=None, retry=""):
    """``{(sample, seed, action, commit, dirty)}`` of cells that have a row with a final status.

    A final status (FINAL, less ``retry``) is a result: an ok and complete row, or a
    failure that would repeat (a failed embedding's not_run cells, an adapter that
    declined). An aborted row, a sha256 mismatch or any other status is not final,
    so its cell runs again. The key holds the dirty flag, so a row from a dirty tree
    never counts as done for a clean run of the same commit. A state cell's seed is
    None (the state reads the raw points only). Embedding rows are not cells.
    Streams the rows and keeps only the keys.
    """
    from manyruns import store

    final, keys = _final(retry), set()
    for ledger in ledgers(out, prefix):
        for row in store.read(ledger):
            geo = (row.get("extra") or {}).get("geomancer") or {}
            if geo.get("kind") not in ("graph", "state"):
                continue
            status = geo.get("status")
            if status not in final or (status == "ok" and not (row.get("ok") and row.get("complete"))):
                continue
            rev = geo.get("geomancer_revision") or {}
            seed = None if geo["kind"] == "state" else geo.get("seed")
            action = "state" if geo["kind"] == "state" else geo.get("action")
            keys.add((geo.get("sample"), seed, action, rev.get("commit"), rev.get("dirty")))
    return keys


def _key(cell, revision):
    seed = None if cell["kind"] == "state" else cell["seed"]
    return (cell["sample"], seed, cell["action"], revision["commit"], revision["dirty"])


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
    # The machine-local path of the points is kept here, never in the step params.
    return {"kind": kind, "source": ctx["source"], "sample": cell["sample"], "seed": cell["seed"],
            "points_sha256": ctx["sample"]["sha256"], "points_npy": ctx["sample"]["npy"],
            "inputs": ctx["inputs"],
            "geomancer_revision": ctx["revision"], "manyruns_revision": ctx["manyruns"],
            "engine_revisions": ctx["engines"],
            "actions_sha256": ctx["actions_sha256"],
            "caller": os.environ.get("GEOMANCER_CALLER"),
            "caller_version": os.environ.get("GEOMANCER_CALLER_VERSION"), **fields}


def _session(ctx, out_dir, **kwargs):
    from manyruns import session

    return session.Session(project=ctx["sample"]["name"], engine=ENGINE, out_dir=out_dir,
                           seed=ctx["seed"], fast_dev_run=False, modality=ctx["modality"],
                           dataset_name=ctx["sample"]["name"], array=ctx["X"], **kwargs)


def _graph_fields(ctx, cell, **fields):
    return {"action": cell["action"], "embedding": cell["embedding"],
            "extractor": cell["extractor"], "scale": cell["scale"],
            "action_spec_sha256": action_spec_sha256(cell, ctx["sample"]), **fields}


def _not_run(ctx, cell, out_dir, ledger, reason, embedding_run_id, status="not_run"):
    """A cell that did not run: an honest row with ``status`` (not_run or aborted) and why."""
    s = _session(ctx, out_dir)
    if cell["kind"] == "state":
        res = s.close(abandoned=True, not_run=[{"index": 0, "name": "ssb_state", "reason": reason}])
        geo = _geo(ctx, cell, "state", action="state", status=status, detail=reason,
                   action_spec_sha256=action_spec_sha256(cell, ctx["sample"]))
    else:
        res = s.close(abandoned=True, not_run=[{"index": 0, "name": "ssb_graph", "reason": reason}])
        geo = _geo(ctx, cell, "graph", **_graph_fields(ctx, cell, embedding_run_id=embedding_run_id,
                                                          graph=None, status=status, detail=reason))
    return _append(res, ledger, geo, ctx["modality"])


def _graph_run(ctx, cell, s, out_dir, ledger, embedding_run_id, rows):
    """Apply ssb_graph on ``s``, keep the graph, append the row (and its summary to ``rows``)."""
    _clear_workdir(out_dir)
    s.apply(graph_step(cell, ctx["sample"]))
    res = s.close()
    status, detail = _status(res)
    try:
        graph = _keep_graph(res, out_dir, ctx["out"])
    except RuntimeError as exc:
        geo = _geo(ctx, cell, "graph", **_graph_fields(
            ctx, cell, embedding_run_id=embedding_run_id, graph=None,
            status="graph_sha256_mismatch", detail=str(exc)))
        rows.append(_append(res, ledger, geo, ctx["modality"]))
        raise
    geo = _geo(ctx, cell, "graph", **_graph_fields(ctx, cell, embedding_run_id=embedding_run_id,
                                                      graph=graph, status=status, detail=detail))
    rows.append(_append(res, ledger, geo, ctx["modality"]))


def _emb_sha256(res):
    # The embedding's bytes at append time, so score.py can check a copied grid's file.
    for step in res.get("steps") or []:
        path = (step.get("artifacts") or {}).get("emb")
        if path and Path(path).is_file():
            return provenance.sha256_file(path)
    return None


def _embedding_unit(ctx, unit, out_dir, ledger, rows):
    recipe = embedding_recipe(unit["arm"])
    root = _session(ctx, out_dir, recipe=recipe)
    rec = root.apply(dict(recipe["steps"][0]))
    res = root.close()
    status, detail = _status(res)
    geo = _geo(ctx, unit, "embedding", embedding=unit["arm"], recipe=recipe["name"],
               embedding_spec_sha256=provenance.digest(_spec(recipe["steps"][0])),
               emb_sha256=_emb_sha256(res), status=status, detail=detail)
    rows.append(_append(res, ledger, geo, ctx["modality"]))
    ctx["embedding_run_id"] = res["run_id"]
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
        _graph_run(ctx, cell, child, out_dir, ledger, res["run_id"], rows)


def _raw_unit(ctx, unit, ledger, rows):
    for cell in unit["cells"]:
        out_dir = ctx["unit_dir"] / cell["action"]
        _graph_run(ctx, cell, _session(ctx, out_dir), out_dir, ledger, None, rows)


def _state_unit(ctx, unit, out_dir, ledger, rows):
    cell = unit["cells"][0]
    s = _session(ctx, out_dir)
    s.apply(state_step(ctx["sample"]))
    res = s.close()
    status, detail = _status(res)
    geo = _geo(ctx, unit, "state", action="state", status=status, detail=detail,
               action_spec_sha256=action_spec_sha256(cell, ctx["sample"]))
    rows.append(_append(res, ledger, geo, ctx["modality"]))


def _cell_dir(ctx, unit, cell):
    return ctx["unit_dir"] / cell["action"] if unit["arm"] == "raw" else ctx["unit_dir"]


def run_unit(unit, sample, source, out, ledger, revision, inputs=None):
    """Run one (sample, seed, arm) unit; returns ``(row summaries, error or None)``.

    When the unit raises, every cell of it that has no row yet gets an ``aborted``
    row naming the error, so each planned cell ends with a row and a status.
    """
    out = Path(out)
    if unit["arm"] == "state":
        unit_dir = out / "cells" / sample["name"] / "state"
    else:
        unit_dir = out / "cells" / sample["name"] / f"seed{unit['seed']}" / unit["arm"]
    ctx = {"source": source, "sample": sample, "seed": unit["seed"], "out": out,
           "X": np.load(sample["npy"], allow_pickle=False), "modality": MODALITY[source],
           "revision": revision, "manyruns": manyruns_revision(),
           "engines": provenance.engine_revisions(), "inputs": inputs,
           "actions_sha256": provenance.sha256_file(actions_module.__file__),
           "unit_dir": unit_dir, "embedding_run_id": None}
    rows = []
    try:
        if unit["arm"] == "state":
            _state_unit(ctx, unit, unit_dir, ledger, rows)
        elif unit["arm"] == "raw":
            _raw_unit(ctx, unit, ledger, rows)
        else:
            _embedding_unit(ctx, unit, unit_dir, ledger, rows)
        return rows, None
    except Exception as exc:
        traceback.print_exc()
        error = f"{type(exc).__name__}: {exc}"
    have = {r["action"] for r in rows if r["kind"] in ("graph", "state")}
    for cell in unit["cells"]:
        if cell["action"] in have:
            continue
        try:
            rows.append(_not_run(ctx, cell, _cell_dir(ctx, unit, cell), ledger,
                                 f"unit aborted: {error}", ctx["embedding_run_id"], status="aborted"))
        except Exception as exc:  # the row itself failed: the error list still names the unit
            error += f"; no aborted row for {cell['action']}: {type(exc).__name__}: {exc}"
    return rows, error


def _work(job):
    unit, sample, source, out, ledger, revision, inputs = job
    # One ledger per process and per invocation: O_APPEND is not proven safe on NFS,
    # and a process never appends to a file that a killed process left half-written.
    ledger = f"{ledger}-p{os.getpid()}"
    started = time.time()
    try:
        rows, error = run_unit(unit, sample, source, out, ledger, revision, inputs)
    except Exception as exc:
        traceback.print_exc()
        rows, error = [], f"{type(exc).__name__}: {exc}"
    return unit, rows, error, time.time() - started


def shard_tag(cfg):
    return f"{cfg['source']}-{int(cfg['shard']):03d}of{int(cfg['of']):03d}-"


def _ledger_dir(out, cfg):
    # <source>-<shard>of<of>-<job>, then -p<pid> per process (_work).
    job = os.environ.get("SLURM_JOB_ID") or f"local{int(time.time())}"
    return Path(out) / "ledgers" / f"{shard_tag(cfg)}{job}"


def run(cfg):
    """Run every pending cell of ``cfg``'s shard. Returns ``{"units", "rows", "errors", ...}``."""
    setup_env(int(cfg["threads"]))
    source = cfg["source"]
    if source not in MODALITY:
        raise ValueError(f"unknown source {source!r}; expected synthetic or real")
    if cfg["inputs"]:
        inputs = Path(cfg["inputs"])
    else:
        from manylatents.singlecell.data import scshapebench

        inputs = Path(scshapebench.scshapebench_root()) / "grid_inputs" / source
    inputs = inputs.resolve()
    out = Path(cfg["out"])
    check_installs(cfg["allow_editable"])
    tools = check_tools(TOOLS if cfg["state"] else TOOLS[:1], cfg["allow_tool_override"])
    revision = provenance.geomancer_revision()
    check_revision(revision, "the geomancer checkout", cfg["allow_dirty"])
    samples, input_errors, inputs_info = load_samples(inputs)
    check_revision(inputs_info["code_revision"] or {}, f"the code that wrote {inputs}/manifest.json",
                   cfg["allow_dirty"])
    # The adapters find the points here by sample and sha256 (provenance.grid_points).
    os.environ["GEOMANCER_GRID_INPUTS"] = str(inputs)
    by_name = {s["name"]: s for s in samples}
    names = sorted(by_name) if cfg["samples"] is None else _names(cfg["samples"], by_name)
    names = shard(names[:cfg["limit"]], int(cfg["shard"]), int(cfg["of"]))
    finished = done(out, shard_tag(cfg), cfg["retry"])
    cells = [c for c in plan(names, parse_seeds(cfg["seeds"]),
                             _names(cfg["actions"], actions_module.ACTIONS), cfg["state"])
             if _key(c, revision) not in finished]
    todo = units(cells)
    ledger = _ledger_dir(out, cfg)
    print(f"{source}: {len(names)} samples, {len(cells)} pending cells in {len(todo)} units, "
          f"{len(input_errors)} input errors; tools {tools}; ledger {ledger}-p<pid>", flush=True)
    workers = int(cfg["workers"])
    jobs = [(u, by_name[u["sample"]], source, str(out), str(ledger), revision, inputs_info)
            for u in todo]
    if workers > 1:
        # spawn, not fork: the parent has already imported manyruns and its engine.
        pool = ProcessPoolExecutor(workers, mp_context=get_context("spawn"))
        results = pool.map(_work, jobs)
    else:
        pool, results = None, map(_work, jobs)
    summary = {"units": 0, "rows": 0, "errors": [], "input_errors": input_errors,
               "status": {}, "seconds": {}, "tools": tools}
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
    """Exit 1 when a unit raised, 2 when a step was unsupported (a config error), else 0.

    A skipped row is an adapter that declined, which is a result; a tool that is not
    built is refused before any cell runs (check_tools).
    """
    cfg = override(config, sys.argv[1:] if argv is None else argv)
    summary = run(cfg)
    print(json.dumps({k: v for k, v in summary.items() if k != "seconds"}, indent=1), flush=True)
    if summary["errors"]:
        return 1
    return 2 if summary["status"].get("unsupported") else 0


if __name__ == "__main__":
    sys.exit(main())
