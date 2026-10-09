"""The one module that computes. Tests replace `execute_step`; nothing else touches compute.

A step runs in a fresh `manyruns.session.Session` built from the source data, exactly as the
`manyruns>` REPL applies steps. Inputs come from the selected upstream iteration:

* before PCA, the expression matrix is RECOMPUTED by replaying the selected preprocessing
  iterations from the source, checked against the digest each one recorded;
* from PCA on, stored arrays are loaded (`emb`, `pseudotime`, `time_labels`).

`manyruns.pipeline` is imported here only, for its public loaders (allowlisted in
`tests/test_layer.py`); compute itself goes through `Session.apply`.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any, Optional

from geomancer.training import diagnostics, playdata, timebins
from geomancer.training.workflow import LOCAL, PSEUDOTIME, REAL

OK, ERROR = "ok", "error"

#: Steps whose output is per-cell time bins for MIOFlow.
TIME_STEPS = frozenset({"discretize_time", "subcluster_time"})


# ── loading ─────────────────────────────────────────────────────────────────────────────
def load_source(project: dict) -> dict:
    """The source matrix, labels and gene axis, through manyruns' public loaders.

    Engine-generated datasets (a manylatents handle with no file) load nothing here; the engine
    generates them inside the first latent step, from `dataset` and `data_kwargs`.
    """
    from manyruns import pipeline

    out = {"array": None, "labels": None, "kind": None, "counts": None, "genes": None,
           "truth": None}
    folder = project.get("data_folder")
    if not folder:
        return out
    time_key = project.get("time_key")
    path = Path(folder)
    with pipeline.quiet():
        if path.is_dir():
            out["array"], out["labels"], out["kind"] = pipeline.load_labeled(path, time_key=time_key)
            return out
        obj = pipeline.load_array(path)
        is_anndata = hasattr(obj, "obs") and hasattr(obj, "X")
        out["array"] = obj.X if is_anndata else pipeline.as_matrix(obj)
        out["labels"], out["kind"] = pipeline.labels_of(obj, time_key=time_key)
        out["counts"], out["genes"] = pipeline.gene_axis(obj)
        if is_anndata and "true_time" in getattr(obj, "obs", {}):
            # Play datasets carry their generative progress; scored, never shown by default.
            out["truth"] = obj.obs["true_time"].to_numpy(dtype=float)
    return out


def digest(matrix: Any) -> dict:
    """Shape and sha256 of a dense or sparse matrix, without copying a contiguous array."""
    import numpy as np

    h = hashlib.sha256()
    if hasattr(matrix, "indptr"):  # scipy sparse (CSR/CSC)
        h.update(b"sparse")
        for part in (matrix.data, matrix.indices, matrix.indptr):
            h.update(memoryview(np.ascontiguousarray(part)))
    else:
        h.update(memoryview(np.ascontiguousarray(matrix)).cast("B"))
    return {"shape": list(getattr(matrix, "shape", ())), "sha256": h.hexdigest()}


def build_session(job: dict, source: dict):
    """A manyruns Session over the source data, writing into this iteration's directory."""
    from manyruns.session import Session

    p = job["project"]
    session = Session(
        project=p["project"], engine=p["engine"], out_dir=Path(job["out_dir"]),
        seed=int(job["seed"]), fast_dev_run=bool(job["smoke"]), modality=p.get("modality", "unknown"),
        recipe={}, array=source["array"], labels=source["labels"], label_kind=source["kind"],
        counts=source["counts"], genes=source["genes"], dataset=p.get("dataset"),
        dataset_name=p.get("dataset_name"), data_kwargs=p.get("data_kwargs") or {},
        declared_shape=p.get("declared_shape"), device=job["device"],
        # Same choice as manyruns' own tuning loop: the O(n^2) suite is not paid per attempt.
        metrics=[],
    )
    if job["time_mode"] == PSEUDOTIME and session.state.get("label_kind") == "time":
        # Pseudotime mode derives time from DPT; measured time still colours the plots.
        session.state["labels"] = None
        session.state["label_kind"] = None
    return session


# ── execution ───────────────────────────────────────────────────────────────────────────
def execute_step(job: dict) -> dict:
    """Run one step. `job` keys: project, step, time_mode, inputs ({slot: array}), replay
    ([{step, expect}]), out_dir, seed, device, smoke. Returns what the step wrote."""
    if job["step"]["name"] in LOCAL:
        return _execute_local(job)
    from manyruns import artifacts

    source = load_source(job["project"])
    session = build_session(job, source)
    for item in job.get("replay") or []:
        rec = session.apply(item["step"])
        expect = item.get("expect")
        if expect and rec.get("outcome") == OK:
            got = digest(session.state["X"])
            if got != expect:
                return {"outcome": ERROR, "detail": (
                    f"recomputing {item['step']['name']} gave a different matrix than when it "
                    f"was recorded (sha256 {got['sha256'][:12]} vs {expect['sha256'][:12]}); the "
                    "source data or software changed")}
    session.steps.clear()
    inputs = job.get("inputs") or {}
    for slot in ("emb", "pseudotime"):
        if slot in inputs:
            session.state[slot] = inputs[slot]
    if "emb" in inputs:
        session.provided = {"embedding"}
    if "time_labels" in inputs:
        session.state["labels"] = inputs["time_labels"]
        session.state["label_kind"] = "time"
    elif job["time_mode"] == REAL and job["step"]["name"] == "mioflow":
        return {"outcome": ERROR, "detail": "real-time mode reached mioflow without refined bins"}
    before_labels = session.state.get("labels")
    before_x = session.state.get("X")
    rec = session.apply(job["step"])
    produced = dict(rec.get("artifacts") or {})
    carried = {slot: produced[slot] for slot in ("emb", "pseudotime") if slot in produced}
    # Only a time step produces bins. A narrowing prep step also re-reads labels (a new object
    # with the same content), which must not be mistaken for derived time.
    if (rec.get("outcome") == OK and job["step"]["name"] in TIME_STEPS
            and session.state.get("label_kind") == "time"
            and session.state.get("labels") is not before_labels):
        import numpy as np

        path = artifacts.save_array(job["out_dir"], session.run_id, 0, job["step"]["name"],
                                    "time_labels", np.asarray(session.state["labels"]).astype(str))
        produced["time_labels"] = path or "not written: save failed"
        if path:
            carried["time_labels"] = path
    found = {}
    if rec.get("outcome") == OK:
        name = job["step"]["name"]
        found = diagnostics.compute(
            name, emb=session.state.get("emb") if name == "phate" else None,
            time_labels=session.state.get("labels") if name == "discretize_time" else None,
            g=session.g, knn=(job["step"].get("params") or {}).get("knn"),
            source_time=source["labels"] if source["kind"] == "time" else None)
    truth = {}
    if rec.get("outcome") == OK and source["truth"] is not None:
        name = job["step"]["name"]
        truth = playdata.truth_scores(
            name, source["truth"], emb=session.state.get("emb"),
            pseudotime=session.state.get("pseudotime") if name == "dpt" else None,
            time_labels=session.state.get("labels") if name in TIME_STEPS else None)
    recomputed = None
    if (rec.get("outcome") == OK and job["step"].get("group") == "prep"
            and session.state.get("X") is not None and session.state.get("X") is not before_x):
        recomputed = {"X": digest(session.state["X"])}
    return {
        "outcome": rec.get("outcome"), "detail": rec.get("detail"),
        "params": rec.get("params") or {}, "bounded": rec.get("bounded"),
        "seconds": rec.get("seconds"), "artifacts": produced, "carried": carried,
        "recomputed": recomputed, "plots": list(session.plots), "g": dict(session.g),
        "run_id": session.run_id, "caveats": list(session.ctx.get("caveats") or []),
        "model": ("trained model kept in memory only; not persisted"
                  if session.state.get("model") is not None else None),
        "diagnostics": found, "truth": truth,
    }


def _execute_local(job: dict) -> dict:
    """`subcluster_time`: refined bins around real timepoints (see `timebins`)."""
    import numpy as np
    from manyruns import artifacts

    inputs = job.get("inputs") or {}
    missing = [s for s in ("emb", "pseudotime") if s not in inputs]
    if missing:
        return {"outcome": ERROR, "detail": f"subcluster_time needs {missing} from upstream"}
    source = load_source(job["project"])
    if source["kind"] != "time" or source["labels"] is None:
        return {"outcome": ERROR, "detail": "real-time mode needs measured time labels"}
    params = job["step"]["params"]
    refined, found = timebins.refine(inputs["emb"], inputs["pseudotime"], source["labels"],
                                           int(params["k_per_timepoint"]), int(job["seed"]))
    out_dir = Path(job["out_dir"])
    run_id = uuid.uuid4().hex[:12]
    # Fixed-point, never scientific notation: manyruns' time parser rejects "1e-05".
    labels = np.char.mod("%.6f", refined)
    path = artifacts.save_array(out_dir, run_id, 0, "subcluster_time", "time_labels", labels)
    plots = []
    plot = _scatter(inputs["emb"], refined, out_dir / "plots" / "subcluster_time.png",
                    "refined time (subclusters within each timepoint)")
    if plot:
        plots.append(plot)
    return {"outcome": OK, "detail": None, "params": dict(params), "bounded": None,
            "seconds": None, "artifacts": {"time_labels": path or "not written: save failed"},
            "carried": {"time_labels": path} if path else {}, "recomputed": None,
            "plots": plots, "g": found, "run_id": run_id, "caveats": [], "model": None,
            "diagnostics": diagnostics.compute("subcluster_time", g=found),
            "truth": (playdata.truth_scores("subcluster_time", source["truth"],
                                            time_labels=refined)
                      if source["truth"] is not None else {})}


def _scatter(emb: Any, colour: Any, path: Path, title: str) -> Optional[str]:
    import numpy as np

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    emb = np.asarray(emb)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    points = ax.scatter(emb[:, 0], emb[:, 1] if emb.shape[1] > 1 else np.zeros(len(emb)),
                        c=colour, s=8, cmap="viridis")
    fig.colorbar(points, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
