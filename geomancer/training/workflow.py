"""The workflow training mode walks, and the parameters each step offers. Pure data.

PHATE → DPT → time → MIOFlow, preceded by the preprocessing the CFlows Methods describe
(counts → normalize → log1p → PCA 50), with a gene filter first to keep log1p's dense matrix
small. Two time modes, chosen when a session starts:

    pseudotime  DPT ordering binned into equal-width timepoints (data with no time axis)
    real        each real timepoint subclustered on PHATE and ordered by DPT, giving refined
                bins around the measured times (CFlows Methods, lines 626–633)

Every value a reviewer might change is read from `choices.py`, not written here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from geomancer.training import choices

PSEUDOTIME, REAL = "pseudotime", "real"
TIME_MODES = (PSEUDOTIME, REAL)

#: Steps whose output is the expression matrix itself. They are recomputed from the source
#: (cheap and deterministic) instead of stored, so large dense matrices never reach disk.
RECOMPUTED = frozenset({"filter_genes", "normalize", "transform"})

#: Steps computed in Geomancer rather than by a manyruns executor.
LOCAL = frozenset({"subcluster_time"})


def steps(time_mode: str) -> list[dict]:
    """The ordered step dicts for one time mode. Each is what manyruns' `Session.apply` takes."""
    if time_mode not in TIME_MODES:
        raise ValueError(f"time mode must be one of {TIME_MODES}, got {time_mode!r}")
    time_step = (
        {"name": "discretize_time", "group": "analysis",
         "params": {"n_timepoints": choices.PSEUDOTIME_BINS}}
        if time_mode == PSEUDOTIME else
        {"name": "subcluster_time", "group": "local",
         "params": ({} if choices.K_PER_TIMEPOINT is None
                    else {"k_per_timepoint": choices.K_PER_TIMEPOINT})})
    return [
        {"name": "filter_genes", "group": "prep",
         "params": {"min_cells": choices.GENE_FILTER_MIN_CELLS}},
        {"name": "normalize", "group": "prep",
         "params": {"target_sum": choices.NORMALIZE_TARGET_SUM}},
        {"name": "transform", "group": "prep", "params": {"method": choices.TRANSFORM_METHOD}},
        {"name": "pca", "group": "latent", "params": {"n_components": choices.PCA_COMPONENTS},
         "limits": {"n_components": ["n_samples", "n_features"]}},
        {"name": "phate", "group": "latent", "params": {"n_components": choices.PHATE_COMPONENTS}},
        {"name": "dpt", "group": "analysis", "params": {}},
        time_step,
        {"name": "mioflow", "group": "lightning", "params": {}},
    ]


def workflow(time_mode: str) -> dict:
    return {"name": f"cflows-{time_mode}", "time_mode": time_mode, "steps": steps(time_mode)}


# ── parameters ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Param:
    kind: str                       # "int" | "number" | "int_or_auto" | "choice"
    default: Any = None             # value, or callable (n_cells, smoke) -> value
    source: str = "engine default"  # where the default comes from
    min: Optional[float] = None
    max: Optional[float] = None
    above: Optional[float] = None   # exclusive lower bound
    nullable: bool = False
    required: bool = False          # no default: the expert must set it
    below_cells: bool = False       # must be < n_cells
    at_most_cells: bool = False     # must be <= n_cells
    options: tuple = ()             # for kind "choice"
    doc: str = ""

    def resolve_default(self, n_cells: Optional[int], smoke: bool) -> Any:
        return self.default(n_cells, smoke) if callable(self.default) else self.default


_ML = "manylatents 0.1.7"
_MR = "manyruns 0.1.0"
_LAB = "geomancer choices.py"

#: What each step offers to change. Engine defaults are checked against the pinned signatures
#: by `tests/test_training_workflow.py`, so a repin that moves one fails loudly.
PARAMS: dict[str, dict[str, Param]] = {
    "filter_genes": {
        "min_cells": Param("int", 3, f"{_MR} prep.filter_genes", min=0,
                           doc="keep genes detected in at least this many cells"),
    },
    "normalize": {
        "target_sum": Param("number", 1e4, f"{_MR} prep.normalize", above=0,
                            doc="library size each cell is scaled to"),
    },
    "transform": {
        "method": Param("choice", "log1p", f"{_MR} prep.transform", options=("log1p", "sqrt"),
                        doc="variance-stabilising transform"),
    },
    "pca": {
        "n_components": Param("int", 2, f"{_ML} PCAModule", min=1,
                              doc="principal components kept (clamped to the data's size)"),
    },
    "phate": {
        "n_components": Param("int", 2, f"{_ML} PHATEModule", min=1, doc="embedding dimensions"),
        "knn": Param("int", 5, f"{_ML} PHATEModule", min=1, below_cells=True,
                     doc="nearest neighbours for the kernel"),
        "decay": Param("number", 40, f"{_ML} PHATEModule", above=0, nullable=True,
                       doc="alpha-decay of the kernel; null disables decay"),
        "t": Param("int_or_auto", 15, f"{_ML} PHATEModule", min=1,
                   doc="diffusion time, or 'auto' (von Neumann entropy knee)"),
        "gamma": Param("number", 1, f"{_ML} PHATEModule", min=-1, max=1,
                       doc="informational distance constant"),
        "n_pca": Param("int", 100, f"{_ML} PHATEModule", min=1, nullable=True,
                       doc="PCA components inside PHATE; null skips it"),
        "n_landmark": Param("int", 2000, f"{_ML} PHATEModule", min=1, nullable=True,
                            doc="landmarks for the diffusion operator; null is exact"),
    },
    "dpt": {
        "n_neighbors": Param("int", lambda n, _s: min(15, max(2, (n or 16) - 1)),
                             f"{_MR} steps._step_dpt: min(15, n_cells - 1)", min=2,
                             below_cells=True, doc="neighbours for the diffusion graph"),
        "n_dcs": Param("int", 2, f"{_MR} steps._step_dpt", min=1,
                       doc="diffusion components DPT accumulates over"),
    },
    "discretize_time": {
        "n_timepoints": Param("int", 5, f"{_MR} steps._step_discretize_time", min=2,
                              at_most_cells=True, doc="equal-width pseudotime bins"),
    },
    "subcluster_time": {
        "k_per_timepoint": Param("int", None, _LAB, min=1, required=True,
                                 doc="k-means subclusters within each real timepoint"),
    },
    "mioflow": {
        "n_local_epochs": Param("int", lambda _n, s: 0 if s else 10,
                                f"{_MR} mioflow._resolve_kwargs (0 with --smoke)", min=0),
        "n_global_epochs": Param("int", lambda _n, s: 2 if s else 50,
                                 f"{_MR} mioflow._resolve_kwargs (2 with --smoke)", min=0),
        "n_post_local_epochs": Param("int", 0, f"{_ML} MIOFlow", min=0),
        "sample_size": Param("int", 256, f"{_MR} mioflow._resolve_kwargs", min=1, nullable=True,
                             doc="points sampled per timepoint; null uses every point"),
        "hidden_dim": Param("int", 16, f"{_MR} mioflow._resolve_kwargs", min=1,
                            doc="ODE network width"),
        "lambda_ot": Param("number", 1.0, f"{_ML} MIOFlow", min=0),
        "lambda_energy": Param("number", 0.01, f"{_ML} MIOFlow", min=0),
        "lambda_density": Param("number", 0.0, f"{_ML} MIOFlow", min=0),
        "energy_time_steps": Param("int", 10, f"{_ML} MIOFlow", min=1),
        "n_trajectories": Param("int", 100, f"{_ML} MIOFlow", min=1),
        "n_bins": Param("int", 100, f"{_ML} MIOFlow", min=1),
    },
}


class ParamError(ValueError):
    """An invalid parameter, refused before anything runs."""


def coerce(raw: str) -> Any:
    """`"40"` -> 40, `"0.5"` -> 0.5, `"true"` -> True, `"null"` -> None, else the string."""
    low = raw.strip().lower()
    if low in ("null", "none"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def parse_assignments(tokens: list[str]) -> dict:
    """`["knn=10", "t=auto"]` -> `{"knn": 10, "t": "auto"}`. Every token must parse."""
    out: dict[str, Any] = {}
    for tok in tokens:
        key, sep, value = tok.partition("=")
        if not sep or not key or value == "":
            raise ParamError(f"cannot read {tok!r}: write parameters as key=value, e.g. knn=10")
        if key in out:
            raise ParamError(f"parameter {key!r} given twice")
        out[key] = coerce(value)
    return out


def looks_like_assignment(text: str) -> bool:
    toks = text.split()
    return bool(toks) and all("=" in t and not t.startswith("=") for t in toks)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    import math

    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def check_value(step: str, name: str, value: Any, spec: Param, n_cells: Optional[int]) -> None:
    where = f"{step}.{name}"
    if value is None:
        if spec.required:
            raise ParamError(f"{where} has no default and must be set ({spec.doc}), "
                             f"e.g. {name}=3")
        if not spec.nullable:
            raise ParamError(f"{where} cannot be null")
        return
    if spec.kind == "choice":
        if value not in spec.options:
            raise ParamError(f"{where} must be one of {', '.join(spec.options)}, got {value!r}")
        return
    if spec.kind == "int_or_auto" and value == "auto":
        return
    if spec.kind in ("int", "int_or_auto") and not _is_int(value):
        raise ParamError(f"{where} must be an integer"
                         + (" or 'auto'" if spec.kind == "int_or_auto" else "")
                         + f", got {value!r}")
    if spec.kind == "number" and not _is_number(value):
        raise ParamError(f"{where} must be a finite number, got {value!r}")
    if spec.min is not None and value < spec.min:
        raise ParamError(f"{where} must be >= {spec.min:g}, got {value!r}")
    if spec.max is not None and value > spec.max:
        raise ParamError(f"{where} must be <= {spec.max:g}, got {value!r}")
    if spec.above is not None and value <= spec.above:
        raise ParamError(f"{where} must be > {spec.above:g}, got {value!r}")
    if n_cells is not None and spec.below_cells and value >= n_cells:
        raise ParamError(f"{where} must be < the {n_cells} cells in the input, got {value!r}")
    if n_cells is not None and spec.at_most_cells and value > n_cells:
        raise ParamError(f"{where} must be <= the {n_cells} cells in the input, got {value!r}")


def validate(step: dict, overrides: dict, n_cells: Optional[int]) -> None:
    """Refuse unknown names, wrong types, out-of-range and missing required values."""
    name = step["name"]
    specs = PARAMS.get(name, {})
    for key in overrides:
        if key not in specs:
            offered = ", ".join(specs) or "none"
            raise ParamError(f"{name} has no parameter {key!r}; offered: {offered}")
    merged = {**dict(step.get("params") or {}), **overrides}
    for key, spec in specs.items():
        if key in merged:
            check_value(name, key, merged[key], spec, n_cells)
        elif spec.required:
            check_value(name, key, None, spec, n_cells)


def resolve(step_name: str, params: dict, n_cells: Optional[int], smoke: bool) -> dict:
    """Every offered parameter with its effective value and where that value came from."""
    out = {k: {"value": v, "source": "requested"} for k, v in params.items()}
    for key, spec in PARAMS.get(step_name, {}).items():
        if key not in out:
            out[key] = {"value": spec.resolve_default(n_cells, smoke), "source": spec.source}
    return out


def diff(old: Optional[dict], new: dict) -> dict:
    """`{name: [old, new]}` for every resolved value that differs."""
    if old is None:
        return {}
    keys = sorted(set(old) | set(new))
    return {k: [(old.get(k) or {}).get("value"), (new.get(k) or {}).get("value")]
            for k in keys if (old.get(k) or {}).get("value") != (new.get(k) or {}).get("value")}


def rule(p: Param) -> str:
    """A one-line description of a parameter's legal values."""
    if p.kind == "choice":
        return "one of " + ", ".join(p.options)
    bits = [p.kind.replace("_", " ")]
    if p.min is not None:
        bits.append(f">= {p.min:g}")
    if p.above is not None:
        bits.append(f"> {p.above:g}")
    if p.max is not None:
        bits.append(f"<= {p.max:g}")
    if p.below_cells:
        bits.append("< n_cells")
    if p.at_most_cells:
        bits.append("<= n_cells")
    if p.nullable:
        bits.append("or null")
    if p.required:
        bits.append("REQUIRED")
    return ", ".join(bits)
