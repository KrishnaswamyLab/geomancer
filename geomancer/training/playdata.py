"""Synthetic time courses with known ground truth, for play sessions.

Each cell has a latent progress `true_time` in [0, 1]. Cells are measured at a few discrete
timepoints (`obs["timepoint"]`, e.g. `day0`), each a noisy snapshot of progress, so measured
time is coarse and progress is the truth an expert's edits can be scored against. Genes
respond smoothly to progress (rising, falling or transient), and counts are Poisson draws, so
the full preprocessing path applies.

Files are written locally (default `data/play/`, which Git ignores) and are never committed.
`truth.*` scores computed from `true_time` are recorded on each iteration but hidden from the
researcher by default, so play sessions stay blind.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: The pool: variants that differ in how hard the trajectory is to recover. Measured with
#: `make_pool(seed=0)` at the workflow's defaults (geomancer 0.1.0, manyruns v0.1.0); |ρ| is
#: against `true_time`:
#:
#:   variant          PHATE kNN graph   DPT unreachable   DPT |ρ|   PHATE best-axis |ρ|
#:   play_separated   6 components      54%               0.08      0.85
#:   play_noisy       connected          0%               0.91      0.92
#:   play_overlap     2 components       0%               0.77      0.97
#:   play_sparse_time 7 components      69%               0.13      0.66
#:
#: Strong signal or few timepoints separate the snapshots into islands, which DPT cannot
#: order (the failure the CFlows paper describes for Diffusion Pseudotime); the diagnostics
#: flag it (`phate.knn_disconnected`, `dpt.unreachable_frac`), and repairing the graph is the
#: expert's job in those sessions.
VARIANTS = {
    "play_separated":   {"cells": 600, "genes": 300, "timepoints": 4, "spread": 0.05, "depth": 3.0},
    "play_noisy":       {"cells": 600, "genes": 300, "timepoints": 4, "spread": 0.05, "depth": 0.8},
    "play_overlap":     {"cells": 600, "genes": 300, "timepoints": 4, "spread": 0.15, "depth": 2.0},
    "play_sparse_time": {"cells": 600, "genes": 300, "timepoints": 3, "spread": 0.08, "depth": 2.0},
}


def generate(cells: int, genes: int, timepoints: int, spread: float, depth: float,
             seed: int = 0):
    """An AnnData of counts with `obs["timepoint"]` (measured) and `obs["true_time"]`."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    snapshots = np.linspace(0, 1, timepoints)
    which = rng.integers(0, timepoints, cells)
    progress = np.clip(snapshots[which] + rng.normal(0, spread, cells), 0, 1)
    kind = rng.integers(0, 3, genes)            # 0 rising, 1 falling, 2 transient
    centre = rng.uniform(0.2, 0.8, genes)
    width = rng.uniform(0.1, 0.3, genes)
    u = progress[:, None]
    response = np.where(kind == 0, 1 / (1 + np.exp(-(u - centre) / (width / 4))),
               np.where(kind == 1, 1 / (1 + np.exp((u - centre) / (width / 4))),
                        np.exp(-((u - centre) ** 2) / (2 * width ** 2))))
    baseline = rng.lognormal(-1.0, 0.8, genes)
    library = rng.lognormal(0, 0.3, cells)[:, None]
    counts = rng.poisson(library * baseline * np.exp(depth * response)).astype(np.float32)
    days = np.round(snapshots * 10 * (timepoints - 1)).astype(int)
    obs = pd.DataFrame({"timepoint": [f"day{days[w]}" for w in which],
                        "true_time": progress}, index=[f"cell{i}" for i in range(cells)])
    var = pd.DataFrame(index=[f"gene{j}" for j in range(genes)])
    return ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)


def make_pool(out_dir: Path | str = "data/play", seed: int = 0) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (name, spec) in enumerate(VARIANTS.items()):
        path = out / f"{name}.h5ad"
        generate(**spec, seed=seed + i).write_h5ad(path)
        paths.append(path)
    return paths


def truth_scores(step: str, truth: np.ndarray, *, emb=None, pseudotime=None,
                 time_labels=None) -> dict:
    """|Spearman| between a step's output and true progress. Higher is closer to the truth."""
    from scipy.stats import spearmanr

    def rho(x) -> float | None:
        x = np.asarray(x, dtype=float)
        ok = np.isfinite(x)
        if ok.sum() < 3 or np.ptp(x[ok]) == 0:
            return None
        return round(abs(float(spearmanr(x[ok], truth[ok]).statistic)), 4)

    if step in ("pca", "phate", "mioflow") and emb is not None:
        scores = [rho(col) for col in np.asarray(emb).T]
        scores = [s for s in scores if s is not None]
        return {f"truth.{step}_best_axis": max(scores)} if scores else {}
    if step == "dpt" and pseudotime is not None:
        return {"truth.dpt": rho(pseudotime)}
    if step in ("discretize_time", "subcluster_time") and time_labels is not None:
        return {f"truth.{step}": rho(np.asarray(time_labels).astype(float))}
    return {}
