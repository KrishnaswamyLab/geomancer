"""Cheap per-step numbers shown beside each plot and read by the feedback rules.

Each one backs a specific rule or question; anything no rule uses stays out. All are O(n·k)
or cheaper — no pairwise distance matrices.

    phate            knn_components, largest_component_share, knn_disconnected,
                     time_silhouette (only when measured time exists)
    dpt              unreachable_frac, geodesic_agreement (computed by manyruns already)
    discretize_time  bins_populated, min/max cells per bin
    subcluster_time  bins, min/max cells per bin (computed by `timebins`)
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def compute(step: str, *, emb: Any = None, time_labels: Any = None, g: Optional[dict] = None,
            knn: Optional[int] = None, source_time: Any = None) -> dict:
    g = g or {}
    if step == "phate" and emb is not None:
        return phate(np.asarray(emb, dtype=float), knn or 5, source_time)
    if step == "dpt":
        return {k: g[src] for k, src in (("dpt.unreachable_frac", "dpt_unreachable_frac"),
                                         ("dpt.geodesic_agreement", "dpt_geodesic_agreement"))
                if src in g}
    if step == "discretize_time" and time_labels is not None:
        _, counts = np.unique(np.asarray(time_labels).astype(str), return_counts=True)
        return {"discretize_time.bins_populated": int(len(counts)),
                "discretize_time.min_cells_per_bin": int(counts.min()),
                "discretize_time.max_cells_per_bin": int(counts.max())}
    if step == "subcluster_time":
        return {k: v for k, v in g.items()
                if k.startswith("subcluster_time.") and not isinstance(v, dict)}
    return {}


def phate(emb: np.ndarray, knn: int, source_time: Any = None) -> dict:
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import kneighbors_graph

    n = len(emb)
    k = int(max(1, min(knn, n - 1)))
    graph = kneighbors_graph(emb, n_neighbors=k, mode="connectivity")
    n_comp, labels = connected_components(graph, directed=False)
    largest = np.bincount(labels).max() / n
    out = {"phate.knn_components": int(n_comp),
           "phate.largest_component_share": round(float(largest), 4),
           "phate.knn_disconnected": bool(n_comp > 1)}
    if source_time is not None:
        times = np.asarray(source_time).astype(str)
        if 1 < len(set(times)) < n:
            from sklearn.metrics import silhouette_score

            sample = min(n, 2000)  # silhouette is O(sample^2); a fixed sample bounds it
            idx = np.random.default_rng(0).choice(n, sample, replace=False) if sample < n else slice(None)
            out["phate.time_silhouette"] = round(float(silhouette_score(emb[idx], times[idx])), 4)
    return out
