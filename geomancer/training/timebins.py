"""Refined time bins around real timepoints (CFlows Methods, lines 626–633).

Within each measured timepoint, cells are subclustered on their PHATE coordinates, the
subclusters are ordered (by default by mean DPT, the reading of "local pseudotime ordering"),
and each subcluster gets a refined time inside a symmetric window around the measured time,
shifted by one constant if needed so no time is negative (recorded as `offset`).
The window's half-width is half the smaller gap to a neighbouring timepoint, so refined bins
from adjacent timepoints never overlap and each measured time stays at its window's centre.

Not implemented from the paper: resampling an equal number of cells per subcluster. It would
change which cells MIOFlow trains on and needs its own provenance first.
"""
from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

from geomancer.training import choices


def numeric_times(labels: Any) -> np.ndarray:
    """Per-cell numeric time from labels like `0.5`, `"day3"` or `"D12"` (the first number)."""
    out = []
    for raw in np.asarray(labels).astype(str):
        match = re.search(r"-?\d+(?:\.\d+)?", raw)
        if match is None:
            raise ValueError(f"cannot read a time from label {raw!r}; real-time mode needs "
                             "numeric timepoints (e.g. 0, 3, 'day7')")
        out.append(float(match.group()))
    return np.asarray(out, dtype=float)


def _order_mean_dpt(members: list[np.ndarray], pseudotime: np.ndarray) -> list[int]:
    """Subcluster indices sorted by mean DPT; a subcluster with no reachable cell goes last."""
    def key(i: int) -> float:
        values = pseudotime[members[i]]
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else np.inf
    return sorted(range(len(members)), key=key)


#: Orderings by name; `choices.SUBCLUSTER_ORDER` picks one. Add a function to add an option.
ORDERINGS: dict[str, Callable[[list, np.ndarray], list[int]]] = {"mean_dpt": _order_mean_dpt}


def refine(emb: np.ndarray, pseudotime: np.ndarray, time_labels: Any, k: int,
           seed: int) -> tuple[np.ndarray, dict]:
    """Refined per-cell times and diagnostics. Deterministic for a given seed."""
    from sklearn.cluster import KMeans

    emb = np.asarray(emb, dtype=float)
    pseudotime = np.asarray(pseudotime, dtype=float)
    times = numeric_times(time_labels)
    if not (len(emb) == len(pseudotime) == len(times)):
        raise ValueError(f"row counts disagree: embedding {len(emb)}, pseudotime "
                         f"{len(pseudotime)}, time labels {len(times)}")
    levels = np.unique(times)
    if len(levels) < 2:
        raise ValueError("real-time mode needs at least two distinct timepoints")
    order = ORDERINGS[choices.SUBCLUSTER_ORDER]
    refined = np.empty(len(times), dtype=float)
    per_timepoint = {}
    for j, t in enumerate(levels):
        gaps = [g for g in (t - levels[j - 1] if j > 0 else None,
                            levels[j + 1] - t if j + 1 < len(levels) else None) if g is not None]
        half = min(gaps) / 2
        idx = np.flatnonzero(times == t)
        k_eff = int(min(k, len(idx)))
        if k_eff <= 1:
            refined[idx] = t
            per_timepoint[f"{t:g}"] = {"k": 1, "sizes": [len(idx)]}
            continue
        fit = KMeans(n_clusters=k_eff, n_init=10, random_state=int(seed)).fit(emb[idx])
        members = [idx[fit.labels_ == c] for c in range(k_eff)]
        for rank, c in enumerate(order(members, pseudotime)):
            refined[members[c]] = t + half * (2 * (rank + 0.5) / k_eff - 1)
        per_timepoint[f"{t:g}"] = {"k": k_eff, "sizes": sorted(len(m) for m in members)}
    # manyruns reads time with an unsigned-number parser (`vocab.parse_timepoints`): ONE
    # negative label makes it fall back to evenly spaced ranks, erasing the real spacing these
    # bins exist to keep. MIOFlow rescales time to [0, 1], so a constant shift changes nothing.
    offset = float(-refined.min()) if refined.min() < 0 else 0.0
    refined = refined + offset
    sizes = [s for v in per_timepoint.values() for s in v["sizes"]]
    diagnostics = {
        "subcluster_time.timepoints": len(levels),
        "subcluster_time.bins": len(sizes),
        "subcluster_time.min_cells_per_bin": int(min(sizes)),
        "subcluster_time.max_cells_per_bin": int(max(sizes)),
        "subcluster_time.order": choices.SUBCLUSTER_ORDER,
        "subcluster_time.offset": round(offset, 6),
        "subcluster_time.per_timepoint": per_timepoint,
    }
    return refined, diagnostics
