"""States: what a policy sees about a sample before any action runs.

Each function maps points (n x d) to a fixed-length vector. They are cheap, read
the input only, and never look at the truth graph.
"""

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

K = 10
SPECTRUM = 5


def _none(points):
    return np.zeros(0)


def _lid_mle(distances):
    # Levina & Bickel (2004) maximum-likelihood local intrinsic dimension, per point.
    d = np.maximum(distances[:, 1:], 1e-12)
    ratios = np.log(d[:, -1:] / d[:, :-1])
    return (d.shape[1] - 1) / np.maximum(ratios.sum(1), 1e-12)


def _input_geometry(points):
    n, dim = points.shape
    k = min(K, n - 1)
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points)
    lid = _lid_mle(distances)
    n_components, labels = connected_components(
        kneighbors_graph(points, k, include_self=False), directed=False)
    largest = np.bincount(labels).max() / n
    spread = distances[:, -1] / max(np.median(distances[:, -1]), 1e-12)
    return np.array([
        np.log(n), dim,
        np.median(lid), np.subtract(*np.percentile(lid, [75, 25])),
        n_components, largest,
        np.percentile(spread, 90),
    ], dtype=float)


def _pca_spectrum(points):
    centered = points - points.mean(0)
    values = np.linalg.svd(centered, compute_uv=False) ** 2
    ratios = values / max(values.sum(), 1e-12)
    return np.pad(ratios[:SPECTRUM], (0, max(0, SPECTRUM - ratios.size)))


STATES = {"none": _none, "input_geometry": _input_geometry, "pca_spectrum": _pca_spectrum}
