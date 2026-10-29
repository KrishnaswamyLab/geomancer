"""Actions: an embedding, then a graph extractor, at one scale.

`ACTIONS` maps a name such as ``"phate+paga@coarse"`` to a function from points
(n x d) to a scShapeBench graph payload. The embedding only chooses where the
graph is built; node positions are always means of the original points, which is
the space the evaluator scores in. The graph builders follow scShapeBench's
method benchmark (synthetic/method_benchmark/run_methods.py), and the
``default`` scale uses its defaults.
"""

import hashlib
import math
from collections import OrderedDict
from functools import partial

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

DIM = 2
SEED = 0


def _raw(points):
    return points


def _pca(points):
    return PCA(n_components=min(DIM, *points.shape), random_state=SEED).fit_transform(points)


def _diffmap(points, k=10, t=1):
    # Coifman & Lafon (2006): Gaussian kernel with a per-point bandwidth (the
    # distance to the k-th neighbour), density-normalised, then the top
    # nontrivial eigenvectors scaled by eigenvalue**t.
    n = points.shape[0]
    k = min(k, n - 1)
    distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    sigma = np.sort(distances, axis=1)[:, k]
    kernel = np.exp(-(distances**2) / np.maximum(np.outer(sigma, sigma), 1e-12))
    q = kernel.sum(1)
    kernel = kernel / np.outer(q, q)
    d = kernel.sum(1)
    symmetric = kernel / np.sqrt(np.outer(d, d))
    values, vectors = eigh(symmetric)
    order = np.argsort(values)[::-1][1 : DIM + 1]
    psi = vectors[:, order] / np.sqrt(d)[:, None]
    return psi * values[order] ** t


def _phate(points):
    import phate

    knn = min(5, points.shape[0] - 1)
    op = phate.PHATE(n_components=DIM, knn=knn, random_state=SEED, verbose=0, n_jobs=1)
    return op.fit_transform(points)


EMBEDDINGS = {"raw": _raw, "pca": _pca, "diffmap": _diffmap, "phate": _phate}

SCALES = {
    "fine": {"knn": 10, "resolution": 2.0, "cubes": 10, "eps_scale": 1.0},
    "default": {"knn": 15, "resolution": 1.0, "cubes": 6, "eps_scale": 1.5},
    "coarse": {"knn": 30, "resolution": 0.5, "cubes": 4, "eps_scale": 2.5},
}


def _payload(points, labels, edges):
    nodes = sorted(set(labels.tolist()))
    index = {node: i for i, node in enumerate(nodes)}
    return {
        "graph_representation": "simple_graph",
        "nodes": list(range(len(nodes))),
        "edges": sorted({tuple(sorted((index[u], index[v]))) for u, v in edges if u != v}),
        "node_positions": {
            str(index[node]): points[labels == node].mean(axis=0).tolist() for node in nodes
        },
    }


def _paga(coords, points, scale):
    import anndata as ad
    import scanpy as sc

    sc.settings.verbosity = 0
    adata = ad.AnnData(np.asarray(coords, dtype=float))
    knn = min(max(scale["knn"], 2), points.shape[0] - 1)
    sc.pp.neighbors(adata, n_neighbors=knn, use_rep="X")
    sc.tl.leiden(adata, resolution=scale["resolution"], random_state=SEED, key_added="leiden",
                 flavor="igraph", n_iterations=2, directed=False)
    sc.tl.paga(adata, groups="leiden")
    labels = adata.obs["leiden"].astype(int).to_numpy()
    connectivities = adata.uns["paga"]["connectivities"].tocoo()
    edges = [(int(u), int(v)) for u, v, w in
             zip(connectivities.row, connectivities.col, connectivities.data) if w > 0.0]
    return _payload(points, labels, edges)


def _mapper_eps(points, min_samples, eps_scale):
    k = min(max(min_samples, 2), points.shape[0])
    distances, _ = NearestNeighbors(n_neighbors=k).fit(points).kneighbors(points)
    base = float(np.median(distances[:, -1]))
    if not math.isfinite(base) or base <= 0.0:
        positive = pdist(points)
        positive = positive[positive > 0.0]
        base = float(np.median(positive)) if positive.size else 1.0
    return max(eps_scale * base, np.finfo(float).eps)


def _mapper(coords, points, scale, min_samples=3, overlap=0.35):
    import kmapper as km

    mapper = km.KeplerMapper(verbose=0)
    lens = np.asarray(coords, dtype=float)[:, :DIM]
    graph = mapper.map(
        lens,
        X=points,
        clusterer=DBSCAN(eps=_mapper_eps(points, min_samples, scale["eps_scale"]),
                         min_samples=min_samples),
        cover=km.Cover(n_cubes=scale["cubes"], perc_overlap=overlap),
        remove_duplicate_nodes=True,
    )
    keys = sorted(graph.get("nodes", {}))
    if not keys:
        return {"graph_representation": "simple_graph", "nodes": [0], "edges": [],
                "node_positions": {"0": points.mean(axis=0).tolist()}}
    index = {key: i for i, key in enumerate(keys)}
    edges = sorted({tuple(sorted((index[u], index[v])))
                    for u, targets in graph.get("links", {}).items() if u in index
                    for v in targets if v in index and v != u})
    return {
        "graph_representation": "simple_graph",
        "nodes": list(range(len(keys))),
        "edges": [list(edge) for edge in edges],
        "node_positions": {str(index[key]): points[graph["nodes"][key]].mean(axis=0).tolist()
                           for key in keys},
    }


GRAPHS = {"paga": _paga, "mapper": _mapper}

# The last few embeddings, so the six graph/scale actions on one sample embed once.
_CACHE: OrderedDict = OrderedDict()
_CACHE_SIZE = 2 * len(EMBEDDINGS)


def embed(name, points):
    """Embedding `name` of `points`, cached by content."""
    key = (name, points.shape, hashlib.sha1(np.ascontiguousarray(points).tobytes()).hexdigest())
    if key not in _CACHE:
        _CACHE[key] = EMBEDDINGS[name](points)
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return _CACHE[key]


def _action(embedding, graph, scale, points):
    points = np.asarray(points, dtype=float)
    payload = GRAPHS[graph](embed(embedding, points), points, SCALES[scale])
    payload["edges"] = [list(edge) for edge in payload["edges"]]
    return payload


ACTIONS = {
    f"{e}+{g}@{s}": partial(_action, e, g, s)
    for e in EMBEDDINGS for g in GRAPHS for s in SCALES
}


def run(name, points):
    """Run one action by name."""
    return ACTIONS[name](points)
