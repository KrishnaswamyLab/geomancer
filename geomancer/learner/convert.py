"""Write the grid's input points: one ``.npy`` per sample and a hashed manifest.

    python -m geomancer.learner.convert source=synthetic
    python -m geomancer.learner.convert source=real workers=4

The grid runs on arrays, so each sample is converted once, here, and every run
cites the ``.npy`` by sha256. Output goes under ``out`` (default
``<root>/grid_inputs``, outside the repo): ``synthetic/<sample>.npy`` and
``real/<dataset>.npy``, each directory with one ``manifest.json``.

Synthetic: ``<sample>.npy`` is the ``noisy_points`` of ``sample_XXXXX.npz``
unchanged. The manifest has the sha256 of the ``.npz`` (which also holds the
truth graph that score.py reads), the ``.json`` meta and the ``.npy``.

Real: the high_support datasets that are present. Each is a seeded uniform
subsample of at most ``max_cells`` cells, then ONE PCA method for every dataset
(a stored ``obsm["X_pca"]`` is never used; 16 of the 79 have none). The manifest
has the sha256 of the source h5ad and of the ``.npy``, the params, and n_obs.

Both manifests carry the geomancer code revision. A sample that fails is an
entry with status "error" and its error, and is counted; it is never dropped.
"""

import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from geomancer.learner import provenance
from geomancer.learner.table import override

config = {
    "source": "synthetic",  # synthetic, real, or all
    "root": None,  # scShapeBench directory; None means <MANYLATENTS_DATA>/scshapebench
    "out": None,  # None means <root>/grid_inputs
    "limit": None,
    "datasets": "high_support",  # real: the label set whose datasets are converted
    "max_cells": 5000,
    "n_pcs": 50,
    "workers": 1,
}

SEED = 0
HVG = 2000
REAL_METHOD = ("seeded uniform subsample; normalize_total + log1p if X holds integer counts; "
               "top-HVG genes (scanpy seurat flavor) when n_vars > hvg; sklearn PCA")
VERSIONS = ("numpy", "scipy", "scikit-learn", "anndata", "scanpy", "manylatents-omics")


def _loader():
    from manylatents.singlecell.data import scshapebench

    return scshapebench


def _save(path, array):
    # Write then rename, so a killed run never leaves a truncated .npy behind.
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.ascontiguousarray(array))
    os.replace(tmp, path)


def _write_manifest(folder, manifest):
    tmp = folder / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, folder / "manifest.json")


def _manifest(kind, root, params, samples, seconds):
    status = [entry["status"] for entry in samples.values()]
    return {
        "kind": kind,
        "root": str(root),
        "params": params,
        "code_revision": provenance.geomancer_revision(),
        "versions": provenance.versions(VERSIONS),
        "counts": {"ok": status.count("ok"), "error": status.count("error")},
        "seconds": round(seconds, 2),
        "samples": samples,
    }


def convert_synthetic(root, out, limit=None):
    """``<out>/synthetic/<sample>.npy`` = noisy_points, and its manifest. Returns the manifest."""
    root, folder = Path(root), Path(out) / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    ids = _synthetic_ids(root)[:limit]
    started, samples = time.time(), {}
    for sample_id in ids:
        name = f"sample_{sample_id:05d}"
        npz = root / "data" / "synthetic" / f"{name}.npz"
        meta = npz.with_suffix(".json")
        try:
            with np.load(npz, allow_pickle=False) as arrays:
                points = arrays["noisy_points"]
            target = folder / f"{name}.npy"
            _save(target, points)
            samples[name] = {
                "status": "ok", "npz": str(npz), "npz_sha256": provenance.sha256_file(npz),
                "meta_sha256": provenance.sha256_file(meta) if meta.is_file() else None,
                "npy": target.name, "npy_sha256": provenance.sha256_file(target),
                "shape": list(points.shape), "dtype": str(points.dtype),
            }
        except Exception as exc:
            samples[name] = {"status": "error", "npz": str(npz), "error": f"{type(exc).__name__}: {exc}"}
    manifest = _manifest("synthetic", root, {"array": "noisy_points"}, samples, time.time() - started)
    _write_manifest(folder, manifest)
    return manifest


def _synthetic_ids(root):
    # The loader's listing, so the grid sees exactly the samples the loader would.
    return _loader().list_synthetic(root=root)


def is_counts(X):
    """True when the first 10,000 stored values of X are all integers."""
    import scipy.sparse as sp

    values = X.data if sp.issparse(X) else np.asarray(X).ravel()
    values = values[:10_000]
    return bool(values.size) and bool(np.all(values == np.round(values)))


def computed_pca(X, n_pcs, seed=SEED):
    """The one PCA method for every real dataset (see REAL_METHOD)."""
    import anndata as ad
    import scanpy as sc
    from sklearn.decomposition import PCA

    adata = ad.AnnData(X.astype(np.float32))
    if is_counts(X):
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    if adata.n_vars > HVG:
        sc.pp.highly_variable_genes(adata, n_top_genes=HVG)
        adata = adata[:, adata.var["highly_variable"]]
    dense = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    return PCA(n_components=min(n_pcs, *dense.shape), random_state=seed).fit_transform(dense)


def real_points(path, max_cells, n_pcs, seed=SEED):
    """One h5ad's seeded uniform subsample of cells, as ``computed_pca`` components.

    The draw is the one the earlier out-of-band real table made (seed 0, sorted indices),
    so the cells are the same.
    """
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        n = adata.n_obs
        idx = np.sort(np.random.default_rng(seed).choice(n, min(n, max_cells), replace=False))
        X = adata[idx].to_memory().X
    finally:
        adata.file.close()
    meta = {"n_obs": n, "n_points": len(idx), "n_vars": int(X.shape[1]), "counts": is_counts(X)}
    return computed_pca(X, n_pcs, seed), meta


def _real_work(job):
    dataset_id, path, folder, max_cells, n_pcs = job
    started = time.time()
    try:
        points, meta = real_points(path, max_cells, n_pcs)
        target = Path(folder) / f"{dataset_id}.npy"
        _save(target, points.astype(np.float64))
        return dataset_id, {
            "status": "ok", "source": str(path), "source_sha256": provenance.sha256_file(path),
            "npy": target.name, "npy_sha256": provenance.sha256_file(target),
            "shape": list(points.shape), **meta, "seconds": round(time.time() - started, 2),
        }
    except Exception as exc:
        traceback.print_exc()
        return dataset_id, {"status": "error", "source": str(path),
                            "error": f"{type(exc).__name__}: {exc}"}


def convert_real(root, out, datasets="high_support", max_cells=5000, n_pcs=50, limit=None, workers=1):
    """``<out>/real/<dataset>.npy`` for the ``datasets`` label set present under ``root``."""
    loader, root, folder = _loader(), Path(root), Path(out) / "real"
    folder.mkdir(parents=True, exist_ok=True)
    labels = loader.load_shape_labels(datasets, root=root)
    ids = sorted(set(loader.list_real(root=root)) & set(labels))[:limit]
    jobs = [(d, str(loader.real_path(d, root=root)), str(folder), max_cells, n_pcs) for d in ids]
    started = time.time()
    results = map(_real_work, jobs) if workers == 1 else ProcessPoolExecutor(workers).map(_real_work, jobs)
    samples = {}
    for done, (dataset_id, entry) in enumerate(results, 1):
        samples[dataset_id] = entry
        print(f"[{done}/{len(jobs)}] {dataset_id} {entry['status']} "
              f"elapsed={time.time() - started:.0f}s", flush=True)
    params = {"datasets": datasets, "max_cells": max_cells, "n_pcs": n_pcs, "hvg": HVG,
              "seed": SEED, "method": REAL_METHOD}
    manifest = _manifest("real", root, params, samples, time.time() - started)
    _write_manifest(folder, manifest)
    return manifest


def main(argv=None):
    cfg = override(config, sys.argv[1:] if argv is None else argv)
    if cfg["source"] not in ("synthetic", "real", "all"):
        raise ValueError(f"unknown source {cfg['source']!r}; expected synthetic, real or all")
    root = Path(cfg["root"]) if cfg["root"] else _loader().scshapebench_root()
    out = Path(cfg["out"]) if cfg["out"] else root / "grid_inputs"
    for source in ("synthetic", "real"):
        if cfg["source"] not in (source, "all"):
            continue
        if source == "synthetic":
            manifest = convert_synthetic(root, out, cfg["limit"])
        else:
            manifest = convert_real(root, out, cfg["datasets"], cfg["max_cells"], cfg["n_pcs"],
                                    cfg["limit"], cfg["workers"])
        print(f"{source}: {manifest['counts']} in {manifest['seconds']}s -> {out / source}", flush=True)


if __name__ == "__main__":
    main()
