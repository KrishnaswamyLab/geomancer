import json
import types

import numpy as np
import pytest

pytest.importorskip("scanpy")

from geomancer.learner import convert, provenance

CLASSES = ["archetypal", "multi_branch", "simple_traj", "clusters"]


def _synthetic_root(tmp_path, n=3):
    folder = tmp_path / "data" / "synthetic"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        stem = folder / f"sample_{i:05d}"
        np.savez(f"{stem}.npz", noisy_points=rng.normal(size=(40 + i, 3)),
                 clean_points=rng.normal(size=(40 + i, 3)), nodes=np.arange(2),
                 edges=np.array([[0, 1]]), positions=rng.normal(size=(2, 3)))
        (folder / f"sample_{i:05d}.json").write_text(json.dumps({"id": i}))
    return tmp_path


def _sha(path):
    return provenance.sha256_file(path)


def test_synthetic_sample_becomes_its_noisy_points_with_hashes_and_revision(tmp_path):
    root = _synthetic_root(tmp_path / "root")
    out = tmp_path / "out"

    manifest = convert.convert_synthetic(root, out)

    assert manifest["counts"] == {"ok": 3, "error": 0}
    assert manifest["code_revision"]["commit"] == provenance.geomancer_revision()["commit"]
    entry = manifest["samples"]["sample_00001"]
    npz = root / "data" / "synthetic" / "sample_00001.npz"
    points = np.load(out / "synthetic" / "sample_00001.npy")
    with np.load(npz) as arrays:
        assert np.array_equal(points, arrays["noisy_points"])
    assert entry["status"] == "ok"
    assert entry["npz_sha256"] == _sha(npz)
    assert entry["npy_sha256"] == _sha(out / "synthetic" / "sample_00001.npy")
    assert entry["shape"] == [41, 3]
    on_disk = json.loads((out / "synthetic" / "manifest.json").read_text())
    assert on_disk == json.loads(json.dumps(manifest))


def test_an_unreadable_synthetic_sample_is_kept_and_counted_as_an_error(tmp_path):
    root = _synthetic_root(tmp_path / "root")
    (root / "data" / "synthetic" / "sample_00002.npz").write_bytes(b"not an npz")

    manifest = convert.convert_synthetic(root, tmp_path / "out")

    assert manifest["counts"] == {"ok": 2, "error": 1}
    entry = manifest["samples"]["sample_00002"]
    assert entry["status"] == "error" and entry["error"]
    assert not (tmp_path / "out" / "synthetic" / "sample_00002.npy").exists()


def _real_root(tmp_path, n_cells=120, n_genes=40):
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    folder = tmp_path / "data" / "scRNAseq"
    folder.mkdir(parents=True)
    for i, kind in enumerate(["X_pca", "counts", "counts"], 1):
        counts = rng.poisson(2.0, (n_cells, n_genes)).astype(np.float32)
        X = counts if kind == "counts" else np.log1p(counts)
        adata = ad.AnnData(sp.csr_matrix(X))
        if kind == "X_pca":
            adata.obsm["X_pca"] = rng.normal(size=(n_cells, 50)).astype(np.float32)
        adata.write_h5ad(folder / f"SCD-{i:04d}.h5ad")
    # SCD-0003 is present but not high_support; SCD-0999 is high_support but absent.
    labels = {d: {c: int(c == "clusters") for c in CLASSES} for d in ("SCD-0001", "SCD-0002", "SCD-0999")}
    return types.SimpleNamespace(
        list_real=lambda root=None: ["SCD-0001", "SCD-0002", "SCD-0003"],
        real_path=lambda d, root=None: folder / f"{d}.h5ad",
        load_shape_labels=lambda aggregation="high_support", root=None: labels,
    )


def test_real_points_are_one_computed_pca_on_a_seeded_subsample(tmp_path, monkeypatch):
    loader = _real_root(tmp_path / "root")
    monkeypatch.setattr(convert, "_loader", lambda: loader)
    out = tmp_path / "out"

    manifest = convert.convert_real(tmp_path / "root", out, max_cells=50, n_pcs=10)
    again = convert.convert_real(tmp_path / "root", tmp_path / "again", max_cells=50, n_pcs=10)

    # Only the high_support datasets that are present.
    assert sorted(manifest["samples"]) == ["SCD-0001", "SCD-0002"]
    assert manifest["counts"] == {"ok": 2, "error": 0}
    assert manifest["params"] == {"datasets": "high_support", "max_cells": 50, "n_pcs": 10,
                                  "hvg": convert.HVG, "seed": convert.SEED,
                                  "method": convert.REAL_METHOD}
    for d in ("SCD-0001", "SCD-0002"):
        entry = manifest["samples"][d]
        points = np.load(out / "real" / f"{d}.npy")
        assert points.shape == (50, 10) and np.isfinite(points).all()
        assert np.array_equal(points, np.load(tmp_path / "again" / "real" / f"{d}.npy"))
        assert entry["source_sha256"] == _sha(loader.real_path(d))
        assert entry["npy_sha256"] == _sha(out / "real" / f"{d}.npy")
        assert entry["n_obs"] == 120 and entry["n_points"] == 50
        assert entry["npy_sha256"] == again["samples"][d]["npy_sha256"]
    # The stored X_pca is never used: every dataset goes through the same PCA.
    assert manifest["samples"]["SCD-0001"]["counts"] is False
    assert manifest["samples"]["SCD-0002"]["counts"] is True
    assert manifest["code_revision"]["commit"] == provenance.geomancer_revision()["commit"]


def test_the_subsample_is_the_one_the_old_real_table_drew(tmp_path):
    # Same seed and draw as table.real_points, so the cells match the earlier real table.
    import anndata as ad

    loader = _real_root(tmp_path / "root")
    path = loader.real_path("SCD-0002")

    points, meta = convert.real_points(path, max_cells=50, n_pcs=10)

    n = ad.read_h5ad(path).n_obs
    idx = np.sort(np.random.default_rng(convert.SEED).choice(n, 50, replace=False))
    X = ad.read_h5ad(path)[idx].X
    assert np.allclose(points, convert.computed_pca(X, 10, convert.SEED))
    assert meta == {"n_obs": 120, "n_points": 50, "n_vars": 40, "counts": True}


def test_cli_converts_synthetic_into_the_default_grid_inputs_dir(tmp_path, monkeypatch):
    root = _synthetic_root(tmp_path / "root")

    convert.main([f"root={root}", "source=synthetic", "limit=2"])

    manifest = json.loads((root / "grid_inputs" / "synthetic" / "manifest.json").read_text())
    assert sorted(manifest["samples"]) == ["sample_00000", "sample_00001"]


def test_revision_reports_the_checkout_commit_or_the_env_override(monkeypatch):
    import subprocess
    from pathlib import Path

    repo = Path(provenance.__file__).resolve().parents[2]
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    monkeypatch.delenv("GEOMANCER_GIT_SHA", raising=False)

    assert provenance.geomancer_revision()["commit"] == head
    monkeypatch.setenv("GEOMANCER_GIT_SHA", "abc123")
    assert provenance.geomancer_revision() == {"commit": "abc123", "dirty": None,
                                               "source": "GEOMANCER_GIT_SHA"}
