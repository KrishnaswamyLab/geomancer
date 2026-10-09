"""The workflow definition, parameter specs and refined time bins (pure; no session)."""
import inspect

import numpy as np
import pytest

from geomancer.training import timebins, workflow as wf


def test_both_time_modes_share_everything_but_the_time_step():
    pseudo, real = wf.steps(wf.PSEUDOTIME), wf.steps(wf.REAL)
    assert [s["name"] for s in pseudo][:-2] == [s["name"] for s in real][:-2]
    assert pseudo[-2]["name"] == "discretize_time" and real[-2]["name"] == "subcluster_time"
    assert all(s["name"] in wf.PARAMS for s in pseudo + real)
    with pytest.raises(ValueError):
        wf.steps("weekly")


def test_engine_defaults_match_the_pinned_signatures():
    from manylatents.algorithms.latent.pca import PCAModule
    from manylatents.algorithms.latent.phate import PHATEModule
    from manylatents.algorithms.lightning.mioflow import MIOFlow

    for step, cls in (("pca", PCAModule), ("phate", PHATEModule), ("mioflow", MIOFlow)):
        sig = inspect.signature(cls.__init__).parameters
        for name, spec in wf.PARAMS[step].items():
            if spec.source.startswith("manylatents"):
                assert sig[name].default == spec.default, f"{step}.{name}"


def test_assignment_parsing():
    assert wf.parse_assignments(["knn=10", "t=auto", "decay=null", "lambda_ot=0.5"]) == \
        {"knn": 10, "t": "auto", "decay": None, "lambda_ot": 0.5}
    for bad in (["knn"], ["=3"], ["knn="], ["knn=1", "knn=2"]):
        with pytest.raises(wf.ParamError):
            wf.parse_assignments(bad)
    assert wf.looks_like_assignment("knn=10 t=20")
    assert not wf.looks_like_assignment("too blurry, knn=10 maybe")


def test_validation_covers_choices_and_required_values():
    step = {"name": "subcluster_time", "params": {}}
    with pytest.raises(wf.ParamError, match="must be set"):
        wf.validate(step, {}, 100)
    wf.validate(step, {"k_per_timepoint": 3}, 100)
    with pytest.raises(wf.ParamError, match="one of"):
        wf.validate({"name": "transform", "params": {}}, {"method": "log2"}, None)


# ── refined time bins ───────────────────────────────────────────────────────────────────
def _timecourse(seed=0):
    rng = np.random.default_rng(seed)
    times = np.repeat([0.0, 1.0, 3.0], 60)
    emb = np.c_[times + rng.normal(scale=0.3, size=180), rng.normal(size=180)]
    return emb, emb[:, 0].copy(), times.astype(str)


def test_refined_bins_stay_in_symmetric_non_overlapping_windows():
    emb, pt, labels = _timecourse()
    refined, diag = timebins.refine(emb, pt, labels, k=3, seed=1)
    windows = {0.0: 0.5, 1.0: 0.5, 3.0: 1.0}  # half of the smaller neighbouring gap
    times = timebins.numeric_times(labels)
    for t, half in windows.items():
        r = refined[times == t]
        assert r.min() > t - half and r.max() < t + half
        assert np.isclose(np.mean(np.unique(r)), t)  # the measured time is the centre
    assert diag["subcluster_time.bins"] == 9 and diag["subcluster_time.timepoints"] == 3


def test_subclusters_are_ordered_by_mean_dpt_and_deterministic():
    emb, pt, labels = _timecourse()
    refined, _ = timebins.refine(emb, pt, labels, k=3, seed=1)
    times = timebins.numeric_times(labels)
    for t in (0.0, 1.0, 3.0):
        idx = times == t
        means = [pt[idx][refined[idx] == r].mean() for r in np.unique(refined[idx])]
        assert means == sorted(means)
    again, _ = timebins.refine(emb, pt, labels, k=3, seed=1)
    assert np.array_equal(refined, again)


def test_refine_handles_small_timepoints_and_refuses_one_timepoint():
    emb, pt, labels = _timecourse()
    refined, diag = timebins.refine(emb[:62], pt[:62], labels[:62], k=5, seed=0)
    assert diag["subcluster_time.per_timepoint"]["1"]["k"] == 2  # only 2 cells at t=1
    with pytest.raises(ValueError, match="two distinct timepoints"):
        timebins.refine(emb[:60], pt[:60], labels[:60], k=2, seed=0)
    assert list(timebins.numeric_times(["day3", "D12", "0.5"])) == [3.0, 12.0, 0.5]
    with pytest.raises(ValueError, match="cannot read a time"):
        timebins.numeric_times(["early"])
