"""The trace contract, export bundles and ground-truth play data.

Sessions here use the mocked compute boundary (`FakeEngine`); the play-data generator and
truth scores are real (numpy/scipy only).
"""
import json

import numpy as np
import pytest

from geomancer.training import contract, execute, export, loop, playdata
from geomancer.training import session as sess, workflow as wf
from training_fakes import FakeEngine


@pytest.fixture
def full(tmp_path, monkeypatch):
    """A session that exercises every event kind."""
    monkeypatch.chdir(tmp_path)
    fake = FakeEngine()
    monkeypatch.setattr(execute, "execute_step", fake)
    project = {"project": "c", "engine": "manylatents", "data_folder": str(tmp_path / "x.h5ad"),
               "dataset_name": None, "data_kwargs": {}, "modality": "scrna"}
    (tmp_path / "x.h5ad").write_bytes(b"stand-in source")
    s = sess.TrainingSession.create("c", project, wf.PSEUDOTIME, source_shape=[30, 5],
                                    expert="e1")
    for step in s.step_names()[:5]:
        s.run(step, select=True)
    with pytest.raises(sess.TrainingError):
        s.run("phate", {"knn": 0})
    s.add_note("phate", 1, "first look")
    loop.feedback_once(s, "phate", 1, "too blurry", accept=1, show_plots=False)
    loop.feedback_once(s, "phate", 1, "too blurry", reject=True)
    answers = iter(["knn=7", "denser", "done"])
    loop.run_loop(s, "phate", read=lambda _p: next(answers), write=lambda *_: None,
                  show_plots=False)
    leave = iter(["done", "none"])  # dpt has nothing kept yet, so done asks
    loop.run_loop(s, "dpt", read=lambda _p: next(leave), write=lambda *_: None,
                  show_plots=False)
    s.select("phate", 2)
    return s


def test_every_emitted_event_meets_the_contract(full):
    rows = full.trace()
    assert {r["event"] for r in rows} == set(contract.EVENTS)
    for row in rows:
        assert contract.problems(row) == [], row["event"]


def test_contract_detects_breaks():
    assert contract.problems({"event": "nonsense"})
    row = {"schema_version": 2, "at": "t", "session_id": "s", "expert": None,
           "event": "feedback", "feedback_id": "f", "kind": "feedback", "step": "phate",
           "iteration": "phate#1", "text": "x", "params": {}, "diagnostics": {}}
    assert "feedback: missing proposals" in contract.problems(row)


def test_learning_examples_join_on_feedback_id(full):
    rows = full.trace()
    said = {r["feedback_id"]: r for r in rows if r["event"] == "feedback" and r["kind"] == "feedback"}
    decided = [r for r in rows if r["event"] == "feedback_decision"]
    assert {d["decision"] for d in decided} == {"accepted", "rejected"}
    for d in decided:
        example = said[d["feedback_id"]]
        assert example["text"] and example["params"]["t"] == 15
    accepted = next(d for d in decided if d["decision"] == "accepted")
    assert full.iteration("phate", 2)["feedback_id"] == accepted["feedback_id"]


def test_export_is_portable_and_contains_no_data(full, tmp_path):
    dest = export.export(full, tmp_path / "exports", plots=True)
    manifest = json.loads((dest / "MANIFEST.json").read_text())
    assert manifest["events"] == len(full.trace()) and manifest["schema_version"] == 2
    text = "".join(p.read_text() for p in dest.rglob("*.json*"))
    assert str(tmp_path) not in text and "/Users/" not in text and "/home/" not in text
    assert json.loads((dest / "session.json").read_text())["project"]["data_folder"] == "x.h5ad"
    assert not list(dest.rglob("*.npy")) and not list(dest.rglob("*.h5ad"))
    assert len(list(dest.glob("iterations/*/*/iteration.json"))) == len(full.iterations())


def test_export_refuses_a_broken_trace(full, tmp_path):
    with open(full.root / "trace.jsonl", "a") as f:
        f.write(json.dumps({"event": "made_up"}) + "\n")
    with pytest.raises(sess.TrainingError, match="breaks the trace contract"):
        export.export(full, tmp_path / "exports")
    assert not list((tmp_path / "exports").glob("*")) if (tmp_path / "exports").exists() else True


def test_scrub_reduces_absolute_paths():
    assert export.scrub({"a": "/x/y/data.h5ad", "b": ["rel/p.npy", "synthetic:t"]}) == \
        {"a": "data.h5ad", "b": ["rel/p.npy", "synthetic:t"]}


# ── play data (real, cheap) ─────────────────────────────────────────────────────────────
def test_play_data_has_measured_time_truth_and_counts(tmp_path):
    adata = playdata.generate(cells=200, genes=50, timepoints=3, spread=0.05, depth=2.0, seed=1)
    assert adata.shape == (200, 50)
    assert set(adata.obs["timepoint"]) == {"day0", "day10", "day20"}
    x = adata.X.toarray()
    assert (x >= 0).all() and np.allclose(x, np.round(x))
    measured = adata.obs["timepoint"].str[3:].astype(float).to_numpy()
    assert np.corrcoef(measured, adata.obs["true_time"])[0, 1] > 0.9
    paths = playdata.make_pool(tmp_path / "play")
    assert [p.name for p in paths] == [f"{n}.h5ad" for n in playdata.VARIANTS]


def test_truth_scores_reward_recovering_progress():
    rng = np.random.default_rng(0)
    truth = rng.random(100)
    good = np.c_[truth + rng.normal(0, 0.01, 100), rng.normal(size=100)]
    assert playdata.truth_scores("phate", truth, emb=good)["truth.phate_best_axis"] > 0.95
    assert playdata.truth_scores("dpt", truth, pseudotime=rng.random(100))["truth.dpt"] < 0.3
    assert playdata.truth_scores("dpt", truth, pseudotime=np.full(100, np.nan)) == {"truth.dpt": None}
