"""The real-compute gate: full walks through manylatents, scanpy and MIOFlow (no mocks).

Opt-in (`pytest -m real`, about two minutes on a laptop CPU); CI's pinned lane runs it.
Feedback uses the rule tier only — the model tier is never called here.
"""
import json

import pytest

pytestmark = pytest.mark.real


@pytest.fixture
def train(isolated_wrapper, capsys):
    def run(*argv):
        code = isolated_wrapper.main(["train", *argv])
        out = capsys.readouterr()
        assert code == 0, out.err[-2000:]
        return out.out
    return run


def test_pseudotime_walk_feedback_and_lazy_recompute(train):
    from geomancer.training.session import TrainingSession

    train("start", "synthetic_timecourse", "-s", "gate", "--time", "pseudotime", "--expert", "ci")
    for _ in range(8):
        train("continue", "-s", "gate", "--select", "--no-plots")
    s = TrainingSession.open("gate")
    assert {r["state"] for r in s.chain()} == {"valid"}
    assert s.iteration("normalize", 1)["status"] == "not_applicable"  # synthetic is not counts
    assert s.iteration("mioflow", 1)["outputs"]["g"]["mioflow.n_timepoints"] == 5
    phate = s.iteration("phate", 1)["outputs"]["diagnostics"]
    assert phate["phate.knn_components"] >= 1 and "phate.time_silhouette" in phate

    out = train("feedback", "-s", "gate", "phate", "1", "too blurry", "--accept", "1",
                "--no-plots")
    assert "t 15 → 10" in out
    out = train("select", "-s", "gate", "phate", "2")
    assert "now stale, not recomputed: dpt, discretize_time, mioflow" in out
    s = TrainingSession.open("gate")
    assert [len(s.iterations(x)) for x in ("dpt", "discretize_time", "mioflow")] == [1, 1, 1]
    train("continue", "-s", "gate", "--select", "--no-plots")
    s = TrainingSession.open("gate")
    assert len(s.iterations("dpt")) == 2 and len(s.iterations("mioflow")) == 1
    assert s.iteration("dpt", 2)["inputs"]["upstream"]["phate"] == "phate#2"


def test_real_time_walk_on_counts_with_truth_and_export(train, tmp_path):
    from geomancer.training import playdata
    from geomancer.training.session import TrainingSession

    path = tmp_path / "play.h5ad"
    # play_noisy as make_pool(seed=0) writes it: a connected trajectory DPT can order.
    playdata.generate(**playdata.VARIANTS["play_noisy"], seed=1).write_h5ad(path)
    out = train("start", str(path), "-s", "counts", "--expert", "ci")
    assert "time mode: real" in out
    for _ in range(6):
        train("continue", "-s", "counts", "--select", "--no-plots")
    train("continue", "-s", "counts", "k_per_timepoint=2", "--select", "--no-plots")
    train("continue", "-s", "counts", "--select", "--no-plots")
    s = TrainingSession.open("counts")
    assert {r["state"] for r in s.chain()} == {"valid"}
    assert s.iteration("normalize", 1)["status"] == "ok"
    assert s.iteration("pca", 1)["inputs"]["replayed"] == [
        "filter_genes#1", "normalize#1", "transform#1"]
    assert s.iteration("mioflow", 1)["outputs"]["g"]["mioflow.n_timepoints"] == 8  # 4 × k=2
    assert s.iteration("dpt", 1)["outputs"]["diagnostics"]["dpt.unreachable_frac"] == 0
    truth = s.iteration("dpt", 1)["outputs"]["truth"]
    assert truth["truth.dpt"] > 0.8  # measured 0.91: DPT recovers the generative progress
    assert "truth" not in train("show", "-s", "counts", "dpt", "1", "--no-plots")
    assert "truth.dpt" in train("show", "-s", "counts", "dpt", "1", "--no-plots", "--truth")

    # Revising preprocessing replays from the source and recomputes only on request.
    train("run", "-s", "counts", "normalize", "target_sum=5000", "--select", "--no-plots")
    train("continue", "-s", "counts", "--select", "--no-plots")
    s = TrainingSession.open("counts")
    assert s.iteration("transform", 2)["inputs"]["replayed"] == ["filter_genes#1", "normalize#2"]

    out = train("export", "-s", "counts", "--out", str(tmp_path / "exports"))
    dest = next((tmp_path / "exports").iterdir())
    assert json.loads((dest / "MANIFEST.json").read_text())["iterations"] == len(s.iterations())
