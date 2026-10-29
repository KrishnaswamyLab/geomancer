import json

import numpy as np
import pytest

from geomancer.learner import train


def _table(tmp_path, n=200):
    # One feature decides the best action: x < 0 -> action a, x >= 0 -> action b.
    # Action c is mediocre everywhere, so it is the best single fixed action.
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    rewards = {"a": np.where(x < 0, 0.9, 0.0), "b": np.where(x >= 0, 0.9, 0.0),
               "c": np.full(n, 0.6)}
    (tmp_path / "config.json").write_text(json.dumps({"actions": ["a", "b", "c"]}))
    (tmp_path / "rewards.jsonl").write_text("".join(
        json.dumps({"sample": i, "action": a, "reward": float(r[i]), "status": "ok"}) + "\n"
        for i in range(n) for a, r in rewards.items()))
    (tmp_path / "states.jsonl").write_text("".join(
        json.dumps({"sample": i, "none": [], "input_geometry": [float(x[i])]}) + "\n"
        for i in range(n)))
    return tmp_path


def _summary(tmp_path, *overrides):
    out = tmp_path / "out"
    train.main([f"table={_table(tmp_path)}", f"out={out}", "seeds=1", *overrides])
    return json.loads((out / "summary.json").read_text())["policies"]


def test_learners_find_the_state_dependent_action(tmp_path):
    summary = _summary(tmp_path)
    reward = {p: m["reward"]["mean"] for p, m in summary.items()}

    assert reward["oracle"] == pytest.approx(0.9)
    assert reward["best_fixed"] == pytest.approx(0.6)
    assert reward["regression_greedy"] > 0.8
    assert reward["contextual_bandit"] > 0.6
    assert reward["random"] < reward["best_fixed"]


def test_without_state_a_learner_is_no_better_than_best_fixed(tmp_path):
    summary = _summary(tmp_path, "state=none")

    assert summary["regression_greedy"]["reward"]["mean"] == pytest.approx(0.6)


def test_refusal_counts_and_precision(tmp_path):
    summary = _summary(tmp_path, "refuse_below=0.95", "policies=regression_greedy")
    m = summary["regression_greedy"]

    # Nothing reaches 0.95, so every refusal is justified.
    assert m["refusal_rate"]["mean"] > 0.5
    assert m["refusal_precision"]["mean"] == pytest.approx(1.0)


def test_silent_failure_is_a_bad_choice_when_a_good_one_existed():
    cfg = dict(train.config)
    R = np.array([[0.1, 0.9], [0.6, 0.7]])

    metrics = train.score(np.array([0, 0]), None, R, cfg)

    assert metrics["silent_failure_rate"] == pytest.approx(0.5)
    assert metrics["regret"] == pytest.approx((0.8 + 0.1) / 2)


def test_gain_over_best_fixed_is_paired_by_seed_and_fold(tmp_path):
    summary = _summary(tmp_path)

    gain = summary["oracle"]["gain_over_best_fixed"]
    assert gain["mean"] == pytest.approx(0.3)
    assert gain["wins"] == gain["of"] == 5
    assert summary["best_fixed"]["gain_over_best_fixed"]["mean"] == 0.0
