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
    summary = _summary(tmp_path, "refuse_below=0.95", "policies=random,regression_greedy")
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


def test_a_table_with_outer_folds_fits_on_its_nested_rewards(tmp_path):
    # In rewards.jsonl action a is best; in rewards_nested.jsonl (what a policy may
    # fit on) action b is. Policies must split on outer_folds and fit on the nested ones.
    samples = [f"s{i}" for i in range(6)]
    outer = [samples[:3], samples[3:]]
    (tmp_path / "config.json").write_text(json.dumps(
        {"actions": ["a", "b"], "reward": "real_label", "outer_folds": outer}))
    (tmp_path / "states.jsonl").write_text("".join(
        json.dumps({"sample": s, "none": []}) + "\n" for s in samples)
        + json.dumps({"sample": "broken", "error": "OSError: unreadable"}) + "\n")
    (tmp_path / "rewards.jsonl").write_text("".join(
        json.dumps({"sample": s, "action": a, "reward": r}) + "\n"
        for s in samples + ["broken"] for a, r in (("a", 0.9), ("b", 0.5))))
    (tmp_path / "rewards_nested.jsonl").write_text("".join(
        json.dumps({"sample": s, "action": a, "outer_fold": k, "reward": r}) + "\n"
        for k, f in enumerate(outer) for s in samples if s not in f
        for a, r in (("a", 0.1), ("b", 0.6))))
    out = tmp_path / "out"

    train.main([f"table={tmp_path}", "state=none", f"out={out}", "seeds=1",
                "policies=best_fixed,oracle"])

    result = json.loads((out / "summary.json").read_text())
    assert result["excluded"] == ["broken"] and result["split"].startswith("outer_folds")
    assert result["policies"]["best_fixed"]["reward"]["mean"] == pytest.approx(0.5)
    assert result["policies"]["oracle"]["reward"]["mean"] == pytest.approx(0.9)
    assert result["policies"]["best_fixed"]["n"] == 2


# ── cells without a reward are masked, never imputed (hard rule 7) ────────────

def _masked_table(tmp_path, missing=()):
    # Action "a" scores 0.0 on sample 0 (a graph that ran and scored 0); "b" failed on
    # sample 0 (reward None); "dead" failed on every sample; sample 2 has no cell that ran.
    rows = [{"sample": 0, "action": "a", "reward": 0.0, "status": "ok"},
            {"sample": 0, "action": "b", "reward": None, "status": "not_run"},
            {"sample": 1, "action": "a", "reward": 0.2, "status": "ok"},
            {"sample": 1, "action": "b", "reward": 0.8, "status": "ok"},
            {"sample": 2, "action": "a", "reward": None, "status": "error"},
            {"sample": 2, "action": "b", "reward": None, "status": "error"}]
    rows += [{"sample": i, "action": "dead", "reward": None, "status": "not_run"} for i in range(3)]
    (tmp_path / "config.json").write_text(json.dumps(
        {"actions": ["a", "b", "dead"], "missing": [list(m) for m in missing]}))
    (tmp_path / "rewards.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "states.jsonl").write_text("".join(
        json.dumps({"sample": i, "input_geometry": [float(i)]}) + "\n" for i in range(3)))
    return tmp_path


def test_a_failed_cell_is_masked_not_scored_as_zero(tmp_path):
    table = train.load_table(_masked_table(tmp_path), "input_geometry")

    # Sample 2 has no observed reward and "dead" never ran: both leave, and are listed.
    assert table["samples"] == [0, 1] and table["excluded_reasons"] == {2: "no observed reward"}
    assert table["actions"] == ["a", "b"] and table["dropped_actions"] == ["dead"]
    assert table["R"][0, 0] == 0.0 and np.isnan(table["R"][0, 1])
    assert table["masked"] == {"not_run": 4, "error": 2}


def test_a_cell_with_no_row_refuses_the_table_unless_allowed(tmp_path):
    path = _masked_table(tmp_path, missing=[(1, "dead")])

    with pytest.raises(ValueError, match="no row"):
        train.load_table(path, "input_geometry")
    assert train.load_table(path, "input_geometry", allow_missing=True)["masked"]["missing"] == 1


def test_a_choice_whose_cell_has_no_reward_scores_zero_and_is_counted():
    """Decided 2026-10-29: picking a route that gave no graph costs what no graph is worth,
    0, at evaluation. Fitting still uses observed cells only."""
    cfg = dict(train.config)
    R = np.array([[np.nan, 0.5], [0.4, 0.8]])

    metrics = train.score(np.array([0, 1]), None, R, cfg)

    assert metrics["unobserved_rate"] == 0.5
    assert metrics["reward"] == pytest.approx(0.4)  # (0 + 0.8) / 2
    assert metrics["regret"] == pytest.approx(0.25)  # (0.5 - 0 + 0.8 - 0.8) / 2
    assert metrics["silent_failure_rate"] == 0.0  # a route with no graph failed visibly


def test_policies_fit_on_observed_cells_only(tmp_path):
    table = train.load_table(_masked_table(tmp_path), "input_geometry")
    cfg = dict(train.config)
    X, R = table["X"], table["R"]
    rng = np.random.default_rng(0)

    choice, _ = train._best_fixed(X, R, X, cfg, rng)
    # "b" has mean 0.8 over its one observed cell; a zero-imputed "b" would lose to "a" (0.1).
    assert list(choice) == [1, 1]
    for name in ("regression_greedy", "contextual_bandit"):
        choice, predicted = train.POLICIES[name](X, R, X, cfg, rng)
        assert np.isfinite(predicted).all(), name


def test_the_summary_names_every_file_its_numbers_rest_on(tmp_path):
    from geomancer.learner import provenance

    path = _table(tmp_path)
    out = tmp_path / "out"
    train.main([f"table={path}", f"out={out}", "seeds=1", "policies=best_fixed"])

    summary = json.loads((out / "summary.json").read_text())
    files = summary["table"]["files"]
    for name in ("rewards.jsonl", "states.jsonl", "config.json"):
        assert files[name] == provenance.sha256_file(path / name)
    assert summary["code_revision"]["commit"] == provenance.geomancer_revision()["commit"]
    assert summary["masked"] == {}
