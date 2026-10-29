"""Replay a reward table: fit each policy on some samples, score it on the rest.

    python -m geomancer.learner.train table=outputs/tables/synthetic_v1 state=input_geometry

Pluggable only along the ablation axes: `state` (a column of states.jsonl),
`POLICIES`, and `reward` (a column of rewards.jsonl). Everything else is a
config value. Writes ``metrics.jsonl`` (one row per policy x seed x fold) and
``summary.json`` (means over seeds and folds) under ``out``.

A decision here is one step: choosing an action never changes the next state.
So DAgger reduces to behaviour cloning, and the full-information learner is
`regression_greedy`. `contextual_bandit` sees only the reward of the action it
chose, one training sample at a time.

A table whose config.json has ``outer_folds`` (the real table) is split on those
folds, and policies are fit on ``rewards_nested.jsonl``: rewards recomputed
without the test fold, so no test label reaches a policy through the rewards it
fits on. Other tables are split at random into ``folds``.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from geomancer.learner.table import override

config = {
    "table": "outputs/tables/synthetic_v1",
    "state": "input_geometry",
    "reward": "reward",
    "policies": "random,best_fixed,regression_greedy,contextual_bandit,oracle",
    "folds": 5,  # random folds; ignored when the table has outer_folds
    "seeds": 3,
    "ridge": 1.0,
    "ucb_alpha": 0.5,
    "refuse_below": None,  # refuse when the predicted best reward is below this
    "silent_low": 0.2,  # a choice this bad ...
    "silent_high": 0.5,  # ... when another action reached this is a silent failure
    "out": None,  # default: <table>/train/<state>
}


def load_table(path, state, reward="reward"):
    """Rewards as a samples x actions matrix, states as samples x features.

    With ``outer_folds`` in config.json, also ``folds`` (sample indices per outer
    fold) and ``R_fit`` (outer folds x samples x actions, from rewards_nested.jsonl).
    Samples whose states row is an error have no state, so they are left out and
    listed in ``excluded``.
    """
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    actions = cfg["actions"]
    states, excluded = {}, []
    for line in (path / "states.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "error" in row:
            excluded.append(row["sample"])
        else:
            states[row["sample"]] = row[state]
    samples = sorted(states)
    index = {s: i for i, s in enumerate(samples)}
    col = {a: j for j, a in enumerate(actions)}

    def matrix(rows):
        R = np.zeros((len(samples), len(actions)))
        for row in rows:
            if row["sample"] not in excluded:
                R[index[row["sample"]], col[row["action"]]] = row.get(reward, 0.0) or 0.0
        return R

    def read(name):
        return [json.loads(line) for line in (path / name).read_text().splitlines()]

    X = np.array([states[s] for s in samples], dtype=float).reshape(len(samples), -1)
    table = {"samples": samples, "actions": actions, "X": X, "R": matrix(read("rewards.jsonl")),
             "excluded": excluded, "folds": None, "R_fit": None, "split": "random"}
    if "outer_folds" in cfg:
        nested = read("rewards_nested.jsonl")
        table["folds"] = [np.array([index[s] for s in f if s in index], dtype=int)
                          for f in cfg["outer_folds"]]
        table["R_fit"] = np.stack([matrix([r for r in nested if r["outer_fold"] == k])
                                   for k in range(len(cfg["outer_folds"]))])
        table["split"] = "outer_folds, fit on rewards_nested"
    elif cfg.get("reward") == "real_label":
        table["split"] = ("random; WARNING: no rewards_nested, so the training rewards were made "
                          "by classifiers fit on the test samples' labels")
    return table


def _features(X_train, X):
    mean, std = X_train.mean(0), X_train.std(0)
    std[std == 0] = 1.0
    return np.c_[np.ones(len(X)), (X - mean) / std]


def _random(X_train, R_train, X_test, cfg, rng):
    return rng.integers(R_train.shape[1], size=len(X_test)), None


def _best_fixed(X_train, R_train, X_test, cfg, rng):
    return np.full(len(X_test), R_train.mean(0).argmax()), None


def _regression_greedy(X_train, R_train, X_test, cfg, rng):
    # One ridge regression per action on the full reward row; choose the argmax.
    F, G = _features(X_train, X_train), _features(X_train, X_test)
    W = np.linalg.solve(F.T @ F + cfg["ridge"] * np.eye(F.shape[1]), F.T @ R_train)
    predicted = G @ W
    return predicted.argmax(1), predicted.max(1)


def _contextual_bandit(X_train, R_train, X_test, cfg, rng):
    # LinUCB (Li et al. 2010) over the training samples in a random order, seeing
    # only the chosen action's reward; then greedy on the test samples.
    F, G = _features(X_train, X_train), _features(X_train, X_test)
    n_actions, dim = R_train.shape[1], F.shape[1]
    A = np.stack([cfg["ridge"] * np.eye(dim)] * n_actions)
    b = np.zeros((n_actions, dim))
    for i in rng.permutation(len(F)):
        theta = np.linalg.solve(A, b[..., None])[..., 0]
        bonus = np.sqrt(np.einsum("d,ade,e->a", F[i], np.linalg.inv(A), F[i]))
        a = int(np.argmax(theta @ F[i] + cfg["ucb_alpha"] * bonus))
        A[a] += np.outer(F[i], F[i])
        b[a] += R_train[i, a] * F[i]
    theta = np.linalg.solve(A, b[..., None])[..., 0]
    predicted = G @ theta.T
    return predicted.argmax(1), predicted.max(1)


POLICIES = {
    "random": _random,
    "best_fixed": _best_fixed,
    "regression_greedy": _regression_greedy,
    "contextual_bandit": _contextual_bandit,
    "oracle": None,  # chooses the best action on each test sample: the upper bound
}


def score(choice, predicted, R_test, cfg):
    """Mean reward, regret to the oracle, silent failures and refusals."""
    best = R_test.max(1)
    refused = (np.zeros(len(choice), bool) if cfg["refuse_below"] is None or predicted is None
               else predicted < cfg["refuse_below"])
    got = R_test[np.arange(len(choice)), choice]
    acted = ~refused
    silent = acted & (got <= cfg["silent_low"]) & (best >= cfg["silent_high"])
    metrics = {
        "reward": float(np.where(acted, got, 0.0).mean()),
        "regret": float((best - np.where(acted, got, 0.0)).mean()),
        "silent_failure_rate": float(silent.mean()),
        "refusal_rate": float(refused.mean()),
    }
    if cfg["refuse_below"] is not None and refused.any():
        # A refusal is justified when no action reached the threshold.
        metrics["refusal_precision"] = float((best[refused] < cfg["refuse_below"]).mean())
    return metrics


def run(table, cfg):
    """Every policy on every seed and fold; one metrics row each."""
    X, R = table["X"], table["R"]
    outer = table.get("folds")
    names = cfg["policies"].split(",")
    rows = []
    for seed in range(cfg["seeds"]):
        rng = np.random.default_rng(seed)
        folds = outer if outer is not None else np.array_split(rng.permutation(len(R)), cfg["folds"])
        for k, test in enumerate(folds):
            if outer is not None:
                train = np.concatenate([f for j, f in enumerate(outer) if j != k])
                R_fit = table["R_fit"][k]
            else:
                train, R_fit = np.setdiff1d(np.arange(len(R)), test), R
            if not len(test) or not len(train):
                continue
            for name in names:
                if name == "oracle":
                    choice, predicted = R[test].argmax(1), R[test].max(1)
                else:
                    # NumPy on macOS Accelerate warns spuriously inside matmul; the
                    # finiteness check below is what guards the result.
                    with np.errstate(all="ignore"):
                        choice, predicted = POLICIES[name](X[train], R_fit[train], X[test], cfg,
                                                           np.random.default_rng([seed, k]))
                    if predicted is not None and not np.isfinite(predicted).all():
                        raise FloatingPointError(f"{name}: non-finite predictions, seed {seed} fold {k}")
                rows.append({"policy": name, "seed": seed, "fold": k,
                             **score(choice, predicted, R[test], cfg)})
    return rows


def summarize(rows, baseline="best_fixed"):
    """Mean and std over seeds x folds, plus the paired reward gain over `baseline`.

    Every policy sees the same test samples in a given seed and fold, so the
    gain is taken per (seed, fold) before averaging.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(row)
    # Metric names from every row: refusal_precision exists only where a policy refused.
    keys = list(dict.fromkeys(k for r in rows for k in r if k not in ("policy", "seed", "fold")))
    base = {(r["seed"], r["fold"]): r["reward"] for r in grouped.get(baseline, [])}
    summary = {}
    for policy, rs in grouped.items():
        summary[policy] = {k: {"mean": float(np.mean([r[k] for r in rs if k in r])),
                               "std": float(np.std([r[k] for r in rs if k in r]))}
                           for k in keys if any(k in r for r in rs)} | {"n": len(rs)}
        if base:
            gain = [r["reward"] - base[(r["seed"], r["fold"])] for r in rs]
            summary[policy][f"gain_over_{baseline}"] = {
                "mean": float(np.mean(gain)), "std": float(np.std(gain)),
                "wins": int(sum(g > 0 for g in gain)), "of": len(gain)}
    return summary


def main(argv=None):
    cfg = override(config, sys.argv[1:] if argv is None else argv)
    table = load_table(cfg["table"], cfg["state"], cfg["reward"])
    out = Path(cfg["out"] or Path(cfg["table"]) / "train" / cfg["state"])
    out.mkdir(parents=True, exist_ok=True)
    rows = run(table, cfg)
    (out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(
        {"config": cfg, "split": table["split"], "excluded": table["excluded"], "policies": summary},
        indent=2))
    print(f"split: {table['split']}" + (f"; excluded {table['excluded']}" if table["excluded"] else ""))
    for policy, m in summary.items():
        print(f"{policy:18s} reward={m['reward']['mean']:.3f}±{m['reward']['std']:.3f} "
              f"regret={m['regret']['mean']:.3f} silent={m['silent_failure_rate']['mean']:.3f} "
              f"refused={m['refusal_rate']['mean']:.3f}"
              + (f" gain={m['gain_over_best_fixed']['mean']:+.3f}±{m['gain_over_best_fixed']['std']:.3f}"
                 f" ({m['gain_over_best_fixed']['wins']}/{m['gain_over_best_fixed']['of']})"
                 if "gain_over_best_fixed" in m else ""))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
