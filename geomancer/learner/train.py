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

A cell without a reward (a refused or failed cell, reward None, or a cell with no
row) is NaN and masked, never imputed: a cell that did not run is not a graph
that scored 0. Policies fit on observed cells only. A choice whose cell has no
reward is counted (``unobserved_rate``) and left out of the reward and regret
means. A table with cells that have no row at all is refused, because the grid
is incomplete (``allow_missing`` masks them). A sample with no observed reward and
an action with none are left out and listed. summary.json counts the masked cells
by status, and names by sha256 every table file the numbers rest on.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from geomancer.learner import provenance
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
    "allow_missing": False,  # mask cells with no row instead of refusing the table
    "out": None,  # default: <table>/train/<state>
}


def load_table(path, state, reward="reward", allow_missing=False):
    """Rewards as a samples x actions matrix (NaN where unobserved), states as samples x features.

    With ``outer_folds`` in config.json, also ``folds`` (sample indices per outer
    fold) and ``R_fit`` (outer folds x samples x actions, from rewards_nested.jsonl).
    Samples whose states row is an error have no state, so they are left out and
    listed in ``excluded``; so are samples with no observed reward. ``masked``
    counts the cells without a reward by status.
    """
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    actions = list(cfg["actions"])
    missing = cfg.get("missing") or []
    if missing and not allow_missing:
        raise ValueError(f"{len(missing)} cells have no row (first {missing[:3]}); the grid is "
                         "incomplete: resubmit it, or pass allow_missing=true to mask them")
    states, excluded, reasons = {}, [], {}
    for line in (path / "states.jsonl").read_text().splitlines():
        row = json.loads(line)
        if "error" in row:
            excluded.append(row["sample"])
            reasons[row["sample"]] = row["error"]
        else:
            states[row["sample"]] = row[state]

    def read(name):
        return [json.loads(line) for line in (path / name).read_text().splitlines()]

    rows = read("rewards.jsonl")
    masked = {}
    for row in rows:
        if row.get(reward) is None:
            masked[row["status"]] = masked.get(row["status"], 0) + 1
    if missing:
        masked["missing"] = len(missing)
    observed = {(r["sample"], r["action"]) for r in rows if r.get(reward) is not None}
    for s in sorted(states, key=str):
        if not any((s, a) in observed for a in actions):
            del states[s]
            excluded.append(s)
            reasons[s] = "no observed reward"
    samples = sorted(states)
    dropped = [a for a in actions if not any((s, a) in observed for s in samples)]
    actions = [a for a in actions if a not in dropped]
    index = {s: i for i, s in enumerate(samples)}
    col = {a: j for j, a in enumerate(actions)}

    def matrix(rows):
        R = np.full((len(samples), len(actions)), np.nan)
        for row in rows:
            if row["sample"] in index and row["action"] in col and row.get(reward) is not None:
                R[index[row["sample"]], col[row["action"]]] = row[reward]
        return R

    files = {name: provenance.sha256_file(path / name)
             for name in ("rewards.jsonl", "states.jsonl", "config.json", "rewards_nested.jsonl")
             if (path / name).is_file()}
    X = np.array([states[s] for s in samples], dtype=float).reshape(len(samples), -1)
    table = {"samples": samples, "actions": actions, "X": X, "R": matrix(rows),
             "excluded": excluded, "excluded_reasons": reasons, "dropped_actions": dropped,
             "masked": masked, "folds": None, "R_fit": None, "split": "random",
             # What the numbers rest on: every reward row cites its run_id, the table's
             # files are named by sha256, and its config names the ledgers and scores.
             "provenance": {"rewards_sha256": files["rewards.jsonl"], "files": files,
                            "table": cfg.get("provenance")}}
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


def _observed_mean(R):
    # The mean reward of each action over its observed cells; -inf for an action with none.
    seen = np.isfinite(R)
    total = np.where(seen, R, 0.0).sum(0)
    return np.where(seen.any(0), total / np.maximum(seen.sum(0), 1), -np.inf)


def _best_fixed(X_train, R_train, X_test, cfg, rng):
    return np.full(len(X_test), _observed_mean(R_train).argmax()), None


def _regression_greedy(X_train, R_train, X_test, cfg, rng):
    # One ridge regression per action on that action's observed cells; choose the argmax.
    F, G = _features(X_train, X_train), _features(X_train, X_test)
    predicted = np.full((len(X_test), R_train.shape[1]), -np.inf)
    for a in range(R_train.shape[1]):
        seen = np.isfinite(R_train[:, a])
        if seen.any():
            Fa = F[seen]
            w = np.linalg.solve(Fa.T @ Fa + cfg["ridge"] * np.eye(F.shape[1]), Fa.T @ R_train[seen, a])
            predicted[:, a] = G @ w
    return predicted.argmax(1), predicted.max(1)


def _contextual_bandit(X_train, R_train, X_test, cfg, rng):
    # LinUCB (Li et al. 2010) over the training samples in a random order, seeing
    # only the chosen action's reward; then greedy on the test samples. A chosen cell
    # without a reward gives no feedback, so it updates nothing.
    F, G = _features(X_train, X_train), _features(X_train, X_test)
    n_actions, dim = R_train.shape[1], F.shape[1]
    A = np.stack([cfg["ridge"] * np.eye(dim)] * n_actions)
    b = np.zeros((n_actions, dim))
    for i in rng.permutation(len(F)):
        theta = np.linalg.solve(A, b[..., None])[..., 0]
        bonus = np.sqrt(np.einsum("d,ade,e->a", F[i], np.linalg.inv(A), F[i]))
        a = int(np.argmax(theta @ F[i] + cfg["ucb_alpha"] * bonus))
        if not np.isfinite(R_train[i, a]):
            continue
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
    """Mean reward, regret to the oracle, silent failures, refusals and unobserved choices.

    The best reward of a sample is over its observed cells. A choice whose cell has
    no reward is counted in ``unobserved_rate`` and left out of the reward and regret
    means; a refusal counts as reward 0, as before.
    """
    best = np.where(np.isfinite(R_test), R_test, -np.inf).max(1)
    refused = (np.zeros(len(choice), bool) if cfg["refuse_below"] is None or predicted is None
               else predicted < cfg["refuse_below"])
    got = R_test[np.arange(len(choice)), choice]
    acted = ~refused
    unobserved = acted & ~np.isfinite(got)
    scored = ~unobserved
    value = np.where(acted & scored, np.nan_to_num(got), 0.0)
    silent = acted & scored & (np.nan_to_num(got) <= cfg["silent_low"]) & (best >= cfg["silent_high"])

    def mean(x):
        return float(x[scored].mean()) if scored.any() else float("nan")

    metrics = {
        "reward": mean(value),
        "regret": mean(best - value),
        "silent_failure_rate": float(silent.mean()),
        "refusal_rate": float(refused.mean()),
        "unobserved_rate": float(unobserved.mean()),
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
                    known = np.where(np.isfinite(R[test]), R[test], -np.inf)
                    choice, predicted = known.argmax(1), known.max(1)
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
    table = load_table(cfg["table"], cfg["state"], cfg["reward"], cfg["allow_missing"])
    out = Path(cfg["out"] or Path(cfg["table"]) / "train" / cfg["state"])
    out.mkdir(parents=True, exist_ok=True)
    rows = run(table, cfg)
    (out / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(
        {"config": cfg, "split": table["split"], "excluded": table["excluded"],
         "excluded_reasons": {str(k): v for k, v in table["excluded_reasons"].items()},
         "dropped_actions": table["dropped_actions"], "masked": table["masked"],
         "code_revision": provenance.geomancer_revision(),
         "table": table["provenance"], "policies": summary},
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
