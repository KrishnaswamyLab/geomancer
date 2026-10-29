"""Build the reward table: every action on every sample, once.

    python -m geomancer.learner.table limit=20 workers=4 out=outputs/tables/smoke

Writes ``rewards.jsonl`` (one row per sample x action), ``states.jsonl`` (one
row per sample, every state vector) and ``config.json`` under ``out``. A failed
action is a row with reward 0, status "error" and the error, never a gap.
"""

import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from geomancer.learner import actions as actions_module
from geomancer.learner import rewards as rewards_module
from geomancer.learner import state as state_module

config = {
    "source": "synthetic",
    "root": None,  # scShapeBench directory; None means ${omics_data:}/scshapebench
    "limit": None,
    "actions": "all",  # or comma-separated action names
    "reward": "synthetic_graph",
    "workers": 4,
    "out": "outputs/tables/synthetic",
}


def override(config, argv):
    """nanoGPT-style ``key=value`` overrides; unknown keys are an error."""
    config = dict(config)
    for arg in argv:
        key, _, raw = arg.partition("=")
        if key not in config:
            raise KeyError(f"unknown config key {key!r}; known: {sorted(config)}")
        try:
            config[key] = json.loads(raw)
        except json.JSONDecodeError:
            config[key] = raw
    return config


def score_sample(sample, actions, reward):
    """One row per action for one sample."""
    rows = []
    for name, fn in actions.items():
        try:
            score = reward(fn(sample["points"]), sample)
            rows.append({"sample": sample["id"], "action": name, "status": "ok", **score})
        except Exception as exc:
            rows.append({"sample": sample["id"], "action": name, "reward": 0.0,
                         "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return rows


def sample_states(sample):
    return {"sample": sample["id"],
            **{name: fn(sample["points"]).tolist() for name, fn in state_module.STATES.items()}}


def _work(job):
    sample, names, reward_name = job
    chosen = {n: actions_module.ACTIONS[n] for n in names}
    try:
        return score_sample(sample, chosen, rewards_module.REWARDS[reward_name]), sample_states(sample)
    except Exception:
        traceback.print_exc()
        raise


def build_table(samples, actions, reward, out, workers=1):
    """Score every action on every sample; write rewards, states and config to ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reward_name = next(k for k, v in rewards_module.REWARDS.items() if v is reward)
    names = list(actions)
    (out / "config.json").write_text(json.dumps({"actions": names, "reward": reward_name}, indent=2))
    jobs = ((sample, names, reward_name) for sample in samples)
    started = time.time()
    with open(out / "rewards.jsonl", "w") as rewards_file, open(out / "states.jsonl", "w") as states_file:
        results = map(_work, jobs) if workers == 1 else ProcessPoolExecutor(workers).map(_work, jobs)
        for done, (rows, states) in enumerate(results, 1):
            rewards_file.writelines(json.dumps(row) + "\n" for row in rows)
            states_file.write(json.dumps(states) + "\n")
            errors = sum(row["status"] == "error" for row in rows)
            print(f"[{done}] sample={states['sample']} errors={errors} "
                  f"elapsed={time.time() - started:.0f}s", flush=True)
    return out


def load_samples(source, root=None, limit=None):
    """Samples as ``{"id", "points", "graph", "meta"}`` from the manylatents-omics loader."""
    if source != "synthetic":
        raise ValueError(f"unknown source {source!r}; only 'synthetic' is implemented")
    try:
        from manylatents.singlecell.data import scshapebench
    except ImportError as exc:
        raise SystemExit(
            "the scShapeBench loader is in manylatents-omics after 0.1.3 "
            "(latent-reasoning-works/manylatents-omics#74); install that branch") from exc
    ids = scshapebench.list_synthetic(root=root)[:limit]
    for sample_id in ids:
        sample = scshapebench.load_synthetic(sample_id, root=root)
        yield {"id": sample_id, "points": sample["points"], "graph": sample["graph"],
               "meta": sample["meta"]}


def main(argv=None):
    cfg = override(config, sys.argv[1:] if argv is None else argv)
    names = (list(actions_module.ACTIONS) if cfg["actions"] == "all"
             else cfg["actions"].split(","))
    chosen = {n: actions_module.ACTIONS[n] for n in names}
    out = build_table(load_samples(cfg["source"], cfg["root"], cfg["limit"]), chosen,
                      rewards_module.REWARDS[cfg["reward"]], cfg["out"], cfg["workers"])
    (out / "run.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
