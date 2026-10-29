"""Join the grid's run records and score files into the tables train.py reads.

    python -m geomancer.learner.table grid=outputs/ssb source=synthetic
    python -m geomancer.learner.table grid=outputs/ssb source=real

This computes nothing. drive.py writes one run record per cell into the ledgers
under ``<grid>/ledgers``; score.py reduces them into ``<grid>/scores/<source>.jsonl``
by run_id. Here each cell (sample, seed, action) takes its row (score.cells: the
latest ok and complete row, else the latest), and that row's score by run_id.

One table per seed, ``<out>/seed<s>/`` (default ``<grid>/tables/<source>/seed<s>``),
because train.py reads one reward per (sample, action). The seed changes the
embeddings only; the state runs once per sample, so every seed's table has the
same states. Each table has:

- ``rewards.jsonl``: one row per cell, with its ``run_id``, ``status`` and score.
  A cell that was refused or failed is a row with that status and reward None,
  never a gap; train.py reads None as reward 0, the reward of an action that gave
  no graph. ``config.json`` counts the cells by status and lists the superseded
  rows and the missing cells (no row at all).
- ``rewards_nested.jsonl`` (real only): for each outer fold k, the rewards of the
  datasets outside k from classifiers fit without fold k. train.py fits on these.
- ``states.jsonl``: one row per sample, every state vector, from its ssb_state
  run (``run_id``); a sample without an ok state run is an error row, which
  train.py leaves out and lists.
- ``config.json``: the actions, the reward, the seed, the counts, and the sha256 of
  every ledger and of the score file read.

A table mixes no geomancer commits: rows from two commits are refused unless
``allow_mixed_commits=true``. A chosen row without a score row means the scores
are older than the ledger, and is refused: rerun score.py.
"""

import json
import sys
from pathlib import Path

from geomancer.learner import actions as actions_module
from geomancer.learner import provenance
from geomancer.learner import state as state_module

config = {
    "grid": "outputs/ssb",  # drive.py's out
    "source": "synthetic",  # synthetic or real
    "scores": None,  # None means <grid>/scores/<source>.jsonl
    "out": None,  # None means <grid>/tables/<source>
    "allow_mixed_commits": False,
}

# Score fields that are bookkeeping, not results; the rest go into the reward row.
BOOKKEEPING = ("reward_kind", "outer_fold", "graph_sha256", "truth_sha256")


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


def _reward_row(score_row):
    row = {k: v for k, v in score_row.items() if k not in BOOKKEEPING}
    return {"sample": row.pop("sample"), "action": row.pop("action"), "seed": row.pop("seed"), **row}


def state_row(row):
    """One states.jsonl row from an ssb_state run: every vector, rebuilt from its scalars."""
    from geomancer.learner import score

    g, vec = score.geo(row), row.get("g_vector") or {}
    names = {row.get("dataset_name"), g.get("sample"), vec.get("ssb_state.sample")}
    if not score.finished(row):
        return {"sample": g.get("sample"), "run_id": row["run_id"],
                "error": f"state run {g.get('status') or 'failed'}: {g.get('detail')}"}
    if len(names) != 1:
        return {"sample": g.get("sample"), "run_id": row["run_id"],
                "error": f"the state run's sample names disagree: {sorted(map(str, names))}"}
    out = {"sample": g["sample"], "run_id": row["run_id"]}
    for name in state_module.STATES:
        dim = vec[f"ssb_state.{name}_dim"]
        out[name] = [vec[f"ssb_state.{name}_{i}"] for i in range(dim)]
    return out


def _order(names):
    known = list(actions_module.ACTIONS)
    return sorted(names, key=lambda a: (known.index(a) if a in known else len(known), a))


def build(grid, source, scores=None, out=None, allow_mixed_commits=False):
    """Write one table per seed; returns their directories, sorted by seed."""
    from geomancer.learner import score

    grid = Path(grid)
    scores = Path(scores) if scores else grid / "scores" / f"{source}.jsonl"
    out = Path(out) if out else grid / "tables" / source
    rows = [r for r in score.read_ledgers(grid) if score.geo(r).get("source") == source]
    graph_rows = [r for r in rows if score.geo(r).get("kind") == "graph"]
    chosen, superseded = score.cells(graph_rows)
    score_rows = [json.loads(line) for line in scores.read_text().splitlines()]
    top = {s["run_id"]: s for s in score_rows if s.get("outer_fold") is None}
    if missing := sorted(r["run_id"] for r in chosen.values() if r["run_id"] not in top):
        raise ValueError(f"{len(missing)} rows have no score in {scores} (first {missing[:5]}); "
                         "the scores are older than the ledger: rerun score.py")
    commits = sorted({str((score.geo(r).get("geomancer_revision") or {}).get("commit"))
                      for r in chosen.values()})
    if len(commits) > 1 and not allow_mixed_commits:
        raise ValueError(f"the rows come from {len(commits)} geomancer commits {commits}; "
                         "run the grid from one commit, or pass allow_mixed_commits=true")
    states, _ = score.cells([r for r in rows if score.geo(r).get("kind") == "state"])
    state_by_sample = {sample: row for (sample, _, _), row in states.items()}
    meta = json.loads(scores.with_suffix(".json").read_text())
    provenance_block = {
        "ledgers": score.ledger_hashes(grid),
        "scores": {"path": str(scores), "sha256": provenance.sha256_file(scores),
                   "manifest_sha256": provenance.sha256_file(scores.with_suffix(".json"))},
        "geomancer_commits": commits,
        "manyruns_revisions": sorted({json.dumps(score.geo(r).get("manyruns_revision"), sort_keys=True)
                                      for r in chosen.values()}),
    }
    names = _order({a for (_, _, a) in chosen})
    samples = sorted({s for (s, _, _) in chosen})
    written = []
    graph_by_id = {r["run_id"]: r for r in graph_rows}
    for seed in sorted({seed for (_, seed, _) in chosen}):
        folder = out / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        mine = {k: r for k, r in chosen.items() if k[1] == seed}
        reward_rows = [_reward_row(top[mine[(s, seed, a)]["run_id"]])
                       for s in samples for a in names if (s, seed, a) in mine]
        counts = {}
        for r in reward_rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        (folder / "rewards.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reward_rows))
        state_rows = [state_row(state_by_sample[s]) if s in state_by_sample
                      else {"sample": s, "run_id": None, "error": "no ok state run"} for s in samples]
        (folder / "states.jsonl").write_text("".join(json.dumps(r) + "\n" for r in state_rows))
        superseded_here = sorted(i for i in superseded if score.cell_key(graph_by_id[i])[1] == seed)
        table_cfg = {
            "source": source, "seed": seed, "actions": names, "reward": meta["reward"],
            "counts": counts, "n_cells": len(reward_rows),
            "missing": [[s, a] for s in samples for a in names if (s, seed, a) not in mine],
            "superseded": superseded_here,
            "states_excluded": [r["sample"] for r in state_rows if "error" in r],
            "provenance": provenance_block,
        }
        if source == "real":
            nested = [_reward_row(s) | {"outer_fold": s["outer_fold"]} for s in score_rows
                      if s.get("outer_fold") is not None and s.get("seed") == seed
                      and s["status"] != "superseded"]
            (folder / "rewards_nested.jsonl").write_text("".join(json.dumps(r) + "\n" for r in nested))
            table_cfg.update(classifiers=meta.get("classifiers"), outer_folds=meta["outer_folds"])
        (folder / "config.json").write_text(json.dumps(table_cfg, indent=2))
        print(f"seed {seed}: {len(samples)} samples x {len(names)} actions, cells {counts}, "
              f"{len(table_cfg['missing'])} missing, {len(superseded_here)} superseded -> {folder}",
              flush=True)
        written.append(folder)
    return written


def main(argv=None):
    cfg = override(config, sys.argv[1:] if argv is None else argv)
    if cfg["source"] not in ("synthetic", "real"):
        raise ValueError(f"unknown source {cfg['source']!r}; expected synthetic or real")
    for folder in build(cfg["grid"], cfg["source"], cfg["scores"], cfg["out"], cfg["allow_mixed_commits"]):
        (folder / "run.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {cfg['out'] or Path(cfg['grid']) / 'tables' / cfg['source']}")


if __name__ == "__main__":
    main()
