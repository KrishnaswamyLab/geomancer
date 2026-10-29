"""Join the grid's run records and score files into the tables train.py reads.

    python -m geomancer.learner.table grid=outputs/ssb source=synthetic
    python -m geomancer.learner.table grid=outputs/ssb source=real

This computes nothing. drive.py writes one run record per cell into the ledgers
under ``<grid>/ledgers``; score.py reduces them into ``<grid>/scores/<source>.jsonl``
by run_id. Here each cell (sample, seed, action) takes its row (score.cells: the
latest ok and complete row, else the latest), and that row's score by run_id.

The expected cells are every sample of convert.py's manifest (the one the score
file names by sha256) times every action asked for, so a sample that failed
conversion, is absent from disk, or has no row at all is counted, never dropped
from the denominator.

One table per seed, ``<out>/seed<s>/`` (default ``<grid>/tables/<source>/seed<s>``),
because train.py reads one reward per (sample, action). The seed changes the
embeddings only; the state runs once per sample, so every seed's table has the
same states. Each table has:

- ``rewards.jsonl``: one row per cell that has a run record, with its ``run_id``,
  ``status`` and score. A cell that was refused or failed is a row with that status
  and reward None, never a gap; train.py masks it, never imputes it.
- ``rewards_nested.jsonl`` (real only): for each outer fold k, the rewards of the
  datasets outside k from classifiers fit without fold k. train.py fits on these.
- ``states.jsonl``: one row per expected sample, every state vector, from its
  ssb_state run (``run_id``); a sample without an ok, checked state run (or not
  converted) is an error row, which train.py leaves out and lists.
- ``config.json``: the actions, the reward, the seed, the counts by status, the
  cells with no row (``missing``), the samples not converted, the superseded rows,
  the attempts per cell, and the provenance: the sha256 of every ledger, of the
  score file and of the inputs manifest, and the (commit, dirty) of the code.

Refusals, each with its switch: rows from two geomancer commits or two engine
revisions (``allow_mixed_commits``); rows, scores or inputs made from a dirty tree
(``allow_dirty``); a state run whose tool env was an override
(``allow_unreproducible``). A chosen row without a score row, or a score file
whose ledger hashes are not the ledgers' now, means the scores are older than
the ledger, and is refused: rerun score.py.
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
    "actions": "all",  # the expected actions, or comma-separated names
    "samples": None,  # the expected samples: None means every sample of the inputs manifest
    "allow_mixed_commits": False,
    "allow_dirty": False,
    "allow_unreproducible": False,
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


def state_row(row, points_sha256=None, allow_unreproducible=False):
    """One states.jsonl row from an ssb_state run: every vector, rebuilt from its scalars.

    The run must be ok and complete, its three sample names must agree, its points
    must be the manifest's, and its provenance must hold (score.provenance_problem);
    otherwise the row is an error with the reason.
    """
    from geomancer.learner import score

    g, vec = score.geo(row), row.get("g_vector") or {}
    names = {row.get("dataset_name"), g.get("sample"), vec.get("ssb_state.sample")}

    def error(reason):
        return {"sample": g.get("sample"), "run_id": row["run_id"], "error": reason}

    if not score.finished(row):
        return error(f"state run {g.get('status') or 'failed'}: {g.get('detail')}")
    if len(names) != 1:
        return error(f"the state run's sample names disagree: {sorted(map(str, names))}")
    if points_sha256 is not None and {g.get("points_sha256"), vec.get("ssb_state.points_sha256")} != {points_sha256}:
        return error(f"the state run's points are not the manifest's {points_sha256}")
    if problem := score.provenance_problem(row, "ssb_state", allow_unreproducible):
        return error(f"{problem[0]}: {problem[1]}")
    out = {"sample": g["sample"], "run_id": row["run_id"]}
    for name in state_module.STATES:
        dim = vec[f"ssb_state.{name}_dim"]
        out[name] = [vec[f"ssb_state.{name}_{i}"] for i in range(dim)]
    return out


def _order(names):
    known = list(actions_module.ACTIONS)
    return sorted(names, key=lambda a: (known.index(a) if a in known else len(known), a))


def _names(value, known):
    if value in (None, "all"):
        return list(known)
    names = [v.strip() for v in (value.split(",") if isinstance(value, str) else value) if v.strip()]
    if unknown := sorted(set(names) - set(known)):
        raise KeyError(f"unknown {unknown}; known: {sorted(known)}")
    return names


def _revision(row):
    rev = ((row.get("extra") or {}).get("geomancer") or {}).get("geomancer_revision") or {}
    return rev.get("commit"), rev.get("dirty")


def _refuse_dirty(what, revision, allow_dirty):
    if not allow_dirty and (revision or {}).get("dirty") is not False:
        raise ValueError(f"{what} came from commit {(revision or {}).get('commit')} with "
                         f"dirty={(revision or {}).get('dirty')}; pass allow_dirty=true to accept it")


def inputs_manifest(meta):
    """convert.py's manifest that the score file names, checked against its sha256."""
    if not meta.get("inputs") or not meta.get("inputs_manifest_sha256"):
        raise ValueError("the score file names no inputs manifest; rerun score.py with inputs=")
    path = Path(meta["inputs"]) / "manifest.json"
    if provenance.sha256_file(path) != meta["inputs_manifest_sha256"]:
        raise ValueError(f"{path} is not the inputs manifest the scores were made from")
    return json.loads(path.read_text()), provenance.sha256_file(path)


def build(grid, source, scores=None, out=None, allow_mixed_commits=False, actions="all",
          samples=None, allow_dirty=False, allow_unreproducible=False):
    """Write one table per seed; returns their directories, sorted by seed."""
    from geomancer.learner import score

    grid = Path(grid)
    scores = Path(scores) if scores else grid / "scores" / f"{source}.jsonl"
    out = Path(out) if out else grid / "tables" / source
    meta = json.loads(scores.with_suffix(".json").read_text())
    if meta.get("ledgers") != score.ledger_hashes(grid):
        raise ValueError(f"the ledgers under {grid} are not the ones {scores} was made from; "
                         "the scores are older than the ledger: rerun score.py")
    manifest, manifest_sha = inputs_manifest(meta)
    _refuse_dirty("the scores", meta.get("code_revision"), allow_dirty)
    _refuse_dirty("the inputs manifest", manifest.get("code_revision"), allow_dirty)
    entries = manifest["samples"]
    universe = sorted(entries) if samples is None else _names(samples, sorted(entries))
    names = _order(_names(actions, actions_module.ACTIONS))
    not_converted = {s: {"status": entries[s]["status"], "error": entries[s].get("error")}
                     for s in universe if entries[s].get("status") != "ok"}
    converted = [s for s in universe if s not in not_converted]

    rows = [r for r in score.read_ledgers(grid) if score.geo(r).get("source") == source]
    graph_rows = [r for r in rows if score.geo(r).get("kind") == "graph"
                  and score.geo(r).get("sample") in converted and score.geo(r).get("action") in names]
    chosen, superseded = score.cells(graph_rows)
    attempts = {}
    for r in graph_rows:
        attempts[score.cell_key(r)] = attempts.get(score.cell_key(r), 0) + 1
    score_rows = [json.loads(line) for line in scores.read_text().splitlines()]
    top = {s["run_id"]: s for s in score_rows if s.get("outer_fold") is None}
    if missing := sorted(r["run_id"] for r in chosen.values() if r["run_id"] not in top):
        raise ValueError(f"{len(missing)} rows have no score in {scores} (first {missing[:5]}); "
                         "the scores are older than the ledger: rerun score.py")
    states, _ = score.cells([r for r in rows if score.geo(r).get("kind") == "state"
                             and score.geo(r).get("sample") in converted])
    state_by_sample = {sample: row for (sample, _, _), row in states.items()}
    cited = list(chosen.values()) + list(state_by_sample.values())
    revisions = sorted({json.dumps(_revision(r)) for r in cited})
    commits = sorted({str(_revision(r)[0]) for r in cited})
    if len(commits) > 1 and not allow_mixed_commits:
        raise ValueError(f"the rows come from {len(commits)} geomancer commits {commits}; "
                         "run the grid from one commit, or pass allow_mixed_commits=true")
    for r in cited:
        _refuse_dirty(f"row {r['run_id']}", score.geo(r).get("geomancer_revision"), allow_dirty)
    engines = sorted({json.dumps(score.geo(r).get("engine_revisions"), sort_keys=True) for r in cited})
    if len(engines) > 1 and not allow_mixed_commits:
        raise ValueError(f"the rows come from {len(engines)} engine and loader revisions; "
                         "run the grid from one install, or pass allow_mixed_commits=true")
    tools = {}
    for r in cited:
        for k, v in (r.get("g_vector") or {}).items():
            if k.endswith((".tool_lock", ".tool_reproducible")):
                tools.setdefault(k, set()).add(json.dumps(v))
    provenance_block = {
        "ledgers": meta["ledgers"],
        "scores": {"path": str(scores), "sha256": provenance.sha256_file(scores),
                   "manifest_sha256": provenance.sha256_file(scores.with_suffix(".json")),
                   "code_revision": meta.get("code_revision")},
        "inputs": {"path": meta["inputs"], "manifest_sha256": manifest_sha,
                   "code_revision": manifest.get("code_revision"),
                   "n_requested": manifest.get("n_requested"), "counts": manifest.get("counts")},
        "geomancer_commits": commits,
        "geomancer_revisions": [json.loads(r) for r in revisions],
        "manyruns_revisions": sorted({json.dumps(score.geo(r).get("manyruns_revision"), sort_keys=True)
                                      for r in cited}),
        "engine_revisions": [json.loads(e) for e in engines],
        "tools": {k: sorted(json.loads(v) for v in vs) for k, vs in sorted(tools.items())},
    }
    written = []
    graph_by_id = {r["run_id"]: r for r in graph_rows}
    seeds = sorted({seed for (_, seed, _) in chosen}) or [None]
    for seed in seeds:
        folder = out / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        mine = {k: r for k, r in chosen.items() if k[1] == seed}
        # A SEEDLESS ROUTE RUNS ONCE (drive.plan): this seed's table reuses that run, marked
        # `seed_applies: False`, unless an older grid ran it for this seed too.
        reused = set()
        for (s, sd, a), r in sorted(chosen.items(), key=lambda kv: str(kv[0][1])):
            if actions_module.seedless(a) and (s, seed, a) not in mine:
                mine[(s, seed, a)] = r
                reused.add((s, a))
        reward_rows = []
        for s in converted:
            for a in names:
                if (s, seed, a) in mine:
                    row = _reward_row(top[mine[(s, seed, a)]["run_id"]])
                    if (s, a) in reused:
                        row.update(seed=seed, seed_applies=False)
                    reward_rows.append(row)
        counts = {}
        for r in reward_rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        (folder / "rewards.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reward_rows))
        state_rows = []
        for s in universe:
            if s in not_converted:
                state_rows.append({"sample": s, "run_id": None,
                                   "error": f"not converted: {not_converted[s]['status']}"})
            elif s in state_by_sample:
                state_rows.append(state_row(state_by_sample[s], entries[s].get("npy_sha256"),
                                            allow_unreproducible))
            else:
                state_rows.append({"sample": s, "run_id": None, "error": "no ok state run"})
        (folder / "states.jsonl").write_text("".join(json.dumps(r) + "\n" for r in state_rows))
        superseded_here = sorted(i for i in superseded if score.cell_key(graph_by_id[i])[1] == seed)
        hist = {}
        for (s, sd, a), n in attempts.items():
            if sd == seed:
                hist[str(n)] = hist.get(str(n), 0) + 1
        missing_cells = [[s, a] for s in converted for a in names if (s, seed, a) not in mine]
        table_cfg = {
            "source": source, "seed": seed, "actions": names, "reward": meta["reward"],
            "samples": universe, "n_expected": len(universe) * len(names),
            "counts": counts, "n_cells": len(reward_rows),
            "missing": missing_cells,
            "not_converted": not_converted,
            "superseded": superseded_here,
            "attempts": dict(sorted(hist.items())),
            "states_excluded": [r["sample"] for r in state_rows if "error" in r],
            "provenance": provenance_block,
        }
        assert table_cfg["n_cells"] + len(missing_cells) + len(not_converted) * len(names) == table_cfg["n_expected"]
        if source == "real":
            nested = [_reward_row(s) | {"outer_fold": s["outer_fold"]} for s in score_rows
                      if s.get("outer_fold") is not None and s.get("seed") == seed
                      and s["status"] != "superseded" and s.get("sample") in converted
                      and s.get("action") in names]
            # The reused seedless runs' nested rewards come with them, relabelled to this seed.
            nested += [_reward_row(s) | {"outer_fold": s["outer_fold"], "seed": seed,
                                         "seed_applies": False}
                       for s in score_rows
                       if s.get("outer_fold") is not None and s["status"] != "superseded"
                       and (s.get("sample"), s.get("action")) in reused
                       and s.get("run_id") == mine[(s.get("sample"), seed, s.get("action"))]["run_id"]]
            (folder / "rewards_nested.jsonl").write_text("".join(json.dumps(r) + "\n" for r in nested))
            table_cfg.update(classifiers=meta.get("classifiers"), outer_folds=meta["outer_folds"])
        (folder / "config.json").write_text(json.dumps(table_cfg, indent=2))
        print(f"seed {seed}: {len(universe)} samples x {len(names)} actions, cells {counts}, "
              f"{len(missing_cells)} missing, {len(not_converted)} samples not converted, "
              f"{len(superseded_here)} superseded -> {folder}", flush=True)
        written.append(folder)
    return written


def main(argv=None):
    cfg = override(config, sys.argv[1:] if argv is None else argv)
    if cfg["source"] not in ("synthetic", "real"):
        raise ValueError(f"unknown source {cfg['source']!r}; expected synthetic or real")
    for folder in build(cfg["grid"], cfg["source"], cfg["scores"], cfg["out"], cfg["allow_mixed_commits"],
                        cfg["actions"], cfg["samples"], cfg["allow_dirty"], cfg["allow_unreproducible"]):
        (folder / "run.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {cfg['out'] or Path(cfg['grid']) / 'tables' / cfg['source']}")


if __name__ == "__main__":
    main()
