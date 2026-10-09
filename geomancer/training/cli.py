"""`geomancer train …` — the command line for expert-guided training mode.

    geomancer train start DATA -s NAME        open data; time mode auto-detected
    geomancer train status -s NAME            steps, selections, validity, history
    geomancer train params -s NAME STEP       offered parameters, current values, defaults
    geomancer train run -s NAME STEP [k=v …]  one new iteration of STEP; nothing downstream
    geomancer train select -s NAME STEP N     the iteration that goes forward
    geomancer train continue -s NAME [k=v …]  run the next step that is not valid
    geomancer train show -s NAME STEP N       one iteration: params, provenance, outputs, plots
    geomancer train note -s NAME STEP N TEXT  attach an optional expert note
    geomancer train feedback -s NAME STEP N "too blurry" [--accept K|--edit k=v|--reject]
    geomancer train compare -s NAME STEP A B  parameter, diagnostic and result differences
    geomancer train loop -s NAME [STEP]       run, look, give feedback or edit, keep, repeat
    geomancer train trace -s NAME             the append-only interaction trace
    geomancer train list                      training sessions in this directory

Parameters are `key=value`, as in the `manyruns>` REPL (`phate knn=10`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from geomancer.training import workflow as wf
from geomancer.training.session import (OK, SESSION_FILE, TRACE_FILE, USABLE, TrainingError,
                                        TrainingSession, iteration_id)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="geomancer train",
                                description="Expert-guided training mode: run the CFlows workflow "
                                            "one step at a time, keep every iteration, choose "
                                            "what goes forward.")
    sub = p.add_subparsers(dest="verb", required=True)

    def with_session(sp):
        sp.add_argument("-s", "--session", required=True, help="training session name")
        sp.add_argument("--expert", help="pseudonymous expert id for this command's records")
        return sp

    st = with_session(sub.add_parser("start", help="open a dataset and start a session"))
    st.add_argument("data", help="a data path or a catalog dataset name")
    st.add_argument("--time", choices=("auto", *wf.TIME_MODES), default="auto",
                    help="real: subcluster measured timepoints (CFlows); pseudotime: bin DPT. "
                         "auto picks real when the data has a time axis")
    st.add_argument("--time-key", help="obs column holding time, when not auto-detected")
    st.add_argument("--engine", default="manylatents",
                    help="compute engine; must persist arrays (default manylatents)")
    st.add_argument("--seed", type=int, default=42, help="default seed for every iteration")
    st.add_argument("--device", default="cpu")
    st.add_argument("--smoke", action="store_true",
                    help="MIOFlow single-batch plumbing check instead of full training")
    st.add_argument("--data-kwargs", nargs="*", metavar="KEY=VALUE",
                    help="generator settings for engine-generated datasets")
    st.add_argument("--proposer", choices=("rules", "rules+model"), default="rules",
                    help="who turns feedback into proposals; rules+model asks an LLM when no rule "
                         "matches (needs geomancer[agents] and an Anthropic key)")
    st.add_argument("--collect", action="store_true",
                    help="play-session mode: the loop asks 'why?' after every decision")

    s = with_session(sub.add_parser("status", help="selections, validity and history"))
    s.add_argument("--json", action="store_true")

    pa = with_session(sub.add_parser("params", help="offered parameters of a step"))
    pa.add_argument("step")

    r = with_session(sub.add_parser("run", help="run one new iteration of a step"))
    r.add_argument("step")
    r.add_argument("assignments", nargs="*", metavar="KEY=VALUE")
    r.add_argument("--from", dest="base", type=int, metavar="N",
                   help="start from iteration N's params (default: selected, else latest)")
    _run_flags(r)

    c = with_session(sub.add_parser("continue", help="run the next step that is not valid"))
    c.add_argument("assignments", nargs="*", metavar="KEY=VALUE")
    _run_flags(c)

    se = with_session(sub.add_parser("select", help="choose the iteration that goes forward"))
    se.add_argument("step")
    se.add_argument("number", type=int)

    sh = with_session(sub.add_parser("show", help="inspect one iteration"))
    sh.add_argument("step")
    sh.add_argument("number", type=int)
    sh.add_argument("--json", action="store_true")
    sh.add_argument("--no-plots", action="store_true")

    n = with_session(sub.add_parser("note", help="attach an expert note to an iteration"))
    n.add_argument("step")
    n.add_argument("number", type=int)
    n.add_argument("text")

    t = with_session(sub.add_parser("trace", help="print the interaction trace"))
    t.add_argument("--json", action="store_true")

    fb = with_session(sub.add_parser("feedback", help="describe what is wrong; get proposals"))
    fb.add_argument("step")
    fb.add_argument("number", type=int)
    fb.add_argument("text")
    decide = fb.add_mutually_exclusive_group()
    decide.add_argument("--accept", type=int, metavar="K", help="run proposal K")
    decide.add_argument("--edit", nargs="+", metavar="KEY=VALUE", help="run your own change")
    decide.add_argument("--reject", action="store_true", help="record that none fits")
    _run_flags(fb)

    cmp_ = with_session(sub.add_parser("compare", help="differences between two iterations"))
    cmp_.add_argument("step")
    cmp_.add_argument("a", type=int)
    cmp_.add_argument("b", type=int)

    lp = with_session(sub.add_parser("loop", help="interactive: run, look, feedback, keep"))
    lp.add_argument("step", nargs="?", help="default: the next step that is not valid")
    lp.add_argument("--collect", action="store_true", help="ask 'why?' after every decision")
    lp.add_argument("--no-plots", action="store_true", help="print plot paths only")

    sub.add_parser("list", help="training sessions under ./outputs")
    return p


def _run_flags(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--seed", type=int, help="override the session seed for this iteration")
    sp.add_argument("--note", help="optional expert note recorded with the iteration")
    sp.add_argument("--select", action="store_true",
                    help="select the new iteration if it succeeds")
    sp.add_argument("--no-plots", action="store_true", help="print plot paths only")


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return VERBS[args.verb](args)
    except (TrainingError, wf.ParamError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted: the iteration is recorded as interrupted; "
              "earlier results are unchanged", file=sys.stderr)
        return 130


# ── verbs ───────────────────────────────────────────────────────────────────────────────
def cmd_start(args) -> int:
    from geomancer.training import execute

    if (TrainingSession.root_for(args.session) / SESSION_FILE).exists():
        raise TrainingError(f"training session {args.session!r} already exists; "
                            f"resume with `geomancer train status -s {args.session}`")
    if args.engine == "mock":
        raise TrainingError("the mock engine writes no arrays, so no step could consume another's "
                            "output; use --engine manylatents")
    project = resolve_data(args.data, args.session, args.engine, args.data_kwargs, args.time_key)
    source = execute.load_source(project)
    has_time = source["kind"] == "time" and source["labels"] is not None
    if args.time == "auto":
        mode = wf.REAL if has_time else wf.PSEUDOTIME
    elif args.time == wf.REAL and not has_time:
        raise TrainingError("--time real needs measured time labels; none were found"
                            + ("" if args.time_key else " (name the column with --time-key)"))
    else:
        mode = args.time
    x = source["array"]
    shape = list(x.shape) if getattr(x, "shape", None) is not None else project.get("declared_shape")
    if args.proposer == "rules+model":
        import importlib.util

        if not any(importlib.util.find_spec(m) for m in ("anthropic", "manyagents")):
            raise TrainingError("--proposer rules+model needs the optional model tier: "
                                "install geomancer[agents] (and set ANTHROPIC_API_KEY)")
    session = TrainingSession.create(args.session, project, mode, seed=args.seed,
                                     device=args.device, smoke=args.smoke, expert=args.expert,
                                     proposer=args.proposer, collect=args.collect,
                                     source_shape=shape)
    print(f"started training session {args.session!r} at {session.root}")
    print(f"time mode: {mode}" + (" (measured time found)" if has_time else ""))
    _print_status(session)
    return 0


def cmd_status(args) -> int:
    session = _open(args)
    if args.json:
        print(json.dumps({"session": session.data, "chain": session.chain()}, indent=2,
                         default=str))
        return 0
    _print_status(session)
    return 0


def cmd_params(args) -> int:
    session = _open(args)
    session.step(args.step)
    base = session.base_iteration(args.step, None)
    requested = session.current_params(args.step)
    resolved = wf.resolve(args.step, requested, session.n_cells(args.step),
                          session.data["defaults"]["smoke"])
    specs = wf.PARAMS.get(args.step, {})
    origin = (base or {}).get("iteration_id") or "the workflow"
    print(f"{args.step} parameters (current values from {origin}):")
    for key, entry in resolved.items():
        p = specs.get(key)
        print(f"  {key:<20} {entry['value']!r:<10} [{entry['source']}]  "
              f"{wf.rule(p) if p else ''}" + (f" — {p.doc}" if p and p.doc else ""))
    if not resolved:
        print("  (this step offers no parameters)")
    print(f"change with: geomancer train run -s {session.name} {args.step} KEY=VALUE")
    return 0


def cmd_run(args) -> int:
    session = _open(args)
    overrides = wf.parse_assignments(args.assignments)
    rec = session.run(args.step, overrides, base=args.base, note=args.note, seed=args.seed,
                      select=args.select)
    return report(session, rec, args.no_plots)


def cmd_continue(args) -> int:
    session = _open(args)
    step, chain = session.next_step()
    if step is None:
        print("every step has a valid selection; nothing to continue. "
              f"Revise a step with `geomancer train run -s {session.name} STEP KEY=VALUE`.")
        return 0
    row = next(r for r in chain if r["step"] == step)
    others = [i for i in row["compatible"] if i != row["selected"]]
    if others:
        raise TrainingError(
            f"{step} already has iterations computed on the current selections "
            f"({', '.join(others)}) but none is selected. Select one with "
            f"`geomancer train select -s {session.name} {step} N`, or rerun it explicitly "
            f"with `geomancer train run -s {session.name} {step} KEY=VALUE`.")
    if row["state"] in ("stale", "broken"):
        print(f"{row['selected']} is {row['state']}: {row['reason']}.")
    base = session.base_iteration(step, None)
    print(f"continuing: running {step} on "
          + (", ".join(f"{k}={v}" for k, v in row["upstream"].items()) or "the source data")
          + f" with params from {(base or {}).get('iteration_id') or 'the workflow'}")
    overrides = wf.parse_assignments(args.assignments)
    rec = session.run(step, overrides, note=args.note, seed=args.seed, select=args.select,
                      reason="continue")
    return report(session, rec, args.no_plots)


def cmd_select(args) -> int:
    session = _open(args)
    change = session.select(args.step, args.number)
    ident = iteration_id(args.step, args.number)
    if not change["changed"]:
        print(f"{ident} is already selected")
        return 0
    print(f"selected {ident}" + (f" (was {change['previous']})" if change.get("previous") else ""))
    if change["invalidated"]:
        print(f"now stale, not recomputed: {', '.join(change['invalidated'])}")
    if change["revalidated"]:
        print(f"valid again (computed from this selection): {', '.join(change['revalidated'])}")
    _print_status(session)
    return 0


def cmd_show(args) -> int:
    session = _open(args)
    it = session.iteration(args.step, args.number)
    if args.json:
        print(json.dumps(it, indent=2))
        return 0
    print_iteration(session, it, show_plots=not args.no_plots)
    return 0


def cmd_note(args) -> int:
    session = _open(args)
    session.add_note(args.step, args.number, args.text)
    print(f"noted on {iteration_id(args.step, args.number)}")
    return 0


def cmd_trace(args) -> int:
    session = _open(args)
    rows = session.trace()
    if args.json:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        return 0
    for row in rows:
        detail = {k: v for k, v in row.items()
                  if k not in ("schema_version", "at", "session_id", "event", "versions",
                               "project", "source", "defaults", "choices")
                  and v not in (None, [], {})}
        print(f"{row.get('at', '?')}  {row.get('event', '?'):<20} "
              + " ".join(f"{k}={json.dumps(v, default=str)}" for k, v in detail.items()))
    print(f"({len(rows)} events in {session.root / TRACE_FILE})")
    return 0


def cmd_list(args) -> int:
    roots = TrainingSession.list()
    if not roots:
        print("no training sessions under ./outputs")
    for root in roots:
        data = json.loads((root / SESSION_FILE).read_text())
        print(f"{data['name']:<24} {data['workflow']['name']:<20} updated {data['updated_at']}")
    return 0


def cmd_feedback(args) -> int:
    from geomancer.training import loop

    session = _open(args)
    return loop.feedback_once(session, args.step, args.number, args.text, accept=args.accept,
                              edit=args.edit, reject=args.reject, select=args.select,
                              seed=args.seed, note=args.note, show_plots=not args.no_plots)


def cmd_compare(args) -> int:
    from geomancer.training import loop

    session = _open(args)
    print("\n".join(loop.compare(session, args.step, args.a, args.b)))
    return 0


def cmd_loop(args) -> int:
    from geomancer.training import loop

    session = _open(args)
    return loop.run_loop(session, args.step, collect=args.collect or None,
                         show_plots=not args.no_plots)


VERBS = {"start": cmd_start, "status": cmd_status, "params": cmd_params, "run": cmd_run,
         "continue": cmd_continue, "select": cmd_select, "show": cmd_show, "note": cmd_note,
         "trace": cmd_trace, "list": cmd_list, "feedback": cmd_feedback,
         "compare": cmd_compare, "loop": cmd_loop}


# ── data resolution (public manyruns APIs only) ─────────────────────────────────────────
def resolve_data(raw: str, name: str, engine: str, kwargs: Optional[list],
                 time_key: Optional[str]) -> dict:
    """A data path or a catalog dataset name -> what `execute.load_source` needs.

    Never downloads: a catalog sample whose file is absent is refused with where it should be.
    """
    from manyruns import app, catalog

    overrides = {}
    for pair in kwargs or []:
        key, _, value = pair.partition("=")
        if key and value:
            overrides[key] = wf.coerce(value)
    src = Path(raw)
    project = {"project": name, "engine": engine, "dataset": None, "data_folder": None,
               "dataset_name": None, "data_kwargs": overrides, "time_key": time_key,
               "declared_shape": None}
    if src.exists():
        try:
            modality = app.detect_modality(src)
        except ValueError as exc:
            raise TrainingError(str(exc))
        return {**project, "data_folder": str(src), "modality": modality}
    try:
        ds = catalog.load_dataset(raw)
    except Exception as exc:  # noqa: BLE001 - unknown name or invalid declaration
        raise TrainingError(f"cannot resolve {raw!r}: not an existing path or a catalog "
                            f"dataset ({exc})")
    handle = ds.get("handle") or {}
    project.update(dataset_name=raw, modality=ds.get("modality", "unknown"))
    if handle.get("kind") == "path":
        ref = str(handle["ref"])
        if ref.startswith("synthetic:"):
            return {**project, "data_folder": ref}
        from manyruns.shell import dataset_ref_path

        found = dataset_ref_path(ref)
        if found is None:
            raise TrainingError(f"{raw!r} names {ref}, which is not on this machine; put the file "
                                "there or pass its path (training mode never downloads)")
        return {**project, "data_folder": str(found)}
    pinned = dict(ds.get("params") or {})
    dims = ds.get("dims") or {}
    effective = {**pinned, **overrides}
    shape = ([dims.get("n_samples"), dims.get("n_features")]
             if dims and effective == pinned else None)
    return {**project, "dataset": handle.get("ref"), "data_kwargs": effective,
            "declared_shape": shape}


# ── rendering ───────────────────────────────────────────────────────────────────────────
def _open(args) -> TrainingSession:
    session = TrainingSession.open(args.session)
    if session.recovered:
        print(f"warning: {session.recovered}", file=sys.stderr)
    if getattr(args, "expert", None):
        session.expert = args.expert
    return session


_MARK = {"valid": "✓", "stale": "!", "unselected": "·", "broken": "✗", "blocked": "…"}


def _print_status(session: TrainingSession) -> None:
    d = session.data
    proj = d["project"]
    print(f"session {d['name']} · data {proj.get('dataset_name') or proj.get('data_folder') or proj.get('dataset')}"
          f" · time {session.time_mode} · engine {proj['engine']} · seed {d['defaults']['seed']}"
          + (f" · expert {session.expert}" if session.expert else "")
          + (" · smoke" if d["defaults"]["smoke"] else ""))
    notes = d.get("notes") or {}
    for row in session.chain():
        print(f" {_MARK.get(row['state'], '?')} {row['step']:<16} {row['state']:<10} "
              f"selected {row['selected'] or '—'}"
              + (f"  ({row['reason']})" if row["reason"] and row["state"] != "unselected" else ""))
        for it in session.iterations(row["step"]):
            ident = it["iteration_id"]
            tags = []
            if ident == row["selected"]:
                tags.append("selected")
            if ident in row["compatible"]:
                tags.append("compatible")
            changed = (it.get("params") or {}).get("changed") or {}
            delta = " ".join(f"{k}:{a!r}→{b!r}" for k, (a, b) in changed.items())
            print(f"     {ident:<20} {it.get('status'):<14} {it.get('started_at', '')[:19]}"
                  f"  {delta or _short_params(it)}"
                  + (f"  [{', '.join(tags)}]" if tags else "")
                  + (f"  note: {it['note']}" if it.get("note") else "")
                  + (f"  +{len(notes[ident])} note(s)" if ident in notes else "")
                  + (f"  error: {it['error'][:80]}" if it.get("error") else ""))
    step, _ = session.next_step()
    print(f"next: geomancer train continue -s {d['name']}   (would run {step})" if step
          else "all steps valid.")


def _short_params(it: dict) -> str:
    req = (it.get("params") or {}).get("requested") or {}
    return " ".join(f"{k}={v!r}" for k, v in req.items()) or "(declared defaults)"


def report(session: TrainingSession, rec: dict, no_plots: bool) -> int:
    print_iteration(session, rec, show_plots=not no_plots)
    ident = rec["iteration_id"]
    if rec.get("duplicate_of"):
        print(f"note: same parameters, inputs and seed as {rec['duplicate_of']}; "
              "this reproduces it rather than trying something new")
    if rec["status"] in USABLE:
        sel = session.data["selected"].get(rec["step"])
        if sel != ident:
            print(f"keep it with: geomancer train select -s {session.name} {rec['step']} "
                  f"{rec['number']}   (selected now: {sel or 'none'})")
        else:
            print(f"{ident} is selected.")
        return 0
    print(f"{ident} did not succeed; earlier iterations and selections are unchanged.")
    return 1


def print_iteration(session: TrainingSession, it: dict, *, show_plots: bool) -> None:
    params = it.get("params") or {}
    out = it.get("outputs") or {}
    print(f"{it['iteration_id']}  status {it.get('status')}  "
          f"{it.get('seconds', '?')}s  seed {(it.get('repro') or {}).get('seed')}")
    if it.get("not_applicable"):
        print(f"  passed through (not applicable): {it['not_applicable']}")
    up = (it.get("inputs") or {}).get("upstream") or {}
    print("  inputs:  " + (", ".join(f"{k}={v}" for k, v in up.items() if v) or "source data"))
    for slot, entry in ((it.get("inputs") or {}).get("state") or {}).items():
        print(f"           {slot} ← {entry['path']} (from {entry['from']})")
    if (it.get("inputs") or {}).get("replayed"):
        print(f"           matrix recomputed from source via {', '.join(it['inputs']['replayed'])}")
    print("  params:  " + " ".join(f"{k}={v['value']!r}" for k, v in
                                    (params.get("resolved") or {}).items()))
    if params.get("changed"):
        print(f"  changed vs {params.get('base_iteration')}: "
              + " ".join(f"{k} {a!r}→{b!r}" for k, (a, b) in params["changed"].items()))
    if it.get("error"):
        print(f"  error:   {it['error']}")
    g = out.get("g") or {}
    step = it["step"]
    shown = {k: v for k, v in g.items()
             if not k.startswith("_") and k not in ("n_samples", "n_features")
             and (k.startswith(step) or k.startswith(f"{step}_")) and not isinstance(v, dict)}
    if shown:
        print("  results: " + ", ".join(f"{k}={_fmt(v)}" for k, v in shown.items()))
    if out.get("diagnostics"):
        print("  diagnostics: " + ", ".join(f"{k}={_fmt(v)}" for k, v in out["diagnostics"].items()))
    for slot, entry in (out.get("state") or {}).items():
        if entry.get("from") == it["iteration_id"]:
            print(f"  output:  {slot} → {entry['path']}")
    for c in out.get("caveats") or []:
        print(f"  caveat:  {c}")
    if out.get("model"):
        print(f"  model:   {out['model']}")
    for note in (session.data.get("notes") or {}).get(it["iteration_id"], []):
        print(f"  note:    {note['text']}  ({note['at'][:19]})")
    for rel in out.get("plots") or []:
        path = session.root / rel
        if not (show_plots and _draw(path)):
            print(f"  plot:    {path}")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return json.dumps(v, default=str) if isinstance(v, (list, dict)) else str(v)


def _draw(path: Path) -> bool:
    """Inline image through manyruns' figure renderer when the terminal supports it."""
    from manyruns import figures

    shim = type("_Console", (), {"file": sys.stdout, "is_terminal": sys.stdout.isatty()})()
    try:
        return bool(figures.draw(shim, path))
    except Exception:  # noqa: BLE001 - a picture is a convenience; the path always prints
        return False
