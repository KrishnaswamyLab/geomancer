"""The interactive loop and the one-shot `feedback` / `compare` verbs.

Inside `geomancer train loop -s NAME [STEP]`:

    <words>        what looks wrong ("too blurry") → numbered proposals to accept, edit, reject
    k=v …          a direct edit, run at once; then an optional "why?" (Enter skips)
    keep [N]       select the current iteration (or N) to go forward
    compare A [B]  differences between two iterations (B defaults to the current one)
    show [N]       one iteration in full, with its plot
    status         the whole session
    done           leave; asks which iteration to keep if none is selected
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from geomancer.training import feedback as fb, workflow as wf
from geomancer.training.session import USABLE, TrainingError, TrainingSession

HELP = __doc__.split("\n\n", 1)[1].strip()


# ── shared pieces ───────────────────────────────────────────────────────────────────────
def context(session: TrainingSession, step: str, number: int, text: str) -> fb.Context:
    it = session.iteration(step, number)
    if it.get("status") not in USABLE:
        raise TrainingError(f"{it['iteration_id']} has status {it.get('status')}; give feedback "
                            "on an iteration that ran")
    params = {k: v.get("value") for k, v in ((it.get("params") or {}).get("resolved") or {}).items()}
    return fb.Context(step=step, text=text, params=params,
                      diagnostics=(it.get("outputs") or {}).get("diagnostics") or {},
                      n_cells=session.n_cells(step))


def describe(result: fb.Result, step: str) -> list[str]:
    lines = []
    for i, p in enumerate(result.proposals, 1):
        where = p.source.get("rule") or p.source.get("model") or p.proposer
        said = f"; you said {', '.join(repr(m) for m in p.matched)}" if p.matched else ""
        lines.append(f"  {i}. {p.param} {p.old!r} → {p.new!r} — {p.reason} [{p.proposer}: {where}{said}]")
    for c in result.contradicted:
        why = (f"{c['requires']} is {c['observed']!r}" if "requires" in c else c.get("reason", ""))
        lines.append(f"  (not proposed: rule {c['rule']} matched {c['matched']}, but {why})")
    for n in result.notes:
        lines.append(f"  ({n})")
    if result.unmatched:
        offered = ", ".join(wf.PARAMS.get(step, {})) or "none"
        lines.append(f"  no proposal for that. Name a change directly, e.g. one of: {offered}")
    return lines


def compare(session: TrainingSession, step: str, a: int, b: int) -> list[str]:
    ia, ib = session.iteration(step, a), session.iteration(step, b)
    out = [f"{ia['iteration_id']} vs {ib['iteration_id']}"]

    def values(it, key):
        if key == "params":
            return {k: v.get("value") for k, v in ((it.get("params") or {}).get("resolved") or {}).items()}
        if key == "diagnostics":
            return (it.get("outputs") or {}).get("diagnostics") or {}
        g = (it.get("outputs") or {}).get("g") or {}
        return {k: v for k, v in g.items() if k.startswith(step) and not isinstance(v, dict)}

    for key in ("params", "diagnostics", "results"):
        va, vb = values(ia, key), values(ib, key)
        diffs = [f"    {k}: {va.get(k)!r} → {vb.get(k)!r}" for k in sorted(set(va) | set(vb))
                 if va.get(k) != vb.get(k)]
        out.append(f"  {key}:" + ("" if diffs else " same"))
        out.extend(diffs)
    up_a, up_b = ((ia.get("inputs") or {}).get("upstream"), (ib.get("inputs") or {}).get("upstream"))
    if up_a != up_b:
        out.append(f"  inputs differ: {up_a} vs {up_b}")
    for it in (ia, ib):
        for rel in (it.get("outputs") or {}).get("plots") or []:
            out.append(f"  plot {it['iteration_id']}: {session.root / rel}")
    return out


def _run_from_feedback(session, step, number, fid, decision, overrides, proposal=None, **kw):
    rec = session.run(step, overrides, base=number, reason=f"feedback-{decision}",
                      feedback_id=fid, **kw)
    session.record_decision(fid, decision, step=step, proposal=proposal, overrides=overrides,
                            iteration=rec["iteration_id"])
    return rec


def feedback_once(session: TrainingSession, step: str, number: int, text: str, *,
                  accept: Optional[int] = None, edit: Optional[list] = None, reject: bool = False,
                  select: bool = False, seed: Optional[int] = None, note: Optional[str] = None,
                  show_plots: bool = True) -> int:
    from geomancer.training.cli import report

    ctx = context(session, step, number, text)
    result = fb.propose(ctx, fb.proposers_for(session.data["defaults"].get("proposer", "rules")))
    fid = session.record_feedback(step, number, text, result.as_dict())
    print(f"feedback on {step}#{number} recorded ({fid}):")
    print("\n".join(describe(result, step)))
    kw = {"select": select, "seed": seed, "note": note}
    if accept is not None:
        if not 1 <= accept <= len(result.proposals):
            raise TrainingError(f"no proposal {accept}; there are {len(result.proposals)}")
        p = result.proposals[accept - 1]
        rec = _run_from_feedback(session, step, number, fid, "accepted", {p.param: p.new},
                                 proposal=p.as_dict(), **kw)
        return report(session, rec, not show_plots)
    if edit:
        rec = _run_from_feedback(session, step, number, fid, "edited",
                                 wf.parse_assignments(edit), **kw)
        return report(session, rec, not show_plots)
    if reject:
        session.record_decision(fid, "rejected", step=step)
        print("recorded: none of these fits")
        return 0
    if result.proposals:
        print(f"decide with: geomancer train feedback -s {session.name} {step} {number} "
              f"{text!r} --accept 1   (or --edit k=v, --reject)")
    return 0


# ── the loop ────────────────────────────────────────────────────────────────────────────
def run_loop(session: TrainingSession, step: Optional[str] = None, *,
             read: Optional[Callable[[str], str]] = None, write: Callable[[str], Any] = print,
             collect: Optional[bool] = None, show_plots: bool = True) -> int:
    from geomancer.training.cli import _print_status, print_iteration

    read = read or (lambda prompt: input(prompt))
    collect = session.data["defaults"].get("collect", False) if collect is None else collect
    if step is None:
        step, _ = session.next_step()
        if step is None:
            write("every step has a valid selection; name a step to revisit, e.g. "
                  f"geomancer train loop -s {session.name} phate")
            return 0
    session.step(step)
    row = _row(session, step)
    blocked = [r for r in session.chain()[:session.step_names().index(step)] if r["state"] != "valid"]
    if blocked:
        write(f"refused: upstream {blocked[0]['step']} is {blocked[0]['state']}; "
              f"loop over it first: geomancer train loop -s {session.name} {blocked[0]['step']}")
        return 1
    session.record("loop_started", step=step, collect=collect)
    if row["selected"] in row["compatible"]:
        current = session.iteration(step, int(row["selected"].rpartition("#")[2]))
    elif row["compatible"]:
        current = session.iteration(step, int(row["compatible"][-1].rpartition("#")[2]))
    else:
        write(f"running {step} on the current selections …")
        current = session.run(step, reason="loop")
    show = lambda it: print_iteration(session, it, show_plots=show_plots)  # noqa: E731
    show(current)
    write(f"commands: words about the plot · k=v · keep [N] · compare A [B] · show [N] · "
          f"status · done · help")

    def ask_why(prompt: str, decision: str, iteration: dict, **extra) -> None:
        try:
            text = read(prompt).strip()
        except EOFError:
            text = ""
        if text:
            session.record_feedback(step, iteration["number"], text, kind="why",
                                    decision=decision, **extra)

    while True:
        try:
            line = read(f"{current['iteration_id']}> ").strip()
        except EOFError:
            line = "done"
        if not line:
            continue
        word, _, rest = line.partition(" ")
        word = word.lower()
        try:
            if word in ("help", "?"):
                write(HELP)
            elif word == "status":
                _print_status(session)
            elif word == "show":
                show(session.iteration(step, int(rest)) if rest.strip() else current)
            elif word == "compare":
                nums = [int(x) for x in rest.split()]
                a, b = (nums + [current["number"]])[:2] if nums else (None, None)
                if a is None:
                    write("usage: compare A [B]")
                else:
                    write("\n".join(compare(session, step, a, b)))
            elif word == "keep":
                n = int(rest) if rest.strip() else current["number"]
                change = session.select(step, n)
                write(f"kept {step}#{n}"
                      + (f"; now stale: {', '.join(change['invalidated'])}" if change["invalidated"] else ""))
                if collect:
                    ask_why("why this one? (Enter to skip) ", "kept", session.iteration(step, n))
            elif word in ("done", "quit", "exit"):
                if _row(session, step)["state"] != "valid":
                    answer = read(f"no iteration of {step} is kept. Keep which? "
                                  "(number, or 'none') ").strip().lower()
                    if answer.isdigit():
                        session.select(step, int(answer))
                        write(f"kept {step}#{answer}")
                    else:
                        session.record("left_without_keeping", step=step)
                        write(f"left {step} without keeping an iteration")
                kept = session.data["selected"].get(step)
                session.record("loop_ended", step=step, kept=kept)
                if kept:
                    write(f"leaving {step}; {kept} goes forward. Next: "
                          f"geomancer train continue -s {session.name}")
                return 0
            elif wf.looks_like_assignment(line):
                overrides = wf.parse_assignments(line.split())
                rec = session.run(step, overrides, base=current["number"], reason="loop-edit")
                show(rec)
                if rec.get("duplicate_of"):
                    write(f"note: same parameters, inputs and seed as {rec['duplicate_of']}")
                if rec["status"] in USABLE:
                    current = rec
                ask_why("why this change? (Enter to skip) ", "direct_edit", rec,
                        overrides=overrides)
            else:
                ctx = context(session, step, current["number"], line)
                result = fb.propose(ctx, fb.proposers_for(
                    session.data["defaults"].get("proposer", "rules")))
                fid = session.record_feedback(step, current["number"], line, result.as_dict())
                write("\n".join(describe(result, step)))
                if not result.proposals:
                    continue
                choice = read("accept N · edit k=v · reject > ").strip()
                verb, _, arg = choice.partition(" ")
                if verb.lower() in ("accept", "a") or verb.isdigit():
                    k = int(verb if verb.isdigit() else (arg or "1"))
                    p = result.proposals[k - 1]
                    rec = _run_from_feedback(session, step, current["number"], fid, "accepted",
                                             {p.param: p.new}, proposal=p.as_dict())
                elif verb.lower() in ("edit", "e") or wf.looks_like_assignment(choice):
                    tokens = (arg if verb.lower() in ("edit", "e") else choice).split()
                    rec = _run_from_feedback(session, step, current["number"], fid, "edited",
                                             wf.parse_assignments(tokens))
                else:
                    session.record_decision(fid, "rejected", step=step)
                    write("recorded: none of these fits")
                    if collect:
                        ask_why("what would you change instead? (Enter to skip) ", "rejected",
                                current, about_feedback=fid)
                    continue
                show(rec)
                if rec["status"] in USABLE:
                    current = rec
                if collect:
                    ask_why("why? (Enter to skip) ", "accepted_or_edited", rec, about_feedback=fid)
        except (TrainingError, wf.ParamError, ValueError, IndexError) as exc:
            write(f"refused: {exc}")


def _row(session: TrainingSession, step: str) -> dict:
    return next(r for r in session.chain() if r["step"] == step)
