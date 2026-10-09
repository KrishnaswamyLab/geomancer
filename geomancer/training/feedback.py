"""Plain-language feedback → proposed parameter changes. No side effects, no session.

A PROPOSER turns what the researcher said about a result ("too blurry", "branches merged"),
the step's current parameters and its diagnostics into ranked proposals. Two exist:

    RuleProposer   the versioned table in `choices.RULES` (deterministic, always on)
    ModelProposer  an LLM limited to this step's offered parameters and a direction; it is
                   asked only when no rule matched, and only when the session enabled it

A learned proposer trained on recorded sessions would be a third implementation of the same
interface. Every proposal records which proposer produced it, so a trace says where each
suggestion came from. Nothing here runs anything: the researcher accepts, edits or rejects.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol

from geomancer.training import choices, workflow as wf


@dataclass
class Proposal:
    step: str
    param: str
    old: Any
    new: Any
    reason: str
    proposer: str               # "rules" | "model" | a learned proposer's name
    source: dict = field(default_factory=dict)  # rule id + version, or model id + prompt hash
    matched: list = field(default_factory=list)  # the researcher's words that triggered it

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Context:
    step: str
    text: str
    params: dict                # the critiqued iteration's resolved values {name: value}
    diagnostics: dict
    n_cells: Optional[int] = None


@dataclass
class Result:
    proposals: list
    contradicted: list          # rules whose words matched but whose diagnostic disagreed
    consulted: list             # proposer names asked, in order
    notes: list                 # e.g. "model tier unavailable"

    @property
    def unmatched(self) -> bool:
        return not self.proposals

    def as_dict(self) -> dict:
        return {"proposals": [p.as_dict() for p in self.proposals],
                "contradicted": self.contradicted, "consulted": self.consulted,
                "notes": self.notes, "unmatched": self.unmatched}


class Proposer(Protocol):
    name: str

    def propose(self, ctx: Context) -> tuple[list[Proposal], list[dict]]:
        """Proposals, plus rules/reasons that matched but were contradicted."""


# ── moving one parameter one step ───────────────────────────────────────────────────────
def step_value(step: str, param: str, current: Any, op: str, amount: float,
               n_cells: Optional[int]) -> Optional[Any]:
    """Apply one step to a value, clipped to the parameter's legal range. None if it cannot
    move (non-numeric with no starting point, or already at a bound)."""
    spec = wf.PARAMS.get(step, {}).get(param)
    if spec is None:
        return None
    base = current
    if not isinstance(base, (int, float)) or isinstance(base, bool):
        base = choices.CONCRETE_FROM.get(param)
        if base is None:
            return None
    new = base * amount if op == "mul" else base + amount
    integer = spec.kind in ("int", "int_or_auto")
    if integer:
        rounded = int(round(new))
        if rounded == base:  # a multiplicative step too small to move an integer
            rounded = int(base) + (1 if new > base else -1)
        new = rounded
    lo = spec.min
    if spec.above is not None:
        lo = spec.above + (1 if integer else 1e-9)
    hi = spec.max
    if n_cells is not None and spec.below_cells:
        hi = n_cells - 1 if hi is None else min(hi, n_cells - 1)
    if n_cells is not None and spec.at_most_cells:
        hi = n_cells if hi is None else min(hi, n_cells)
    if lo is not None:
        new = max(new, lo)
    if hi is not None:
        new = min(new, hi)
    if not integer:
        new = round(float(new), 6)
    return None if new == current else new


def _matches(text: str, phrase: str) -> bool:
    """Whole-phrase match that ignores negated mentions ("not blurry", "isn't noisy")."""
    for m in re.finditer(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", text):
        before = text[max(0, m.start() - 12):m.start()]
        if not re.search(r"\b(not|no|isn't|aren't|never|without)\s+(too\s+|very\s+)?$", before):
            return True
    return False


class RuleProposer:
    name = "rules"

    def __init__(self, rules=None, version: Optional[str] = None):
        self.rules = choices.RULES if rules is None else rules
        self.version = version or choices.RULES_VERSION

    def propose(self, ctx: Context) -> tuple[list[Proposal], list[dict]]:
        text = ctx.text.lower()
        scored, contradicted = [], []
        for order, rule in enumerate(self.rules):
            if rule.step != ctx.step:
                continue
            words = [w for w in rule.words if _matches(text, w)]
            if not words:
                continue
            if rule.requires and not ctx.diagnostics.get(rule.requires):
                contradicted.append({"rule": rule.id, "matched": words,
                                     "requires": rule.requires,
                                     "observed": ctx.diagnostics.get(rule.requires)})
                continue
            current = ctx.params.get(rule.param)
            new = step_value(ctx.step, rule.param, current, rule.op, rule.amount, ctx.n_cells)
            if new is None:
                contradicted.append({"rule": rule.id, "matched": words,
                                     "reason": f"{rule.param} cannot move further from {current!r}"})
                continue
            scored.append((-len(words), order, Proposal(
                step=ctx.step, param=rule.param, old=current, new=new, reason=rule.reason,
                proposer=self.name, source={"rule": rule.id, "rules_version": self.version},
                matched=words)))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [p for _, _, p in scored], contradicted


class ModelProposer:
    """An LLM picks ONE offered parameter and a direction; the step size comes from the rule
    table (or ±1 / ×1.1). It receives the words, parameter names and values, and diagnostic
    numbers — never data, plots or paths. Off unless the session enabled it."""

    name = "model"

    def __init__(self, call=None):
        self._call = call

    def propose(self, ctx: Context) -> tuple[list[Proposal], list[dict]]:
        offered = {k: v for k, v in ctx.params.items() if k in wf.PARAMS.get(ctx.step, {})}
        if not offered:
            return [], []
        prompt = (f"Step: {ctx.step}\nParameters and current values: {json.dumps(offered, default=str)}\n"
                  f"Diagnostics: {json.dumps(ctx.diagnostics, default=str)}\n"
                  f"The scientist said: {ctx.text!r}\n"
                  "Which single parameter should move, and which way?")
        system = ("You map a scientist's description of a plot to ONE parameter of the step that "
                  "produced it, and a direction. Never invent a parameter; never choose a magnitude.")
        schema = {"type": "object",
                  "properties": {"param": {"type": "string", "enum": sorted(offered)},
                                 "direction": {"type": "string", "enum": ["up", "down"]},
                                 "why": {"type": "string"}},
                  "required": ["param", "direction"], "additionalProperties": False}
        call, model = self._call, None
        if call is None:
            from manyruns import agents

            call, model = agents.call_tool, getattr(agents, "MODEL", None)
        got = call(system=system, prompt=prompt, name="adjust",
                   description="Name one parameter of this step to move, and the direction.",
                   schema=schema) or {}
        param, direction = got.get("param"), got.get("direction")
        if param not in offered or direction not in ("up", "down"):
            return [], []
        op, amount = _model_step(ctx.step, param, direction)
        new = step_value(ctx.step, param, offered[param], op, amount, ctx.n_cells)
        if new is None:
            return [], []
        return [Proposal(step=ctx.step, param=param, old=offered[param], new=new,
                         reason=got.get("why") or f"model suggests {param} {direction}",
                         proposer=self.name,
                         source={"model": model,
                                 "prompt_sha256": hashlib.sha256((system + prompt).encode()).hexdigest()})], []


def _model_step(step: str, param: str, direction: str) -> tuple[str, float]:
    """The rule table's step for this parameter and direction, else ±1 for an int, ×1.1/÷1.1."""
    for rule in choices.RULES:
        if rule.step == step and rule.param == param:
            up = (rule.op == "mul" and rule.amount > 1) or (rule.op == "add" and rule.amount > 0)
            if up == (direction == "up"):
                return rule.op, rule.amount
    spec = wf.PARAMS[step][param]
    if spec.kind in ("int", "int_or_auto"):
        return "add", 1 if direction == "up" else -1
    return "mul", 1.1 if direction == "up" else 1 / 1.1


def proposers_for(setting: str) -> list:
    """The session's `proposer` setting → proposers, consulted in order."""
    out: list = [RuleProposer()]
    if setting == "rules+model":
        out.append(ModelProposer())
    return out


def propose(ctx: Context, proposers: list) -> Result:
    """Rules first; a later proposer is consulted only when nothing before it proposed."""
    proposals, contradicted, consulted, notes = [], [], [], []
    for proposer in proposers:
        if proposals:
            break
        consulted.append(proposer.name)
        try:
            got, against = proposer.propose(ctx)
        except Exception as exc:  # noqa: BLE001 - an optional tier must not break the loop
            notes.append(f"{proposer.name} unavailable: {type(exc).__name__}")
            continue
        proposals.extend(got)
        contradicted.extend(against)
    seen, ranked = set(), []
    for p in proposals:
        if p.param not in seen:
            seen.add(p.param)
            ranked.append(p)
    return Result(ranked[:choices.MAX_PROPOSALS], contradicted, consulted, notes)
