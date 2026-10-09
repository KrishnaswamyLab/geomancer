"""The trace format a learner outside Geomancer may rely on (schema version 2).

Every line of `trace.jsonl` is one JSON object with the COMMON fields, an `event` name from
EVENTS, and at least that event's required fields. Fields may be added within a schema
version; renaming or removing one is a new version. `tests/test_training_contract.py` checks
every event a full session emits against this table.

The learning example for a proposer is a `feedback` event (words, `params`, `diagnostics`,
`proposals`) joined by `feedback_id` to its `feedback_decision` (accepted / edited /
rejected, final `overrides`) and, through `resulting_iteration`, to what that change produced.
"""
from __future__ import annotations

SCHEMA_VERSION = 2

COMMON = ("schema_version", "at", "session_id", "expert", "event")

EVENTS: dict[str, tuple[str, ...]] = {
    "session_started": ("workflow", "time_mode", "steps", "project", "source", "defaults",
                        "choices", "versions"),
    "iteration_started": ("iteration", "step", "reason", "upstream", "overrides", "changed",
                          "seed", "duplicate_of", "feedback_id"),
    "iteration_finished": ("iteration", "status", "error"),
    "rejected": ("step", "overrides", "reason"),
    "selected": ("step", "iteration", "previous", "invalidated", "revalidated"),
    "note": ("iteration", "text"),
    "feedback": ("feedback_id", "kind", "step", "iteration", "text", "params", "diagnostics"),
    "feedback_decision": ("feedback_id", "decision", "step", "proposal", "overrides",
                          "resulting_iteration"),
    "loop_started": ("step", "collect"),
    "loop_ended": ("step", "kept"),
    "left_without_keeping": ("step",),
}

#: Extra fields required on a `feedback` event of kind "feedback" (words that asked for
#: proposals), as opposed to kind "why" (a reason given for the researcher's own decision).
PROPOSAL_FIELDS = ("proposals", "contradicted", "consulted", "notes", "unmatched")
PROPOSAL_ITEM = ("step", "param", "old", "new", "reason", "proposer", "source", "matched")


def problems(row: dict) -> list[str]:
    """Why a trace row does not meet the contract; empty when it does."""
    out = [f"missing {k}" for k in COMMON if k not in row]
    name = row.get("event")
    if name not in EVENTS:
        return out + [f"unknown event {name!r}"]
    if row.get("schema_version") != SCHEMA_VERSION:
        out.append(f"schema_version {row.get('schema_version')!r}")
    out += [f"{name}: missing {k}" for k in EVENTS[name] if k not in row]
    if name == "feedback" and row.get("kind") == "feedback":
        out += [f"feedback: missing {k}" for k in PROPOSAL_FIELDS if k not in row]
        for i, p in enumerate(row.get("proposals") or []):
            out += [f"feedback proposal {i}: missing {k}" for k in PROPOSAL_ITEM if k not in p]
    return out
