"""A portable bundle of one session's records, for pooling sessions to learn from.

Contains `session.json`, `trace.jsonl`, every `iteration.json`, a `MANIFEST.json`, and
optionally the plots. Never the input data or any array. Absolute paths are reduced to file
names, and the bundle is refused if a home or temporary-directory path still appears.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
from pathlib import Path
from typing import Any

from geomancer.training import choices, contract
from geomancer.training.session import TrainingError, TrainingSession

_MACHINE = re.compile(r"/(?:Users|home)/[^/\s\"']+|/(?:private/)?var/folders/|/(?:private/)?tmp/")


def scrub(value: Any) -> Any:
    """Absolute paths → their file name; everything else unchanged."""
    if isinstance(value, str):
        if value.startswith("/") and len(value) > 1:
            return Path(value).name
        return value
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def export(session: TrainingSession, out: Path | str = "exports", *, plots: bool = False) -> Path:
    rows = session.trace()
    bad = [(i, p) for i, row in enumerate(rows) for p in contract.problems(row)]
    if bad:
        raise TrainingError(f"trace row {bad[0][0]} breaks the trace contract: {bad[0][1]}")
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = Path(out) / f"{session.root.parent.name}-{stamp}"
    dest.mkdir(parents=True)
    try:
        _write(dest / "session.json", scrub(session.data))
        (dest / "trace.jsonl").write_text(
            "".join(json.dumps(scrub(r), sort_keys=True) + "\n" for r in rows))
        iterations = session.iterations()
        for it in iterations:
            folder = dest / "iterations" / it["step"] / f"{it['number']:03d}"
            _write(folder / "iteration.json", scrub(it))
            if plots:
                for rel in (it.get("outputs") or {}).get("plots") or []:
                    src = session.root / rel
                    if src.is_file():
                        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dest / rel)
        _write(dest / "MANIFEST.json", {
            "schema_version": contract.SCHEMA_VERSION, "session_id": session.data["session_id"],
            "name": session.name, "exported_at": stamp, "events": len(rows),
            "iterations": len(iterations), "rules_version": choices.RULES_VERSION,
            "plots": plots, "contains": "records only; no input data and no arrays",
        })
        leaks = [p for p in dest.rglob("*") if p.suffix in (".json", ".jsonl")
                 and _MACHINE.search(p.read_text())]
        if leaks:
            raise TrainingError(f"a machine path survived scrubbing in {leaks[0].name}; "
                                "nothing was exported")
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return dest


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
