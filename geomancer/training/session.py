"""Iterations, selection, staleness, persistence and the interaction trace. No compute.

An ITERATION is one execution of one step with one parameter set, on the upstream iterations
selected when it ran. Iterations are never overwritten. A selected iteration is VALID only when
it was computed from exactly the currently selected upstream iterations (and, for the first
step, from unchanged source data). Changing a selection therefore makes everything downstream
stale without running it, and returning to an older selection revalidates exactly what was
computed from it. The chain is linear on purpose: every CFlows stage feeds the next.

Layout, under `outputs/<session>/training/`:

    session.json            workflow, source, selections, notes, derived validity (+ .bak)
    trace.jsonl             append-only interaction events
    iterations/<step>/NNN/  iteration.json, plus the arrays and plots manyruns wrote

`iteration.json` files are the source of truth for history, so a failed `session.json` write
can lose at most a selection change, never an iteration.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from geomancer.training import choices, workflow as wf

SCHEMA_VERSION = 2
SESSION_FILE = "session.json"
TRACE_FILE = "trace.jsonl"

#: State slots handed from one step to the next (manyruns persists `emb` and `pseudotime`;
#: `time_labels` is persisted here because manyruns never writes labels).
CARRIED = ("emb", "pseudotime", "time_labels")

OK, NOT_APPLICABLE, ERROR, SKIPPED, INTERRUPTED, RUNNING = (
    "ok", "not_applicable", "error", "skipped", "interrupted", "running")
USABLE = frozenset({OK, NOT_APPLICABLE})


class TrainingError(Exception):
    """A refusal with a message the researcher can act on. Nothing was executed."""


# ── helpers ─────────────────────────────────────────────────────────────────────────────
def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


_HASHES: dict[tuple, str] = {}


def sha256_file(path: Path) -> str:
    """sha256 of a file, cached by (path, size, mtime) so status checks don't re-read it."""
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _HASHES:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        _HASHES[key] = h.hexdigest()
    return _HASHES[key]


def jsonable(value: Any) -> Any:
    """Numbers, strings and containers survive; arrays and objects become a short tag."""
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if hasattr(value, "shape") and getattr(value, "shape", None) not in ((), None):
        return f"<array shape={tuple(value.shape)}>"
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return f"<{type(value).__name__}>"


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write-then-rename in the same directory, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


_VERSIONS: Optional[dict] = None


def versions() -> dict:
    """Software provenance for the trace. No environment variables, paths or secrets."""
    global _VERSIONS
    if _VERSIONS is None:
        from importlib import metadata

        out = {"python": platform.python_version(), "platform": platform.platform()}
        for name in ("geomancer", "manyruns", "manylatents", "manylatents-omics", "phate",
                     "scanpy", "anndata", "torch", "lightning", "numpy", "scipy",
                     "scikit-learn"):
            try:
                out[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
        try:
            direct = json.loads(metadata.distribution("manyruns").read_text("direct_url.json")
                                or "{}")
            out["manyruns_revision"] = (direct.get("vcs_info") or {}).get("commit_id")
        except (metadata.PackageNotFoundError, ValueError):
            pass
        _VERSIONS = out
    return dict(_VERSIONS)


def fingerprint_source(project: dict, previous: Optional[dict] = None) -> dict:
    """Identify the input data without copying it. A file's hash is reused while its size and
    mtime match the previous fingerprint; generated sources are identified by generator."""
    folder, dataset = project.get("data_folder"), project.get("dataset")
    if folder and not str(folder).startswith("synthetic:"):
        path = Path(folder).resolve()
        if path.is_file():
            st = path.stat()
            prev = previous or {}
            same = (prev.get("path") == str(path) and prev.get("size") == st.st_size
                    and prev.get("mtime_ns") == st.st_mtime_ns and prev.get("sha256"))
            fp = {"kind": "file", "path": str(path), "size": st.st_size,
                  "mtime_ns": st.st_mtime_ns,
                  "sha256": prev["sha256"] if same else sha256_file(path)}
            fp["digest"] = fp["sha256"]
        elif path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file())
            listing = [[str(p.relative_to(path)), p.stat().st_size, sha256_file(p)] for p in files]
            fp = {"kind": "directory", "path": str(path), "files": len(listing),
                  "digest": hashlib.sha256(json.dumps(listing).encode()).hexdigest()}
        else:
            fp = {"kind": "missing", "path": str(path), "digest": None}
    else:
        from importlib import metadata

        engine = "manylatents" if dataset and not folder else "manyruns"
        try:
            gen = f"{engine} {metadata.version(engine)}"
        except metadata.PackageNotFoundError:
            gen = engine
        fp = {"kind": "generated", "ref": folder or dataset,
              "dataset_name": project.get("dataset_name"),
              "data_kwargs": project.get("data_kwargs") or {}, "generator": gen}
        stable = {k: fp[k] for k in ("ref", "dataset_name", "data_kwargs", "generator")}
        fp["digest"] = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    return fp


def verify_entry(entry: dict, root: Path) -> Optional[str]:
    path = root / entry["path"]
    if not path.is_file():
        return f"missing artifact {entry['path']}"
    if entry.get("sha256") and sha256_file(path) != entry["sha256"]:
        return f"artifact {entry['path']} changed on disk (sha256 mismatch)"
    return None


def load_inputs(state: dict, root: Path) -> dict:
    """Load the carried arrays of an upstream output state, refusing any that changed."""
    from manyruns import artifacts

    arrays = {}
    for slot, entry in state.items():
        problem = verify_entry(entry, root)
        if problem:
            raise TrainingError(f"input {slot} from {entry.get('from')} is unusable: {problem}")
        try:
            arrays[slot] = artifacts.load_array(root / entry["path"])
        except (OSError, ValueError) as exc:
            raise TrainingError(f"input {slot} from {entry.get('from')} cannot be loaded "
                                f"({entry['path']}): {exc}")
    return arrays


def iteration_id(step: str, number: int) -> str:
    return f"{step}#{number}"


def split_id(ident: str) -> tuple[str, int]:
    step, _, number = ident.rpartition("#")
    return step, int(number)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ── the session ─────────────────────────────────────────────────────────────────────────
class TrainingSession:
    """One researcher's guided walk through the workflow on one dataset."""

    def __init__(self, root: Path, data: dict, recovered: Optional[str] = None):
        self.root = Path(root)
        self.data = data
        self.recovered = recovered
        self.expert: Optional[str] = data.get("expert")
        self._source_now: Optional[dict] = None

    # -- location and lifecycle --
    @staticmethod
    def root_for(name: str, base: Path | str = "outputs") -> Path:
        from manyruns.app import _slug

        return Path(base) / _slug(name) / "training"

    @classmethod
    def create(cls, name: str, project: dict, time_mode: str, *, seed: int = 42,
               device: str = "cpu", smoke: bool = False, expert: Optional[str] = None,
               proposer: str = "rules", collect: bool = False,
               source_shape: Optional[list] = None,
               base: Path | str = "outputs") -> "TrainingSession":
        root = cls.root_for(name, base)
        if (root / SESSION_FILE).exists() or (root / f"{SESSION_FILE}.bak").exists():
            raise TrainingError(f"training session {name!r} already exists at {root}; "
                                f"resume it with `geomancer train status -s {name}`")
        flow = wf.workflow(time_mode)
        flow["sha256"] = hashlib.sha256(json.dumps(flow["steps"], sort_keys=True,
                                                   default=str).encode()).hexdigest()
        data = {
            "schema_version": SCHEMA_VERSION, "session_id": uuid.uuid4().hex, "name": name,
            "created_at": now(), "updated_at": now(), "expert": expert,
            "project": {k: project.get(k) for k in ("project", "engine", "dataset",
                                                    "data_folder", "dataset_name",
                                                    "data_kwargs", "modality", "time_key",
                                                    "declared_shape")},
            "workflow": flow,
            "defaults": {"seed": int(seed), "device": device, "smoke": bool(smoke),
                         "proposer": proposer, "collect": bool(collect)},
            "source": {"fingerprint": fingerprint_source(project), "shape": source_shape},
            "selected": {}, "notes": {},
        }
        root.mkdir(parents=True, exist_ok=True)
        session = cls(root, data)
        with session.lock():
            session.save()
            session.event("session_started", workflow=flow["name"], time_mode=time_mode,
                          steps=[s["name"] for s in flow["steps"]], project=data["project"],
                          source=data["source"], defaults=data["defaults"],
                          choices=_choices_snapshot(), versions=versions())
        return session

    @classmethod
    def open(cls, name: str, base: Path | str = "outputs") -> "TrainingSession":
        root = cls.root_for(name, base)
        main, bak = root / SESSION_FILE, root / f"{SESSION_FILE}.bak"
        recovered = None
        try:
            data = json.loads(main.read_text())
        except FileNotFoundError:
            if not bak.exists():
                raise TrainingError(f"no training session {name!r} under {root.parent.parent}; "
                                    f"start one with `geomancer train start DATA -s {name}`")
            data, recovered = None, f"{SESSION_FILE} is missing"
        except (OSError, ValueError) as exc:
            data, recovered = None, f"{SESSION_FILE} is unreadable ({type(exc).__name__})"
        if data is None:
            try:
                data = json.loads(bak.read_text())
            except (OSError, ValueError) as exc:
                raise TrainingError(f"{recovered} and its backup is unusable ({exc}); "
                                    f"iteration records under {root / 'iterations'} are intact")
            recovered += (f"; recovered the previous save from {bak.name}. "
                          "The latest selection change may be lost.")
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise TrainingError(f"session {name!r} was written with schema version {version}; "
                                f"this geomancer reads version {SCHEMA_VERSION}. Start a new "
                                "session (old sessions are not migrated).")
        return cls(root, data, recovered=recovered)

    @classmethod
    def list(cls, base: Path | str = "outputs") -> list[Path]:
        base = Path(base)
        return sorted(p.parent for p in base.glob(f"*/training/{SESSION_FILE}")) if base.is_dir() else []

    def save(self) -> None:
        """Atomically replace session.json, keeping the previous version as a backup."""
        self.data["updated_at"] = now()
        self.data["status"] = {c["step"]: {k: c[k] for k in ("selected", "state", "reason")}
                               for c in self.chain()}
        main = self.root / SESSION_FILE
        if main.exists():
            try:
                json.loads(main.read_text())
                shutil.copy2(main, main.with_name(f"{SESSION_FILE}.bak"))
            except (OSError, ValueError):
                pass  # never replace a good backup with a corrupt main file
        write_json_atomic(main, self.data)
        self.recovered = None

    @contextlib.contextmanager
    def lock(self):
        """One mutating command per session at a time."""
        path = self.root / ".lock"
        self.root.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                try:
                    pid = int(path.read_text().strip() or 0)
                except (OSError, ValueError):
                    pid = 0
                if pid and _alive(pid):
                    raise TrainingError(f"session is in use by process {pid}; wait for it to finish")
                path.unlink(missing_ok=True)  # left by a process that no longer exists
        else:
            raise TrainingError(f"could not lock {path}")
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self._settle_interrupted()
            yield
        finally:
            path.unlink(missing_ok=True)

    def event(self, event_name: str, /, **fields: Any) -> None:
        """Append one line to the interaction trace. Never rewrites earlier lines."""
        row = {"schema_version": SCHEMA_VERSION, "at": now(),
               "session_id": self.data["session_id"], "expert": self.expert, "event": event_name,
               **jsonable(fields)}
        with open(self.root / TRACE_FILE, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def trace(self) -> list[dict]:
        path = self.root / TRACE_FILE
        rows = []
        if path.is_file():
            for line in path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    rows.append({"event": "unreadable_line", "text": line[:200]})
        return rows

    # -- views --
    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def time_mode(self) -> str:
        return self.data["workflow"]["time_mode"]

    @property
    def steps(self) -> list[dict]:
        return self.data["workflow"]["steps"]

    def step_names(self) -> list[str]:
        return [s["name"] for s in self.steps]

    def step(self, name: str) -> dict:
        for s in self.steps:
            if s["name"] == name:
                return s
        raise TrainingError(f"{name!r} is not a step of this session's workflow; "
                            f"steps: {', '.join(self.step_names())}")

    def iterations(self, step: Optional[str] = None) -> list[dict]:
        """Every recorded iteration, read from disk, oldest first."""
        out = []
        for name in ([step] if step else self.step_names()):
            folder = self.root / "iterations" / name
            if not folder.is_dir():
                continue
            for d in sorted(folder.iterdir()):
                f = d / "iteration.json"
                if d.is_dir() and d.name.isdigit() and f.is_file():
                    try:
                        out.append(json.loads(f.read_text()))
                    except ValueError:
                        out.append({"iteration_id": iteration_id(name, int(d.name)),
                                    "step": name, "number": int(d.name), "status": ERROR,
                                    "error": "iteration.json is unreadable"})
        return out

    def iteration(self, step: str, number: int) -> dict:
        for it in self.iterations(step):
            if it["number"] == number:
                return it
        raise TrainingError(f"no iteration {iteration_id(step, number)}")

    def selected_iteration(self, step: str) -> Optional[dict]:
        sel = self.data["selected"].get(step)
        return self.iteration(*split_id(sel)) if sel else None

    def source_now(self) -> dict:
        if self._source_now is None:
            self._source_now = fingerprint_source(self.data["project"],
                                                  self.data["source"].get("fingerprint"))
        return self._source_now

    def problems(self, it: dict) -> list[str]:
        """Why this iteration cannot be used, independent of what is selected upstream."""
        if it.get("status") not in USABLE:
            return [f"status is {it.get('status')}"]
        out = []
        for entry in (it.get("outputs") or {}).get("state", {}).values():
            problem = verify_entry(entry, self.root)
            if problem:
                out.append(problem)
        return out

    def chain(self) -> list[dict]:
        """Selection and validity for every step, in workflow order.

        `valid`: selected, intact, and computed from exactly the selected upstream iterations
        (the first step also from unchanged source data). `stale`: selected but computed from
        something no longer selected. `unselected`, `broken` (selected but unusable) and
        `blocked` (an upstream step is not valid) complete the set.
        """
        selected = self.data.get("selected", {})
        by_id = {it["iteration_id"]: it for it in self.iterations()}
        out, upstream, prior_valid = [], {}, True
        source = self.source_now()
        for i, s in enumerate(self.steps):
            name = s["name"]
            sel = selected.get(name)
            it = by_id.get(sel) if sel else None
            row = {"step": name, "selected": sel, "upstream": dict(upstream)}
            if sel is None:
                row.update(state="unselected", reason="no iteration selected")
            elif it is None:
                row.update(state="broken", reason=f"{sel} has no iteration record on disk")
            elif self.problems(it):
                row.update(state="broken", reason="; ".join(self.problems(it)))
            else:
                mismatch = self._mismatch(it, upstream, source, i == 0)
                if mismatch:
                    row.update(state="stale", reason=mismatch)
                elif not prior_valid:
                    row.update(state="blocked", reason="an upstream step is not valid")
                else:
                    row.update(state="valid", reason="")
            row["compatible"] = [c["iteration_id"] for c in self.iterations(name)
                                 if prior_valid and not self.problems(c)
                                 and not self._mismatch(c, upstream, source, i == 0)]
            out.append(row)
            prior_valid = prior_valid and row["state"] == "valid"
            upstream[name] = sel
        return out

    def _mismatch(self, it: dict, upstream: dict, source: dict, first: bool) -> str:
        used = (it.get("inputs") or {}).get("upstream") or {}
        for step, want in upstream.items():
            if used.get(step) != want:
                return (f"computed from {used.get(step) or 'nothing'}, but {step} selection is "
                        f"{want or 'empty'}")
        if first:
            had = ((it.get("inputs") or {}).get("source") or {}).get("digest")
            if had != source.get("digest"):
                return "source data changed since this iteration ran"
        return ""

    # -- actions --
    def select(self, step: str, number: int) -> dict:
        """Choose the iteration of `step` that goes forward. Never runs anything."""
        self.step(step)
        ident = iteration_id(step, number)
        with self.lock():
            it = self.iteration(step, number)
            problems = self.problems(it)
            if problems:
                raise TrainingError(f"cannot select {ident}: {'; '.join(problems)}")
            row = next(c for c in self.chain() if c["step"] == step)
            if ident not in row["compatible"]:
                blocked = [c for c in self.chain()
                           if c["step"] in row["upstream"] and c["state"] != "valid"]
                if blocked:
                    raise TrainingError(f"cannot select {ident}: upstream {blocked[0]['step']} is "
                                        f"{blocked[0]['state']} ({blocked[0]['reason']})")
                reason = self._mismatch(it, row["upstream"], self.source_now(),
                                        step == self.steps[0]["name"])
                raise TrainingError(f"cannot select {ident}: it was {reason}. Select its "
                                    f"upstream iterations first, or rerun {step}.")
            previous = self.data["selected"].get(step)
            if previous == ident:
                return {"changed": False, "previous": previous, "invalidated": [],
                        "revalidated": []}
            before = {c["step"]: c["state"] for c in self.chain()}
            self.data["selected"][step] = ident
            after = {c["step"]: c["state"] for c in self.chain()}
            change = {
                "changed": True, "previous": previous,
                "invalidated": [s for s in after if before[s] == "valid" and after[s] != "valid"],
                "revalidated": [s for s in after if before[s] != "valid" and after[s] == "valid"
                                and s != step],
            }
            self.save()
            self.event("selected", step=step, iteration=ident, previous=previous,
                       invalidated=change["invalidated"], revalidated=change["revalidated"])
        return change

    def add_note(self, step: str, number: int, text: str) -> None:
        ident = iteration_id(step, number)
        with self.lock():
            self.iteration(step, number)
            self.data.setdefault("notes", {}).setdefault(ident, []).append(
                {"at": now(), "text": text, "expert": self.expert})
            self.save()
            self.event("note", iteration=ident, text=text)

    def record_feedback(self, step: str, number: int, text: str, result: Optional[dict] = None,
                        *, kind: str = "feedback", **extra: Any) -> str:
        """Record what the researcher said about an iteration, verbatim, and what was proposed.

        `kind` is "feedback" (words that asked for proposals) or "why" (a reason given for a
        direct edit). Returns the id that links the decision and any resulting iteration."""
        feedback_id = "fb-" + uuid.uuid4().hex[:12]
        with self.lock():
            it = self.iteration(step, number)
            self.event("feedback", feedback_id=feedback_id, kind=kind, step=step,
                       iteration=it["iteration_id"], text=text,
                       params=_resolved_values(it),
                       diagnostics=(it.get("outputs") or {}).get("diagnostics"),
                       **(result or {}), **extra)
        return feedback_id

    def record_decision(self, feedback_id: str, decision: str, *, step: str,
                        proposal: Optional[dict] = None, overrides: Optional[dict] = None,
                        iteration: Optional[str] = None) -> None:
        """accepted | edited | rejected, with the final params and the iteration they made."""
        with self.lock():
            self.event("feedback_decision", feedback_id=feedback_id, decision=decision,
                       step=step, proposal=proposal, overrides=overrides,
                       resulting_iteration=iteration)

    def record(self, event_name: str, /, **fields: Any) -> None:
        """Any other interaction event (loop started/ended), under the session lock."""
        with self.lock():
            self.event(event_name, **fields)

    def next_step(self) -> tuple[Optional[str], list[dict]]:
        chain = self.chain()
        for row in chain:
            if row["state"] != "valid":
                return row["step"], chain
        return None, chain

    def base_iteration(self, step: str, number: Optional[int]) -> Optional[dict]:
        """Whose params a new iteration starts from: explicit, else selected, else latest."""
        if number is not None:
            return self.iteration(step, number)
        sel = self.selected_iteration(step)
        if sel:
            return sel
        done = [i for i in self.iterations(step) if i.get("status") in USABLE]
        return done[-1] if done else None

    def current_params(self, step: str, number: Optional[int] = None) -> dict:
        base = self.base_iteration(step, number)
        return dict(((base or {}).get("params") or {}).get("requested")
                    or self.step(step).get("params") or {})

    def n_cells(self, step: str) -> Optional[int]:
        """Rows reaching `step`: from the selected upstream arrays, else the source shape."""
        index = self.step_names().index(step)
        if index:
            prior = self.selected_iteration(self.step_names()[index - 1])
            for entry in ((prior or {}).get("outputs") or {}).get("state", {}).values():
                shape = entry.get("shape")
                if shape:
                    return int(shape[0])
        shape = (self.data.get("source") or {}).get("shape")
        return int(shape[0]) if shape and shape[0] else None

    def run(self, step: str, overrides: Optional[dict] = None, *, base: Optional[int] = None,
            note: Optional[str] = None, seed: Optional[int] = None, select: bool = False,
            reason: str = "run", feedback_id: Optional[str] = None,
            executor: Optional[Callable[[dict], dict]] = None) -> dict:
        """Execute ONE new iteration of `step` on the selected upstream iterations.

        Refuses before writing anything when an upstream step is not valid or a parameter is
        invalid. Never runs another step. Selects only when `select=True` and it succeeded.
        """
        overrides = dict(overrides or {})
        spec = self.step(step)
        with self.lock():
            chain = self.chain()
            names = self.step_names()
            index = names.index(step)
            for row in chain[:index]:
                if row["state"] != "valid":
                    self.event("rejected", step=step, overrides=overrides,
                               reason=f"upstream {row['step']} is {row['state']}")
                    raise TrainingError(
                        f"cannot run {step}: upstream {row['step']} is {row['state']}"
                        f"{' — ' + row['reason'] if row['reason'] else ''}. "
                        + _upstream_hint(row))
            upstream = dict(chain[index]["upstream"])
            base_it = self.base_iteration(step, base)
            base_params = self.current_params(step, base)
            requested = {**base_params, **overrides}
            prior = self.selected_iteration(names[index - 1]) if index else None
            in_state = ((prior or {}).get("outputs") or {}).get("state", {})
            try:
                arrays = load_inputs(in_state, self.root)
                n_cells = _n_rows(arrays) or self.n_cells(step)
                wf.validate({**spec, "params": base_params}, overrides, n_cells)
            except (TrainingError, wf.ParamError) as exc:
                self.event("rejected", step=step, overrides=overrides, reason=str(exc))
                raise TrainingError(str(exc)) from exc
            defaults = self.data["defaults"]
            seed = int(defaults["seed"] if seed is None else seed)
            resolved = wf.resolve(step, requested, n_cells, defaults["smoke"])
            duplicate = next((i["iteration_id"] for i in self.iterations(step)
                              if i.get("status") in USABLE
                              and (i.get("params") or {}).get("requested") == requested
                              and (i.get("inputs") or {}).get("upstream") == upstream
                              and (i.get("repro") or {}).get("seed") == seed), None)
            replay = self._replay(index) if step in wf.RECOMPUTED or step == "pca" else []
            number = self._allocate(step)
            ident = iteration_id(step, number)
            out_dir = self.root / "iterations" / step / f"{number:03d}"
            source = self.source_now() if index == 0 else None
            rec = {
                "schema_version": SCHEMA_VERSION, "session_id": self.data["session_id"],
                "iteration_id": ident, "step": step, "number": number, "status": RUNNING,
                "reason": reason, "expert": self.expert, "feedback_id": feedback_id,
                "started_at": now(), "finished_at": None, "duplicate_of": duplicate,
                "params": {"declared": dict(spec.get("params") or {}), "overrides": overrides,
                           "requested": requested, "resolved": resolved,
                           "base_iteration": (base_it or {}).get("iteration_id"),
                           "changed": wf.diff((base_it or {}).get("params", {}).get("resolved"),
                                              resolved)},
                "inputs": {"upstream": upstream, "state": in_state,
                           "replayed": [r["iteration_id"] for r in replay],
                           "source": ({k: source.get(k) for k in ("kind", "digest", "ref",
                                                                 "path", "size")}
                                      if source else None)},
                "repro": {"seed": seed, "engine": self.data["project"]["engine"],
                          "device": defaults["device"], "smoke": defaults["smoke"],
                          "time_mode": self.time_mode,
                          "workflow_sha256": self.data["workflow"]["sha256"],
                          "versions": versions()},
                "note": note, "outputs": None, "error": None,
            }
            write_json_atomic(out_dir / "iteration.json", rec)
            self.event("iteration_started", iteration=ident, step=step, reason=reason,
                       upstream=upstream, overrides=overrides, changed=rec["params"]["changed"],
                       seed=seed, note=note, duplicate_of=duplicate, feedback_id=feedback_id)
            job = {"project": self.data["project"], "step": {**spec, "params": requested},
                   "time_mode": self.time_mode, "inputs": arrays,
                   "replay": [{"step": r["step"], "expect": r["expect"]} for r in replay],
                   "out_dir": out_dir, "seed": seed, "device": defaults["device"],
                   "smoke": defaults["smoke"]}
            if executor is None:
                from geomancer.training import execute

                executor = execute.execute_step
            try:
                result = executor(job)
            except KeyboardInterrupt:
                self._finish(rec, out_dir, INTERRUPTED, error="interrupted (KeyboardInterrupt)")
                raise
            except Exception as exc:  # noqa: BLE001 - a failed attempt is recorded, not fatal
                result = {"outcome": ERROR, "detail": f"{type(exc).__name__}: {exc}"}
            self._commit(rec, out_dir, in_state, result, spec)
            if select and rec["status"] in USABLE:
                self.data["selected"][step] = ident
                self.save()
                self.event("selected", step=step, iteration=ident, previous=None,
                           via="--select", invalidated=[], revalidated=[])
            return rec

    def _replay(self, index: int) -> list[dict]:
        """The selected recomputed-step iterations before `index`, to rebuild the matrix."""
        out = []
        for name in self.step_names()[:index]:
            if name not in wf.RECOMPUTED:
                continue
            it = self.selected_iteration(name)
            out.append({"iteration_id": it["iteration_id"],
                        "step": {**self.step(name), "params": it["params"]["requested"]},
                        "expect": ((it.get("outputs") or {}).get("recomputed") or {}).get("X")})
        return out

    def _allocate(self, step: str) -> int:
        folder = self.root / "iterations" / step
        folder.mkdir(parents=True, exist_ok=True)
        taken = [int(d.name) for d in folder.iterdir() if d.is_dir() and d.name.isdigit()]
        number = max(taken, default=0) + 1
        while True:
            try:
                (folder / f"{number:03d}").mkdir()
                return number
            except FileExistsError:
                number += 1

    def _commit(self, rec: dict, out_dir: Path, in_state: dict, result: dict, spec: dict) -> None:
        outcome = result.get("outcome")
        if outcome == "ok":
            status = OK
        elif outcome == "skipped" and spec.get("group") == "prep" and \
                choices.NOT_APPLICABLE == "pass_through":
            status = NOT_APPLICABLE
        elif outcome == "skipped":
            status = SKIPPED
        else:
            status = ERROR
        error = None if status == OK else (result.get("detail") or f"outcome {outcome}")
        state = {slot: dict(entry) for slot, entry in in_state.items()}
        byproducts = {}
        for slot, path in (result.get("artifacts") or {}).items():
            p = Path(str(path))
            if not p.is_file():
                if slot in CARRIED and status == OK:
                    status, error = ERROR, f"output {slot} was not persisted: {path}"
                byproducts[slot] = {"note": str(path)}
                continue
            entry = {"path": os.path.relpath(p.resolve(), self.root.resolve()),
                     "sha256": sha256_file(p), "from": rec["iteration_id"],
                     "shape": _npy_shape(p)}
            if slot in CARRIED and slot in (result.get("carried") or {}):
                state[slot] = entry
            else:
                byproducts[slot] = entry
        rec["outputs"] = {
            "state": state if status in USABLE else {}, "byproducts": byproducts,
            "recomputed": result.get("recomputed"),
            "plots": [os.path.relpath(Path(p).resolve(), self.root.resolve())
                      for p in result.get("plots") or [] if Path(p).is_file()],
            "g": jsonable(result.get("g")), "executed_params": jsonable(result.get("params")),
            "bounded": jsonable(result.get("bounded")), "caveats": result.get("caveats"),
            "diagnostics": jsonable(result.get("diagnostics")),
            "model": result.get("model"), "manyruns_run_id": result.get("run_id"),
            "engine_seconds": result.get("seconds"),
        }
        if status == NOT_APPLICABLE:
            rec["not_applicable"] = error
            error = None
        self._finish(rec, out_dir, status, error=error)

    def _finish(self, rec: dict, out_dir: Path, status: str, *, error: Optional[str]) -> None:
        rec["status"], rec["error"], rec["finished_at"] = status, error, now()
        rec["seconds"] = round((_dt.datetime.fromisoformat(rec["finished_at"])
                                - _dt.datetime.fromisoformat(rec["started_at"])).total_seconds(), 3)
        write_json_atomic(out_dir / "iteration.json", rec)
        self.event("iteration_finished", iteration=rec["iteration_id"], status=status,
                   error=error, not_applicable=rec.get("not_applicable"), seconds=rec["seconds"],
                   diagnostics=(rec.get("outputs") or {}).get("diagnostics"),
                   outputs={k: v.get("path") for k, v in ((rec.get("outputs") or {})
                                                          .get("state") or {}).items()
                            if v.get("from") == rec["iteration_id"]})
        try:
            self.save()
        except OSError as exc:  # the iteration record above is already durable
            print(f"warning: could not save {SESSION_FILE} ({exc}); "
                  "history is intact in iteration records", file=sys.stderr)

    def _settle_interrupted(self) -> None:
        """Under the lock no other process is running, so `running` means it died."""
        for it in self.iterations():
            if it.get("status") == RUNNING:
                out_dir = self.root / "iterations" / it["step"] / f"{it['number']:03d}"
                it.update(status=INTERRUPTED, finished_at=None,
                          error="the process ended before this iteration finished")
                write_json_atomic(out_dir / "iteration.json", it)
                self.event("iteration_finished", iteration=it["iteration_id"],
                           status=INTERRUPTED, error=it["error"], recovered=True)


def _resolved_values(it: dict) -> dict:
    return {k: v.get("value") for k, v in ((it.get("params") or {}).get("resolved") or {}).items()}


def _upstream_hint(row: dict) -> str:
    if row["state"] == "unselected" and row["compatible"]:
        return f"Select one: geomancer train select {row['step']} N"
    if row["state"] == "unselected":
        return "Run it first: geomancer train continue"
    return "Recompute explicitly: geomancer train continue"


def _n_rows(arrays: dict) -> Optional[int]:
    for value in arrays.values():
        shape = getattr(value, "shape", None)
        if shape:
            return int(shape[0])
    return None


def _npy_shape(path: Path) -> Optional[list]:
    """The array shape from a .npy header, without loading the data."""
    if path.suffix != ".npy":
        return None
    try:
        import numpy as np

        with open(path, "rb") as f:
            major, _minor = np.lib.format.read_magic(f)
            reader = (np.lib.format.read_array_header_1_0 if major == 1
                      else np.lib.format.read_array_header_2_0)
            shape, _order, _dtype = reader(f)
        return list(shape)
    except Exception:  # noqa: BLE001 - shape is a convenience for bounds checks
        return None


def _choices_snapshot() -> dict:
    """The reviewable choices in force when the session started, recorded in the trace."""
    return {k: getattr(choices, k) for k in (
        "GENE_FILTER_MIN_CELLS", "NORMALIZE_TARGET_SUM", "TRANSFORM_METHOD", "PCA_COMPONENTS",
        "NOT_APPLICABLE", "PHATE_COMPONENTS", "PSEUDOTIME_BINS", "K_PER_TIMEPOINT",
        "SUBCLUSTER_ORDER", "RULES_VERSION")}
