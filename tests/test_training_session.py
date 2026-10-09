"""Iterations, selection, staleness, persistence and failure handling.

MOCKED BOUNDARY: `geomancer.training.execute.execute_step` is replaced by `FakeEngine`
(`tests/training_fakes.py`). These tests check bookkeeping, not science.
"""
import json
import os

import pytest

from geomancer.training import choices, execute, session as sess, workflow as wf
from geomancer.training.session import TrainingError, TrainingSession
from training_fakes import N, FakeEngine, digest

PREP = ["filter_genes", "normalize", "transform"]


@pytest.fixture
def engine(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(execute, "execute_step", fake)
    return fake


def make(tmp_path, monkeypatch, mode=wf.PSEUDOTIME, name="demo", **kw):
    monkeypatch.chdir(tmp_path)
    project = {"project": name, "engine": "manylatents", "dataset": None,
               "data_folder": "synthetic:time-course", "dataset_name": "synthetic_timecourse",
               "data_kwargs": {}, "modality": "synthetic"}
    return TrainingSession.create(name, project, mode, source_shape=[N, 5], **kw)


@pytest.fixture
def session(tmp_path, monkeypatch, engine):
    return make(tmp_path, monkeypatch)


def states(s):
    return {row["step"]: row["state"] for row in s.chain()}


def iteration_dirs(s):
    return sorted(str(p.relative_to(s.root)) for p in (s.root / "iterations").glob("*/*"))


def walk(s, upto=None):
    """Run and select every step in order (through `upto`)."""
    for step in s.step_names():
        assert s.run(step, select=True)["status"] in sess.USABLE, step
        if step == upto:
            return


# ── the workflow and its preprocessing ──────────────────────────────────────────────────
def test_pseudotime_workflow_order_and_declared_values(session):
    assert session.step_names() == [*PREP, "pca", "phate", "dpt", "discretize_time", "mioflow"]
    assert session.step("filter_genes")["params"] == {"min_cells": choices.GENE_FILTER_MIN_CELLS}
    assert session.step("pca")["params"] == {"n_components": choices.PCA_COMPONENTS}


def test_real_time_workflow_swaps_the_time_step(tmp_path, monkeypatch, engine):
    s = make(tmp_path, monkeypatch, mode=wf.REAL)
    assert s.step_names()[-2] == "subcluster_time"


def test_prep_and_pca_recompute_the_matrix_from_selected_iterations(session, engine):
    walk(session, upto="transform")
    rec = session.run("pca", select=True)
    assert engine.calls[-1]["replay"] == PREP
    assert rec["inputs"]["replayed"] == ["filter_genes#1", "normalize#1", "transform#1"]
    assert "X" in session.iteration("transform", 1)["outputs"]["recomputed"]
    phate = session.run("phate")
    assert engine.calls[-1]["replay"] == []                  # phate reads the stored PCA output
    assert phate["inputs"]["state"]["emb"]["from"] == "pca#1"


def test_not_applicable_prep_passes_through(session, engine):
    engine.not_applicable = {"normalize", "transform"}
    walk(session, upto="pca")
    it = session.iteration("normalize", 1)
    assert it["status"] == sess.NOT_APPLICABLE and "not counts" in it["not_applicable"]
    assert states(session)["pca"] == "valid"


def test_not_applicable_can_be_made_to_stop(session, engine, monkeypatch):
    monkeypatch.setattr(choices, "NOT_APPLICABLE", "stop")
    engine.not_applicable = {"normalize"}
    walk(session, upto="filter_genes")
    rec = session.run("normalize", select=True)
    assert rec["status"] == sess.SKIPPED and session.data["selected"].get("normalize") is None
    with pytest.raises(TrainingError, match="upstream normalize is unselected"):
        session.run("transform")


# ── parameters ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("overrides, message", [
    ({"knnn": 10}, "no parameter 'knnn'"),
    ({"knn": "ten"}, "must be an integer"),
    ({"knn": 0}, ">= 1"),
    ({"knn": N}, f"< the {N} cells"),
    ({"t": "never"}, "or 'auto'"),
    ({"gamma": 2}, "<= 1"),
    ({"decay": 0}, "> 0"),
    ({"n_components": None}, "cannot be null"),
])
def test_invalid_parameters_are_refused_before_execution(session, engine, overrides, message):
    walk(session, upto="pca")
    calls = len(engine.calls)
    with pytest.raises(TrainingError, match=message):
        session.run("phate", overrides)
    assert len(engine.calls) == calls
    assert not (session.root / "iterations" / "phate").exists() or \
        not list((session.root / "iterations" / "phate").iterdir())
    assert session.trace()[-1]["event"] == "rejected"


def test_choice_and_required_parameters(tmp_path, monkeypatch, engine):
    s = make(tmp_path, monkeypatch, mode=wf.REAL)
    with pytest.raises(TrainingError, match="one of log1p, sqrt"):
        walk(s, upto="normalize")
        s.run("transform", {"method": "log2"})
    for step in ("transform", "pca", "phate", "dpt"):
        s.run(step, select=True)
    with pytest.raises(TrainingError, match="has no default and must be set"):
        s.run("subcluster_time")
    rec = s.run("subcluster_time", {"k_per_timepoint": 3})
    assert rec["status"] == "ok"


def test_resolved_defaults_and_sources(session, engine):
    walk(session, upto="pca")
    rec = session.run("phate", {"knn": 10, "t": "auto"})
    resolved = rec["params"]["resolved"]
    assert resolved["knn"] == {"value": 10, "source": "requested"}
    assert resolved["decay"]["value"] == 40 and "PHATEModule" in resolved["decay"]["source"]


# ── iterations and provenance ───────────────────────────────────────────────────────────
def test_reruns_keep_distinct_iterations_and_record_changes(session, engine):
    walk(session, upto="pca")
    a = session.run("phate", {"knn": 10})
    b = session.run("phate", {"t": 30})
    c = session.run("phate", {"knn": 7}, base=1)
    assert [r["iteration_id"] for r in (a, b, c)] == ["phate#1", "phate#2", "phate#3"]
    assert b["params"]["requested"] == {"n_components": 3, "knn": 10, "t": 30}
    assert b["params"]["changed"] == {"t": [15, 30]}
    assert c["params"]["changed"] == {"knn": [10, 7]}
    paths = {r["outputs"]["state"]["emb"]["path"] for r in (a, b, c)}
    assert len(paths) == 3 and all((session.root / p).is_file() for p in paths)


def test_duplicate_parameters_are_flagged(session, engine):
    walk(session, upto="pca")
    session.run("phate", {"knn": 10})
    again = session.run("phate", {"knn": 10})
    assert again["duplicate_of"] == "phate#1"
    assert session.run("phate", {"knn": 10}, seed=7)["duplicate_of"] is None


def test_inputs_are_the_selected_upstream_artifacts(session, engine):
    from manyruns import artifacts

    walk(session, upto="pca")
    session.run("phate")
    second = session.run("phate", {"knn": 9})
    session.select("phate", 2)
    dpt = session.run("dpt")
    emb = second["outputs"]["state"]["emb"]
    assert dpt["inputs"]["upstream"]["phate"] == "phate#2"
    assert dpt["inputs"]["state"]["emb"] == emb
    assert engine.calls[-1]["inputs"]["emb"] == digest(artifacts.load_array(session.root / emb["path"]))
    assert dpt["outputs"]["state"]["pseudotime"]["from"] == "dpt#1"


# ── staleness ───────────────────────────────────────────────────────────────────────────
DOWN = ["dpt", "discretize_time", "mioflow"]


def test_changing_phate_marks_downstream_stale_without_running_it(session, engine):
    walk(session)
    calls = len(engine.calls)
    session.run("phate", {"knn": 12})
    change = session.select("phate", 2)
    assert change["invalidated"] == DOWN
    assert len(engine.calls) == calls + 1
    assert [states(session)[s] for s in DOWN] == ["stale"] * 3
    assert len(iteration_dirs(session)) == 9


def test_changing_preprocessing_marks_everything_after_it_stale(session, engine):
    walk(session)
    session.run("filter_genes", {"min_cells": 5})
    change = session.select("filter_genes", 2)
    assert change["invalidated"] == session.step_names()[1:]


def test_stale_outputs_cannot_be_consumed(session, engine):
    walk(session)
    session.run("phate", {"knn": 12}, select=True)
    for step in ("discretize_time", "mioflow"):
        with pytest.raises(TrainingError, match="upstream dpt is stale"):
            session.run(step)
    with pytest.raises(TrainingError, match="cannot select dpt#1"):
        session.select("dpt", 1)


def test_returning_to_an_older_upstream_revalidates_only_its_descendants(session, engine):
    walk(session)
    session.run("phate", {"knn": 12}, select=True)
    session.run("dpt", select=True)
    session.select("phate", 1)
    assert states(session)["dpt"] == "stale"
    session.select("dpt", 1)
    assert set(states(session).values()) == {"valid"}


def test_source_change_invalidates_the_first_step(session, engine, monkeypatch):
    walk(session, upto="filter_genes")
    monkeypatch.setattr(session, "_source_now", {"digest": "different"})
    assert states(session)["filter_genes"] == "stale"


def test_tampered_artifact_is_reported_and_refused(session, engine):
    walk(session, upto="pca")
    emb = session.root / session.iteration("pca", 1)["outputs"]["state"]["emb"]["path"]
    emb.write_bytes(b"not the array")
    row = next(r for r in session.chain() if r["step"] == "pca")
    assert row["state"] == "broken" and "sha256 mismatch" in row["reason"]
    with pytest.raises(TrainingError, match="upstream pca is broken"):
        session.run("phate")


# ── persistence ─────────────────────────────────────────────────────────────────────────
def test_save_and_reopen_preserve_selection_history_and_validity(session, engine):
    walk(session)
    session.run("phate", {"knn": 12}, select=True)
    session.add_note("phate", 2, "tighter neighbourhoods look cleaner")
    before = (session.data["selected"], session.chain(), session.iterations())
    again = TrainingSession.open("demo")
    assert (again.data["selected"], again.chain(), again.iterations()) == before
    assert json.loads((session.root / "session.json").read_text())["status"]["dpt"]["state"] == "stale"


def test_corrupt_session_file_recovers_from_backup(session, engine):
    walk(session, upto="pca")
    session.run("phate", select=True)
    session.run("phate", {"knn": 3}, select=True)
    (session.root / "session.json").write_text("{ truncated")
    again = TrainingSession.open("demo")
    assert "recovered" in again.recovered
    assert again.data["selected"]["phate"] == "phate#1"
    assert len(again.iterations("phate")) == 2


def test_old_schema_is_refused(session):
    data = json.loads((session.root / "session.json").read_text())
    data["schema_version"] = 1
    (session.root / "session.json").write_text(json.dumps(data))
    with pytest.raises(TrainingError, match="schema version 1"):
        TrainingSession.open("demo")


def test_records_carry_expert_and_no_secrets(tmp_path, monkeypatch, engine):
    monkeypatch.setenv("SECRET_TOKEN", "sk-do-not-log-this")
    s = make(tmp_path, monkeypatch, expert="jo")
    s.run("filter_genes", note="first look")
    text = (s.root / "trace.jsonl").read_text() + json.dumps(s.iterations())
    assert "sk-do-not-log-this" not in text and "SECRET_TOKEN" not in text
    assert {r["expert"] for r in s.trace()} == {"jo"}
    assert s.iteration("filter_genes", 1)["expert"] == "jo"


# ── failures and interruptions ──────────────────────────────────────────────────────────
def test_failed_attempt_is_recorded_and_previous_success_kept(session, engine):
    walk(session, upto="pca")
    session.run("phate", select=True)
    engine.fail = {"phate": RuntimeError("PHATE blew up")}
    rec = session.run("phate", {"knn": 8}, select=True)
    assert rec["status"] == "error" and "PHATE blew up" in rec["error"]
    assert session.data["selected"]["phate"] == "phate#1"
    with pytest.raises(TrainingError, match="status is error"):
        session.select("phate", 2)


def test_partially_written_output_is_never_accepted(session, engine):
    walk(session, upto="transform")
    engine.drop_output = "pca"
    rec = session.run("pca", select=True)
    assert rec["status"] == "error" and "was not persisted" in rec["error"]
    assert "pca" not in session.data["selected"]


def test_keyboard_interrupt_is_recorded_and_reraised(session, engine):
    walk(session, upto="filter_genes")
    engine.fail = {"normalize": KeyboardInterrupt()}
    with pytest.raises(KeyboardInterrupt):
        session.run("normalize")
    assert session.iteration("normalize", 1)["status"] == "interrupted"
    assert not (session.root / ".lock").exists()


def test_killed_process_leaves_an_interrupted_record(session, engine):
    walk(session, upto="filter_genes")
    it = session.iteration("filter_genes", 1)
    dead = session.root / "iterations/filter_genes/002"
    dead.mkdir()
    sess.write_json_atomic(dead / "iteration.json",
                           {**it, "iteration_id": "filter_genes#2", "number": 2,
                            "status": "running"})
    (session.root / ".lock").write_text("999999")
    session.run("normalize")
    assert session.iteration("filter_genes", 2)["status"] == "interrupted"
    assert session.run("filter_genes")["iteration_id"] == "filter_genes#3"


def test_session_in_use_is_refused(session, engine):
    (session.root / ".lock").write_text(str(os.getpid()))
    with pytest.raises(TrainingError, match="in use"):
        session.run("filter_genes")
    (session.root / ".lock").unlink()


def test_unloadable_input_is_a_refusal_not_a_traceback(session, engine):
    import numpy as np

    walk(session, upto="pca")
    emb = session.iteration("pca", 1)["outputs"]["state"]["emb"]
    np.save(session.root / emb["path"], np.array([object()], dtype=object))
    sess._HASHES.clear()
    data = json.loads((session.root / "iterations/pca/001/iteration.json").read_text())
    data["outputs"]["state"]["emb"]["sha256"] = None  # skip the hash check, reach the loader
    sess.write_json_atomic(session.root / "iterations/pca/001/iteration.json", data)
    with pytest.raises(TrainingError, match="cannot be loaded"):
        session.run("phate")
