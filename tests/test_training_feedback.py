"""Plain-language feedback → proposals (pure), and the loop and verbs that record it.

The loop and verb tests use the mocked compute boundary (`FakeEngine`); the model tier is
always a stub here. Nothing in this file is scientific evidence.
"""
import json

import pytest

from geomancer.training import choices, execute, feedback as fb, loop, session as sess
from geomancer.training import workflow as wf
from training_fakes import FakeEngine

PHATE = {"n_components": 3, "knn": 5, "decay": 40, "t": 15, "gamma": 1, "n_pca": 100,
         "n_landmark": 2000}


def ctx(text, step="phate", params=None, diagnostics=None, n_cells=300):
    return fb.Context(step=step, text=text, params=dict(params or PHATE),
                      diagnostics=diagnostics or {}, n_cells=n_cells)


def rules(c):
    return fb.propose(c, [fb.RuleProposer()])


# ── rules (pure) ────────────────────────────────────────────────────────────────────────
def test_blurry_lowers_t_and_noisy_raises_it():
    p, = rules(ctx("It looks too blurry")).proposals
    assert (p.param, p.old, p.new, p.proposer) == ("t", 15, 10, "rules")
    assert p.source == {"rule": "phate-oversmoothed", "rules_version": choices.RULES_VERSION}
    assert p.matched == ["blurry"]
    assert rules(ctx("very noisy and grainy")).proposals[0].new == 22


def test_negated_words_do_not_match():
    assert rules(ctx("not blurry at all, isn't noisy either")).unmatched


def test_rule_needing_a_diagnostic_is_reported_when_contradicted():
    connected = rules(ctx("looks fragmented", diagnostics={"phate.knn_disconnected": False}))
    assert connected.unmatched
    assert connected.contradicted[0]["rule"] == "phate-fragmented"
    split = rules(ctx("looks fragmented", diagnostics={"phate.knn_disconnected": True}))
    assert split.proposals[0].param == "knn" and split.proposals[0].new == 7


def test_ambiguous_feedback_gives_ranked_distinct_proposals():
    result = rules(ctx("blurry and smeared, also one blob"))
    assert [p.param for p in result.proposals] == ["t", "decay"]  # two words beat one
    assert len(result.proposals) <= choices.MAX_PROPOSALS


def test_steps_are_clipped_to_legal_ranges():
    assert fb.step_value("phate", "knn", 298, "add", 2, n_cells=300) == 299
    assert fb.step_value("phate", "knn", 299, "add", 2, n_cells=300) is None   # at the bound
    assert fb.step_value("phate", "t", "auto", "mul", 0.67, None) == 13         # from 20
    assert fb.step_value("phate", "t", 1, "mul", 0.67, None) is None            # min 1
    assert fb.step_value("discretize_time", "n_timepoints", 2, "add", -1, 300) is None
    assert fb.step_value("phate", "decay", 40, "mul", 0.75, None) == 30.0


def test_unknown_words_are_unmatched_and_rules_are_step_specific():
    assert rules(ctx("hmm, interesting")).unmatched
    assert rules(ctx("too blurry", step="dpt", params={"n_neighbors": 15, "n_dcs": 2})).unmatched
    coarse = rules(ctx("bins are too coarse", step="discretize_time",
                       params={"n_timepoints": 5}))
    assert coarse.proposals[0].new == 6


def test_model_tier_is_consulted_only_when_rules_find_nothing():
    calls = []

    def stub(**kw):
        calls.append(kw)
        return {"param": "knn", "direction": "up", "why": "looks sparse"}

    proposers = [fb.RuleProposer(), fb.ModelProposer(call=stub)]
    assert fb.propose(ctx("too blurry"), proposers).proposals[0].proposer == "rules"
    assert calls == []
    result = fb.propose(ctx("the clusters feel lonely"), proposers)
    p, = result.proposals
    assert (p.proposer, p.param, p.new) == ("model", "knn", 7)  # the rule table's step
    assert len(p.source["prompt_sha256"]) == 64
    sent = calls[0]["prompt"]
    assert "lonely" in sent and "/" not in sent.split("Diagnostics")[0]  # no paths sent
    assert calls[0]["schema"]["properties"]["param"]["enum"] == sorted(PHATE)


def test_model_tier_failure_degrades_to_unmatched():
    def broken(**kw):
        raise RuntimeError("no key")

    result = fb.propose(ctx("feels lonely"), [fb.RuleProposer(), fb.ModelProposer(call=broken)])
    assert result.unmatched and "model unavailable" in result.notes[0]
    invented = fb.propose(ctx("feels lonely"), [fb.ModelProposer(
        call=lambda **kw: {"param": "made_up", "direction": "up"})])
    assert invented.unmatched


# ── recorded through a session (mocked compute) ─────────────────────────────────────────
@pytest.fixture
def engine(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(execute, "execute_step", fake)
    return fake


@pytest.fixture
def at_phate(tmp_path, monkeypatch, engine):
    monkeypatch.chdir(tmp_path)
    project = {"project": "fb", "engine": "manylatents", "data_folder": "synthetic:time-course",
               "dataset_name": "synthetic_timecourse", "data_kwargs": {}, "modality": "synthetic"}
    s = sess.TrainingSession.create("fb", project, wf.PSEUDOTIME, source_shape=[30, 5],
                                    expert="jo")
    for step in s.step_names()[:4]:
        s.run(step, select=True)
    s.run("phate")
    return s


def events(s, kind):
    return [r for r in s.trace() if r["event"] == kind]


def test_feedback_accept_runs_the_proposal_and_links_everything(at_phate, capsys):
    s = at_phate
    assert loop.feedback_once(s, "phate", 1, "too blurry", accept=1, show_plots=False) == 0
    rec = s.iteration("phate", 2)
    assert rec["params"]["changed"] == {"t": [15, 10]} and rec["reason"] == "feedback-accepted"
    said, = events(s, "feedback")
    decided, = events(s, "feedback_decision")
    assert said["text"] == "too blurry" and said["iteration"] == "phate#1"
    assert said["proposals"][0]["new"] == 10 and said["expert"] == "jo"
    assert said["params"]["t"] == 15
    assert decided["feedback_id"] == said["feedback_id"] == rec["feedback_id"]
    assert decided["decision"] == "accepted" and decided["resulting_iteration"] == "phate#2"


def test_feedback_edit_reject_and_undecided(at_phate, capsys):
    s = at_phate
    loop.feedback_once(s, "phate", 1, "too blurry", edit=["t=12", "knn=8"], show_plots=False)
    assert s.iteration("phate", 2)["params"]["overrides"] == {"t": 12, "knn": 8}
    loop.feedback_once(s, "phate", 1, "too blurry", reject=True)
    loop.feedback_once(s, "phate", 1, "something odd")
    assert "no proposal for that" in capsys.readouterr().out
    assert [e["decision"] for e in events(s, "feedback_decision")] == ["edited", "rejected"]
    assert len(events(s, "feedback")) == 3 and len(s.iterations("phate")) == 2


def scripted(lines):
    it = iter(lines)

    def read(prompt):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    return read


def test_loop_feedback_edit_why_keep_and_done(at_phate):
    s = at_phate
    out = []
    code = loop.run_loop(s, "phate", read=scripted([
        "too blurry", "accept 1",          # phate#2 (t 15 → 10)
        "knn=10", "denser looked cleaner",  # phate#3, with a reason
        "compare 1",
        "keep",                             # keeps phate#3
        "done",
    ]), write=out.append, show_plots=False)
    assert code == 0
    assert s.data["selected"]["phate"] == "phate#3"
    why = [e for e in events(s, "feedback") if e["kind"] == "why"]
    assert why[0]["text"] == "denser looked cleaner" and why[0]["iteration"] == "phate#3"
    assert why[0]["overrides"] == {"knn": 10}
    kinds = [e["event"] for e in s.trace()]
    assert kinds.index("loop_started") < kinds.index("loop_ended")
    assert events(s, "loop_ended")[0]["kept"] == "phate#3"
    assert any("params:" in line for line in out)


def test_loop_done_without_keeping_asks(at_phate):
    s = at_phate
    loop.run_loop(s, "phate", read=scripted(["done", "none"]), write=lambda *_: None,
                  show_plots=False)
    assert events(s, "left_without_keeping")
    loop.run_loop(s, "phate", read=scripted(["done", "1"]), write=lambda *_: None,
                  show_plots=False)
    assert s.data["selected"]["phate"] == "phate#1"


def test_loop_reports_refusals_and_keeps_going(at_phate):
    out = []
    loop.run_loop(at_phate, "phate", read=scripted(["knn=zero", "keep 9", "done", "1"]),
                  write=out.append, show_plots=False)
    assert sum("refused" in line for line in out) == 2


def test_collect_mode_asks_why_after_decisions(at_phate):
    s = at_phate
    loop.run_loop(s, "phate", read=scripted(["too blurry", "reject", "branches still merge",
                                             "keep", "it is the clearest", "done"]),
                  write=lambda *_: None, collect=True, show_plots=False)
    why = {e["decision"]: e["text"] for e in events(s, "feedback") if e["kind"] == "why"}
    assert why == {"rejected": "branches still merge", "kept": "it is the clearest"}


def test_natural_variants_of_merged_match():
    assert rules(ctx("the branches look merged")).proposals[0].param == "t"
