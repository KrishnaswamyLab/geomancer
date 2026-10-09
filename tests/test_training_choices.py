"""What each reviewable value in `geomancer/training/choices.py` does.

When the lab answers a review question, edit `choices.py` and these tests say whether the
new value is coherent with the rest of the workflow.
"""
import pytest

from geomancer.training import choices, timebins, workflow as wf


def test_preprocessing_values_reach_the_workflow():
    steps = {s["name"]: s for s in wf.steps(wf.PSEUDOTIME)}
    assert steps["filter_genes"]["params"]["min_cells"] == choices.GENE_FILTER_MIN_CELLS
    assert steps["normalize"]["params"]["target_sum"] == choices.NORMALIZE_TARGET_SUM
    assert steps["transform"]["params"]["method"] == choices.TRANSFORM_METHOD
    assert steps["pca"]["params"]["n_components"] == choices.PCA_COMPONENTS
    assert steps["phate"]["params"]["n_components"] == choices.PHATE_COMPONENTS
    assert steps["discretize_time"]["params"]["n_timepoints"] == choices.PSEUDOTIME_BINS
    for name in ("filter_genes", "normalize", "transform", "pca", "phate", "discretize_time"):
        wf.validate(steps[name], {}, 1000)  # every declared value is itself legal


def test_not_applicable_policy_is_one_of_two_words():
    assert choices.NOT_APPLICABLE in ("pass_through", "stop")


def test_k_per_timepoint_default_becomes_a_declared_param(monkeypatch):
    real = {s["name"]: s for s in wf.steps(wf.REAL)}
    if choices.K_PER_TIMEPOINT is None:
        assert real["subcluster_time"]["params"] == {}
    monkeypatch.setattr(choices, "K_PER_TIMEPOINT", 4)
    real = {s["name"]: s for s in wf.steps(wf.REAL)}
    assert real["subcluster_time"]["params"] == {"k_per_timepoint": 4}
    wf.validate(real["subcluster_time"], {}, 100)


def test_subcluster_order_names_an_implemented_ordering():
    assert choices.SUBCLUSTER_ORDER in timebins.ORDERINGS


@pytest.mark.parametrize("rule", choices.RULES, ids=lambda r: r.id)
def test_every_rule_moves_an_offered_parameter(rule):
    assert rule.step in wf.PARAMS, rule.step
    assert rule.param in wf.PARAMS[rule.step], f"{rule.step}.{rule.param}"
    assert rule.op in ("mul", "add") and rule.amount != 0
    assert rule.words and all(w == w.lower() for w in rule.words)
    assert rule.reason


def test_rule_ids_are_unique_and_versioned():
    ids = [r.id for r in choices.RULES]
    assert len(ids) == len(set(ids))
    assert choices.RULES_VERSION
