"""Rewards: score a predicted graph payload against one sample's truth.

Each function returns a dict whose ``reward`` is in [0, 1], plus the scores it
was taken from, so the table keeps them for later analysis.
"""


def synthetic_graph(pred, sample):
    """scShapeBench's full-graph-recovery evaluator; reward = its primary metric.

    The primary metric is the relaxed topology score. Persistence similarity and
    graph edit distance are kept alongside it.
    """
    # The evaluator filters sys.path entries for other Python versions and sets
    # PYTHONNOUSERSITE when it is imported; import it only when scoring.
    from scshapebench.synthetic import evaluation as ev

    truth_graph, truth_positions, truth_reduced = ev.build_graph_from_payload(
        sample["graph"], positions_required_for_persistence=True)
    pred_graph, pred_positions, pred_reduced = ev.build_graph_from_payload(
        pred, positions_required_for_persistence=True)
    result = ev.evaluate_full_graph_recovery(
        truth_graph=truth_graph,
        pred_graph=pred_graph,
        truth_positions=truth_positions,
        pred_positions=pred_positions,
        observed_points=sample["points"],
        truth_already_reduced_multigraph=truth_reduced,
        pred_already_reduced_multigraph=pred_reduced,
    )
    headline = result["headline_scores"]
    return {
        "reward": headline["topology_score"],
        **headline,
        "graph_edit_distance": result["graph_edit_distance"],
        "graph_edit_distance_exact": result["graph_edit_distance_exact"],
    }


REWARDS = {"synthetic_graph": synthetic_graph}
