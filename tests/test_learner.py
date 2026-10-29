import json

import numpy as np
import pytest

pytest.importorskip("kmapper")
pytest.importorskip("scshapebench")

from geomancer.learner import actions, rewards, state, table


def _circle(n=200, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    points = np.c_[np.cos(theta), np.sin(theta)] + rng.normal(0, 0.03, (n, 2))
    # True graph: a square loop, positions on the circle.
    corners = {str(i): [np.cos(i * np.pi / 2), np.sin(i * np.pi / 2)] for i in range(4)}
    graph = {"nodes": [0, 1, 2, 3], "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
             "node_positions": corners}
    return {"id": 0, "points": points, "graph": graph, "meta": {}}


def test_action_names_cover_the_grid():
    names = actions.ACTIONS
    assert "raw+paga@default" in names and "phate+mapper@coarse" in names
    assert len(names) == len(actions.EMBEDDINGS) * len(actions.GRAPHS) * len(actions.SCALES)


def test_every_graph_method_returns_a_payload_in_point_space():
    sample = _circle()
    for graph in actions.GRAPHS:
        payload = actions.run(f"raw+{graph}@default", sample["points"])
        assert payload["nodes"], graph
        for position in payload["node_positions"].values():
            assert len(position) == sample["points"].shape[1]


def test_truth_scores_one_against_itself():
    sample = _circle()

    score = rewards.synthetic_graph(sample["graph"], sample)

    assert score["topology_score"] == pytest.approx(1.0)
    assert score["graph_edit_distance"] == 0


def test_state_vectors_have_fixed_length():
    points = _circle()["points"]
    for name, fn in state.STATES.items():
        vector = fn(points)
        assert vector.ndim == 1 and np.isfinite(vector).all(), name
    assert state.STATES["none"](points).size == 0


def test_failed_action_is_a_zero_reward_row_with_its_error(monkeypatch):
    def broken(points):
        raise RuntimeError("boom")

    rows = table.score_sample(_circle(), {"broken": broken}, rewards.synthetic_graph)

    assert rows == [{"sample": 0, "action": "broken", "reward": 0.0, "status": "error",
                     "error": "RuntimeError: boom"}]


def test_build_table_writes_rewards_and_states(tmp_path):
    names = ["raw+paga@default", "raw+mapper@default"]
    chosen = {n: actions.ACTIONS[n] for n in names}

    table.build_table([_circle()], chosen, rewards.synthetic_graph, tmp_path, workers=1)

    rows = [json.loads(line) for line in (tmp_path / "rewards.jsonl").read_text().splitlines()]
    assert sorted(r["action"] for r in rows) == sorted(names)
    assert all(0.0 <= r["reward"] <= 1.0 for r in rows)
    states = json.loads((tmp_path / "states.jsonl").read_text().splitlines()[0])
    assert set(states) == {"sample", *state.STATES}
    assert json.loads((tmp_path / "config.json").read_text())["actions"] == names


def test_overrides_parse_like_nanogpt():
    config = {"limit": None, "workers": 4, "out": "x"}

    assert table.override(config, ["limit=10", "out=y"]) == {"limit": 10, "workers": 4, "out": "y"}
    with pytest.raises(KeyError):
        table.override(config, ["nope=1"])


def test_paga_with_no_edge_between_clusters_is_nodes_without_edges():
    # sc.tl.paga raises when no kNN edge joins two Leiden clusters: one tight
    # blob, or blobs far apart. The action must return the clusters as nodes.
    rng = np.random.default_rng(0)
    one_blob = rng.normal(0, 0.01, (60, 2))
    three_blobs = np.concatenate([rng.normal(c, 0.01, (60, 2)) for c in (0, 100, 200)])

    single = actions._paga(one_blob, one_blob, {"knn": 30, "resolution": 0.01})
    separate = actions._paga(three_blobs, three_blobs, {"knn": 30, "resolution": 0.5})

    assert single["nodes"] == [0] and single["edges"] == []
    assert len(separate["nodes"]) >= 3 and separate["edges"] == []
