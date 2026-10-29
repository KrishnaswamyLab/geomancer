"""The ssb_graph and ssb_state lab tools and their locks.

The adapters are run here the way manyruns runs them: a separate interpreter,
given only request.json (pipeline/external.py at v0.1.0). The interpreter is this
venv's, which has every package the tool lock pins.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("kmapper")
pytest.importorskip("scanpy")

from geomancer.learner import actions, provenance, state

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "geomancer" / "configs" / "tool"


def _circle(n=150, seed=0, dim=3):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    points = np.c_[np.cos(theta), np.sin(theta), np.zeros(n)][:, :dim]
    return points + rng.normal(0, 0.03, points.shape)


def _request(tmp_path, tool, coords, points, **params):
    work = tmp_path / f"tool-{tool}"
    work.mkdir(parents=True)
    points_path = tmp_path / "sample_00007.npy"
    np.save(points_path, points)
    np.save(work / "input.npy", coords)
    merged = {"points": str(points_path), "points_sha256": provenance.sha256_file(points_path),
              **params}
    request = {"tool": tool, "version": "0.1.0", "params": merged, "seed": 0,
               "input": str(work / "input.npy"), "output": str(work / "output.h5ad"),
               "result": str(work / "result.json"), "workdir": str(work)}
    (work / "request.json").write_text(json.dumps(request))
    return work


def _run(tool, work, env=None):
    import os

    proc = subprocess.run([sys.executable, str(TOOLS / f"{tool}.py"), str(work / "request.json")],
                          capture_output=True, text=True, env={**os.environ, **(env or {})})
    assert proc.returncode == 0, proc.stderr
    return json.loads((work / "result.json").read_text())


def test_manifests_pass_the_manyruns_tool_check():
    from manyruns import toolchain

    for name in ("ssb_graph", "ssb_state"):
        cfg = yaml.safe_load((TOOLS / f"{name}.yaml").read_text())
        assert toolchain.check_tool(cfg, name) == [], name
        assert cfg["input"] == {"kind": "state_matrix"}
        # Amendment 1: the truth never enters a run.
        assert not any("truth" in key for key in cfg.get("params") or {}), name


@pytest.mark.parametrize("extractor", sorted(actions.GRAPHS))
def test_raw_arm_graph_is_the_actions_graph_and_carries_its_hash(tmp_path, extractor):
    points = _circle()
    work = _request(tmp_path, "ssb_graph", points, points, extractor=extractor, scale="default")

    result = _run("ssb_graph", work, {"GEOMANCER_CALLER": "geomancer.test",
                                      "GEOMANCER_CALLER_VERSION": "9.9"})

    assert result["ok"] is True, result
    graph = json.loads((work / "graph.json").read_text())
    assert graph == json.loads(json.dumps(actions.graph(extractor, "default", points, points)))
    s = result["scalars"]
    assert s["graph_sha256"] == hashlib.sha256((work / "graph.json").read_bytes()).hexdigest()
    assert s["n_nodes"] == len(graph["nodes"]) and s["n_edges"] == len(graph["edges"])
    assert s["n_cycles"] == s["n_edges"] - s["n_nodes"] + s["n_components"]
    assert s["sample"] == "sample_00007"
    assert s["points_sha256"] == provenance.sha256_file(tmp_path / "sample_00007.npy")
    assert s["input_sha256"] == provenance.sha256_file(work / "input.npy")
    assert s["adapter_sha256"] == provenance.sha256_file(TOOLS / "ssb_graph.py")
    assert s["actions_sha256"] == provenance.sha256_file(Path(actions.__file__))
    assert s["caller"] == "geomancer.test" and s["caller_version"] == "9.9"
    assert s["geomancer_revision"] == provenance.geomancer_revision()["commit"]
    assert s["version_kmapper"] and s["version_scanpy"] and s["version_numpy"]
    # Structural scalars only: no reward, no score (amendment 1).
    assert not any(k in s for k in ("reward", "topology_score", "relaxed_topology_score"))
    assert (work / "output.h5ad").is_file()


def test_embedded_arm_builds_on_the_embedding_and_places_nodes_in_point_space(tmp_path):
    points = _circle(dim=3)
    coords = actions._pca(points)
    work = _request(tmp_path, "ssb_graph", coords, points, extractor="paga", scale="coarse")

    result = _run("ssb_graph", work)

    graph = json.loads((work / "graph.json").read_text())
    assert result["ok"] is True
    assert graph == json.loads(json.dumps(actions.graph("paga", "coarse", coords, points)))
    assert all(len(p) == 3 for p in graph["node_positions"].values())


def test_points_that_do_not_match_their_hash_are_declined(tmp_path):
    points = _circle()
    work = _request(tmp_path, "ssb_graph", points, points, extractor="paga", scale="default")
    request = json.loads((work / "request.json").read_text())
    request["params"]["points_sha256"] = "0" * 64
    (work / "request.json").write_text(json.dumps(request))

    result = _run("ssb_graph", work)

    assert result["ok"] is False and "sha256" in result["reason"]
    assert not (work / "graph.json").exists()


def test_an_input_with_other_rows_than_the_points_is_declined(tmp_path):
    points = _circle()
    work = _request(tmp_path, "ssb_graph", points[:-1], points, extractor="paga", scale="default")

    result = _run("ssb_graph", work)

    assert result["ok"] is False and "rows" in result["reason"]


def test_state_tool_returns_every_state_vector_as_scalars(tmp_path):
    points = _circle()
    work = _request(tmp_path, "ssb_state", points, points)

    result = _run("ssb_state", work)

    assert result["ok"] is True, result
    s = result["scalars"]
    for name, fn in state.STATES.items():
        vector = fn(points)
        assert [s[f"{name}_{i}"] for i in range(vector.size)] == pytest.approx(vector.tolist())
        assert s[f"{name}_dim"] == vector.size
    assert s["sample"] == "sample_00007"
    assert s["state_sha256"] == provenance.sha256_file(Path(state.__file__))


def test_state_tool_declines_an_input_that_is_not_the_raw_points(tmp_path):
    points = _circle()
    work = _request(tmp_path, "ssb_state", points + 1.0, points)

    result = _run("ssb_state", work)

    assert result["ok"] is False and "raw points" in result["reason"]


def test_locks_are_fully_hashed_for_both_platforms():
    for name in ("ssb_graph", "ssb_state"):
        version = yaml.safe_load((TOOLS / f"{name}.yaml").read_text())["version"]
        for tag in ("macos-arm64", "linux-x86_64"):
            lock = TOOLS / "locks" / f"{name}-{version}-{tag}.lock"
            text = lock.read_text()
            pins = [line for line in text.splitlines() if line and not line.startswith((" ", "#"))]
            assert pins, lock
            assert all("==" in pin and pin.rstrip().endswith("\\") for pin in pins), lock
            assert "geomancer" not in [pin.split("==")[0] for pin in pins]


def test_lock_inputs_are_the_manifest_requires_without_the_tool_itself():
    from geomancer.learner import lock

    cfg = yaml.safe_load((TOOLS / "ssb_graph.yaml").read_text())

    lines = lock.requirements(cfg)

    assert lines and not any(line.startswith("ssb_graph") for line in lines)
    assert all(line in [str(r).split("#")[0].strip() for r in cfg["requires"]] for line in lines)
