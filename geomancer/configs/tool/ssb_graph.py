"""ssb_graph adapter: extract one graph from an embedding, or from the raw points.

manyruns runs this file in the tool's own interpreter with one argument, the
path of request.json (manyruns/pipeline/external.py at v0.1.0). It reads
input.npy (the embedding the run holds, or the raw points X for the raw arm) and
``params.points`` (the sample's original points, checked against
``params.points_sha256``). It calls geomancer.learner.actions.graph, which builds
the graph on the input and places each node at the mean of its original points.

It writes graph.json, a minimal output.h5ad (manyruns skips the step without
one) and result.json with structural scalars and provenance. It never reads
the truth and never scores: scoring is geomancer/learner/score.py, after the run.

geomancer is imported from ``$GEOMANCER_SRC`` (the checkout root) when that is
set, else from the interpreter's own packages. The tool lock pins the
third-party packages only, so the geomancer code that ran is named by
``geomancer_revision`` and ``actions_sha256`` in the result.
"""

import json
import os
import sys
from pathlib import Path


def _write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2))


def structure(payload):
    """Counts of nodes, edges, components, independent cycles, leaves and branch points."""
    nodes = list(payload["nodes"])
    parent = {node: node for node in nodes}
    degree = dict.fromkeys(nodes, 0)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for u, v in payload["edges"]:
        degree[u] += 1
        degree[v] += 1
        parent[find(u)] = find(v)
    components = len({find(node) for node in nodes})
    return {
        "n_nodes": len(nodes),
        "n_edges": len(payload["edges"]),
        "n_components": components,
        # The cycle rank of a simple graph: edges - nodes + components.
        "n_cycles": len(payload["edges"]) - len(nodes) + components,
        "n_leaves": sum(d == 1 for d in degree.values()),
        "n_branch_points": sum(d >= 3 for d in degree.values()),
        "n_isolated": sum(d == 0 for d in degree.values()),
    }


def run(request):
    if src := os.environ.get("GEOMANCER_SRC"):
        sys.path.insert(0, src)
    import numpy as np

    from geomancer.learner import actions, provenance

    params = request["params"]
    points = provenance.verified_points(params["points"], params["points_sha256"])
    coords = np.load(request["input"], allow_pickle=False)
    if coords.shape[0] != points.shape[0]:
        raise ValueError(f"the input has {coords.shape[0]} rows and the points have {points.shape[0]} rows")
    payload = actions.graph(params["extractor"], params["scale"], coords, points)
    graph_path = Path(request["workdir"]) / "graph.json"
    graph_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    import anndata as ad

    ad.AnnData(np.zeros((0, 0), dtype=np.float32)).write_h5ad(request["output"])
    return {
        **structure(payload),
        "sample": Path(params["points"]).stem,
        "points_sha256": params["points_sha256"],
        "input_sha256": provenance.sha256_file(request["input"]),
        "graph_sha256": provenance.sha256_file(graph_path),
        "actions_sha256": provenance.sha256_file(actions.__file__),
        **provenance.tool_scalars(__file__),
    }


def main(path):
    request = json.loads(Path(path).read_text())
    try:
        scalars = run(request)
    except Exception as exc:
        # Declined, with the reason; manyruns records the step as skipped and keeps the row.
        _write(request["result"], {"ok": False, "reason": f"{type(exc).__name__}: {exc}"})
        return
    _write(request["result"], {"ok": True, "scalars": scalars})


if __name__ == "__main__":
    main(sys.argv[1])
