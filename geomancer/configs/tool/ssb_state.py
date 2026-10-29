"""ssb_state adapter: the learner's context vectors for one sample, as scalars.

manyruns runs this file in the tool's own interpreter with the path of
request.json (manyruns/pipeline/external.py at v0.1.0). input.npy must be the
raw points: the adapter checks it against the sample's points (``params.sample``
and ``params.points_sha256``, found in ``$GEOMANCER_GRID_INPUTS`` as in
ssb_graph.py) and declines anything else, so a state is never measured on an
embedding by mistake.

Each vector in geomancer.learner.state.STATES becomes ``<name>_<i>`` scalars
plus ``<name>_dim``. The states read the input only and never the truth.
geomancer is imported as in ssb_graph.py (``$GEOMANCER_SRC`` first).
"""

import json
import os
import sys
from pathlib import Path


def _write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2))


def run(request):
    if src := os.environ.get("GEOMANCER_SRC"):
        sys.path.insert(0, src)
    import numpy as np

    from geomancer.learner import provenance, state

    params = request["params"]
    points = provenance.grid_points(params["sample"], params["points_sha256"])
    X = np.load(request["input"], allow_pickle=False)
    if X.shape != points.shape or not np.array_equal(np.asarray(X, float), np.asarray(points, float)):
        raise ValueError("the input is not the raw points; ssb_state measures the raw points only")
    scalars = {}
    for name, fn in state.STATES.items():
        vector = np.asarray(fn(points), dtype=float)
        scalars[f"{name}_dim"] = int(vector.size)
        scalars.update({f"{name}_{i}": float(v) for i, v in enumerate(vector)})

    import anndata as ad

    ad.AnnData(np.zeros((0, 0), dtype=np.float32)).write_h5ad(request["output"])
    return {
        **scalars,
        "sample": params["sample"],
        "points_sha256": params["points_sha256"],
        "input_sha256": provenance.sha256_file(request["input"]),
        "state_sha256": provenance.sha256_file(state.__file__),
        **provenance.tool_scalars(__file__),
    }


def main(path):
    request = json.loads(Path(path).read_text())
    try:
        scalars = run(request)
    except Exception as exc:
        _write(request["result"], {"ok": False, "reason": f"{type(exc).__name__}: {exc}"})
        return
    _write(request["result"], {"ok": True, "scalars": scalars})


if __name__ == "__main__":
    main(sys.argv[1])
