"""The mocked compute boundary shared by the training tests.

`FakeEngine` replaces `geomancer.training.execute.execute_step`, the single function that
calls manyruns. It writes small deterministic arrays through manyruns' own
`artifacts.save_array` and records what each step was handed. It is bookkeeping, not science.
"""
import hashlib
import json

import numpy as np

N = 30


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.jobs = []
        self.fail = {}             # step name -> exception to raise
        self.drop_output = None    # step whose carried file is "not written"
        self.not_applicable = set()  # prep steps that skip as not applicable

    def __call__(self, job):
        from manyruns import artifacts

        step, params, inputs = job["step"]["name"], job["step"]["params"], job["inputs"]
        self.jobs.append(job)
        self.calls.append({"step": step, "params": dict(params),
                           "inputs": {k: digest(v) for k, v in inputs.items()},
                           "replay": [r["step"]["name"] for r in job.get("replay") or []]})
        if step in self.fail:
            raise self.fail[step]
        seed = int(hashlib.sha256(json.dumps(
            [step, params, sorted(self.calls[-1]["inputs"].items()), job["seed"]],
            sort_keys=True, default=str).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        if job["step"]["group"] == "prep":
            if step in self.not_applicable:
                return {"outcome": "skipped", "detail": f"{step}: not counts"}
            return {"outcome": "ok", "params": dict(params), "artifacts": {}, "carried": {},
                    "recomputed": {"X": {"shape": [N, 5], "sha256": f"{seed:x}"}},
                    "plots": [], "g": {}, "run_id": "fake"}
        if step == "pca":
            produced = {"emb": rng.normal(size=(N, min(params.get("n_components", 2), 5)))}
        elif step == "phate":
            assert "emb" in inputs
            produced = {"emb": rng.normal(size=(N, params.get("n_components", 2)))}
        elif step == "dpt":
            assert "emb" in inputs
            produced = {"pseudotime": rng.random(N)}
        elif step == "discretize_time":
            assert "pseudotime" in inputs
            produced = {"time_labels": rng.integers(0, params["n_timepoints"], N).astype(str)}
        elif step == "subcluster_time":
            assert {"emb", "pseudotime"} <= set(inputs)
            produced = {"time_labels": rng.random(N).astype(str)}
        else:
            assert {"emb", "time_labels"} <= set(inputs)
            produced = {"emb": rng.normal(size=(N, inputs["emb"].shape[1]))}
        paths = {}
        for slot, arr in produced.items():
            if self.drop_output == step:
                paths[slot] = str(job["out_dir"] / "state" / "never-written.npy")
            else:
                paths[slot] = artifacts.save_array(job["out_dir"], "fake", 0, step, slot, arr)
        return {"outcome": "ok", "detail": None, "params": dict(params), "artifacts": paths,
                "carried": dict(paths), "plots": [], "g": {f"{step}.fake": 1.0},
                "run_id": "fake", "caveats": []}
