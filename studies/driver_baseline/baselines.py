"""Non-LLM baselines on the synthetic cases, scored like drive.py.

- no_change: always answers {"changes": {}} (never allowed, never declines).
- always_decline: always declines.
- majority: always moves the most common gold knob in its most common direction.
- portal_rules: the portal's own keyword rules (rules_version "1", read from
  btd2026.github.io/scShapeBench); declines when no rule matches.
Usage: python baselines.py cases.jsonl
"""

import json
import sys
from collections import Counter, defaultdict

from drive import score

RULES = [  # (id, knob, op, amount, words), copied from the portal bundle
    (
        "phate-oversmoothed",
        "t",
        "mul",
        0.67,
        [
            "blurry",
            "blurred",
            "over-smoothed",
            "oversmoothed",
            "too smooth",
            "washed out",
            "branches merged",
            "merged branches",
            "merged",
            "merging",
            "smeared",
        ],
    ),
    (
        "phate-noisy",
        "t",
        "mul",
        1.5,
        ["noisy", "speckled", "grainy", "fuzzy edges", "jagged", "scattered"],
    ),
    (
        "phate-fragmented",
        "knn",
        "add",
        2,
        ["fragmented", "islands", "broken apart", "disconnected", "pieces", "split up"],
    ),
    (
        "phate-one-blob",
        "decay",
        "mul",
        0.75,
        ["one blob", "blob", "too compressed", "collapsed", "squashed", "no structure"],
    ),
]


def portal_rules(case):
    text, cur = case["text"].lower(), case["current"]
    changes = {}
    for _, knob, op, amount, words in RULES:
        if any(w in text for w in words):
            v = cur[knob] * amount if op == "mul" else cur[knob] + amount
            changes[knob] = max(1, round(v))
    return {"changes": changes} if changes else {"decline": "no rule matched"}


def main(path):
    with open(path) as f:
        cases = [json.loads(line) for line in f]
    act = [c["gold"] for c in cases if not c["gold"].get("decline")]
    knob, direction = Counter((g["knob"], g["direction"]) for g in act).most_common(1)[
        0
    ][0]

    def majority(c):
        return {
            "changes": {
                knob: c["current"][knob] * 2
                if direction == "up"
                else c["current"][knob] // 2
            }
        }

    policies = {
        "no_change": lambda c: {"changes": {}},
        "always_decline": lambda c: {"decline": "x"},
        f"majority ({knob} {direction})": majority,
        "portal_rules": portal_rules,
    }
    rows = []
    for name, pol in policies.items():
        agg = defaultdict(lambda: defaultdict(list))
        for c in cases:
            ans = pol(c)
            s = score(c, json.dumps(ans))
            g = "decline" if c["gold"].get("decline") else c["specificity"]
            for k, v in s.items():
                agg[g][k].append(v)
            rows.append(
                {"id": c["id"], "driver": name, "answer": json.dumps(ans), "score": s}
            )
        print(f"== {name}")
        for g, d in agg.items():
            print("  ", g, {k: f"{sum(v)}/{len(v)}" for k, v in d.items()})
    with open("results/out-baselines.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main(sys.argv[1])
