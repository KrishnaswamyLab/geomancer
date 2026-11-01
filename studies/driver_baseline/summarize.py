"""Print the README table from the saved driver outputs.

"Main knob" is the knob an answer changed by the largest factor; it scores
when that knob is the gold knob and moved the gold way. Usage, from this
folder: python summarize.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

OUTPUTS = [
    "out-baselines.jsonl",
    "out-qwen3_1.7b.jsonl",
    "out-qwen3_8b.jsonl",
    "out-qwen3_8b-think.jsonl",
]


def main_knob_right(case, answer):
    try:
        ans = json.loads(answer)
    except json.JSONDecodeError:
        return False
    ch = ans.get("changes") if isinstance(ans, dict) else None
    cur, gold = case["current"], case["gold"]
    ch = {
        k: v
        for k, v in (ch or {}).items()
        if k in cur and isinstance(v, (int, float)) and v > 0 and v != cur[k]
    }
    if not ch:
        return False
    k = max(ch, key=lambda k: abs(math.log(ch[k] / cur[k])))
    return k == gold["knob"] and (ch[k] > cur[k]) == (gold["direction"] == "up")


def main():
    here = Path(__file__).parent
    with open(here / "cases.jsonl") as f:
        cases = {c["id"]: c for c in map(json.loads, f)}
    table = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for name in OUTPUTS:
        with open(here / "results" / name) as f:
            for r in map(json.loads, f):
                c, s = cases[r["id"]], r["score"]
                driver = r["driver"] + (", thinking" if r.get("think") else "")
                if c["gold"].get("decline"):
                    cols = {"declined": s["declined_right"]}
                elif c["specificity"] == "explicit":
                    cols = {"exact value": s["exact_value"]}
                else:
                    cols = {
                        "main knob": main_knob_right(c, r["answer"]),
                        "strict": s["strict"],
                    }
                for col, ok in cols.items():
                    table[driver][col][0] += bool(ok)
                    table[driver][col][1] += 1
    cols = ["main knob", "strict", "exact value", "declined"]
    print("| Driver | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for driver, row in table.items():
        cells = [
            f"{row[c][0]}/{row[c][1]} ({100 * row[c][0] / row[c][1]:.0f}%)"
            for c in cols
        ]
        print(f"| {driver} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
