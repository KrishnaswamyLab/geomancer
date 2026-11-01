"""Ask a driver model to answer each synthetic case with knob changes; score it.

The driver sees the portal's knobs (meaning, presets, allowed range, current
value) and the person's message. It answers JSON: {"changes": {knob: value}}
or {"decline": "<reason>"}. Scores: allowed (valid JSON, known knob, in
range), right knob, right direction, exact value (explicit cases), and
correct decline. Usage: python drive.py <driver model> <cases.jsonl> <out.jsonl>
"""

import json
import sys
import urllib.request
from collections import defaultdict

from make_prompts import KNOBS

MEANING = {
    "knn": "Neighbors (k): how many nearby cells each cell listens to. Higher is smoother and more connected.",
    "t": "Diffusion time (t): how far information spreads. Higher merges more; lower keeps fine branches.",
    "decay": "Decay: how quickly a cell stops listening to farther cells. Lower spreads the plot out.",
}

SYSTEM = """You tune a PHATE embedding for a biologist. You can change ONLY these knobs:
{knobs}
Answer with JSON only:
- {{"changes": {{"<knob>": <integer>, ...}}}} to change one or more knobs, or
- {{"decline": "<short reason>"}} if the request cannot be met by these knobs.
Any integer in the allowed range is valid; presets are suggestions."""


def card(current):
    return "\n".join(
        f"- {k} (current {current[k]}, allowed {KNOBS[k]['range'][0]}-{KNOBS[k]['range'][1]}, "
        f"presets {KNOBS[k]['presets']}): {MEANING[k]}"
        for k in KNOBS
    )


def ask(model, case, think):
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "format": "json",
            "think": think,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM.format(knobs=card(case["current"])),
                },
                {"role": "user", "content": case["text"]},
            ],
        }
    ).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat", body, {"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["message"][
        "content"
    ]


def score(case, raw):
    gold, cur = case["gold"], case["current"]
    try:
        ans = json.loads(raw)
    except json.JSONDecodeError:
        ans = {}
    if not isinstance(ans, dict):
        ans = {}
    if gold.get("decline"):
        return {
            "allowed": bool(ans),
            "declined_right": "decline" in ans and not ans.get("changes"),
        }
    ch = ans.get("changes") or {}
    ok = (
        isinstance(ch, dict)
        and ch
        and all(
            k in KNOBS
            and isinstance(v, int)
            and KNOBS[k]["range"][0] <= v <= KNOBS[k]["range"][1]
            for k, v in ch.items()
        )
    )
    # Every score is over all cases: an answer that is not allowed scores False.
    r = {
        "allowed": bool(ok),
        "declined_wrongly": "decline" in ans and not ch,
        "right_knob": False,
        "right_direction": False,
        "only_that_knob": False,
        "strict": False,
    }
    if "value" in gold:
        r["exact_value"] = False
    if not ok:
        return r
    k = gold["knob"]
    r["right_knob"] = k in ch
    r["right_direction"] = (
        k in ch
        and ((ch[k] > cur[k]) == (gold["direction"] == "up"))
        and ch[k] != cur[k]
    )
    r["only_that_knob"] = set(ch) == {k}
    # strict: the right knob, the right way, and nothing else moved.
    r["strict"] = r["right_direction"] and r["only_that_knob"]
    if "value" in gold:
        r["exact_value"] = ch.get(k) == gold["value"]
    return r


def main(model, cases_path, out, think=False):
    with open(cases_path) as f:
        cases = [json.loads(line) for line in f]
    rows, agg = [], defaultdict(lambda: defaultdict(list))
    for c in cases:
        raw = ask(model, c, think)
        s = score(c, raw)
        rows.append({**c, "driver": model, "think": think, "answer": raw, "score": s})
        group = "decline" if c["gold"].get("decline") else c["specificity"]
        for key, val in s.items():
            agg[group][key].append(val)
        print(c["id"], s, raw[:100].replace("\n", " "), flush=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"\n== {model} think={think}")
    for g, d in agg.items():
        print(g, {k: f"{sum(v)}/{len(v)}" for k, v in d.items()})


if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        think=len(sys.argv) > 4 and sys.argv[4] == "think",
    )
