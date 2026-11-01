"""Write synthetic biologist requests for the portal's PHATE knobs.

Each case is a plain dict: the plot's current knob values, the scene (what
the plot shows and what the person wants), a persona, a specificity level,
and the gold answer. A generator model writes only the words; the gold
answer comes from the scene, never from the model.

Design follows the few-shot "communication characteristics" recipe
(under-specification, over-specification, jargon level) and checks lexical
diversity. Usage: python make_prompts.py <model>[,<model>...] <out.jsonl>
"""

import json
import random
import re
import sys
import urllib.request

KNOBS = {
    "knn": {"presets": [5, 10, 15, 30, 100, 200, 500, 1000], "range": [2, 1000]},
    "t": {"presets": [5, 12, 20, 50, 75, 100], "range": [1, 200]},
    "decay": {"presets": [15, 40, 60, 80], "range": [1, 200]},
}
START = {"knn": 100, "t": 12, "decay": 60}

# scene id -> (what the plot shows, the gold answer). Directions follow the
# portal's own rules (rules_version "1") and the PHATE knob meanings.
SCENES = {
    # Neutral wording: none of the portal's rule trigger words, so the
    # generator cannot copy them from the scene.
    "merged_branches": (
        "you expected two or more lineages to split apart (for example erythroid and myeloid), but on the plot they follow one path and you cannot tell where they diverge",
        {"knob": "t", "direction": "down"},
    ),
    "noisy": (
        "the cells form no recognisable shape; cells that sit next to each other on the plot do not look related",
        {"knob": "t", "direction": "up"},
    ),
    "fragmented": (
        "cells you know form one continuous differentiation process appear as several separate groups with empty gaps between them",
        {"knob": "knn", "direction": "up"},
    ),
    "one_blob": (
        "all cells sit in one dense round mass and you cannot see any internal organisation",
        {"knob": "decay", "direction": "down"},
    ),
    # Out of the action space: a correct driver declines.
    "fewer_clusters": (
        "the shape of the plot looks right to you, but the cluster colours split it into too many labels; you want fewer cluster labels and the same layout",
        {"decline": True},
    ),
    "recolor": (
        "the shape looks fine; you want the cells coloured by donor instead of by cell type",
        {"decline": True},
    ),
    "batch": (
        "cells from your two sequencing batches sit apart on the plot even where they are the same cell type, and you want batch correction",
        {"decline": True},
    ),
}

PERSONAS = {
    "wet_lab": "a wet-lab biologist who knows the biology well but has never heard of PHATE settings",
    "bioinformatician": "a computational biologist who knows what PHATE is but speaks informally",
    "pi": "a busy PI typing a short, terse message between meetings",
}
SPECIFICITY = {
    "vague": "Say only how the plot looks or what is wrong with it, in everyday lab words. Do not name any setting or number.",
    "goal": "Say what biological structure you expect to see (for example branches, a continuum, separate populations) and that the plot does not show it. Do not name any setting or number.",
    "explicit": "Name the exact setting and the exact new value: {explicit}.",
}

FEWSHOT = """Examples of real-sounding messages (not from this dataset):
- "the erythroid and myeloid arms are just mashed together here, can you pull them apart?"
- "hmm this looks like noise, I can't see anything"
- "can we bump k to 30 and see"
"""

LEAK = re.compile(
    r"\b(k|knn|neighbou?rs?|t|diffusion|decay|alpha|kernel)\b\s*[=:]?\s*\d*",
    re.IGNORECASE,
)


def ollama(model, prompt, temperature=0.9):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": 120},
        }
    ).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        body,
        {"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=300).read())[
        "response"
    ].strip()


def explicit_target(rng):
    knob = rng.choice(list(KNOBS))
    value = rng.choice([v for v in KNOBS[knob]["presets"] if v != START[knob]])
    return knob, value


def main(models, out, n_variants=2, seed=0):
    rng = random.Random(seed)
    cases = []
    for scene, (shows, gold) in SCENES.items():
        for persona, who in PERSONAS.items():
            levels = ["vague", "goal"] + ([] if gold.get("decline") else ["explicit"])
            for level in levels:
                for model in models:
                    for v in range(n_variants):
                        g = dict(gold)
                        instr = SPECIFICITY[level]
                        if level == "explicit":
                            knob, value = explicit_target(rng)
                            g = {
                                "knob": knob,
                                "value": value,
                                "direction": "up" if value > START[knob] else "down",
                            }
                            instr = instr.format(explicit=f"{knob} = {value}")
                        prompt = (
                            f"You are {who}. You are looking at a PHATE plot of your single-cell data, where {shows}.\n"
                            f"Write the one message you would type to the analysis assistant. {instr}\n"
                            f"Write 1 to 2 sentences, as you would really type them. Output only the message.\n\n{FEWSHOT}"
                        )
                        for attempt in range(3):
                            text = ollama(model, prompt).strip().strip('"')
                            if level == "explicit" or not LEAK.search(text):
                                break
                        else:
                            continue
                        cases.append(
                            {
                                "id": f"{scene}-{persona}-{level}-{model.split(':')[0]}-{v}",
                                "scene": scene,
                                "persona": persona,
                                "specificity": level,
                                "text": text,
                                "current": START,
                                "gold": g,
                                "generator": model,
                            }
                        )
                        print(cases[-1]["id"], "|", text, flush=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(c) + "\n" for c in cases)
    words = [w.lower() for c in cases for w in re.findall(r"[a-zA-Z']+", c["text"])]
    print(
        f"\n{len(cases)} cases, type/token ratio {len(set(words)) / max(1, len(words)):.2f}"
    )


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2])
