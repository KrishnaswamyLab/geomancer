# Driver baseline: words to knob changes on the portal's PHATE knobs

A first baseline for an LLM study of issue #35: can a model turn what a
biologist types into the right change of the portal's knobs?

**Result:** untuned Qwen sets an explicitly named knob correctly (94%), but on
vague or goal requests it rarely picks the right knob. The best LLM (qwen3:8b,
thinking) moves the right knob most, the right way, in 33% of cases. The
portal's four keyword rules score 26%. Qwen also changes all three knobs in
most answers, and declines out-of-scope requests in at most 54% of cases.

## Setup

**Action space.** The three PHATE knobs of the portal's training mode
(https://btd2026.github.io/scShapeBench/), with its presets, ranges and
defaults (`knn` 100, `t` 12, `decay` 60). Every case starts at the defaults.

**Cases** (`cases.jsonl`, 201). A case is one scene x one persona x one
specificity, written by a generator model. The gold answer comes from the
scene, never from the generator.

| Scene | What the plot shows | Gold |
|---|---|---|
| merged_branches | lineages that should split follow one path | `t` down |
| noisy | no recognisable shape | `t` up |
| fragmented | one continuous process in separate groups | `knn` up |
| one_blob | one dense round mass | `decay` down |
| fewer_clusters | layout fine, too many cluster labels | decline |
| recolor | layout fine, wants colour by donor | decline |
| batch | batches sit apart, wants batch correction | decline |

The four action golds follow the portal's own rules (`rules_version` "1").

| Persona | Specificity |
|---|---|
| wet-lab biologist, bioinformatician, terse PI | vague (how the plot looks), goal (what structure is expected), explicit (names a knob and a value; the value is the gold) |

Generators: qwen3:32b and llama3.1:8b-instruct (temperature 0.9, 2 variants
each). Scene texts avoid the rules' trigger words, and vague or goal messages
that name a knob are rewritten (up to 3 tries) or dropped.

**Drivers** see a knob card (meaning, presets, range, current value) and the
message, and answer JSON: `{"changes": {...}}` or `{"decline": "..."}`
(Ollama, temperature 0, JSON format).

## Results

From `python summarize.py`. Every score is over all cases of its kind:
81 vague or goal, 48 explicit, 72 out-of-scope.

| Driver | main knob | strict | exact value | declined |
|---|---|---|---|---|
| no_change | 0/81 (0%) | 0/81 (0%) | 0/48 (0%) | 0/72 (0%) |
| always_decline | 0/81 (0%) | 0/81 (0%) | 0/48 (0%) | 72/72 (100%) |
| majority (decay down) | 18/81 (22%) | 18/81 (22%) | 0/48 (0%) | 0/72 (0%) |
| portal_rules | 21/81 (26%) | 21/81 (26%) | 0/48 (0%) | 61/72 (85%) |
| qwen3:1.7b | 11/81 (14%) | 2/81 (2%) | 45/48 (94%) | 17/72 (24%) |
| qwen3:8b | 10/81 (12%) | 0/81 (0%) | 45/48 (94%) | 39/72 (54%) |
| qwen3:8b, thinking | 27/81 (33%) | 0/81 (0%) | 45/48 (94%) | 22/72 (31%) |

- **main knob:** the knob the answer changed by the largest factor is the gold knob, moved the gold way.
- **strict:** the gold knob, moved the gold way, and no other knob changed.
- **portal_rules** declines whenever no keyword matches, so its decline score is not a skill: it also declines 54 of the 81 real requests.
- qwen3:8b changed all three knobs in 67 of 81 vague or goal answers.

## Known limits

1. **Gold labels are one opinion.** The scene-to-knob golds are the portal's rules, not measured changes. They need a check by the portal's authors.
2. **Low wording variety.** The type/token ratio of the messages is 0.11.
3. **Generator effect.** The rules score 7/37 on qwen3-written and 14/44 on llama3.1-written vague or goal cases. Report per generator, or hold one out.
4. **Some leakage remains.** 38 of 153 non-explicit messages contain a rule trigger word ("scattered", "blob"). They are kept, as people do use these words.
5. **Same family.** qwen3:32b wrote half the cases that qwen3 drivers answer.

An earlier run (not kept) seeded the scene texts with the rules' trigger words; 40 of 75 messages copied them and the rules scored far higher.

## Rerun

Needs a local Ollama server with the models pulled. Run each command from this folder on the laptop.

```bash
python3 make_prompts.py qwen3:32b,llama3.1:8b-instruct-fp16 cases.jsonl
```

```bash
python3 baselines.py cases.jsonl
```

```bash
python3 drive.py qwen3:8b cases.jsonl results/out-qwen3_8b.jsonl
```

```bash
python3 drive.py qwen3:8b cases.jsonl results/out-qwen3_8b-think.jsonl think
```

```bash
python3 summarize.py
```

`make_prompts.py` overwrites `cases.jsonl`; generation is not seeded at the
model, so a rerun gives new messages.
