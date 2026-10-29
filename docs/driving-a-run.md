# Driving a run with a model

This page is for anyone who wants a model of their own to work a manyruns analysis with a
scientist, or alone. It explains the parts of the stack, each seam between them, the action
space a model chooses from, what is recorded, and how to attach a model and measure it.

Status on 2026-10-29: the seat and the reference driver are in two draft pull requests,
[manyruns#8](https://github.com/latent-reasoning-works/manyruns/pull/8) and
[manyagents#31](https://github.com/latent-reasoning-works/manyagents/pull/31). geomancer still
pins manyruns `v0.1.0`, which has no seat; the pin moves when #8 is released. Until then, run
from those branches (see [Try it](#try-it)).

## 1. The stack

Four repositories, each with one job. The arrows are Python dependencies.

```
  geomancer ──depends on──▶ manyruns ──depends on──▶ manylatents (+ manylatents-omics)
  (KL lab config)            (controller)             (compute)

  manyagents  ── no import ──  talks to a running manyruns over a local socket
  (drivers: any model)
```

| Repository | Its job | What it never does |
|---|---|---|
| **manylatents** | The compute: embeddings (PHATE, UMAP, PCA, diffusion maps), clustering, geometric metrics. Pure algorithms with Hydra configs. | Decide what to run, or record why |
| **manyruns** | The **controller**. It reads a dataset, works out which analyses are legal, runs one step at a time by calling manylatents, shows the result, and records every run and every choice. It owns what is legal. | Compute an embedding itself, or call a model to decide for it |
| **manyagents** | Drivers. It puts any model (local through Ollama or vLLM, or hosted) behind one adapter interface, and its `manyagents-drive` command attaches that model to a running manyruns session. | Import manyruns, or decide what is legal |
| **geomancer** | The Krishnaswamy Lab's configuration on manyruns: datasets, recipes, lab tools, and the knob card in `geomancer/configs/driver/`. | Hold engine code |

So manyruns is a controller that depends on manylatents for every number it shows. A model never
computes anything and never touches manylatents directly. It only chooses, through manyruns,
among moves manyruns has declared legal.

## 2. One run, step by step

A **recipe** is a list of steps, for example `normalize → pca → phate`. Each step names a
manylatents algorithm and its parameters. manyruns runs the recipe one step at a time:

1. **Observe.** manyruns reads the dataset and what it provides (counts, labels, time).
2. **Legal moves.** It lists the recipes the data can honestly support, and refuses the rest
   with a reason (a trajectory on data with no time axis, for example).
3. **Step.** It calls manylatents to run the next step.
4. **Ask.** For a step with knobs (PHATE, UMAP, Leiden), the **tune loop** stops and asks:
   `accept / tune / cancel (or key=value)`. Whoever has control answers. A new value refits the
   step; `accept` keeps it and moves on.
5. **Record.** Every run goes to `outputs/index.jsonl`, and every tune loop writes one
   **decision row** saying what was tried, what was kept, and who chose.

A model takes part only at step 4: it answers the tune loop's question, if it is allowed to.

## 3. The seams

A seam is the place where two parts meet and what is allowed to cross it.

| Seam | Between | What crosses | Where it is defined |
|---|---|---|---|
| **Recipe step** | manyruns → manylatents | a step's group, name and parameters; back come arrays and metrics | manyruns `pipeline/runner.py`, recipe YAML |
| **Lab config** | geomancer → manyruns | datasets, recipes, tools, the knob card | `geomancer/configs/` |
| **The seat** | a driver → a running manyruns session | JSON lines over a local socket | manyruns contract §8 |
| **The offer** | manyruns → driver | the action space for the question now waiting | manyruns `manyruns/driver.py` |
| **Control** | scientist → manyruns | how far the driver may go (the level) | run screen keys, `MANYRUNS_DRIVER_LEVEL` |
| **The record** | manyruns → anyone | decision rows: what was offered, tried, kept, by whom | manyruns `decisions.py`, `tune.py` |
| **Conformance** | fixtures → a driver's scorer | test cases with the verdict each choice must get | manyruns `tests/fixtures/driver_conformance.json`, `manyagents-conformance` |

### The seat

Every stepped run opens a **seat**: a Unix socket that only your user can open. The run's output
folder holds a pointer to it:

```json
outputs/<project>/driver.json
{"socket": "/tmp/mr-xxxx/d.sock", "pid": 4242, "tool_version": "...", "protocol_version": 1}
```

manyruns never calls a model. A driver **attaches** to the seat, the way any program can run
the CLI. One driver may attach at a time. The full protocol is §8 of the manyruns environment
contract (`docs/environment-contract.md` on manyruns#8). In short:

- Send `hello` first, with your `chooser` (your name on every row, e.g. `llm:mymodel@v3`) and
  your `bound` (how your choices are constrained: `logits`, `schema`, `prompt` or `none`). The
  reply carries `protocol_version` and the offer already waiting.
- Requests: `actions` (the current offer), `say` (text for the scientist), `propose` (a card),
  `apply` (a choice that runs at once).
- Events pushed to you: `control`, `offer`, `question`, `message` (the scientist asked you
  something), `said`, `applied`, `taken`, `refused`, `finished`.

### Control

The scientist decides how far the driver may go. No request can raise it.

| Level | The driver may | Tag on screen |
|---|---|---|
| `interpret` | read, and answer the scientist (`say`) | `HUMAN` |
| `propose` | also put up a card; the scientist takes it with `ctrl+y` | `HUMAN` |
| `drive` | also `apply` its own choices | red `ON AIR · MACHINE · <chooser>` |
| manual (`ctrl+b`) | talk only, until `ctrl+b` again restores its level | yellow `MANUAL · <chooser> standing by` |

`ctrl+g` steps through the levels. A run with nobody at the keys gets its level when it starts:
`MANYRUNS_DRIVER_LEVEL=drive manyruns`.

## 4. The action space

There are two surfaces.

**Which analysis (the ledger).** `manyruns actions <dataset>` prints the recipes, which can run on
this data, why the others cannot, and a `legal_id`. It is a command, not on the seat yet.

**Which values (the step offer).** While the tune loop waits, the seat's offer lists the knobs of
that step. These are all the knobs manyruns offers today:

| Step | Knob | Default | Minimum | Maximum | What it does (from `knobs.json`) |
|---|---|---|---|---|---|
| phate | `knn` | 5 | 1 | unknown | neighbours per point; lower keeps side branches apart |
| phate | `t` | `auto` (takes an integer) | 1 | unknown | diffusion time; higher is smoother and more line-like |
| phate | `decay` | 40 | 1 | unknown | kernel decay; lower connects distant points more |
| phate | `n_components` | 2 | 1 | unknown | output dimensions; does not change the shape |
| umap | `n_neighbors` | 15 | 1 | unknown | higher shows more global structure |
| umap | `min_dist` | 0.1 | 0.0 | unknown | higher spreads points out |
| umap | `n_components` | 2 | 1 | unknown | output dimensions |
| leiden | `resolution` | 1.0 | 0.1 | unknown | higher gives more, smaller clusters |

A recipe's own `params:` override the defaults. A maximum is known only when the recipe declares
a cap in `limits:` that the data resolves, for example `knn: n_samples`. No shipped recipe caps a
tunable knob yet (the only `limits:` today are on PCA's `n_components`), so every maximum above
is reported as unknown. An unknown maximum is never invented; a lab recipe can declare one.

An offer, exactly as the seat sends it:

```json
{"surface": "step", "step": "phate", "question_id": 2, "legal_id": "e14cfabcdf1b",
 "params": {
   "t":            {"current": "auto", "type": "integer", "also": ["auto"],
                    "minimum": 1, "maximum": null, "maximum_known": false},
   "decay":        {"current": 40, "type": "integer", "minimum": 1,
                    "maximum": null, "maximum_known": false},
   "knn":          {"current": 5,  "type": "integer", "minimum": 1,
                    "maximum": null, "maximum_known": false},
   "n_components": {"current": 2,  "type": "integer", "minimum": 1,
                    "maximum": null, "maximum_known": false}},
 "tool_version": "manyruns 0.1.0"}
```

A choice is one or more of these knobs with new values, or a decision:

```json
{"verb": "apply", "legal_id": "e14cfabcdf1b", "params": {"knn": 10}}
{"verb": "apply", "legal_id": "e14cfabcdf1b", "decision": "accept"}
```

manyruns checks every choice itself, whatever your model's schema said. It refuses, with the
reason, and runs nothing for:

- a `legal_id` that is not the current one (the question moved on: ask for `actions` again);
- a knob that is not in the offer, or a value of the wrong type;
- a value below the minimum or above a known maximum (refused, never clamped);
- a value equal to the current one.

## 5. What is recorded: the training shape

Each tune loop writes one decision row to the run's `decisions.jsonl`. When a driver took part,
the row is version 2. This one is real, with a few empty fields left out: a model proposed
`knn=10` as a card, and the scientist took it and then accepted.

```json
{"schema_version": 2, "surface": "tune", "run_id": "f721babfc1c4",
 "offered": [
   {"step": "phate", "params": {"n_components": 3, "knn": 5},  "label": "phate knn=5 n_components=3"},
   {"step": "phate", "params": {"n_components": 3, "knn": 10}, "label": "phate knn=10 n_components=3",
    "by": "human", "proposed_by": "llm:qwen3:8b"}],
 "chosen": "phate knn=10 n_components=3",
 "chooser": "human", "bound": "none",
 "proposed": {"params": {"knn": 10}, "chooser": "llm:qwen3:8b", "bound": "schema",
              "rationale": "for: make these look more like branches"},
 "rationale": "for: make these look more like branches"}
```

| Field | Meaning |
|---|---|
| `offered` | every value tried in this loop, in order; `by` names who set it, `proposed_by` whose card it was |
| `chosen` | the one kept (`null` when the scientist cancelled) |
| `chooser`, `bound` | who made the final choice, and how it was constrained |
| `proposed`, `rationale` | the card behind the kept value, if there was one |

A row whose chooser is a model is not training data for that same model. Rows where a model
proposed and a person judged are the most useful ones to train on.

**The row shape changes in the next manyruns release.** The example above is what the draft
seat writes today. The released seat keeps the same information in a different shape:

| Today (draft) | Next release |
|---|---|
| `"schema_version": 2` | `"schema_version": 1`, with the keys below added |
| `chooser` on the row | `"human"` when the scientist made the last choice; the driver's chooser when the driver accepted at level `drive` |
| `rationale` on the row and in `proposed` | inside `proposed.meta` only |
| `by`, `proposed_by` on each `offered` entry | unchanged |
| the offer hash `legal_id` | `offer_sha256`; `legal_id` keeps its ledger meaning |

Read rows by these keys, not by `schema_version`.

## 6. Try it

On the laptop, with Ollama running a local model. Until the pull requests merge, use their
branches: manyruns `feat/driver-socket` and manyagents `feat/drive-seat`.

Terminal 1, start manyruns and open a run (for example `swissroll`, then `embed`):

```sh
uv run manyruns
```

Terminal 2, attach a model with the lab's knob card:

```sh
manyagents-drive outputs/<project> --agent ollama --model qwen3:8b --knobs geomancer/configs/driver/knobs.json
```

For your own served model, use `--agent openai` with an OpenAI-compatible endpoint (for example
`vllm serve` with tool calling on).

Measure a model offline against the conformance cases:

```sh
manyagents-conformance <manyruns>/tests/fixtures/driver_conformance.json --agent ollama --model qwen3:8b
```

## 7. Baselines to beat

`manyagents-conformance` on the 20 cases, one run each at temperature 0. "Right knob" means the
right knob moved in the right direction (or to the asked value).

| Model | Legal moves | Right knob | Correctly declined |
|---|---|---|---|
| qwen3:1.7b | 0.71 | 0.31 | 0.57 |
| qwen3:8b | 0.80 | 0.46 | 0.71 |
| qwen3:8b, thinking | 0.81 | 0.69 | 0.57 |

These are a baseline, not a benchmark: 20 hand-labelled cases. A model trained on portal traces
should beat them on held-out cases.

## 8. What is not here yet

- **Portal knobs that manyruns does not offer.** The offer has only the knobs in section 4. A
  portal knob such as a graph `K` with presets 40, 50 and 60 needs a recipe step that declares it
  (and `presets:` on that step, which the offer then carries).
- **The ledger on the seat.** A driver reads recipe choices with `manyruns actions` for now.
