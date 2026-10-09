# Training mode: design and review notes

`geomancer train` lets an expert walk the CFlows workflow one step at a time, revise a
step's parameters as often as they like, keep every attempt, choose what goes forward, and
leave a record of what they did and why. "Training" means expert-guided execution and trace
collection. It does not fine-tune a model, learn a policy, or search hyperparameters on its own.

References to "the paper" are to the Cflows preprint (bioRxiv 10.1101/2023.03.28.534644,
version of 2025-08-07); line numbers are the preprint's own.

## Decisions

| # | Decision |
| --- | --- |
| D1 | Proof is two-level: an automated real-compute gate (`pytest -m real`, required in CI's pinned lane) and an acceptance session with lab researchers on lab data. Acceptance passes when the expert confirms the trace matches what they did and why; acceptance rate and cycles-to-satisfied are recorded, not thresholds. |
| D2 | The repeated-edit loop centres on PHATE, then the time step. PHATE is the space everything else happens in (lines 138–144, 535–549, 613–615) and the paper gives no PHATE settings; temporal binning is a named lever (lines 145–149); DPT's failure on disconnected cells (lines 229–230, Fig. 2) makes its neighbour count a diagnostic fix. |
| D3 | The workflow is defined in code, not as a recipe file: `filter_genes → normalize → transform (log1p) → pca (50) → phate → dpt → {discretize_time ∣ subcluster_time} → mioflow`. PCA to 50 follows lines 539–543. Every step is an ordinary step with iterations, selection and staleness. |
| D4 | A preprocessing step that does not apply (normalizing data that is not counts) passes through as `not_applicable`, selectable, with manyruns' reason recorded. **Flagged for review** (R4). |
| D5 | The expression matrix is recomputed from the source when PCA needs it and checked against the hash each preprocessing iteration recorded; from PCA on, arrays are stored. Large dense matrices never reach disk. |
| D6 | Time mode is chosen at `start` and recorded. **Real**: each measured timepoint is subclustered on PHATE coordinates with k-means, subclusters are ordered by mean DPT, and refined bins sit symmetrically within ±half the smaller neighbouring gap (lines 626–633). No resampling. `k_per_timepoint` has no default. **Pseudotime**: DPT binned into equal-width timepoints, for data with no time axis. |
| D7 | Granger and HVG are deferred until manylatents#292. HVG is reserved for Granger's gene space (lines 187–191, 368–369), not PCA input. |
| D8 | Selecting an iteration is satisfaction; continuing past a step is recorded. |
| D9 | Feedback → proposals: pluggable proposers, a versioned rule table first, an optional model tier only when no rule matches, every proposal confirmed by the researcher; fixed step sizes clipped to legal ranges; at most three ranked; no guessing. |
| D10 | Diagnostics are cheap and each backs a rule or question: PHATE kNN components and largest-component share (and time silhouette when measured time exists), DPT unreachable fraction and geodesic agreement, cells per time bin. |
| D11 | Interaction: `feedback` (one shot) and `loop` (interactive); identical reruns are flagged, not refused; a direct edit is followed by an optional "why?"; `--collect` asks after every decision. |
| D12 | The model tier is opt-in per session (`--proposer rules+model`, `geomancer[agents]`). It receives the words, parameter names and values and diagnostic numbers only, never data, plots or paths; its model id and prompt hash are recorded; replay reuses recorded parameters and never calls it again. |
| D13 | Every event carries a pseudonymous expert id. The trace format is a tested contract (`geomancer/training/contract.py`, schema version 2); version 1 sessions are refused, not migrated. |
| D14 | A learner lives outside Geomancer and reads exported traces; its first target is feedback + diagnostics + parameters → the change the expert accepted. It would plug in as another proposer. |
| D15 | Play sessions use ground-truth synthetic data (`make-play-data`) plus lab data. Truth scores are recorded but hidden by default, so sessions stay blind. |
| D16 | Five-module split with pluggable proposers, cached artifact hashes, and a deliberately linear chain (every CFlows stage feeds the next). Data loads through `manyruns.pipeline`'s public loaders, allowlisted in `tests/test_layer.py`. |

## Review list

Every item is a value in `geomancer/training/choices.py`; changing an answer is a one-line
edit, and `tests/test_training_choices.py` checks the new value is coherent.

| # | Question | Current answer |
| --- | --- | --- |
| R1 | Do the feedback rules point the right way, and which words do experts actually use? | `RULES` (version 1): blurry/merged → `t` ×0.67; noisy → `t` ×1.5; fragmented (only if the kNN graph is disconnected) → `knn` +2; one blob → `decay` ×0.75; bins too coarse/fine → ±1 |
| R2 | Gene filter before log1p: does keeping genes seen in ≥3 cells match lab practice? | `GENE_FILTER_MIN_CELLS = 3` |
| R3 | PCA on highly variable genes, or on all filtered genes? | all filtered genes (HVG deferred, D7) |
| R4 | **Flagged:** a preprocessing step that does not apply passes through silently, with its reason recorded. Acceptable, or should it stop and ask? | `NOT_APPLICABLE = "pass_through"` (`"stop"` makes it unselectable) |
| R5 | A standard number of subclusters per timepoint? | `K_PER_TIMEPOINT = None` (must be set per run) |
| R6 | Is ordering subclusters by mean DPT the paper's "local pseudotime ordering"? | `SUBCLUSTER_ORDER = "mean_dpt"` |

## How it stores things

```
outputs/<session>/training/
  session.json            workflow, source fingerprint, selections, notes, derived validity
  session.json.bak        the previous save
  trace.jsonl             append-only events (the interaction trace)
  iterations/<step>/NNN/  iteration.json + the arrays and plots manyruns wrote for it
```

An iteration is never overwritten. `iteration.json` records requested, resolved and changed
parameters (with where each default came from), the upstream iterations and input artifacts
with sha256, the replayed preprocessing, timestamps, status and error, outputs with hashes,
diagnostics, hidden truth scores, the seed and software versions, the expert and any feedback
that caused it. A selected iteration is **valid** only when computed from the currently
selected upstream iterations and, for the first step, unchanged source data; otherwise it is
**stale**, **broken** (missing or modified artifact) or **blocked**. Changing a selection runs
nothing; `continue` recomputes the first step that is not valid, one step at a time.

## The trace contract and learning examples

Each `trace.jsonl` line has `schema_version`, `at`, `session_id`, `expert` and `event`, plus
the fields `contract.EVENTS` lists for that event. A learning example is a `feedback` event
(verbatim `text`, the critiqued iteration's `params` and `diagnostics`, the `proposals` with
their proposer and rule version or model id) joined on `feedback_id` to its
`feedback_decision` (`accepted`, `edited` with final `overrides`, or `rejected`) and to the
`resulting_iteration`. `kind: "why"` events carry the reason an expert gave for their own
edit, selection or rejection. `geomancer train export -s NAME` writes a portable bundle with
absolute paths reduced to file names, no input data and no arrays, and refuses to write if
the trace breaks the contract or a machine path survives.

## Play data

`geomancer train make-play-data` writes four synthetic count datasets to `data/play/` (never
committed). Each cell has a latent `true_time`; cells are measured at a few discrete
timepoints. Measured at the workflow's defaults:

| Variant | PHATE kNN graph | DPT unreachable | DPT \|ρ\| vs truth | PHATE best-axis \|ρ\| |
| --- | --- | --- | --- | --- |
| `play_separated` | 6 components | 54% | 0.08 | 0.85 |
| `play_noisy` | connected | 0% | 0.91 | 0.92 |
| `play_overlap` | 2 components | 0% | 0.77 | 0.97 |
| `play_sparse_time` | 7 components | 69% | 0.13 | 0.66 |

Strongly separated snapshots break DPT exactly as the paper describes; the diagnostics flag
it, and repairing the graph is the expert's job in those sessions. `show --truth` reveals the
scores after a session.

## Running a session

```sh
geomancer train start DATA -s NAME --expert ID          # add --collect for play sessions
geomancer train loop -s NAME                            # filter_genes; `keep`, `done`
geomancer train continue -s NAME --select               # or loop each step in turn
geomancer train loop -s NAME phate                      # words, k=v, compare, keep, done
geomancer train continue -s NAME k_per_timepoint=3      # real-time mode needs k
geomancer train status -s NAME
geomancer train trace -s NAME
geomancer train export -s NAME --plots
```

## Known limitations

- MIOFlow's trained model stays in memory; only its outputs are stored.
- Interactive steps skip manyruns' full metric suite, as manyruns' own tuning loop does.
- In pseudotime mode, MIOFlow trains on bins derived from the same PHATE embedding; treat the
  result as exploration, not as evidence that a trajectory exists.
- Rules match whole phrases and skip simple negations; they do not understand sentences.
- The chain is linear: changing any step makes every later step stale.

## Upstream follow-ups (not filed)

- manyruns: a public data loader outside `manyruns.pipeline`, to drop the allowlist.
- manylatents: a sparse-preserving log1p; today log1p densifies (about 1 GB peak for pbmc3k).
- manyruns: `pipeline/loading.py`'s `_resolve_data_object` comment still says it normalizes
  and runs PCA; the code no longer does.
