"""Every choice the lab is asked to review, in one place.

Each value below is a provisional answer to a question the CFlows paper leaves open, or one
made only to get a first session running. Changing an answer is a one-line edit here; the tests
in `tests/test_training_choices.py` pin what each value does, so an edit that breaks the
workflow fails loudly. `docs/training-mode.md` lists the open questions behind each one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ── preprocessing ───────────────────────────────────────────────────────────────────────
#: `filter_genes` keeps genes detected in at least this many cells, before log1p densifies the
#: matrix. 3 is manyruns' own default. REVIEW: does this match what the lab does before PCA?
GENE_FILTER_MIN_CELLS = 3

#: Library size `normalize` scales every cell to (manyruns' own declaration).
NORMALIZE_TARGET_SUM = 10000

#: `transform` method: "log1p" or "sqrt" (the two manyruns offers).
TRANSFORM_METHOD = "log1p"

#: PCA width before PHATE. The CFlows Methods project onto the top 50 components (lines
#: 539–543). manyruns clamps it to the data's own dimensions and records the clamp.
PCA_COMPONENTS = 50

#: What happens when a preprocessing step does not apply (non-count data reaching `normalize`,
#: no gene names reaching `filter_genes`):
#:   "pass_through" — recorded as `not_applicable`, selectable, output equals input;
#:   "stop"         — recorded as `skipped`, not selectable; the chain stops there.
#: REVIEW (flagged): is silently passing through acceptable, or should it stop and ask?
NOT_APPLICABLE = "pass_through"

# ── embedding ───────────────────────────────────────────────────────────────────────────
#: PHATE output dimensions (manyruns' own `cflows` declaration).
PHATE_COMPONENTS = 3

# ── time ────────────────────────────────────────────────────────────────────────────────
#: Pseudotime mode: equal-width bins over the DPT ordering (manyruns' `pseudotime` declaration).
PSEUDOTIME_BINS = 5

#: Real-time mode: subclusters per real timepoint. `None` means no default — the expert must
#: set it before `subcluster_time` runs, because the paper gives no number.
#: REVIEW: is there a standard number? Setting an int here makes it the default.
K_PER_TIMEPOINT: Optional[int] = None

#: How subclusters within a timepoint are ordered. Only "mean_dpt" exists: order by each
#: subcluster's mean DPT value, the reading of the paper's "local pseudotime ordering"
#: (lines 629–630). REVIEW: is that how it was done? A new ordering is one function in
#: `geomancer/training/timebins.py` plus its name here.
SUBCLUSTER_ORDER = "mean_dpt"

# ── plain-language feedback ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Rule:
    """When the researcher's words match, propose moving one parameter one step.

    `words` are matched as whole phrases, case-insensitively. `op` is "mul" (multiply by
    `amount`) or "add" (add `amount`). `requires` names a diagnostic that must be true for the
    rule to apply; if it is false the rule is reported as contradicted instead of proposed.
    """
    id: str
    step: str
    words: tuple[str, ...]
    param: str
    op: str
    amount: float
    reason: str
    requires: Optional[str] = None


#: Bump whenever RULES change; every proposal records the version that produced it.
RULES_VERSION = "1"

#: REVIEW: these directions are a first reading of PHATE's parameters, not established practice.
#: What words do experts actually use? Add a row per phrase family.
RULES: tuple[Rule, ...] = (
    Rule("phate-oversmoothed", "phate",
         ("blurry", "blurred", "over-smoothed", "oversmoothed", "too smooth", "washed out",
          "branches merged", "merged branches", "smeared"),
         "t", "mul", 0.67,
         "a shorter diffusion time keeps finer structure"),
    Rule("phate-noisy", "phate",
         ("noisy", "speckled", "grainy", "fuzzy edges", "jagged", "scattered"),
         "t", "mul", 1.5,
         "a longer diffusion time denoises the embedding"),
    Rule("phate-fragmented", "phate",
         ("fragmented", "islands", "broken apart", "disconnected", "pieces", "split up"),
         "knn", "add", 2,
         "more neighbours connect the kNN graph",
         requires="phate.knn_disconnected"),
    Rule("phate-one-blob", "phate",
         ("one blob", "blob", "too compressed", "collapsed", "squashed", "no structure"),
         "decay", "mul", 0.75,
         "a lower decay widens the kernel's tails, spreading the embedding"),
    Rule("bins-too-coarse", "discretize_time",
         ("too coarse", "steps too big", "too few bins", "not enough bins"),
         "n_timepoints", "add", 1, "more bins give finer time steps"),
    Rule("bins-too-fine", "discretize_time",
         ("too fine", "bins too sparse", "too many bins", "empty bins"),
         "n_timepoints", "add", -1, "fewer bins give each bin more cells"),
    Rule("subclusters-too-coarse", "subcluster_time",
         ("too coarse", "steps too big", "not enough subclusters"),
         "k_per_timepoint", "add", 1, "more subclusters refine time within each timepoint"),
    Rule("subclusters-too-fine", "subcluster_time",
         ("too fine", "too many subclusters", "subclusters too small"),
         "k_per_timepoint", "add", -1, "fewer subclusters keep each refined bin populated"),
)

#: At most this many proposals are shown for one piece of feedback.
MAX_PROPOSALS = 3

#: Where a multiplicative step starts when the current value is not a number (PHATE `t="auto"`):
#: manyruns' own tuning strip starts `t` from 20 (`tui/params.CONCRETE_FROM`).
CONCRETE_FROM = {"t": 20}
