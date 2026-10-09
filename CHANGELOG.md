# Changelog

## Unreleased

- `geomancer train`: expert-guided training mode over the CFlows workflow
  (`filter_genes → normalize → log1p → pca → phate → dpt → time → mioflow`). Run one step
  at a time, rerun with new parameters, keep every iteration, select what goes forward,
  continue explicitly. Downstream results go stale, never rerun, when an upstream
  selection changes. Real-time mode subclusters measured timepoints (CFlows Methods);
  pseudotime mode bins DPT. State and an append-only trace persist under
  `outputs/<session>/training/` (schema version 2).
- Plain-language feedback (`train feedback`, `train loop`): a versioned rule table and
  per-step diagnostics propose parameter changes the researcher accepts, edits or rejects;
  an optional model tier (`--proposer rules+model`) is consulted only when no rule matches.
  Words, proposals, decisions and resulting iterations are linked in the trace; direct
  edits take an optional "why?". `train compare` shows differences between iterations.
- Reviewable choices in `geomancer/training/choices.py`.
- Direct dependencies on numpy, scikit-learn, scipy and matplotlib; optional
  `geomancer[agents]` extra.

## 0.1.0 — G-1 skeleton

- The Krishnaswamy Lab's public configuration layer, built on manyruns by
  Latent Reasoning Works, pinned from git to `v0.1.0` (`79551f0`).
- A CLI wrapper with caller declaration and a temporary merged config directory.
- Deletable synthetic recipe, dataset, and metrics examples.
- Required pinned CI and advisory upstream-dev CI; no publication workflow.
- Tool examples await manyruns 0.1.1. The empty lab tool directory is omitted
  because Git cannot track empty directories; the wrapper preserves bundled tools.
