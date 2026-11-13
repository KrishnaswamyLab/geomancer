# Changelog

## Unreleased

- Pin manyruns to `main` commit `4d19cf1` instead of `v0.1.0`. It brings the
  caller provenance that the tests already expected: the `caller` field on run
  records and the "built on manyruns" `--version` line. It also brings the tune
  surface rule: a `left` or `right` step that does not change the value is
  `null`, so a knob at `t="auto"` offers `left: null, right: 20`.

## 0.1.0 — G-1 skeleton

- The Krishnaswamy Lab's public configuration layer, built on manyruns by
  Latent Reasoning Works, pinned from git to `v0.1.0` (`79551f0`).
- A CLI wrapper with caller declaration and a temporary merged config directory.
- Deletable synthetic recipe, dataset, and metrics examples.
- Required pinned CI and advisory upstream-dev CI; no publication workflow.
- Tool examples await manyruns 0.1.1. The empty lab tool directory is omitted
  because Git cannot track empty directories; the wrapper preserves bundled tools.
