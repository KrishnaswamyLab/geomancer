geomancer is built on [manyruns](https://github.com/latent-reasoning-works/manyruns) by **Latent Reasoning Works**, and adds the **Krishnaswamy Lab**'s public configuration layer.

[![Built on manyruns](https://img.shields.io/badge/built_on-manyruns-blue)](https://github.com/latent-reasoning-works/manyruns)

# geomancer

Install from the lab repository with uv and Python 3.11–3.12 (macOS or Linux):

```sh
uv tool install --python 3.12 git+https://github.com/KrishnaswamyLab/geomancer
geomancer --version
geomancer check
geomancer tools check
geomancer run example_blobs --recipe example_pca --engine mock --project example
```

The repository must first be created and pushed by its owner. geomancer is never
published to PyPI. It pins manyruns from Git at `v0.1.0`, commit
`79551f04127fdad75aa396eeebcb27351ebd284b`. Under this version, `--version` reports
`geomancer <manyruns version> · revision <manyruns commit>`. A mock run proves
CLI and record plumbing only; its numbers are not scientific results.

## What this lab adds

G-1 supplies a thin wrapper and one **deletable example** in each of three kinds:

| Kind | Example | Purpose |
| --- | --- | --- |
| Recipe | `example_pca` | Two-dimensional PCA, with bounds from the input shape |
| Dataset | `example_blobs` | Synthetic Gaussian blobs from the public manylatents generator |
| Metrics | `example_geometry` | Trustworthiness and continuity, with declared ranges |

These are examples a lab deletes when its own public configurations are ready.
They are not lab recipes or results. No data files ship. The metrics example can
be loaded with `manyruns.catalog.load_suite("example_geometry")` after the wrapper
has configured the directories. The CLI still uses the bundled `default` suite;
a uniquely named suite does not replace it. v0.1.0's `check` covers recipes and
datasets; the tests validate metrics with `catalog.check_suite` and `check_ranges`.

There is no lab tool example in G-1: it requires manyruns 0.1.1. The empty lab
`configs/tool/` directory is omitted because Git does not track empty directories.
Bundled manyruns tools remain available; `tools check` in 0.1.0 reports missing
locks or environments without failing, so it does not prove a tool can run.

The wrapper reads only the working directory's `.env`, then sets
`MANYRUNS_RECIPE_DIR`, `MANYRUNS_DATASET_DIR`, `MANYRUNS_METRICS_DIR`, and
`MANYRUNS_TOOL_DIR` to temporary copies containing bundled configs plus lab configs.
It copies nested locks and adjacent adapters and refuses duplicate filenames.
One temporary root is reused per process and removed at normal process exit.
A directory override already set in the shell wins; a cwd `.env` value comes next;
the merged default comes last. A user directory replaces the combined catalog
outright on 0.1.0. Do not commit environment files.

`MANYRUNS_CALLER=geomancer` and its version are declared together unless a caller
name is already set. The wrapper delegates the CLI to manyruns; caller-aware
version formatting and run records arrive upstream in 0.1.1.

## Development and provenance

The initial offline skeleton intentionally omits `uv.lock`. On a networked machine:

```sh
uv lock
uv lock --check
uv sync --locked
uv run --locked geomancer --version
uv run --locked geomancer check
uv run --locked geomancer tools check
uv run --locked pytest -q
```

The project declares torch directly on Linux and maps it to an explicit
[PyTorch CPU index](https://docs.astral.sh/uv/guides/integration/pytorch/).
Relock after changing this source so `uv sync --locked` installs CPU wheels on
Linux without downloading unused CUDA dependencies. macOS keeps using PyPI's
CPU wheels; unrelated packages cannot use the explicit index.
With uv 0.11.8, `--torch-backend` is accepted by `uv tool install` (as used in
the isolated consumer test) and `uv pip install`, but not by `uv lock`,
`uv sync`, or `uv run`.

Commit the resulting `uv.lock`. Tests include the mock record assertions and an
isolated `uv tool install git+file://...` smoke checking the printed revision
against the lock. Until the lock exists, those checks skip locally and CI fails.
Do not interpret the offline tests as proof of resolution or consumer installation.

See [CONTRIBUTING.md](CONTRIBUTING.md) for local upstream overlays, pin changes,
and the promotion procedure. The required `pinned` lane tests the locked version;
the advisory `upstream` lane tests `dev` and first proves the overlay is different.

The upstream [lab registry](https://github.com/latent-reasoning-works/manyruns/blob/main/docs/labs.md)
is planned for manyruns 0.1.1 and is absent at the current pin. Register the lab
there when it lands. Cite both projects using [CITATION.cff](CITATION.cff).
The MIT notices, including credit for the copied upstream public-safety gate and
build inclusion rules, are in [LICENSE](LICENSE).
