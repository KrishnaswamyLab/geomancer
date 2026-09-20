# Contributing

geomancer is the Krishnaswamy Lab's public layer over manyruns by Latent Reasoning
Works. Use uv and Python 3.11–3.12. It is installed from Git, never published to PyPI.

Start with `uv sync --locked --torch-backend=cpu`, then `uv run --locked pytest -q`.
The initial offline skeleton deliberately has no `uv.lock`: its owner must first
run `uv lock --torch-backend=cpu`, validate it, and commit it before CI can pass.
Do not replace the git source with a published dependency or commit local overlays.

Give recipes stable dotted `id` values, and resolvable `source` metadata when
published. Keep every lab filename distinct from bundled names. Examples are
explicitly deletable: replace them when real public lab configurations are ready.
Do not commit data, credentials, environment files, or private organization names.
The public-safety helper and tracked-file gate are copied from manyruns v0.1.0;
its upstream-only name-hash dependency and bundled-lock regression are removed,
and the data placeholder exception is removed. No private name list is shipped.

To fix manyruns while working here:

1. Clone manyruns beside this checkout, on a branch off `dev`. Use
   `uv run --with-editable ../manyruns geomancer ...` and
   `uv run --with-editable ../manyruns pytest` on every local invocation.
2. Send the failing test and fix upstream to manyruns `dev`; run its required checks.
3. A geomancer development branch may pin `branch = "dev"` or a development `rev`
   and regenerate its lock. Pushes to that branch may pass, but its PR targeting
   `main` must reject the development pin. There is no bypass switch.
4. After the upstream release PR reaches `main` and is tagged, change geomancer's
   source to that tag, relock, test, and merge. A full 40-hex commit already on
   upstream `main` is the exception for a fix needed before a tag; a development
   commit is not. CI verifies ancestry for this exception.

Tick the PR's upstream-candidate checkbox when real expected outputs on more
than one public or synthetic dataset justify promotion. Open a manyruns `dev` PR
citing the lab PR, with the complete config dependency closure and a maintenance
owner. Preserve IDs and copyright notices. At the next pin bump, delete promoted
copies: the no-shadowing test is the reminder.

The upstream [lab registry](https://github.com/latent-reasoning-works/manyruns/blob/main/docs/labs.md)
and [upstreaming procedure](https://github.com/latent-reasoning-works/manyruns/blob/main/docs/upstreaming.md)
are planned for manyruns 0.1.1; they are absent at the G-1 pin. Register the lab
there when those pages land. Include `geomancer --version` in compatibility reports.

`pinned` is the required CI lane. `upstream` overlays manyruns `dev` on every
runtime command and is advisory; its revision must differ from the lock. The
pytest `--upstream` option changes only the runtime revision expectation; it never
relaxes the declaration or lock checks. The isolated install smoke always checks
the actual pinned consumer install, even when the rest of the suite uses `dev`.
