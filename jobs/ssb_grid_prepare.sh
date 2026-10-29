#!/bin/bash
# Prepare one frozen checkout of geomancer for the scShapeBench action grid.
# Run this on the Mila LOGIN node (it needs the network), once per commit:
#
#   bash jobs/ssb_grid_prepare.sh <full 40-hex commit>
#
# Why a frozen checkout: the lab tools import geomancer from $GEOMANCER_SRC on every
# run, and `shop sbatch --sync` fast-forwards $SCRATCH/geomancer. A grid that ran from
# that checkout would change code under its running tasks. Each grid task runs from
# $SCRATCH/geomancer-grid-<commit> instead, which nothing moves.
#
# It builds the worktree's venv from uv.lock and the two tool envs from their hashed
# locks (configs/tool/locks/*-linux-x86_64.lock, ~0.7 GB in ~/.cache/manyruns/tools),
# runs each tool's selftest, and prints the commit to pass to the grid jobs.
set -euo pipefail

SHA="${1:?usage: ssb_grid_prepare.sh <full commit sha>}"
REPO="${GEOMANCER_REPO:-$SCRATCH/geomancer}"
SRC="$SCRATCH/geomancer-grid-$SHA"

git -C "$REPO" fetch --quiet origin
if [ ! -d "$SRC" ]; then
  git -C "$REPO" worktree add --detach "$SRC" "$SHA"
fi
HEAD="$(git -C "$SRC" rev-parse HEAD)"
if [ "$HEAD" != "$SHA" ]; then
  echo "refused: $SRC is at $HEAD, not $SHA" >&2
  exit 1
fi
if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=all)" ]; then
  echo "refused: $SRC has uncommitted or untracked files" >&2
  exit 1
fi

cd "$SRC"
uv sync --locked --extra learner
uv run --no-sync geomancer tools install ssb_graph
uv run --no-sync geomancer tools install ssb_state
uv run --no-sync geomancer tools check
uv run --no-sync geomancer tools list

echo "ready: $SRC"
echo "pass to every grid job: --export=ALL,GEOMANCER_GIT_SHA=$SHA"
