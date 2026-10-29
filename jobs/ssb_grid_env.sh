# Shared setup for the grid's three Mila jobs (ssb_grid_{convert,drive,reduce}.sbatch).
# Sourced, never run. It refuses unless the frozen checkout for GEOMANCER_GIT_SHA is
# prepared, cds into it and sets the variables every step needs.
#
# Layout, one directory per commit so that a resubmission resumes and a new commit
# never mixes into an old grid:
#   $SCRATCH/geomancer/outputs/grid/<sha12>/inputs/{synthetic,real}  convert.py
#   $SCRATCH/geomancer/outputs/grid/<sha12>/<source>/                drive, score, table
#     ledgers/  cells/  graphs/  scores/  tables/<source>/seed<s>/
# Each source has its own grid directory because table.py checks the score file
# against the sha256 of every ledger under the grid: a real array that appends while
# the synthetic table is built would otherwise invalidate the synthetic scores.

# A shell from ssh or from Slurm does not read ~/.bashrc, so name uv's home and keep its cache
# on scratch (the home quota is small) here rather than relying on the login profile.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
: "${GEOMANCER_GIT_SHA:?pass --export=ALL,GEOMANCER_GIT_SHA=<full sha> (the commit jobs/ssb_grid_prepare.sh built)}"
SRC="$SCRATCH/geomancer-grid-$GEOMANCER_GIT_SHA"
GRID_ROOT="${GEOMANCER_GRID_ROOT:-$SCRATCH/geomancer/outputs/grid/${GEOMANCER_GIT_SHA:0:12}}"
INPUTS="$GRID_ROOT/inputs"
if [ ! -d "$SRC/.venv" ]; then
  echo "refused: $SRC is not prepared; run jobs/ssb_grid_prepare.sh $GEOMANCER_GIT_SHA on the login node" >&2
  exit 1
fi
mkdir -p "$GRID_ROOT"
# Run from the frozen checkout: `python -m` puts the working directory first on
# sys.path, so running from $SCRATCH/geomancer would import that (moving) checkout.
cd "$SRC"

export MANYLATENTS_DATA="${MANYLATENTS_DATA:-$SCRATCH/lrw/data}"
# The adapters import geomancer from here; geomancer checks it is at GEOMANCER_GIT_SHA.
export GEOMANCER_SRC="$SRC"
# Two BLAS/OpenMP threads per worker and per tool process.
THREADS=2
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-2} / THREADS ))
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS NUMBA_NUM_THREADS=$THREADS
# No MANYRUNS_TOOL_*_PYTHON: the tools run in their locked envs, so rows say
# tool_reproducible true. The driver refuses a tool that is not built.
unset MANYRUNS_TOOL_SSB_GRAPH_PYTHON MANYRUNS_TOOL_SSB_STATE_PYTHON

run() { uv run --no-sync --project "$SRC" python -m "$@"; }

echo "grid: commit $GEOMANCER_GIT_SHA, checkout $SRC, outputs $GRID_ROOT, workers $WORKERS x $THREADS threads"
