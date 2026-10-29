#!/bin/bash
# The scShapeBench action grid on Mila: convert the inputs, run the cells, then score
# and join them. One script, three steps, chosen by GEOMANCER_GRID_STEP.
#
# Every step runs from the frozen checkout $SCRATCH/geomancer-grid-$GEOMANCER_GIT_SHA
# that jobs/ssb_grid_prepare.sh built on the login node, never from $SCRATCH/geomancer,
# which `shop sbatch --sync` moves. The commit must be passed in; the script refuses
# without it, and geomancer refuses a checkout that is dirty or at another commit.
#
# Steps (submit from the laptop, in this order; each later one with --dependency):
#   convert  one task. Writes grid_inputs/<source>/<sample>.npy and manifest.json.
#            --export=ALL,GEOMANCER_GIT_SHA=<sha>,GEOMANCER_GRID_STEP=convert,GEOMANCER_GRID_SOURCE=all
#   drive    a SLURM array; task i runs shard i of the array's task count.
#            synthetic: --array=0-99 (10 samples x 3 seeds per task)
#            real:      --array=0-78 (1 dataset x 3 seeds per task; set --mem=40G)
#            --export=ALL,GEOMANCER_GIT_SHA=<sha>,GEOMANCER_GRID_STEP=drive,GEOMANCER_GRID_SOURCE=synthetic
#   reduce   one task, after the array (--dependency=afterany:<array job id>), so a
#            failed shard is counted in the table rather than blocking it.
#            --export=ALL,GEOMANCER_GIT_SHA=<sha>,GEOMANCER_GRID_STEP=reduce,GEOMANCER_GRID_SOURCE=synthetic
#
# Resubmitting the drive array resumes: each shard skips the cells that already have a
# final row at this commit. Keep the same --array range, because a shard reads only its
# own ledgers.
#
# Sizing (measured on a loaded laptop, so upper bounds; remeasure on one Mila node
# first): a synthetic sample-seed takes about 85 s at 775 points, most of it tool
# process start; a real dataset-seed (5000 points) takes about 15 min. One worker peaks
# at 4.9 GB on a 5000-point dataset, and each task also reads its shard's ledgers.
# Memory per task is workers x 6 GB + 3 GB: 4 workers -> 27 GB.
#SBATCH --job-name=ssb_grid
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
set -euo pipefail

: "${GEOMANCER_GIT_SHA:?pass --export=ALL,GEOMANCER_GIT_SHA=<sha> (the commit jobs/ssb_grid_prepare.sh built)}"
STEP="${GEOMANCER_GRID_STEP:?set GEOMANCER_GRID_STEP to convert, drive or reduce}"
SOURCE="${GEOMANCER_GRID_SOURCE:-synthetic}"
SRC="$SCRATCH/geomancer-grid-$GEOMANCER_GIT_SHA"
OUT="${GEOMANCER_GRID_OUT:-$SCRATCH/geomancer/outputs/ssb}"
if [ ! -d "$SRC/.venv" ]; then
  echo "refused: $SRC is not prepared; run jobs/ssb_grid_prepare.sh $GEOMANCER_GIT_SHA on the login node" >&2
  exit 1
fi
mkdir -p "$OUT" outputs/slurm
# Run from the frozen checkout: `python -m` puts the working directory first on
# sys.path, so running from $SCRATCH/geomancer would import that checkout instead.
cd "$SRC"

export MANYLATENTS_DATA="${MANYLATENTS_DATA:-$SCRATCH/lrw/data}"
# The adapters import geomancer from here; geomancer checks it is at GEOMANCER_GIT_SHA.
export GEOMANCER_SRC="$SRC"
# Two BLAS/OpenMP threads per worker and per tool process, so 4 workers fill 8 CPUs.
THREADS=2
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-2} / THREADS ))
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS NUMBA_NUM_THREADS=$THREADS
# No MANYRUNS_TOOL_*_PYTHON: the tools run in their locked envs, so rows say
# tool_reproducible true. The driver refuses a tool that is not built.
unset MANYRUNS_TOOL_SSB_GRAPH_PYTHON MANYRUNS_TOOL_SSB_STATE_PYTHON

run() { uv run --no-sync --project "$SRC" python -m "$@"; }

case "$STEP" in
  convert)
    run geomancer.learner.convert source="$SOURCE" workers="$WORKERS"
    ;;
  drive)
    : "${SLURM_ARRAY_TASK_ID:?the drive step is a SLURM array (--array=0-<n-1>)}"
    run geomancer.learner.drive source="$SOURCE" out="$OUT" \
      shard="$SLURM_ARRAY_TASK_ID" of="$SLURM_ARRAY_TASK_COUNT" seeds=0,1,2 \
      workers="$WORKERS" threads="$THREADS"
    ;;
  reduce)
    classifiers=""
    if [ "$SOURCE" = real ]; then classifiers="classifiers=rf,stats_svm,gnn"; fi
    run geomancer.learner.score grid="$OUT" source="$SOURCE" workers="$WORKERS" $classifiers
    run geomancer.learner.table grid="$OUT" source="$SOURCE"
    ;;
  *)
    echo "unknown GEOMANCER_GRID_STEP=$STEP; expected convert, drive or reduce" >&2
    exit 1
    ;;
esac
