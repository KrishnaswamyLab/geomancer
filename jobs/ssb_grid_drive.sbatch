#!/bin/bash
# Grid step 2 of 3: run the cells. A SLURM array; task i runs shard i of
# GEOMANCER_GRID_OF shards (default 100 synthetic, 79 real). A shard is a set of whole
# samples (synthetic) or datasets (real): all 24 actions x seeds 0,1,2 of each, plus
# one ssb_state run per sample.
#   synthetic: --array=0-99, 10 samples per task (1000 samples, 72,000 graph cells)
#   real:      --array=0-78, 1 dataset per task (79 high_support datasets)
# Outputs: $SCRATCH/geomancer/outputs/grid/<sha12>/<source>/ (jobs/ssb_grid_env.sh).
# Each task writes its own ledger directory, ledgers/<source>-<i>of<n>-<jobid>-p<pid>/.
#
# Resume: resubmit the same command with the same --array range. drive.py skips every
# cell that already has a final row (ok, error, not_run, skipped) at this commit and
# reruns aborted cells and cells with no row. It reads only its own shard's ledgers,
# so the shard count must not change. It is GEOMANCER_GRID_OF, not the array's task
# count, so that only the failed shards can be resubmitted (--array=3,17). A new
# commit is a new grid directory.
#
# The whole sequence, all on the LAPTOP unless marked (replace <sha> with the full
# commit; every job gets --export=ALL,GEOMANCER_GIT_SHA=<sha>):
#   0. MILA LOGIN NODE: bash jobs/ssb_grid_prepare.sh <sha>
#   1. shop sbatch mila jobs/ssb_grid_convert.sbatch --sync -- --export=ALL,GEOMANCER_GIT_SHA=<sha>
#   2. shop sbatch mila jobs/ssb_grid_drive.sbatch -- --dependency=afterok:<convert id> \
#        --array=0-99 --export=ALL,GEOMANCER_GIT_SHA=<sha>,GEOMANCER_GRID_SOURCE=synthetic
#   3. shop sbatch mila jobs/ssb_grid_drive.sbatch -- --dependency=afterok:<convert id> \
#        --array=0-78 --mem=32G \
#        --export=ALL,GEOMANCER_GIT_SHA=<sha>,GEOMANCER_GRID_SOURCE=real
#   4. shop sbatch mila jobs/ssb_grid_reduce.sbatch -- --dependency=afterany:<step 2 id> \
#        --export=ALL,GEOMANCER_GIT_SHA=<sha>,GEOMANCER_GRID_SOURCE=synthetic
#   5. the same as 4 with <step 3 id> and GEOMANCER_GRID_SOURCE=real
#
# SIZING (laptop, 2026-10-29, clean tree at 0add5c4, locked tool envs, workers=1, load
# average 7 to 16 on 18 cores; remeasure on one Mila task before trusting the limits):
# - synthetic, one (sample, seed), all 24 actions: 54 to 73 s at 775 and 1431 points,
#   21 s at 38 points (only the raw arm runs). 87% of it is ssb_graph tool runs, about
#   3 s each, mostly interpreter start and imports. Peak RSS 1.3 GB.
#   -> a task of 10 samples x 3 seeds is about 30 min of worker time, so about 8 min
#   with 4 workers. --time=1:00:00 leaves a margin for a slower node.
# - real, 5000 cells x 50 PCs (SCD-0002), all 24 actions: 122 s per seed (pca unit 42 s,
#   phate 42 s, diffmap 17 s until it fails, raw 17 s). One task (1 dataset x 3 seeds)
#   took 180 s with 2 workers. Peak RSS 5.0 GB at 5000 cells, so 4 workers x 5 GB + the
#   parent -> --mem=32G.
# - the default metric suite costs 0.23 s per branched graph run at 775 points, 0.62 s
#   at 1431 and 2.8 s at 5000; 0.005 s on raw and state runs. It is 8% of the
#   synthetic wall time.
# - disk: the n x n affinity and kernel files of pca and phate, capped at 16 MiB each:
#   43 MB per synthetic sample (3 seeds) at 775 points, 143 MB at 1431, about 26 GB for
#   all 1000 samples; graphs/ is 2 to 3 KB per graph.
#SBATCH --job-name=ssb_drive
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
set -euo pipefail
: "${GEOMANCER_GIT_SHA:?pass --export=ALL,GEOMANCER_GIT_SHA=<full sha>}"
# shellcheck source=jobs/ssb_grid_env.sh
source "$SCRATCH/geomancer-grid-$GEOMANCER_GIT_SHA/jobs/ssb_grid_env.sh"
SOURCE="${GEOMANCER_GRID_SOURCE:?set GEOMANCER_GRID_SOURCE to synthetic or real}"
: "${SLURM_ARRAY_TASK_ID:?this is a SLURM array job; pass --array=0-<n-1>}"
OUT="$GRID_ROOT/$SOURCE"
case "$SOURCE" in
  synthetic) OF="${GEOMANCER_GRID_OF:-100}" ;;
  real) OF="${GEOMANCER_GRID_OF:-79}" ;;
  *) echo "unknown GEOMANCER_GRID_SOURCE=$SOURCE; expected synthetic or real" >&2; exit 1 ;;
esac
if [ "$SLURM_ARRAY_TASK_ID" -ge "$OF" ]; then
  echo "refused: array task $SLURM_ARRAY_TASK_ID is outside the $OF shards" >&2
  exit 1
fi
if [ ! -f "$INPUTS/$SOURCE/manifest.json" ]; then
  echo "refused: $INPUTS/$SOURCE/manifest.json is missing; run jobs/ssb_grid_convert.sbatch first" >&2
  exit 1
fi

run geomancer.learner.drive source="$SOURCE" inputs="$INPUTS/$SOURCE" out="$OUT" \
  shard="$SLURM_ARRAY_TASK_ID" of="$OF" seeds=0,1,2 \
  workers="$WORKERS" threads="$THREADS"
