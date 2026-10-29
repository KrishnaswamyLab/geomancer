#!/bin/bash
# Grid step 3 of 3: score the graphs after the runs (score.py), then join the rows and
# scores into one table per seed (table.py). Submit it with
# --dependency=afterany:<drive array job id>, so a failed shard is counted in the
# table (as missing or aborted cells) instead of blocking it. table.py counts the
# missing cells in config.json, and train.py refuses a table with missing cells
# unless allow_missing=true. To fill them, resubmit the drive array, then this job.
#
# Submit from the laptop (see jobs/ssb_grid_drive.sbatch for the whole sequence):
#   synthetic: shop sbatch mila jobs/ssb_grid_reduce.sbatch -- --dependency=afterany:<id> \
#                --export=ALL,GEOMANCER_GIT_SHA=<sha>,GEOMANCER_GRID_SOURCE=synthetic
#   real:      the same with GEOMANCER_GRID_SOURCE=real
#
# It reads the ledgers and writes only scores/ and tables/, so it is safe to rerun.
# score.py needs every embedding run's *_emb.npy under cells/ (the join check hashes
# it); do not delete cells/ before this job.
#
# Sizing: rewards.synthetic_graph took 0.28 s per graph on the laptop (775 points),
# so 72,000 graphs are about 5.6 CPU-hours: 16 CPUs -> 8 workers, under 1 h. Real is
# one classifier fit per (action, seed, outer fold): small, but rf,stats_svm,gnn.
#SBATCH --job-name=ssb_reduce
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
set -euo pipefail
: "${GEOMANCER_GIT_SHA:?pass --export=ALL,GEOMANCER_GIT_SHA=<full sha>}"
# shellcheck source=jobs/ssb_grid_env.sh
source "$SCRATCH/geomancer-grid-$GEOMANCER_GIT_SHA/jobs/ssb_grid_env.sh"
SOURCE="${GEOMANCER_GRID_SOURCE:?set GEOMANCER_GRID_SOURCE to synthetic or real}"
OUT="$GRID_ROOT/$SOURCE"

# Real reward: rf is the headline (the first classifier gives `reward`); stats_svm and gnn
# are recorded beside it as a robustness check (decided 2026-10-29). Synthetic scoring takes
# no classifier.
classifiers=rf
if [ "$SOURCE" = real ]; then classifiers=rf,stats_svm,gnn; fi
run geomancer.learner.score grid="$OUT" source="$SOURCE" inputs="$INPUTS/$SOURCE" \
  workers="$WORKERS" classifiers="$classifiers"
run geomancer.learner.table grid="$OUT" source="$SOURCE"
