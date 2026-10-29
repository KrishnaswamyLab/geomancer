#!/bin/bash
# Grid step 1 of 3: convert the inputs once, to hashed .npy files with a manifest.
# Writes $SCRATCH/geomancer/outputs/grid/<sha12>/inputs/{synthetic,real}/ (see
# jobs/ssb_grid_env.sh). Synthetic is a copy of noisy_points (seconds); real is a
# seeded subsample to 5000 cells and one PCA for each of the 79 high_support datasets.
# A dataset that is not on disk or that fails is an entry in the manifest, counted.
#
# Submit from the laptop (see jobs/ssb_grid_drive.sbatch for the whole sequence):
#   shop sbatch mila jobs/ssb_grid_convert.sbatch -- --export=ALL,GEOMANCER_GIT_SHA=<sha>
#
# Rerunning rewrites the manifests, and drive.py checks every .npy against them, so
# do not rerun it while a drive array of the same commit is running.
#SBATCH --job-name=ssb_convert
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
set -euo pipefail
: "${GEOMANCER_GIT_SHA:?pass --export=ALL,GEOMANCER_GIT_SHA=<full sha>}"
# shellcheck source=jobs/ssb_grid_env.sh
source "$SCRATCH/geomancer-grid-$GEOMANCER_GIT_SHA/jobs/ssb_grid_env.sh"

run geomancer.learner.convert source="${GEOMANCER_GRID_SOURCE:-all}" out="$INPUTS" workers="$WORKERS"
