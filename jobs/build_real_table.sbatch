#!/bin/bash
# Build the learner's real-data reward table on the 79 scShapeBench high_support datasets.
# Stage 1 (graphs): 24 actions on each dataset, 5000 cells per dataset.
# Stage 2 (rewards): out-of-fold probabilities of scShapeBench's fixed classifiers.
# Submit from a laptop checkout: shop sbatch mila jobs/build_real_table.sbatch --sync
# Re-run only stage 2: STAGES=rewards shop sbatch mila jobs/build_real_table.sbatch --sync
#
# Sizing (measured on a laptop at 5000 cells): diffmap peaks near 10 GB per worker and
# the 24 actions take 2-3 min per dataset, so 6 workers at 16 GB each finish in about 1 h.
#SBATCH --job-name=build_real_table
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
set -euo pipefail

export MANYLATENTS_DATA="${MANYLATENTS_DATA:-$SCRATCH/lrw/data}"
# Two BLAS threads per worker, so 6 workers fill the 12 CPUs.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
OUT=outputs/tables/real_v1
STAGES="${STAGES:-graphs rewards}"
mkdir -p "$OUT" outputs/slurm

# The scShapeBench loader is not in geomancer's lock: it is on the manylatents-omics
# branch of latent-reasoning-works/manylatents-omics#74, pinned here by commit.
# high_support labels are read from parquet, so pyarrow is added too.
LOADER="manylatents-omics @ git+https://github.com/latent-reasoning-works/manylatents-omics@61c6a63717d597ef92c7e27c557b297e92d956f1"

for stage in $STAGES; do
  uv run --extra learner --with "$LOADER" --with pyarrow \
    python -m geomancer.learner.table source=real stage="$stage" out="$OUT" \
    max_cells=5000 n_pcs=50 points=X_pca workers=6 classifiers=rf,stats_svm,gnn
done

wc -l "$OUT"/graphs.jsonl "$OUT"/states.jsonl
if [ -f "$OUT"/rewards.jsonl ]; then wc -l "$OUT"/rewards.jsonl; fi
