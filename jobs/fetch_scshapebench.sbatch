#!/bin/bash
# Download scShapeBench from Hugging Face into $MANYLATENTS_DATA/scshapebench.
# Full dataset is ~116 GB. Re-running resumes; finished files are skipped.
# Submit from a laptop checkout: shop sbatch mila jobs/fetch_scshapebench.sbatch --sync
#SBATCH --job-name=fetch_scshapebench
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
set -euo pipefail

export MANYLATENTS_DATA="${MANYLATENTS_DATA:-$SCRATCH/lrw/data}"
# Keep the Hugging Face cache off the home quota.
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
DEST="$MANYLATENTS_DATA/scshapebench"
mkdir -p "$DEST" outputs/slurm

uv run --no-project --with "huggingface_hub>=0.34" python - "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="scShape-Benchmark/scShapeBench",
    repo_type="dataset",
    local_dir=sys.argv[1],
    max_workers=4,
)
print(f"wrote {path}")
PY

du -sh "$DEST"
find "$DEST/data/scRNAseq" -name '*.h5ad' | wc -l
