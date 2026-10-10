#!/bin/bash
# Catalog every wrfout file. Headers only, so one node is plenty.
#   sbatch slurm/inventory.sbatch   # defaults to all 12 volumes under /bsushare/leaf-shared/subset-output-wrf
#SBATCH --job-name=wrf-inventory
#SBATCH --nodes=1
#SBATCH --partition=bsudfq          # Borah's default CPU partition (48 cores, 192 GB per node)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?submit from the repo folder}/slurm/env.sh"
ROOT=${1:-/bsushare/leaf-shared/subset-output-wrf}
# d02 only: the d01 folders in vol04 are the parent domain.
python wrf_inventory.py --root "$ROOT" --domain d02 --out inventory --workers "$SLURM_CPUS_PER_TASK"
