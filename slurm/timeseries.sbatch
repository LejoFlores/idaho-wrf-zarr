#!/bin/bash
# After surface_met and land_hydro are written and verified:
#   python make_timeseries_store.py init --config config.yaml
#   N=$(python make_timeseries_store.py ntasks --config config.yaml)
#   sbatch --array=0-$((N-1)) slurm/timeseries.sbatch
#SBATCH --job-name=wrf-timeseries
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?submit from the repo folder}/slurm/env.sh"
python make_timeseries_store.py write --config "${CONFIG:-config.yaml}" --band-rows "${BAND_ROWS:-128}"
