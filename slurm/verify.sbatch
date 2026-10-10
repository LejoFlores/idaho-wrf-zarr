#!/bin/bash
#   sbatch --export=ALL,STORE=surface_met slurm/verify.sbatch
#SBATCH --job-name=wrf-verify
#SBATCH --partition=bsudfq          # Borah's default CPU partition (48 cores, 192 GB per node)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/bsuhome/%u/idaho-wrf-zarr/logs/%x-%j.out
set -euo pipefail
source "${WRF_ZARR_REPO:-$HOME/idaho-wrf-zarr}/slurm/env.sh"
python wrf_to_zarr.py status --config "${CONFIG:-config.yaml}" --store "${STORE:?set STORE}"
python verify_zarr.py --config "${CONFIG:-config.yaml}" --store "$STORE" --samples "${SAMPLES:-100}"
