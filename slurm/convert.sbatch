#!/bin/bash
# One array task writes one block of time for one store. Submit via submit_conversion.sh,
# or by hand after `python wrf_to_zarr.py init`:
#   N=$(python wrf_to_zarr.py ntasks --config config.yaml --store surface_met)
#   sbatch --array=0-$((N-1))%40 --export=ALL,STORE=surface_met slurm/convert.sbatch
#SBATCH --job-name=wrf-zarr
#SBATCH --partition=bsudfq          # Borah's default CPU partition (48 cores, 192 GB per node)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/bsuhome/%u/idaho-wrf-zarr/logs/%x-%A_%a.out
set -euo pipefail
source "${WRF_ZARR_REPO:-$HOME/idaho-wrf-zarr}/slurm/env.sh"
python wrf_to_zarr.py write --config "${CONFIG:-config.yaml}" --store "${STORE:?set STORE}"
