#!/bin/bash
# How much smaller would a store be with bitrounding? Reads only. Output goes to bitround/<jobid>/.
#   sbatch slurm/bitround.sbatch
#   sbatch --export=ALL,EXTRA="--hours 48 --start 2010-10-01 --end 2011-09-30" slurm/bitround.sbatch
#SBATCH --job-name=wrf-bitround
#SBATCH --partition=bsudfq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/bsuhome/%u/idaho-wrf-zarr/logs/%x-%j.out
set -euo pipefail
source "${WRF_ZARR_REPO:-$HOME/idaho-wrf-zarr}/slurm/env.sh"
python -u tools/bitround_test.py --config "${CONFIG:-config.yaml}" --out "bitround/${SLURM_JOB_ID}" ${EXTRA:-}
