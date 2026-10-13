#!/bin/bash
# Check each water year's NetCDF against the stores; delete only with DELETE=1.
#   sbatch --array=1987-2017 slurm/retire.sbatch                     # dry run, all years
#   sbatch --array=1987 --export=ALL,DELETE=1 slurm/retire.sbatch    # delete one year
#   add DUPLICATES=1 to also delete copies identical to the file the stores used
# Reports go to retire/wy_<YEAR>.tsv.
#SBATCH --job-name=wrf-retire
#SBATCH --partition=bsudfq          # Borah's default CPU partition (48 cores, 192 GB per node)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/bsuhome/%u/idaho-wrf-zarr/logs/%x-%A_%a.out
set -euo pipefail
source "${WRF_ZARR_REPO:-$HOME/idaho-wrf-zarr}/slurm/env.sh"
python -u tools/retire_netcdf.py --config "${CONFIG:-config.yaml}" --water-year "${SLURM_ARRAY_TASK_ID}" \
    ${DELETE:+--delete} ${DUPLICATES:+--include-duplicates}
