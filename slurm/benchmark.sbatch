#!/bin/bash
# Functional test and read benchmark: seasonal cycle of precipitable water from
# atmos_3d + surface_met, written as a PNG. Output goes to bench/<jobid>/.
#   sbatch slurm/benchmark.sbatch
#   sbatch --export=ALL,EXTRA="--start 2000-10-01 --end 2001-09-30" slurm/benchmark.sbatch   # one year
#SBATCH --job-name=wrf-bench
#SBATCH --partition=bsudfq          # Borah's default CPU partition (48 cores, 192 GB per node)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/bsuhome/%u/idaho-wrf-zarr/logs/%x-%j.out
set -euo pipefail
source "${WRF_ZARR_REPO:-$HOME/idaho-wrf-zarr}/slurm/env.sh"
python -u tools/benchmark_pw.py --config "${CONFIG:-config.yaml}" --stride "${STRIDE:-6}" \
    --out "bench/${SLURM_JOB_ID}" ${EXTRA:-}
