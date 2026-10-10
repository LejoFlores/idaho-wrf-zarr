# Sourced by every job: load Python and move to the scripts folder.
# Edit to match how you get Python on Borah, e.g. a conda module and the env
# made from environment.yml.
module load conda 2>/dev/null || true
conda activate wrf-zarr 2>/dev/null || source activate wrf-zarr
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
mkdir -p logs
export HDF5_USE_FILE_LOCKING=FALSE   # avoids lock errors reading NetCDF on parallel file systems
