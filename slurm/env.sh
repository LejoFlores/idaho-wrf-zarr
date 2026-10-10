# Sourced by every job: activate the conda env made from environment.yml and
# move to the repo folder. Uses your own miniconda (/bsuhome/$USER/miniconda3)
# if present, else Borah's conda module.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  module load conda/23.1.0
  eval "$(conda shell.bash hook)"
fi
conda activate "${WRF_ZARR_ENV:-wrf-zarr}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
mkdir -p logs
export HDF5_USE_FILE_LOCKING=FALSE   # avoids lock errors reading NetCDF over NFS (Borah's Isilon)
