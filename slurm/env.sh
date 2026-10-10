# Sourced by every job: activate the conda env made from environment.yml and
# move to the repo folder (default ~/idaho-wrf-zarr; set WRF_ZARR_REPO to move it).
# Finds conda wherever it lives:
#   1. $WRF_ZARR_CONDA_SH, if you set it to a conda.sh path
#   2. the conda you submitted from (sbatch passes $CONDA_EXE along)
#   3. ~/miniforge3, ~/miniconda3, ~/mambaforge or ~/anaconda3
#   4. Borah's conda/23.1.0 module
# Set WRF_ZARR_ENV to use an env with a different name.
_conda_sh=""
for c in "${WRF_ZARR_CONDA_SH:-}" \
         "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh}" \
         "$HOME"/{miniforge3,miniconda3,mambaforge,anaconda3}/etc/profile.d/conda.sh; do
  if [ -n "$c" ] && [ -f "$c" ]; then _conda_sh=$c; break; fi
done
if [ -n "$_conda_sh" ]; then
  source "$_conda_sh"
else
  module load conda/23.1.0
  eval "$(conda shell.bash hook)"
fi
conda activate "${WRF_ZARR_ENV:-wrf-zarr}"
echo "python: $(command -v python)"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the repo, wherever jobs were submitted from
mkdir -p logs
export HDF5_USE_FILE_LOCKING=FALSE   # avoids lock errors reading NetCDF over NFS (Borah's Isilon)
