#!/bin/bash
# Copy a Zarr store between file systems with parallel rsync, then check the copy.
# Does not delete the source; do that yourself after the check passes.
#   sbatch --export=ALL,SRC=/bsuscratch/aflores/idaho-wrf-1km-zarr/atmos_3d.zarr,DST=/bsushare/leaf-shared/idaho-wrf-1km-zarr/ slurm/move_store.sbatch
#SBATCH --job-name=wrf-move
#SBATCH --partition=bsudfq          # Borah's default CPU partition (48 cores, 192 GB per node)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/bsuhome/%u/idaho-wrf-zarr/logs/%x-%j.out
set -euo pipefail
: "${SRC:?set SRC to the .zarr folder}" "${DST:?set DST to the folder that will hold it}"
name=$(basename "$SRC")
mkdir -p "$DST/$name"
echo "$(date '+%F %T') copying $SRC -> $DST/$name"
# Small metadata first, then one rsync per array folder, ${PARALLEL:-12} at a time.
rsync -a --exclude='*/' "$SRC/" "$DST/$name/"
find "$SRC" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
  xargs -P "${PARALLEL:-12}" -I{} sh -c 'rsync -a "$0/{}/" "$1/{}/" && echo "$(date "+%T") done {}"' "$SRC" "$DST/$name"
rsync -a "$SRC/" "$DST/$name/"   # catch-all pass: anything missed, plus folder timestamps
echo "$(date '+%F %T') checking"
left=$(rsync -an --itemize-changes "$SRC/" "$DST/$name/" | wc -l)
n_src=$(find "$SRC" -type f | wc -l); n_dst=$(find "$DST/$name" -type f | wc -l)
b_src=$(find "$SRC" -type f -printf '%s\n' | awk '{s+=$1} END {print s}')
b_dst=$(find "$DST/$name" -type f -printf '%s\n' | awk '{s+=$1} END {print s}')
echo "files: $n_src source, $n_dst copy; bytes: $b_src source, $b_dst copy; rsync differences left: $left"
if [ "$left" -eq 0 ] && [ "$n_src" -eq "$n_dst" ] && [ "$b_src" -eq "$b_dst" ]; then
  echo "COPY OK"
else
  echo "COPY INCOMPLETE"; exit 1
fi
