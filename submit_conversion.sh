#!/bin/bash
# Initialise the stores and submit their conversion arrays, then a verify job
# for each that starts when its array finishes.
#   ./submit_conversion.sh                      # every store in config.yaml
#   ./submit_conversion.sh surface_met land_hydro
# Throttle with MAX_RUNNING (default 40 tasks at once per store).
set -euo pipefail
cd "$(dirname "$0")"
source slurm/env.sh
CONFIG=${CONFIG:-config.yaml}
STORES=${*:-$(python -c "import yaml;print(' '.join(yaml.safe_load(open('$CONFIG'))['stores']))")}
for s in $STORES; do
  python wrf_to_zarr.py init --config "$CONFIG" --store "$s"
  [ "$s" = static ] && continue
  n=$(python wrf_to_zarr.py ntasks --config "$CONFIG" --store "$s")
  jid=$(sbatch --parsable --array=0-$((n-1))%${MAX_RUNNING:-40} --export=ALL,STORE="$s",CONFIG="$CONFIG" slurm/convert.sbatch)
  sbatch --dependency=afterany:"$jid" --export=ALL,STORE="$s",CONFIG="$CONFIG" slurm/verify.sbatch
  echo "$s: $n tasks in job $jid"
done
