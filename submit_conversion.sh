#!/bin/bash
# Initialise the stores and submit their conversion arrays, then a verify job
# for each that starts when its array finishes.
#   ./submit_conversion.sh                      # every store in config.yaml
#   ./submit_conversion.sh surface_met land_hydro
# Throttle with MAX_RUNNING (default 40 tasks at once per store). Borah's bsudfq QOS allows
# 384 CPUs per user, so with 2-CPU tasks keep the sum across running stores under 192.
# PARTITION=short sends the jobs to Borah's `short` partition instead (60 nodes, 7-day limit,
# no per-user CPU cap in the 2026-10-05 probe).
set -euo pipefail
cd "$(dirname "$0")"
source slurm/env.sh
CONFIG=${CONFIG:-config.yaml}
STORES=${*:-$(python -c "import yaml;print(' '.join(yaml.safe_load(open('$CONFIG'))['stores']))")}
for s in $STORES; do
  python wrf_to_zarr.py init --config "$CONFIG" --store "$s"
  [ "$s" = static ] && continue
  n=$(python wrf_to_zarr.py ntasks --config "$CONFIG" --store "$s")
  jid=$(sbatch --parsable ${PARTITION:+-p $PARTITION} --array=0-$((n-1))%${MAX_RUNNING:-40} --export=ALL,STORE="$s",CONFIG="$CONFIG" slurm/convert.sbatch)
  sbatch ${PARTITION:+-p $PARTITION} --dependency=afterany:"$jid" --export=ALL,STORE="$s",CONFIG="$CONFIG" slurm/verify.sbatch
  echo "$s: $n tasks in job $jid"
done
