#!/bin/bash
# End-to-end check of the pipeline on tiny synthetic WRF files (under a minute per layout).
# Run this on Borah before pointing the scripts at the real archive:
#   bash tests/run_synthetic_test.sh                 # both layouts
#   bash tests/run_synthetic_test.sh per-variable    # just Borah's layout
set -euo pipefail
export PYTHONWARNINGS=ignore
here=$(cd "$(dirname "$0")/.." && pwd)
layouts=${*:-wrfout per-variable}
for layout in $layouts; do
  work=$(mktemp -d)
  # Starts on 30 September, so the per-variable run crosses a month and a water-year boundary.
  python "$here/tests/make_synthetic_wrfout.py" "$work/wrf" --layout "$layout" --start 1986-09-30 --rate 3 >/dev/null
  python "$here/wrf_inventory.py" --root "$work/wrf" --out "$work/inv" --workers 2 >/dev/null
  sed -e "s#^inventory_dir:.*#inventory_dir: $work/inv#" -e "s#  root: .*#  root: $work/zarr#" \
      -e 's#start: "1986-10-01T00:00"#start: "1986-09-30T00:00"#' -e 's#end: "2017-09-30T23:00"#end: "1986-10-02T23:00"#' \
      -e "s#^include_paths:.*#include_paths: '.'#" -e "s#^grid_file:.*#grid_file: null#" \
      -e "s#    include_paths: wrf_allfield#    include_paths: '.'#" \
      -e 's/hours_per_task: 168/hours_per_task: 48/' -e 's/chunks: {time: 8760, y: 32, x: 32}/chunks: {time: 24, y: 8, x: 8}/' \
      "$here/config.yaml" > "$work/config.yaml"
  cd "$here"
  echo "== $layout"
  for s in static surface_met land_hydro atmos_3d; do
    python wrf_to_zarr.py init --config "$work/config.yaml" --store "$s" >/dev/null
    if [ "$s" != static ]; then
      n=$(python wrf_to_zarr.py ntasks --config "$work/config.yaml" --store "$s")
      for t in $(seq 0 $((n-1))); do python wrf_to_zarr.py write --config "$work/config.yaml" --store "$s" --task "$t" >/dev/null; done
    fi
    printf '%-12s ' "$s"; python verify_zarr.py --config "$work/config.yaml" --store "$s" --samples 30 | tail -1
  done
  python make_timeseries_store.py init --config "$work/config.yaml" >/dev/null
  n=$(python make_timeseries_store.py ntasks --config "$work/config.yaml")
  for t in $(seq 0 $((n-1))); do python make_timeseries_store.py write --config "$work/config.yaml" --task "$t" --band-rows 16 >/dev/null; done
  printf '%-12s ' benchmark
  python tools/benchmark_pw.py --config "$work/config.yaml" --ptop 5000 --stride 1 --samples 3 --workers 2 \
      --out "$work/bench" > "$work/bench.log" || { cat "$work/bench.log"; exit 1; }
  grep -c '^- ok' "$work/bench.log" | sed 's/$/ checks ok/'
  rm -rf "$work"
done
echo "synthetic pipeline test passed"
