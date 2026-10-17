# Idaho 1 km WRF to Zarr

Scripts to catalog the Idaho 1 km WRF output on Borah and convert it to
CF-described Zarr stores for AI training and evaluation. The storage
recommendation behind the layout is in the Claude Doc "WRF data storage plan".

The source on Borah is one NetCDF file per variable per month,
`/bsushare/leaf-shared/subset-output-wrf/volXX/wrf_out/wy_YYYY/d02/VAR_YYYY-MM.nc`,
WY1987-2017 across 12 volumes, about 136 TiB. The scripts also read classic
wrfout files.

Tested end to end on small synthetic files in both layouts (`tests/`). They have
not yet run on the real archive, so expect to adjust variable lists and SLURM
settings after the inventory.

## Steps

```bash
# Clone into your home folder; jobs, logs and the inventory live here:
cd /bsuhome/aflores && git clone git@github.com:LejoFlores/idaho-wrf-zarr.git && cd idaho-wrf-zarr
dev-session    # Borah: build envs on a compute node; the login node has a newer GLIBC
conda env create -f environment.yml && conda activate wrf-zarr
bash tests/run_synthetic_test.sh            # under two minutes; checks the install

# 1. Inventory (headers only). Send back inventory/inventory_report.md.
#    Its water-year table shows which volume holds each WY and whether it is complete.
sbatch slurm/inventory.sbatch     # scans d02 in vol01-vol12 under /bsushare/leaf-shared/subset-output-wrf

# 2. Edit config.yaml: variable lists (check inventory/variables.csv), and grid_file if the
#    report says no files carry projection attributes.

# 3. Convert. Initialises each store, submits one array per store, then a verify job.
./submit_conversion.sh                      # or: ./submit_conversion.sh surface_met
python wrf_to_zarr.py status --config config.yaml --store surface_met   # what's left

# 4. After surface_met and land_hydro verify: the time-series copy.
python make_timeseries_store.py init --config config.yaml
sbatch --array=0-$(( $(python make_timeseries_store.py ntasks --config config.yaml) - 1 )) slurm/timeseries.sbatch

# 5. Functional test and read benchmark (needs matplotlib): seasonal cycle of precipitable
#    water from atmos_3d, as a PNG plus timings, in bench/<jobid>/.
sbatch slurm/benchmark.sbatch

# 6. Retire the NetCDF one water year at a time. Dry run first; it re-checks every file
#    against the stores and deletes only files that pass (report in retire/wy_<YEAR>.tsv).
sbatch --array=1987-2017 slurm/retire.sbatch                     # dry run
sbatch --array=1987 --export=ALL,DELETE=1 slurm/retire.sbatch    # delete one year
```

Jobs run from `~/idaho-wrf-zarr` wherever you submit them, and write logs to `~/idaho-wrf-zarr/logs/`.
The inventory goes to `~/idaho-wrf-zarr/inventory/`; the Zarr stores go to `output.root` in
`config.yaml`, on `/bsushare` next to the data. To keep the repo elsewhere, set `WRF_ZARR_REPO`
and change the `--output` lines in `slurm/*.sbatch`.
`slurm/env.sh` activates the `wrf-zarr` env using the conda you submit from (or
`$WRF_ZARR_CONDA_SH`, a few usual install folders, then the `conda/23.1.0` module);
each job log prints which python it got. Jobs run on `bsudfq`; its QOS caps a user
at 384 CPUs, which `MAX_RUNNING` in `submit_conversion.sh` respects. `PARTITION=short
./submit_conversion.sh` uses the `short` partition instead, which has no per-user cap.

`tools/probe_borah.sh` reports conda, storage on `/bsushare`, partitions and GPUs, and the
variables, sizes and headers in one water year of the archive. It is read-only:
`bash tools/probe_borah.sh > probe_$(hostname -s)_$(date +%Y%m%d).txt 2>&1`.

## Files

| File | What it does |
| --- | --- |
| `wrf_inventory.py` | Lists every file, output time, variable, gap and duplicate; records grid, projection, WRF version and physics options |
| `wrf_to_zarr.py` | `init`, `write`, `status` for each store in `config.yaml` |
| `verify_zarr.py` | Compares a store with the NetCDF through a separate code path; non-zero exit on any mismatch |
| `make_timeseries_store.py` | Rechunked copy of key variables for point time series |
| `config.yaml` | Stores, variables, chunking, paths |
| `submit_conversion.sh`, `slurm/` | SLURM jobs |
| `tests/` | Synthetic WRF file generator (both layouts) and the end-to-end test |
| `tools/benchmark_pw.py` | Precipitable-water seasonal cycle from the stores: checks against the NetCDF, read timings, PNG |
| `tools/bitround_test.py` | What bitrounding (lossy, numcodecs BitRound) would save on a store: bit information per variable, sizes and errors at keepbits 1-23, PW error. Reads only. `sbatch slurm/bitround.sbatch` |
| `tools/compare_stores.py` | Check a moved store is identical to the original copy (metadata, hour_present, random hours bit for bit) |
| `examples/parallel_zarr_pw.ipynb` | Example: reading the stores in parallel with Dask workers (PW seasonal cycle, scaling test, access patterns) |
| `tools/retire_netcdf.py` | Deletes a water year's NetCDF only where the stores hold the same data; dry run by default |
| `tools/probe_borah.sh` | Read-only report on Borah and one water year of the archive |

## What the conversion does

- One store per group: `static`, `surface_met`, `land_hydro`, `atmos_3d`, plus `surface_timeseries`.
- A regular hourly time axis from the config's start to end. Hours with no file stay NaN, and
  `hour_present` is 0 there, so training samplers can skip them.
- Dimensions renamed to `time, level, soil_layer, y, x`; projection `x`/`y` in metres with a
  `lambert_conformal` grid mapping, plus 2D `lat`/`lon`.
- Staggered variables averaged onto the mass grid.
- Accumulations (with `I_<VAR>` buckets) become hourly amounts named `<VAR>_1h`; `PRECIP` is
  their sum. A drop in a running total is read as a reset (a restart), and that hour takes the
  new total. The hour after a gap is NaN.
- Winds stay grid-relative (`x_wind`, `y_wind`). `COSALPHA`/`SINALPHA` in `static` rotate them.
- Lossless Blosc/zstd compression; Zarr v3; `atmos_3d` is sharded by day to keep file counts down.

Each write unit leaves a marker in `<output root>/_progress/<store>/`, so a failed or timed-out
task can be resubmitted and skips what it already wrote. Tasks write whole chunks (or shards) and
never share one, so they can run in parallel safely.

## Known limits

- Only Lambert conformal (`MAP_PROJ=1`) is handled.
- If output is not hourly, edit `time:`.
- If the per-variable files carry no WRF global attributes, set `grid_file` to a wrfout, wrfinput
  or geo_em file for d02. Without XLAT/XLONG anywhere, lat/lon are computed from the projection.
- Duplicate hours (restart overlaps) keep the first file's copy; set `duplicate_times: last` to flip it.
