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
# Clone next to the data so config.yaml's default paths work:
cd /bsushare/leaf-shared && git clone git@github.com:LejoFlores/idaho-wrf-zarr.git && cd idaho-wrf-zarr
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
```

Submit jobs from the repo folder; they find `slurm/env.sh` through `$SLURM_SUBMIT_DIR`.
`slurm/env.sh` activates the `wrf-zarr` env from `~/miniconda3`, or from the
`conda/23.1.0` module if that's missing. Jobs run on `bsudfq`; its QOS caps a user
at 384 CPUs, which `MAX_RUNNING` in `submit_conversion.sh` respects.

## Files

| File | What it does |
| --- | --- |
| `wrf_inventory.py` | Lists every file, output time, variable, gap and duplicate; records grid, projection, WRF version and physics options |
| `wrf_to_zarr.py` | `init`, `write`, `status` for each store in `config.yaml` |
| `verify_zarr.py` | Compares a store with the NetCDF through a separate code path; non-zero exit on any mismatch |
| `make_timeseries_store.py` | Rechunked copy of key variables for point time series |
| `config.yaml` | Stores, variables, chunking, paths |
| `submit_conversion.sh`, `slurm/` | SLURM jobs |
| `tests/` | Synthetic wrfout generator and the end-to-end test |

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
