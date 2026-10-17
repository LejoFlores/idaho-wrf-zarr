#!/usr/bin/env python
"""Check that the store the config points at is identical to another copy of it,
e.g. after moving it between file systems. Compares variables, shapes, hour_present
for every hour, and every variable bit for bit at --samples random complete hours.

    python tools/compare_stores.py --config config.yaml --store atmos_3d \
        --other /bsuscratch/aflores/idaho-wrf-1km-zarr/atmos_3d.zarr

(verify_zarr.py can't do this once the NetCDF originals are retired.)
Exit status 1 if anything differs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wrf_to_zarr as w  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--config", default="config.yaml")
p.add_argument("--store", required=True)
p.add_argument("--other", required=True, help="path of the other copy")
p.add_argument("--samples", type=int, default=20)
a = p.parse_args()

cfg = w.load_config(a.config)
here = w.store_path(cfg, a.store)
print(f"config store: {here}\nother copy:   {a.other}")
x = xr.open_zarr(here, consolidated=True)
y = xr.open_zarr(a.other, consolidated=True)
bad = []
if set(x.variables) != set(y.variables):
    bad.append(f"variables differ: {sorted(set(x.variables) ^ set(y.variables))}")
for v in sorted(set(x.variables) & set(y.variables)):
    if x[v].shape != y[v].shape:
        bad.append(f"{v}: shape {x[v].shape} vs {y[v].shape}")
hx, hy = x["hour_present"].values, y["hour_present"].values
if not np.array_equal(hx, hy, equal_nan=True):
    bad.append(f"hour_present differs at {int((~((hx == hy) | (np.isnan(hx) & np.isnan(hy)))).sum())} hours")
print(f"{x.sizes['time']:,} hours, {int((hx == 1).sum()):,} complete")
names = [v for v in x.data_vars if "time" in x[v].dims and v != "hour_present"]
hours = np.random.default_rng(1).choice(np.flatnonzero(hx == 1), min(a.samples, int((hx == 1).sum())), replace=False)
for n, i in enumerate(sorted(hours), 1):
    for v in names:
        if not np.array_equal(x[v].isel(time=int(i)).values, y[v].isel(time=int(i)).values, equal_nan=True):
            bad.append(f"{v} differs at {x.time.values[i]}")
    print(f"  hour {n} of {len(hours)} ({str(x.time.values[i])[:13]}): {len(names)} variables compared", flush=True)
print("\n" + ("\n".join(bad) if bad else f"IDENTICAL: metadata, hour_present and {len(hours)} random hours of every variable"))
sys.exit(1 if bad else 0)
