#!/usr/bin/env python
"""Compare every surface variable in lev_interpolated/ with the same month in vol*/wrf_out/.

    python tools/compare_trees.py [inventory]

Before include_paths was anchored, the surface stores read these WY2000 months from
lev_interpolated/. If every line says "identical", those stores need no rework.
"""
import re
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

inv = Path(sys.argv[1] if len(sys.argv) > 1 else "inventory")
f = pd.read_csv(inv / "manifest_files.csv", keep_default_na=False)
f = f[f["error"] == ""]
interp = f[f["path"].str.contains("/lev_interpolated/")]
main = f[f["path"].str.contains(r"/subset-output-wrf/vol\d+/wrf_out/", regex=True)]
bad = 0
for _, r in interp.iterrows():
    month = re.search(r"(\d{4}-\d{2})", r["path"]).group(1)
    for var in r["data_vars"].split():
        hit = main[main["path"].str.endswith(f"/{var}_{month}.nc")]
        if hit.empty:
            continue
        with netCDF4.Dataset(r["path"]) as a, netCDF4.Dataset(hit["path"].iloc[0]) as b:
            if a[var].ndim != 3:
                continue  # 3D fields are on different levels by design
            x, y = np.asarray(a[var][:], "f8"), np.asarray(b[var][:], "f8")
            if x.shape != y.shape:
                print(f"{var} {month}: shapes differ {x.shape} vs {y.shape}"); bad += 1; continue
            d = float(np.nanmax(np.abs(x - y)))
            print(f"{var} {month}: {'identical' if d == 0 else f'max |difference| {d:.3g}'}")
            bad += d != 0
print(f"\n{bad} variable-months differ")
