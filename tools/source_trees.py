#!/usr/bin/env python
"""Summarise the separate folder trees the inventory found, so we can pick which feed the stores.

    python tools/source_trees.py [inventory]

For each tree (vol*/wrf_out, lev_interpolated/..., wrf_allfield, ...): file count, size,
water years and variables; the dimensions of the 3D variables in each; and, where two
trees hold the same variable for the same month, whether their values agree.
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
root = "/bsushare/leaf-shared/subset-output-wrf/"
rel = f["path"].str.replace(root, "", regex=False)
f["tree"] = rel.str.replace(r"vol\d+", "volNN", regex=True).str.replace(r"/wy_\d{4}.*|/[^/]+$", "", regex=True)
f["vars"] = f["data_vars"].str.split()

print("== trees")
for tree, g in f.groupby("tree"):
    wys = sorted({int(w) for w in g["water_year"] if str(w).isdigit()})
    names = sorted(set().union(*g["vars"]))
    print(f"{tree}: {len(g):,} files, {g['size_bytes'].sum() / 1e12:.2f} TB, "
          f"WY {wys[0] if wys else '-'}-{wys[-1] if wys else '-'} ({len(wys)} years), {len(names)} variables")
    print("   ", " ".join(names[:80]) + (" ..." if len(names) > 80 else ""))

print("\n== 3D variable dimensions, one file per tree")
for var in ("U", "V", "T", "QVAPOR", "W", "PH", "CLDFRA", "QCLOUD"):
    for tree, g in f.groupby("tree"):
        hit = g[g["vars"].apply(lambda v: var in v)]
        if hit.empty:
            continue
        path = hit["path"].iloc[0]
        with netCDF4.Dataset(path) as nc:
            v = nc[var]
            extra = ""
            for d in v.dimensions:
                if d in nc.variables and d not in ("Time", "time"):
                    vals = np.asarray(nc[d][:]).ravel()
                    extra += f" {d}={vals[:3].tolist()}..{vals[-2:].tolist()} {getattr(nc[d], 'units', '')}"
            print(f"{var:7s} {tree}: {dict(zip(v.dimensions, v.shape))}{extra}")

print("\n== same variable and month in two trees: do the values agree?")
f["month"] = rel.str.extract(r"(\d{4}-\d{2})")[0]
for var in ("RAINNC", "T2", "SNOW", "U"):
    rows = f[f["vars"].apply(lambda v: var in v)].dropna(subset=["month"])
    for month, g in rows.groupby("month"):
        if g["tree"].nunique() < 2:
            continue
        a, b = g.drop_duplicates("tree").head(2)["path"]
        with netCDF4.Dataset(a) as na, netCDF4.Dataset(b) as nb:
            xa, xb = na[var], nb[var]
            if xa.shape[1:] != xb.shape[1:]:
                print(f"{var} {month}: shapes differ {xa.shape} vs {xb.shape}")
            else:
                d = float(np.nanmax(np.abs(np.asarray(xa[0], "f8") - np.asarray(xb[0], "f8"))))
                print(f"{var} {month}: first hour max |difference| {d:.3g}  ({len(xa)} vs {len(xb)} hours)")
            print(f"   {a}\n   {b}")
        break
