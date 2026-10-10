#!/usr/bin/env python
"""Print the few facts config.yaml still needs that inventory_report.md leaves out.

    python tools/inventory_details.py [inventory]
"""
import json
import sys
from pathlib import Path

import netCDF4
import pandas as pd

inv = Path(sys.argv[1] if len(sys.argv) > 1 else "inventory")
s = json.loads((inv / "summary.json").read_text())
g = s["d02"]["globals"]
print("BUCKET_MM:", g.get("BUCKET_MM"), " BUCKET_J:", g.get("BUCKET_J"))
print("unreadable files:", s["_files_with_errors"] or "none")

f = pd.read_csv(inv / "manifest_files.csv", keep_default_na=False)
names = f["data_vars"].str.split()
full = f.loc[names.apply(lambda v: "HGT" in v), "path"].tolist()
print("files holding HGT (complete wrfout):", full)

for var in ("U", "W", "RAINNC", "I_RAINNC"):
    path = f.loc[names.apply(lambda v: var in v), "path"].iloc[0]
    with netCDF4.Dataset(path) as nc:
        v = nc[var]
        print(f"{var}: {path}\n   dims {dict(zip(v.dimensions, v.shape))} dtype {v.dtype} "
              f"BUCKET_MM={getattr(nc, 'BUCKET_MM', None)} BUCKET_J={getattr(nc, 'BUCKET_J', None)}")
