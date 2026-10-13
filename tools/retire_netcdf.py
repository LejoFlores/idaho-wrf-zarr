#!/usr/bin/env python
"""Delete one water year of original NetCDF files, but only files whose contents
are confirmed to be in the Zarr stores. Dry run unless --delete is given.

A file is deletable when every variable in it feeds a store, and:
  * every hour the store took from this file was written (hour_present not NaN)
  * at --checks random hours from this file, the store equals the file:
    exactly for direct variables; as the hourly difference of running totals
    (bucket counters included) for accumulated ones. I_<VAR> bucket files only
    need their hours written, since the <VAR> check already uses them.

Kept, and listed in the report:
  * files with a variable no store holds (none expected in wrf_out)
  * spin-up duplicates: files whose hours were all taken from another file
    (WY1994/2006/2009 start on 16 September). Delete them with --include-duplicates.
  * unreadable files (they are not in the manifest; SWDOWN_2009-09.nc is one)
  * anything not matching include_paths: wrf_allfield, lev_interpolated, logs

Writes retire/wy_<YEAR>.tsv (path, GB, status, note). Exit status 1 if any file
failed a check.

    python tools/retire_netcdf.py --config config.yaml --water-year 1987            # dry run
    python tools/retire_netcdf.py --config config.yaml --water-year 1987 --delete   # delete
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_zarr as vz  # noqa: E402
import wrf_to_zarr as w  # noqa: E402

STORES = ("surface_met", "land_hydro", "atmos_3d")


def coverage(cfg):
    """{source var: [(store, out name, kind)]} and per-store (Source, dataset, destagger)."""
    cover, stores = {}, {}
    for s in STORES:
        src = w.Source(cfg, s)
        plan, _ = w.plan_variables(cfg, s, src)
        ds = xr.open_zarr(w.store_path(cfg, s), consolidated=True)
        # hour_present is read once here: in atmos_3d it spans ~11,000 shards, so
        # reading it per file made each file take minutes.
        stores[s] = (src, ds, cfg["stores"][s].get("destagger", True), ds["hour_present"].values)
        for out, kind, spec in plan:
            if kind != "sum":  # sums are rebuilt from parts that are stored on their own
                cover.setdefault(spec, []).append((s, out, kind))
    return cover, stores


def check_file(cfg, fid, path, names, cover, stores, taxis_ns, rng, n_checks):
    """Return (status, note)."""
    targets = []
    for v in names:
        base = v[2:] if v.startswith("I_") and v[2:] in cover else None
        if v not in cover and base is None:
            return "keep", f"{v} is in no store"
        targets += [(v, s, out, "bucket" if base else kind) for s, out, kind in cover[base or v]]

    any_chosen = False
    for v, s, out, kind in targets:
        src, ds, do_ds, hp = stores[s]
        ts, fids, _ = src.index(v)
        mine = ts[fids == fid]
        if not len(mine):
            continue
        any_chosen = True
        idx = np.searchsorted(taxis_ns, mine)
        inside = (idx < len(taxis_ns)) & (taxis_ns[np.minimum(idx, len(taxis_ns) - 1)] == mine)
        idx = idx[inside]
        if np.isnan(hp[idx]).any():
            return "FAIL", f"{s}: hours from this file were never written"
        if kind == "bucket" or not len(idx):
            continue
        for i in rng.choice(idx, min(n_checks, len(idx)), replace=False):
            t = pd.Timestamp(taxis_ns[i])
            got = ds[out].isel(time=int(i)).values
            if kind == "direct":
                if not np.array_equal(got, vz.raw_field(src, v, t, do_ds), equal_nan=True):
                    return "FAIL", f"{s}/{out} differs from the file at {t}"
            else:
                tp = pd.Timestamp(taxis_ns[i - 1]) if i > 0 else None
                if tp is None or src.locate(v, tp) is None:
                    continue
                d = vz.raw_total(cfg, src, v, t) - vz.raw_total(cfg, src, v, tp)
                if (d < -1e-3).any():
                    continue  # a reset hour
                if not np.allclose(got, np.maximum(d, 0), atol=1e-3, rtol=1e-4):
                    return "FAIL", f"{s}/{out} differs from the file at {t}"
    if not any_chosen:
        return "duplicate", "all hours taken from another file (spin-up overlap)"
    return "ok", ""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--water-year", type=int, required=True)
    p.add_argument("--checks", type=int, default=2, help="random hours compared per file")
    p.add_argument("--delete", action="store_true", help="actually delete the files that pass")
    p.add_argument("--include-duplicates", action="store_true", help="also delete spin-up duplicates")
    a = p.parse_args()

    cfg = w.load_config(a.config)
    src = w.Source(cfg)  # global include_paths: wrf_out only
    files = src.files[src.files["water_year"].astype(str) == str(a.water_year)]
    if files.empty:
        sys.exit(f"no files for water year {a.water_year}")
    cover, stores = coverage(cfg)
    taxis_ns = cfg["_time_axis"].as_unit("ns").asi8
    rng = np.random.default_rng(abs(a.water_year))

    rows = []
    for n, (fid, r) in enumerate(files.sort_values("path").iterrows(), 1):
        status, note = check_file(cfg, fid, r["path"], r["data_vars"].split(), cover, stores, taxis_ns,
                                  rng, a.checks)
        rows.append((r["path"], int(r["size_bytes"]) / 1e9, status, note))
        if status == "FAIL":
            print(f"FAIL  {r['path']}: {note}", flush=True)
        if n % 100 == 0:
            print(f"  checked {n:,} of {len(files):,}", flush=True)

    rep = pd.DataFrame(rows, columns=["path", "GB", "status", "note"])
    out = Path("retire")
    out.mkdir(exist_ok=True)
    rep.to_csv(out / f"wy_{a.water_year}.tsv", sep="\t", index=False)
    summary = rep.groupby("status")["GB"].agg(["count", "sum"])
    print(f"\nwater year {a.water_year}:\n{summary.to_string(float_format=lambda x: f'{x:,.1f}')}")

    failed = (rep["status"] == "FAIL").any()
    go = ["ok"] + (["duplicate"] if a.include_duplicates else [])
    victims = rep[rep["status"].isin(go)]
    if not a.delete:
        print(f"\ndry run: would delete {len(victims):,} files, {victims['GB'].sum():,.1f} GB. "
              f"Report in {out}/wy_{a.water_year}.tsv")
    elif failed:
        print("\nnot deleting anything: some files failed a check (see the report)")
    else:
        for path in victims["path"]:
            os.remove(path)
        print(f"\ndeleted {len(victims):,} files, {victims['GB'].sum():,.1f} GB")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
