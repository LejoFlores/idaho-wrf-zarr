#!/usr/bin/env python
"""Build a copy of key variables chunked for long time series at points.

The training stores hold a day of whole maps per chunk, so reading 32 years
at one SNOTEL site touches about 11,700 chunks. This copy holds about a year
of a 32 x 32 km tile per chunk, so the same read touches 32.

    python make_timeseries_store.py init  --config config.yaml
    python make_timeseries_store.py ntasks --config config.yaml
    python make_timeseries_store.py write --config config.yaml --task 0

Each task writes one time chunk (about a year) for every variable, in bands
of --band-rows rows to bound memory: a band holds time_chunk x band_rows x nx
float32 values per variable (8760 x 128 x 700 is about 3 GB).
"""
import argparse
import os
import sys

import dask.array as da
import numpy as np
import xarray as xr

import wrf_to_zarr as w


def sources(cfg):
    out = {}
    for store, names in cfg["timeseries"]["from"].items():
        ds = xr.open_zarr(w.store_path(cfg, store), consolidated=True)
        missing = [n for n in names if n not in ds]
        if missing:
            w.warnings.warn(f"{store}: not in the store, skipped: {missing}")
        out[store] = ds[[n for n in names if n in ds]]
    return out


def cmd_init(cfg, overwrite):
    ts = cfg["timeseries"]
    path = w.store_path(cfg, ts["name"])
    if path.exists() and not overwrite:
        sys.exit(f"{path} exists; pass --overwrite to replace it")
    srcs = sources(cfg)
    first = next(iter(srcs.values()))
    c = ts["chunks"]
    data, enc = {}, {}
    for store, ds in srcs.items():
        for v in ds.data_vars:
            if v == "lambert_conformal":
                continue
            shape = ds[v].shape
            chunks = (c["time"], c["y"], c["x"])
            data[v] = (ds[v].dims, da.full(shape, np.nan, dtype="f4", chunks=chunks),
                       {**ds[v].attrs, "source_store": store})
            enc[v] = {"chunks": chunks, "compressors": [w.compressor(cfg)]}
    coords = {k: first.coords[k] for k in ("time", "y", "x", "lat", "lon") if k in first.coords}
    out = xr.Dataset(data, coords=coords)
    out["lambert_conformal"] = xr.open_zarr(w.store_path(cfg, next(iter(srcs))))["lambert_conformal"].load()
    out.attrs = {**first.attrs, "store": ts["name"], "comment": "rechunked copy for time series reads"}
    enc["time"] = {k: first.time.encoding[k] for k in ("units", "calendar", "dtype") if k in first.time.encoding}
    for k in ("lat", "lon"):
        if k in out.coords:
            enc[k] = {"chunks": (c["y"], c["x"])}
    out.to_zarr(path, mode="w", compute=False, zarr_format=cfg["output"]["zarr_format"],
                consolidated=True, encoding=enc)
    print(f"initialised {path}: {len(data)} variables, {n_tasks(cfg):,} tasks")


def n_tasks(cfg):
    return -(-len(cfg["_time_axis"]) // int(cfg["timeseries"]["chunks"]["time"]))


def cmd_write(cfg, task, band_rows):
    ts = cfg["timeseries"]
    path = w.store_path(cfg, ts["name"])
    T, cy = int(ts["chunks"]["time"]), int(ts["chunks"]["y"])
    if band_rows % cy:
        sys.exit(f"--band-rows must be a multiple of the y chunk ({cy})")
    nt = len(cfg["_time_axis"])
    t0, t1 = task * T, min((task + 1) * T, nt)
    for store, ds in sources(cfg).items():
        ny = ds.sizes["y"]
        for v in ds.data_vars:
            for y0 in range(0, ny, band_rows):
                y1 = min(y0 + band_rows, ny)
                block = ds[v].isel(time=slice(t0, t1), y=slice(y0, y1)).values
                xr.Dataset({v: (ds[v].dims, block)}).to_zarr(
                    path, region={"time": slice(t0, t1), "y": slice(y0, y1)}, mode="r+", consolidated=False)
            print(f"task {task}: {store}/{v} hours {t0}-{t1 - 1}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["init", "ntasks", "write"])
    p.add_argument("--config", required=True)
    p.add_argument("--task", type=int)
    p.add_argument("--band-rows", type=int, default=128)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    cfg = w.load_config(a.config)
    if a.command == "init":
        cmd_init(cfg, a.overwrite)
    elif a.command == "ntasks":
        print(n_tasks(cfg))
    else:
        task = a.task if a.task is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        cmd_write(cfg, task, a.band_rows)


if __name__ == "__main__":
    main()
