#!/usr/bin/env python
"""Check a converted store against the original WRF files.

Reads the source files with xarray, a separate code path from the converter,
so a bug in one is unlikely to hide in the other. Checks:

  * every write unit finished, and hour_present matches the manifest
  * projection x/y rebuild the source's XLAT/XLONG, if it has them
  * at --samples random hours, every direct variable matches exactly
    (after destaggering), and every <VAR>_1h equals the difference of
    running totals where no reset happened

    python verify_zarr.py --config config.yaml --store surface_met --samples 50

Exits non-zero if any check fails, so it can gate deleting the NetCDF.
"""
import argparse
import sys

import numpy as np
import pyproj
import xarray as xr

import wrf_to_zarr as w


def raw(path, k, name):
    with xr.open_dataset(path, decode_times=False, mask_and_scale=False) as ds:
        return ds[name].isel({ds[name].dims[0]: k}).load(), dict(ds.attrs)


def raw_field(src, name, t, destagger):
    path, k = src.locate(name, t)
    da, _ = raw(path, k, name)
    da = da.astype("f4")
    if destagger:
        for d in [d for d in da.dims if d in w.STAGGERED]:
            n = da.sizes[d]
            da = 0.5 * (da.isel({d: slice(0, n - 1)}).variable + da.isel({d: slice(1, n)}).variable)
            da = xr.DataArray(da)
    return np.asarray(da.values)


def raw_total(cfg, src, name, t):
    path, k = src.locate(name, t)
    da, attrs = raw(path, k, name)
    v = da.values.astype("f8")
    if src.has(f"I_{name}"):
        key = "BUCKET_J" if "J" in da.attrs.get("units", "") else "BUCKET_MM"
        size = float(attrs.get(key, cfg.get(key.lower(), 0.0)) or 0.0)
        if size > 0:
            bpath, bk = src.locate(f"I_{name}", t)
            v = v + size * raw(bpath, bk, f"I_{name}")[0].values
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    cfg = w.load_config(a.config)
    src = w.Source(cfg, a.store)
    ds = xr.open_zarr(w.store_path(cfg, a.store), consolidated=True)
    do_ds = cfg["stores"][a.store].get("destagger", True)
    plan, _ = w.plan_variables(cfg, a.store, src)
    failures = []

    def check(ok, msg):
        print(("ok    " if ok else "FAIL  ") + msg)
        if not ok:
            failures.append(msg)

    # Grid: projection coordinates against WRF's own lat/lon, when the source has them.
    if ds.lat.attrs.get("source", "").startswith("XLAT"):
        crs = pyproj.CRS.from_wkt(ds["lambert_conformal"].attrs["crs_wkt"])
        inv = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = inv.transform(*np.meshgrid(ds.x.values, ds.y.values))
        err = max(np.abs(lat - ds.lat.values).max(), np.abs(lon - ds.lon.values).max())
        check(err < 0.01, f"grid: projection x/y rebuild XLAT/XLONG to {err:.2e} degrees (~{err * 111e3:.0f} m)")
    else:
        print("skip  grid: no XLAT/XLONG in the source; lat/lon were computed from the projection. "
              "Set grid_file to a wrfout or geo_em file to check them.")

    if a.store == "static":
        for out, _, v in plan:
            check(np.array_equal(ds[out].values, raw_field(src, v, src.index(v)[0][0], do_ds), equal_nan=True),
                  f"static {out} matches")
        sys.exit(1 if failures else 0)

    # Completeness.
    taxis = cfg["_time_axis"]
    svars = w.source_vars(plan)
    hp = ds["hour_present"].values
    expected = np.array([np.mean([src.locate(v, t) is not None for v in svars]) for t in taxis], "f4")
    check(not np.isnan(hp).any(), f"all hours converted ({int(np.isnan(hp).sum()):,} never written)")
    check(np.allclose(np.nan_to_num(hp, nan=-1), expected),
          f"hour_present matches the manifest ({int((expected == 1).sum()):,} of {len(taxis):,} hours complete)")

    # Values at random hours where this hour and the previous are complete.
    rng = np.random.default_rng(a.seed)
    full = expected == 1
    candidates = np.flatnonzero(full[1:] & full[:-1]) + 1
    picks = rng.choice(candidates, size=min(a.samples, len(candidates)), replace=False)
    bad = {}
    for i in sorted(picks):
        t, tp = taxis[i], taxis[i - 1]
        for out, kind, spec in plan:
            got = ds[out].isel(time=int(i)).values
            if kind == "direct":
                ok = np.array_equal(got, raw_field(src, spec, t, do_ds), equal_nan=True)
            else:
                parts = spec["sum_of"] if kind == "sum" else [spec]
                diffs = [raw_total(cfg, src, v, t) - raw_total(cfg, src, v, tp) for v in parts]
                if any((d < -1e-3).any() for d in diffs):
                    continue  # a reset hour; the converter's rule applies, nothing to compare
                ok = np.allclose(got, sum(np.maximum(d, 0) for d in diffs), atol=1e-3, rtol=1e-4)
            if not ok:
                bad.setdefault(out, []).append(str(t))
    for out, _, _ in plan:
        check(out not in bad, f"{out}: {len(picks)} sampled hours compared with source"
              + (f"; mismatches at {bad[out][:5]}" if out in bad else ""))
    print(f"\n{len(failures)} failed checks")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
