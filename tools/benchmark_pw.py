#!/usr/bin/env python
"""Functional test and read benchmark of the Zarr stores: the seasonal cycle of
precipitable water (PW), written to a PNG.

PW is the column integral of water vapour, PW = (1/g) * integral of q dp. The
archive has no P or PB, but WRF's eta coordinate gives each layer's share of
the dry-air column, -DNW = ZNW[k] - ZNW[k+1], and the surface pressure fixes
the column mass:

    PSFC - P_TOP = MU_d * sum_k (-DNW_k)(1 + QVAPOR_k)     (condensate ignored)
    PW           = MU_d * sum_k (-DNW_k) QVAPOR_k / g

So with A = sum_k (-DNW_k) QVAPOR_k,  PW = (PSFC - P_TOP) * A / (1 + A) / g  [mm].

Reads QVAPOR from atmos_3d, PSFC from surface_met, ZNW from static and P_TOP
from the grid file. Writes to --out:
  report.md               benchmark timings and test results (paste this back)
  benchmark.json          the same numbers, machine readable
  pw_seasonal_cycle.png   domain-mean seasonal cycle and Jan/Jul maps
  pw_monthly.nc           monthly-mean PW maps, for reuse

Checks (exit status 1 if any fails):
  * PW from the stores equals PW computed from the original NetCDF files
  * the domain-mean climatology is physically plausible (1-40 mm, summer > winter)

    python tools/benchmark_pw.py --config config.yaml --stride 6
    python tools/benchmark_pw.py --config config.yaml --start 2000-10-01 --end 2001-09-30   # one year
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import dask
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wrf_to_zarr as w  # noqa: E402

G = 9.81


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self.t0


def log(msg):
    print(f"{pd.Timestamp.now():%H:%M:%S}  {msg}", flush=True)


def column_weights(cfg, static, nlev):
    """-DNW per mass level: each layer's share of the dry column (positive, sums to 1).

    The static store's ZNW was destaggered onto mass levels by the converter, so
    take the full-level ZNW from the grid file, or rebuild it from ZNU (WRF puts
    ZNU halfway between full levels): ZNW[0] = 1, ZNW[k+1] = 2 ZNU[k] - ZNW[k].
    """
    znw = None
    if cfg.get("grid_file"):
        with netCDF4.Dataset(cfg["grid_file"]) as nc:
            if "ZNW" in nc.variables:
                znw = np.asarray(nc.variables["ZNW"][:], "f8").reshape(-1, nlev + 1)[0]
    if znw is None:
        znw_s = np.asarray(static["ZNW"].values, "f8").squeeze()
        if znw_s.size == nlev + 1:
            znw = znw_s
        else:
            znu = np.asarray(static["ZNU"].values, "f8").squeeze()
            znw = np.empty(nlev + 1)
            znw[0] = 1.0
            for k in range(nlev):
                znw[k + 1] = 2 * znu[k] - znw[k]
            znw[-1] = 0.0
    wts = znw[:-1] - znw[1:]
    if not np.isclose(wts.sum(), 1, atol=1e-3) or (wts <= 0).any():
        sys.exit(f"could not get WRF eta levels: weights sum to {wts.sum():.5f}, min {wts.min():.2e}")
    return xr.DataArray(wts, dims="level")


def p_top(cfg, override):
    if override is not None:
        return float(override)
    if cfg.get("grid_file"):
        with netCDF4.Dataset(cfg["grid_file"]) as nc:
            if "P_TOP" in nc.variables:
                return float(np.asarray(nc.variables["P_TOP"][:]).ravel()[0])
    sys.exit("P_TOP not found: set grid_file in the config or pass --ptop")


def pw_from(q, psfc, wts, ptop):
    """q (..., level, y, x) and psfc (..., y, x) to PW in mm. Works on numpy or xarray."""
    a = (q * wts).sum("level") if isinstance(q, xr.DataArray) else np.tensordot(wts, q, axes=(0, -3))
    return (psfc - ptop) * a / (1 + a) / G


def timed(ds):
    """The store's data variables that have a time axis, without hour_present."""
    return [v for v in ds.data_vars if "time" in ds[v].dims and v != "hour_present"]


def bench_reads(a3, sm, rng, n):
    """Latency of the access patterns training and analysis use."""
    nt, out = a3.sizes["time"], {}
    good = np.flatnonzero(a3["hour_present"].values == 1)
    hours = rng.choice(good, min(n, len(good)), replace=False)
    nbytes_q = a3["QVAPOR"].isel(time=0).nbytes

    ts = []
    for i in hours:
        with Timer() as t:
            a3["QVAPOR"].isel(time=int(i)).values
        ts.append(t.s)
    out["one_hour_QVAPOR"] = {"what": "one hour of QVAPOR, full 3D grid", "n": n,
                              "median_s": float(np.median(ts)), "max_s": float(np.max(ts)),
                              "MB_per_s": nbytes_q / 1e6 / float(np.median(ts))}

    names = timed(a3)
    nbytes_all = sum(a3[v].isel(time=0).nbytes for v in names)
    ts = []
    for i in hours:
        with Timer() as t:
            a3[names].isel(time=int(i)).load()
        ts.append(t.s)
    out["one_hour_all_3d"] = {"what": f"one hour of all {len(names)} 3D variables (one training sample)",
                              "n": n, "MB": nbytes_all / 1e6, "median_s": float(np.median(ts)),
                              "max_s": float(np.max(ts)), "MB_per_s": nbytes_all / 1e6 / float(np.median(ts))}

    i0 = int(rng.integers(0, nt - 24 + 1))
    with Timer() as t:
        sm[timed(sm)].isel(time=slice(i0, i0 + 24)).load()
    nb = sum(sm[v].isel(time=slice(0, 24)).nbytes for v in timed(sm))
    out["one_day_surface_met"] = {"what": "one day of every surface_met variable", "MB": nb / 1e6,
                                  "s": t.s, "MB_per_s": nb / 1e6 / t.s}

    y, x = a3.sizes["y"] // 2, a3.sizes["x"] // 2
    nm = min(744, nt)
    i0 = int(rng.integers(0, nt - nm + 1))
    with Timer() as t:
        a3["QVAPOR"].isel(time=slice(i0, i0 + nm), y=y, x=x).values
    out["point_month_QVAPOR"] = {"what": "31 days of QVAPOR at one grid column (atmos_3d is chunked "
                                         "for whole maps, so this reads 744 full hours)", "s": t.s}
    with Timer() as t:
        sm["PSFC"].isel(time=slice(i0, i0 + nm), y=y, x=x).values
    out["point_month_PSFC"] = {"what": "31 days of PSFC at one grid point from surface_met", "s": t.s}
    return out


def stored_sizes(path, names):
    """Bytes on disk per variable, from the store's own listing (no du needed)."""
    import zarr
    g = zarr.open_group(path, mode="r")
    return {v: int(g[v].nbytes_stored()) for v in names}


def source_check(cfg, a3, sm, wts, ptop, rng, n):
    """PW from the stores vs PW from the original NetCDF, at n random hours."""
    s3, ss = w.Source(cfg, "atmos_3d"), w.Source(cfg, "surface_met")
    cache = w.FileCache()
    good = np.flatnonzero((a3["hour_present"].values == 1) & (sm["hour_present"].values == 1))
    worst, rows = 0.0, []
    try:
        for i in rng.choice(good, min(n, len(good)), replace=False):
            t = pd.Timestamp(a3.time.values[i])
            (pq, kq), (pp, kp) = s3.locate("QVAPOR", t), ss.locate("PSFC", t)
            q = w.read_nc(cache(pq), "QVAPOR", kq, False)[0].astype("f8")
            ps = w.read_nc(cache(pp), "PSFC", kp, False)[0].astype("f8")
            ref = pw_from(q, ps, wts.values, ptop)
            got = pw_from(a3["QVAPOR"].isel(time=int(i)).astype("f8"),
                          sm["PSFC"].isel(time=int(i)).astype("f8"), wts, ptop).values
            d = float(np.nanmax(np.abs(got - ref)))
            worst = max(worst, d)
            rows.append((str(t), float(np.nanmean(ref)), d))
    finally:
        cache.close()
    return worst, rows


def monthly_pw(a3, sm, wts, ptop, stride, start, end):
    times = pd.DatetimeIndex(a3.time.values)
    keep = (times.hour % stride == 0) & (times >= start) & (times <= end)
    idx = np.flatnonzero(keep)
    months = times[idx].to_period("M")
    maps, labels, counts = [], [], []
    hp = (a3["hour_present"] == 1) & (sm["hour_present"] == 1)
    for k, m in enumerate(months.unique()):
        ii = idx[months == m]
        good = hp.isel(time=ii)
        pw = pw_from(a3["QVAPOR"].isel(time=ii), sm["PSFC"].isel(time=ii), wts, ptop).where(good)
        mean, n = dask.compute(pw.mean("time"), good.sum())
        maps.append(mean.values.astype("f4"))
        labels.append(m.to_timestamp())
        counts.append(int(n))
        if k % 12 == 11:
            log(f"  through {m}: domain-mean PW {np.nanmean(maps[-1]):.2f} mm")
    return np.stack(maps), pd.DatetimeIndex(labels), np.array(counts), len(idx)


def plot(monthly, lat, lon, png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = pd.Series(np.nanmean(monthly["pw"].values, axis=(1, 2)), index=monthly.time.values)
    wy_order = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    by_month = series.groupby(series.index.month)
    clim = by_month.mean().reindex(wy_order)
    lo, hi = by_month.quantile(0.1).reindex(wy_order), by_month.quantile(0.9).reindex(wy_order)
    clim_maps = monthly["pw"].groupby("time.month").mean()

    fig = plt.figure(figsize=(12, 8), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    ax = fig.add_subplot(gs[0, :])
    xs = np.arange(12)
    ax.fill_between(xs, lo.values, hi.values, color="#9ecae1", alpha=0.6, label="10th–90th percentile across years")
    ax.plot(xs, clim.values, "o-", color="#08519c", lw=2, label="mean")
    ax.set_xticks(xs, ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"])
    ax.set_ylabel("Domain-mean precipitable water (mm)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=False)

    vmax = float(np.nanpercentile(clim_maps.values, 99.5))
    for j, (mo, name) in enumerate([(1, "January"), (7, "July")]):
        if mo not in clim_maps.month.values:
            continue
        a = fig.add_subplot(gs[1, j])
        m = a.pcolormesh(lon, lat, clim_maps.sel(month=mo).values, cmap="YlGnBu", vmin=0, vmax=vmax,
                         shading="auto")
        a.set_title(f"{name} mean PW")
        a.set_aspect(1 / np.cos(np.deg2rad(float(np.mean(lat)))))
        a.set_xlabel("Longitude")
        a.set_ylabel("Latitude")
        fig.colorbar(m, ax=a, label="mm", shrink=0.85)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return clim


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--stride", type=int, default=6, help="use every Nth hour (6 = 00/06/12/18 UTC)")
    p.add_argument("--start", default=None, help="first time (default: start of the record)")
    p.add_argument("--end", default=None, help="last time (default: end of the record)")
    p.add_argument("--ptop", type=float, default=None, help="model top pressure in Pa (default: from grid_file)")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--samples", type=int, default=10, help="random hours for the latency and source checks")
    p.add_argument("--no-sizes", action="store_true", help="skip measuring bytes on disk per variable")
    p.add_argument("--out", default=None, help="output folder (default bench/<timestamp>)")
    a = p.parse_args()

    cfg = w.load_config(a.config)
    out = Path(a.out or f"bench/{pd.Timestamp.now():%Y%m%d-%H%M}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    dask.config.set(scheduler="threads", num_workers=a.workers)
    res = {"workers": a.workers, "stride_hours": a.stride, "checks": {}}

    with Timer() as t:
        a3 = xr.open_zarr(w.store_path(cfg, "atmos_3d"), consolidated=True)
        sm = xr.open_zarr(w.store_path(cfg, "surface_met"), consolidated=True)
        st = xr.open_zarr(w.store_path(cfg, "static"), consolidated=True)
    res["open_stores_s"] = t.s
    if a3.sizes["time"] != sm.sizes["time"] or (a3.time.values[[0, -1]] != sm.time.values[[0, -1]]).any():
        sys.exit("atmos_3d and surface_met have different time axes")
    wts, ptop = column_weights(cfg, st, a3.sizes["level"]), p_top(cfg, a.ptop)
    res["p_top_Pa"] = ptop
    log(f"opened stores in {t.s:.1f} s; {a3.sizes['time']:,} hours, {a3.sizes['level']} levels, "
        f"P_TOP {ptop:.0f} Pa, {a.workers} threads")

    log("read latency benchmarks")
    res["reads"] = bench_reads(a3, sm, rng, a.samples)
    for k, v in res["reads"].items():
        log(f"  {k}: {json.dumps({x: round(y, 3) if isinstance(y, float) else y for x, y in v.items() if x != 'what'})}")

    if not a.no_sizes:
        log("measuring bytes on disk per variable")
        with Timer() as t:
            names = timed(a3)
            sizes = stored_sizes(w.store_path(cfg, "atmos_3d"), names)
        res["atmos_3d_bytes_on_disk"] = sizes
        res["atmos_3d_decoded_bytes"] = {v: int(a3[v].nbytes) for v in names}
        log(f"  atmos_3d: {sum(sizes.values()) / 1e12:.2f} TB on disk ({t.s:.0f} s to list)")

    log("functional test: PW from the stores vs the original NetCDF")
    worst, rows = source_check(cfg, a3, sm, wts, ptop, rng, min(a.samples, 5))
    res["checks"]["store_matches_netcdf"] = {"ok": worst < 1e-3, "max_abs_diff_mm": worst, "hours": rows}
    log(f"  max |store - NetCDF| = {worst:.2e} mm over {len(rows)} hours")

    start = pd.Timestamp(a.start) if a.start else pd.Timestamp(a3.time.values[0])
    end = pd.Timestamp(a.end) if a.end else pd.Timestamp(a3.time.values[-1])
    log(f"monthly PW, every {a.stride} h, {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    with Timer() as t:
        maps, months, counts, n_hours = monthly_pw(a3, sm, wts, ptop, a.stride, start, end)
    gb = n_hours * (a3["QVAPOR"].isel(time=0).nbytes + sm["PSFC"].isel(time=0).nbytes) / 1e9
    res["seasonal_cycle"] = {"hours_read": n_hours, "months": len(months), "decoded_GB": gb, "s": t.s,
                             "decoded_GB_per_s": gb / t.s, "hours_per_s": n_hours / t.s}
    log(f"  {n_hours:,} hours, {gb:,.0f} GB decoded in {t.s / 60:.1f} min ({gb / t.s:.2f} GB/s)")

    monthly = xr.Dataset({"pw": (("time", "y", "x"), maps, {"units": "mm", "long_name": "monthly mean precipitable water"}),
                          "n_hours": (("time",), counts)},
                         coords={"time": months, "lat": a3["lat"], "lon": a3["lon"]})
    monthly.to_netcdf(out / "pw_monthly.nc")
    yrs = f"{months[0]:%Y-%m} to {months[-1]:%Y-%m}"
    clim = plot(monthly, a3["lat"].values, a3["lon"].values, out / "pw_seasonal_cycle.png",
                f"Precipitable water, WRF 1 km Idaho, {yrs} (every {a.stride} h, from atmos_3d Zarr)")
    res["climatology_mm"] = {str(k): round(float(v), 2) for k, v in clim.items()}
    have = clim.dropna()
    plausible = bool(have.between(1, 40).all()) and (len(have) < 12 or clim[7] > clim[1])
    res["checks"]["plausible_climatology"] = {"ok": plausible, "range_mm": [float(have.min()), float(have.max())]}

    (out / "benchmark.json").write_text(json.dumps(res, indent=2, default=str))
    (out / "report.md").write_text(report(res, out))
    print("\n" + (out / "report.md").read_text())
    failed = [k for k, v in res["checks"].items() if not v["ok"]]
    sys.exit(1 if failed else 0)


def report(r, out):
    L = [f"# Zarr benchmark ({r['workers']} threads)", "", "## Checks"]
    for k, v in r["checks"].items():
        L.append(f"- {'ok  ' if v['ok'] else 'FAIL'} {k}: " + ", ".join(f"{a}={b}" for a, b in v.items()
                                                                     if a not in ("ok", "hours")))
    L += ["", "## Reads", f"- open three stores: {r['open_stores_s']:.1f} s"]
    for v in r["reads"].values():
        s = v.get("median_s", v.get("s"))
        rate = f", {v['MB_per_s']:.0f} MB/s" if "MB_per_s" in v else ""
        L.append(f"- {v['what']}: {s:.2f} s{' median' if 'median_s' in v else ''}{rate}")
    sc = r["seasonal_cycle"]
    L += ["", "## Seasonal cycle of PW",
          f"- {sc['hours_read']:,} hours over {sc['months']} months, {sc['decoded_GB']:,.0f} GB decoded in "
          f"{sc['s'] / 60:.1f} min ({sc['decoded_GB_per_s']:.2f} GB/s, {sc['hours_per_s']:.1f} hours/s)",
          "- domain-mean climatology (mm): " + ", ".join(f"{k}:{v}" for k, v in r["climatology_mm"].items())]
    if "atmos_3d_bytes_on_disk" in r:
        L += ["", "## atmos_3d size on disk", "| variable | on disk (TB) | ratio |", "|---|---|---|"]
        tot = 0
        for v, b in sorted(r["atmos_3d_bytes_on_disk"].items(), key=lambda kv: -kv[1]):
            tot += b
            L.append(f"| {v} | {b / 1e12:.2f} | {r['atmos_3d_decoded_bytes'][v] / max(b, 1):.1f}x |")
        L.append(f"| total | {tot / 1e12:.2f} | |")
    L += ["", f"Files in {out}: pw_seasonal_cycle.png, pw_monthly.nc, benchmark.json"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
