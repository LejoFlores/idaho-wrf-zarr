#!/usr/bin/env python
"""Write small WRF-shaped NetCDF files for testing the pipeline.

Two layouts:
  wrfout        classic files, all variables, --hours-per-file hours each
  per-variable  one file per variable per month, VAR_YYYY-MM.nc, in a
                volXX/wrf_out/wy_YYYY/d02/ tree like Borah's

The files carry a Times char variable, WRF dimension names, staggered winds,
accumulated precipitation with an I_RAINNC bucket, and Lambert conformal
global attributes, with XLAT/XLONG consistent with them. Values are not
physically meaningful.

    python tests/make_synthetic_wrfout.py OUTDIR --layout per-variable --start 1986-10-30
"""
import argparse
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pyproj

NX, NY, NZ, NSOIL = 40, 30, 5, 4
DX = 1000.0
ATTRS = dict(
    TITLE=" OUTPUT FROM WRF V4.1.2 MODEL (synthetic)", MAP_PROJ=1, TRUELAT1=30.0,
    TRUELAT2=60.0, STAND_LON=-114.0, MOAD_CEN_LAT=44.0, CEN_LAT=44.2, CEN_LON=-115.0,
    DX=DX, DY=DX, GRID_ID=2, MP_PHYSICS=8, BL_PBL_PHYSICS=1, SF_SURFACE_PHYSICS=4,
    CU_PHYSICS=0, RA_LW_PHYSICS=4, RA_SW_PHYSICS=4, GRID_FDDA=0, BUCKET_MM=100.0,
)
DIMS = {"south_north": NY, "west_east": NX, "west_east_stag": NX + 1, "bottom_top": NZ,
        "soil_layers_stag": NSOIL}
T0 = pd.Timestamp("1985-10-01")


def latlon():
    crs = pyproj.CRS.from_proj4(
        f"+proj=lcc +lat_1={ATTRS['TRUELAT1']} +lat_2={ATTRS['TRUELAT2']} "
        f"+lat_0={ATTRS['MOAD_CEN_LAT']} +lon_0={ATTRS['STAND_LON']} +a=6370000 +b=6370000 +units=m")
    fwd = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xc, yc = fwd.transform(ATTRS["CEN_LON"], ATTRS["CEN_LAT"])
    x = xc + (np.arange(NX) - (NX - 1) / 2) * DX
    y = yc + (np.arange(NY) - (NY - 1) / 2) * DX
    lon, lat = inv.transform(*np.meshgrid(x, y))
    return lat.astype("f4"), lon.astype("f4")


def fields(times, accum_start, rate):
    """{name: (dims, units, description, stagger, dtype, data)} for the given times."""
    lat, lon = latlon()
    nt = len(times)
    h = np.array([(t - T0) / pd.Timedelta(hours=1) for t in times])
    h3 = h[:, None, None]
    b = np.ones((nt, NY, NX), "f4")
    acc = rate * (h - accum_start)
    two = ("south_north", "west_east")
    return {
        "XLAT": (two, "degree_north", "LATITUDE, SOUTH IS NEGATIVE", "", "f4", b * lat),
        "XLONG": (two, "degree_east", "LONGITUDE, WEST IS NEGATIVE", "", "f4", b * lon),
        "HGT": (two, "m", "Terrain Height", "", "f4", b * (1500 + 10 * lat)),
        "LANDMASK": (two, "", "LAND MASK", "", "f4", b),
        "T2": (two, "K", "TEMP at 2 M", "", "f4", 270 + 0.01 * h3 + 0 * b),
        "Q2": (two, "kg kg-1", "QV at 2 M", "", "f4", b * 0.004),
        "PSFC": (two, "Pa", "SFC PRESSURE", "", "f4", b * 85000),
        "SNOW": (two, "kg m-2", "SNOW WATER EQUIVALENT", "", "f4", b * h3),
        "U10": (two, "m s-1", "U at 10 M", "", "f4", b * 2),
        "V10": (two, "m s-1", "V at 10 M", "", "f4", b * -1),
        "RAINNC": (two, "mm", "ACCUMULATED TOTAL GRID SCALE PRECIPITATION", "", "f4",
                   b * (acc % 100)[:, None, None]),
        "I_RAINNC": (two, "", "BUCKET FOR RAINNC", "", "i4", b * (acc // 100)[:, None, None]),
        "RAINC": (two, "mm", "ACCUMULATED TOTAL CUMULUS PRECIPITATION", "", "f4", 0 * b),
        "SMOIS": (("soil_layers_stag",) + two, "m3 m-3", "SOIL MOISTURE", "", "f4",
                  np.ones((nt, NSOIL, NY, NX), "f4") * 0.3),
        "U": (("bottom_top", "south_north", "west_east_stag"), "m s-1", "x-wind component", "X", "f4",
              np.broadcast_to(np.arange(NX + 1, dtype="f4"), (nt, NZ, NY, NX + 1))),
        "T": (("bottom_top",) + two, "K", "perturbation potential temperature", "", "f4",
              np.ones((nt, NZ, NY, NX), "f4") * h3[:, None]),
        "ZNU": (("bottom_top",), "", "eta values on half (mass) levels", "", "f4",
                np.broadcast_to(np.linspace(0.99, 0.1, NZ, dtype="f4"), (nt, NZ))),
    }


def write(path, times, flds):
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.setncatts(ATTRS)
        nc.createDimension("Time", None)
        nc.createDimension("DateStrLen", 19)
        for d in sorted({d for f in flds.values() for d in f[0]}):
            nc.createDimension(d, DIMS[d])
        t = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[:] = np.array([list(s.strftime("%Y-%m-%d_%H:%M:%S")) for s in times], dtype="S1")
        for name, (dims, units, desc, stagger, dtype, data) in flds.items():
            v = nc.createVariable(name, dtype, ("Time",) + dims, zlib=True)
            v.units, v.description, v.stagger, v.MemoryOrder = units, desc, stagger, "XY "
            v[:] = data


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outdir")
    p.add_argument("--layout", choices=["wrfout", "per-variable"], default="wrfout")
    p.add_argument("--start", default="1985-10-01", help="first output time")
    p.add_argument("--hours", type=int, default=72)
    p.add_argument("--hours-per-file", type=int, default=6, help="wrfout layout only")
    p.add_argument("--gap", type=int, default=30, help="hour index left out, to test gap handling")
    p.add_argument("--restart", type=int, default=50, help="hour index at which accumulations reset")
    p.add_argument("--rate", type=float, default=0.5, help="precipitation rate, mm/h")
    a = p.parse_args()
    start = pd.Timestamp(a.start)
    hours = [h for h in range(a.hours) if h != a.gap]
    h0 = (start - T0) / pd.Timedelta(hours=1)
    out = Path(a.outdir)

    if a.layout == "wrfout":
        # Accumulations restart at the first file at or after --restart, as after a WRF restart.
        for f0 in range(0, a.hours, a.hours_per_file):
            idx = [h for h in hours if f0 <= h < f0 + a.hours_per_file]
            if not idx:
                continue
            times = [start + pd.Timedelta(hours=h) for h in idx]
            acc0 = h0 + (a.restart if f0 >= a.restart else 0)
            write(out / f"{times[0]:%Y}" / f"wrfout_d02_{times[0]:%Y-%m-%d_%H:%M:%S}", times,
                  fields(times, acc0, a.rate))
    else:
        t_all = [start + pd.Timedelta(hours=h) for h in hours]
        flds = fields(t_all, h0, a.rate)
        reset = np.array(hours) >= a.restart
        if reset.any():  # accumulations restart from zero at --restart
            later = fields(t_all, h0 + a.restart, a.rate)
            for k in ("RAINNC", "I_RAINNC"):
                d = flds[k][5].copy()
                d[reset] = later[k][5][reset]
                flds[k] = flds[k][:5] + (d,)
        months = pd.DatetimeIndex(t_all).to_period("M")
        for m in months.unique():
            sel = np.flatnonzero(months == m)
            times = [t_all[i] for i in sel]
            wy = times[0].year + (times[0].month >= 10)
            d = out / f"vol{(wy % 2) + 1:02d}" / "wrf_out" / f"wy_{wy}" / "d02"
            for name, f in flds.items():
                if name in ("XLAT", "XLONG"):
                    continue  # assume no lat/lon in per-variable files; the grid comes from attributes
                write(d / f"{name}_{m}.nc", times, {name: f[:5] + (f[5][sel],)})
            (d / "wrf_check.log").write_text("synthetic\n")
    print(f"wrote synthetic {a.layout} files under {out}")


if __name__ == "__main__":
    main()
