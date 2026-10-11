#!/usr/bin/env python
"""Convert WRF output into CF-described Zarr stores, one per variable group.

Works from the manifest written by wrf_inventory.py, so it never has to search
the file system or guess which file holds which variable and hour. Handles
classic wrfout files and the one-variable-per-month layout (VAR_YYYY-MM.nc). Each store gets a regular
hourly time axis; hours with no source file stay NaN, and the store's
`hour_present` variable gives the fraction of its source variables found.

Steps, each its own command:

    init    create an empty store with all metadata and coordinates
            (for `static`, this also writes the data: there is no time axis)
    ntasks  print how many array tasks `write` needs
    write   fill one task's block of time; safe to run tasks in parallel,
            because every task writes whole chunks (or shards) and no two
            tasks share one
    status  list the task ids that have not finished, for resubmission

    python wrf_to_zarr.py init   --config config.yaml --store surface_met
    python wrf_to_zarr.py write  --config config.yaml --store surface_met --task 0
    python wrf_to_zarr.py status --config config.yaml --store surface_met

Transformations, all recorded in each variable's attributes:
  * WRF dimension names become time, level, soil_layer, y, x.
  * Staggered variables are averaged onto the mass grid (destagger: true).
  * Accumulated variables (and their I_<VAR> buckets) become hourly amounts
    named <VAR>_1h. A drop in the running total is read as a reset, such as a
    restart, and the hour takes the new total.
  * Winds stay grid-relative (CF x_wind/y_wind). COSALPHA and SINALPHA in the
    static store rotate them to earth-relative.
"""
import argparse
import json
import os
import shutil
import sys
import time as _time
import warnings
from pathlib import Path

import dask.array as da
import netCDF4
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
import yaml
import zarr

warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")

STAGGERED = {"west_east_stag": "west_east", "south_north_stag": "south_north", "bottom_top_stag": "bottom_top"}
RENAME = {"Time": "time", "south_north": "y", "west_east": "x", "bottom_top": "level",
          "soil_layers_stag": "soil_layer", "south_north_stag": "y_stag", "west_east_stag": "x_stag",
          "bottom_top_stag": "level_stag"}
STANDARD_NAMES = {
    "T2": "air_temperature", "Q2": "humidity_mixing_ratio", "QVAPOR": "humidity_mixing_ratio",
    "PSFC": "surface_air_pressure", "SWDOWN": "surface_downwelling_shortwave_flux_in_air",
    "GLW": "surface_downwelling_longwave_flux_in_air", "TSK": "surface_temperature",
    "HFX": "surface_upward_sensible_heat_flux", "LH": "surface_upward_latent_heat_flux",
    "SNOW": "surface_snow_amount", "SNOWH": "surface_snow_thickness", "HGT": "surface_altitude",
    "SMOIS": "volume_fraction_of_condensed_water_in_soil", "TSLB": "soil_temperature",
    "U10": "x_wind", "V10": "y_wind", "U": "x_wind", "V": "y_wind", "W": "upward_air_velocity",
    "PBLH": "atmosphere_boundary_layer_thickness", "ALBEDO": "surface_albedo",
    "SFROFF_1h": "surface_runoff_amount", "UDROFF_1h": "subsurface_runoff_amount",
    "ACSNOM_1h": "surface_snow_melt_amount", "SNOWNC_1h": "snowfall_amount",
}
EARTH_RADIUS = 6370000.0  # WRF's sphere


# ----------------------------------------------------------------------------- setup

def load_config(path):
    """Read the config. Relative paths in it are taken relative to the config file's folder."""
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    here = path.parent
    for keys in (("inventory_dir",), ("grid_file",), ("output", "root")):
        d = cfg
        for k in keys[:-1]:
            d = d.get(k, {})
        if d.get(keys[-1]):
            d[keys[-1]] = str((here / Path(d[keys[-1]]).expanduser()).resolve())
    cfg["_time_axis"] = pd.date_range(cfg["time"]["start"], cfg["time"]["end"],
                                      freq=f"{cfg['time'].get('freq_hours', 1)}h").as_unit("ns")
    return cfg


def output_root(cfg, name):
    """output.root, unless the store sets its own root (e.g. to put a big store on scratch)."""
    return Path((cfg.get("stores", {}).get(name) or {}).get("root") or cfg["output"]["root"])


def store_path(cfg, name):
    return output_root(cfg, name) / f"{name}.zarr"


def progress_dir(cfg, name):
    return output_root(cfg, name) / "_progress" / name


class Source:
    """Maps (variable, time) to (file path, index in file), from the inventory."""

    def __init__(self, cfg, store=None):
        inv = Path(cfg["inventory_dir"])
        f = pd.read_csv(inv / "manifest_files.csv", keep_default_na=False)
        f = f[(f["error"] == "") & (f["domain"] == cfg["domain"]) & (f["data_vars"] != "")]
        # include_paths: a regex a file's path must match; a store may set its own.
        include = (cfg.get("stores", {}).get(store) or {}).get("include_paths", cfg.get("include_paths"))
        if include:
            f = f[f["path"].str.contains(include, regex=True)]
        if f.empty:
            sys.exit(f"no readable files for domain {cfg['domain']} in {inv} matching {include!r}")
        self.files = f.set_index("file_id")
        self.paths = self.files["path"]
        irr = pd.read_csv(inv / "manifest_times.csv", parse_dates=["time"])
        self.irregular = {fid: g.sort_values("index_in_file")["time"].values.astype("datetime64[ns]").astype("i8")
                          for fid, g in irr.groupby("file_id")}
        self.by_var = {}
        for fid, names in self.files["data_vars"].items():
            for v in names.split():
                self.by_var.setdefault(v, []).append(fid)
        self.keep = cfg.get("duplicate_times", "first")
        self._index = {}
        self.grid_file = cfg.get("grid_file") or None
        self.first_path = self.paths.iloc[0]

    def has(self, var):
        return var in self.by_var

    def times_of(self, fid):
        r = self.files.loc[fid]
        if fid in self.irregular:
            return self.irregular[fid]
        t0 = pd.Timestamp(r["first_time"]).as_unit("ns").value
        step = int(r["step_seconds"]) * 10**9
        return t0 + np.arange(int(r["n_times"]), dtype="i8") * step

    def index(self, var):
        """Sorted (times, file_ids, positions) for one variable, duplicates resolved."""
        if var not in self._index:
            fids = sorted(self.by_var[var], key=lambda i: self.paths[i])
            if self.keep == "last":
                fids = fids[::-1]
            t = np.concatenate([self.times_of(i) for i in fids])
            fid = np.concatenate([np.full(len(self.times_of(i)), i) for i in fids])
            k = np.concatenate([np.arange(len(self.times_of(i))) for i in fids])
            _, first = np.unique(t, return_index=True)  # first occurrence in the chosen order
            self._index[var] = (t[first], fid[first], k[first])
        return self._index[var]

    def locate(self, var, t):
        if var not in self.by_var:
            return None
        ts, fid, k = self.index(var)
        tv = pd.Timestamp(t).as_unit("ns").value
        j = np.searchsorted(ts, tv)
        if j < len(ts) and ts[j] == tv:
            return self.paths[fid[j]], int(k[j])
        return None

    def first_location(self, var):
        ts, fid, k = self.index(var)
        return self.paths[fid[0]], int(k[0])

    def sample_path(self, var):
        return self.paths[sorted(self.by_var[var], key=lambda i: self.paths[i])[0]]


class FileCache:
    """Keep recently used files open across the hours and variables of a task."""

    def __init__(self, max_open=64):
        self.open, self.max_open = {}, max_open

    def __call__(self, path):
        if path in self.open:
            self.open[path] = self.open.pop(path)  # most recently used goes last
        else:
            if len(self.open) >= self.max_open:
                self.open.pop(next(iter(self.open))).close()
            nc = netCDF4.Dataset(path)
            nc.set_auto_mask(False)
            self.open[path] = nc
        return self.open[path]

    def close(self):
        for nc in self.open.values():
            nc.close()
        self.open = {}


# ----------------------------------------------------------------------------- reading

def destagger(arr, dims):
    """Average a staggered array onto the mass grid. dims excludes Time."""
    out_dims = list(dims)
    for ax, d in enumerate(dims):
        if d in STAGGERED:
            sl0 = [slice(None)] * arr.ndim
            sl1 = [slice(None)] * arr.ndim
            sl0[ax], sl1[ax] = slice(0, -1), slice(1, None)
            arr = 0.5 * (arr[tuple(sl0)] + arr[tuple(sl1)])
            out_dims[ax] = STAGGERED[d]
    return arr, out_dims


def read_nc(nc, name, k, do_destagger):
    v = nc.variables[name]
    arr = np.asarray(v[k], dtype="f4")
    dims = list(v.dimensions[1:])
    fill = getattr(v, "_FillValue", None)
    if fill is not None:
        arr = np.where(arr == fill, np.nan, arr)
    if do_destagger:
        arr, dims = destagger(arr, dims)
    return arr, dims


def read_field(src, cache, name, t, do_destagger):
    loc = src.locate(name, t)
    if loc is None:
        return None
    return read_nc(cache(loc[0]), name, loc[1], do_destagger)[0]


def bucket_size(cfg, nc, name):
    units = getattr(nc.variables[name], "units", "")
    key = "BUCKET_J" if "J" in units else "BUCKET_MM"
    return float(getattr(nc, key, cfg.get(key.lower(), 0.0)) or 0.0)


def read_total(cfg, src, cache, name, t):
    """Running total of an accumulated variable, including its I_<VAR> bucket counter.

    The bucket may sit in the same file (wrfout) or its own I_<VAR>_YYYY-MM.nc file."""
    loc = src.locate(name, t)
    if loc is None:
        return None
    nc = cache(loc[0])
    arr = read_nc(nc, name, loc[1], False)[0]
    bucket = f"I_{name}"
    if src.has(bucket):
        size = bucket_size(cfg, nc, name)
        bloc = src.locate(bucket, t)
        if size > 0:
            if bloc is None:
                return None  # can't reconstruct the total without the bucket
            arr = arr + size * np.asarray(cache(bloc[0]).variables[bucket][bloc[1]], dtype="f4")
        elif bloc is not None and np.any(np.asarray(cache(bloc[0]).variables[bucket][bloc[1]])):
            raise ValueError(f"{bucket} is non-zero at {t} but the bucket size is unknown: "
                             "set bucket_mm / bucket_j in config.yaml")
    return arr


# ----------------------------------------------------------------------------- grid and metadata

def grid_dataset(cfg, src):
    """The file whose global attributes define the projection: grid_file, or any source file."""
    return netCDF4.Dataset(src.grid_file or src.first_path)


def projection(nc, nx, ny):
    for a in ("MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "MOAD_CEN_LAT", "CEN_LAT", "CEN_LON", "DX", "DY"):
        if not hasattr(nc, a):
            sys.exit(f"{nc.filepath()} lacks global attribute {a}; set grid_file in the config to a "
                     "wrfout, wrfinput or geo_em file for this domain")
    if int(nc.MAP_PROJ) != 1:
        raise NotImplementedError(f"MAP_PROJ={nc.MAP_PROJ}; only Lambert conformal (1) is handled")
    crs = pyproj.CRS.from_proj4(
        f"+proj=lcc +lat_1={nc.TRUELAT1} +lat_2={nc.TRUELAT2} +lat_0={nc.MOAD_CEN_LAT} "
        f"+lon_0={nc.STAND_LON} +a={EARTH_RADIUS} +b={EARTH_RADIUS} +units=m +no_defs")
    fwd = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xc, yc = fwd.transform(float(nc.CEN_LON), float(nc.CEN_LAT))
    x = xc + (np.arange(nx) - (nx - 1) / 2) * float(nc.DX)
    y = yc + (np.arange(ny) - (ny - 1) / 2) * float(nc.DY)
    gm = xr.DataArray(np.int32(0), attrs={
        "grid_mapping_name": "lambert_conformal_conic",
        "standard_parallel": [float(nc.TRUELAT1), float(nc.TRUELAT2)],
        "longitude_of_central_meridian": float(nc.STAND_LON),
        "latitude_of_projection_origin": float(nc.MOAD_CEN_LAT),
        "false_easting": 0.0, "false_northing": 0.0, "earth_radius": EARTH_RADIUS,
        "crs_wkt": crs.to_wkt(),
    })
    return x, y, gm, crs


def coords_for(cfg, src):
    with grid_dataset(cfg, src) as nc:
        nc.set_auto_mask(False)
        nx, ny = len(nc.dimensions["west_east"]), len(nc.dimensions["south_north"])
        x, y, gm, crs = projection(nc, nx, ny)
        latlon = None
        if "XLAT" in nc.variables and "XLONG" in nc.variables:
            pick = lambda v: np.asarray(v[0] if v.ndim == 3 else v[:], "f8")
            latlon = pick(nc.variables["XLAT"]), pick(nc.variables["XLONG"])
        globals_ = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in nc.__dict__.items()
                    if isinstance(v, (str, int, float, np.generic))}
    lat_src = "XLAT/XLONG from the grid file"
    if latlon is None:
        inv = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = inv.transform(*np.meshgrid(x, y))
        latlon, lat_src = (lat, lon), "computed from the projection"
    c = {
        "x": ("x", x, {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"}),
        "y": ("y", y, {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"}),
        "lat": (("y", "x"), latlon[0], {"standard_name": "latitude", "units": "degrees_north", "source": lat_src}),
        "lon": (("y", "x"), latlon[1], {"standard_name": "longitude", "units": "degrees_east", "source": lat_src}),
    }
    cache = FileCache()
    try:
        for var, dim, attrs in [("ZNU", "level", {"long_name": "eta value on mass levels", "units": "1",
                                                  "axis": "Z", "positive": "down"}),
                                ("ZS", "soil_layer", {"long_name": "depth of soil layer center", "units": "m",
                                                      "positive": "down"})]:
            if src.has(var):
                path, k = src.first_location(var)
                c[dim] = (dim, read_nc(cache(path), var, k, False)[0], attrs)
    finally:
        cache.close()
    return c, gm, globals_


def var_attrs(nc, name, out_name=None, extra=None):
    v = nc.variables[name]
    out_name = out_name or name
    a = {"long_name": getattr(v, "description", name).strip() or name,
         "units": getattr(v, "units", "").strip() or "1",
         "wrf_name": name, "grid_mapping": "lambert_conformal"}
    if out_name in STANDARD_NAMES:
        a["standard_name"] = STANDARD_NAMES[out_name]
    a.update(extra or {})
    return a


def global_attrs(cfg, wrf_globals, name, skipped):
    a = {"Conventions": "CF-1.10", "store": name,
         "history": f"{pd.Timestamp.now('UTC'):%Y-%m-%dT%H:%MZ} wrf_to_zarr.py from {cfg['inventory_dir']}",
         "variables_missing_from_source": skipped, **cfg.get("dataset_attrs", {})}
    a.update({f"WRF_{k}": v for k, v in wrf_globals.items() if k != "history"})
    return a


def drop_unused_coords(ds):
    used = {d for v in ds.data_vars.values() for d in v.dims}
    return ds.drop_vars([c for c in ("level", "soil_layer") if c in ds.coords and c not in used])


def compressor(cfg):
    c = cfg["output"].get("compressor", {"cname": "zstd", "clevel": 3, "shuffle": "shuffle"})
    return zarr.codecs.BloscCodec(cname=c["cname"], clevel=c["clevel"], shuffle=c["shuffle"])


# ----------------------------------------------------------------------------- store layout

def plan_variables(cfg, name, src):
    """Return ([(out_name, kind, spec)], skipped) for a store, leaving out what the files lack."""
    sc = cfg["stores"][name]
    plan, skipped = [], []
    for v in sc.get("variables", []):
        (plan.append((v, "direct", v)) if src.has(v) else skipped.append(v))
    for v in sc.get("accumulated", []):
        (plan.append((f"{v}_1h", "accum", v)) if src.has(v) else skipped.append(v))
    for out, d in (sc.get("derived") or {}).items():
        parts = [p for p in d["sum_of"] if src.has(p)]
        if parts:
            plan.append((out, "sum", dict(d, sum_of=parts)))
        else:
            skipped.append(out)
    if skipped:
        warnings.warn(f"{name}: not in source files, skipped: {skipped}")
    return plan, sorted(set(skipped))


def source_vars(plan):
    """Every source variable a plan reads; hour_present is the share of these found."""
    out = []
    for _, kind, spec in plan:
        out += spec["sum_of"] if kind == "sum" else [spec]
    return sorted(set(out))


def unit_hours(cfg, name):
    """Hours in one write unit: a shard if sharded, else a chunk."""
    sc = cfg["stores"][name]
    return int((sc.get("shards") or sc["chunks"])["time"])


def sample(src, var, do_ds):
    """Shape, output dims and the open file of a variable's first output, for metadata."""
    path, k = src.first_location(var)
    nc = netCDF4.Dataset(path)
    nc.set_auto_mask(False)
    arr, dims = read_nc(nc, var, k, do_ds)
    return arr, [RENAME.get(d, d) for d in dims], nc


def cmd_init(cfg, name, overwrite):
    src = Source(cfg, name)
    path = store_path(cfg, name)
    if path.exists() and not overwrite:
        sys.exit(f"{path} exists; pass --overwrite to replace it")
    # A replaced store starts empty, so the old store's progress markers must go too.
    shutil.rmtree(progress_dir(cfg, name), ignore_errors=True)
    sc = cfg["stores"][name]
    coords, gm, wrf_globals = coords_for(cfg, src)
    plan, skipped = plan_variables(cfg, name, src)
    do_ds = sc.get("destagger", True)

    if name == "static":
        data = {}
        for out, _, v in plan:
            arr, dims, nc = sample(src, v, do_ds)
            data[out] = (dims, arr, var_attrs(nc, v))
            nc.close()
        ds = drop_unused_coords(xr.Dataset(data, coords=coords))
        ds["lambert_conformal"] = gm
        ds.attrs = global_attrs(cfg, wrf_globals, name, skipped)
        ds.to_zarr(path, mode="w", zarr_format=cfg["output"]["zarr_format"], consolidated=True,
                   encoding={v: {"compressors": [compressor(cfg)]} for v in data})
        print(f"wrote {path}")
        return

    taxis = cfg["_time_axis"]
    t_chunk = int(sc["chunks"]["time"])
    t_shard = (sc.get("shards") or {}).get("time")
    t_dask = int(t_shard or t_chunk)
    data, enc = {}, {}
    for out, kind, spec in plan:
        arr, dims, nc = sample(src, spec if kind != "sum" else spec["sum_of"][0], do_ds)
        try:
            if kind == "accum":
                attrs = var_attrs(nc, spec, out, {
                    "cell_methods": "time: sum", "comment":
                    f"hourly amount: difference of WRF accumulated {spec} (with I_{spec} bucket if present) "
                    "between consecutive hours; a negative difference is treated as a reset"})
            elif kind == "sum":
                attrs = {"long_name": spec.get("long_name", out), "units": spec.get("units", "1"),
                         "cell_methods": "time: sum", "grid_mapping": "lambert_conformal",
                         "comment": "sum of hourly amounts of " + ", ".join(spec["sum_of"])}
                if "standard_name" in spec:
                    attrs["standard_name"] = spec["standard_name"]
            else:
                stag = do_ds and any(d in STAGGERED for d in nc.variables[spec].dimensions)
                attrs = var_attrs(nc, spec, out, {"comment": "destaggered to the mass grid by averaging "
                                                             "neighbours"} if stag else None)
        finally:
            nc.close()
        dims = ["time"] + dims
        shape, chunks = (len(taxis),) + arr.shape, (t_chunk,) + arr.shape
        data[out] = (dims, da.full(shape, np.nan, dtype="f4", chunks=(t_dask,) + arr.shape), attrs)
        enc[out] = {"chunks": chunks, "compressors": [compressor(cfg)]}
        if t_shard:
            enc[out]["shards"] = (int(t_shard),) + arr.shape
    data["hour_present"] = (["time"], da.full(len(taxis), np.nan, dtype="f4", chunks=t_dask),
                            {"long_name": "fraction of this store's source variables found for this hour; "
                                          "NaN if not yet converted", "units": "1",
                             "source_variables": source_vars(plan)})
    enc["hour_present"] = {"chunks": (t_chunk,)}
    if t_shard:
        enc["hour_present"]["shards"] = (int(t_shard),)
    ds = xr.Dataset(data, coords={**coords, "time": ("time", taxis, {"standard_name": "time", "axis": "T"})})
    ds["lambert_conformal"] = gm
    ds.attrs = global_attrs(cfg, wrf_globals, name, skipped)
    ds = drop_unused_coords(ds)
    enc["time"] = {"units": f"hours since {taxis[0]:%Y-%m-%d %H:%M:%S}", "calendar": "proleptic_gregorian",
                   "dtype": "int32"}
    ds.to_zarr(path, mode="w", compute=False, zarr_format=cfg["output"]["zarr_format"],
               consolidated=True, encoding=enc)
    progress_dir(cfg, name).mkdir(parents=True, exist_ok=True)
    for f in progress_dir(cfg, name).glob("unit_*.json"):
        f.unlink()
    print(f"initialised {path}: {len(plan)} variables, {len(taxis):,} hours, {n_units(cfg, name):,} write "
          f"units, {n_tasks(cfg, name):,} tasks")


# ----------------------------------------------------------------------------- writing

def n_units(cfg, name):
    return -(-len(cfg["_time_axis"]) // unit_hours(cfg, name))


def units_per_task(cfg, name):
    u = unit_hours(cfg, name)
    h = int(cfg["stores"][name].get("hours_per_task", u))
    if h % u:
        sys.exit(f"{name}: hours_per_task={h} must be a multiple of the write unit ({u} h)")
    return h // u


def n_tasks(cfg, name):
    return -(-n_units(cfg, name) // units_per_task(cfg, name))


def build_unit(cfg, name, src, cache, plan, i0, i1):
    """Read hours [i0, i1) of the time axis. Returns ({out_name: array}, hour_present)."""
    taxis = cfg["_time_axis"]
    do_ds = cfg["stores"][name].get("destagger", True)
    times = taxis[i0:i1]
    prev_t = times[0] - (taxis[1] - taxis[0])
    out, hourly = {}, {}

    def stack(arrs):
        shape = next((a.shape for a in arrs if a is not None), None)
        if shape is None:
            return None
        res = np.full((len(arrs),) + shape, np.nan, "f4")
        for j, a in enumerate(arrs):
            if a is not None:
                res[j] = a
        return res

    def accum_hourly(v):
        if v not in hourly:
            tot = [read_total(cfg, src, cache, v, t) for t in [prev_t] + list(times)]
            res = [None] * len(times)
            for j in range(len(times)):
                cur, before = tot[j + 1], tot[j]
                if cur is None or before is None:
                    continue
                d = cur - before
                reset = d < -1e-3 * np.maximum(1.0, np.abs(before))
                res[j] = np.maximum(np.where(reset, cur, d), 0.0)  # clip float round-off below zero
            hourly[v] = stack(res) if any(r is not None for r in res) else None
        return hourly[v]

    for o, kind, spec in plan:
        if kind == "direct":
            res = stack([read_field(src, cache, spec, t, do_ds) for t in times])
        elif kind == "accum":
            res = accum_hourly(spec)
        else:
            parts = [accum_hourly(p) for p in spec["sum_of"]]
            res = None if any(p is None for p in parts) else sum(parts)
        if res is not None:
            out[o] = res
    svars = source_vars(plan)
    present = np.array([np.mean([src.locate(v, t) is not None for v in svars]) for t in times], "f4")
    return out, present


def cmd_write(cfg, name, task, redo=False):
    if name == "static":
        sys.exit("static is written by init")
    src = Source(cfg, name)
    path = store_path(cfg, name)
    pdir = progress_dir(cfg, name)
    u, k = unit_hours(cfg, name), units_per_task(cfg, name)
    nt = len(cfg["_time_axis"])
    plan, _ = plan_variables(cfg, name, src)
    store_dims = {v: a.dims for v, a in xr.open_zarr(path, consolidated=True).data_vars.items()}
    cache = FileCache()
    try:
        for unit in range(task * k, min((task + 1) * k, n_units(cfg, name))):
            marker = pdir / f"unit_{unit:06d}.json"
            if marker.exists() and not redo:
                continue
            t_start = _time.time()
            i0, i1 = unit * u, min((unit + 1) * u, nt)
            arrays, present = build_unit(cfg, name, src, cache, plan, i0, i1)
            region = {"time": slice(i0, i1)}
            for o, arr in arrays.items():
                xr.Dataset({o: (store_dims[o], arr)}).to_zarr(path, region=region, mode="r+", consolidated=False)
            xr.Dataset({"hour_present": (("time",), present)}).to_zarr(path, region=region, mode="r+",
                                                                      consolidated=False)
            marker.write_text(json.dumps({"unit": unit, "i0": i0, "i1": i1, "mean_present": float(present.mean()),
                                          "hours": i1 - i0, "seconds": round(_time.time() - t_start, 1)}))
            print(f"unit {unit}: hours {i0}-{i1 - 1}, {present.mean():.0%} of source fields present, "
                  f"{_time.time() - t_start:.1f} s", flush=True)
    finally:
        cache.close()


def cmd_status(cfg, name):
    done = {int(f.stem.split("_")[1]) for f in progress_dir(cfg, name).glob("unit_*.json")}
    k, nu = units_per_task(cfg, name), n_units(cfg, name)
    todo = [t for t in range(n_tasks(cfg, name)) if any(u not in done for u in range(t * k, min((t + 1) * k, nu)))]
    print(f"{name}: {len(done):,}/{nu:,} units done; {len(todo):,} tasks unfinished")
    if todo:
        print("resubmit with: --array=" + compress_ranges(todo))


def compress_ranges(ids):
    out, start = [], ids[0]
    for a, b in zip(ids, ids[1:] + [None]):
        if b != a + 1:
            out.append(f"{start}-{a}" if a > start else f"{start}")
            start = b
    return ",".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["init", "ntasks", "write", "status"])
    p.add_argument("--config", required=True)
    p.add_argument("--store", required=True, help="a key under `stores` in the config")
    p.add_argument("--task", type=int, help="array task id for `write` (default: $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--overwrite", action="store_true", help="init: replace an existing store")
    p.add_argument("--redo", action="store_true", help="write: rewrite units already marked done")
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.store not in cfg["stores"]:
        sys.exit(f"unknown store {a.store}; config has {list(cfg['stores'])}")
    if a.command == "init":
        cmd_init(cfg, a.store, a.overwrite)
    elif a.command == "ntasks":
        print(n_tasks(cfg, a.store))
    elif a.command == "write":
        task = a.task if a.task is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        cmd_write(cfg, a.store, task, a.redo)
    else:
        cmd_status(cfg, a.store)


if __name__ == "__main__":
    main()
