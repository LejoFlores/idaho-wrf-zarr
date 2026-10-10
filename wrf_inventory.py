#!/usr/bin/env python
"""Catalog a tree of WRF output files and write a manifest.

Handles both layouts:
  * classic wrfout files, many variables per file, and
  * one variable per file per month (VAR_YYYY-MM.nc), as on Borah under
    /bsushare/leaf-shared/subset-output-wrf/volXX/wrf_out/wy_YYYY/d02/.

Reads only headers and time variables, so it is cheap to run over the whole
archive. Outputs, in --out:

    manifest_files.csv   one row per file: path, size, volume, water year, domain,
                         variables held, first/last time, number of times, step
    manifest_times.csv   output times of files whose times are not evenly spaced
    variables.csv        per variable: dims, units, description, stagger, files,
                         GB, first/last time, hours present and missing
    coverage.csv         hours present per variable and water year
    water_years.csv      per water-year folder: volume, files, size, completeness
    summary.json         per domain: grid, projection, WRF version, physics, gaps
    inventory_report.md  the same, readable; paste this back

    python wrf_inventory.py --root /bsushare/leaf-shared/subset-output-wrf \\
        --domain d02 --out inventory/ --workers $SLURM_CPUS_PER_TASK
"""
import argparse
import fnmatch
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

# Global attributes worth recording: projection, grid, and model configuration.
GLOBAL_KEYS = [
    "TITLE", "START_DATE", "SIMULATION_START_DATE", "GRID_ID", "PARENT_ID", "DX", "DY",
    "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "MOAD_CEN_LAT", "CEN_LAT", "CEN_LON",
    "POLE_LAT", "POLE_LON", "WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION",
    "BOTTOM-TOP_GRID_DIMENSION", "I_PARENT_START", "J_PARENT_START", "PARENT_GRID_RATIO",
    "DT", "MP_PHYSICS", "RA_LW_PHYSICS", "RA_SW_PHYSICS", "SF_SFCLAY_PHYSICS",
    "SF_SURFACE_PHYSICS", "BL_PBL_PHYSICS", "CU_PHYSICS", "SF_LAKE_PHYSICS",
    "SURFACE_INPUT_SOURCE", "SST_UPDATE", "GRID_FDDA", "GFDDA_INTERVAL_M", "GFDDA_END_H",
    "GRID_SFDDA", "SGFDDA_INTERVAL_M", "SPEC_BDY_WIDTH", "BUCKET_MM", "BUCKET_J",
    "NUM_LAND_CAT", "ISWATER", "MMINLU", "HYPSOMETRIC_OPT", "USE_THETA_M", "history",
]
# Not data: time and coordinate helpers that WRF post-processing tools carry along.
HELPERS = {"Times", "XTIME", "time", "Time", "time_bnds", "XLAT", "XLONG", "XLAT_U", "XLONG_U",
           "XLAT_V", "XLONG_V", "lat", "lon", "latitude", "longitude", "crs"}
TIME_DIMS = {"Time", "time"}
DOMAIN_RE = re.compile(r"_d(\d\d)_")
DOMAIN_DIR_RE = re.compile(r"/(d\d\d)/")
VOLUME_RE = re.compile(r"/(vol\d+)/")
WY_RE = re.compile(r"/wy_(\d{4})/")
DIM_KEYS = ["west_east", "south_north", "bottom_top", "soil_layers_stag"]


def find_files(root, patterns):
    """Walk with os.scandir, which is much faster than glob on parallel file systems."""
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif any(fnmatch.fnmatch(e.name, p) for p in patterns):
                        yield e.path
        except PermissionError as err:
            print(f"warning: {err}", file=sys.stderr)


def read_times(nc):
    """Output times as a DatetimeIndex, from WRF's Times or a CF time variable."""
    if "Times" in nc.variables:
        raw = np.asarray(nc.variables["Times"][:])
        return pd.DatetimeIndex([b"".join(r).decode().strip().replace("_", " ") for r in raw]).as_unit("ns")
    for name in ("XTIME", "time", "Time"):
        v = nc.variables.get(name)
        if v is not None and hasattr(v, "units") and "since" in v.units:
            d = netCDF4.num2date(v[:], v.units, getattr(v, "calendar", "standard"),
                                 only_use_cftime_datetimes=False, only_use_python_datetimes=True)
            return pd.DatetimeIndex(pd.to_datetime([str(x) for x in np.atleast_1d(d)])).round("s").as_unit("ns")
    return pd.DatetimeIndex([]).as_unit("ns")


def scan_file(path):
    mv, mw = VOLUME_RE.search(path), WY_RE.search(path)
    row = {"path": path, "size_bytes": os.path.getsize(path), "error": "",
           "volume": mv.group(1) if mv else "", "water_year": int(mw.group(1)) if mw else -1}
    try:
        with netCDF4.Dataset(path) as nc:
            row["format"] = nc.data_model
            m, md = DOMAIN_RE.search(os.path.basename(path)), DOMAIN_DIR_RE.search(path)
            row["domain"] = (f"d{m.group(1)}" if m else md.group(1) if md
                             else f"d{int(getattr(nc, 'GRID_ID', 0)):02d}")
            t = read_times(nc)
            row["n_times"] = len(t)
            row["first_time"] = str(t[0]) if len(t) else ""
            row["last_time"] = str(t[-1]) if len(t) else ""
            steps = np.unique(np.diff(t.asi8)) if len(t) > 1 else np.array([])
            row["step_seconds"] = int(steps[0] // 10**9) if len(steps) == 1 and steps[0] > 0 else 0
            row["times"] = [] if row["step_seconds"] or len(t) <= 1 else [str(x) for x in t]
            dims = {k: len(v) for k, v in nc.dimensions.items()}
            for k in DIM_KEYS:
                row[f"n_{k}"] = dims.get(k, 0)
            vars_ = {}
            for name, v in nc.variables.items():
                if name in HELPERS or not v.dimensions or v.dimensions[0] not in TIME_DIMS:
                    continue
                filt = v.filters() or {}
                vars_[name] = {
                    "dims": " ".join(v.dimensions),
                    "units": getattr(v, "units", ""),
                    "description": getattr(v, "description", getattr(v, "long_name", "")),
                    "stagger": getattr(v, "stagger", ""),
                    "dtype": str(v.dtype),
                    "compressed": bool(filt.get("zlib") or filt.get("zstd") or filt.get("szip")),
                    "chunking": str(v.chunking()),
                }
            row["data_vars"] = " ".join(sorted(vars_))
            row["vars"] = vars_
            row["globals"] = {k: _jsonable(getattr(nc, k)) for k in GLOBAL_KEYS if hasattr(nc, k)}
            row["has_latlon"] = "XLAT" in nc.variables and "XLONG" in nc.variables
            if row["has_latlon"]:
                lat = np.asarray(nc.variables["XLAT"][0] if nc.variables["XLAT"].ndim == 3 else nc.variables["XLAT"][:])
                lon = np.asarray(nc.variables["XLONG"][0] if nc.variables["XLONG"].ndim == 3 else nc.variables["XLONG"][:])
                row["globals"]["corners_latlon"] = [
                    [round(float(lat[j, i]), 4), round(float(lon[j, i]), 4)]
                    for j, i in [(0, 0), (0, -1), (-1, 0), (-1, -1)]]
    except Exception as err:  # keep going; a bad file is a finding, not a crash
        row["error"] = f"{type(err).__name__}: {err}"
    return row


def _jsonable(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def file_times(r):
    """All output times of one scanned file, as int64 nanoseconds."""
    if r["step_seconds"]:
        return pd.Timestamp(r["first_time"]).as_unit("ns").value + np.arange(r["n_times"], dtype="i8") * r["step_seconds"] * 10**9
    if r["times"]:
        return pd.DatetimeIndex(r["times"]).as_unit("ns").asi8
    return pd.DatetimeIndex([r["first_time"]]).as_unit("ns").asi8 if r["first_time"] else np.array([], "i8")


def gaps_of(t_ns, step_ns):
    t = np.unique(t_ns)
    d = np.diff(t)
    return [{"after": str(pd.Timestamp(t[i])), "before": str(pd.Timestamp(t[i + 1])),
             "missing_steps": int(d[i] // step_ns) - 1} for i in np.flatnonzero(d > step_ns)]


def water_year_of(t_ns):
    t = pd.DatetimeIndex(t_ns)
    return t.year + (t.month >= 10)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, action="append", help="directory to search (repeatable)")
    p.add_argument("--pattern", action="append",
                   help="file name glob, repeatable (default: '*.nc' and 'wrfout*')")
    p.add_argument("--domain", action="append", help="only these domains, e.g. --domain d02 (repeatable)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--step-hours", type=float, default=1.0, help="expected output interval")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(f for r in a.root for f in find_files(r, a.pattern or ["*.nc", "wrfout*"]))
    if a.domain:
        paths = [f for f in paths if any(f"/{d}/" in f or f"_{d}_" in os.path.basename(f) for d in a.domain)]
    print(f"found {len(paths):,} files", flush=True)
    if not paths:
        sys.exit("no files matched")

    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(scan_file, paths, chunksize=32)):
            rows.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  scanned {i + 1:,}", flush=True)
    ok = [r for r in rows if not r["error"]]
    step_ns = int(a.step_hours * 3600 * 10**9)

    cols = ["path", "size_bytes", "volume", "water_year", "domain", "format", "data_vars", "n_times",
            "first_time", "last_time", "step_seconds", "has_latlon"] + [f"n_{k}" for k in DIM_KEYS] + ["error"]
    files = pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows])
    files.insert(0, "file_id", range(len(files)))
    files.to_csv(out / "manifest_files.csv", index=False)
    irregular = [(fid, k, t) for fid, r in enumerate(rows) if not r["error"] for k, t in enumerate(r["times"])]
    pd.DataFrame(irregular, columns=["file_id", "index_in_file", "time"]).to_csv(out / "manifest_times.csv", index=False)

    # Per variable: attributes, size, and time coverage.
    by_var = defaultdict(list)
    for r in ok:
        for name in r["vars"]:
            by_var[(r["domain"], name)].append(r)
    vrows, cov = [], []
    for (dom, name), rs in sorted(by_var.items()):
        t = np.unique(np.concatenate([file_times(r) for r in rs]))
        n_dup = int(sum(len(file_times(r)) for r in rs) - len(t))
        gaps = gaps_of(t, step_ns)
        attrs = rs[0]["vars"][name]
        vrows.append(dict(domain=dom, name=name, **attrs, n_files=len(rs),
                          gb=round(sum(r["size_bytes"] / len(r["vars"]) for r in rs) / 1e9, 2),
                          first_time=str(pd.Timestamp(t[0])) if len(t) else "",
                          last_time=str(pd.Timestamp(t[-1])) if len(t) else "",
                          hours=len(t), missing_hours=sum(g["missing_steps"] for g in gaps),
                          n_gaps=len(gaps), duplicate_hours=n_dup))
        if len(t):
            for wy, n in pd.Series(water_year_of(t)).value_counts().sort_index().items():
                cov.append(dict(domain=dom, name=name, water_year=int(wy), hours=int(n)))
    var_table = pd.DataFrame(vrows)
    var_table.to_csv(out / "variables.csv", index=False)
    cov = pd.DataFrame(cov)
    if len(cov):
        cov["hours_expected"] = [int((pd.Timestamp(f"{w}-10-01") - pd.Timestamp(f"{w - 1}-10-01")) / pd.Timedelta(hours=a.step_hours))
                                 for w in cov["water_year"]]
    cov.to_csv(out / "coverage.csv", index=False)

    # Per water-year folder.
    wy_rows = []
    groups = defaultdict(list)
    for r in ok:
        if r["water_year"] > 0:
            groups[(r["domain"], r["water_year"])].append(r)
    for (dom, wy), rs in sorted(groups.items()):
        t = np.unique(np.concatenate([file_times(r) for r in rs]))
        start, end = (pd.Timestamp(f"{y}-10-01").as_unit("ns").value for y in (wy - 1, wy))
        inside = t[(t >= start) & (t < end)]
        wy_rows.append(dict(domain=dom, water_year=wy, volumes=",".join(sorted({r["volume"] for r in rs})),
                            n_files=len(rs), n_variables=len({v for r in rs for v in r["vars"]}),
                            gb=round(sum(r["size_bytes"] for r in rs) / 1e9, 1),
                            first_time=str(pd.Timestamp(t[0])) if len(t) else "",
                            last_time=str(pd.Timestamp(t[-1])) if len(t) else "",
                            hours_in_wy=len(inside), hours_expected=int((end - start) // step_ns),
                            hours_outside_wy=len(t) - len(inside)))
    wy_table = pd.DataFrame(wy_rows)
    wy_table.to_csv(out / "water_years.csv", index=False)

    # Per domain: grid, projection, model configuration.
    summary = {}
    for dom in sorted({r["domain"] for r in ok}):
        rs = [r for r in ok if r["domain"] == dom]
        g = next((r["globals"] for r in rs if "MAP_PROJ" in r["globals"]), rs[0]["globals"])
        changed = {}
        for k in GLOBAL_KEYS:
            if k == "history":
                continue
            vals = {json.dumps(r["globals"].get(k)) for r in rs if k in r["globals"]}
            if len(vals) > 1:
                changed[k] = sorted(vals)[:10]
        summary[dom] = {
            "n_files": len(rs), "total_bytes": int(sum(r["size_bytes"] for r in rs)),
            "n_variables": int((var_table["domain"] == dom).sum()) if len(var_table) else 0,
            "files_with_projection_attrs": sum("MAP_PROJ" in r["globals"] for r in rs),
            "files_with_latlon": sum(r["has_latlon"] for r in rs),
            "grid": {k: max(r[f"n_{k}"] for r in rs) for k in DIM_KEYS},
            "globals": g, "globals_that_change": changed,
            "netcdf_formats": dict(Counter(r["format"] for r in rs)),
            "step_seconds_seen": dict(Counter(str(r["step_seconds"]) for r in rs)),
        }
    summary["_files_with_errors"] = files.loc[files["error"] != "", ["path", "error"]].head(200).to_dict("records")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(out / "inventory_report.md", summary, files, var_table, cov, wy_table)
    print(f"wrote manifest for {len(files):,} files ({len(rows) - len(ok)} unreadable) to {out}")


def write_report(path, summary, files, var_table, cov, wy_table):
    L = ["# WRF output inventory", ""]
    for dom, s in summary.items():
        if dom.startswith("_"):
            continue
        g = s["globals"]
        L += [f"## Domain {dom}", "",
              f"- Files: {s['n_files']:,} ({s['total_bytes'] / 1e12:.2f} TB), {s['n_variables']} variables",
              f"- Grid: {s['grid']}, DX={g.get('DX')} m",
              f"- Projection: MAP_PROJ={g.get('MAP_PROJ')}, TRUELAT1={g.get('TRUELAT1')}, TRUELAT2={g.get('TRUELAT2')}, "
              f"STAND_LON={g.get('STAND_LON')}, MOAD_CEN_LAT={g.get('MOAD_CEN_LAT')}, CEN_LAT={g.get('CEN_LAT')}, "
              f"CEN_LON={g.get('CEN_LON')}",
              f"- Files carrying projection attributes: {s['files_with_projection_attrs']:,}; carrying XLAT/XLONG: "
              f"{s['files_with_latlon']:,}",
              f"- Corners (lat, lon): {g.get('corners_latlon')}",
              f"- Model: {str(g.get('TITLE', '')).strip()}",
              f"- Physics: MP={g.get('MP_PHYSICS')}, PBL={g.get('BL_PBL_PHYSICS')}, SFCLAY={g.get('SF_SFCLAY_PHYSICS')}, "
              f"LSM={g.get('SF_SURFACE_PHYSICS')}, CU={g.get('CU_PHYSICS')}, LW={g.get('RA_LW_PHYSICS')}, "
              f"SW={g.get('RA_SW_PHYSICS')}, GRID_FDDA={g.get('GRID_FDDA')}, SST_UPDATE={g.get('SST_UPDATE')}",
              f"- Time steps between outputs (s, count of files): {s['step_seconds_seen']}",
              f"- NetCDF formats: {s['netcdf_formats']}",
              f"- Post-processing history (first file): {str(g.get('history', ''))[:300]}"]
        if s["globals_that_change"]:
            L.append(f"- Attributes that differ between files: {list(s['globals_that_change'])}")
        L.append("")
    if len(wy_table):
        L += ["## Water years", "",
              "| domain | WY | volume | files | variables | GB | first | last | hours / expected | outside WY |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in wy_table.iterrows():
            L.append(f"| {r['domain']} | {r['water_year']} | {r['volumes']} | {r['n_files']:,} | {r['n_variables']} | "
                     f"{r['gb']:,} | {r['first_time']} | {r['last_time']} | {r['hours_in_wy']:,} / "
                     f"{r['hours_expected']:,} | {r['hours_outside_wy']} |")
        L.append("")
    if len(cov):
        short = cov[cov["hours"] < cov["hours_expected"]]
        L += ["## Incomplete variable-years", "",
              f"{len(short)} of {len(cov)} variable-water-year pairs are short of a full year"
              + (" (first 60 shown; all in coverage.csv)." if len(short) > 60 else "."), ""]
        if len(short):
            L += ["| domain | variable | WY | hours / expected |", "| --- | --- | --- | --- |"]
            L += [f"| {r['domain']} | {r['name']} | {r['water_year']} | {r['hours']:,} / {r['hours_expected']:,} |"
                  for _, r in short.head(60).iterrows()]
        L.append("")
    L += [f"Files that failed to open: {int((files['error'] != '').sum())}", "", "## Variables", "",
          "| domain | name | dims | units | description | files | GB | first | last | missing h | compressed | chunking |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in var_table.sort_values("gb", ascending=False).iterrows():
        L.append(f"| {r['domain']} | {r['name']} | {r['dims']} | {r['units']} | {r['description']} | {r['n_files']} | "
                 f"{r['gb']:,} | {r['first_time']} | {r['last_time']} | {r['missing_hours']:,} | {r['compressed']} | "
                 f"{r['chunking']} |")
    Path(path).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
