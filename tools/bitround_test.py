#!/usr/bin/env python
"""How much smaller would atmos_3d be with bitrounding, and what would it cost?

Bitrounding (Klöwer et al. 2021, doi:10.1038/s43588-021-00156-2) keeps the first
`keepbits` of a float32's 23 mantissa bits, rounding to nearest, and zeroes the
rest; the zeros then compress well. This is what numcodecs.BitRound (usable as a
Zarr codec) and netCDF's quantize do. The rounding error is at most 2^-(keepbits+1)
of each value.

For random hours in --start..--end, for each atmos_3d variable, this script:
  * measures the bitwise real information (as xbitinfo does): the mutual
    information between each bit and the same bit in the neighbouring grid
    point along x, with insignificant values (99% binomial test) set to zero.
    keepbits at 99% and 99.9% of the information follow from its cumulative sum.
  * rounds every hour to keepbits 1..23 and compresses each result as the store
    does (Blosc zstd level 3), with byte shuffle (the store's setting) and with
    bitshuffle, which suits rounded data better.
  * records the rounding error, and for QVAPOR the error in precipitable water.

Sizes are reported against the store's current lossless codec, and scaled by the
variable's real bytes on disk to estimate the whole store. Writes to --out:
  report.md, bitround.json, bitinfo.png (information per bit, per variable)

Reads only; nothing in the stores changes.

    python tools/bitround_test.py --config config.yaml --hours 24
"""
import argparse
import json
import os
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import BitRound, Blosc
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrf_to_zarr as w  # noqa: E402
from benchmark_pw import column_weights, p_top, pw_from, stored_sizes, timed, log  # noqa: E402

KEEPBITS = list(range(1, 24))            # 23 = lossless
SHOW = [4, 6, 8, 10, 12, 16]             # keepbits shown in the report tables
SHUFFLES = {"shuffle": Blosc.SHUFFLE, "bitshuffle": Blosc.BITSHUFFLE}
NBITS, EXP_BITS = 32, 8                  # float32: sign, 8 exponent, 23 mantissa


def bit_counts(a):
    """Counts for the mutual information of each bit with its x neighbour.

    Returns an (32, 3) array of [n(bit=1 here), n(bit=1 next), n(both)], bit 0
    being the sign bit (most significant first), and the number of pairs."""
    u = np.ascontiguousarray(a, dtype="f4").view("u4")
    here, nxt = u[..., :-1], u[..., 1:]
    ok = np.isfinite(a[..., :-1]) & np.isfinite(a[..., 1:])
    here, nxt = here[ok], nxt[ok]
    both = here & nxt
    c = np.empty((NBITS, 3), "i8")
    for i in range(NBITS):
        m = np.uint32(1 << (NBITS - 1 - i))
        c[i] = (np.count_nonzero(here & m), np.count_nonzero(nxt & m), np.count_nonzero(both & m))
    return c, int(here.size)


def bit_information(c, n, confidence=0.99):
    """Mutual information per bit (bits), zeroed where not significant."""
    p1a, p1b, p11 = c[:, 0] / n, c[:, 1] / n, c[:, 2] / n
    joint = np.stack([1 - p1a - p1b + p11, p1b - p11, p1a - p11, p11], axis=1)   # 00, 01, 10, 11
    marg_a = np.stack([1 - p1a, 1 - p1a, p1a, p1a], axis=1)
    marg_b = np.stack([1 - p1b, p1b, 1 - p1b, p1b], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(joint > 0, joint * np.log2(joint / (marg_a * marg_b)), 0.0)
    mi = np.clip(terms.sum(axis=1), 0, None)
    # Two independent random bits still show some MI in a finite sample: the
    # threshold is 1 - H(p) for p at the binomial confidence bound.
    p = 0.5 + 0.5 * NormalDist().inv_cdf(1 - (1 - confidence) / 2) / np.sqrt(n)
    threshold = 1 - (-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
    mi[mi <= threshold] = 0
    return mi


def keepbits_for(info, level):
    """Mantissa bits needed to keep `level` of the total information (as xbitinfo)."""
    if info.sum() == 0:
        return None                               # nothing significant: no basis to choose
    cdf = np.cumsum(info) / info.sum()
    last = int(np.argmax(cdf >= level))          # index of the last bit needed
    return int(np.clip(last + 1 - (1 + EXP_BITS), 0, 23))


def one_hour(a, keepbits, clevel, cname, psfc=None, wts=None, ptop=None):
    """Compressed sizes and errors at each keepbits for one variable at one hour."""
    a = np.ascontiguousarray(a, dtype="f4")
    finite = np.isfinite(a)
    std = float(np.nanstd(a)) or 1.0
    nz = finite & (a != 0)
    out = {"sizes": {s: {} for s in SHUFFLES}, "rmse_over_std": {}, "max_rel": {}, "max_abs": {},
           "pw_max_abs_mm": {}}
    out["counts"], out["npairs"] = bit_counts(a)
    ref_pw = pw_from(a.astype("f8"), psfc, wts, ptop) if psfc is not None else None
    for kb in keepbits:
        r = a if kb == 23 else BitRound(kb).encode(a).view("f4").reshape(a.shape)
        for s, code in SHUFFLES.items():
            out["sizes"][s][kb] = len(Blosc(cname=cname, clevel=clevel, shuffle=code).encode(r))
        d = (r.astype("f8") - a)[finite]
        out["rmse_over_std"][kb] = float(np.sqrt(np.mean(d * d)) / std)
        out["max_abs"][kb] = float(np.abs(d).max()) if d.size else 0.0
        out["max_rel"][kb] = float(np.max(np.abs(r[nz].astype("f8") - a[nz]) / np.abs(a[nz]))) if nz.any() else 0.0
        if ref_pw is not None:
            out["pw_max_abs_mm"][kb] = float(np.nanmax(np.abs(pw_from(r.astype("f8"), psfc, wts, ptop) - ref_pw)))
    out["raw"] = a.nbytes
    return out


def merge(acc, h):
    if acc is None:
        return {"sizes": {s: dict(v) for s, v in h["sizes"].items()}, "counts": h["counts"].copy(), "npairs": h["npairs"],
                "raw": h["raw"], "rmse_over_std": {k: [v] for k, v in h["rmse_over_std"].items()},
                "max_rel": dict(h["max_rel"]), "max_abs": dict(h["max_abs"]),
                "pw_max_abs_mm": dict(h["pw_max_abs_mm"]), "hours": 1}
    for s in SHUFFLES:
        for kb, b in h["sizes"][s].items():
            acc["sizes"][s][kb] += b
    acc["counts"] += h["counts"]
    acc["npairs"] += h["npairs"]
    acc["raw"] += h["raw"]
    acc["hours"] += 1
    for kb in h["rmse_over_std"]:
        acc["rmse_over_std"][kb].append(h["rmse_over_std"][kb])
        acc["max_rel"][kb] = max(acc["max_rel"][kb], h["max_rel"][kb])
        acc["max_abs"][kb] = max(acc["max_abs"][kb], h["max_abs"][kb])
        if kb in h["pw_max_abs_mm"]:
            acc["pw_max_abs_mm"][kb] = max(acc["pw_max_abs_mm"][kb], h["pw_max_abs_mm"][kb])
    return acc


def plot_info(info, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(info)
    m = np.array([info[v]["info"] for v in names])
    fig, ax = plt.subplots(figsize=(12, 0.4 * len(names) + 1.6), layout="constrained")
    im = ax.imshow(m, aspect="auto", cmap="viridis", vmin=0, vmax=max(1e-3, float(m.max())))
    ax.set_yticks(range(len(names)), names)
    ax.set_xticks([0, 1, 5, 9, 14, 19, 24, 29, 31], ["sign", "e1", "e5", "m1", "m6", "m11", "m16", "m21", "m23"])
    ax.axvline(0.5, color="w", lw=0.8)
    ax.axvline(8.5, color="w", lw=0.8)
    for j, v in enumerate(names):
        for lvl, mk in ((0.99, "|"), (0.999, "x")):
            kb = info[v][f"keepbits_{lvl}"]
            if kb is not None:
                ax.plot(8.5 + kb, j, marker=mk, color="w" if mk == "|" else "#fdae61", ms=10 if mk == "|" else 7, mew=2)
    ax.set_xlabel("bit (sign, exponent e1-e8, mantissa m1-m23); white | keeps 99% of the information, orange x 99.9%")
    ax.set_title("Real bitwise information in atmos_3d (mutual information with the x neighbour)")
    fig.colorbar(im, ax=ax, label="information (bits)", pad=0.01)
    fig.savefig(png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--store", default="atmos_3d")
    p.add_argument("--start", default="1999-10-01", help="sample hours from here (default: WY2000)")
    p.add_argument("--end", default="2000-09-30T23:00")
    p.add_argument("--hours", type=int, default=24, help="random hours sampled")
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--ptop", type=float, default=None, help="model top pressure in Pa (default: from grid_file)")
    p.add_argument("--no-sizes", action="store_true", help="estimate the store from the sample instead of "
                                                           "listing bytes on disk")
    p.add_argument("--out", default=None, help="output folder (default bitround/<timestamp>)")
    a = p.parse_args()

    t0 = time.perf_counter()
    cfg = w.load_config(a.config)
    comp = cfg["output"].get("compressor", {"cname": "zstd", "clevel": 3})
    numcodecs.blosc.use_threads = False          # parallel over variables instead
    out = Path(a.out or f"bitround/{pd.Timestamp.now():%Y%m%d-%H%M}")
    out.mkdir(parents=True, exist_ok=True)

    ds = xr.open_zarr(w.store_path(cfg, a.store), consolidated=True)
    names = timed(ds)
    times = pd.DatetimeIndex(ds.time.values)
    good = np.flatnonzero((ds["hour_present"].values == 1) & (times >= pd.Timestamp(a.start)) &
                          (times <= pd.Timestamp(a.end)))
    if not len(good):
        sys.exit(f"no complete hours between {a.start} and {a.end}")
    hours = np.sort(np.random.default_rng(0).choice(good, min(a.hours, len(good)), replace=False))

    pw = None
    if "QVAPOR" in names:
        sm = xr.open_zarr(w.store_path(cfg, "surface_met"), consolidated=True)
        st = xr.open_zarr(w.store_path(cfg, "static"), consolidated=True)
        pw = (sm, column_weights(cfg, st, ds.sizes["level"]).values, p_top(cfg, a.ptop))
    log(f"{a.store}: {len(names)} variables, {len(hours)} hours from {times[hours[0]]:%Y-%m-%d} to "
        f"{times[hours[-1]]:%Y-%m-%d}, {a.workers} threads")

    acc = {v: None for v in names}
    with ThreadPoolExecutor(a.workers) as pool:
        for n, i in enumerate(hours, 1):
            fields = ds[names].isel(time=int(i)).load()
            jobs = {}
            for v in names:
                kw = {}
                if v == "QVAPOR" and pw is not None:
                    kw = dict(psfc=pw[0]["PSFC"].isel(time=int(i)).values.astype("f8"), wts=pw[1], ptop=pw[2])
                jobs[v] = pool.submit(one_hour, fields[v].values, KEEPBITS, comp.get("clevel", 3),
                                      comp.get("cname", "zstd"), **kw)
            for v, f in jobs.items():
                acc[v] = merge(acc[v], f.result())
            log(f"  hour {n} of {len(hours)} ({times[i]:%Y-%m-%d %H}h) done")

    on_disk = None
    if not a.no_sizes:
        log("listing bytes on disk per variable")
        on_disk = stored_sizes(w.store_path(cfg, a.store), names)

    res = {"store": a.store, "hours": [str(times[i]) for i in hours], "codec": comp, "variables": {}}
    for v in names:
        r = acc[v]
        info = bit_information(r["counts"], r["npairs"])
        lossless = r["sizes"]["shuffle"][23]
        disk = on_disk[v] if on_disk else lossless / r["raw"] * ds[v].nbytes
        rel = {s: {kb: b / lossless for kb, b in r["sizes"][s].items()} for s in SHUFFLES}
        res["variables"][v] = {
            "info": info.tolist(), "info_total_bits": float(info.sum()),
            "keepbits_0.99": keepbits_for(info, 0.99), "keepbits_0.999": keepbits_for(info, 0.999),
            "sample_ratio_lossless": r["raw"] / lossless, "bytes_on_disk_now": int(disk),
            "size_vs_now": rel, "rmse_over_std": {kb: float(np.mean(x)) for kb, x in r["rmse_over_std"].items()},
            "max_rel_err": r["max_rel"], "max_abs_err": r["max_abs"], "units": ds[v].attrs.get("units", ""),
        }
        if r["pw_max_abs_mm"]:
            res["pw_max_abs_mm"] = r["pw_max_abs_mm"]

    ru = resource.getrusage(resource.RUSAGE_SELF)
    res.update(total_wall_s=time.perf_counter() - t0, cpu_s=ru.ru_utime + ru.ru_stime,
               peak_rss_GB=ru.ru_maxrss / 1e6, host=os.uname().nodename, disk_source="listed" if on_disk else "sample")
    (out / "bitround.json").write_text(json.dumps(res, indent=1))
    plot_info(res["variables"], out / "bitinfo.png")
    (out / "report.md").write_text(report(res, out))
    print("\n" + (out / "report.md").read_text())


def report(r, out):
    V = r["variables"]
    now = sum(v["bytes_on_disk_now"] for v in V.values())

    def total(kb_of, shuffle):
        return sum(v["bytes_on_disk_now"] * v["size_vs_now"][shuffle][kb_of(name)] for name, v in V.items())

    def kb_or_lossless(k):
        return lambda name: 23 if V[name][k] is None else max(1, V[name][k])

    L = [f"# Bitrounding test: {r['store']} on {r['host']}", "",
         f"- {len(r['hours'])} random complete hours, {r['hours'][0][:10]} to {r['hours'][-1][:10]}; "
         f"wall time {r['total_wall_s'] / 60:.1f} min, peak memory {r['peak_rss_GB']:.1f} GB",
         f"- store size now: {now / 1e12:.2f} TB ({'bytes on disk' if r['disk_source'] == 'listed' else 'estimated from the sample'}); "
         f"codec Blosc {r['codec'].get('cname', 'zstd')} level {r['codec'].get('clevel', 3)} with byte shuffle, lossless", "",
         "## Whole-store size estimate", "",
         "| keepbits | with byte shuffle (TB) | with bitshuffle (TB) |", "|---|---|---|",
         f"| 23 (lossless) | {now / 1e12:.2f} | {total(lambda n: 23, 'bitshuffle') / 1e12:.2f} |"]
    for kb in reversed(SHOW):
        L.append(f"| {kb} for every variable | {total(lambda n, k=kb: k, 'shuffle') / 1e12:.2f} | "
                 f"{total(lambda n, k=kb: k, 'bitshuffle') / 1e12:.2f} |")
    for k, label in (("keepbits_0.999", "per variable, 99.9% of information"), ("keepbits_0.99", "per variable, 99% of information")):
        L.append(f"| {label} | {total(kb_or_lossless(k), 'shuffle') / 1e12:.2f} | "
                 f"{total(kb_or_lossless(k), 'bitshuffle') / 1e12:.2f} |")
    L += ["", "(A variable with no significant information, which happens for nearly empty hydrometeor "
              "fields, is counted as lossless in the per-variable rows.)", "",
          "## Per variable", "",
          "Size is relative to the store now, with bitshuffle. Error is the RMS rounding error as a share of "
          "the field's standard deviation; the worst relative error on any value is at most 2^-(keepbits+1).", "",
          "| variable | units | info (bits) | keepbits 99% | keepbits 99.9% | size at 99% | size at 99.9% | "
          + " | ".join(f"size kb={k}" for k in reversed(SHOW)) + " | RMS err kb=8 |",
          "|---" * (8 + len(SHOW)) + "|"]
    for name, v in sorted(V.items(), key=lambda kv: -kv[1]["bytes_on_disk_now"]):
        s = v["size_vs_now"]["bitshuffle"]
        k99, k999 = v["keepbits_0.99"], v["keepbits_0.999"]
        f = lambda k: "-" if k is None else f"{s[max(1, k)]:.2f}"  # noqa: E731
        L.append(f"| {name} | {v['units']} | {v['info_total_bits']:.1f} | {'-' if k99 is None else k99} | "
                 f"{'-' if k999 is None else k999} | {f(k99)} | {f(k999)} | "
                 + " | ".join(f"{s[k]:.2f}" for k in reversed(SHOW)) + f" | {v['rmse_over_std'][8]:.1e} |")
    if "pw_max_abs_mm" in r:
        L += ["", "## Precipitable water from rounded QVAPOR", "",
              "Worst error in PW at any grid point and hour, against PW from the unrounded store.", "",
              "| keepbits | " + " | ".join(str(k) for k in SHOW) + f" | 99% ({V['QVAPOR']['keepbits_0.99']}) |",
              "|---" * (len(SHOW) + 2) + "|",
              "| max PW error (mm) | " + " | ".join(f"{r['pw_max_abs_mm'][k]:.1e}" for k in SHOW)
              + f" | {r['pw_max_abs_mm'][max(1, V['QVAPOR']['keepbits_0.99'] or 23)]:.1e} |"]
    L += ["", f"Files in {out}: bitinfo.png (information per bit), bitround.json (every keepbits 1-23)"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
