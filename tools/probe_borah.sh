#!/usr/bin/env bash
# Report what this project needs to know about Borah and the WRF archive. Read-only; run on a
# login node from the repo folder and send the file back:
#   bash tools/probe_borah.sh > probe_$(hostname -s)_$(date +%Y%m%d).txt 2>&1
# Extends idaho-summer-precip's probe_env.sh with conda, storage on /bsushare, GPUs, and a
# look inside one water year of the WRF output. Writes nothing; takes a minute or two.
set -u
DATA=${DATA:-/bsushare/leaf-shared/subset-output-wrf}
WY_DIR=${WY_DIR:-$DATA/vol01/wrf_out/wy_1987/d02}
sec() { printf '\n===== %s =====\n' "$1"; }

sec "host, date"
hostname; date -u; cat /etc/os-release 2>/dev/null | head -2

sec "conda and python"
for t in conda mamba micromamba python3; do printf '%-11s %s\n' "$t" "$(command -v $t 2>/dev/null || echo -)"; done
echo "CONDA_EXE=${CONDA_EXE:-}"; echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
command -v conda >/dev/null && { conda --version; conda info --base; conda env list; conda config --show channels 2>/dev/null; }
ls -d "$HOME"/{miniforge3,miniconda3,mambaforge,anaconda3} 2>/dev/null
python3 --version 2>&1
if command -v conda >/dev/null && conda env list | grep -q '^wrf-zarr '; then
  echo "-- wrf-zarr env:"
  conda run -n wrf-zarr python -c "
import importlib
for m in ['xarray','zarr','dask','netCDF4','pyproj','pandas','yaml','s3fs','xwrf']:
    try: print(f'{m:8s}', importlib.import_module(m).__version__)
    except Exception as e: print(f'{m:8s} missing ({type(e).__name__})')"
fi

sec "storage: space, inodes, quotas"
for d in "$DATA" /bsushare/leaf-shared /bsushare /bsuhome /bsuscratch "$HOME"; do
  [ -d "$d" ] && { echo "-- $d"; df -h "$d" | tail -1; df -i "$d" | tail -1; }
done
quota -s 2>/dev/null || echo "quota: no output"
ls -ld /bsushare/leaf-shared 2>/dev/null; id -gn; groups
ls /bsushare/leaf-shared 2>/dev/null | head -30

sec "Slurm: CPU and GPU partitions, QOS"
sinfo -o '%P %a %l %D %c %m %G' 2>/dev/null
sacctmgr -nP show assoc user="$USER" format=account,partition,qos 2>/dev/null | head
sacctmgr -nP show qos format=name,maxwall,maxtresperuser,maxjobsperuser,maxsubmitjobsperuser 2>/dev/null | head -20
scontrol show config 2>/dev/null | grep -E 'MaxArraySize|MaxJobCount'

sec "modules with ncdump"
module -t avail 2>&1 | grep -iE '^netcdf|^nco|^cdo' | sort -u

sec "one water year of WRF output: $WY_DIR"
if [ -d "$WY_DIR" ]; then
  ls "$WY_DIR" | wc -l
  echo "-- size and file count per variable"
  ls -l "$WY_DIR"/*.nc | awk '{n=$NF; sub(/.*\//,"",n); sub(/_[0-9][0-9][0-9][0-9]-[0-9][0-9]\.nc$/,"",n); s[n]+=$5; c[n]++}
    END{for(k in s) printf "%-16s %3d files %10.1f GB\n",k,c[k],s[k]/1e9}' | sort -k4 -nr
  echo "-- non-.nc files"; ls "$WY_DIR" | grep -v '\.nc$'
  NCDUMP=$(command -v ncdump || true)
  if [ -z "$NCDUMP" ]; then
    module load netcdf-c/4.10.0/openmpi/4.1.3/gcc/12.1.0 2>/dev/null || module load netcdf/gcc/64/4.7.4 2>/dev/null
    NCDUMP=$(command -v ncdump || true)
  fi
  hdr() {  # header with chunking and compression (-s), via ncdump or python
    if [ -n "$NCDUMP" ]; then "$NCDUMP" -hs "$1" | head -${2:-80}
    else python3 -c "import netCDF4,sys; d=netCDF4.Dataset(sys.argv[1]); print(d); [print(v) for v in d.variables.values()]" "$1" 2>&1 | head -${2:-80}; fi
  }
  f2=$(ls "$WY_DIR"/T2_*.nc 2>/dev/null | head -1)
  f3=$(ls "$WY_DIR"/{U,T,QVAPOR,P,PH}_*.nc 2>/dev/null | head -1)
  fa=$(ls "$WY_DIR"/RAINNC_*.nc 2>/dev/null | head -1)
  for f in "$f2" "$f3" "$fa"; do [ -n "$f" ] && { echo; echo "-- $f"; hdr "$f" 90; }; done
  echo; echo "-- times in the T2 file"
  [ -n "$f2" ] && [ -n "$NCDUMP" ] && "$NCDUMP" -v Times "$f2" 2>/dev/null | grep -A3 '^ Times =' | head -4
  echo; echo "-- wrf_check.log (first 30 lines)"; head -30 "$WY_DIR/wrf_check.log" 2>/dev/null
  echo; echo "-- other water-year folders hold the same variables?"
  for d in "$DATA"/vol*/wrf_out/wy_*/d02; do
    printf '%s %s\n' "$(ls "$d" | sed 's/_[0-9-]*\.nc$//' | sort -u | md5sum | cut -c1-8)" "$d"
  done | sort | awk '{print $1}' | uniq -c
else
  echo "not found"
fi
ls "$DATA"/vol*/wrf_out/ 2>/dev/null | head -3
find "$DATA" -maxdepth 4 \( -iname 'wrfinput*' -o -iname 'geo_em*' -o -iname 'wrfout*' -o -iname 'namelist*' \) 2>/dev/null | head

sec "outbound network"
for u in https://github.com https://cds.climate.copernicus.eu https://s3.us-west-2.amazonaws.com https://www.openstoragenetwork.org; do
  printf '%-45s ' "$u"; curl -s -o /dev/null -m 10 -w '%{http_code}\n' "$u" || echo fail
done
