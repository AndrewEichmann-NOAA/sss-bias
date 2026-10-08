#!/usr/bin/env python3
"""
Fetch real Argo GDAC QC (PSAL_QC/PSAL_ADJUSTED_QC) and delayed-mode salinity
for every near-surface obsForge Argo profile across the project's full
matchup date range, to replace the ad hoc valid-range filter (DESIGN.md 30)
with real per-obs quality control (DESIGN.md 31's segment test, scaled up).

Chunked by month (like fetch_raw_argo.py), each month's result saved and
skipped if already present -- safe to interrupt and resume. For each month:
  1. Load obsForge near-surface Argo obs with NO valid-range filter
     (min_salinity=0, max_salinity=45 -- obsForge's own crude bound), so
     real QC does the filtering rather than a numeric heuristic.
  2. Match each to the GDAC profile index (argopy.ArgoIndex) by
     (lat, lon, datetime).
  3. Fetch each matched profile file directly over HTTPS (bypassing
     argopy's own profile() fetcher, which errors on this dataset's
     filename pattern) to recover real QC and the delayed-mode value.
"""

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from build_raw_smap_matchups import load_argo_for_window
from enrich_argo_metadata import load_index, match_to_index
from fetch_raw_argo import month_chunks

GDAC_BASE = "https://data-argo.ifremer.fr/dac/"
MAX_DEPTH = 5.0
QC_ORDER = {'1': 1, '2': 2, '5': 2, '8': 2, '3': 3, '4': 4, '9': 9, ' ': 9, '': 9}


def fetch_profile(file_path, session, retries=2):
    """Returns (delayed_salinity, worst_psal_qc, data_mode) for near-surface
    (PRES<=MAX_DEPTH) levels of a profile's primary (N_PROF=0) profile, or
    None if the fetch/parse failed.
    """
    url = GDAC_BASE + file_path
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=20)
            if r.status_code != 200:
                return None
            import xarray as xr
            ds = xr.open_dataset(io.BytesIO(r.content))

            pres = ds['PRES'].values[0]
            near_surface = pres <= MAX_DEPTH
            if not near_surface.any():
                return None

            psal_adj = ds['PSAL_ADJUSTED'].values[0] if 'PSAL_ADJUSTED' in ds else np.full_like(pres, np.nan)
            psal_raw = ds['PSAL'].values[0] if 'PSAL' in ds else np.full_like(pres, np.nan)
            vals = np.where(np.isfinite(psal_adj), psal_adj, psal_raw)[near_surface]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                return None
            delayed_salinity = float(np.mean(vals))

            qc_var = 'PSAL_ADJUSTED_QC' if 'PSAL_ADJUSTED_QC' in ds else 'PSAL_QC'
            qc_raw = ds[qc_var].values[0][near_surface] if qc_var in ds else np.array([])
            qc_codes = [(c.decode() if isinstance(c, bytes) else str(c)).strip() for c in qc_raw]
            qc_ranks = [QC_ORDER.get(c, 9) for c in qc_codes] or [9]
            worst_qc = max(qc_ranks)

            data_mode = ds['DATA_MODE'].values[0] if 'DATA_MODE' in ds else None
            mode = data_mode.decode() if isinstance(data_mode, bytes) else str(data_mode)
            return delayed_salinity, worst_qc, mode
        except Exception:
            time.sleep(1)
    return None


def fetch_month(t0, t1, args):
    argo_df = load_argo_for_window(args.argo_base_dir, t0, t1, args.max_depth,
                                    min_salinity=0.0, max_salinity=45.0, verbose=False)
    if argo_df.empty:
        return argo_df

    index_df = load_index(t0.strftime('%Y-%m-%d'), t1.strftime('%Y-%m-%d'), args.cachedir)
    max_time_delta = pd.Timedelta(minutes=args.max_time_delta_minutes)
    argo_renamed = argo_df.rename(columns={'lat': 'argo_lat', 'lon': 'argo_lon', 'datetime': 'argo_datetime'})
    matched = match_to_index(argo_renamed, index_df, args.max_dist_km, max_time_delta)
    matched = matched[matched['matched']].reset_index(drop=True)
    if matched.empty:
        return matched

    unique_files = matched['file'].drop_duplicates().tolist()
    results = {}
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_profile, f, session): f for f in unique_files}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    matched['gdac_result'] = matched['file'].map(results)
    matched = matched[matched['gdac_result'].notna()].copy()
    matched['delayed_salinity'] = matched['gdac_result'].apply(lambda x: x[0])
    matched['worst_psal_qc'] = matched['gdac_result'].apply(lambda x: x[1])
    matched['data_mode'] = matched['gdac_result'].apply(lambda x: x[2])
    matched['discrepancy'] = matched['salinity'] - matched['delayed_salinity']
    return matched.drop(columns=['gdac_result'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--argo-base-dir', default='/Users/afeman/Desktop/work/sss-bias/data/common_obsForge')
    parser.add_argument('--start-date', default='2022-06-01')
    parser.add_argument('--end-date', default='2025-05-01')
    parser.add_argument('--max-depth', type=float, default=MAX_DEPTH)
    parser.add_argument('--cachedir', default='/Users/afeman/Desktop/work/sss-bias/data/argopy_cache_v2')
    parser.add_argument('--max-dist-km', type=float, default=1.0)
    parser.add_argument('--max-time-delta-minutes', type=float, default=10.0)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--out-dir', default='/Users/afeman/Desktop/work/sss-bias/data/gdac_argo_qc')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = month_chunks(args.start_date, args.end_date, chunk_months=1)
    print(f"{len(chunks)} monthly chunks, {args.start_date} to {args.end_date}")

    total_rows = 0
    for i, (t0, t1) in enumerate(chunks):
        out_path = out_dir / f"gdac_argo_qc_{t0.strftime('%Y%m')}.parquet"
        if out_path.exists():
            print(f"  [{i+1}/{len(chunks)}] {t0.date()} to {t1.date()}: already fetched, skipping")
            continue

        t_start = time.time()
        print(f"  [{i+1}/{len(chunks)}] {t0.date()} to {t1.date()}: fetching...")
        df = fetch_month(t0, t1, args)
        if df is None or df.empty:
            print(f"    0 rows")
            df = pd.DataFrame()
        df.to_parquet(out_path, index=False)
        total_rows += len(df)
        print(f"    {len(df)} rows -> {out_path} ({time.time()-t_start:.0f}s)")

    print(f"\nDone. Total rows fetched this run: {total_rows}")


if __name__ == '__main__':
    main()
