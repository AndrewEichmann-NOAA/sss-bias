#!/usr/bin/env python3
"""
Fetch near-surface Argo profiles DIRECTLY from the GDAC, independent of the
local obsForge archive (DESIGN.md 33/34): unlike fetch_gdac_argo_qc.py, which
only enriches profiles obsForge already has a local record of, this queries
the GDAC's own global profile index for every profile in a date range,
regardless of whether obsForge's local mirror has (or is missing) that
period -- e.g. recovers real coverage for the 2022-01-05 to 2022-04-30 local
obsForge gap found in 33.

For each profile in the index: fetches the file directly over HTTPS (same
approach as fetch_gdac_argo_qc.py/test_sensor_aging_hypothesis.py -- argopy's
own profile() fetcher errors on this dataset's filename pattern), keeps
near-surface (PRES<=max_depth) levels, and records the delayed-mode-
preferred salinity (PSAL_ADJUSTED, falling back to PSAL), worst QC among
those levels, and DATA_MODE. Position/time come from the GDAC index itself
(already profile-level lat/lon/date), not re-derived from the file.

Known gap: obsForge's `oceanBasin` categorical code has no GDAC equivalent,
so it is NOT populated here (left NaN) -- a basin_0..5 one-hot feature built
from this table would need a separate lat/lon-based basin classifier. Not
implemented; flagging rather than guessing.
"""

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from enrich_argo_metadata import load_index
from fetch_raw_argo import month_chunks

GDAC_BASE = "https://data-argo.ifremer.fr/dac/"
MAX_DEPTH = 5.0
QC_ORDER = {'1': 1, '2': 2, '5': 2, '8': 2, '3': 3, '4': 4, '9': 9, ' ': 9, '': 9}


def fetch_profile(file_path, session, max_depth, retries=2):
    """Returns (delayed_salinity, mean_depth, worst_psal_qc, data_mode) for
    near-surface (PRES<=max_depth) levels of a profile's primary (N_PROF=0)
    profile, or None if the fetch/parse failed or has no near-surface data.
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
            near_surface = pres <= max_depth
            if not near_surface.any():
                return None

            psal_adj = ds['PSAL_ADJUSTED'].values[0] if 'PSAL_ADJUSTED' in ds else np.full_like(pres, np.nan)
            psal_raw = ds['PSAL'].values[0] if 'PSAL' in ds else np.full_like(pres, np.nan)
            vals = np.where(np.isfinite(psal_adj), psal_adj, psal_raw)[near_surface]
            depths = pres[near_surface]
            valid = np.isfinite(vals)
            if not valid.any():
                return None
            delayed_salinity = float(np.mean(vals[valid]))
            mean_depth = float(np.mean(depths[valid]))

            qc_var = 'PSAL_ADJUSTED_QC' if 'PSAL_ADJUSTED_QC' in ds else 'PSAL_QC'
            qc_raw = ds[qc_var].values[0][near_surface] if qc_var in ds else np.array([])
            qc_codes = [(c.decode() if isinstance(c, bytes) else str(c)).strip() for c in qc_raw]
            qc_ranks = [QC_ORDER.get(c, 9) for c in qc_codes] or [9]
            worst_qc = max(qc_ranks)

            data_mode = ds['DATA_MODE'].values[0] if 'DATA_MODE' in ds else None
            mode = data_mode.decode() if isinstance(data_mode, bytes) else str(data_mode)
            return delayed_salinity, mean_depth, worst_qc, mode
        except Exception:
            time.sleep(1)
    return None


def fetch_month(t0, t1, args):
    index_df = load_index(t0.strftime('%Y-%m-%d'), t1.strftime('%Y-%m-%d'), args.cachedir)
    if index_df.empty:
        return index_df

    unique_files = index_df['file'].drop_duplicates().tolist()
    results = {}
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_profile, f, session, args.max_depth): f for f in unique_files}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    df = index_df.copy()
    df['fetch_result'] = df['file'].map(results)
    df = df[df['fetch_result'].notna()].copy()
    df['salinity'] = df['fetch_result'].apply(lambda x: x[0])
    df['depth'] = df['fetch_result'].apply(lambda x: x[1])
    df['worst_psal_qc'] = df['fetch_result'].apply(lambda x: x[2])
    df['data_mode'] = df['fetch_result'].apply(lambda x: x[3])
    df = df.drop(columns=['fetch_result']).rename(columns={'latitude': 'lat', 'longitude': 'lon', 'date': 'datetime'})
    return df[['lat', 'lon', 'datetime', 'wmo', 'cyc', 'file', 'dac', 'salinity', 'depth', 'worst_psal_qc', 'data_mode']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', default='2021-01-01')
    parser.add_argument('--end-date', default='2025-11-30')
    parser.add_argument('--max-depth', type=float, default=MAX_DEPTH)
    parser.add_argument('--cachedir', default='/Users/afeman/Desktop/work/sss-bias/data/argopy_cache_v2')
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--out-dir', default='/Users/afeman/Desktop/work/sss-bias/data/gdac_argo_direct')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = month_chunks(args.start_date, args.end_date, chunk_months=1)
    print(f"{len(chunks)} monthly chunks, {args.start_date} to {args.end_date} (global, obsForge-independent)")

    total_rows = 0
    for i, (t0, t1) in enumerate(chunks):
        out_path = out_dir / f"gdac_argo_direct_{t0.strftime('%Y%m')}.parquet"
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
