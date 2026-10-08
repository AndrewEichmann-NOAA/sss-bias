#!/usr/bin/env python3
"""
First segment-scale test of whether raw Argo GDAC data offers a real
improvement over obsForge's IODA-processed Argo (DESIGN.md 14/17/30):
specifically, whether GDAC's own PSAL_QC would have caught the stuck-sensor
profiles that obsForge's crude valid-range filter only partially catches.

Segment: one calendar month of obsForge near-surface Argo obs, loaded with
NO valid-range filter applied (min_salinity=0, max_salinity=45 -- obsForge's
own crude bound, see build_matchups.py's load_argo_near_surface) so both the
filter's passes and its misses are visible side by side. Each profile is
matched to the GDAC profile index (argopy.ArgoIndex, cached locally by an
earlier enrich_argo_metadata.py run) by (lat, lon, datetime), then the
matched files are fetched directly over HTTPS -- bypassing argopy's own
profile() fetcher, which errors on this dataset's filename pattern, same
workaround as test_sensor_aging_hypothesis.py -- to recover real PSAL_QC and
the delayed-mode-preferred salinity value.
"""

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from build_raw_smap_matchups import load_argo_for_window
from enrich_argo_metadata import load_index, match_to_index

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--argo-base-dir', default='/Users/afeman/Desktop/work/sss-bias/data/common_obsForge')
    parser.add_argument('--start-date', default='2022-06-01')
    parser.add_argument('--end-date', default='2022-07-01')
    parser.add_argument('--max-depth', type=float, default=MAX_DEPTH)
    parser.add_argument('--cachedir', default='/Users/afeman/Desktop/work/sss-bias/data/argopy_cache')
    parser.add_argument('--max-dist-km', type=float, default=1.0)
    parser.add_argument('--max-time-delta-minutes', type=float, default=10.0)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--out', default='/Users/afeman/Desktop/work/sss-bias/data/matchups/gdac_qc_recovery_check.parquet')
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    print(f"Loading obsForge Argo near-surface obs, {args.start_date} to {args.end_date}, "
          f"NO valid-range filter (min_salinity=0, max_salinity=45)...")
    argo_df = load_argo_for_window(args.argo_base_dir, start_date, end_date,
                                    args.max_depth, min_salinity=0.0, max_salinity=45.0)
    print(f"  {len(argo_df)} unique near-surface profiles")

    print("Loading GDAC index and matching...")
    index_df = load_index(args.start_date, args.end_date, args.cachedir)
    print(f"  {len(index_df)} index records in range")
    max_time_delta = pd.Timedelta(minutes=args.max_time_delta_minutes)
    argo_renamed = argo_df.rename(columns={'lat': 'argo_lat', 'lon': 'argo_lon', 'datetime': 'argo_datetime'})
    matched = match_to_index(argo_renamed, index_df, args.max_dist_km, max_time_delta)
    matched = matched[matched['matched']].reset_index(drop=True)
    print(f"  {len(matched)}/{len(argo_df)} ({100*len(matched)/len(argo_df):.1f}%) matched to a GDAC file "
          f"within {args.max_dist_km}km/{args.max_time_delta_minutes}min")

    unique_files = matched['file'].drop_duplicates().tolist()
    print(f"\nFetching {len(unique_files)} unique profile files from GDAC...")
    results = {}
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_profile, f, session): f for f in unique_files}
        done = 0
        for fut in as_completed(futures):
            f = futures[fut]
            results[f] = fut.result()
            done += 1
            if done % 500 == 0:
                print(f"  fetched {done}/{len(unique_files)}")

    n_ok = sum(1 for v in results.values() if v is not None)
    print(f"Successfully fetched {n_ok}/{len(unique_files)} profile files")

    matched['gdac_result'] = matched['file'].map(results)
    matched = matched[matched['gdac_result'].notna()].copy()
    matched['delayed_salinity'] = matched['gdac_result'].apply(lambda x: x[0])
    matched['worst_psal_qc'] = matched['gdac_result'].apply(lambda x: x[1])
    matched['data_mode'] = matched['gdac_result'].apply(lambda x: x[2])
    matched['discrepancy'] = matched['salinity'] - matched['delayed_salinity']

    matched.drop(columns=['gdac_result']).to_parquet(args.out, index=False)
    print(f"\nSaved {len(matched)} rows to {args.out}")

    print("\n=== worst near-surface PSAL_QC vs. obsForge salinity range ===")
    for lo, hi, label in [(0, 20, '<20 (old filter rejects)'), (20, 30, '20-30 (old filter passes, new rejects)'),
                          (30, 45, '>=30 (both filters pass)')]:
        sub = matched[(matched['salinity'] >= lo) & (matched['salinity'] < hi)]
        if sub.empty:
            print(f"  {label}: 0 profiles")
            continue
        qc_counts = sub['worst_psal_qc'].value_counts().sort_index()
        print(f"  {label}: n={len(sub)}, worst_psal_qc counts: {qc_counts.to_dict()}")

    print("\n=== |obsForge - delayed-mode| discrepancy, overall ===")
    print(matched['discrepancy'].abs().describe())

    print("\n=== DATA_MODE distribution ===")
    print(matched['data_mode'].value_counts())


if __name__ == '__main__':
    main()
