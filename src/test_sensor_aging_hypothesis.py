#!/usr/bin/env python3
"""
Test the sensor-aging hypothesis from the float-leakage diagnostic: is the
gap between "seen" (long-lived, in train) and "unseen" (newly-deployed)
floats explained by real-time-vs-delayed-mode salinity discrepancy being
larger for older floats?

For each test-set row with a recovered WMO ID, fetches the actual GDAC
profile file directly over HTTPS (bypassing argopy's profile() fetcher,
which errors on this dataset's filename pattern -- see session notes) and
compares our existing real-time obsForge-derived near-surface salinity
against the same profile's delayed-mode-preferred (PSAL_ADJUSTED, falling
back to PSAL) near-surface average. Compares |discrepancy| between the
'seen' and 'unseen' groups.
"""

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from features import add_features, split_data_naive

GDAC_BASE = "https://data-argo.ifremer.fr/dac/"
MAX_DEPTH = 5.0


def fetch_near_surface_salinity(file_path, session, retries=2):
    url = GDAC_BASE + file_path
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=20)
            if r.status_code != 200:
                return None
            import xarray as xr
            ds = xr.open_dataset(io.BytesIO(r.content))

            # Primary profile only (N_PROF=0); near-surface PRES<=MAX_DEPTH.
            pres = ds['PRES'].values[0]
            near_surface = pres <= MAX_DEPTH
            if not near_surface.any():
                return None

            if 'PSAL_ADJUSTED' in ds:
                psal = ds['PSAL_ADJUSTED'].values[0]
            else:
                psal = np.full_like(pres, np.nan)
            psal_raw = ds['PSAL'].values[0] if 'PSAL' in ds else np.full_like(pres, np.nan)

            vals = np.where(np.isfinite(psal), psal, psal_raw)[near_surface]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                return None
            data_mode = ds['DATA_MODE'].values[0] if 'DATA_MODE' in ds else None
            mode = data_mode.decode() if isinstance(data_mode, bytes) else str(data_mode)
            return float(np.mean(vals)), mode
        except Exception:
            time.sleep(1)
    return None


def build_group_table(sensor):
    df = pd.read_parquet(f'/Users/afeman/Desktop/work/sss-bias/data/matchups/{sensor}_argo_matchups.parquet')
    df = add_features(df)
    train, val, test = split_data_naive(df)
    train_floats = set(train['wmo'].dropna())

    def group_of(row):
        if pd.isna(row):
            return 'unknown'
        return 'seen' if row in train_floats else 'unseen'

    test = test.copy()
    test['float_group'] = test['wmo'].apply(group_of)
    return test[test['float_group'] != 'unknown'].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lookup', default='/Users/afeman/Desktop/work/sss-bias/data/matchups/argo_wmo_lookup.parquet')
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--out', default='/Users/afeman/Desktop/work/sss-bias/data/matchups/aging_hypothesis_check.parquet')
    args = parser.parse_args()

    lookup = pd.read_parquet(args.lookup, columns=['argo_lat', 'argo_lon', 'argo_datetime', 'wmo', 'cyc', 'file'])

    frames = []
    for sensor in ['smap', 'smos']:
        g = build_group_table(sensor)
        g['sensor'] = sensor
        frames.append(g)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.merge(lookup, on=['argo_lat', 'argo_lon', 'argo_datetime'], how='left')
    combined = combined.dropna(subset=['file'])

    unique_files = combined[['file']].drop_duplicates()
    print(f"{len(combined)} test rows (seen+unseen) across both sensors, "
          f"{len(unique_files)} unique profile files to fetch")

    results = {}
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_near_surface_salinity, f, session): f
                   for f in unique_files['file']}
        done = 0
        for fut in as_completed(futures):
            f = futures[fut]
            results[f] = fut.result()
            done += 1
            if done % 500 == 0:
                print(f"  fetched {done}/{len(unique_files)}")

    n_ok = sum(1 for v in results.values() if v is not None)
    print(f"Successfully fetched {n_ok}/{len(unique_files)} profile files")

    combined['delayed_result'] = combined['file'].map(results)
    combined = combined[combined['delayed_result'].notna()]
    combined['delayed_salinity'] = combined['delayed_result'].apply(lambda x: x[0])
    combined['data_mode'] = combined['delayed_result'].apply(lambda x: x[1])
    combined['discrepancy'] = combined['argo_salinity'] - combined['delayed_salinity']
    combined['abs_discrepancy'] = combined['discrepancy'].abs()

    combined.drop(columns=['delayed_result']).to_parquet(args.out, index=False)
    print(f"\nSaved to {args.out}")

    print("\n=== |real-time - delayed-mode-preferred| discrepancy by float group ===")
    for sensor in ['smap', 'smos']:
        sub = combined[combined['sensor'] == sensor]
        print(f"\n{sensor.upper()}:")
        print(sub.groupby('float_group')['abs_discrepancy'].describe()[['count', 'mean', '50%', 'std']])

    print("\n=== DATA_MODE distribution by float group (combined) ===")
    print(pd.crosstab(combined['float_group'], combined['data_mode']))


if __name__ == '__main__':
    main()
