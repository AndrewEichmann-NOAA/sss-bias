#!/usr/bin/env python3
"""
Recover WMO float ID (and eventually real per-obs QC) for the Argo profiles
already present in our SMAP/SMOS matchup tables, by matching them against
the public GDAC's profile index -- see DESIGN.md 17.

This does NOT bulk-download raw Argo data. It only nearest-neighbor matches
our existing matchup rows' (lat, lon, datetime) against the index (already
downloaded once, queried locally, no network per-row) to identify which
GDAC file/WMO ID/cycle corresponds to each row we already have. Fetching the
actual profile files (for real PSAL_QC) is a separate, later step once this
matching is validated.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

warnings.filterwarnings('ignore')

EARTH_RADIUS_KM = 6371.0
DEFAULT_MAX_DIST_KM = 1.0
DEFAULT_MAX_TIME_DELTA = pd.Timedelta(minutes=10)


def load_index(start_date, end_date, cachedir):
    from argopy import ArgoIndex

    idx = ArgoIndex(index_file='core', cache=True, cachedir=cachedir)
    idx.load()
    idx.query.date([-180, 180, -90, 90, start_date, end_date])
    df = idx.to_dataframe()
    df = df[['date', 'latitude', 'longitude', 'wmo', 'cyc', 'file', 'dac']]
    # Some profiles (e.g. under ice, no GPS fix) have missing position -- can't
    # be spatially matched, so drop rather than let them break the BallTree.
    return df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)


def unique_argo_rows(matchup_paths):
    frames = []
    for path in matchup_paths:
        df = pd.read_parquet(path, columns=['argo_lat', 'argo_lon', 'argo_datetime'])
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=['argo_lat', 'argo_lon', 'argo_datetime']).reset_index(drop=True)


def match_to_index(argo_df, index_df, max_dist_km, max_time_delta):
    index_rad = np.radians(index_df[['latitude', 'longitude']].to_numpy())
    tree = BallTree(index_rad, metric='haversine')

    argo_rad = np.radians(argo_df[['argo_lat', 'argo_lon']].to_numpy())
    dist_rad, idx_pos = tree.query(argo_rad, k=1)
    dist_km = dist_rad[:, 0] * EARTH_RADIUS_KM
    idx_pos = idx_pos[:, 0]

    matched = index_df.iloc[idx_pos].reset_index(drop=True)
    result = argo_df.reset_index(drop=True).copy()
    result['wmo'] = matched['wmo'].to_numpy()
    result['cyc'] = matched['cyc'].to_numpy()
    result['file'] = matched['file'].to_numpy()
    result['dac'] = matched['dac'].to_numpy()
    result['index_date'] = matched['date'].to_numpy()
    result['dist_km'] = dist_km
    result['time_delta'] = (result['argo_datetime'] - result['index_date']).abs()

    result['matched'] = (result['dist_km'] <= max_dist_km) & (result['time_delta'] <= max_time_delta)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups-dir', default='/Users/afeman/Desktop/work/sss-bias/data/matchups')
    parser.add_argument('--cachedir', default='/Users/afeman/Desktop/work/sss-bias/data/argopy_cache')
    parser.add_argument('--start-date', default='2021-01-01')
    parser.add_argument('--end-date', default='2025-12-01')
    parser.add_argument('--max-dist-km', type=float, default=DEFAULT_MAX_DIST_KM)
    parser.add_argument('--max-time-delta-minutes', type=float, default=10.0)
    parser.add_argument('--out', default='/Users/afeman/Desktop/work/sss-bias/data/matchups/argo_wmo_lookup.parquet')
    args = parser.parse_args()

    matchup_paths = [
        f'{args.matchups_dir}/smap_argo_matchups.parquet',
        f'{args.matchups_dir}/smos_argo_matchups.parquet',
    ]

    print("Loading unique Argo rows from existing matchup tables...")
    argo_df = unique_argo_rows(matchup_paths)
    print(f"  {len(argo_df)} unique (lat, lon, datetime) Argo profiles across both sensors")

    print("Loading and filtering GDAC index...")
    index_df = load_index(args.start_date, args.end_date, args.cachedir)
    print(f"  {len(index_df)} index records in range")

    print("Matching...")
    max_time_delta = pd.Timedelta(minutes=args.max_time_delta_minutes)
    result = match_to_index(argo_df, index_df, args.max_dist_km, max_time_delta)

    n_matched = result['matched'].sum()
    print(f"\nMatched {n_matched} / {len(result)} ({100*n_matched/len(result):.1f}%) "
          f"within {args.max_dist_km}km / {args.max_time_delta_minutes}min")
    print("\nDistance (km) for matched rows:")
    print(result.loc[result['matched'], 'dist_km'].describe())
    print("\nUnmatched rows -- distance to nearest index entry (should be large if genuinely absent):")
    print(result.loc[~result['matched'], 'dist_km'].describe())

    result.to_parquet(args.out, index=False)
    print(f"\nSaved lookup table to {args.out}")


if __name__ == '__main__':
    main()
