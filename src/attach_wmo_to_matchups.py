#!/usr/bin/env python3
"""
Merge the recovered WMO float ID (data/matchups/argo_wmo_lookup.parquet,
built by enrich_argo_metadata.py) into the SMAP/SMOS matchup tables, so
downstream code (features.py split_data) can do float-aware train/val/test
partitioning. Rows with no confident index match keep wmo = NaN -- see
DESIGN.md 17/18 for what that implies for the split.

Overwrites data/matchups/<sensor>_argo_matchups.parquet in place, adding a
single 'wmo' column (float64, NaN where unmatched). No other columns change.
"""

import argparse

import pandas as pd


def attach(matchups_path, lookup_path):
    matchups = pd.read_parquet(matchups_path)
    lookup = pd.read_parquet(lookup_path, columns=['argo_lat', 'argo_lon', 'argo_datetime', 'wmo', 'matched'])
    lookup = lookup[lookup['matched']].drop(columns='matched')

    before = len(matchups)
    merged = matchups.merge(lookup, on=['argo_lat', 'argo_lon', 'argo_datetime'], how='left')
    assert len(merged) == before, "merge changed row count -- lookup key must not be unique"

    n_with_wmo = merged['wmo'].notna().sum()
    print(f"  {n_with_wmo} / {len(merged)} ({100*n_with_wmo/len(merged):.1f}%) rows got a WMO float ID")

    merged.to_parquet(matchups_path, index=False)
    print(f"  Overwrote {matchups_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matchups-dir', default='/Users/afeman/Desktop/work/sss-bias/data/matchups')
    parser.add_argument('--lookup', default='/Users/afeman/Desktop/work/sss-bias/data/matchups/argo_wmo_lookup.parquet')
    args = parser.parse_args()

    for sensor in ['smap', 'smos']:
        path = f'{args.matchups_dir}/{sensor}_argo_matchups.parquet'
        print(f"{sensor}:")
        attach(path, args.lookup)


if __name__ == '__main__':
    main()
