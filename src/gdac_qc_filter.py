#!/usr/bin/env python3
"""
Filters an Argo near-surface DataFrame (lat/lon/datetime/salinity/depth/
oceanBasin, as produced by build_matchups.py's load_argo_near_surface or
build_raw_smap_matchups.py's load_argo_for_window) down to profiles with
real, good GDAC QC -- replacing the ad hoc valid-range heuristic (DESIGN.md
30) with actual per-obs quality control (DESIGN.md 31), using the lookup
table built by fetch_gdac_argo_qc.py.

The merge key is an EXACT match on (lat, lon, datetime), not a fuzzy nearest-
neighbor: fetch_gdac_argo_qc.py derived its argo_lat/argo_lon/argo_datetime
from the exact same loader functions (deterministic NetCDF parsing of the
same source files), so the same profile always has bit-identical
coordinates whichever call site loaded it.

Unmatched profiles (no GDAC index match within fetch_gdac_argo_qc.py's
1km/10min tolerance -- ~10% in the segment test, DESIGN.md 31) are dropped
rather than falling back to the old range filter: this is a conservative
choice (only keep what real QC has confirmed), not an attempt to maximize
row count.
"""

from pathlib import Path

import numpy as np
import pandas as pd

QC_COLUMNS = ['argo_lat', 'argo_lon', 'argo_datetime', 'worst_psal_qc', 'delayed_salinity', 'data_mode']


def load_qc_lookup(qc_dir):
    files = sorted(Path(qc_dir).glob('gdac_argo_qc_*.parquet'))
    if not files:
        raise FileNotFoundError(f"No gdac_argo_qc_*.parquet files found in {qc_dir}")
    frames = [pd.read_parquet(f, columns=QC_COLUMNS) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=['argo_lat', 'argo_lon', 'argo_datetime']).reset_index(drop=True)


def apply_qc_filter(argo_df, qc_lookup, good_qc=(1, 2), use_delayed_mode=True, verbose=True):
    """Returns argo_df restricted to profiles with worst_psal_qc in good_qc,
    with `salinity` replaced by the delayed-mode-preferred value if
    use_delayed_mode (DESIGN.md 17.1: prefer the best available label).
    """
    if argo_df is None or argo_df.empty:
        return argo_df

    renamed = argo_df.rename(columns={'lat': 'argo_lat', 'lon': 'argo_lon', 'datetime': 'argo_datetime'})
    merged = renamed.merge(qc_lookup, on=['argo_lat', 'argo_lon', 'argo_datetime'], how='left')

    n_before = len(merged)
    n_no_qc = merged['worst_psal_qc'].isna().sum()
    merged = merged[merged['worst_psal_qc'].isin(good_qc)].copy()
    if verbose:
        print(f"  GDAC QC filter: {n_before} obs -> {len(merged)} kept "
              f"({n_no_qc} had no GDAC match, {n_before - len(merged) - n_no_qc} failed QC)")

    if use_delayed_mode:
        merged['salinity'] = merged['delayed_salinity']

    return merged.rename(columns={'argo_lat': 'lat', 'argo_lon': 'lon', 'argo_datetime': 'datetime'})[
        ['lat', 'lon', 'datetime', 'oceanBasin', 'depth', 'salinity']]


def load_gdac_direct_argo(direct_dir, start_date, end_date, good_qc=(1, 2), verbose=True):
    """Loads Argo profiles fetched directly from the GDAC (fetch_gdac_argo_direct.py,
    DESIGN.md 34) -- independent of obsForge's local archive, so not gated by
    obsForge's completeness (33's found gap, or the general ~50% undercount
    vs. the true GDAC index found while investigating it).

    Returns the same (lat, lon, datetime, oceanBasin, depth, salinity) schema
    as build_matchups.py's load_argo_near_surface, for drop-in use by
    match_to_argo / box_average_match_to_argo. `oceanBasin` is NOT available
    from GDAC, so it's filled in with classify_ocean_basin.classify_basin's
    lat/lon-based approximation (89.3% agreement with real obsForge codes on
    basins 1/2/3/5, which are the ones that actually matter -- DESIGN.md 35).
    """
    files = sorted(Path(direct_dir).glob('gdac_argo_direct_*.parquet'))
    if not files:
        raise FileNotFoundError(f"No gdac_argo_direct_*.parquet files found in {direct_dir}")

    start_ym = pd.Timestamp(start_date).strftime('%Y%m')
    end_ym = pd.Timestamp(end_date).strftime('%Y%m')
    files = [f for f in files if start_ym <= f.stem.split('_')[-1] <= end_ym]

    frames = [pd.read_parquet(f, columns=['lat', 'lon', 'datetime', 'salinity', 'depth', 'worst_psal_qc'])
              for f in files]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=['lat', 'lon', 'datetime', 'salinity', 'depth', 'worst_psal_qc'])

    df = df[(df['datetime'] >= pd.Timestamp(start_date)) & (df['datetime'] < pd.Timestamp(end_date))]
    n_before = len(df)
    df = df[df['worst_psal_qc'].isin(good_qc)].copy()
    if verbose:
        print(f"  GDAC-direct Argo: {n_before} profiles in range -> {len(df)} kept after QC filter")

    from classify_ocean_basin import classify_basin
    df['oceanBasin'] = classify_basin(df['lat'].to_numpy(), df['lon'].to_numpy())
    return df[['lat', 'lon', 'datetime', 'oceanBasin', 'depth', 'salinity']].reset_index(drop=True)
