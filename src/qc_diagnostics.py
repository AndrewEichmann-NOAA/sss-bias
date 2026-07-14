#!/usr/bin/env python3
"""
Measure QC rejection rates at each stage of the satellite-SSS-vs-Argo
matchup pipeline, for a given --sensor (smap or smos):
  1. PreQC failures (bitmask for SMAP, quality-index threshold for SMOS --
     see SENSOR_CONFIG in build_matchups.py)
  2. Satellite / Argo physical valid-range failures
  3. Gross-mismatch (background-check style) failures on matched pairs

Reuses the same filter constants and SENSOR_CONFIG as build_matchups.py.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import BallTree

from process_netcdf_cycles import NetCDFCycleProcessor
from build_matchups import EARTH_RADIUS_KM, SENSOR_CONFIG

MIN_SALINITY = 20.0
MAX_SALINITY = 42.0
MAX_DEPTH = 5.0
MAX_DIST_KM = 50.0
MAX_TIME_DELTA = pd.Timedelta(hours=3)
MAX_ABS_DIFF = 10.0


def sat_stage_counts(cycle_path, cycle_hour, sensor):
    config = SENSOR_CONFIG[sensor]
    f = cycle_path / 'sss' / f'gdas.t{cycle_hour}z.{config["filename"]}'
    if not f.exists():
        return None
    try:
        meta = xr.open_dataset(f, group='MetaData', engine='netcdf4')
        obs = xr.open_dataset(f, group='ObsValue', engine='netcdf4')
        qc = xr.open_dataset(f, group='PreQC', engine='netcdf4')
        sss = obs['seaSurfaceSalinity'].values
        preqc = qc['seaSurfaceSalinity'].values
        lat = meta['latitude'].values
        lon = meta['longitude'].values
        dt = meta['dateTime'].values
        basin = meta['oceanBasin'].values
        meta.close()
        obs.close()
        qc.close()
    except Exception as e:
        print(f"  Error loading {sensor} {f}: {e}")
        return None

    valid = ~np.isnan(sss)
    n_raw = int(valid.sum())
    qc_pass_mask = config['qc_pass'](preqc)
    preqc_fail = valid & ~qc_pass_mask
    in_range = (sss >= MIN_SALINITY) & (sss <= MAX_SALINITY)
    range_fail = valid & qc_pass_mask & ~in_range
    n_pass = valid & qc_pass_mask & in_range

    df = pd.DataFrame({
        'lat': lat[n_pass], 'lon': lon[n_pass], 'datetime': dt[n_pass],
        'oceanBasin': basin[n_pass], 'sss': sss[n_pass],
    })

    return {
        'n_raw': n_raw,
        'n_preqc_fail': int(preqc_fail.sum()),
        'n_range_fail': int(range_fail.sum()),
        'n_pass': int(n_pass.sum()),
        'df': df,
    }


def argo_stage_counts(cycle_path, cycle_hour):
    f = cycle_path / 'insitu' / f'gdas.t{cycle_hour}z.insitu_salt_profile_argo.nc'
    if not f.exists():
        return None
    try:
        meta = xr.open_dataset(f, group='MetaData', engine='netcdf4')
        obs = xr.open_dataset(f, group='ObsValue', engine='netcdf4')
        sal = obs['salinity'].values
        depth = meta['depth'].values
        lat = meta['latitude'].values
        lon = meta['longitude'].values
        dt = meta['dateTime'].values
        basin = meta['oceanBasin'].values
        meta.close()
        obs.close()
    except Exception as e:
        print(f"  Error loading Argo {f}: {e}")
        return None

    near_surface = ~np.isnan(sal) & (depth <= MAX_DEPTH)
    n_raw = int(near_surface.sum())
    in_range = (sal >= MIN_SALINITY) & (sal <= MAX_SALINITY)
    range_fail = near_surface & ~in_range
    n_pass = near_surface & in_range

    df = pd.DataFrame({
        'lat': lat[n_pass], 'lon': lon[n_pass], 'datetime': dt[n_pass],
        'oceanBasin': basin[n_pass], 'salinity': sal[n_pass], 'depth': depth[n_pass],
    })
    n_obs_pass = len(df)
    if not df.empty:
        df = df.groupby(['lat', 'lon', 'datetime', 'oceanBasin'], as_index=False).agg(
            salinity=('salinity', 'mean'), depth=('depth', 'mean'))

    return {
        'n_raw': n_raw,
        'n_range_fail': int(range_fail.sum()),
        'n_pass': n_obs_pass,
        'n_profiles_pass': len(df),
        'df': df,
    }


def match_with_diagnostics(sat_df, argo_df):
    """Returns (n_candidate_pairs_within_space_time_window, n_of_those_failing_gross_diff_check)."""
    if sat_df is None or argo_df is None or sat_df.empty or argo_df.empty:
        return 0, 0

    sat_rad = np.radians(sat_df[['lat', 'lon']].to_numpy())
    tree = BallTree(sat_rad, metric='haversine')
    argo_rad = np.radians(argo_df[['lat', 'lon']].to_numpy())
    dist_rad, idx = tree.query(argo_rad, k=1)
    dist_km = dist_rad[:, 0] * EARTH_RADIUS_KM
    idx = idx[:, 0]

    matched_sat = sat_df.iloc[idx].reset_index(drop=True)
    argo_r = argo_df.reset_index(drop=True)
    time_delta = (argo_r['datetime'] - matched_sat['datetime']).abs()

    candidate_mask = (dist_km <= MAX_DIST_KM) & (time_delta <= MAX_TIME_DELTA)
    n_candidates = int(candidate_mask.sum())

    diff = (matched_sat['sss'] - argo_r['salinity']).abs()
    gross_fail_mask = candidate_mask & (diff > MAX_ABS_DIFF)
    n_gross_fail = int(gross_fail_mask.sum())

    return n_candidates, n_gross_fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor', choices=sorted(SENSOR_CONFIG), default='smap')
    args = parser.parse_args()
    sensor = args.sensor

    base_dir = '/Users/afeman/Desktop/work/sss-bias/data/common_obsForge'
    processor = NetCDFCycleProcessor(base_dir)
    cycle_dirs = processor.find_cycle_directories(None, None)
    print(f"Found {len(cycle_dirs)} cycle directories\n")

    t = dict(
        sat_raw=0, sat_preqc_fail=0, sat_range_fail=0, sat_pass=0,
        argo_raw=0, argo_range_fail=0, argo_pass=0, argo_profiles_pass=0,
        candidates=0, gross_fail=0,
    )

    for i, (date, cycle, cycle_path) in enumerate(cycle_dirs):
        s = sat_stage_counts(cycle_path, cycle, sensor)
        a = argo_stage_counts(cycle_path, cycle)

        if s:
            t['sat_raw'] += s['n_raw']
            t['sat_preqc_fail'] += s['n_preqc_fail']
            t['sat_range_fail'] += s['n_range_fail']
            t['sat_pass'] += s['n_pass']
        if a:
            t['argo_raw'] += a['n_raw']
            t['argo_range_fail'] += a['n_range_fail']
            t['argo_pass'] += a['n_pass']
            t['argo_profiles_pass'] += a['n_profiles_pass']

        if s and a:
            n_cand, n_gross = match_with_diagnostics(s['df'], a['df'])
            t['candidates'] += n_cand
            t['gross_fail'] += n_gross

        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(cycle_dirs)} cycles processed")

    print(f"\n=== {sensor.upper()} (obs with non-fill SSS value) ===")
    print(f"  raw (non-fill):         {t['sat_raw']:>12,}")
    print(f"  PreQC fail (rejected):  {t['sat_preqc_fail']:>12,} "
          f"({100*t['sat_preqc_fail']/t['sat_raw']:.2f}%)")
    print(f"  out of [{MIN_SALINITY},{MAX_SALINITY}] PSU (of PreQC-pass): {t['sat_range_fail']:>12,} "
          f"({100*t['sat_range_fail']/t['sat_raw']:.3f}%)")
    print(f"  pass both filters:      {t['sat_pass']:>12,} "
          f"({100*t['sat_pass']/t['sat_raw']:.2f}%)")

    print("\n=== Argo (near-surface, depth <= 5m, non-fill salinity) ===")
    print(f"  raw:                    {t['argo_raw']:>12,}")
    print(f"  out of [{MIN_SALINITY},{MAX_SALINITY}] PSU (rejected): {t['argo_range_fail']:>12,} "
          f"({100*t['argo_range_fail']/t['argo_raw']:.2f}%)")
    print(f"  pass range filter:      {t['argo_pass']:>12,} "
          f"({100*t['argo_pass']/t['argo_raw']:.2f}%)")
    print(f"  -> unique profiles after averaging same lat/lon/datetime: {t['argo_profiles_pass']:,}")

    print(f"\n=== Matched pairs (within 50km / 3h, both QC-passed) ===")
    print(f"  candidate pairs:        {t['candidates']:>12,}")
    print(f"  gross-mismatch (>|{MAX_ABS_DIFF}| PSU) rejected: {t['gross_fail']:>12,} "
          f"({100*t['gross_fail']/t['candidates']:.2f}%)" if t['candidates'] else "  candidate pairs: 0")
    print(f"  final matchups:         {t['candidates']-t['gross_fail']:>12,}")

    print("\n=== End-to-end suspicious-observation rate ===")
    sat_suspicious = t['sat_preqc_fail'] + t['sat_range_fail']
    print(f"  {sensor.upper()}: {sat_suspicious:,} / {t['sat_raw']:,} flagged suspicious "
          f"({100*sat_suspicious/t['sat_raw']:.2f}%)")
    print(f"  Argo (near-surface obs): {t['argo_range_fail']:,} / {t['argo_raw']:,} flagged suspicious "
          f"({100*t['argo_range_fail']/t['argo_raw']:.2f}%)")
    if t['candidates']:
        print(f"  Matched pairs additionally rejected on gross mismatch: "
              f"{t['gross_fail']:,} / {t['candidates']:,} ({100*t['gross_fail']/t['candidates']:.2f}%)")


if __name__ == '__main__':
    main()
