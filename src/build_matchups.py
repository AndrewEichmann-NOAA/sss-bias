#!/usr/bin/env python3
"""
Build a satellite-SSS-vs-Argo matchup table for bias-correction training.
Supports SMAP and SMOS (--sensor), sharing the same collocation logic.

For each 6h cycle, collocates near-surface (depth <= 5m) Argo salinity
observations with the nearest satellite SSS observation within a distance/
time window, and writes the result to a parquet table (one row per matched
pair). Output columns use a sensor-agnostic `sat_*` prefix so downstream
feature/training code works unchanged for either sensor.

Directory structure: gdas.YYYYMMDD/HH/ocean/{sss,insitu}/

QC note -- the two sensors' PreQC fields mean different things and are NOT
handled the same way:
  - SMAP: a real bitmask, verified ~81% pass at PreQC == 0.
  - SMOS: a continuous quality/uncertainty index (ObsError and SSS variance
    both scale up with it), plus a distinct high-uncertainty catch-all
    bucket at exactly 999 (verified: 21.6% of obs, with a 20x higher
    out-of-range rate than the well-behaved bins). There is no PreQC == 0
    "pass" convention for SMOS. Threshold chosen at PreQC < 600, which
    excludes the small high-error [600,900) bin and the 999 catch-all while
    keeping the well-behaved continuous range -- giving a ~78% pass rate,
    comparable in magnitude to SMAP's 81%.
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import BallTree

from process_netcdf_cycles import NetCDFCycleProcessor

EARTH_RADIUS_KM = 6371.0

SENSOR_CONFIG = {
    'smap': {
        'filename': 'sss_smap_l2.nc',
        'qc_pass': lambda preqc: preqc == 0,
    },
    'smos': {
        'filename': 'sss_smos_l2.nc',
        'qc_pass': lambda preqc: preqc < 600,
    },
}


def load_sat_sss(cycle_path, cycle_hour, sensor, min_salinity, max_salinity):
    """Load satellite SSS obs (lat, lon, datetime, oceanBasin, sss) for one cycle."""
    config = SENSOR_CONFIG[sensor]
    f = cycle_path / 'sss' / f'gdas.t{cycle_hour}z.{config["filename"]}'
    if not f.exists():
        return None

    try:
        meta = xr.open_dataset(f, group='MetaData', engine='netcdf4')
        obs = xr.open_dataset(f, group='ObsValue', engine='netcdf4')
        qc = xr.open_dataset(f, group='PreQC', engine='netcdf4')
        df = pd.DataFrame({
            'lat': meta['latitude'].values,
            'lon': meta['longitude'].values,
            'datetime': meta['dateTime'].values,
            'oceanBasin': meta['oceanBasin'].values,
            'sss': obs['seaSurfaceSalinity'].values,
            'preqc': qc['seaSurfaceSalinity'].values,
        })
        meta.close()
        obs.close()
        qc.close()
    except Exception as e:
        print(f"  Error loading {sensor} {f}: {e}")
        return None

    df = df.dropna(subset=['sss'])
    df = df[config['qc_pass'](df['preqc']) & df['sss'].between(min_salinity, max_salinity)]
    return df.drop(columns='preqc').reset_index(drop=True)


def load_argo_near_surface(cycle_path, cycle_hour, max_depth, min_salinity, max_salinity):
    """Load Argo salinity obs with depth <= max_depth, averaged per profile.

    NOTE: Argo's PreQC field is NOT usable for QC here -- verified it is
    uniformly 0 (pass) across all ~522k obs in a sample cycle, including
    obs with salinity as low as 0.06 PSU (a stuck-sensor profile reporting
    ~0.118 PSU across all depths). Real Argo delayed-mode QC flags were not
    carried through the obsForge/IODA conversion, so we apply a physically
    motivated valid-range filter instead (min_salinity/max_salinity), rather
    than trusting PreQC.
    """
    f = cycle_path / 'insitu' / f'gdas.t{cycle_hour}z.insitu_salt_profile_argo.nc'
    if not f.exists():
        return None

    try:
        meta = xr.open_dataset(f, group='MetaData', engine='netcdf4')
        obs = xr.open_dataset(f, group='ObsValue', engine='netcdf4')
        df = pd.DataFrame({
            'lat': meta['latitude'].values,
            'lon': meta['longitude'].values,
            'datetime': meta['dateTime'].values,
            'oceanBasin': meta['oceanBasin'].values,
            'depth': meta['depth'].values,
            'salinity': obs['salinity'].values,
        })
        meta.close()
        obs.close()
    except Exception as e:
        print(f"  Error loading Argo {f}: {e}")
        return None

    df = df[df['depth'] <= max_depth].dropna(subset=['salinity'])
    df = df[df['salinity'].between(min_salinity, max_salinity)]
    if df.empty:
        return df

    # Average multiple near-surface depth levels from the same profile
    # (same lat/lon/datetime) into a single bulk-salinity value.
    grouped = df.groupby(['lat', 'lon', 'datetime', 'oceanBasin'], as_index=False).agg(
        salinity=('salinity', 'mean'),
        depth=('depth', 'mean'),
    )
    return grouped


def match_cycle(sat_df, argo_df, max_dist_km, max_time_delta, max_abs_diff):
    """Nearest-neighbor match each Argo near-surface obs to a satellite obs.

    max_abs_diff is a gross-error/background-check style filter on the
    matched pair itself: per-obs physical-range QC alone still lets through
    matches like a stuck-sensor Argo profile reporting ~9 PSU in the open
    ocean (physically implausible for that region, but inside a lenient
    [min_salinity, max_salinity] band) paired against a normal ~36 PSU
    satellite retrieval. A >30 PSU discrepancy is a QC failure, not bias
    signal to learn from, so pairs are rejected outright rather than left in
    as extreme training examples.
    """
    if sat_df is None or argo_df is None or sat_df.empty or argo_df.empty:
        return pd.DataFrame()

    sat_rad = np.radians(sat_df[['lat', 'lon']].to_numpy())
    tree = BallTree(sat_rad, metric='haversine')

    argo_rad = np.radians(argo_df[['lat', 'lon']].to_numpy())
    dist_rad, idx = tree.query(argo_rad, k=1)
    dist_km = dist_rad[:, 0] * EARTH_RADIUS_KM
    idx = idx[:, 0]

    matched_sat = sat_df.iloc[idx].reset_index(drop=True)
    result = argo_df.reset_index(drop=True).rename(columns={
        'lat': 'argo_lat', 'lon': 'argo_lon', 'datetime': 'argo_datetime',
        'oceanBasin': 'argo_oceanBasin', 'salinity': 'argo_salinity', 'depth': 'argo_depth',
    })
    result['sat_sss'] = matched_sat['sss'].to_numpy()
    result['sat_lat'] = matched_sat['lat'].to_numpy()
    result['sat_lon'] = matched_sat['lon'].to_numpy()
    result['sat_datetime'] = matched_sat['datetime'].to_numpy()
    result['sat_oceanBasin'] = matched_sat['oceanBasin'].to_numpy()
    result['dist_km'] = dist_km
    result['time_delta'] = (result['argo_datetime'] - result['sat_datetime']).abs()

    abs_diff = (result['sat_sss'] - result['argo_salinity']).abs()
    mask = ((result['dist_km'] <= max_dist_km)
            & (result['time_delta'] <= max_time_delta)
            & (abs_diff <= max_abs_diff))
    return result[mask].reset_index(drop=True)


def build_matchups(base_dir, sensor, start_date, end_date, max_dist_km, max_time_delta_hours,
                    max_depth=5.0, min_salinity=20.0, max_salinity=42.0, max_abs_diff=10.0,
                    verbose=True):
    processor = NetCDFCycleProcessor(base_dir)
    cycle_dirs = processor.find_cycle_directories(start_date, end_date)
    print(f"Found {len(cycle_dirs)} cycle directories")

    max_time_delta = pd.Timedelta(hours=max_time_delta_hours)
    all_matches = []
    total_argo = 0
    total_sat = 0

    for date, cycle, cycle_path in cycle_dirs:
        sat_df = load_sat_sss(cycle_path, cycle, sensor, min_salinity, max_salinity)
        argo_df = load_argo_near_surface(cycle_path, cycle, max_depth, min_salinity, max_salinity)
        matches = match_cycle(sat_df, argo_df, max_dist_km, max_time_delta, max_abs_diff)

        n_argo = 0 if argo_df is None else len(argo_df)
        n_sat = 0 if sat_df is None else len(sat_df)
        total_argo += n_argo
        total_sat += n_sat

        if not matches.empty:
            matches['cycle_date'] = date
            matches['cycle_hour'] = cycle
            all_matches.append(matches)

        if verbose:
            print(f"  {date.strftime('%Y-%m-%d')} {cycle}Z: "
                  f"{n_argo} argo near-surface, {n_sat} {sensor} obs, "
                  f"{len(matches)} matches")

    print(f"\nTotals: {total_argo} argo near-surface obs, {total_sat} {sensor} obs scanned")

    if not all_matches:
        print("No matches found.")
        return pd.DataFrame()

    result = pd.concat(all_matches, ignore_index=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Build satellite-SSS-vs-Argo matchup table")
    parser.add_argument('--sensor', choices=sorted(SENSOR_CONFIG), default='smap')
    parser.add_argument('--base-dir', default='/Users/afeman/Desktop/work/sss-bias/data/common_obsForge')
    parser.add_argument('--start-date', default=None, help='YYYY-MM-DD')
    parser.add_argument('--end-date', default=None, help='YYYY-MM-DD')
    parser.add_argument('--max-dist-km', type=float, default=50.0)
    parser.add_argument('--max-time-delta-hours', type=float, default=3.0)
    parser.add_argument('--max-depth', type=float, default=5.0, help='Argo near-surface depth cutoff (m)')
    parser.add_argument('--min-salinity', type=float, default=20.0, help='Valid-range QC lower bound (PSU)')
    parser.add_argument('--max-salinity', type=float, default=42.0, help='Valid-range QC upper bound (PSU)')
    parser.add_argument('--max-abs-diff', type=float, default=10.0,
                         help='Reject matched pairs with |satellite - Argo| beyond this (PSU); gross-error check')
    parser.add_argument('--out', default=None,
                         help='Defaults to data/matchups/<sensor>_argo_matchups.parquet')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    out = args.out or f'/Users/afeman/Desktop/work/sss-bias/data/matchups/{args.sensor}_argo_matchups.parquet'

    result = build_matchups(
        args.base_dir, args.sensor, start_date, end_date,
        args.max_dist_km, args.max_time_delta_hours,
        max_depth=args.max_depth, min_salinity=args.min_salinity, max_salinity=args.max_salinity,
        max_abs_diff=args.max_abs_diff,
        verbose=not args.quiet,
    )

    if not result.empty:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out_path, index=False)
        print(f"\nWrote {len(result)} matchups to {out_path}")

        print("\nDistance (km) summary:")
        print(result['dist_km'].describe())
        print("\nTime delta summary:")
        print(result['time_delta'].describe())


if __name__ == '__main__':
    main()
