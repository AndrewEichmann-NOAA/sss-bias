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

    NOTE: uses `originalDateTime`, not `dateTime`, as the observation
    timestamp. Argo obs are assimilated on a wider +/-4-cycle window than
    satellite obs, and `dateTime` is snapped to a nearby synoptic cycle slot
    for IODA/DA-window bookkeeping -- verified only ~9-14% of obs have
    `dateTime == originalDateTime` in sampled cycles, with the rest offset
    by up to +/-24h. Using `dateTime` here would silently pair satellite
    retrievals with Argo profiles up to a day apart while believing them to
    be within the 3h match window. `originalDateTime` has no _FillValue
    attribute (like `salinity`, see above), so a plausible-range guard
    (2000-2030) is applied defensively, though a full-archive sample found
    zero implausible values.
    """
    f = cycle_path / 'insitu' / f'gdas.t{cycle_hour}z.insitu_salt_profile_argo.nc'
    if not f.exists():
        return None

    try:
        meta = xr.open_dataset(f, group='MetaData', engine='netcdf4', decode_times=False)
        obs = xr.open_dataset(f, group='ObsValue', engine='netcdf4')
        original_dt = meta['originalDateTime'].values
        df = pd.DataFrame({
            'lat': meta['latitude'].values,
            'lon': meta['longitude'].values,
            'datetime': pd.to_datetime(original_dt, unit='s', origin='unix'),
            'oceanBasin': meta['oceanBasin'].values,
            'depth': meta['depth'].values,
            'salinity': obs['salinity'].values,
        })
        meta.close()
        obs.close()
    except Exception as e:
        print(f"  Error loading Argo {f}: {e}")
        return None

    plausible_lo, plausible_hi = pd.Timestamp('2000-01-01'), pd.Timestamp('2030-01-01')
    df = df[df['datetime'].between(plausible_lo, plausible_hi)]

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


def build_sat_tree(sat_df):
    """Build a haversine BallTree over a satellite cycle's obs, or None if empty."""
    if sat_df is None or sat_df.empty:
        return None
    sat_rad = np.radians(sat_df[['lat', 'lon']].to_numpy())
    return BallTree(sat_rad, metric='haversine')


def match_windowed(argo_df, candidates, max_dist_km, max_time_delta, max_abs_diff):
    """Match each Argo near-surface obs to the best satellite obs across a WINDOW of
    nearby cycles, not just the one cycle sharing Argo's own directory.

    `candidates` is a list of (sat_df, tree) tuples, one per cycle in the window
    (typically +/-4 cycles = +/-24h, matching Argo's wider DA assimilation window --
    see build_matchups.py module docstring and DESIGN.md 15.5). Each candidate cycle
    is queried independently for its own nearest-by-space match to every Argo obs;
    across all candidate cycles, the best (smallest distance) match that ALSO passes
    the time/distance/gross-error filters is kept per Argo obs. This is deliberately
    NOT "pool all candidate cycles into one BallTree and take the single nearest,"
    because a spatially-nearer-but-wrong-time match from a neighboring cycle could
    otherwise mask a valid, slightly-farther, correct-time match -- the whole point
    of searching a cycle window is to recover genuinely valid matches, not just to
    prefer whatever point happens to be closest in space regardless of time.

    max_abs_diff is a gross-error/background-check style filter on the matched pair
    itself: per-obs physical-range QC alone still lets through matches like a
    stuck-sensor Argo profile reporting ~9 PSU in the open ocean (physically
    implausible for that region, but inside a lenient [min_salinity, max_salinity]
    band) paired against a normal ~36 PSU satellite retrieval. A >30 PSU discrepancy
    is a QC failure, not bias signal to learn from, so pairs are rejected outright
    rather than left in as extreme training examples.
    """
    if argo_df is None or argo_df.empty:
        return pd.DataFrame()

    argo_df = argo_df.reset_index(drop=True)
    argo_rad = np.radians(argo_df[['lat', 'lon']].to_numpy())
    argo_salinity = argo_df['salinity'].to_numpy()
    argo_datetime = argo_df['datetime'].to_numpy()
    n = len(argo_df)

    best_dist = np.full(n, np.inf)
    best_sss = np.full(n, np.nan)
    best_lat = np.full(n, np.nan)
    best_lon = np.full(n, np.nan)
    best_basin = np.full(n, np.nan)
    best_datetime = np.full(n, np.datetime64('NaT'), dtype='datetime64[ns]')
    best_time_delta = np.full(n, np.timedelta64('NaT'), dtype='timedelta64[ns]')

    for sat_df, tree in candidates:
        if sat_df is None or tree is None or sat_df.empty:
            continue

        dist_rad, idx = tree.query(argo_rad, k=1)
        dist_km = dist_rad[:, 0] * EARTH_RADIUS_KM
        idx = idx[:, 0]

        cand_sss = sat_df['sss'].to_numpy()[idx]
        cand_lat = sat_df['lat'].to_numpy()[idx]
        cand_lon = sat_df['lon'].to_numpy()[idx]
        cand_basin = sat_df['oceanBasin'].to_numpy()[idx]
        cand_datetime = sat_df['datetime'].to_numpy()[idx]
        cand_time_delta = np.abs(argo_datetime - cand_datetime)
        cand_abs_diff = np.abs(cand_sss - argo_salinity)

        valid = (dist_km <= max_dist_km) & (cand_time_delta <= np.timedelta64(max_time_delta)) \
            & (cand_abs_diff <= max_abs_diff)
        cand_dist = np.where(valid, dist_km, np.inf)

        improve = cand_dist < best_dist
        if not improve.any():
            continue
        best_dist = np.where(improve, cand_dist, best_dist)
        best_sss = np.where(improve, cand_sss, best_sss)
        best_lat = np.where(improve, cand_lat, best_lat)
        best_lon = np.where(improve, cand_lon, best_lon)
        best_basin = np.where(improve, cand_basin, best_basin)
        best_datetime = np.where(improve, cand_datetime, best_datetime)
        best_time_delta = np.where(improve, cand_time_delta, best_time_delta)

    result = argo_df.rename(columns={
        'lat': 'argo_lat', 'lon': 'argo_lon', 'datetime': 'argo_datetime',
        'oceanBasin': 'argo_oceanBasin', 'salinity': 'argo_salinity', 'depth': 'argo_depth',
    })
    result['sat_sss'] = best_sss
    result['sat_lat'] = best_lat
    result['sat_lon'] = best_lon
    result['sat_datetime'] = best_datetime
    result['sat_oceanBasin'] = best_basin
    result['dist_km'] = best_dist
    result['time_delta'] = best_time_delta

    mask = np.isfinite(best_dist)
    return result[mask].reset_index(drop=True)


def build_matchups(base_dir, sensor, start_date, end_date, max_dist_km, max_time_delta_hours,
                    max_depth=5.0, min_salinity=20.0, max_salinity=42.0, max_abs_diff=10.0,
                    cycle_window=4, verbose=True):
    """cycle_window: how many 6h cycles on either side of an Argo obs's own cycle to
    search for a satellite match (default 4 = +/-24h, matching Argo's wider DA
    assimilation window -- see DESIGN.md 15.5). Satellite files are loaded and their
    BallTree built at most once each (cached across the sliding window), so this
    costs ~cycle_window extra tree queries per cycle, not extra file I/O.
    """
    processor = NetCDFCycleProcessor(base_dir)
    cycle_dirs = processor.find_cycle_directories(start_date, end_date)
    print(f"Found {len(cycle_dirs)} cycle directories")

    max_time_delta = pd.Timedelta(hours=max_time_delta_hours)
    all_matches = []
    total_argo = 0
    total_sat_loaded = 0

    sat_cache = {}  # cycle index -> (sat_df, tree), sliding window

    def get_sat(idx):
        if idx not in sat_cache:
            _, cyc, cpath = cycle_dirs[idx]
            sdf = load_sat_sss(cpath, cyc, sensor, min_salinity, max_salinity)
            sat_cache[idx] = (sdf, build_sat_tree(sdf))
        return sat_cache[idx]

    for i, (date, cycle, cycle_path) in enumerate(cycle_dirs):
        argo_df = load_argo_near_surface(cycle_path, cycle, max_depth, min_salinity, max_salinity)

        window = range(max(0, i - cycle_window), min(len(cycle_dirs), i + cycle_window + 1))
        candidates = [get_sat(j) for j in window]

        matches = match_windowed(argo_df, candidates, max_dist_km, max_time_delta, max_abs_diff)

        n_argo = 0 if argo_df is None else len(argo_df)
        total_argo += n_argo

        if not matches.empty:
            matches['cycle_date'] = date
            matches['cycle_hour'] = cycle
            all_matches.append(matches)

        if verbose:
            n_sat_center = len(candidates[min(i, cycle_window)][0]) \
                if candidates and candidates[min(i, cycle_window)][0] is not None else 0
            print(f"  {date.strftime('%Y-%m-%d')} {cycle}Z: "
                  f"{n_argo} argo near-surface, {n_sat_center} {sensor} obs (own cycle), "
                  f"{len(matches)} matches (searched +/-{cycle_window} cycles)")

        # Evict cache entries no longer needed by any future iteration.
        evict_before = i + 1 - cycle_window
        for j in [k for k in sat_cache if k < evict_before]:
            sdf, _ = sat_cache.pop(j)
            if sdf is not None:
                total_sat_loaded += len(sdf)

    # Account for any cache entries still resident at the end of the loop.
    for sdf, _ in sat_cache.values():
        if sdf is not None:
            total_sat_loaded += len(sdf)

    print(f"\nTotals: {total_argo} argo near-surface obs scanned (raw, NOT deduplicated -- see below), "
          f"{total_sat_loaded} {sensor} obs loaded "
          f"(each satellite file loaded at most once, reused across the +/-{cycle_window}-cycle window)")

    if not all_matches:
        print("No matches found.")
        return pd.DataFrame()

    result = pd.concat(all_matches, ignore_index=True)

    # Argo profiles are replicated across every cycle file within their own
    # +/-4-cycle assimilation window (obsForge puts the same profile in each
    # cycle's file so that cycle's DA run has it available). Since each cycle
    # is processed independently, the same real profile is found and matched
    # once per cycle file it appears in (up to 2*cycle_window+1 times),
    # producing byte-identical duplicate rows -- collapse them here.
    n_before_dedup = len(result)
    result = result.drop_duplicates(subset=['argo_lat', 'argo_lon', 'argo_datetime']).reset_index(drop=True)
    print(f"Deduplicated {n_before_dedup} raw matches (one per cycle-file appearance of the same "
          f"Argo profile) down to {len(result)} unique profile-satellite matches.")

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
    parser.add_argument('--cycle-window', type=int, default=4,
                         help='Search +/-N cycles for a satellite match, not just Argo\'s own cycle '
                              '(default 4 = +/-24h, matching Argo\'s wider DA assimilation window)')
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
        max_abs_diff=args.max_abs_diff, cycle_window=args.cycle_window,
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
