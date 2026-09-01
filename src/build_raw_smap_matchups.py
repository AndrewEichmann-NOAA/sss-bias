#!/usr/bin/env python3
"""
Build a matchup table between raw JPL CAP SMAP L2B swath data (data/raw_smap_cap/)
and near-surface Argo salinity -- the "rich-feature" counterpart to
build_matchups.py's IODA-based smap_argo_matchups.parquet.

Unlike build_matchups.py, the satellite side here is NOT chunked into 6h DA
cycles: JPL CAP files are per-orbit-revolution swaths covering a continuous
date range, with no cycle-directory structure. So instead of a sliding
cycle-window match, all raw SMAP obs across the requested date range are
pooled into one haversine BallTree, and each Argo near-surface obs (loaded
via build_matchups.py's existing Argo loader, over the same date range's DA
cycles) is matched against it directly.

Retains the per-pixel fields the operational IODA converter currently drops
(incidence/azimuth angles, brightness temperatures, ancillary wind/SST, ice
concentration, retrieval uncertainty, ...) -- see DESIGN.md 21.1: adding these
to the IODA converter is a small, additive change, not a rearchitecture. This
table exists to measure the "research upper bound" available if that change
were made, against the current lat/lon/season/basin-only baseline.

Timestamp reconstruction: each file's `row_time` is UTC seconds-of-day for its
along-track dimension only (broadcast across all cross-track bins in that
row); the file's absolute date comes from the REV_START_YEAR/DAY_OF_YEAR
global attributes -- verified in DESIGN.md 22 that row_time's first value
equals REV_START_TIME converted to seconds, and that a single file's row_time
span (row_time.max() - row_time.min()) never approaches 86400s, so no
day-rollover correction is needed within a file.

Fill-value handling: bypasses netCDF4's auto-masking entirely (verified
unreliable for anc_sst/ice_concentration in DESIGN.md 22) in favor of reading
raw values and manually replacing each variable's declared _FillValue with NaN.
"""

import argparse
import glob
import re
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from build_matchups import load_argo_near_surface, EARTH_RADIUS_KM
from process_netcdf_cycles import NetCDFCycleProcessor

# Per-pixel fields beyond lat/lon/sss/datetime worth carrying through to the
# matchup table as candidate model inputs. Excludes smap_ambiguity_dir/spd
# (3D, per-ambiguity rather than per-pixel) and the raw look-count/ambiguity-
# count fields (n_h_fore/aft, n_v_fore/aft, num_ambiguities -- uint8 with a
# 0 fill value indistinguishable from a genuine zero count, and not physically
# informative on their own).
RICH_FIELDS = [
    'smap_sss_uncertainty', 'quality_flag',
    'anc_sst', 'anc_spd', 'anc_dir', 'anc_sss', 'anc_swh',
    'inc_fore', 'inc_aft', 'azi_fore', 'azi_aft',
    'antazi_fore', 'antazi_aft', 'ice_concentration',
    'land_fraction_fore', 'land_fraction_aft',
    'tb_h_fore', 'tb_h_aft', 'tb_v_fore', 'tb_v_aft',
    'tb_h_bias_adj', 'tb_v_bias_adj',
    'nedt_h_fore', 'nedt_h_aft', 'nedt_v_fore', 'nedt_v_aft',
    'smap_spd', 'smap_high_spd', 'smap_high_dir', 'smap_high_dir_smooth',
]


def _read_var(ds, name):
    """Read a variable as float64, replacing its declared _FillValue with NaN.

    Requires ds.set_auto_mask(False) on the parent Dataset -- see module
    docstring on why netCDF4's own masking isn't trusted here.
    """
    v = ds.variables[name]
    arr = np.asarray(v[:], dtype=np.float64)
    fill = getattr(v, '_FillValue', None)
    if fill is not None:
        arr = np.where(arr == float(fill), np.nan, arr)
    return arr


def _parse_ascending(path):
    """Ascending/descending isn't a netCDF variable -- it's embedded in the raw
    filename itself (`_A_`/`_D_` in SMAP_L2B_SSS_NRT_..._A_...20220608T....h5,
    see DESIGN.md 21.1). Returns 1.0 for ascending, 0.0 for descending.
    """
    m = re.search(r'_([AD])_\d{8}T', Path(path).name)
    if not m:
        raise ValueError(f"Could not find ascending/descending flag in filename: {path}")
    return 1.0 if m.group(1) == 'A' else 0.0


def load_raw_smap_file(path):
    """Load one JPL CAP L2B swath file into a flat per-pixel DataFrame."""
    ascending = _parse_ascending(path)

    ds = nc.Dataset(path)
    ds.set_auto_mask(False)
    try:
        lat = _read_var(ds, 'lat')
        lon = _read_var(ds, 'lon')
        sss = _read_var(ds, 'smap_sss')
        row_time = _read_var(ds, 'row_time')  # (n_along,), seconds of REV_START's UTC day

        base_date = datetime(int(ds.REV_START_YEAR), 1, 1) + timedelta(days=int(ds.REV_START_DAY_OF_YEAR) - 1)
        base = np.datetime64(base_date, 's')

        n_cross, n_along = lat.shape
        row_time_2d = np.broadcast_to(row_time[None, :], (n_cross, n_along))
        seconds = np.round(row_time_2d).astype('int64')
        dt = base + seconds.astype('timedelta64[s]')

        data = {'lat': lat.ravel(), 'lon': lon.ravel(), 'sss': sss.ravel(), 'datetime': dt.ravel(),
                'ascending': np.full(lat.size, ascending)}
        for field in RICH_FIELDS:
            data[field] = _read_var(ds, field).ravel()
        df = pd.DataFrame(data)
    finally:
        ds.close()

    return df.dropna(subset=['lat', 'lon', 'sss'])


def load_raw_smap_dir(raw_dir, min_salinity, max_salinity, verbose=True,
                       start_date=None, end_date=None, max_time_delta=None):
    """Load and QC-filter raw SMAP CAP files in raw_dir, keeping files
    SEPARATE (one DataFrame per orbit revolution) rather than pooling into a
    single DataFrame/tree.

    Each file spans ~1.6h of a single orbit revolution -- pooling all files
    into one global nearest-neighbor search lets a spatially-closer pixel from
    a totally different day's pass mask the true same-pass match (verified:
    this silently dropped 84% of otherwise-valid matches in this window). Per-
    file candidates let match_to_argo apply the same "best valid match across
    a window of candidate passes" logic as build_matchups.py's match_windowed,
    just keyed by orbit file instead of by 6h DA cycle.

    quality_flag == 0 mirrors build_matchups.py's SENSOR_CONFIG['smap'] QC-pass
    convention -- quality_flag *is* the IODA-converted PreQC field, verbatim
    (DESIGN.md 20).

    If start_date/end_date/max_time_delta are given, only loads files whose
    filename-embedded date falls within [start_date - max_time_delta,
    end_date + max_time_delta] -- the archive has grown to tens of thousands
    of files spanning years, and loading every one of them regardless of the
    requested Argo window wastes memory/time in proportion to total archive
    size rather than the actual date range needed (DESIGN.md 29). Padding by
    max_time_delta keeps every file that could possibly fall in some Argo
    obs's match window; the per-file time-window pruning in match_to_argo /
    box_average_match_to_argo still applies on top of this coarser filter.
    """
    files = sorted(glob.glob(str(Path(raw_dir) / '*.h5')))

    if start_date is not None and end_date is not None and max_time_delta is not None:
        pad = pd.Timedelta(max_time_delta)
        window_start = (start_date - pad).strftime('%Y%m%d')
        window_end = (end_date + pad).strftime('%Y%m%d')
        n_total = len(files)
        files = [f for f in files
                 if (m := re.search(r'_(\d{8})T', f)) and window_start <= m.group(1) <= window_end]
        if verbose:
            print(f"Filtered {n_total} total archive files down to {len(files)} "
                  f"within [{window_start}, {window_end}]")
    elif verbose:
        print(f"Found {len(files)} raw SMAP CAP files (no date filter -- loading entire archive)")

    file_dfs = []
    n_ocean = 0
    n_pass = 0
    for f in files:
        try:
            df = load_raw_smap_file(f)
        except Exception as e:
            print(f"  Error loading {f}: {e}")
            continue
        n_ocean += len(df)
        df = df[(df['quality_flag'] == 0) & df['sss'].between(min_salinity, max_salinity)].reset_index(drop=True)
        n_pass += len(df)
        if not df.empty:
            file_dfs.append(df)

    if verbose:
        print(f"  {n_ocean} ocean (non-fill) pixels across all files, {n_pass} QC-pass and in-range")

    return file_dfs


def load_argo_for_window(base_dir, start_date, end_date, max_depth, min_salinity, max_salinity, verbose=True):
    """Load deduplicated near-surface Argo obs across every 6h DA cycle in
    [start_date, end_date), reusing build_matchups.py's per-cycle loader and
    dedup convention (profiles are replicated across their +/-4-cycle window,
    see DESIGN.md 15.5 / build_matchups.py's own dedup step).
    """
    processor = NetCDFCycleProcessor(base_dir)
    cycle_dirs = processor.find_cycle_directories(start_date, end_date)
    if verbose:
        print(f"Found {len(cycle_dirs)} Argo cycle directories")

    dfs = []
    for date, cycle, cycle_path in cycle_dirs:
        adf = load_argo_near_surface(cycle_path, cycle, max_depth, min_salinity, max_salinity)
        if adf is not None and not adf.empty:
            dfs.append(adf)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    n_before = len(df)
    df = df.drop_duplicates(subset=['lat', 'lon', 'datetime']).reset_index(drop=True)
    if verbose:
        print(f"  {n_before} raw near-surface obs, {len(df)} after cross-cycle dedup")
    return df


def match_to_argo(argo_df, file_dfs, max_dist_km, max_time_delta, max_abs_diff):
    """Match each Argo obs to the best raw SMAP pixel across ALL orbit files
    (not just the spatially nearest pixel overall -- see load_raw_smap_dir's
    docstring for why that's wrong). Each file is queried independently for
    its own nearest pixel to every Argo obs; across all files, the best
    (smallest distance) result that ALSO passes the time/gross-error filters
    is kept per Argo obs. Mirrors build_matchups.py's match_windowed, adapted
    to per-orbit-file candidates and generalized to carry arbitrary satellite
    columns (the rich fields) through via a (file, local row) index instead of
    per-column np.where accumulation.

    Each file only spans ~1-2h, so an Argo obs more than max_time_delta outside
    a file's own [min, max] datetime range can never pass the time filter --
    querying the tree for it is wasted work. Without pruning those out first,
    cost is O(n_files x n_argo): both grow linearly with the requested date
    span, so total cost grows quadratically (DESIGN.md 25: a full year took
    ~2h47m this way, ~19x the 12-week run's cost for a ~4.35x longer span).
    Argo obs are sorted once by datetime so each file can binary-search its
    own relevant window instead of scanning every Argo obs -- this drops the
    per-file candidate count to ~constant regardless of total span, making
    the whole match O(n_files) i.e. linear in the date range.
    """
    argo_df = argo_df.reset_index(drop=True)
    argo_rad = np.radians(argo_df[['lat', 'lon']].to_numpy())
    argo_salinity = argo_df['salinity'].to_numpy()
    argo_datetime = argo_df['datetime'].to_numpy()
    n = len(argo_df)

    best_dist = np.full(n, np.inf)
    best_file_idx = np.full(n, -1, dtype=np.int64)
    best_local_idx = np.full(n, -1, dtype=np.int64)

    order = np.argsort(argo_datetime)
    sorted_datetime = argo_datetime[order]

    for fi, df in enumerate(file_dfs):
        file_datetime = df['datetime'].to_numpy()
        # file_datetime is datetime64[s]; subtracting a pd.Timedelta yields a
        # pandas Timestamp, not a numpy datetime64, which np.searchsorted
        # can't compare against sorted_datetime's datetime64[ns] -- convert
        # back explicitly rather than relying on numpy to coerce it.
        window_start = np.datetime64(file_datetime.min() - max_time_delta)
        window_end = np.datetime64(file_datetime.max() + max_time_delta)

        lo = np.searchsorted(sorted_datetime, window_start, side='left')
        hi = np.searchsorted(sorted_datetime, window_end, side='right')
        if lo >= hi:
            continue
        positions = order[lo:hi]

        tree = BallTree(np.radians(df[['lat', 'lon']].to_numpy()), metric='haversine')
        dist_rad, idx = tree.query(argo_rad[positions], k=1)
        dist_km = dist_rad[:, 0] * EARTH_RADIUS_KM
        idx = idx[:, 0]

        cand_sss = df['sss'].to_numpy()[idx]
        cand_dt = file_datetime[idx]
        time_delta = np.abs(argo_datetime[positions] - cand_dt)
        abs_diff = np.abs(cand_sss - argo_salinity[positions])

        valid = (dist_km <= max_dist_km) & (time_delta <= max_time_delta) & (abs_diff <= max_abs_diff)
        cand_dist = np.where(valid, dist_km, np.inf)

        current_best = best_dist[positions]
        improve = cand_dist < current_best
        if not improve.any():
            continue
        best_dist[positions] = np.where(improve, cand_dist, current_best)
        best_file_idx[positions] = np.where(improve, fi, best_file_idx[positions])
        best_local_idx[positions] = np.where(improve, idx, best_local_idx[positions])

    matched_mask = np.isfinite(best_dist)
    matched_positions = np.nonzero(matched_mask)[0]

    matched_rows = [file_dfs[best_file_idx[i]].iloc[best_local_idx[i]] for i in matched_positions]
    matched_sat = pd.DataFrame(matched_rows).reset_index(drop=True) if matched_rows else pd.DataFrame()

    result = argo_df.iloc[matched_positions].rename(columns={
        'lat': 'argo_lat', 'lon': 'argo_lon', 'datetime': 'argo_datetime',
        'oceanBasin': 'argo_oceanBasin', 'salinity': 'argo_salinity', 'depth': 'argo_depth',
    }).reset_index(drop=True)
    for col in matched_sat.columns:
        result[f'sat_{col}'] = matched_sat[col].to_numpy()
    result['dist_km'] = best_dist[matched_positions]
    result['time_delta'] = np.abs(result['argo_datetime'].to_numpy() - result['sat_datetime'].to_numpy())

    return result


def box_average_match_to_argo(argo_df, file_dfs, max_dist_km, max_time_delta):
    """Schanze, Le Vine, Dinnat & Kao (2020) recommended validation matchup
    (DESIGN.md 28): for each Argo report, average EVERY raw satellite sample
    within a max_dist_km circle and +/-max_time_delta window centered on the
    report, then compare that average to the Argo salinity -- rather than
    match_to_argo's single nearest-neighbor pick. Their own ablation (Fig.
    4/5) is the reason for this: a single nearest sample carries a lot of
    retrieval noise that averaging over the box cancels out, dropping RMSD
    from ~0.5 to ~0.25 g/kg in their Aquarius analysis.

    This is a VALIDATION metric, not a training-data construction method --
    the box is centered on the Argo report, so roughly half the averaged
    samples postdate it, which isn't available at correction time in a
    real-time system (see DESIGN.md 28's discussion of why this isn't sound
    to train the deployed per-observation correction model on).

    Reuses match_to_argo's per-file time-window pruning (sort Argo obs by
    datetime once, binary-search each file's relevant window) to stay linear
    in the requested date span, but replaces the single-nearest-neighbor
    query with a radius query, accumulating a running sum/count per Argo obs
    across all files instead of tracking one best match.
    """
    argo_df = argo_df.reset_index(drop=True)
    argo_rad = np.radians(argo_df[['lat', 'lon']].to_numpy())
    argo_datetime = argo_df['datetime'].to_numpy()
    n = len(argo_df)

    sum_sss = np.zeros(n)
    count = np.zeros(n, dtype=np.int64)

    order = np.argsort(argo_datetime)
    sorted_datetime = argo_datetime[order]
    radius_rad = max_dist_km / EARTH_RADIUS_KM

    for df in file_dfs:
        file_datetime = df['datetime'].to_numpy()
        window_start = np.datetime64(file_datetime.min() - max_time_delta)
        window_end = np.datetime64(file_datetime.max() + max_time_delta)

        lo = np.searchsorted(sorted_datetime, window_start, side='left')
        hi = np.searchsorted(sorted_datetime, window_end, side='right')
        if lo >= hi:
            continue
        positions = order[lo:hi]

        tree = BallTree(np.radians(df[['lat', 'lon']].to_numpy()), metric='haversine')
        neighbor_lists = tree.query_radius(argo_rad[positions], r=radius_rad)

        file_sss = df['sss'].to_numpy()
        for local_i, neighbors in enumerate(neighbor_lists):
            if len(neighbors) == 0:
                continue
            pos = positions[local_i]
            valid = np.abs(argo_datetime[pos] - file_datetime[neighbors]) <= max_time_delta
            if not valid.any():
                continue
            valid_neighbors = neighbors[valid]
            sum_sss[pos] += file_sss[valid_neighbors].sum()
            count[pos] += len(valid_neighbors)

    matched_positions = np.nonzero(count > 0)[0]

    result = argo_df.iloc[matched_positions].rename(columns={
        'lat': 'argo_lat', 'lon': 'argo_lon', 'datetime': 'argo_datetime',
        'oceanBasin': 'argo_oceanBasin', 'salinity': 'argo_salinity', 'depth': 'argo_depth',
    }).reset_index(drop=True)
    result['sat_sss_mean'] = sum_sss[matched_positions] / count[matched_positions]
    result['n_samples'] = count[matched_positions]

    return result


def main():
    parser = argparse.ArgumentParser(description="Build raw JPL CAP SMAP-vs-Argo rich-feature matchup table")
    parser.add_argument('--raw-smap-dir', default='/Users/afeman/Desktop/work/sss-bias/data/raw_smap_cap')
    parser.add_argument('--argo-base-dir', default='/Users/afeman/Desktop/work/sss-bias/data/common_obsForge')
    parser.add_argument('--start-date', default='2022-06-01', help='YYYY-MM-DD')
    parser.add_argument('--end-date', default='2022-06-08', help='YYYY-MM-DD')
    parser.add_argument('--max-dist-km', type=float, default=50.0)
    parser.add_argument('--max-time-delta-hours', type=float, default=3.0)
    parser.add_argument('--max-depth', type=float, default=5.0, help='Argo near-surface depth cutoff (m)')
    parser.add_argument('--min-salinity', type=float, default=20.0, help='Valid-range QC lower bound (PSU)')
    parser.add_argument('--max-salinity', type=float, default=42.0, help='Valid-range QC upper bound (PSU)')
    parser.add_argument('--max-abs-diff', type=float, default=10.0,
                         help='Reject matched pairs with |satellite - Argo| beyond this (PSU); gross-error check '
                              '(only used in the default nearest-neighbor mode, not --box-average).')
    parser.add_argument('--box-average', action='store_true',
                         help='Use the Schanze et al. (2020) validation matchup instead of nearest-neighbor: '
                              'average every satellite sample within the space-time box per Argo report, rather '
                              'than picking the single nearest one. A validation metric, not training data -- '
                              'see box_average_match_to_argo\'s docstring and DESIGN.md 28.')
    parser.add_argument('--out', default='/Users/afeman/Desktop/work/sss-bias/data/matchups/smap_cap_argo_matchups.parquet')
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    print("Loading raw SMAP CAP swath data...")
    file_dfs = load_raw_smap_dir(args.raw_smap_dir, args.min_salinity, args.max_salinity,
                                  start_date=start_date, end_date=end_date,
                                  max_time_delta=pd.Timedelta(hours=args.max_time_delta_hours))
    print(f"  {len(file_dfs)} orbit files with at least one QC-pass, in-range obs\n")

    print("Loading Argo near-surface obs...")
    argo_df = load_argo_for_window(args.argo_base_dir, start_date, end_date,
                                    args.max_depth, args.min_salinity, args.max_salinity)
    print(f"  {len(argo_df)} unique near-surface profiles\n")

    print("Matching...")
    if args.box_average:
        result = box_average_match_to_argo(argo_df, file_dfs, args.max_dist_km,
                                            pd.Timedelta(hours=args.max_time_delta_hours))
    else:
        result = match_to_argo(argo_df, file_dfs, args.max_dist_km,
                                pd.Timedelta(hours=args.max_time_delta_hours), args.max_abs_diff)
    print(f"  {len(result)} matches")

    if result.empty:
        print("No matches found.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    print(f"\nWrote {len(result)} matchups to {out_path}")

    if args.box_average:
        diff = result['sat_sss_mean'] - result['argo_salinity']
        print("\nSalinity difference (satellite box-average - Argo) summary:")
        print(diff.describe())
        print(f"Bias: {diff.mean():.4f} PSU   Std: {diff.std():.4f} PSU   "
              f"RMSD: {np.sqrt((diff ** 2).mean()):.4f} PSU")
        print("\nSamples averaged per matchup:")
        print(result['n_samples'].describe())
    else:
        print("\nDistance (km) summary:")
        print(result['dist_km'].describe())
        print("\nTime delta summary:")
        print(result['time_delta'].describe())


if __name__ == '__main__':
    main()
