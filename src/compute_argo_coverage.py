#!/usr/bin/env python3
"""
Scan the full local common_obsForge archive for every unique near-surface
(<=5m) Argo profile, globally deduplicated across the ~9-cycle-file
replication (see DESIGN.md 16.1), and save (lat, lon) for each to a parquet
table -- this is the "raw Argo coverage" independent of any satellite
matching, used to compare against matched-profile density in
plot_argo_coverage.py.
"""

import numpy as np
import pandas as pd
import xarray as xr

from process_netcdf_cycles import NetCDFCycleProcessor

MAX_DEPTH = 5.0


def main():
    processor = NetCDFCycleProcessor('/Users/afeman/Desktop/work/sss-bias/data/common_obsForge')
    cycle_dirs = processor.find_cycle_directories(None, None)
    print(f"{len(cycle_dirs)} cycle directories")

    all_keys = set()
    n_with_argo = 0
    for i, (date, cycle, cycle_path) in enumerate(cycle_dirs):
        f = cycle_path / 'insitu' / f'gdas.t{cycle}z.insitu_salt_profile_argo.nc'
        if not f.exists():
            continue
        n_with_argo += 1
        try:
            meta = xr.open_dataset(f, group='MetaData', engine='netcdf4', decode_times=False)
            lat = meta['latitude'].values
            lon = meta['longitude'].values
            depth = meta['depth'].values
            odt = meta['originalDateTime'].values
            meta.close()
        except Exception as e:
            print(f"  error {f}: {e}")
            continue

        near_surface = (depth <= MAX_DEPTH) & ~np.isnan(lat) & ~np.isnan(lon)
        keys = zip(lat[near_surface].round(4), lon[near_surface].round(4), odt[near_surface].round(0))
        all_keys.update(keys)

        if (i + 1) % 1000 == 0:
            print(f"  ...{i+1}/{len(cycle_dirs)}, running unique count: {len(all_keys)}")

    print(f"\nCycles with an argo file: {n_with_argo}")
    print(f"Globally deduplicated unique near-surface Argo profiles: {len(all_keys)}")

    df = pd.DataFrame(all_keys, columns=['lat', 'lon', 'original_datetime_epoch'])
    out_path = '/Users/afeman/Desktop/work/sss-bias/data/matchups/all_argo_profiles.parquet'
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
