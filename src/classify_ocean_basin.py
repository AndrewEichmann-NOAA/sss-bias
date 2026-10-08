#!/usr/bin/env python3
"""
Lat/lon-based ocean basin classifier, standing in for obsForge's oceanBasin
code where it isn't available (GDAC-direct Argo profiles, DESIGN.md 34 --
oceanBasin has no GDAC equivalent).

Boundaries were empirically derived and validated against ~40,000 real
(lat, lon, oceanBasin) triples from the obsForge-derived matchup table
(smap_cap_argo_matchups_gdacqc.parquet), not assumed from a textbook
definition -- see DESIGN.md 35. Simple three-threshold rule (Southern Ocean
by latitude, then Atlantic/Indian/Pacific by longitude) achieves 89.3%
overall agreement with the real obsForge codes.

Does NOT reproduce basin 0 (marginal/enclosed seas, e.g. Mediterranean, Red
Sea -- geographically scattered, not a contiguous region a simple lat/lon
rule can capture) or basin 4 (Arctic, n=101, indistinguishable from high-
latitude Atlantic by a simple threshold). Both are fine to drop: permutation
feature importance (DESIGN.md, rich_feature_importance.parquet) found both
negligible (basin_0 rank 28/41, basin_4 rank 35/41, near-zero ΔRMSE),
unlike basin_1/2/3/5 which ranked 6th, 12th, 21st, and 13th respectively.
"""

import numpy as np

SOUTHERN_LAT_CUTOFF = -40.0
ATLANTIC_LON_RANGE = (-70.0, 20.0)
INDIAN_LON_RANGE = (20.0, 125.0)


def classify_basin(lat, lon):
    """Returns an oceanBasin code array matching obsForge's convention:
    1=Atlantic, 2=Pacific (default), 3=Indian, 5=Southern Ocean.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    basin = np.full(lat.shape, 2.0)

    southern = lat < SOUTHERN_LAT_CUTOFF
    atlantic = (~southern) & (lon >= ATLANTIC_LON_RANGE[0]) & (lon < ATLANTIC_LON_RANGE[1])
    indian = (~southern) & (lon >= INDIAN_LON_RANGE[0]) & (lon < INDIAN_LON_RANGE[1])

    basin[atlantic] = 1.0
    basin[indian] = 3.0
    basin[southern] = 5.0
    return basin
