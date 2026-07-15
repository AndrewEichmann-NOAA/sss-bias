"""
Feature engineering and train/val/test split for the satellite-SSS-vs-Argo
matchup table (works for either SMAP or SMOS matchup tables -- both use the
same sat_* column prefix, see build_matchups.py).

All model inputs are derived from the satellite observation alone (its own
SSS, lat/lon, time, oceanBasin) -- at deployment there is no matched Argo
profile to draw on, only the satellite swath being corrected. `argo_salinity`
is only ever used as the regression target, never as a feature.
"""

import numpy as np
import pandas as pd

BASIN_CODES = [0, 1, 2, 3, 4, 5]

FEATURE_COLUMNS = ['sat_sss', 'sat_lat', 'lon_sin', 'lon_cos', 'doy_sin', 'doy_cos'] + \
    [f'basin_{c}' for c in BASIN_CODES]
TARGET_COLUMN = 'argo_salinity'

# Split boundaries from DESIGN.md section 6, chosen after discovering SMAP
# obs in this repo only span 2021-2023. 14-day embargo gaps at each boundary
# absorb float-drift leakage for rows with no recovered float ID (see below).
TRAIN_START = pd.Timestamp('2021-01-01')
TRAIN_END = pd.Timestamp('2022-12-31')
VAL_START = pd.Timestamp('2023-01-15')
VAL_END = pd.Timestamp('2023-06-30')
TEST_START = pd.Timestamp('2023-07-15')
TEST_END = pd.Timestamp('2023-12-31')

_PARTITION_ORDER = {'train': 0, 'val': 1, 'test': 2}


def add_features(df):
    """Add cyclically-encoded time/longitude features and basin one-hots."""
    df = df.copy()

    day_of_year = df['sat_datetime'].dt.dayofyear.astype(float)
    df['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

    lon_rad = np.radians(df['sat_lon'].astype(float))
    df['lon_sin'] = np.sin(lon_rad)
    df['lon_cos'] = np.cos(lon_rad)

    for code in BASIN_CODES:
        df[f'basin_{code}'] = (df['sat_oceanBasin'] == code).astype(float)

    return df


def split_data(df, date_col='cycle_date'):
    """Chronological train/validate/test split with embargo gaps, made
    float-aware where a WMO float ID was recovered (see DESIGN.md 17/18):
    a float is assigned entirely to whichever of train/val/test contains its
    EARLIEST in-window observation, and every other row from that same float
    follows it there -- so no float ever appears in more than one partition.

    Rows with no recovered float ID (`wmo` NaN -- ~5% of SMAP, ~21% of SMOS)
    keep the naive date-only assignment; the embargo gaps are their only
    protection against float-drift leakage, same as before this change.
    Embargo-period rows are dropped exactly as before regardless of wmo --
    this reassignment only ever moves rows *between* train/val/test, never
    pulls an embargo-dropped row back in.
    """
    dates = pd.to_datetime(df[date_col])
    naive = pd.Series(np.nan, index=df.index, dtype=object)
    naive[(dates >= TRAIN_START) & (dates <= TRAIN_END)] = 'train'
    naive[(dates >= VAL_START) & (dates <= VAL_END)] = 'val'
    naive[(dates >= TEST_START) & (dates <= TEST_END)] = 'test'

    in_window = naive.notna()
    has_wmo = df['wmo'].notna() if 'wmo' in df.columns else pd.Series(False, index=df.index)
    eligible = in_window & has_wmo

    owner = pd.DataFrame({
        'wmo': df.loc[eligible, 'wmo'],
        'partition': naive[eligible],
        'date': dates[eligible],
    })
    owner['order'] = owner['partition'].map(_PARTITION_ORDER)
    owning_partition = (owner.sort_values(['wmo', 'order', 'date'])
                             .groupby('wmo')['partition'].first())

    final = naive.copy()
    reassigned = df['wmo'].map(owning_partition) if 'wmo' in df.columns else pd.Series(np.nan, index=df.index)
    mask = reassigned.notna()
    final[mask] = reassigned[mask]
    # A float-owned row outside any train/val/test window (pure embargo) has
    # no entry in `naive` and thus no entry in `owner`/`owning_partition`
    # either, so `reassigned` never pulls embargo rows back in -- only rows
    # already in some window can move to a *different* window.

    train = df[final == 'train'].reset_index(drop=True)
    val = df[final == 'val'].reset_index(drop=True)
    test = df[final == 'test'].reset_index(drop=True)
    return train, val, test


def split_data_naive(df, date_col='cycle_date'):
    """Pure date-based train/validate/test split, ignoring float ID entirely
    (the original pre-18 behavior). Kept alongside split_data() specifically
    to let float_leakage_diagnostic.py test whether float-based leakage is
    empirically material for this model, rather than assuming it -- see the
    discussion in DESIGN.md 18.3.
    """
    dates = pd.to_datetime(df[date_col])
    train = df[(dates >= TRAIN_START) & (dates <= TRAIN_END)].reset_index(drop=True)
    val = df[(dates >= VAL_START) & (dates <= VAL_END)].reset_index(drop=True)
    test = df[(dates >= TEST_START) & (dates <= TEST_END)].reset_index(drop=True)
    return train, val, test


class Standardizer:
    """Z-score using train-set statistics only, to avoid val/test leakage."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
