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
# absorb float-drift leakage (Argo has no float ID to block on directly).
TRAIN_START = pd.Timestamp('2021-01-01')
TRAIN_END = pd.Timestamp('2022-12-31')
VAL_START = pd.Timestamp('2023-01-15')
VAL_END = pd.Timestamp('2023-06-30')
TEST_START = pd.Timestamp('2023-07-15')
TEST_END = pd.Timestamp('2023-12-31')


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
    """Chronological train/validate/test split with embargo gaps."""
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
