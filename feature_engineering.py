import pandas as pd
import numpy as np


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    b = d['BAA10Y']

    d['BAA10Y_roll5_mean']  = b.rolling(5).mean()
    d['BAA10Y_roll21_mean'] = b.rolling(21).mean()
    d['BAA10Y_roll63_mean'] = b.rolling(63).mean()
    d['BAA10Y_roll5_std']   = b.rolling(5).std()
    d['BAA10Y_roll21_std']  = b.rolling(21).std()
    d['BAA10Y_mom5']        = b - b.shift(5)
    d['BAA10Y_mom21']       = b - b.shift(21)

    d['yield_curve_slope']  = d['DGS10'] - d['DGS2']
    d['VIX_roll5_mean']     = d['VIXCLS'].rolling(5).mean()
    d['VIX_roll21_mean']    = d['VIXCLS'].rolling(21).mean()
    d['SP500_mom21']        = d['SP500'] / d['SP500'].shift(21) - 1
    d['DFF_mom21']          = d['DFF'] - d['DFF'].shift(21)
    d['credit_tier_spread'] = d['DBAA'] - d['DAAA']

    d['month_sin'] = np.sin(2 * np.pi * d.index.month / 12)
    d['month_cos'] = np.cos(2 * np.pi * d.index.month / 12)

    return d
