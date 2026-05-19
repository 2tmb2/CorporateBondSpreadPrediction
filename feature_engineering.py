import pandas as pd
import numpy as np


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    b = d['BAA10Y']

    # BAA10Y trend / momentum / regime
    d['BAA10Y_roll21_mean']  = b.rolling(21).mean()
    d['BAA10Y_roll21_std']   = b.rolling(21).std()
    d['BAA10Y_mom5']         = b - b.shift(5)
    d['BAA10Y_mom21']        = b - b.shift(21)
    d['BAA10Y_accel']        = d['BAA10Y_mom5'] - d['BAA10Y_mom5'].shift(5)
    d['BAA10Y_zscore252']    = (b - b.rolling(252).mean()) / b.rolling(252).std()
    d['BAA10Y_RSI14']        = _rsi(b, 14)

    # Yield curve
    ycs = d['DGS10'] - d['DGS2']
    d['yield_curve_slope']   = ycs
    d['yield_curve_chg21']   = ycs - ycs.shift(21)

    # VIX
    d['VIX_roll21_mean']     = d['VIXCLS'].rolling(21).mean()
    d['VIX_mom21']           = d['VIXCLS'] - d['VIXCLS'].shift(21)
    d['VIX_spike']           = d['VIXCLS'] / d['VIX_roll21_mean']

    # Equity
    d['SP500_mom21']         = d['SP500'] / d['SP500'].shift(21) - 1

    # Rates
    d['DFF_mom21']           = d['DFF'] - d['DFF'].shift(21)

    # Credit quality
    cts = d['DBAA'] - d['DAAA']
    d['credit_tier_spread']  = cts
    d['credit_tier_chg21']   = cts - cts.shift(21)

    # Cross-asset flow: +ve = equity up AND spread falling (risk-on)
    d['risk_on_off']         = d['SP500_mom21'] * (-d['BAA10Y_mom21'])

    # Macro
    d['oil_mom21']           = d['DCOILWTICO'] / d['DCOILWTICO'].shift(21) - 1
    d['NFCI_trend']          = d['NFCI'] - d['NFCI'].shift(20)

    # Seasonality
    d['month_sin'] = np.sin(2 * np.pi * d.index.month / 12)
    d['month_cos'] = np.cos(2 * np.pi * d.index.month / 12)

    # Secular time trend — years since 1990-01-01
    # Lets models learn structural regime shifts (e.g. pre/post-2008 credit markets)
    d['time_idx'] = (d.index - pd.Timestamp('1990-01-01')).days / 365.25

    return d
