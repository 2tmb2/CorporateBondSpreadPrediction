import numpy as np
import pandas as pd

def build_df(split: float) -> pd.DataFrame:
    df = pd.read_csv('./data/dataset-no-recessions.csv', parse_dates=['Date']).set_index('Date').sort_index()

    # H = Forecast horizon (days ahead).
    H = 1
    # split = test/train split

    # Supervised setup: features at time t predict BAA10Y at time t+H.
    # Shifting by -H aligns each row's target with the future value.
    y_future = df['BAA10Y'].shift(-H)
    y_now    = df['BAA10Y']
    X_full   = df.drop(columns=['BAA10Y'])

    # Drop rows where either the current spread (needed for naive baseline),
    # the future spread (the target), or any feature is NaN.
    mask = y_future.notna() & y_now.notna() & X_full.notna().all(axis=1)
    X = X_full.loc[mask]
    y = y_future.loc[mask]
    y_t = y_now.loc[mask]  # current spread, kept for the naive last-value baseline

    # Chronological split — NEVER shuffle time-series data.
    # OOB error from the forest plays the role of validation (each sample is
    # scored by the trees that didn't see it), so we only need a held-out test
    # set. This frees up ~15% more data for training.
    n = len(y)
    train_end = int(n * split)

    X_train, X_test = X.iloc[:train_end], X.iloc[train_end:]
    y_train, y_test = y.iloc[:train_end], y.iloc[train_end:]
    y_t_test        = y_t.iloc[train_end:]   # current spread, kept for naive-baseline comparison

    return X_train, y_train, X_test, y_test