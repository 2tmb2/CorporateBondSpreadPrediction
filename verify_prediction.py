"""
verify_prediction.py
--------------------
Fetches the latest BAA10Y spread from FRED and compares it against
the predictions produced by predict.ipynb.

Run after predict.ipynb has been executed and after FRED has published
the target date's data (usually by 3-4 PM ET on the same business day).

Usage:
    python verify_prediction.py
"""

import io
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA10Y"

# ── Paste your predict.ipynb output here after each run ──────────────────────
# Update these values each time you run predict.ipynb.
# current_spread = spread on the feature date (the last known value before prediction)
# predictions    = {H: ensemble_predicted_spread}

PREDICTION_DATE  = date(2026, 5, 16)          # the date being predicted
CURRENT_SPREAD   = 1.6200                      # BAA10Y as of the feature date (2026-05-15)

# Run predict.ipynb with target_date = 2026-05-16 and paste ensemble outputs here.
PREDICTIONS = {
    1:  1.6196,   # H=1  ensemble (input: 2026-05-15) -- update after running predict.ipynb
    7:  1.6545,   # H=7  ensemble (input: 2026-05-09) -- update after running predict.ipynb
    28: 1.7186,   # H=28 ensemble (input: 2026-04-18) -- update after running predict.ipynb
}

# ── Fetch latest BAA10Y from FRED ─────────────────────────────────────────────
print("Fetching BAA10Y from FRED ...")
try:
    resp = requests.get(FRED_URL, timeout=15)
    resp.raise_for_status()
    baa = pd.read_csv(
        io.StringIO(resp.text),
        index_col=0, parse_dates=True,
        na_values=['.'],
    )['BAA10Y'].dropna()
    print(f"  Latest FRED value: {baa.index[-1].date()}  =  {baa.iloc[-1]:.4f}")
    print(f"  Second latest:     {baa.index[-2].date()}  =  {baa.iloc[-2]:.4f}")
except Exception as e:
    print(f"  ERROR fetching from FRED: {e}")
    sys.exit(1)

# ── Find the actual spread on PREDICTION_DATE ─────────────────────────────────
target_ts = pd.Timestamp(PREDICTION_DATE)
if target_ts in baa.index:
    actual = baa[target_ts]
    print(f"\n  Actual BAA10Y on {PREDICTION_DATE}: {actual:.4f}  [FOUND]")
else:
    # Try the nearest available date after prediction date
    later = baa[baa.index >= target_ts]
    if len(later) == 0:
        print(f"\n  BAA10Y for {PREDICTION_DATE} not yet published on FRED.")
        print("  Check back after 3-4 PM ET on the target date (or the next business day).")
        sys.exit(0)
    actual_date = later.index[0].date()
    actual      = later.iloc[0]
    print(f"\n  BAA10Y for {PREDICTION_DATE} not found. Using nearest: {actual_date} = {actual:.4f}")

# ── Compare predictions ───────────────────────────────────────────────────────
print()
print("=" * 65)
print(f"  PREDICTION VERIFICATION  --  target date: {PREDICTION_DATE}")
print(f"  Actual BAA10Y : {actual:.4f}")
print(f"  Current spread (feature date baseline): {CURRENT_SPREAD:.4f}")
print("=" * 65)

actual_dir = "UP" if actual > CURRENT_SPREAD else ("DOWN" if actual < CURRENT_SPREAD else "FLAT")
print(f"  Actual direction (vs feature-date spread): {actual_dir}")
print()
print(f"  {'H':>4}  {'Predicted':>10}  {'Actual':>8}  {'Error':>8}  {'Pred Dir':>9}  {'Correct?':>9}")
print(f"  {'-'*58}")

for H, pred in sorted(PREDICTIONS.items()):
    error    = pred - actual
    pred_dir = "UP" if pred > CURRENT_SPREAD else ("DOWN" if pred < CURRENT_SPREAD else "FLAT")
    correct  = (pred_dir == actual_dir) if actual_dir != "FLAT" else "N/A"
    correct_str = "YES" if correct is True else ("NO" if correct is False else "N/A")
    print(f"  {H:>4}  {pred:>10.4f}  {actual:>8.4f}  {error:>+8.4f}  {pred_dir:>9}  {correct_str:>9}")

print()
rmse = np.sqrt(np.mean([(PREDICTIONS[H] - actual) ** 2 for H in PREDICTIONS]))
print(f"  RMSE across horizons: {rmse:.4f}")
print()
print("NOTE: H=1 is the fairest comparison for next-day prediction.")
print("      H=7 and H=28 use older input features and target the same date,")
print("      so their errors reflect how much conditions changed since their feature date.")
