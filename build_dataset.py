"""
Download all series and merge into a single DAILY CSV: data/dataset.csv

Series classification
---------------------
DAILY (FRED business-day or yfinance):
  BAA10Y, DFF, DGS10, DGS2, DAAA, DBAA, VIXCLS, DCOILWTICO,
  DEXUSEU, DTWEXBGS, TEDRATE, SP500 (^GSPC), LQD volume, HYG volume
  EURUSD_FX_Vol  (computed: 30-day rolling annualised std of EUR/USD returns)

NON-DAILY → forward-filled to daily + companion '<NAME>_days_since' column
  Monthly : CPIAUCSL, PPIACO, PCEPI, UNRATE, INDPRO, RSAFS, UMCSENT, TCU
  Weekly  : NFCI
  Quarterly: GDPC1

days_since = 0 on the observation date itself, increments each calendar day
             until the next observation arrives.

Date range: 1990-01-01 → 2026-01-01  (calendar days, inclusive)
"""

import io
import os
import warnings

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

START = "1990-01-01"
END   = "2026-01-01"

DAILY_INDEX = pd.date_range(start=START, end=END, freq="D")

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fetch_fred(series_id: str) -> pd.Series:
    """Download a FRED series and return as a Series indexed by date."""
    url = FRED_BASE.format(series_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
    s = df.iloc[:, 0]
    s.name = series_id
    s = s.replace(".", np.nan).astype(float)
    return s.loc[START:END]


def to_daily_ffill(s: pd.Series) -> pd.Series:
    """Reindex to DAILY_INDEX and forward-fill only (no look-ahead)."""
    return s.reindex(DAILY_INDEX).ffill()


def days_since(s: pd.Series) -> pd.Series:
    """
    Return a daily series counting calendar days elapsed since the last
    non-NaN observation.  Value is 0 on the observation date itself.
    Before the first observation the value is NaN.
    """
    sparse = s.reindex(DAILY_INDEX)
    obs_set = set(sparse.dropna().index)
    result = np.full(len(DAILY_INDEX), np.nan)
    last_idx = -1
    for i, date in enumerate(DAILY_INDEX):
        if date in obs_set:
            last_idx = i
        if last_idx >= 0:
            result[i] = i - last_idx
    return pd.Series(result, index=DAILY_INDEX, dtype=float)


# ---------------------------------------------------------------------------
# Download daily FRED series
# ---------------------------------------------------------------------------
print("Downloading daily FRED series...")
daily_fred = {
    "BAA10Y"   : "BAA10Y",   # Baa–10Y spread (target Y)
    "DFF"      : "DFF",      # Federal Funds Rate (daily)
    "DGS10"    : "DGS10",    # 10-Year Treasury (daily)
    "DGS2"     : "DGS2",     # 2-Year Treasury (daily)
    "DAAA"     : "DAAA",     # Moody's Aaa yield (daily)
    "DBAA"     : "DBAA",     # Moody's Baa yield (daily)
    "VIXCLS"   : "VIXCLS",   # VIX (daily)
    "DCOILWTICO": "DCOILWTICO",  # WTI Crude Oil (daily)
}
daily_raw: dict[str, pd.Series] = {}
for name, sid in daily_fred.items():
    try:
        daily_raw[name] = fetch_fred(sid)
        print(f"  [OK]   {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

# ---------------------------------------------------------------------------
# Download non-daily FRED series
# ---------------------------------------------------------------------------
print("\nDownloading non-daily FRED series...")
nondaily_fred = {
    # Monthly
    "CPIAUCSL"  : "CPIAUCSL",   # CPI
    "PPIACO"    : "PPIACO",     # PPI
    "PCEPI"     : "PCEPI",      # PCE price index
    "UNRATE"    : "UNRATE",     # Unemployment rate
    "INDPRO"    : "INDPRO",     # Industrial production
    "UMCSENT"   : "UMCSENT",    # Michigan consumer sentiment
    "TCU"       : "TCU",        # Capacity utilisation
    # Weekly
    "NFCI"      : "NFCI",       # Chicago Fed National Financial Conditions Index
    # Quarterly
    "GDPC1"     : "GDPC1",      # Real GDP
    # Money Supply (monthly)
    "M1SL"      : "M1SL",       # M1 Money Stock
    "M2SL"      : "M2SL",       # M2 Money Stock
    # Savings deposits — needed to back out the May-2020 M1 redefinition
    "SAVINGSL"  : "SAVINGSL",   # Savings Deposits (SA, monthly)
}
nondaily_raw: dict[str, pd.Series] = {}
for name, sid in nondaily_fred.items():
    try:
        nondaily_raw[name] = fetch_fred(sid)
        print(f"  [OK]   {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

# ---------------------------------------------------------------------------
# Adjust M1SL for the May-2020 redefinition
# ---------------------------------------------------------------------------
# Beginning May 2020: FRED moved savings deposits (SAVINGSL) into M1 ("other
# liquid deposits"), and SAVINGSL was discontinued.  To put the full series on
# the new (post-May-2020) basis, add SAVINGSL to M1SL for all dates < 2020-05-01.
# Post-May-2020 M1SL already includes savings, so it is left unchanged.
# M2SL needs no adjustment: M2 total is the same under both definitions.
if "M1SL" in nondaily_raw and "SAVINGSL" in nondaily_raw:
    m1 = nondaily_raw["M1SL"].copy()
    savings = nondaily_raw["SAVINGSL"]
    cutoff = pd.Timestamp("2020-05-01")
    pre_mask = m1.index < cutoff
    # Align savings to M1 index (both monthly, but dates may differ slightly)
    # SAVINGSL was discontinued in May 2020, so values only exist pre-cutoff anyway
    savings_aligned = savings.reindex(m1.index)
    m1.loc[pre_mask] = m1.loc[pre_mask] + savings_aligned.loc[pre_mask]
    nondaily_raw["M1SL"] = m1
    print("\nM1SL adjusted: savings deposits added for dates < 2020-05-01 "
          "(maps old M1 onto new post-May-2020 definition)")
    # SAVINGSL is only a helper — drop it so it doesn't appear as its own feature
    del nondaily_raw["SAVINGSL"]

# ---------------------------------------------------------------------------
# Download yfinance series (daily)
# ---------------------------------------------------------------------------
print("\nDownloading yfinance series...")
yf_series: dict[str, pd.Series] = {}
yf_tickers = {
    "SP500"     : ("^GSPC", "Close"),
}
for name, (ticker, col) in yf_tickers.items():
    try:
        df = yf.download(ticker, start=START, end="2026-01-02",
                         auto_adjust=True, progress=False)
        s = df[col].squeeze()
        s.name = name
        s = s.loc[START:END]
        yf_series[name] = s
        print(f"  [OK]   {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

# ---------------------------------------------------------------------------
# Assemble dataset
# ---------------------------------------------------------------------------
print("\nAssembling daily dataset...")
frames: dict[str, pd.Series] = {}

# Daily FRED series — forward-fill weekends/holidays
for name, s in daily_raw.items():
    frames[name] = to_daily_ffill(s)

# yfinance daily — SP500 forward-filled
if "SP500" in yf_series:
    frames["SP500"] = to_daily_ffill(yf_series["SP500"])

# Non-daily series — forward-fill value + days_since column
for name, s in nondaily_raw.items():
    frames[name]                    = to_daily_ffill(s)
    frames[f"{name}_days_since"]    = days_since(s)

# ---------------------------------------------------------------------------
# Build DataFrame and save
# ---------------------------------------------------------------------------
# Desired column order
col_order = [
    # Target
    "BAA10Y",
    # Interest rates (daily)
    "DFF", "DGS10", "DGS2",
    # Inflation (monthly → +days_since)
    "CPIAUCSL", "CPIAUCSL_days_since",
    "PPIACO",   "PPIACO_days_since",
    "PCEPI",    "PCEPI_days_since",
    # Labor / Activity (monthly/quarterly → +days_since)
    "UNRATE",  "UNRATE_days_since",
    "GDPC1",   "GDPC1_days_since",
    "INDPRO",  "INDPRO_days_since",
    # Financial conditions
    "VIXCLS", "SP500",
    "NFCI",   "NFCI_days_since",
    # Credit (daily)
    "DAAA", "DBAA",
    # Consumer / Business (monthly → +days_since)
    "UMCSENT", "UMCSENT_days_since",
    "TCU",     "TCU_days_since",
    # International
    "DCOILWTICO",
    # Money Supply (monthly → +days_since)
    "M1SL", "M1SL_days_since",
    "M2SL", "M2SL_days_since",
]
# Keep only columns that were successfully downloaded
col_order = [c for c in col_order if c in frames]

df = pd.DataFrame({c: frames[c] for c in col_order}, index=DAILY_INDEX)
df.index.name = "Date"
df = df.loc[START:END]

out_path = os.path.join(DATA_DIR, "dataset.csv")
df.to_csv(out_path)

# ---------------------------------------------------------------------------
# Save no-recessions dataset (same columns, recession periods removed)
# ---------------------------------------------------------------------------
RECESSION_PERIODS = [
    ("1990-07-01", "1991-07-01"),
    ("2001-04-01", "2001-12-31"),
    ("2008-01-01", "2009-09-30"),
    ("2020-01-01", "2020-06-30"),
]

recession_mask = pd.Series(False, index=df.index)
for start_r, end_r in RECESSION_PERIODS:
    recession_mask |= (df.index >= start_r) & (df.index <= end_r)

df_nr = df.loc[~recession_mask]
nr_path = os.path.join(DATA_DIR, "dataset-no-recessions.csv")
df_nr.to_csv(nr_path)
print(f"No-recessions saved: {nr_path}  ({df_nr.shape[0]} rows)")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print(f"\nDataset saved : {out_path}")
print(f"Shape         : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Date range    : {df.index.min().date()} → {df.index.max().date()}")
print(f"\n{'Column':<28}  {'Non-NaN':>7}  {'First':>12}  {'Last':>12}  Note")
print("-" * 80)
for col in df.columns:
    n      = df[col].notna().sum()
    first  = df[col].first_valid_index()
    last   = df[col].last_valid_index()
    note   = "days_since" if col.endswith("_days_since") else ""
    print(f"  {col:<26}  {n:7d}  "
          f"{str(first.date()) if first else 'N/A':>12}  "
          f"{str(last.date())  if last  else 'N/A':>12}  {note}")
