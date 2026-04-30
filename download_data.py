"""
Download economic and financial data series for Corporate Bond Spread Prediction.
- FRED series via public FRED CSV endpoint (no API key required)
- ETF volume data (LQD, HYG) via yfinance (not available on FRED)
- Exchange rate volatility computed as 30-day rolling std of daily EUR/USD log returns
"""

import os
import io
import warnings
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

START = "1990-01-01"
END = datetime.today().strftime("%Y-%m-%d")

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

def download_fred(series_id: str) -> pd.DataFrame:
    """Download a FRED series as a DataFrame via the public CSV endpoint."""
    url = FRED_URL.format(series_id=series_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
    df.columns = [series_id]
    df.replace(".", float("nan"), inplace=True)
    df = df.astype(float)
    df = df.loc[START:END]
    return df

# ---------------------------------------------------------------------------
# FRED series: {filename_stem: (series_id, description)}
# ---------------------------------------------------------------------------
FRED_SERIES = {
    # Target
    "BAA10Y":      ("BAA10Y",      "Moody's Baa Corporate Bond Spread over 10-Year Treasury"),
    # Monetary Policy / Interest Rates
    "FEDFUNDS":    ("FEDFUNDS",    "Federal Funds Effective Rate"),
    "GS10":        ("GS10",        "10-Year Treasury Constant Maturity Rate"),
    "GS2":         ("GS2",         "2-Year Treasury Constant Maturity Rate"),
    # Inflation / Price Pressure
    "CPIAUCSL":    ("CPIAUCSL",    "Consumer Price Index (CPI) - All Urban Consumers"),
    "PPIACO":      ("PPIACO",      "Producer Price Index (PPI) - All Commodities"),
    "PCEPI":       ("PCEPI",       "Personal Consumption Expenditures Price Index"),
    # Labor Market / Economic Activity
    "UNRATE":      ("UNRATE",      "Unemployment Rate"),
    "GDPC1":       ("GDPC1",       "Real GDP (Quarterly)"),
    "INDPRO":      ("INDPRO",      "Industrial Production Index"),
    # Financial Market Conditions
    "VIXCLS":      ("VIXCLS",      "CBOE Volatility Index (VIX)"),
    "SP500":       ("SP500",       "S&P 500 Index"),
    "NFCI":        ("NFCI",        "Chicago Fed National Financial Conditions Index"),
    # Credit Market
    "AAA":         ("AAA",         "Moody's Seasoned Aaa Corporate Bond Yield"),
    "BAA":         ("BAA",         "Moody's Seasoned Baa Corporate Bond Yield"),
    # Liquidity Proxies
    "TEDRATE":     ("TEDRATE",     "TED Spread (discontinued Jan 2023; historical data retained)"),
    # Consumer / Business Strength
    "RSAFS":       ("RSAFS",       "Advance Retail Sales: Retail and Food Services"),
    "UMCSENT":     ("UMCSENT",     "University of Michigan Consumer Sentiment"),
    "TCU":         ("TCU",         "Capacity Utilization: Total Index"),
    # International / External Factors
    "DTWEXBGS":    ("DTWEXBGS",    "Trade Weighted US Dollar Index: Broad, Goods & Services"),
    "DCOILWTICO":  ("DCOILWTICO",  "Crude Oil Prices: WTI - Cushing, Oklahoma"),
    "DEXUSEU":     ("DEXUSEU",     "US Dollar / Euro Exchange Rate (used for FX volatility)"),
}

# ---------------------------------------------------------------------------
# yfinance tickers for series not available on FRED
# {filename_stem: (ticker, column, description)}
# ---------------------------------------------------------------------------
YF_VOLUME_SERIES = {
    "LQD_Volume": ("LQD",  "Volume", "iShares iBoxx $ Investment Grade Corporate Bond ETF Volume"),
    "HYG_Volume": ("HYG",  "Volume", "iShares iBoxx $ High Yield Corporate Bond ETF Volume"),
}

print(f"Downloading data from {START} to {END}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Download FRED series
# ---------------------------------------------------------------------------
failed = []
for name, (series_id, desc) in FRED_SERIES.items():
    try:
        df = download_fred(series_id)
        path = os.path.join(DATA_DIR, f"{name}.csv")
        df.to_csv(path)
        print(f"  [OK] {name:15s} — {desc}")
    except Exception as e:
        print(f"  [FAIL] {name:15s} — {e}")
        failed.append(name)

# ---------------------------------------------------------------------------
# Compute Exchange Rate Volatility (30-day rolling std of log returns)
# ---------------------------------------------------------------------------
eurusd_path = os.path.join(DATA_DIR, "DEXUSEU.csv")
if os.path.exists(eurusd_path):
    try:
        eurusd = pd.read_csv(eurusd_path, index_col=0, parse_dates=True)
        eurusd = eurusd.replace(".", float("nan")).astype(float)
        log_ret = eurusd["DEXUSEU"].pct_change().apply(lambda x: x)  # simple returns
        fx_vol = log_ret.rolling(window=30).std() * (252 ** 0.5)     # annualised
        fx_vol.name = "EURUSD_FX_Vol_30d"
        fx_vol_df = fx_vol.to_frame()
        path = os.path.join(DATA_DIR, "EURUSD_FX_Volatility.csv")
        fx_vol_df.to_csv(path)
        print(f"  [OK] EURUSD_FX_Volatility — Exchange Rate Volatility (30-day rolling, annualised)")
    except Exception as e:
        print(f"  [FAIL] EURUSD_FX_Volatility — {e}")

# ---------------------------------------------------------------------------
# Download ETF volume data via yfinance
# ---------------------------------------------------------------------------
for name, (ticker, col, desc) in YF_VOLUME_SERIES.items():
    try:
        df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
        vol = df[[col]].copy()
        vol.columns = [f"{ticker}_{col}"]
        path = os.path.join(DATA_DIR, f"{name}.csv")
        vol.to_csv(path)
        print(f"  [OK] {name:15s} — {desc}")
    except Exception as e:
        print(f"  [FAIL] {name:15s} — {e}")
        failed.append(name)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
all_files = sorted(os.listdir(DATA_DIR))
print(f"Files saved to ./data/  ({len(all_files)} total):")
for f in all_files:
    fpath = os.path.join(DATA_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {f:<40s}  {size_kb:6.1f} KB")

if failed:
    print(f"\nFailed series: {failed}")
else:
    print("\nAll series downloaded successfully.")
