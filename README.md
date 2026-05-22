# CSSE415 Final Project: Corporate Bond Spread

Predicts the Moody's Baa corporate bond spread over the 10-Year Treasury (BAA10Y) at horizons of 1, 7, and 28 days using an ensemble of 10 machine learning models trained on macroeconomic and market data from 1990 to present.

## Environment setup

1. Create a python virtual environment using ```python3 -m venv .venv``` (note that .venv can be replaced with whatever name you want)
2. Download dependencies by running ```pip install -r requirements.txt``` once you have activated the virtual environment.

## Workflow

1. `build_dataset.py` → builds `data/dataset.csv` and `data/dataset-no-recessions.csv`
2. `predict.ipynb` → backtests models, generates charts, trains production models, and outputs the forecast

## File descriptions

### Root

| File | Description |
|------|-------------|
| `build_dataset.py` | Downloads ~30 FRED and yfinance series (daily, monthly, quarterly) from 1990 to present and merges them into a single daily CSV. Non-daily series are forward-filled and accompanied by a `_days_since` column. Also saves a recession-stripped variant. Run this to refresh the dataset before training. |
| `feature_engineering.py` | Shared module containing `add_trend_features(df)`. Called by `predict.ipynb` and the individual model notebooks. Adds 22 engineered columns: BAA10Y momentum/RSI/z-score, yield curve slope, VIX regime signals, equity return, credit tier spread, risk-on/off composite, oil return, NFCI trend, and seasonal encodings. |
| `predict.ipynb` | Main notebook. Runs four sequential phases: (1) leakage-free backtest of all 10 models across Q4 2025 and Q1 2026, (2) saves `backtest_comparison.png` and `accuracy_diagram.png`, (3) retrains all models on the most recent 30 years of data, (4) prints and plots ensemble spread forecasts for H=1, H=7, H=28. Uses direction-conditional precision weighting for the ensemble. |
| `stacking.ipynb` | Experimental stacking ensemble notebook (work in progress). |
| `verify_prediction.py` | After running `predict.ipynb`, paste the ensemble outputs into this script and run it to fetch the actual BAA10Y value from FRED and compare predicted vs. actual for each horizon. Prints RMSE and directional accuracy. |
| `feature_guide.txt` | Plain-text reference describing every raw and engineered feature: what it measures, its source frequency, and why it is relevant to credit spreads. |
| `predict_overview.txt` | Prose overview of `predict.ipynb` for presentation purposes. Covers the ensemble weighting methodology, backtest design, horizon definitions, and a proposed future improvement (regime-matched precision weighting). |
| `requirements.txt` | Python package dependencies. |
| `backtest_comparison.png` | Saved output of Phase 2. 3×3 grid of RMSE bars, directional accuracy bars, and time-series overlays (one column per horizon). |
| `accuracy_diagram.png` | Saved output of Phase 2. DirAcc heatmap, RMSE heatmap, and grouped bar chart across all models and horizons. |

### `data/`

| File | Description |
|------|-------------|
| `dataset.csv` | Full daily dataset from 1990-01-01 to present including recession periods (~13,000 rows, ~30 columns + engineered features). Target column is `BAA10Y`. |
| `dataset-no-recessions.csv` | Same as `dataset.csv` with four recession windows removed (1990–91, 2001, 2008–09, 2020). Used to train the `no-rec` model variants in `predict.ipynb`. |

### `models/with-recessions/` and `models/without-recessions/`

Exploratory notebooks used during development. Each trains a single model type on the respective dataset variant using a 70/15/15 chronological train/val/test split and reports MAE, RMSE, and directional accuracy. These are not used by `predict.ipynb` directly but informed model selection and hyperparameter choices.

| Notebook | Model type |
|----------|------------|
| `baseline-*.ipynb` | Naive baselines (constant mean, constant median, no-change). Establishes the performance floor all other models must beat. |
| `gradientboosted-*.ipynb` | XGBoost gradient boosted trees with early stopping. |
| `linear_models_*.ipynb` | Ridge and Lasso regression with 80/20 cross-validation for alpha selection. |
| `randomforest-*.ipynb` | Random Forest regressor (500 trees). |
| `randomforest-classifier-*.ipynb` | Random Forest classifier predicting spread direction (UP/DOWN) rather than a numeric value. |
| `xgboost-classification-*.ipynb` | XGBoost classifier for directional prediction with precision/recall/F1 evaluation. |
