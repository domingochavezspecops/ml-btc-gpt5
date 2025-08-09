# ml-btc-gpt5

A complete, production-minded pipeline for fetching BTCUSDT perpetual futures data (5m), computing indicators, training ML models to generate long/short signals, optimizing for real-world trading performance, and producing a human-readable trading report with concrete examples.

## Contents

- Data fetching from Binance Futures (5m, last N months)
- Indicators: EMA(55), Keltner-like bands (±1 std), RSI(14)
- Feature engineering from price/volume/indicators
- Labeling with forward horizon + return threshold for LONG/SHORT/NEUTRAL
- Baseline training (Logistic Regression, Random Forest, XGBoost)
- Trading-optimized pipeline (hyperparameter tuning, threshold/persistence filtering, walk-forward backtest with costs)
- Report generation with 10–15 concrete, chart-friendly trade examples

## Repository structure

```
.
├─ fetch_btcusdt_futures_5m.py        # Download klines, compute indicators, save CSVs
├─ train_btcusdt_signals.py           # Baseline ML training and evaluation
├─ optimize_trading_model.py          # Trading-focused optimization (tuning + backtest)
├─ generate_trading_report.py         # Human-readable report + trade examples CSV
├─ data/                              # Raw and processed CSVs (generated)
├─ models/                            # Baseline models (generated)
├─ models_opt/                        # Optimized trading model (generated)
├─ reports/                           # Baseline metrics & reports (generated)
└─ reports_opt/                       # Optimization summaries & trading report (generated)
```

## Requirements

- Python 3.10+
- Packages:
  - requests, pandas, numpy
  - scikit-learn, joblib
  - xgboost (optional but recommended)

Install:

```
pip install requests pandas numpy scikit-learn joblib xgboost
```

## 1) Fetch data + indicators

Fetch last 12 months of 5m BTCUSDT futures klines from Binance and compute indicators. Handles pagination/rate limits and fills missing candles.

```
python fetch_btcusdt_futures_5m.py --months 12 --symbol BTCUSDT --interval 5m --log INFO
```

Outputs:
- data/btcusdt_perp_5m_raw.csv
- data/btcusdt_perp_5m_with_indicators.csv

Notes:
- Timestamps are UTC; data sorted oldest→newest
- Indicators: EMA(55), std(55), Keltner-like bands = ema_55 ± std_55, RSI(14)

## 2) Baseline training (metrics-first)

Train multi-class classifier for LONG/SHORT/NEUTRAL using horizon- and threshold-based labels (no look-ahead; features shifted by 1 bar).

```
python train_btcusdt_signals.py \
  --csv data/btcusdt_perp_5m_with_indicators.csv \
  --horizon 12 --threshold 0.0075 \
  --models lr rf xgb --log INFO
```

Outputs:
- models/best_model_*.joblib
- reports/ (classification reports, confusion matrix, feature importance, walk-forward summary)

## 3) Trading-optimized pipeline (PnL-first)

Optimizes for real-world profitability: hyperparameter tuning, probability threshold optimization, persistence filtering, walk-forward backtesting with transaction costs, and simple risk sizing.

```
python optimize_trading_model.py \
  --csv data/btcusdt_perp_5m_with_indicators.csv \
  --horizon 12 --threshold 0.0075 \
  --models xgb rf --imbalance undersample \
  --cost 0.0012 --log INFO
```

Key features:
- Hyperparameter sweep for RF/XGB over multiple windows
- Threshold optimization for LONG/SHORT probabilities to favor precision over recall
- Signal persistence requirement (e.g., 2 bars) to reduce noise
- Walk-forward evaluation; scoring combines PnL, drawdown, and precision
- Transaction costs included (default 0.12% round-trip)

Outputs:
- models_opt/best_trading_model_<model>.joblib
- reports_opt/opt_result_*.json, opt_summary.json

Example (from a recent run):
- Best model: XGBoost
- Best params: {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 0.8, "reg_lambda": 1.0}
- Recommended thresholds: LONG=0.70, SHORT=0.90
- Persistence: 2 consecutive bars

## 4) Generate human-readable trading report

Produces trading_report.txt and trade_examples.csv with concrete, chart-friendly examples.

```
python generate_trading_report.py \
  --csv data/btcusdt_perp_5m_with_indicators.csv \
  --model models_opt/best_trading_model_xgb.joblib \
  --summary reports_opt/opt_summary.json \
  --log INFO
```

Outputs:
- reports_opt/trading_report.txt
- reports_opt/trade_examples.csv (10–15 entries with UTC timestamps, entry/exit prices, PnL, feature values, and explanations)

## Features & labels (summary)

- Features: price/volume + indicators (ema_55, kc_upper/lower, rsi_14, std_55) + derived (returns, volatility, band position, RSI transforms, wick/body, volume z-scores)
- Labels:
  - LONG (1): within next N bars, price rises ≥ X%
  - SHORT (-1): within next N bars, price falls ≤ -X%
  - NEUTRAL (0) otherwise (conflicts default to neutral)
  - Typical: N=12 bars (1h), X=0.75% (0.0075)

## Risk & trading assumptions

- Transaction cost: default 0.12% round-trip (configure via --cost)
- Persistence filter reduces false positives (signals must repeat)
- Position sizing: example sizing tied to volatility and confidence (customize as needed)

## Repro checklist

1) Install dependencies
2) Fetch data: fetch_btcusdt_futures_5m.py
3) (Optional) Baseline training: train_btcusdt_signals.py
4) Optimize for trading: optimize_trading_model.py
5) Generate report: generate_trading_report.py

## Notes

- This repository is for research/education. Past performance is not indicative of future results. Not financial advice.
- Consider adding a .gitignore to avoid committing large CSVs/models (e.g., data/*.csv, models*/)**
