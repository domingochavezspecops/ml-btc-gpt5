#!/usr/bin/env python3
"""
Generate a comprehensive, human-readable trading report for the optimized BTCUSDT model.

- Loads optimized model artifact and summary
- Recomputes features, creates signals using recommended thresholds and persistence
- Backtests with transaction costs; computes precision/recall and risk metrics
- Produces 10-15 concrete trade examples with timestamps, prices, feature values, and explanations
- Exports:
  - reports_opt/trading_report.txt
  - reports_opt/trade_examples.csv

Usage:
  python generate_trading_report.py \
    --csv data/btcusdt_perp_5m_with_indicators.csv \
    --model models_opt/best_trading_model_xgb.joblib \
    --summary reports_opt/opt_summary.json --log INFO
"""
from __future__ import annotations

import os
import json
import argparse
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib

# Provide EncodedModel so joblib can unpickle artifacts created with it
class EncodedModel:
    def __init__(self, base_model, classes_sorted):
        self.base_model = base_model
        self.label_to_idx = {int(c): i for i, c in enumerate(classes_sorted)}
        self.idx_to_label = {i: int(c) for i, c in enumerate(classes_sorted)}
    def fit(self, X, y):
        import numpy as _np
        y_enc = _np.array([self.label_to_idx[int(v)] for v in y], dtype=int)
        self.base_model.fit(X, y_enc)
        return self
    def predict(self, X):
        import numpy as _np
        y_enc = _np.asarray(self.base_model.predict(X)).astype(int)
        return _np.array([self.idx_to_label[int(v)] for v in y_enc], dtype=int)
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)
    def __getattr__(self, name):
        # Avoid recursion during unpickling when base_model may not yet be set
        try:
            base = object.__getattribute__(self, 'base_model')
        except Exception:
            raise AttributeError(name)
        return getattr(base, name)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    d = df.copy()
    d["ret_1"] = d["close"].pct_change(1)
    d["ret_3"] = d["close"].pct_change(3)
    d["ret_6"] = d["close"].pct_change(6)
    d["ret_12"] = d["close"].pct_change(12)
    d["vol_12"] = d["ret_1"].rolling(12).std()
    d["vol_24"] = d["ret_1"].rolling(24).std()
    d["body"] = (d["close"] - d["open"]) / d["open"]
    d["range_pct"] = (d["high"] - d["low"]) / d["close"]
    d["upper_wick"] = (d["high"] - d[["open", "close"]].max(axis=1)) / d["close"]
    d["lower_wick"] = (d[["open", "close"]].min(axis=1) - d["low"]) / d["close"]
    d["close_to_ema"] = (d["close"] - d["ema_55"]) / d["ema_55"]
    d["band_width"] = (d["kc_upper"] - d["kc_lower"]) / d["ema_55"]
    d["band_pos"] = (d["close"] - d["kc_lower"]) / (d["kc_upper"] - d["kc_lower"])
    d["z_to_ema"] = (d["close"] - d["ema_55"]) / d["std_55"]
    d["rsi"] = d["rsi_14"]
    d["rsi_norm"] = (d["rsi"] - 50.0) / 50.0
    d["rsi_overbought"] = (d["rsi"] > 70).astype(int)
    d["rsi_oversold"] = (d["rsi"] < 30).astype(int)
    d["vol_z24"] = (d["volume"] - d["volume"].rolling(24).mean()) / d["volume"].rolling(24).std()
    d["qvol_z24"] = (d["quote_volume"] - d["quote_volume"].rolling(24).mean()) / d["quote_volume"].rolling(24).std()
    feature_cols = [
        "open","high","low","close","volume","quote_volume",
        "ema_55","kc_upper","kc_lower","rsi_14","std_55",
        "ret_1","ret_3","ret_6","ret_12","vol_12","vol_24",
        "body","range_pct","upper_wick","lower_wick",
        "close_to_ema","band_width","band_pos","z_to_ema",
        "rsi","rsi_norm","rsi_overbought","rsi_oversold",
        "vol_z24","qvol_z24",
    ]
    d[feature_cols] = d[feature_cols].shift(1)
    d = d.replace([np.inf, -np.inf], np.nan)
    return d, feature_cols


def build_labels(df: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    close = df["close"].astype(float)
    fut = pd.concat([close.shift(-k) for k in range(1, horizon+1)], axis=1)
    up_ret = (fut.max(axis=1) - close) / close
    dn_ret = (fut.min(axis=1) - close) / close
    long_sig = up_ret >= threshold
    short_sig = dn_ret <= -threshold
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[long_sig & ~short_sig] = 1
    labels[short_sig & ~long_sig] = -1
    labels.iloc[-horizon:] = np.nan
    return labels


def prob_to_signal_with_indices(proba: np.ndarray, idx_long: int | None, idx_short: int | None, th_long: float, th_short: float) -> np.ndarray:
    p_long = proba[:, idx_long] if idx_long is not None else np.zeros(len(proba))
    p_short = proba[:, idx_short] if idx_short is not None else np.zeros(len(proba))
    sig = np.zeros(len(proba), dtype=int)
    sig[p_long >= th_long] = 1
    sig[p_short >= th_short] = -1
    both = (p_long >= th_long) & (p_short >= th_short)
    sig[both] = np.where(p_long[both] >= p_short[both], 1, -1)
    return sig


def apply_persistence(signals: np.ndarray, persist: int) -> np.ndarray:
    if persist <= 1:
        return signals
    out = np.zeros_like(signals)
    run = 0
    cur = 0
    for i, s in enumerate(signals):
        if s == cur and s != 0:
            run += 1
        else:
            cur = s
            run = 1 if s != 0 else 0
        out[i] = s if (s != 0 and run >= persist) else 0
    return out


def backtest(prices: pd.Series, signals: np.ndarray, horizon: int, cost_roundtrip: float = 0.0012) -> Dict[str, Any]:
    signals = signals.astype(int)
    n = len(prices)
    pos = 0
    entry_price = 0.0
    equity = [1.0]
    rets = []
    num_trades = 0
    wins = 0
    durations = []
    entries = []
    exits = []

    holding = 0
    for i in range(1, n):
        price = prices.iloc[i]
        # exit by horizon
        if pos != 0:
            holding += 1
            if holding >= horizon:
                gross_ret = (price - entry_price) / entry_price if pos == 1 else (entry_price - price) / entry_price
                trade_ret = gross_ret - cost_roundtrip
                equity.append(equity[-1] * (1.0 + trade_ret))
                rets.append(trade_ret)
                num_trades += 1
                wins += 1 if trade_ret > 0 else 0
                durations.append(holding)
                exits.append(i)
                pos = 0
                entry_price = 0.0
                holding = 0
                continue
        # entry
        if pos == 0 and signals[i] != 0:
            pos = signals[i]
            entry_price = price
            equity.append(equity[-1] * (1.0 - cost_roundtrip/2))
            rets.append(-cost_roundtrip/2)
            holding = 0
            entries.append(i)
        else:
            equity.append(equity[-1])
            rets.append(0.0)

    eq = np.array(equity)
    rets = np.array(rets)
    cumret = eq[-1] - 1.0
    sharpe = (rets.mean() / rets.std()) * np.sqrt(288 * 365) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    max_dd = ((eq - peak) / peak).min()
    avg_dur = float(np.mean(durations)) if durations else 0.0

    return {
        'cum_return': float(cumret),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'num_trades': int(num_trades),
        'win_rate': float(wins / num_trades) if num_trades > 0 else 0.0,
        'avg_duration_bars': avg_dur,
        'entries': entries,
        'exits': exits,
        'equity_curve': eq.tolist(),
    }


def feature_interpretations() -> Dict[str, str]:
    return {
        'close_to_ema': 'Price distance from EMA55 (positive=above EMA; momentum/uptrend)',
        'band_pos': 'Position within Keltner band (0=lower, 1=upper); near 1 often implies stretched up move',
        'z_to_ema': 'Z-score of price vs EMA using 55-std; large absolute implies stretched condition',
        'band_width': 'Band width normalized by EMA; regime (compression vs expansion)',
        'rsi': 'RSI(14): >70 overbought, <30 oversold',
        'ret_3': '3-bar return (momentum)',
        'ret_6': '6-bar return (momentum)',
        'vol_12': 'Short-term volatility of returns',
        'upper_wick': 'Upper wick proportion; potential rejection if large',
        'lower_wick': 'Lower wick proportion; potential demand if large',
    }


def main():
    ap = argparse.ArgumentParser(description='Generate human-readable trading report from optimized model.')
    ap.add_argument('--csv', default='data/btcusdt_perp_5m_with_indicators.csv')
    ap.add_argument('--model', default=None, help='Path to joblib model; if omitted, tries to infer from summary')
    ap.add_argument('--summary', default='reports_opt/opt_summary.json')
    ap.add_argument('--outdir', default='reports_opt')
    ap.add_argument('--log', default='INFO')
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format='%(asctime)s | %(levelname)s | %(message)s')
    ensure_dir(args.outdir)

    # Load summary and model artifact
    with open(args.summary, 'r') as f:
        summary = json.load(f)
    best_model_name = summary['model'] if 'model' in summary else summary.get('best_model', 'xgb')
    # opt_summary structure from optimize script
    if 'model' not in summary and 'best_model' in summary:
        best_model_name = summary['best_model']
    model_path = args.model
    if model_path is None:
        model_path = f"models_opt/best_trading_model_{best_model_name}.joblib"
    art = joblib.load(model_path)
    model = art['model']
    feat_selected: List[str] = art['features']
    horizon: int = int(art['horizon'])
    label_threshold: float = float(art['label_threshold'])
    prob_thresholds: Dict[str, float] = art['prob_thresholds']
    persistence: int = int(art['persistence'])
    trading_cost: float = float(art.get('trading_cost', 0.0012))

    # Load data and features
    df = load_data(args.csv)
    df_feat, base_features = engineer_features(df)
    df_feat['label'] = build_labels(df_feat, horizon, label_threshold)

    # Prepare X,y
    X = df_feat[feat_selected]
    y = df_feat['label']
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask].astype(int)
    times = df.loc[X.index, 'open_time']

    # Predict
    try:
        proba = model.predict_proba(X)
    except Exception:
        # Some models might require numpy array
        proba = model.predict_proba(X.values)
    # Determine probability column indices for long/short
    idx_long = None
    idx_short = None
    if hasattr(model, 'label_to_idx'):
        l2i = model.label_to_idx
        idx_long = l2i.get(1, None)
        idx_short = l2i.get(-1, None)
    elif hasattr(model, 'classes_'):
        classes_arr = list(map(int, getattr(model, 'classes_')))
        if 1 in classes_arr:
            idx_long = classes_arr.index(1)
        if -1 in classes_arr:
            idx_short = classes_arr.index(-1)
    else:
        # fallback assume [-1,0,1]
        idx_short = 0
        idx_long = 2 if proba.shape[1] >= 3 else None
    # Generate signals
    sig = prob_to_signal_with_indices(proba, idx_long, idx_short, th_long=prob_thresholds['long'], th_short=prob_thresholds['short'])
    sig = apply_persistence(sig, persistence)

    # Metrics: precision/recall for LONG/SHORT comparing to labels
    from sklearn.metrics import precision_recall_fscore_support
    pr, rc, f1, _ = precision_recall_fscore_support(y, sig, labels=[-1,0,1], zero_division=0)

    # Backtest
    prices = df.loc[X.index, 'close']
    bt = backtest(prices, sig, horizon=horizon, cost_roundtrip=trading_cost)

    # Fold metrics (simple two-fold split for reporting)
    from math import floor
    splits = []
    n = len(X)
    k = 3
    step = floor(n / k)
    for i in range(k):
        s = i*step
        e = (i+1)*step if i < k-1 else n
        splits.append((s,e))
    fold_rows = []
    for fi, (s, e) in enumerate(splits):
        pr_f, rc_f, f1_f, _ = precision_recall_fscore_support(y.iloc[s:e], sig[s:e], labels=[-1,0,1], zero_division=0)
        metrics_f = backtest(prices.iloc[s:e], sig[s:e], horizon=horizon, cost_roundtrip=trading_cost)
        fold_rows.append({
            'fold': fi,
            'cum_return': metrics_f['cum_return'],
            'sharpe': metrics_f['sharpe'],
            'max_drawdown': metrics_f['max_drawdown'],
            'num_trades': metrics_f['num_trades'],
            'win_rate': metrics_f['win_rate'],
            'precision_long': float(pr_f[2]),
            'recall_long': float(rc_f[2]),
            'precision_short': float(pr_f[0]),
            'recall_short': float(rc_f[0]),
        })

    # Build feature importance ranking if available
    def model_importance(model, feature_names: List[str]) -> pd.DataFrame:
        clf = model
        if hasattr(clf, 'base_model'):
            clf = clf.base_model
        if hasattr(clf, 'feature_importances_'):
            imp = np.array(clf.feature_importances_)
        elif hasattr(clf, 'coef_'):
            coef = clf.coef_
            imp = np.mean(np.abs(coef), axis=0) if getattr(coef, 'ndim', 1) > 1 else np.abs(coef)
        else:
            imp = np.zeros(len(feature_names))
        return pd.DataFrame({'feature': feature_names, 'importance': imp}).sort_values('importance', ascending=False).reset_index(drop=True)

    imp_df = model_importance(model, feat_selected)
    interp = feature_interpretations()

    # Build trade examples
    entries = bt['entries']
    exits = bt['exits']
    examples = []
    capital = 1000.0
    key_feats = ['rsi','band_pos','close_to_ema','ret_3','vol_12']
    for en, ex in zip(entries, exits):
        ts = times.iloc[en]
        direction = 'LONG' if sig[en] == 1 else 'SHORT'
        entry_price = float(prices.iloc[en])
        exit_price = float(prices.iloc[ex])
        pct = (exit_price - entry_price) / entry_price if direction == 'LONG' else (entry_price - exit_price) / entry_price
        pnl_pct = float(pct - trading_cost)
        pnl_usd = float(pnl_pct * capital)
        feat_vals = {k: float(df_feat.loc[X.index[en], k]) if k in df_feat.columns else np.nan for k in key_feats}
        # Explanation template
        exp_parts = []
        if not np.isnan(feat_vals.get('rsi', np.nan)):
            if feat_vals['rsi'] > 70:
                exp_parts.append('RSI was overbought')
            elif feat_vals['rsi'] < 30:
                exp_parts.append('RSI was oversold')
            else:
                exp_parts.append('RSI near neutral')
        if not np.isnan(feat_vals.get('band_pos', np.nan)):
            if feat_vals['band_pos'] > 0.8:
                exp_parts.append('price near upper band')
            elif feat_vals['band_pos'] < 0.2:
                exp_parts.append('price near lower band')
        if not np.isnan(feat_vals.get('close_to_ema', np.nan)):
            exp_parts.append(('above EMA' if feat_vals['close_to_ema'] > 0 else 'below EMA'))
        if not np.isnan(feat_vals.get('ret_3', np.nan)):
            exp_parts.append(('recent momentum up' if feat_vals['ret_3'] > 0 else 'recent momentum down'))
        explanation = ', '.join(exp_parts) or 'Model probability exceeded threshold with persistence'

        examples.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M UTC'),
            'direction': direction,
            'entry_price': round(entry_price, 2),
            'exit_time': times.iloc[ex].strftime('%Y-%m-%d %H:%M UTC'),
            'exit_price': round(exit_price, 2),
            'pnl_pct': round(pnl_pct*100, 3),
            'pnl_usd_on_1000': round(pnl_usd, 2),
            **{f: round(feat_vals[f], 4) if not np.isnan(feat_vals[f]) else '' for f in key_feats},
            'explanation': explanation,
        })

    # Select 10-15 examples: top 5 winners, top 5 losers, + random others
    if examples:
        ex_df = pd.DataFrame(examples)
        winners = ex_df.sort_values('pnl_pct', ascending=False).head(5)
        losers = ex_df.sort_values('pnl_pct', ascending=True).head(5)
        rest = ex_df.drop(pd.concat([winners, losers]).index)
        rng = np.random.default_rng(42)
        extra = rest.sample(n=min(3, len(rest)), random_state=42) if len(rest) > 0 else pd.DataFrame()
        ex_sel = pd.concat([winners, losers, extra]).head(15)
    else:
        ex_sel = pd.DataFrame()

    # Write trade examples CSV
    csv_path = os.path.join(args.outdir, 'trade_examples.csv')
    ex_sel.to_csv(csv_path, index=False)

    # Build human-readable report
    lines: List[str] = []
    lines.append('=== Model Summary ===')
    lines.append(f"Best model: {best_model_name}")
    # Hyperparameters: if XGB, try to fetch from base_model
    params = {}
    clf = model
    if hasattr(clf, 'base_model'):
        clf = clf.base_model
    if hasattr(clf, 'get_params'):
        params = clf.get_params()
    key_params = {k: params[k] for k in ['n_estimators','max_depth','learning_rate','subsample','colsample_bytree','reg_lambda'] if k in params}
    lines.append(f"Key hyperparameters: {key_params}")
    lines.append(f"Recommended thresholds: LONG={prob_thresholds['long']}, SHORT={prob_thresholds['short']}")
    lines.append(f"Signal persistence: {persistence} consecutive bars")
    lines.append('Selected features (top 15 by importance):')
    for _, row in imp_df.head(15).iterrows():
        feat = row['feature']
        imp = row['importance']
        desc = feature_interpretations().get(feat, '')
        lines.append(f"  - {feat}: importance={imp:.4f} {('-> ' + desc) if desc else ''}")

    lines.append('\n=== Performance Metrics ===')
    lines.append(f"Overall backtest: cumulative return={bt['cum_return']:.3f}, Sharpe={bt['sharpe']:.2f}, Max Drawdown={bt['max_drawdown']:.3f}")
    lines.append(f"Trades: {bt['num_trades']}, Win rate={bt['win_rate']:.2%}, Avg trade duration={bt['avg_duration_bars']:.1f} bars")
    lines.append(f"Precision/Recall (LONG): {pr[2]:.3f}/{rc[2]:.3f} | (SHORT): {pr[0]:.3f}/{rc[0]:.3f}")
    lines.append('Walk-forward folds:')
    for r in fold_rows:
        lines.append(f"  Fold {r['fold']}: PnL={r['cum_return']:.3f}, Sharpe={r['sharpe']:.2f}, MDD={r['max_drawdown']:.3f}, Trades={r['num_trades']}, Win%={r['win_rate']:.2%}, P_long={r['precision_long']:.3f}, R_long={r['recall_long']:.3f}, P_short={r['precision_short']:.3f}, R_short={r['recall_short']:.3f}")

    lines.append('\n=== Concrete Trade Examples (timestamps in UTC) ===')
    if not ex_sel.empty:
        for _, r in ex_sel.iterrows():
            lines.append(f"- {r['timestamp']}: {r['direction']} entry={r['entry_price']} -> exit {r['exit_time']} @ {r['exit_price']} | PnL={r['pnl_pct']:.2f}% (${r['pnl_usd_on_1000']:.2f} on $1000). Features: RSI={r['rsi']}, band_pos={r['band_pos']}, close_to_ema={r['close_to_ema']}, ret_3={r['ret_3']}, vol_12={r['vol_12']}. Why: {r['explanation']}")
    else:
        lines.append('No trades generated under the current settings.')

    lines.append('\n=== Signal Interpretation Guide ===')
    lines.append('- A LONG signal triggers when model P(LONG) >= threshold and persists for the required bars; similarly for SHORT with its threshold.')
    lines.append('- Persistence filter reduces noise: only act when the same signal appears in consecutive bars (e.g., 2 bars).')
    lines.append('- Manual hints:')
    lines.append('  * LONG setups tend to have RSI normalizing from oversold, price rising above EMA55 (close_to_ema>0), band_pos moving up from lower band, and positive short-term momentum (ret_3>0).')
    lines.append('  * SHORT setups often show RSI rolling over from overbought, price below EMA55 (close_to_ema<0), band_pos near upper band rolling down, and negative momentum (ret_3<0).')
    lines.append('- Position sizing: scale position with confidence and recent volatility. Higher model probability and lower vol_12 -> larger size; high vol -> reduce size.')
    lines.append(f"- Costs modeled at {trading_cost*100:.2f}% round-trip; ensure your live fees/slippage are equal or lower.")

    # Write text report
    txt_path = os.path.join(args.outdir, 'trading_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('Report written to:', txt_path)
    print('Trade examples CSV:', csv_path)


if __name__ == '__main__':
    main()

