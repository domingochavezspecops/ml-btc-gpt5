#!/usr/bin/env python3
"""
Trading-optimized ML pipeline for BTCUSDT 5m signals.

Enhancements:
- Hyperparameter tuning (RF, XGB) with time-series walk-forward CV using a trading score
- Threshold optimization for LONG/SHORT probabilities focused on precision and profit
- Feature selection via model-based importance (top-k sweep)
- Robust walk-forward evaluation with realistic backtest including transaction costs
- Signal filtering (min confidence + persistence) and position sizing from confidence & volatility
- Imbalance handling options (class_weight or neutral undersampling)

Outputs: best model, thresholds, feature importance, backtest metrics and equity curve.

Example:
  python optimize_trading_model.py \
    --csv data/btcusdt_perp_5m_with_indicators.csv \
    --horizon 12 --threshold 0.0075 \
    --models xgb rf --imbalance undersample --log INFO
"""
from __future__ import annotations

import os
import json
import argparse
import logging
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import joblib

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False


class EncodedModel:
    """Wrap a classifier to encode labels {-1,0,1} to {0,1,2} for training (e.g., XGBoost)."""
    def __init__(self, base_model: Any, classes_sorted: List[int]):
        self.base_model = base_model
        self.label_to_idx = {int(c): i for i, c in enumerate(classes_sorted)}
        self.idx_to_label = {i: int(c) for i, c in enumerate(classes_sorted)}

    def fit(self, X, y):
        y_enc = np.array([self.label_to_idx[int(v)] for v in y], dtype=int)
        self.base_model.fit(X, y_enc)
        return self

    def predict(self, X):
        y_enc = np.asarray(self.base_model.predict(X)).astype(int)
        return np.array([self.idx_to_label[int(v)] for v in y_enc], dtype=int)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    def __getattr__(self, name):
        return getattr(self.base_model, name)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
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


def prepare_xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols]
    y = df['label']
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask].astype(int)


def walk_forward_splits(n: int, n_folds: int = 5, train_frac: float = 0.6, val_frac: float = 0.2, test_frac: float = 0.2) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    win = int(n * (train_frac + val_frac + test_frac))
    step = int(win * 0.4)
    splits = []
    start = 0
    while True:
        end = start + win
        if end > n:
            break
        t_end = start + int(win * train_frac / (train_frac + val_frac + test_frac))
        v_end = t_end + int(win * val_frac / (train_frac + val_frac + test_frac))
        splits.append((np.arange(start, t_end), np.arange(t_end, v_end), np.arange(v_end, end)))
        start += step
    return splits


def imbalance_resample(X: pd.DataFrame, y: pd.Series, method: str = 'class_weight', neutral_ratio: float = 1.0) -> Tuple[pd.DataFrame, pd.Series, Any]:
    if method == 'undersample':
        # Downsample neutral to match (sum of action classes) * neutral_ratio
        idx_long = y[y == 1].index
        idx_short = y[y == -1].index
        idx_neu = y[y == 0].index
        target = int((len(idx_long) + len(idx_short)) * neutral_ratio)
        if target < len(idx_neu) and target > 0:
            idx_neu_down = np.random.choice(idx_neu, size=target, replace=False)
            keep = np.concatenate([idx_long, idx_short, idx_neu_down])
            keep.sort()
            return X.loc[keep], y.loc[keep], None
        return X, y, None
    # class_weight path
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return X, y, {int(c): float(w) for c, w in zip(classes, weights)}


def build_model(name: str, classes: np.ndarray, class_weight: Any = None):
    if name == 'rf':
        return RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=5, n_jobs=-1, class_weight=class_weight, random_state=42)
    if name == 'xgb' and HAS_XGB:
        classes_sorted = sorted(int(c) for c in classes)
        base = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            objective='multi:softprob', num_class=len(classes_sorted), tree_method='hist', reg_lambda=1.0, n_jobs=-1, random_state=42
        )
        return EncodedModel(base, classes_sorted)
    raise ValueError('Unknown or unavailable model: %s' % name)


def model_importance(model, feature_names: List[str]) -> pd.DataFrame:
    clf = model
    if isinstance(model, Pipeline):
        clf = model.named_steps.get('clf', model)
    if hasattr(clf, 'feature_importances_'):
        imp = np.array(clf.feature_importances_)
    elif hasattr(clf, 'coef_'):
        coef = clf.coef_
        imp = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        imp = np.zeros(len(feature_names))
    return pd.DataFrame({'feature': feature_names, 'importance': imp}).sort_values('importance', ascending=False).reset_index(drop=True)


def prob_to_signal(proba: np.ndarray, classes: List[int], th_long: float, th_short: float) -> np.ndarray:
    # classes order must align with proba columns
    class_to_idx = {c: i for i, c in enumerate(classes)}
    p_long = proba[:, class_to_idx[1]] if 1 in class_to_idx else np.zeros(len(proba))
    p_short = proba[:, class_to_idx[-1]] if -1 in class_to_idx else np.zeros(len(proba))
    sig = np.zeros(len(proba), dtype=int)
    sig[p_long >= th_long] = 1
    sig[p_short >= th_short] = -1
    # If both exceed (rare), pick higher probability
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


def backtest(prices: pd.Series, signals: np.ndarray, horizon: int, cost_roundtrip: float = 0.0012, vol: pd.Series | None = None, max_size: float = 1.0) -> Dict[str, Any]:
    # prices: close prices aligned with signals
    signals = signals.astype(int)
    n = len(prices)
    pos = 0  # -1, 0, 1
    entry_price = 0.0
    equity = [1.0]
    rets = []
    num_trades = 0
    wins = 0
    holding = 0

    for i in range(1, n):
        price_prev = prices.iloc[i-1]
        price = prices.iloc[i]

        # position sizing: based on rolling vol (vol) and signal strength via persistence already
        size = 1.0
        if vol is not None and not np.isnan(vol.iloc[i]):
            target_vol = 0.01  # target 1% volatility unit
            size = min(max_size, max(0.1, target_vol / max(vol.iloc[i], 1e-6)))

        # exit condition: horizon-based if in position
        if pos != 0:
            holding += 1
            if holding >= horizon:
                # exit
                gross_ret = (price - entry_price) / entry_price if pos == 1 else (entry_price - price) / entry_price
                trade_ret = size * (gross_ret - cost_roundtrip)
                equity.append(equity[-1] * (1.0 + trade_ret))
                rets.append(trade_ret)
                num_trades += 1
                if trade_ret > 0:
                    wins += 1
                pos = 0
                entry_price = 0.0
                holding = 0
                continue

        # entry if flat
        if pos == 0 and signals[i] != 0:
            pos = signals[i]
            entry_price = price
            # apply entry cost immediately as equity drag (half of roundtrip)
            equity.append(equity[-1] * (1.0 - cost_roundtrip/2))
            rets.append(-cost_roundtrip/2)
            holding = 0
        else:
            # mark-to-market if holding
            if pos != 0:
                mtm = (price - price_prev) / price_prev if pos == 1 else (price_prev - price) / price_prev
                rets.append(size * mtm)
                equity.append(equity[-1] * (1.0 + size * mtm))
            else:
                rets.append(0.0)
                equity.append(equity[-1])

    eq = np.array(equity)
    rets = np.array(rets)
    cumret = eq[-1] - 1.0
    # Sharpe (5m bars ~ 288 per day); use per-bar returns annualized assuming 365d
    if rets.std() > 0:
        sharpe = (rets.mean() / rets.std()) * np.sqrt(288 * 365)
    else:
        sharpe = 0.0
    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()

    return {
        'cum_return': float(cumret),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'num_trades': int(num_trades),
        'win_rate': float(wins / num_trades) if num_trades > 0 else 0.0,
        'equity_curve': eq.tolist(),
    }


def trading_score(metrics: Dict[str, Any], prec_long: float, prec_short: float) -> float:
    # prioritize PnL and lower drawdown, reward precision on action classes
    pnl = metrics['cum_return']
    mdd = -metrics['max_drawdown']  # negative number; invert sign
    score = pnl + 0.1 * (prec_long + prec_short) + 0.5 * mdd
    return float(score)


def evaluate_window(model, X_train, y_train, X_val, y_val, X_test, y_test, prices_val, prices_test, horizon: int, th_grid=(0.6,0.7,0.8,0.9), persist_grid=(1,2,3), cost=0.0012) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Fit
    model.fit(X_train, y_train)

    # Validation proba and classes
    proba_val = model.predict_proba(X_val)
    classes_model = sorted(int(c) for c in np.unique(y_train))
    # Optimize thresholds on validation
    best = None
    for th_l in th_grid:
        for th_s in th_grid:
            for pers in persist_grid:
                sig = prob_to_signal(proba_val, classes_model, th_l, th_s)
                sig = apply_persistence(sig, pers)
                pr, rc, f1, _ = precision_recall_fscore_support(y_val, sig, labels=[-1,0,1], zero_division=0)
                metrics_val = backtest(prices_val, sig, horizon, cost_roundtrip=cost)
                score = trading_score(metrics_val, prec_long=float(pr[2]), prec_short=float(pr[0]))
                cand = (score, th_l, th_s, pers, pr)
                if best is None or score > best[0]:
                    best = cand
    assert best is not None
    _, th_l_best, th_s_best, pers_best, pr_best = best

    # Test with best thresholds
    proba_test = model.predict_proba(X_test)
    sig_test = prob_to_signal(proba_test, classes_model, th_l_best, th_s_best)
    sig_test = apply_persistence(sig_test, pers_best)
    pr_t, rc_t, f1_t, _ = precision_recall_fscore_support(y_test, sig_test, labels=[-1,0,1], zero_division=0)
    metrics_test = backtest(prices_test, sig_test, horizon, cost_roundtrip=cost)

    val_summary = {
        'threshold_long': th_l_best,
        'threshold_short': th_s_best,
        'persistence': pers_best,
        'precision_short': float(pr_best[0]),
        'precision_neutral': float(pr_best[1]),
        'precision_long': float(pr_best[2]),
        'metrics': metrics_val,
    }
    test_summary = {
        'precision_short': float(pr_t[0]),
        'precision_neutral': float(pr_t[1]),
        'precision_long': float(pr_t[2]),
        'metrics': metrics_test,
    }
    return val_summary, test_summary


def hyperparam_grid(name: str) -> List[Dict[str, Any]]:
    if name == 'rf':
        grid = []
        for n in [300, 500]:
            for md in [None, 10, 20]:
                for mss in [2, 5, 10]:
                    for mf in ['sqrt', 0.5, None]:
                        grid.append({'n_estimators': n, 'max_depth': md, 'min_samples_split': mss, 'max_features': mf})
        return grid
    if name == 'xgb':
        grid = []
        for n in [300, 500]:
            for md in [4, 6, 8]:
                for lr in [0.05, 0.1]:
                    for ss in [0.8, 1.0]:
                        for cs in [0.8, 1.0]:
                            for rl in [1.0, 2.0]:
                                grid.append({'n_estimators': n, 'max_depth': md, 'learning_rate': lr, 'subsample': ss, 'colsample_bytree': cs, 'reg_lambda': rl})
        return grid
    return []


def apply_hyperparams(model, params: Dict[str, Any]):
    base = model.base_model if isinstance(model, EncodedModel) else model
    for k, v in params.items():
        setattr(base, k, v)


def select_features_by_importance(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, base_features: List[str], topk_list: List[int]) -> List[str]:
    # Fit a quick model and rank features
    classes = np.unique(y_train)
    model = build_model(model_name, classes, class_weight='balanced')
    model.fit(X_train, y_train)
    imp = model_importance(model, base_features)
    ranked = imp['feature'].tolist()
    # Try various top-k; return the best by a simple heuristic (sum of importances)
    best_k = None
    best_sum = -1
    for k in topk_list:
        s = imp['importance'][:k].sum()
        if s > best_sum:
            best_sum = s
            best_k = k
    return ranked[:best_k] if best_k else base_features


def main():
    ap = argparse.ArgumentParser(description='Optimize trading ML model with thresholds and backtest.')
    ap.add_argument('--csv', default='data/btcusdt_perp_5m_with_indicators.csv')
    ap.add_argument('--horizon', type=int, default=12)
    ap.add_argument('--threshold', type=float, default=0.0075)
    ap.add_argument('--models', nargs='+', default=['xgb','rf'])
    ap.add_argument('--imbalance', choices=['class_weight','undersample'], default='class_weight')
    ap.add_argument('--outdir', default='reports_opt')
    ap.add_argument('--modeldir', default='models_opt')
    ap.add_argument('--log', default='INFO')
    ap.add_argument('--cost', type=float, default=0.0012, help='Round-trip trading cost (fraction), e.g., 0.0012 = 0.12%')
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format='%(asctime)s | %(levelname)s | %(message)s')

    ensure_dirs(args.outdir, args.modeldir)

    logging.info('Loading %s', args.csv)
    df = load_data(args.csv)
    df_feat, base_features = engineer_features(df)
    df_feat['label'] = build_labels(df_feat, args.horizon, args.threshold)
    X_all, y_all = prepare_xy(df_feat, base_features)

    splits = walk_forward_splits(len(X_all), n_folds=5)
    logging.info('Prepared %d walk-forward windows', len(splits))

    best_overall = None
    best_artifacts = None

    for model_name in args.models:
        if model_name == 'xgb' and not HAS_XGB:
            logging.warning('Skipping xgb: not installed')
            continue

        # Feature selection (coarse): pick top-k from a base fit on first train window
        first_train_idx, _, _ = splits[0]
        feat_selected = select_features_by_importance(model_name, X_all.iloc[first_train_idx], y_all.iloc[first_train_idx], base_features, topk_list=[15,20,25,30])
        logging.info('Model %s selected %d features', model_name, len(feat_selected))

        # Hyperparameter tuning using the first two windows for speed
        grid = hyperparam_grid(model_name)
        logging.info('Hyperparameter candidates: %d', len(grid))

        # Evaluate a subset if grid is large
        max_try = 24
        if len(grid) > max_try:
            np.random.seed(42)
            grid = list(np.random.choice(grid, size=max_try, replace=False))

        best_local = None
        for gi, params in enumerate(grid):
            # aggregate validation trading score over first 2 windows
            agg_score = 0.0
            for wi, (tr, va, te) in enumerate(splits[:2]):
                X_tr, y_tr = X_all.iloc[tr][feat_selected], y_all.iloc[tr]
                X_va, y_va = X_all.iloc[va][feat_selected], y_all.iloc[va]
                X_te, y_te = X_all.iloc[te][feat_selected], y_all.iloc[te]
                # imbalance handling
                X_tr_res, y_tr_res, cw = imbalance_resample(X_tr, y_tr, method=args.imbalance, neutral_ratio=1.0)
                model = build_model(model_name, np.unique(y_tr_res), class_weight=cw)
                apply_hyperparams(model, params)
                # thresholds optimized inside evaluate_window
                prices_va = df_feat.loc[X_va.index, 'close']
                val_sum, _ = evaluate_window(model, X_tr_res, y_tr_res, X_va, y_va, X_te, y_te, prices_va, prices_va, args.horizon, cost=args.cost)
                score = trading_score(val_sum['metrics'], val_sum['precision_long'], val_sum['precision_short'])
                agg_score += score
            cand = (agg_score, params)
            if best_local is None or agg_score > best_local[0]:
                best_local = cand
        assert best_local is not None
        best_params = best_local[1]
        logging.info('Best params for %s: %s', model_name, best_params)

        # Full walk-forward with best params
        all_eq = []
        summaries = []
        thresholds = []
        for wi, (tr, va, te) in enumerate(splits):
            X_tr, y_tr = X_all.iloc[tr][feat_selected], y_all.iloc[tr]
            X_va, y_va = X_all.iloc[va][feat_selected], y_all.iloc[va]
            X_te, y_te = X_all.iloc[te][feat_selected], y_all.iloc[te]
            X_tr_res, y_tr_res, cw = imbalance_resample(X_tr, y_tr, method=args.imbalance, neutral_ratio=1.0)
            model = build_model(model_name, np.unique(y_tr_res), class_weight=cw)
            apply_hyperparams(model, best_params)
            prices_va = df_feat.loc[X_va.index, 'close']
            prices_te = df_feat.loc[X_te.index, 'close']
            val_sum, test_sum = evaluate_window(model, X_tr_res, y_tr_res, X_va, y_va, X_te, y_te, prices_va, prices_te, args.horizon, cost=args.cost)
            summaries.append({'fold': wi, 'val': val_sum, 'test': test_sum})
            thresholds.append({'fold': wi, 'th_long': val_sum['threshold_long'], 'th_short': val_sum['threshold_short'], 'persistence': val_sum['persistence']})
            all_eq += test_sum['metrics']['equity_curve']

        # Aggregate metrics over folds (using mean of tests)
        test_metrics = [s['test']['metrics'] for s in summaries]
        pnl_mean = float(np.mean([m['cum_return'] for m in test_metrics]))
        sharpe_mean = float(np.mean([m['sharpe'] for m in test_metrics]))
        mdd_mean = float(np.mean([m['max_drawdown'] for m in test_metrics]))
        pr_long_mean = float(np.mean([s['test']['precision_long'] for s in summaries]))
        pr_short_mean = float(np.mean([s['test']['precision_short'] for s in summaries]))
        score_total = pnl_mean + 0.1*(pr_long_mean + pr_short_mean) + 0.5*(-mdd_mean)

        model_result = {
            'model': model_name,
            'features': feat_selected,
            'best_params': best_params,
            'fold_summaries': summaries,
            'thresholds': thresholds,
            'agg': {
                'pnl_mean': pnl_mean,
                'sharpe_mean': sharpe_mean,
                'max_drawdown_mean': mdd_mean,
                'precision_long_mean': pr_long_mean,
                'precision_short_mean': pr_short_mean,
                'score': score_total,
            }
        }

        # Track best overall
        if best_overall is None or score_total > best_overall[0]:
            best_overall = (score_total, model_name)
            best_artifacts = model_result

        # Save per-model results
        with open(os.path.join(args.outdir, f'opt_result_{model_name}.json'), 'w') as f:
            json.dump(model_result, f, indent=2)

    assert best_artifacts is not None
    best_name = best_artifacts['model']
    logging.info('Best overall model: %s | score %.4f', best_name, best_overall[0])

    # Fit final model on all data (train+val folds combined per window is already evaluated; here fit on all X with best params and features)
    feat_selected = best_artifacts['features']
    params = best_artifacts['best_params']
    X_final, y_final = X_all[feat_selected], y_all
    X_final_res, y_final_res, cw = imbalance_resample(X_final, y_final, method=args.imbalance, neutral_ratio=1.0)
    final_model = build_model(best_name, np.unique(y_final_res), class_weight=cw)
    apply_hyperparams(final_model, params)
    final_model.fit(X_final_res, y_final_res)

    # Save final model and metadata including recommended thresholds (average across folds)
    th_l = float(np.mean([t['th_long'] for t in best_artifacts['thresholds']]))
    th_s = float(np.mean([t['th_short'] for t in best_artifacts['thresholds']]))
    pers = int(round(np.mean([t['persistence'] for t in best_artifacts['thresholds']])))

    joblib.dump({
        'model': final_model,
        'features': feat_selected,
        'horizon': args.horizon,
        'label_threshold': args.threshold,
        'prob_thresholds': {'long': th_l, 'short': th_s},
        'persistence': pers,
        'imbalance': args.imbalance,
        'trading_cost': args.cost,
        'source_csv': args.csv,
        'notes': 'Model optimized for trading PnL with threshold/persistence filtering.'
    }, os.path.join(args.modeldir, f'best_trading_model_{best_name}.joblib'))

    # Save overall summary
    with open(os.path.join(args.outdir, 'opt_summary.json'), 'w') as f:
        json.dump(best_artifacts, f, indent=2)

    print('Optimization complete. Best model:', best_name)
    print('Recommended prob thresholds:', {'long': th_l, 'short': th_s}, 'persistence:', pers)


if __name__ == '__main__':
    main()

