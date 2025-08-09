#!/usr/bin/env python3
"""
Train ML models to predict LONG / SHORT / NEUTRAL signals on BTCUSDT 5m data.

- Loads data from ./data/btcusdt_perp_5m_with_indicators.csv by default
- Engineers features (price/volume/indicator-derived)
- Defines labels based on forward N-period horizon with X% thresholds
- Trains multiple models (Logistic Regression, Random Forest, optional XGBoost)
- Evaluates with time-based splits and a simple walk-forward validation
- Saves the best model and reports

Example:
    python train_btcusdt_signals.py --csv data/btcusdt_perp_5m_with_indicators.csv \
        --horizon 12 --threshold 0.0075 --models lr rf xgb --log INFO

Notes:
- No look-ahead bias: features are shifted by 1 bar; labels use future N bars
- Classes: 1=LONG, -1=SHORT, 0=NEUTRAL
"""
from __future__ import annotations

import os
import json
import argparse
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False


@dataclass
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class EncodedModel:
    """Wrap a classifier to encode labels to 0..K-1 for training (e.g., XGBoost),
    but expose original labels for predict()."""
    def __init__(self, base_model: Any, label_to_idx: Dict[int, int]):
        self.base_model = base_model
        self.label_to_idx = {int(k): int(v) for k, v in label_to_idx.items()}
        # Build reverse map
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def fit(self, X, y):
        y_enc = np.array([self.label_to_idx[int(v)] for v in y], dtype=int)
        self.base_model.fit(X, y_enc)
        return self

    def predict(self, X):
        y_enc = self.base_model.predict(X)
        # Some models may return float; cast to int
        y_enc = np.asarray(y_enc).astype(int)
        return np.array([self.idx_to_label[int(v)] for v in y_enc], dtype=int)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    def __getattr__(self, name):
        # Delegate unknown attributes to base_model (e.g., feature_importances_)
        return getattr(self.base_model, name)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure datetime
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Basic derived
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)

    # Volatility proxies
    df["vol_12"] = df["ret_1"].rolling(12).std()
    df["vol_24"] = df["ret_1"].rolling(24).std()

    # Candle shapes
    df["body"] = (df["close"] - df["open"]) / df["open"]
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]

    # Indicator relationships
    df["close_to_ema"] = (df["close"] - df["ema_55"]) / df["ema_55"]
    df["band_width"] = (df["kc_upper"] - df["kc_lower"]) / df["ema_55"]
    df["band_pos"] = (df["close"] - df["kc_lower"]) / (df["kc_upper"] - df["kc_lower"])
    df["z_to_ema"] = (df["close"] - df["ema_55"]) / df["std_55"]

    # RSI transforms
    df["rsi"] = df["rsi_14"]
    df["rsi_norm"] = (df["rsi"] - 50.0) / 50.0
    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
    df["rsi_oversold"] = (df["rsi"] < 30).astype(int)

    # Volume signals
    df["vol_z24"] = (df["volume"] - df["volume"].rolling(24).mean()) / df["volume"].rolling(24).std()
    df["qvol_z24"] = (df["quote_volume"] - df["quote_volume"].rolling(24).mean()) / df["quote_volume"].rolling(24).std()

    # Shift all features by 1 to avoid using info from the decision bar's close
    feature_cols = [
        "open", "high", "low", "close", "volume", "quote_volume",
        "ema_55", "kc_upper", "kc_lower", "rsi_14", "std_55",
        "ret_1", "ret_3", "ret_6", "ret_12",
        "vol_12", "vol_24",
        "body", "range_pct", "upper_wick", "lower_wick",
        "close_to_ema", "band_width", "band_pos", "z_to_ema",
        "rsi", "rsi_norm", "rsi_overbought", "rsi_oversold",
        "vol_z24", "qvol_z24",
    ]
    df[feature_cols] = df[feature_cols].shift(1)

    return df


def build_labels(df: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    """Return label series with values {-1, 0, 1} for SHORT, NEUTRAL, LONG.

    LONG if max future return >= threshold within next `horizon` bars.
    SHORT if min future return <= -threshold within next `horizon` bars.
    If both conditions are true, label 0 (neutral) to avoid hindsight conflict.
    """
    close = df["close"].astype(float)

    # Future windows using shifts (horizon typically small: 6-24)
    fut = [close.shift(-k) for k in range(1, horizon + 1)]
    fut_mat = pd.concat(fut, axis=1)
    fut_max = fut_mat.max(axis=1)
    fut_min = fut_mat.min(axis=1)

    up_ret = (fut_max - close) / close
    dn_ret = (fut_min - close) / close

    long_sig = up_ret >= threshold
    short_sig = dn_ret <= -threshold

    labels = pd.Series(0, index=df.index, dtype=int)
    labels[long_sig & ~short_sig] = 1
    labels[short_sig & ~long_sig] = -1
    # conflicts remain 0

    # Last `horizon` rows have NaN labels; set to NaN to drop later
    labels.iloc[-horizon:] = np.nan

    return labels


def time_split_indices(n: int, train_frac=0.7, val_frac=0.15) -> SplitIndices:
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)
    return SplitIndices(train_idx, val_idx, test_idx)


def walk_forward_splits(n: int, n_folds: int = 3, train_size_frac: float = 0.6, val_size_frac: float = 0.1, test_size_frac: float = 0.1) -> List[SplitIndices]:
    splits: List[SplitIndices] = []
    min_total_frac = train_size_frac + val_size_frac + test_size_frac
    if min_total_frac > 0.95:
        raise ValueError("Sum of train/val/test fractions for walk-forward should be <= 0.95")

    window = int(n * (train_size_frac + val_size_frac + test_size_frac))
    step = int(window * 0.5)  # 50% overlap

    start = 0
    for _ in range(n_folds):
        end = start + window
        if end + int(n * 0.05) > n:  # leave tail room
            break
        train_end = start + int(window * train_size_frac / (train_size_frac + val_size_frac + test_size_frac))
        val_end = train_end + int(window * val_size_frac / (train_size_frac + val_size_frac + test_size_frac))
        train_idx = np.arange(start, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, end)
        splits.append(SplitIndices(train_idx, val_idx, test_idx))
        start += step
    return splits


def prepare_xy(df: pd.DataFrame, feature_cols: List[str], label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].copy()
    y = df[label_col].copy()

    # Replace inf with NaN before dropping
    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove rows with any NaN in X or y
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask].astype(int)

    return X, y


def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series, splits: SplitIndices, model_names: List[str], outdir: str) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Train models on train, select on val by macro F1, evaluate on test. Save reports.
    Returns best_model_name and metrics dict per model.
    """
    results: Dict[str, Dict[str, float]] = {}

    classes = np.unique(y)
    # Compute class weights to mitigate imbalance
    class_weight_vals = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight = {int(c): w for c, w in zip(classes, class_weight_vals)}

    # Train/Val/Test splits
    X_train, y_train = X.iloc[splits.train_idx], y.iloc[splits.train_idx]
    X_val, y_val = X.iloc[splits.val_idx], y.iloc[splits.val_idx]
    X_test, y_test = X.iloc[splits.test_idx], y.iloc[splits.test_idx]

    fitted_models: Dict[str, object] = {}

    for name in model_names:
        if name == 'lr':
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight=class_weight, n_jobs=None)),
            ])
            model = pipe
        elif name == 'rf':
            model = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=5, n_jobs=-1, class_weight=class_weight, random_state=42)
        elif name == 'xgb':
            if not HAS_XGB:
                logging.warning("XGBoost not installed; skipping xgb model.")
                continue
            # Map labels to 0..K-1 for XGBoost
            sorted_classes = sorted(int(c) for c in classes)
            label_to_idx = {c: i for i, c in enumerate(sorted_classes)}
            base = XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective='multi:softprob',
                num_class=len(sorted_classes),
                tree_method='hist',
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=42,
            )
            model = EncodedModel(base, label_to_idx)
        else:
            logging.warning("Unknown model name: %s", name)
            continue

        # Fit on train
        model.fit(X_train, y_train)
        # Validate
        y_val_pred = model.predict(X_val)
        f1_macro_val = f1_score(y_val, y_val_pred, average='macro')
        results[name] = {"f1_macro_val": float(f1_macro_val)}
        fitted_models[name] = model

        # Save classification report on validation
        report = classification_report(y_val, y_val_pred, digits=3)
        with open(os.path.join(outdir, f"val_report_{name}.txt"), 'w') as f:
            f.write(report)

    if not results:
        raise RuntimeError("No models were trained. Ensure at least one of lr, rf, xgb is selected and dependencies installed.")

    # Select best by validation macro F1
    best_model_name = max(results.items(), key=lambda kv: kv[1]["f1_macro_val"])[0]
    best_model = fitted_models[best_model_name]

    # Test evaluation
    y_test_pred = best_model.predict(X_test)
    f1_macro_test = f1_score(y_test, y_test_pred, average='macro')
    prec, rec, f1_per_class, support = precision_recall_fscore_support(y_test, y_test_pred, labels=[-1,0,1], zero_division=0)

    results[best_model_name].update({
        "f1_macro_test": float(f1_macro_test),
        "precision_-1": float(prec[0]),
        "recall_-1": float(rec[0]),
        "f1_-1": float(f1_per_class[0]),
        "precision_0": float(prec[1]),
        "recall_0": float(rec[1]),
        "f1_0": float(f1_per_class[1]),
        "precision_1": float(prec[2]),
        "recall_1": float(rec[2]),
        "f1_1": float(f1_per_class[2]),
    })

    # Save test report and confusion matrix
    report_test = classification_report(y_test, y_test_pred, labels=[-1,0,1], digits=3)
    with open(os.path.join(outdir, f"test_report_{best_model_name}.txt"), 'w') as f:
        f.write(report_test)
    cm = confusion_matrix(y_test, y_test_pred, labels=[-1,0,1])
    np.savetxt(os.path.join(outdir, f"test_confusion_matrix_{best_model_name}.csv"), cm, delimiter=',', fmt='%d')

    # Feature importances
    feat_importances = feature_importance(best_model, X.columns.tolist())
    feat_imp_path = os.path.join(outdir, f"feature_importance_{best_model_name}.csv")
    feat_importances.to_csv(feat_imp_path, index=False)

    return best_model_name, results


def feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    try:
        # Pipeline handling for LR
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            clf = model.named_steps.get('clf')
        else:
            clf = model

        if hasattr(clf, 'feature_importances_'):
            vals = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            coef = clf.coef_
            if coef.ndim == 1:
                vals = np.abs(coef)
            else:
                vals = np.mean(np.abs(coef), axis=0)
        else:
            # Fallback: permutation importance could be added; use zeroes
            vals = np.zeros(len(feature_names))

        return pd.DataFrame({"feature": feature_names, "importance": vals}).sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception as e:
        logging.warning("Failed to compute feature importance: %s", e)
        return pd.DataFrame({"feature": feature_names, "importance": np.nan})


def do_walk_forward(X: pd.DataFrame, y: pd.Series, model_name: str, outdir: str) -> pd.DataFrame:
    splits_list = walk_forward_splits(len(X), n_folds=3)

    rows = []
    for i, sp in enumerate(splits_list):
        # Use same training/eval procedure per fold but single model
        if model_name == 'lr':
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight='balanced')),
            ])
        elif model_name == 'rf':
            model = RandomForestClassifier(n_estimators=300, min_samples_split=5, n_jobs=-1, class_weight='balanced_subsample', random_state=42)
        elif model_name == 'xgb' and HAS_XGB:
            # determine classes in this fold and wrap with encoder
            X_train, y_train = X.iloc[sp.train_idx], y.iloc[sp.train_idx]
            sorted_classes = sorted(int(c) for c in np.unique(y_train))
            label_to_idx = {c: i for i, c in enumerate(sorted_classes)}
            base = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                objective='multi:softprob', num_class=len(sorted_classes), tree_method='hist', reg_lambda=1.0, n_jobs=-1, random_state=42
            )
            model = EncodedModel(base, label_to_idx)
        else:
            continue

        # For non-xgb, define train/test here
        if not (model_name == 'xgb' and HAS_XGB):
            X_train, y_train = X.iloc[sp.train_idx], y.iloc[sp.train_idx]
        X_test, y_test = X.iloc[sp.test_idx], y.iloc[sp.test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1m = f1_score(y_test, y_pred, average='macro')
        pr, rc, f1c, _ = precision_recall_fscore_support(y_test, y_pred, labels=[-1,0,1], zero_division=0)
        rows.append({
            "fold": i,
            "f1_macro": float(f1m),
            "precision_-1": float(pr[0]), "recall_-1": float(rc[0]), "f1_-1": float(f1c[0]),
            "precision_0": float(pr[1]), "recall_0": float(rc[1]), "f1_0": float(f1c[1]),
            "precision_1": float(pr[2]), "recall_1": float(rc[2]), "f1_1": float(f1c[2]),
        })

    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        df_res.to_csv(os.path.join(outdir, f"walk_forward_{model_name}.csv"), index=False)
    return df_res


def main():
    parser = argparse.ArgumentParser(description="Train ML models for BTCUSDT long/short signals.")
    parser.add_argument('--csv', default='data/btcusdt_perp_5m_with_indicators.csv', help='Path to input CSV with indicators')
    parser.add_argument('--horizon', type=int, default=12, help='Forward horizon in bars (5m bars). 12=1 hour.')
    parser.add_argument('--threshold', type=float, default=0.0075, help='Return threshold for signal (e.g., 0.0075=0.75%)')
    parser.add_argument('--models', nargs='+', default=['lr','rf','xgb'], help='Models to train: lr rf xgb')
    parser.add_argument('--outdir', default='reports', help='Directory to save reports')
    parser.add_argument('--modeldir', default='models', help='Directory to save trained models')
    parser.add_argument('--log', default='INFO', help='Logging level')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format='%(asctime)s | %(levelname)s | %(message)s')

    ensure_dirs(args.outdir, args.modeldir)

    logging.info("Loading data from %s", args.csv)
    df = load_data(args.csv)

    logging.info("Engineering features...")
    df_feat = engineer_features(df)

    logging.info("Building labels (horizon=%d, threshold=%.4f)...", args.horizon, args.threshold)
    df_feat['label'] = build_labels(df_feat, args.horizon, args.threshold)

    # Feature set
    feature_cols = [
        "open", "high", "low", "close", "volume", "quote_volume",
        "ema_55", "kc_upper", "kc_lower", "rsi_14", "std_55",
        "ret_1", "ret_3", "ret_6", "ret_12",
        "vol_12", "vol_24",
        "body", "range_pct", "upper_wick", "lower_wick",
        "close_to_ema", "band_width", "band_pos", "z_to_ema",
        "rsi", "rsi_norm", "rsi_overbought", "rsi_oversold",
        "vol_z24", "qvol_z24",
    ]

    X, y = prepare_xy(df_feat, feature_cols, 'label')

    logging.info("Data after cleaning: %d rows, %d features; class distribution: %s", len(X), X.shape[1], y.value_counts().to_dict())

    # Time-based split
    splits = time_split_indices(len(X), train_frac=0.7, val_frac=0.15)

    logging.info("Training models: %s", args.models)
    best_model_name, results = train_and_evaluate_models(X, y, splits, args.models, args.outdir)

    logging.info("Best model: %s | Metrics: %s", best_model_name, results.get(best_model_name))

    # Save best model
    # Refit on train+val for final model
    trainval_idx = np.concatenate([splits.train_idx, splits.val_idx])
    X_trainval, y_trainval = X.iloc[trainval_idx], y.iloc[trainval_idx]

    if best_model_name == 'lr':
        final_model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight='balanced')),
        ])
    elif best_model_name == 'rf':
        final_model = RandomForestClassifier(n_estimators=400, min_samples_split=5, n_jobs=-1, class_weight='balanced_subsample', random_state=42)
    elif best_model_name == 'xgb' and HAS_XGB:
        # Wrap with label encoder based on classes in train+val
        sorted_classes = sorted(int(c) for c in np.unique(y_trainval))
        label_to_idx = {c: i for i, c in enumerate(sorted_classes)}
        base = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            objective='multi:softprob', num_class=len(sorted_classes), tree_method='hist', reg_lambda=1.0, n_jobs=-1, random_state=42
        )
        final_model = EncodedModel(base, label_to_idx)
    else:
        raise RuntimeError("Unexpected best model selection.")

    final_model.fit(X_trainval, y_trainval)
    model_path = os.path.join(args.modeldir, f"best_model_{best_model_name}.joblib")
    joblib.dump({
        'model': final_model,
        'features': feature_cols,
        'horizon': args.horizon,
        'threshold': args.threshold,
        'metadata': {
            'csv': args.csv,
            'rows_used': len(X),
        }
    }, model_path)
    logging.info("Saved best model to %s", model_path)

    # Walk-forward evaluation for the best model
    logging.info("Running walk-forward evaluation for the best model: %s", best_model_name)
    wf_df = do_walk_forward(X, y, best_model_name, args.outdir)
    if not wf_df.empty:
        logging.info("Walk-forward summary (macro F1 mean=%.3f, std=%.3f)", wf_df['f1_macro'].mean(), wf_df['f1_macro'].std())

    # Save summary JSON
    summary = {
        'best_model': best_model_name,
        'metrics': results.get(best_model_name, {}),
        'walk_forward': wf_df.to_dict(orient='records') if not wf_df.empty else [],
    }
    with open(os.path.join(args.outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("Training complete. Best model:", best_model_name)


if __name__ == '__main__':
    main()

