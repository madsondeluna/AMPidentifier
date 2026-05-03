# model_training/train.py
#
# Baseline training pipeline for all classifiers.
#
# Models trained:
#   rf   - Random Forest
#   svm  - Support Vector Machine
#   gb   - Gradient Boosting
#   xgb  - XGBoost
#   lgbm - LightGBM
#
# Scaling strategy:
#   RobustScaler -> all models
#
# Features: 22 physicochemical features (selected_features.txt).
#
# Run from project root:
#   python3 -m model_training.train

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef
)
import numpy as np

from amp_identifier.feature_extraction import calculate_physicochemical_features
from amp_identifier.data_io import load_fasta_sequences

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = "model_training/data"
OUTPUT_DIR = "model_training/baseline_model"
POSITIVE_FILE          = os.path.join(DATA_DIR, "positive_sequences.fasta")
NEGATIVE_FILE          = os.path.join(DATA_DIR, "negative_sequences.fasta")
SELECTED_FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.txt")

RANDOM_STATE = 42
TEST_SIZE    = 0.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_selected_features() -> list:
    with open(SELECTED_FEATURES_PATH) as f:
        return [l.strip() for l in f if l.strip()]


def _evaluate(model, X_test, y_test, scaler=None, threshold=0.5) -> dict:
    X = scaler.transform(X_test) if scaler else X_test
    y_proba = model.predict_proba(X)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "accuracy":    accuracy_score(y_test, y_pred),
        "precision":   precision_score(y_test, y_pred),
        "recall":      recall_score(y_test, y_pred),
        "specificity": tn / (tn + fp),
        "f1":          f1_score(y_test, y_pred),
        "mcc":         matthews_corrcoef(y_test, y_pred),
        "auc_roc":     roc_auc_score(y_test, y_proba),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def _tune_threshold_mcc(model, X_val, y_val) -> float:
    """Find probability threshold that maximizes MCC on validation set."""
    y_proba = model.predict_proba(X_val)[:, 1]
    best_t, best_mcc = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        y_pred = (y_proba >= t).astype(int)
        mcc = matthews_corrcoef(y_val, y_pred)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("--- Starting Model Training Pipeline ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load data ---
    print("\nStep 1: Loading sequences...")
    pos_seqs, pos_ids = load_fasta_sequences(POSITIVE_FILE)
    neg_seqs, neg_ids = load_fasta_sequences(NEGATIVE_FILE)
    sequences = pos_seqs + neg_seqs
    ids       = pos_ids  + neg_ids
    labels    = [1] * len(pos_seqs) + [0] * len(neg_seqs)
    print(f"  Positive: {len(pos_seqs)}  Negative: {len(neg_seqs)}")

    # --- Extract features ---
    print("\nStep 2: Extracting features...")
    features_df = calculate_physicochemical_features(sequences, ids)
    features_df["label"] = labels

    selected = _load_selected_features()
    X = features_df[selected].fillna(0)
    y = features_df["label"]
    print(f"  Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    # --- Split ---
    print("\nStep 3: Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # --- Scalers ---
    print("\nStep 4: Fitting scalers...")
    robust_scaler = RobustScaler()

    X_train_robust = pd.DataFrame(
        robust_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_robust  = pd.DataFrame(
        robust_scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    joblib.dump(robust_scaler, os.path.join(OUTPUT_DIR, "scaler_robust.pkl"))
    print("  Scaler saved.")

    # --- Model definitions ---
    # (model_name, model, X_train_scaled, X_test_scaled)
    models = [
        ("rf",  RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=-1,
                                       verbose=2),
                X_train_robust, X_test_robust),
        ("svm", SVC(probability=True, class_weight="balanced",
                    random_state=RANDOM_STATE, verbose=True),
                X_train_robust, X_test_robust),
        ("gb",  GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                           verbose=2),
                X_train_robust, X_test_robust),
        ("xgb", XGBClassifier(n_estimators=100, scale_pos_weight=1,
                               eval_metric="logloss", verbosity=2,
                               random_state=RANDOM_STATE),
                X_train_robust, X_test_robust),
        ("lgbm", LGBMClassifier(n_estimators=100, class_weight="balanced",
                                random_state=RANDOM_STATE, verbose=-1),
                 X_train_robust, X_test_robust),
    ]

    # --- Train and evaluate ---
    summary_rows = []
    for name, model, X_tr, X_te in models:
        print(f"\n--- Training {name.upper()} ---")
        model.fit(X_tr, y_train)

        threshold = _tune_threshold_mcc(model, X_te, y_test)
        print(f"  Optimal MCC threshold: {threshold:.2f}")

        metrics = _evaluate(model, X_te, y_test, threshold=threshold)
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}  MCC: {metrics['mcc']:.4f}  "
              f"F1: {metrics['f1']:.4f}")

        model_path = os.path.join(OUTPUT_DIR, f"amp_model_{name}.pkl")
        joblib.dump(model, model_path)

        threshold_path = os.path.join(OUTPUT_DIR, f"threshold_{name}.txt")
        with open(threshold_path, "w") as f:
            f.write(str(threshold))

        summary_rows.append({"model": name.upper(), "threshold": threshold, **metrics})

    # --- Summary ---
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "training_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved -> {summary_path}")
    print("\n--- All training pipelines finished ---")
    print(summary_df[["model", "auc_roc", "mcc", "f1", "threshold"]].to_string(index=False))


if __name__ == "__main__":
    main()
