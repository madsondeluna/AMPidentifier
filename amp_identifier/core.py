# amp_identifier/core.py

import os
import joblib
import numpy as np
import pandas as pd
from . import data_io, feature_extraction, reporting

TUNED_DIR = "model_training/tuned_model"
DATA_DIR  = "model_training/data"

SEL_FEAT_PATH = os.path.join(DATA_DIR, "selected_features.txt")

# Scaler each individual model expects (VotingEnsemble handles scaling internally)
SCALER_MAP = {
    "rf":   "robust",
    "svm":  "std",
    "gb":   "robust",
    "xgb":  "robust",
    "lgbm": "robust",
}

THRESHOLD_DEFAULTS = {
    "rf":     0.56,
    "svm":    0.47,
    "gb":     0.55,
    "xgb":    0.48,
    "lgbm":   0.71,
    "voting": 0.56,
}


def _load_selected_features():
    if not os.path.exists(SEL_FEAT_PATH):
        return None
    with open(SEL_FEAT_PATH) as f:
        return [l.strip() for l in f if l.strip()]


def _load_threshold(model_type, override=None):
    if override is not None:
        return float(override)
    path = os.path.join(TUNED_DIR, f"threshold_{model_type}.txt")
    if os.path.exists(path):
        with open(path) as f:
            return float(f.read().strip())
    return THRESHOLD_DEFAULTS.get(model_type, 0.5)


def run_prediction_pipeline(
    input_file: str,
    output_dir: str,
    internal_model_type: str,
    use_ensemble: bool,
    threshold_override=None,
):
    print("\n" + "=" * 72)
    print("AMPidentifier 2.0 — prediction pipeline")
    print("=" * 72 + "\n")

    # Step 1: Load sequences
    print("Step 1/4: Loading sequences")
    sequences, seq_ids = data_io.load_fasta_sequences(input_file)
    if not sequences:
        print("No sequences loaded. Exiting.")
        return
    print(f"  {len(sequences)} sequence(s) found\n")

    # Step 2: Extract features
    print("Step 2/4: Extracting features")
    features_df = feature_extraction.calculate_physicochemical_features(sequences, seq_ids)
    features_df.fillna(0, inplace=True)

    selected = _load_selected_features()
    if selected:
        missing = [f for f in selected if f not in features_df.columns]
        if missing:
            print(f"  Warning: {len(missing)} selected features absent from feature matrix.")
        X_raw = features_df[[f for f in selected if f in features_df.columns]].fillna(0)
    else:
        meta_cols = {"ID", "sequence", "label"}
        X_raw = features_df[[c for c in features_df.columns if c not in meta_cols]].fillna(0)

    feat_path = os.path.join(output_dir, "physicochemical_features.csv")
    reporting.save_features_report(features_df, feat_path)
    print(f"  Features saved to {feat_path}\n")

    # Step 3: Run predictions
    print("Step 3/4: Running predictions")
    model_type = internal_model_type.lower()

    model_path = os.path.join(TUNED_DIR, f"amp_model_{model_type}_tuned.pkl")
    if not os.path.exists(model_path):
        print(f"  Model file not found: {model_path}")
        print("  Run `python3 -m model_training.tune` to generate tuned models.")
        return

    model = joblib.load(model_path)
    print(f"  Loaded: {model_path}")

    if model_type == "voting":
        # VotingEnsemble stores its own scalers
        proba = model.predict_proba(X_raw)[:, 1]
    else:
        # Individual model: apply the appropriate scaler from tuned_model/
        scaler_key  = SCALER_MAP[model_type]
        scaler_path = os.path.join(TUNED_DIR, f"scaler_{scaler_key}.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_sc   = pd.DataFrame(
                scaler.transform(X_raw),
                columns=X_raw.columns, index=X_raw.index,
            )
        else:
            print(f"  Warning: scaler not found at {scaler_path}. Using unscaled features.")
            X_sc = X_raw
        proba = model.predict_proba(X_sc)[:, 1]

    threshold = _load_threshold(model_type, threshold_override)
    predictions = (proba >= threshold).astype(int)

    results_df = pd.DataFrame({
        "ID":             features_df["ID"],
        "sequence":       features_df["sequence"],
        "probability_AMP": np.round(proba, 4),
        "prediction":     predictions,
        "label":          predictions.astype(str),
    })
    results_df["label"] = results_df["prediction"].map({1: "AMP", 0: "non-AMP"})

    pred_path = os.path.join(output_dir, f"predictions_{model_type}.csv")
    results_df.to_csv(pred_path, index=False)
    print(f"  Predictions saved to {pred_path}\n")

    # Step 4: Summary
    print("Step 4/4: Summary")
    n_total   = len(results_df)
    n_amp     = int(predictions.sum())
    n_non_amp = n_total - n_amp
    pct_amp   = n_amp / n_total * 100 if n_total > 0 else 0.0

    bar_len  = 40
    amp_bars = int(pct_amp * bar_len / 100)
    non_bars = bar_len - amp_bars

    print()
    print("=" * 72)
    print("PREDICTION SUMMARY")
    print("=" * 72)
    print(f"  Model          : {model_type.upper()}")
    print(f"  Threshold      : {threshold:.2f}")
    print(f"  Total sequences: {n_total}")
    print()
    print(f"  AMP detected   : {n_amp:4d} ({pct_amp:5.1f}%)  [{'|' * amp_bars}{' ' * non_bars}]")
    print(f"  Non-AMP        : {n_non_amp:4d} ({100 - pct_amp:5.1f}%)  [{' ' * amp_bars}{'|' * non_bars}]")
    print()
    print(f"  Output         : {pred_path}")
    print("=" * 72)
    print()
    print("  Citation:")
    print("  Luna-Aragão, M.A. et al. (2025). AMPidentifier 2.0.")
    print("  Journal of Chemical Information and Modeling.")
    print("=" * 72 + "\n")
