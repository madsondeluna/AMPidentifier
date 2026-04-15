# amp_identifier/core.py

import os
import sys
import joblib
import numpy as np
import pandas as pd
from . import data_io, feature_extraction, reporting

TUNED_DIR = "model_training/tuned_model"
DATA_DIR  = "model_training/data"

SEL_FEAT_PATH = os.path.join(DATA_DIR, "selected_features.txt")

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

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

def _bold(t):   return _c("1",  t)
def _dim(t):    return _c("2",  t)
def _cyan(t):   return _c("36", t)
def _green(t):  return _c("32", t)
def _yellow(t): return _c("33", t)
def _red(t):    return _c("31", t)
def _gray(t):   return _c("90", t)

# ---------------------------------------------------------------------------
# Box layout constants  (all widths are visible character counts)
# ---------------------------------------------------------------------------
#
#   ╔══════════════════════════════════════════════════════════════════════╗
#   ║  <label:18><value:50>                                               ║
#   ╠══════════════════════════════════════════════════════════════════════╣
#   ║  <label:9><count:6>  (<pct:5>%)  <bar:BAR_W><pad:7>                ║
#   ╚══════════════════════════════════════════════════════════════════════╝
#
#   BOX_W  = 72   (total line width including ║ on each side)
#   INNER  = 70   (characters between ║║)
#   BAR_W  = 32   (visible block characters in the progress bar)
#   BAR_ROW_FIXED = 2+9+6+3+5+3+2+BAR_W = 62  =>  trailing pad = 70-62 = 8

BOX_W  = 72
INNER  = BOX_W - 2
BAR_W  = 32
_BAR_ROW_FIXED = 2 + 9 + 6 + 3 + 5 + 3 + 2 + BAR_W   # = 62
_BAR_TRAIL     = INNER - _BAR_ROW_FIXED                # = 8


def _bar(ratio: float) -> str:
    filled = round(max(0.0, min(1.0, ratio)) * BAR_W)
    return _green("\u2588" * filled) + _gray("\u2591" * (BAR_W - filled))


def _box_top():  print("\u2554" + "\u2550" * INNER + "\u2557")
def _box_div():  print("\u2560" + "\u2550" * INNER + "\u2563")
def _box_bot():  print("\u255a" + "\u2550" * INNER + "\u255d")


def _box_info(label: str, value: str):
    # visible layout: "  {label:<18}{value:<50}"  = 2+18+50 = 70 = INNER
    val_w = INNER - 20            # 50
    if len(value) > val_w:
        value = value[:val_w - 3] + "..."
    # Pad raw strings first, then colorize — ANSI codes must not enter f-string width specs
    row = "  " + _bold(f"{label:<18}") + f"{value:<{val_w}}"
    print(f"\u2551{row}\u2551")


def _box_bar(label: str, count: int, pct: float):
    # visible layout: "  {label:<9}{count:>6}  ({pct:5.1f}%)  {bar}{pad}"
    # = 2+9+6+3+5+3+2+BAR_W+_BAR_TRAIL = INNER
    left = f"  {label:<9}{count:>6}  ({pct:5.1f}%)  "
    bar  = _bar(pct / 100.0)
    pad  = " " * _BAR_TRAIL
    print(f"\u2551{left}{bar}{pad}\u2551")


def _box_dim(text: str):
    # dim text row: "  {text:<INNER-2}"
    inner_text_w = INNER - 2
    raw = f"  {text:<{inner_text_w}}"
    print(f"\u2551{_dim(raw)}\u2551")


# ---------------------------------------------------------------------------
# Step / log helpers
# ---------------------------------------------------------------------------

def _step(n: int, total: int, label: str):
    tag = _bold(_cyan(f"[{n}/{total}]"))
    print(f"  {tag}  {label}")

def _ok(msg: str):   print(f"        {_gray(msg)}")
def _warn(msg: str): print(f"        {_yellow('warning:')} {msg}")
def _err(msg: str):  print(f"        {_red('error:')} {msg}")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_prediction_pipeline(
    input_file: str,
    output_dir: str,
    internal_model_type: str,
    use_ensemble: bool,
    threshold_override=None,
):
    print()
    print(f"  {_gray('-' * (BOX_W - 4))}")
    print()

    # Step 1: Load sequences
    _step(1, 4, "Loading sequences")
    sequences, seq_ids = data_io.load_fasta_sequences(input_file)
    if not sequences:
        _err("no sequences loaded. Exiting.")
        return
    _ok(f"{len(sequences)} sequence(s) found")
    print()

    # Step 2: Extract features
    _step(2, 4, "Extracting features")
    features_df = feature_extraction.calculate_physicochemical_features(sequences, seq_ids)
    features_df.fillna(0, inplace=True)

    selected = _load_selected_features()
    if selected:
        missing = [f for f in selected if f not in features_df.columns]
        if missing:
            _warn(f"{len(missing)} selected feature(s) absent from feature matrix")
        X_raw = features_df[[f for f in selected if f in features_df.columns]].fillna(0)
    else:
        meta_cols = {"ID", "sequence", "label"}
        X_raw = features_df[[c for c in features_df.columns if c not in meta_cols]].fillna(0)

    feat_path = os.path.join(output_dir, "physicochemical_features.csv")
    reporting.save_features_report(features_df, feat_path)
    _ok(f"{X_raw.shape[1]} features x {X_raw.shape[0]} sequences")
    _ok(f"saved -> {feat_path}")
    print()

    # Step 3: Run predictions
    model_type = internal_model_type.lower()
    threshold  = _load_threshold(model_type, threshold_override)

    _step(3, 4, f"Running predictions  {_dim(f'model={model_type.upper()}  threshold={threshold:.2f}')}")

    model_path = os.path.join(TUNED_DIR, f"amp_model_{model_type}_tuned.pkl")
    if not os.path.exists(model_path):
        _err(f"model not found: {model_path}")
        _err("run  python3 -m model_training.tune  to generate tuned models")
        return

    model = joblib.load(model_path)
    _ok(f"loaded {model_path}")

    if model_type == "voting":
        proba = model.predict_proba(X_raw)[:, 1]
    else:
        scaler_key  = SCALER_MAP[model_type]
        scaler_path = os.path.join(TUNED_DIR, f"scaler_{scaler_key}.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_sc   = pd.DataFrame(
                scaler.transform(X_raw),
                columns=X_raw.columns, index=X_raw.index,
            )
        else:
            _warn(f"scaler not found at {scaler_path} — using unscaled features")
            X_sc = X_raw
        proba = model.predict_proba(X_sc)[:, 1]

    predictions = (proba >= threshold).astype(int)

    results_df = pd.DataFrame({
        "ID":              features_df["ID"],
        "sequence":        features_df["sequence"],
        "probability_AMP": np.round(proba, 4),
        "prediction":      predictions,
        "label":           predictions.astype(str),
    })
    results_df["label"] = results_df["prediction"].map({1: "AMP", 0: "non-AMP"})

    pred_path = os.path.join(output_dir, f"predictions_{model_type}.csv")
    results_df.to_csv(pred_path, index=False)
    _ok(f"saved -> {pred_path}")
    print()

    # Step 4: Summary
    _step(4, 4, "Summary")
    print()

    n_total   = len(results_df)
    n_amp     = int(predictions.sum())
    n_non_amp = n_total - n_amp
    pct_amp   = n_amp     / n_total * 100 if n_total > 0 else 0.0
    pct_non   = n_non_amp / n_total * 100 if n_total > 0 else 0.0

    _box_top()
    _box_info("Model",            model_type.upper())
    _box_info("Threshold",        f"{threshold:.2f}")
    _box_info("Total sequences",  str(n_total))
    _box_div()
    _box_bar("AMP",     n_amp,     pct_amp)
    _box_bar("non-AMP", n_non_amp, pct_non)
    _box_div()
    _box_info("Output", pred_path)
    _box_div()
    _box_dim("Luna-Aragao, M.A. et al. (2025). AMPidentifier 2.0.")
    _box_dim("Journal of Chemical Information and Modeling.")
    _box_bot()
    print()
