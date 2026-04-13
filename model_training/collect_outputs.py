# model_training/collect_outputs.py
#
# Evaluates a single model in an isolated process and saves outputs to disk.
# Called by plot_tuning.py via subprocess — one call per model.
#
# Usage:
#   python -m model_training.collect_outputs <model_name>
#   model_name: rf | svm | gb | xgb | mlp | stack | deep
#
# Outputs saved to model_training/tuned_model/outputs/<name>_outputs.npz:
#   proba     - predicted probabilities on test set (float32)
#   y_test    - true labels (int8)
#   threshold - MCC-optimal threshold (scalar)
#   importance - feature importances (float32, tree models only; zeros otherwise)

import os
import sys
import gc
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from amp_identifier.feature_extraction import calculate_physicochemical_features
from amp_identifier.data_io import load_fasta_sequences

DATA_DIR      = "model_training/data"
TUNED_DIR     = "model_training/tuned_model"
OUT_DIR       = os.path.join(TUNED_DIR, "outputs")
POSITIVE_FILE = os.path.join(DATA_DIR, "positive_sequences.fasta")
NEGATIVE_FILE = os.path.join(DATA_DIR, "negative_sequences.fasta")
SEL_FEAT_PATH = os.path.join(DATA_DIR, "selected_features.txt")
RANDOM_STATE  = 42
TEST_SIZE     = 0.20

SCALER_MAP = {
    "rf":    "robust",
    "svm":   "std",
    "gb":    "robust",
    "xgb":   "robust",
    "mlp":   "qt",
    "stack": "robust",
}
TREE_MODELS = {"rf", "gb", "xgb"}

AA_VOCAB = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
MAX_LEN  = 200


def _load_features_and_split():
    pos_seqs, pos_ids = load_fasta_sequences(POSITIVE_FILE)
    neg_seqs, neg_ids = load_fasta_sequences(NEGATIVE_FILE)
    sequences = pos_seqs + neg_seqs
    ids       = pos_ids + neg_ids
    labels    = [1] * len(pos_seqs) + [0] * len(neg_seqs)

    features_df = calculate_physicochemical_features(sequences, ids)
    features_df["label"] = labels

    with open(SEL_FEAT_PATH) as f:
        selected = [l.strip() for l in f if l.strip()]

    X = features_df[selected].fillna(0)
    y = np.array(features_df["label"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, sequences, labels


def _scale(X_train, X_test, scaler_key):
    if scaler_key == "robust":
        scaler = RobustScaler()
    elif scaler_key == "qt":
        scaler = QuantileTransformer(output_distribution="normal",
                                     random_state=RANDOM_STATE)
    else:
        scaler = StandardScaler()
    scaler.fit(X_train)
    return pd.DataFrame(scaler.transform(X_test),
                        columns=X_test.columns, index=X_test.index)


def _threshold_mcc(proba, y_true):
    best_t, best_mcc = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        preds = (proba >= t).astype(int)
        mcc   = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, float(t)
    return best_t


def run_classical(name):
    print(f"  Loading features...")
    X_train, X_test, y_train, y_test, _, _ = _load_features_and_split()

    scaler_key = SCALER_MAP[name]
    X_test_sc  = _scale(X_train, X_test, scaler_key)
    del X_train
    gc.collect()

    model_path = os.path.join(TUNED_DIR, f"amp_model_{name}_tuned.pkl")
    print(f"  Loading model {name.upper()}...")
    model = joblib.load(model_path)

    print(f"  Computing probabilities...")
    proba = model.predict_proba(X_test_sc)[:, 1].astype(np.float32)

    importance = np.zeros(X_test_sc.shape[1], dtype=np.float32)
    if name in TREE_MODELS and hasattr(model, "feature_importances_"):
        importance = model.feature_importances_.astype(np.float32)

    thresh = load_threshold(name)
    del model
    gc.collect()

    return proba, y_test.astype(np.int8), thresh, importance


def run_deep(sequences, labels):
    import torch

    _, seq_test, _, y_test = train_test_split(
        sequences, labels,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels,
    )
    y_test = np.array(y_test, dtype=np.int8)

    model_path = os.path.join(TUNED_DIR, "amp_model_deep.pt")
    if not os.path.exists(model_path):
        print("  Deep model not found.")
        sys.exit(1)

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    from model_training.train_deep import AmpDeepModel
    state = torch.load(model_path, map_location=device, weights_only=True)
    model = AmpDeepModel().to(device)
    model.load_state_dict(state)
    model.eval()

    def _encode(seq):
        t = [AA_VOCAB.get(aa.upper(), 0) for aa in seq]
        if len(t) >= MAX_LEN:
            return t[:MAX_LEN]
        return t + [0] * (MAX_LEN - len(t))

    tensors = torch.tensor([_encode(s) for s in seq_test], dtype=torch.long)
    all_proba = []
    with torch.no_grad():
        for i in range(0, len(tensors), 256):
            xb     = tensors[i:i + 256].to(device)
            logits = model(xb).cpu().numpy()
            all_proba.append(1.0 / (1.0 + np.exp(-logits)))

    proba  = np.concatenate(all_proba).astype(np.float32)
    thresh = load_threshold("deep")
    return proba, y_test, thresh, np.zeros(0, dtype=np.float32)


def load_threshold(name):
    path = os.path.join(TUNED_DIR, f"threshold_{name}.txt")
    if os.path.exists(path):
        with open(path) as f:
            return float(f.read().strip())
    return 0.5


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m model_training.collect_outputs <model_name>")
        sys.exit(1)

    name = sys.argv[1].lower()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}_outputs.npz")

    print(f"Collecting outputs for {name.upper()}...")

    if name == "deep":
        pos_seqs, _ = load_fasta_sequences(POSITIVE_FILE)
        neg_seqs, _ = load_fasta_sequences(NEGATIVE_FILE)
        sequences   = pos_seqs + neg_seqs
        labels      = [1] * len(pos_seqs) + [0] * len(neg_seqs)
        proba, y_test, thresh, importance = run_deep(sequences, labels)
    elif name in SCALER_MAP:
        proba, y_test, thresh, importance = run_classical(name)
    else:
        print(f"Unknown model: {name}")
        sys.exit(1)

    np.savez_compressed(out_path,
                        proba=proba,
                        y_test=y_test,
                        threshold=np.array([thresh], dtype=np.float32),
                        importance=importance)
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
