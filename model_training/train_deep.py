# model_training/train_deep.py
#
# Deep learning pipeline: 1D-CNN + BiLSTM on raw amino acid sequences.
#
# Architecture:
#   Embedding(21, 64, padding_idx=0)
#   -> Conv1d(64->128, k=3, pad=1) + BN + ReLU
#   -> Conv1d(128->128, k=5, pad=2) + BN + ReLU
#   -> MaxPool1d(2)
#   -> BiLSTM(input=128, hidden=128, layers=2, bidirectional=True, dropout=0.3)
#   -> GlobalMaxPool over sequence dim
#   -> FC(256->128) + ReLU + Dropout(0.3)
#   -> FC(128->1)  [logits]
#
# Training:
#   BCEWithLogitsLoss (no pos_weight; dataset is balanced)
#   AdamW(lr=1e-3, weight_decay=1e-4)
#   ReduceLROnPlateau(patience=5, factor=0.5)
#   EarlyStopping(patience=10) on val AUC-ROC
#   Threshold tuned by MCC on validation split (10% of training data)
#
# Outputs (model_training/tuned_model/):
#   amp_model_deep.pt      - best model weights
#   result_deep.csv        - metrics (same columns as tuning_report.csv)
#   threshold_deep.txt     - optimal MCC threshold
#
# Run from project root:
#   python -m model_training.train_deep

import os
import sys
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef,
)

from amp_identifier.data_io import load_fasta_sequences

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = "model_training/data"
OUTPUT_DIR = "model_training/tuned_model"

POSITIVE_FILE = os.path.join(DATA_DIR, "positive_sequences.fasta")
NEGATIVE_FILE = os.path.join(DATA_DIR, "negative_sequences.fasta")

RANDOM_STATE = 42
TEST_SIZE    = 0.20   # must match train.py / tune.py split
VAL_SIZE     = 0.10   # fraction of training data used for validation

MAX_LEN      = 200
BATCH_SIZE   = 64
MAX_EPOCHS   = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE_ES  = 10     # early stopping patience (val AUC)
PATIENCE_LR  = 5      # ReduceLROnPlateau patience

# Amino acid vocabulary: PAD=0, standard 20 AAs indexed 1-20
AA_VOCAB = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
VOCAB_SIZE = len(AA_VOCAB) + 1  # 21 (index 0 = PAD)

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def encode_sequence(seq: str) -> list[int]:
    """Convert amino acid string to integer indices; unknown AAs map to 0."""
    return [AA_VOCAB.get(aa.upper(), 0) for aa in seq]


def pad_or_truncate(indices: list[int], max_len: int) -> list[int]:
    if len(indices) >= max_len:
        return indices[:max_len]
    return indices + [0] * (max_len - len(indices))


class PeptideDataset(Dataset):
    def __init__(self, sequences: list[str], labels: list[int]):
        self.data = [
            torch.tensor(pad_or_truncate(encode_sequence(s), MAX_LEN), dtype=torch.long)
            for s in sequences
        ]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AmpDeepModel(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 64,
                 cnn_channels: int = 128, lstm_hidden: int = 128,
                 lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.cnn = nn.Sequential(
            nn.Conv1d(embed_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # BiLSTM: input_size=cnn_channels, output per step = lstm_hidden*2
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * 2  # bidirectional

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.embedding(x)                # (batch, seq_len, embed_dim)
        emb = emb.transpose(1, 2)              # (batch, embed_dim, seq_len)
        cnn_out = self.cnn(emb)                # (batch, cnn_channels, seq_len//2)
        cnn_out = cnn_out.transpose(1, 2)      # (batch, seq_len//2, cnn_channels)
        lstm_out, _ = self.lstm(cnn_out)       # (batch, seq_len//2, lstm_hidden*2)
        pooled = lstm_out.max(dim=1).values    # (batch, lstm_hidden*2)
        logits = self.classifier(pooled)       # (batch, 1)
        return logits.squeeze(1)               # (batch,)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def evaluate_loader(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb).cpu()
            all_logits.append(logits)
            all_labels.append(yb)
    logits_cat = torch.cat(all_logits).numpy()
    labels_cat = torch.cat(all_labels).numpy()
    proba = 1 / (1 + np.exp(-logits_cat))  # sigmoid
    return proba, labels_cat


def tune_threshold_mcc(proba: np.ndarray, labels: np.ndarray) -> float:
    best_t, best_mcc = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        preds = (proba >= t).astype(int)
        mcc = matthews_corrcoef(labels, preds)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, float(t)
    return best_t


def compute_metrics(proba: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "accuracy":    float(accuracy_score(labels, preds)),
        "precision":   float(precision_score(labels, preds, zero_division=0)),
        "recall":      float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1":          float(f1_score(labels, preds, zero_division=0)),
        "mcc":         float(matthews_corrcoef(labels, preds)),
        "auc_roc":     float(roc_auc_score(labels, proba)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load sequences ---
    print("\nStep 1: Loading sequences...")
    pos_seqs, _ = load_fasta_sequences(POSITIVE_FILE)
    neg_seqs, _ = load_fasta_sequences(NEGATIVE_FILE)
    sequences = pos_seqs + neg_seqs
    labels    = [1] * len(pos_seqs) + [0] * len(neg_seqs)
    print(f"  Positive: {len(pos_seqs)}  Negative: {len(neg_seqs)}  Total: {len(sequences)}")

    # --- Replicate same train/test split as tune.py / train.py ---
    print("\nStep 2: Splitting data (80/20 stratified, random_state=42)...")
    (seq_train, seq_test,
     y_train,   y_test) = train_test_split(
        sequences, labels,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels,
    )
    print(f"  Train: {len(seq_train)}  Test: {len(seq_test)}")

    # --- Further split train -> train_sub + val (for early stopping) ---
    print(f"\nStep 3: Splitting train into sub-train/val ({int((1-VAL_SIZE)*100)}/{int(VAL_SIZE*100)})...")
    (seq_sub, seq_val,
     y_sub,   y_val) = train_test_split(
        seq_train, y_train,
        test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train,
    )
    print(f"  Sub-train: {len(seq_sub)}  Val: {len(seq_val)}")

    # --- DataLoaders ---
    train_ds = PeptideDataset(seq_sub, y_sub)
    val_ds   = PeptideDataset(seq_val, y_val)
    test_ds  = PeptideDataset(seq_test, y_test)

    g = torch.Generator()
    g.manual_seed(RANDOM_STATE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model ---
    print("\nStep 4: Building model...")
    model = AmpDeepModel().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=PATIENCE_LR, factor=0.5,
    )

    # --- Training loop ---
    print(f"\nStep 5: Training (max {MAX_EPOCHS} epochs, early stopping patience={PATIENCE_ES})...\n")
    best_val_auc  = -1.0
    best_epoch    = 0
    patience_cnt  = 0
    best_state    = None

    header = f"{'Epoch':>6}  {'Train Loss':>10}  {'Val AUC':>8}  {'Val MCC':>8}  {'LR':>10}  {'Time':>7}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(yb)

        train_loss = running_loss / len(train_ds)

        val_proba, val_labels = evaluate_loader(model, val_loader)
        val_auc = roc_auc_score(val_labels, val_proba)
        val_t   = tune_threshold_mcc(val_proba, val_labels)
        val_mcc = matthews_corrcoef(val_labels, (val_proba >= val_t).astype(int))

        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        marker = " *" if val_auc > best_val_auc else ""
        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_auc:>8.4f}  {val_mcc:>8.4f}  "
              f"{current_lr:>10.2e}  {_fmt(elapsed):>7}{marker}")
        sys.stdout.flush()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            patience_cnt = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE_ES:
                print(f"\nEarly stopping at epoch {epoch} (best val AUC={best_val_auc:.4f} at epoch {best_epoch})")
                break

    # --- Restore best weights ---
    model.load_state_dict(best_state)

    # --- Tune threshold on validation set ---
    print("\nStep 6: Tuning threshold (MCC) on validation set...")
    val_proba, val_labels = evaluate_loader(model, val_loader)
    threshold = tune_threshold_mcc(val_proba, val_labels)
    print(f"  Optimal threshold: {threshold:.2f}")

    # --- Evaluate on test set ---
    print("\nStep 7: Evaluating on test set...")
    test_proba, test_labels = evaluate_loader(model, test_loader)
    metrics = compute_metrics(test_proba, test_labels, threshold)
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}  MCC: {metrics['mcc']:.4f}  "
          f"F1: {metrics['f1']:.4f}  Acc: {metrics['accuracy']:.4f}")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")

    # --- Save outputs ---
    print("\nStep 8: Saving model and results...")

    model_path = os.path.join(OUTPUT_DIR, "amp_model_deep.pt")
    torch.save(best_state, model_path)
    print(f"  Model saved -> {model_path}")

    threshold_path = os.path.join(OUTPUT_DIR, "threshold_deep.txt")
    with open(threshold_path, "w") as f:
        f.write(str(threshold))
    print(f"  Threshold saved -> {threshold_path}")

    result_row = {
        "model":           "DEEP",
        "best_cv_roc_auc": round(best_val_auc, 4),
        "threshold":       threshold,
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
    }
    result_df = pd.DataFrame([result_row])
    result_path = os.path.join(OUTPUT_DIR, "result_deep.csv")
    result_df.to_csv(result_path, index=False)
    print(f"  Results saved -> {result_path}")

    print("\n--- Deep learning training complete ---")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
