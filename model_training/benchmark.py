# model_training/benchmark.py
#
# Evaluates all 7 tuned models on an independent benchmark set.
#
# Input:
#   benchmarking/benchmark.fasta — FASTA where each header contains
#   "label=1" (AMP) or "label=0" (non-AMP), e.g.:
#     >pos_1 label=1
#     GLFDIVKKVVGALGSL...
#
# Outputs (benchmarking/):
#   benchmark_results.csv       — per-model metrics
#   fig_bench_roc.png           — ROC curves on benchmark set
#   fig_bench_metrics.png       — grouped bar chart of all metrics
#
# Run from project root:
#   python -m model_training.benchmark

import os
import gc
import re
import sys

import numpy as np
import pandas as pd
import joblib
import matplotlib as mpl
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, matthews_corrcoef,
    precision_score, recall_score, f1_score, accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer

from amp_identifier.feature_extraction import calculate_physicochemical_features
from amp_identifier.data_io import load_fasta_sequences
from model_training.voting import VotingEnsemble

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCH_FASTA   = "benchmarking/benchmark.fasta"
DATA_DIR      = "model_training/data"
TUNED_DIR     = "model_training/tuned_model"
OUT_DIR       = "benchmarking"
POSITIVE_FILE = os.path.join(DATA_DIR, "positive_sequences.fasta")
NEGATIVE_FILE = os.path.join(DATA_DIR, "negative_sequences.fasta")
SEL_FEAT_PATH = os.path.join(DATA_DIR, "selected_features.txt")
RANDOM_STATE  = 42
TEST_SIZE     = 0.20

# ---------------------------------------------------------------------------
# NPG style (matches plot_tuning.py)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COL1 = 3.50
COL2 = 7.20

COLORS = {
    "RF":     "#4DBBD5",
    "SVM":    "#E64B35",
    "GB":     "#00A087",
    "XGB":    "#3C5488",
    "VOTING": "#8491B4",
}
LINESTYLES = {
    "RF":     "-",
    "SVM":    (0, (5, 1)),
    "GB":     "-.",
    "XGB":    ":",
    "VOTING": (0, (1, 1)),
}
ALPHA = 0.85
MODELS_ORDERED = ["RF", "SVM", "GB", "XGB", "VOTING"]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def savefig(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def parse_benchmark_fasta(path):
    """Return (sequences, labels) from a FASTA with 'label=N' in headers."""
    sequences, labels = [], []
    seq_buf = []
    label = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if seq_buf and label is not None:
                    sequences.append("".join(seq_buf))
                    labels.append(label)
                seq_buf = []
                m = re.search(r"label=([01])", line)
                label = int(m.group(1)) if m else None
            else:
                seq_buf.append(line)
    if seq_buf and label is not None:
        sequences.append("".join(seq_buf))
        labels.append(label)
    return sequences, labels


def load_threshold(name):
    path = os.path.join(TUNED_DIR, f"threshold_{name.lower()}.txt")
    if os.path.exists(path):
        with open(path) as f:
            return float(f.read().strip())
    return 0.5


def compute_metrics(y_true, proba, threshold):
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1   = f1_score(y_true, preds, zero_division=0)
    mcc  = matthews_corrcoef(y_true, preds)
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_roc = auc(fpr, tpr)
    return dict(
        threshold=threshold,
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        specificity=round(spec, 4),
        f1=round(f1, 4),
        mcc=round(mcc, 4),
        auc_roc=round(auc_roc, 4),
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
    ), fpr, tpr

# ---------------------------------------------------------------------------
# Feature loading and scaling
# ---------------------------------------------------------------------------

def build_scaler_cache(bench_sequences, bench_ids):
    """
    Returns (X_bench_scaled, selected_features) where X_bench_scaled is a dict
    mapping scaler key -> transformed DataFrame.
    Scaler is fit on the training split of the original pipeline.
    """
    print("Loading training sequences for scaler fitting...")
    pos_seqs, pos_ids = load_fasta_sequences(POSITIVE_FILE)
    neg_seqs, neg_ids = load_fasta_sequences(NEGATIVE_FILE)
    seqs_all = pos_seqs + neg_seqs
    ids_all  = pos_ids + neg_ids
    labels_all = [1] * len(pos_seqs) + [0] * len(neg_seqs)

    print(f"  Computing features for {len(seqs_all)} training sequences...")
    feat_df = calculate_physicochemical_features(seqs_all, ids_all)
    feat_df["label"] = labels_all

    with open(SEL_FEAT_PATH) as f:
        selected = [l.strip() for l in f if l.strip()]

    X = feat_df[selected].fillna(0)
    y = np.array(feat_df["label"])

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    del feat_df, X, y
    gc.collect()

    print(f"  Computing features for {len(bench_sequences)} benchmark sequences...")
    bench_feat = calculate_physicochemical_features(bench_sequences, bench_ids)
    X_bench = bench_feat[selected].fillna(0)
    del bench_feat
    gc.collect()

    scalers = {
        "robust": RobustScaler(),
        "std":    StandardScaler(),
        "qt":     QuantileTransformer(output_distribution="normal",
                                      random_state=RANDOM_STATE),
    }
    scaled = {}
    for key, scaler in scalers.items():
        scaler.fit(X_train)
        scaled[key] = pd.DataFrame(
            scaler.transform(X_bench),
            columns=X_bench.columns,
            index=X_bench.index,
        )
    scaled["raw"] = X_bench

    del X_train
    gc.collect()
    return scaled


SCALER_MAP = {
    "rf":     "robust",
    "svm":    "std",
    "gb":     "robust",
    "xgb":    "robust",
    "voting": "raw",
}

# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def eval_classical(name, X_bench_scaled, y_bench):
    key = SCALER_MAP[name.lower()]
    X   = X_bench_scaled[key]
    model_path = os.path.join(TUNED_DIR, f"amp_model_{name.lower()}_tuned.pkl")
    model = joblib.load(model_path)
    proba = model.predict_proba(X)[:, 1].astype(np.float64)
    del model
    gc.collect()
    thresh = load_threshold(name)
    return compute_metrics(y_bench, proba, thresh)



# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_bench_roc(roc_data):
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.95))
    for name in MODELS_ORDERED:
        fpr, tpr, auc_val = roc_data[name]
        lw = 1.2 if name == "GB" else 0.9
        ax.plot(fpr, tpr,
                color=COLORS[name], linewidth=lw,
                linestyle=LINESTYLES[name], alpha=ALPHA,
                label=f"{name} ({auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.7, linestyle=":")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Independent Benchmark")
    n_mod = len(MODELS_ORDERED)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=n_mod,
        fontsize=6,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.8,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    savefig(fig, "fig_bench_roc.png")


def fig_bench_metrics(results):
    metrics = ["accuracy", "precision", "recall", "specificity", "f1", "mcc", "auc_roc"]
    labels  = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "MCC", "AUC-ROC"]
    n_met   = len(metrics)
    n_mod   = len(MODELS_ORDERED)
    x       = np.arange(n_met)
    width   = 0.10
    offsets = np.linspace(-(n_mod - 1) / 2, (n_mod - 1) / 2, n_mod) * width

    fig, ax = plt.subplots(figsize=(COL2, COL1 * 1.1))
    for i, name in enumerate(MODELS_ORDERED):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width,
                      color=COLORS[name], alpha=ALPHA, label=name,
                      linewidth=0)
        for bar, v in zip(bars, vals):
            if v >= 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.006,
                    f"{v * 100:.1f}%",
                    ha="center", va="bottom",
                    fontsize=3.5, rotation=90,
                    color=COLORS[name],
                )

    ref_lines = [
        (0.50, "50%", (0, (1, 2)),      0.35),
        (0.80, "80%", (0, (4, 2)),      0.35),
        (0.90, "90%", (0, (6, 2, 1, 2)), 0.35),
    ]
    ref_handles = []
    for yval, label, ls, alpha in ref_lines:
        line = ax.axhline(
            yval, color="#aaaaaa", linewidth=0.8,
            linestyle=ls, alpha=alpha, zorder=0,
        )
        ref_handles.append(
            mpl.lines.Line2D([], [], color="#aaaaaa", linewidth=0.8,
                             linestyle=ls, alpha=alpha, label=label)
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.20))
    ax.set_ylabel("Score")
    ax.set_title("Model Performance — Independent Benchmark (n=4,736)")

    model_handles, model_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=model_handles + ref_handles,
        labels=model_labels + [h.get_label() for h in ref_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=n_mod + len(ref_lines),
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout()
    savefig(fig, "fig_bench_metrics.png")

def fig_bench_confusion(results):
    n_mod    = len(MODELS_ORDERED)
    cell_in  = 1.8                          # each subplot: 1.8" x 1.8"
    fig, axes = plt.subplots(
        1, n_mod,
        figsize=(n_mod * cell_in + 0.4, cell_in + 1.0),
    )
    axes_flat = list(axes)

    for idx, name in enumerate(MODELS_ORDERED):
        ax    = axes_flat[idx]
        m     = results[name]
        cm    = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=int)
        tot   = cm.sum()
        color = COLORS[name]

        ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=tot)
        ax.set_box_aspect(1)

        for row in range(2):
            for col in range(2):
                val = cm[row, col]
                pct = val / tot * 100
                ax.text(col, row, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center",
                        fontsize=6.5,
                        color="white" if val > tot * 0.45 else "#222222")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("True", fontsize=7)
        ax.set_xticklabels(["Non-AMP", "AMP"], fontsize=6.5)
        ax.set_yticklabels(["Non-AMP", "AMP"], fontsize=6.5)
        ax.set_title(name, fontsize=8, color=color, fontweight="bold")

    fig.suptitle("Confusion Matrices — Independent Benchmark", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig(fig, "fig_bench_confusion.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Parsing benchmark FASTA...")
    bench_seqs, bench_labels = parse_benchmark_fasta(BENCH_FASTA)
    bench_ids = [f"bench_{i}" for i in range(len(bench_seqs))]
    y_bench   = np.array(bench_labels, dtype=np.int8)
    n_pos = int(np.sum(y_bench == 1))
    n_neg = int(np.sum(y_bench == 0))
    print(f"  {len(bench_seqs)} sequences: {n_pos} AMP, {n_neg} non-AMP")

    print("Building scaler cache from training data...")
    X_bench_scaled = build_scaler_cache(bench_seqs, bench_ids)

    results  = {}
    roc_data = {}
    for name in MODELS_ORDERED:
        print(f"Evaluating {name}...")
        m, fpr, tpr = eval_classical(name, X_bench_scaled, y_bench)
        results[name]  = m
        roc_data[name] = (fpr, tpr, m["auc_roc"])
        print(f"  AUC-ROC={m['auc_roc']:.4f}  MCC={m['mcc']:.4f}  "
              f"F1={m['f1']:.4f}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")

    # Save CSV
    rows = []
    for name in MODELS_ORDERED:
        rows.append({"model": name, **results[name]})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    print("Generating figures...")
    fig_bench_roc(roc_data)
    fig_bench_metrics(results)
    fig_bench_confusion(results)
    print("Done.")


if __name__ == "__main__":
    main()
