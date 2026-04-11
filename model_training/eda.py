# model_training/eda.py
#
# Phase 2.5: Exploratory Data Analysis for publication figures.
#
# Figures generated:
#   fig01_length_distribution.png       -- sequence length histogram (AMP vs non-AMP)
#   fig02_aa_composition.png            -- amino acid frequency (AMP vs non-AMP)
#   fig03_physicochemical_dist.png      -- KDE of 8 global descriptors + 4 top AAC features
#
# Run from project root:
#   python -m model_training.eda

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from amp_identifier.feature_extraction import calculate_physicochemical_features
from amp_identifier.data_io import load_fasta_sequences

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = "model_training/data"
OUT_DIR    = "model_training/eda"
POSITIVE_FILE          = os.path.join(DATA_DIR, "positive_sequences.fasta")
NEGATIVE_FILE          = os.path.join(DATA_DIR, "negative_sequences.fasta")
SELECTED_FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.txt")

RANDOM_STATE = 42
FIGURE_DPI   = 200

PALETTE = {"AMP": "#2166AC", "non-AMP": "#D6604D"}
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    print("Loading sequences...")
    pos_seqs, pos_ids = load_fasta_sequences(POSITIVE_FILE)
    neg_seqs, neg_ids = load_fasta_sequences(NEGATIVE_FILE)

    sequences = pos_seqs + neg_seqs
    ids       = pos_ids  + neg_ids
    labels    = ["AMP"] * len(pos_seqs) + ["non-AMP"] * len(neg_seqs)
    y_bin     = [1] * len(pos_seqs) + [0] * len(neg_seqs)

    print("Extracting features...")
    features_df = calculate_physicochemical_features(sequences, ids)
    features_df["label"]  = labels
    features_df["y"]      = y_bin
    features_df["length"] = features_df["sequence"].str.len()

    selected = open(SELECTED_FEATURES_PATH).read().splitlines()
    print(f"  {len(pos_seqs)} AMPs | {len(neg_seqs)} non-AMPs | "
          f"{features_df.shape[0]} total | {len(selected)} features")
    return features_df, selected


# ---------------------------------------------------------------------------
# Fig 01 — Length distribution
# ---------------------------------------------------------------------------
def fig01_length(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, label, color in zip(axes,
                                 ["AMP", "non-AMP"],
                                 [PALETTE["AMP"], PALETTE["non-AMP"]]):
        sub = df[df["label"] == label]["length"]
        ax.hist(sub, bins=40, color=color, edgecolor="none", alpha=0.85)
        ax.axvline(sub.median(), color="#333333", linestyle="--", linewidth=1.2,
                   label=f"median = {sub.median():.0f} aa")
        ax.set_xlabel("Sequence length (aa)")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=9)

    fig.suptitle("Sequence length distribution", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig01_length_distribution.png")


# ---------------------------------------------------------------------------
# Fig 02 — Amino acid composition
# ---------------------------------------------------------------------------
def fig02_aa_composition(df: pd.DataFrame):
    def _mean_aac(sub):
        return {aa: sub["sequence"].str.count(aa).div(sub["length"]).mean()
                for aa in AMINO_ACIDS}

    amp_aac    = _mean_aac(df[df["label"] == "AMP"])
    nonamp_aac = _mean_aac(df[df["label"] == "non-AMP"])

    x = np.arange(len(AMINO_ACIDS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - w/2, [amp_aac[a]    for a in AMINO_ACIDS], w,
           label="AMP",     color=PALETTE["AMP"],     edgecolor="none", alpha=0.9)
    ax.bar(x + w/2, [nonamp_aac[a] for a in AMINO_ACIDS], w,
           label="non-AMP", color=PALETTE["non-AMP"], edgecolor="none", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(AMINO_ACIDS)
    ax.set_xlabel("Amino acid")
    ax.set_ylabel("Mean relative frequency")
    ax.set_title("Amino acid composition: AMP vs non-AMP")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig02_aa_composition.png")


# ---------------------------------------------------------------------------
# Fig 03 — Physicochemical distributions (KDE, 8 global descriptors)
# ---------------------------------------------------------------------------
def fig03_physicochemical(df: pd.DataFrame):
    props = [
        ("Charge",        "Net charge"),
        ("pI",            "Isoelectric point (pI)"),
        ("MW",            "Molecular weight (Da)"),
        ("HydrophRatio",  "Hydrophobic ratio"),
        ("BomanInd",      "Boman index"),
        ("AliphaticInd",  "Aliphatic index"),
        ("Aromaticity",   "Aromaticity"),
        ("InstabilityInd","Instability index"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for ax, (col, xlabel) in zip(axes, props):
        for label, color in PALETTE.items():
            sub = df[df["label"] == label][col].dropna()
            sub.plot.kde(ax=ax, label=label, color=color, linewidth=1.8)
            ax.axvline(sub.median(), color=color, linestyle=":", linewidth=1.1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Global physicochemical descriptor distributions: AMP vs non-AMP",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "fig03_physicochemical_dist.png")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _save(fig, filename: str):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df, selected = load_data()

    print("\nFig 01 — Length distribution...")
    fig01_length(df)

    print("Fig 02 — Amino acid composition...")
    fig02_aa_composition(df)

    print("Fig 03 — Physicochemical distributions...")
    fig03_physicochemical(df)

    print("\n--- EDA complete ---")
    print(f"All figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
