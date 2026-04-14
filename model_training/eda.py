# model_training/eda.py
#
# Exploratory Data Analysis — Macrel-inspired feature set.
#
# Figures generated:
#   fig01_length_distribution.png       -- sequence length histogram (AMP vs non-AMP)
#   fig02_aa_composition.png            -- per-residue frequency (AMP vs non-AMP)
#   fig03_global_descriptors.png        -- KDE of 6 global descriptors + hydrophobic moment
#   fig04_grouped_aac.png               -- grouped amino acid composition (9 groups)
#   fig05_local_features.png            -- FET and solvent access positional features (6)
#
# Run from project root:
#   python3 -m model_training.eda

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
DATA_DIR      = "model_training/data"
OUT_DIR       = "model_training/eda"
POSITIVE_FILE = os.path.join(DATA_DIR, "positive_sequences.fasta")
NEGATIVE_FILE = os.path.join(DATA_DIR, "negative_sequences.fasta")

FIGURE_DPI = 200
PALETTE    = {"AMP": "#2166AC", "non-AMP": "#D6604D"}
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

    print("Extracting features...")
    df = calculate_physicochemical_features(sequences, ids)
    df["label"]  = labels
    df["length"] = df["sequence"].str.len()

    print(f"  {len(pos_seqs)} AMPs | {len(neg_seqs)} non-AMPs")
    return df


# ---------------------------------------------------------------------------
# Fig 01 — Sequence length distribution
# ---------------------------------------------------------------------------
def fig01_length(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, label, color in zip(axes, ["AMP", "non-AMP"],
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
# Fig 02 — Per-residue amino acid composition
# ---------------------------------------------------------------------------
def fig02_aa_composition(df: pd.DataFrame):
    def _mean_freq(sub):
        return {aa: sub["sequence"].str.count(aa).div(sub["length"]).mean()
                for aa in AMINO_ACIDS}

    amp_f    = _mean_freq(df[df["label"] == "AMP"])
    nonamp_f = _mean_freq(df[df["label"] == "non-AMP"])

    x = np.arange(len(AMINO_ACIDS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - w/2, [amp_f[a]    for a in AMINO_ACIDS], w,
           label="AMP",     color=PALETTE["AMP"],     edgecolor="none", alpha=0.9)
    ax.bar(x + w/2, [nonamp_f[a] for a in AMINO_ACIDS], w,
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
# Fig 03 — Global descriptors + hydrophobic moment (7 features, KDE)
# ---------------------------------------------------------------------------
def fig03_global_descriptors(df: pd.DataFrame):
    props = [
        ("Charge",             "Net charge"),
        ("pI",                 "Isoelectric point (pI)"),
        ("InstabilityInd",     "Instability index"),
        ("AliphaticInd",       "Aliphatic index"),
        ("BomanInd",           "Boman index"),
        ("HydrophRatio",       "Hydrophobic ratio"),
        ("HydrophobicMoment",  "Hydrophobic moment (100 deg)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for ax, (col, xlabel) in zip(axes, props):
        for label, color in PALETTE.items():
            sub = df[df["label"] == label][col].dropna()
            sub.plot.kde(ax=ax, label=label, color=color, linewidth=1.8)
            ax.axvline(sub.median(), color=color, linestyle=":", linewidth=1.1)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    axes[-1].set_visible(False)
    fig.suptitle("Global physicochemical descriptors: AMP vs non-AMP",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "fig03_global_descriptors.png")


# ---------------------------------------------------------------------------
# Fig 04 — Grouped amino acid composition (9 groups, bar chart + KDE)
# ---------------------------------------------------------------------------
def fig04_grouped_aac(df: pd.DataFrame):
    groups = [
        ("f_acidic",    "f_acidic (DE)"),
        ("f_basic",     "f_basic (KRH)"),
        ("f_polar",     "f_polar (STNQ)"),
        ("f_nonpolar",  "f_nonpolar (AVLIMFYWP)"),
        ("f_aliphatic", "f_aliphatic (AVLIM)"),
        ("f_aromatic",  "f_aromatic (FYW)"),
        ("f_charged",   "f_charged (DEKRH)"),
        ("f_small",     "f_small (AGSDT)"),
        ("f_tiny",      "f_tiny (AGS)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, groups):
        for label, color in PALETTE.items():
            sub = df[df["label"] == label][col].dropna()
            sub.plot.kde(ax=ax, label=label, color=color, linewidth=1.8)
            ax.axvline(sub.median(), color=color, linestyle=":", linewidth=1.1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Fraction of residues")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    fig.suptitle("Grouped amino acid composition: AMP vs non-AMP",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "fig04_grouped_aac.png")


# ---------------------------------------------------------------------------
# Fig 05 — Local positional features: FET + solvent access (6 features, KDE)
# ---------------------------------------------------------------------------
def fig05_local_features(df: pd.DataFrame):
    features = [
        ("FET_low_D1",  "FET_low_D1\n(ILVWAMGT, lowest FET)"),
        ("FET_mid_D1",  "FET_mid_D1\n(FYSQCN, intermediate)"),
        ("FET_high_D1", "FET_high_D1\n(PHKEDR, highest FET)"),
        ("SA_buried_D1",  "SA_buried_D1\n(ALFCGIVW)"),
        ("SA_exposed_D1", "SA_exposed_D1\n(RKQEND)"),
        ("SA_inter_D1",   "SA_inter_D1\n(MSPTHY)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, features):
        for label, color in PALETTE.items():
            sub = df[df["label"] == label][col].dropna()
            sub = sub[sub > 0]   # exclude sequences with no residue in the group
            if len(sub) > 1:
                sub.plot.kde(ax=ax, label=label, color=color, linewidth=1.8)
            ax.axvline(sub.median(), color=color, linestyle=":", linewidth=1.1)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("Relative position (0-1)")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7)

    fig.suptitle("Local positional features: FET and solvent accessibility",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "fig05_local_features.png")


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

    df = load_data()

    print("\nFig 01 — Length distribution...")
    fig01_length(df)

    print("Fig 02 — Amino acid composition...")
    fig02_aa_composition(df)

    print("Fig 03 — Global descriptors...")
    fig03_global_descriptors(df)

    print("Fig 04 — Grouped AAC...")
    fig04_grouped_aac(df)

    print("Fig 05 — Local positional features...")
    fig05_local_features(df)

    print("\n--- EDA complete ---")
    print(f"Figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
