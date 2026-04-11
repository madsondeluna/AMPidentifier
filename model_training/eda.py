# model_training/eda.py
#
# Phase 2.5: Exploratory Data Analysis for publication figures.
#
# Figures generated:
#   fig01_length_distribution.png       -- sequence length histogram (AMP vs non-AMP)
#   fig02_aa_composition.png            -- amino acid frequency (AMP vs non-AMP)
#   fig03_physicochemical_dist.png      -- KDE/violin of 6 global descriptors
#   fig04_charge_pi_scatter.png         -- charge vs pI scatter colored by class
#   fig05_pca.png                       -- PCA of 159 selected features
#   fig06_tsne.png                      -- t-SNE of 159 selected features
#   fig07_top_features_boxplot.png      -- top 12 RF-important features (boxplot)
#   fig08_class_balance.png             -- class balance and length quantiles
#
# Run from project root:
#   python -m model_training.eda

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler

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
RF_IMPORTANCE_PATH     = os.path.join("model_training/feature_analysis",
                                      "feature_selection_report.txt")

RANDOM_STATE = 42
FIGURE_DPI   = 200
TSNE_SAMPLE  = 3000   # subsample for t-SNE speed

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


def _load_top_features(n: int = 12) -> list:
    """Parse top features from feature_selection_report.txt."""
    top = []
    inside = False
    with open(RF_IMPORTANCE_PATH) as f:
        for line in f:
            if "Top 20 features by RF importance" in line:
                inside = True
                continue
            if inside:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].rstrip(".").isdigit():
                    top.append(parts[1])
                if len(top) == n:
                    break
    return top


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
# Fig 03 — Physicochemical distributions (KDE + rug)
# ---------------------------------------------------------------------------
def fig03_physicochemical(df: pd.DataFrame):
    props = [
        ("Charge",        "Net charge"),
        ("pI",            "Isoelectric point (pI)"),
        ("MW",            "Molecular weight (Da)"),
        ("HydrophRatio",  "Hydrophobic ratio"),
        ("BomanInd",      "Boman index"),
        ("AliphaticInd",  "Aliphatic index"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for ax, (col, xlabel) in zip(axes, props):
        for label, color in PALETTE.items():
            sub = df[df["label"] == label][col].dropna()
            sub.plot.kde(ax=ax, label=label, color=color, linewidth=1.8)
            ax.axvline(sub.median(), color=color, linestyle=":", linewidth=1.1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Physicochemical property distributions: AMP vs non-AMP",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, "fig03_physicochemical_dist.png")


# ---------------------------------------------------------------------------
# Fig 04 — Charge vs pI scatter
# ---------------------------------------------------------------------------
def fig04_charge_pi(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label, color in PALETTE.items():
        sub = df[df["label"] == label]
        ax.scatter(sub["pI"], sub["Charge"], c=color, label=label,
                   alpha=0.25, s=8, edgecolors="none", rasterized=True)
    ax.set_xlabel("Isoelectric point (pI)")
    ax.set_ylabel("Net charge")
    ax.set_title("Charge vs pI")
    ax.legend(markerscale=3)
    fig.tight_layout()
    _save(fig, "fig04_charge_pi_scatter.png")


# ---------------------------------------------------------------------------
# Fig 05 — PCA
# ---------------------------------------------------------------------------
def fig05_pca(df: pd.DataFrame, selected: list):
    X = df[selected].fillna(0).values
    y = df["label"].values

    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_sc)
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label, color in PALETTE.items():
        mask = y == label
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=label,
                   alpha=0.25, s=8, edgecolors="none", rasterized=True)
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
    ax.set_title("PCA of 159 selected features")
    ax.legend(markerscale=3)
    fig.tight_layout()
    _save(fig, "fig05_pca.png")


# ---------------------------------------------------------------------------
# Fig 06 — t-SNE
# ---------------------------------------------------------------------------
def fig06_tsne(df: pd.DataFrame, selected: list):
    X = df[selected].fillna(0).values
    y = df["label"].values

    # Subsample for speed
    rng = np.random.default_rng(RANDOM_STATE)
    n = min(TSNE_SAMPLE, len(X))
    idx = rng.choice(len(X), n, replace=False)
    X_sub, y_sub = X[idx], y[idx]

    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X_sub)

    # PCA pre-reduction to 50 dims before t-SNE (standard practice)
    pca_pre = PCA(n_components=min(50, X_sc.shape[1]), random_state=RANDOM_STATE)
    X_pca = pca_pre.fit_transform(X_sc)

    print(f"  Running t-SNE on {n} samples...")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
                random_state=RANDOM_STATE, n_jobs=-1)
    coords = tsne.fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label, color in PALETTE.items():
        mask = y_sub == label
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=label,
                   alpha=0.35, s=10, edgecolors="none", rasterized=True)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE of 159 selected features (n={n})")
    ax.legend(markerscale=3)
    fig.tight_layout()
    _save(fig, "fig06_tsne.png")


# ---------------------------------------------------------------------------
# Fig 07 — Top features boxplot
# ---------------------------------------------------------------------------
def fig07_top_features_boxplot(df: pd.DataFrame):
    top_feats = _load_top_features(12)
    plot_df = df[top_feats + ["label"]].melt(id_vars="label",
                                              var_name="feature",
                                              value_name="value")

    # Shorten CTD names for readability
    def _short(name):
        return (name.replace("CTD_", "")
                    .replace("solvent_access", "solv")
                    .replace("hydrophobicity", "hydro")
                    .replace("polarizability", "polar")
                    .replace("secondary_struct", "2struct")
                    .replace("charge", "chg"))

    plot_df["feature"] = plot_df["feature"].apply(_short)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(
        data=plot_df, x="feature", y="value", hue="label",
        palette=PALETTE, linewidth=0.8, fliersize=2,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Feature value")
    ax.set_title("Top 12 features by RF importance: AMP vs non-AMP")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="")
    fig.tight_layout()
    _save(fig, "fig07_top_features_boxplot.png")


# ---------------------------------------------------------------------------
# Fig 08 — Class balance + length quantile table
# ---------------------------------------------------------------------------
def fig08_class_balance(df: pd.DataFrame):
    fig = plt.figure(figsize=(11, 4.5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.6])

    # -- Pie chart
    ax_pie = fig.add_subplot(gs[0])
    counts = df["label"].value_counts()
    colors = [PALETTE["AMP"], PALETTE["non-AMP"]]
    wedges, texts, autotexts = ax_pie.pie(
        counts.values, labels=counts.index, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 10},
    )
    ax_pie.set_title("Class balance", fontsize=11)

    # -- Length quantile table
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")
    stats = []
    for label in ["AMP", "non-AMP"]:
        sub = df[df["label"] == label]["length"]
        stats.append([
            label,
            f"{int(sub.min())}",
            f"{sub.quantile(0.25):.0f}",
            f"{sub.median():.0f}",
            f"{sub.mean():.1f}",
            f"{sub.quantile(0.75):.0f}",
            f"{int(sub.max())}",
        ])
    col_labels = ["Class", "Min", "Q1", "Median", "Mean", "Q3", "Max"]
    tbl = ax_tbl.table(
        cellText=stats,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    ax_tbl.set_title("Sequence length statistics (aa)", fontsize=11, pad=20)

    fig.tight_layout()
    _save(fig, "fig08_class_balance.png")


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

    print("Fig 04 — Charge vs pI scatter...")
    fig04_charge_pi(df)

    print("Fig 05 — PCA...")
    fig05_pca(df, selected)

    print("Fig 06 — t-SNE...")
    fig06_tsne(df, selected)

    print("Fig 07 — Top features boxplot...")
    fig07_top_features_boxplot(df)

    print("Fig 08 — Class balance...")
    fig08_class_balance(df)

    print("\n--- EDA complete ---")
    print(f"All figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
