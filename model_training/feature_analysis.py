# model_training/feature_analysis.py
#
# Phase 2: correlation analysis and feature selection.
#
# Pipeline:
#   1. Load sequences and extract the full feature set (~577 features).
#   2. Variance threshold: remove features with variance <= VARIANCE_THRESHOLD.
#   3. Structural filter: remove CTD Composition features (CTD_*_C1/C2/C3).
#      Rationale: C1+C2+C3=1 by definition (Dubchak et al. 1995), creating
#      perfect multicollinearity when all three groups are present. Additionally,
#      CTD_C features are linear combinations of AAC features (e.g.
#      CTD_charge_C1 = AAC_K + AAC_R). CTD Transition (T) and Distribution (D)
#      are retained as they encode positional information absent from AAC.
#   4. Pearson correlation filter: remove one of each pair with |r| > CORR_THRESHOLD.
#      The feature with higher mean absolute correlation to the rest is dropped.
#   5. RF importance ranking (text report only, no plot).
#   6. Save:
#       - model_training/data/selected_features.txt
#       - model_training/feature_analysis/fig_correlation_heatmap_before.png
#       - model_training/feature_analysis/fig_correlation_heatmap_after.png
#       - model_training/feature_analysis/feature_selection_report.txt
#
# Run from project root:
#   python -m model_training.feature_analysis

import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from amp_identifier.feature_extraction import calculate_physicochemical_features
from amp_identifier.data_io import load_fasta_sequences

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = "model_training/data"
OUT_DIR    = "model_training/feature_analysis"
POSITIVE_FILE          = os.path.join(DATA_DIR, "positive_sequences.fasta")
NEGATIVE_FILE          = os.path.join(DATA_DIR, "negative_sequences.fasta")
SELECTED_FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.txt")

RANDOM_STATE       = 42
TEST_SIZE          = 0.2
VARIANCE_THRESHOLD = 0.001   # remove features with variance <= this value
CORR_THRESHOLD     = 0.90    # remove one of each pair with |r| > this value
RF_N_ESTIMATORS    = 200

FIGURE_DPI = 200


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_data():
    print("Loading sequences...")
    pos_seqs, pos_ids = load_fasta_sequences(POSITIVE_FILE)
    neg_seqs, neg_ids = load_fasta_sequences(NEGATIVE_FILE)
    sequences = pos_seqs + neg_seqs
    ids       = pos_ids  + neg_ids
    labels    = [1] * len(pos_seqs) + [0] * len(neg_seqs)
    print(f"  Positive: {len(pos_seqs)}  Negative: {len(neg_seqs)}")

    print("Extracting features...")
    t0 = time.time()
    features_df = calculate_physicochemical_features(sequences, ids)
    features_df["label"] = labels
    print(f"  Done in {time.time()-t0:.1f}s  Shape: {features_df.shape}")

    X = features_df.drop(columns=["ID", "sequence", "label"]).fillna(0)
    y = features_df["label"]
    return X, y


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def _variance_filter(X: pd.DataFrame) -> pd.DataFrame:
    sel = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    sel.fit(X)
    kept = X.columns[sel.get_support()].tolist()
    print(f"Variance filter (>{VARIANCE_THRESHOLD}): "
          f"removed {len(X.columns) - len(kept)}, kept {len(kept)}")
    return X[kept]


def _structural_filter(X: pd.DataFrame) -> pd.DataFrame:
    """Remove CTD_*_C1/C2/C3: structurally collinear and redundant with AAC."""
    ctd_c = re.compile(r"^CTD_\w+_C[123]$")
    to_drop = [c for c in X.columns if ctd_c.match(c)]
    kept = [c for c in X.columns if c not in to_drop]
    print(f"Structural filter (CTD_C): removed {len(to_drop)}, kept {len(kept)}")
    return X[kept]


def _correlation_filter(X: pd.DataFrame) -> tuple:
    """Remove one feature from each pair with |r| > CORR_THRESHOLD.
    Drops the feature with higher mean absolute correlation to all others.
    Returns (filtered_X, full_corr_matrix_before_filtering).
    """
    corr = X.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        for partner in upper.index[upper[col] > CORR_THRESHOLD].tolist():
            if col in to_drop or partner in to_drop:
                continue
            mean_col     = corr[col].drop(index=col).mean()
            mean_partner = corr[partner].drop(index=partner).mean()
            to_drop.add(col if mean_col > mean_partner else partner)

    kept = [c for c in X.columns if c not in to_drop]
    print(f"Correlation filter (|r|>{CORR_THRESHOLD}): "
          f"removed {len(to_drop)}, kept {len(kept)}")
    return X[kept], corr



# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
def _plot_heatmap(corr: pd.DataFrame, title: str, filename: str):
    """Clustered heatmap of absolute Pearson correlation matrix.

    Colorbar positioned to the right of the heatmap, outside the dendrogram
    area, to avoid overlap.
    """
    n = corr.shape[0]
    cell_size = max(0.12, min(0.20, 13.0 / n))
    figsize   = (n * cell_size + 3, n * cell_size + 1)
    font_size = max(4, min(7, int(110 / n)))

    g = sns.clustermap(
        corr,
        cmap="coolwarm",
        vmin=0, vmax=1,
        figsize=figsize,
        xticklabels=True,
        yticklabels=True,
        linewidths=0,
        # Place colorbar at the right edge, vertically centered
        cbar_pos=(1.01, 0.35, 0.018, 0.30),
        cbar_kws={"label": "|Pearson r|"},
        dendrogram_ratio=(0.12, 0.12),
    )

    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), fontsize=font_size, rotation=90
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), fontsize=font_size, rotation=0
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.fig.suptitle(title, y=1.005, fontsize=9)

    path = os.path.join(OUT_DIR, filename)
    g.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(g.fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load
    X, y = _load_data()
    n_initial = X.shape[1]

    # 2. Variance filter
    print("\n--- Step 1: Variance filter ---")
    X_var = _variance_filter(X)
    n_after_var = X_var.shape[1]

    # 3. Structural filter
    print("\n--- Step 2: Structural filter ---")
    X_struct = _structural_filter(X_var)
    n_after_struct = X_struct.shape[1]

    # 4. Correlation filter
    print("\n--- Step 3: Correlation filter ---")
    print("  Computing correlation matrix (before)...")
    corr_before = X_struct.corr(method="pearson").abs()

    print("  Plotting heatmap (before)...")
    _plot_heatmap(
        corr_before,
        f"Pairwise Absolute Pearson Correlation of Physicochemical Features Before Filtering (n = {n_after_struct})",
        "fig_correlation_heatmap_before.png",
    )

    X_final, _ = _correlation_filter(X_struct)
    n_final = X_final.shape[1]

    print("  Computing correlation matrix (after Pearson filter)...")
    corr_after = X_final.corr(method="pearson").abs()

    print("  Plotting heatmap (after)...")
    _plot_heatmap(
        corr_after,
        f"Pairwise Absolute Pearson Correlation of Physicochemical Features After Filtering (n = {n_final})",
        "fig_correlation_heatmap_after.png",
    )

    # 5. RF importance ranking (report only)
    print("\n--- Step 4: RF importance ranking ---")
    X_train, _, y_train, _ = train_test_split(
        X_final, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = RobustScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    print(f"Fitting RF ({RF_N_ESTIMATORS} trees) for importance ranking...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_sc, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    importance = pd.Series(rf.feature_importances_, index=X_train_sc.columns).sort_values(ascending=False)

    # 6. Save selected features
    with open(SELECTED_FEATURES_PATH, "w") as f:
        for feat in X_final.columns:
            f.write(feat + "\n")
    print(f"\nSelected features -> {SELECTED_FEATURES_PATH}  ({n_final} features)")

    # 7. Text report
    groups = {"GlobalDesc": 0, "AAC": 0, "CTD_T": 0, "CTD_D": 0}
    for feat in X_final.columns:
        if feat.startswith("AAC_"):
            groups["AAC"] += 1
        elif re.match(r"^CTD_\w+_T\d+$", feat):
            groups["CTD_T"] += 1
        elif re.match(r"^CTD_\w+_D\d+", feat):
            groups["CTD_D"] += 1
        else:
            groups["GlobalDesc"] += 1

    report_lines = [
        "Feature Selection Report",
        "=" * 60,
        "",
        f"Initial features              : {n_initial}",
        f"Samples                       : {X.shape[0]}",
        f"  Positive (AMP)              : {int(y.sum())}",
        f"  Negative (non-AMP)          : {int((y == 0).sum())}",
        "",
        f"Step 1 — Variance filter      : threshold = {VARIANCE_THRESHOLD}",
        f"  Removed                     : {n_initial - n_after_var}",
        f"  Remaining                   : {n_after_var}",
        "",
        f"Step 2 — Structural filter    : CTD_*_C1/C2/C3 removed",
        f"  Removed                     : {n_after_var - n_after_struct}",
        f"  Remaining                   : {n_after_struct}",
        f"  Rationale: C1+C2+C3=1 (Dubchak et al. 1995) and CTD_C",
        f"  features are linear combinations of AAC descriptors.",
        "",
        f"Step 3 — Correlation filter   : |r| > {CORR_THRESHOLD}",
        f"  Removed                     : {n_after_struct - n_final}",
        f"  Remaining                   : {n_final}",
        "",
        f"Final selected features       : {n_final}",
        f"  GlobalDesc                  : {groups['GlobalDesc']}",
        f"  AAC                         : {groups['AAC']}",
        f"  CTD Transition (T)          : {groups['CTD_T']}",
        f"  CTD Distribution (D)        : {groups['CTD_D']}",
        "",
        "Top 20 features by RF importance (RobustScaler, 200 trees):",
    ]
    for i, (feat, val) in enumerate(importance.head(20).items(), 1):
        report_lines.append(f"  {i:>2}. {feat:<42} {val:.6f}")

    txt_path = os.path.join(OUT_DIR, "feature_selection_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Report -> {txt_path}")
    print("\n--- Feature analysis complete ---")


if __name__ == "__main__":
    main()
