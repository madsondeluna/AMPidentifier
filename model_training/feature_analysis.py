# model_training/feature_analysis.py
#
# Phase 2: correlation analysis and feature selection.
#
# Pipeline:
#   1. Load sequences and extract the full feature set (~577 features).
#   2. Variance threshold: remove features with variance <= VARIANCE_THRESHOLD.
#   3. Pearson correlation filter: remove one of each pair with |r| > CORR_THRESHOLD.
#      The feature with lower mean absolute correlation to the rest of the set is kept.
#   4. Random Forest importance ranking on the filtered feature set.
#   5. Save:
#       - model_training/data/selected_features.txt  (feature names, one per line)
#       - model_training/feature_analysis/fig_variance_distribution.png
#       - model_training/feature_analysis/fig_correlation_heatmap_before.png
#       - model_training/feature_analysis/fig_correlation_heatmap_after.png
#       - model_training/feature_analysis/fig_correlation_histogram.png
#       - model_training/feature_analysis/fig_feature_importance.png
#       - model_training/feature_analysis/feature_selection_report.txt
#
# Run from project root:
#   python -m model_training.feature_analysis

import os
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
from sklearn.preprocessing import StandardScaler

from amp_identifier.feature_extraction import calculate_physicochemical_features
from amp_identifier.data_io import load_fasta_sequences

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR    = "model_training/data"
OUT_DIR     = "model_training/feature_analysis"
POSITIVE_FILE = os.path.join(DATA_DIR, "positive_sequences.fasta")
NEGATIVE_FILE = os.path.join(DATA_DIR, "negative_sequences.fasta")
SELECTED_FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.txt")

RANDOM_STATE     = 42
TEST_SIZE        = 0.2
VARIANCE_THRESHOLD = 0.001  # remove features with variance <= this value
CORR_THRESHOLD   = 0.95     # remove one of each pair with |r| > this value
RF_N_ESTIMATORS  = 200      # quick RF for importance ranking
TOP_N_IMPORTANCE = 40       # how many top features to plot

FIGURE_DPI = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_data():
    print("Loading sequences...")
    pos_seqs, pos_ids = load_fasta_sequences(POSITIVE_FILE)
    neg_seqs, neg_ids = load_fasta_sequences(NEGATIVE_FILE)
    sequences = pos_seqs + neg_seqs
    ids       = pos_ids  + neg_ids
    labels    = [1] * len(pos_seqs) + [0] * len(neg_seqs)
    print(f"  Positive: {len(pos_seqs)}  Negative: {len(neg_seqs)}")

    print("Extracting features (~577 per sequence)...")
    t0 = time.time()
    features_df = calculate_physicochemical_features(sequences, ids)
    features_df["label"] = labels
    print(f"  Done in {time.time()-t0:.1f}s  Shape: {features_df.shape}")

    X = features_df.drop(columns=["ID", "sequence", "label"]).fillna(0)
    y = features_df["label"]
    return X, y


def _variance_filter(X: pd.DataFrame) -> pd.DataFrame:
    sel = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    sel.fit(X)
    kept = X.columns[sel.get_support()].tolist()
    removed = len(X.columns) - len(kept)
    print(f"Variance filter (threshold={VARIANCE_THRESHOLD}): "
          f"removed {removed}, kept {len(kept)}")
    return X[kept]


def _correlation_filter(X: pd.DataFrame) -> tuple:
    """Remove one feature from each pair with |r| > CORR_THRESHOLD.

    Strategy: for each correlated pair, drop the feature with higher mean
    absolute correlation to all other features (more redundant overall).
    """
    corr = X.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > CORR_THRESHOLD].tolist()
        for partner in correlated:
            if col in to_drop or partner in to_drop:
                continue
            # drop the one with higher mean abs correlation to all others
            mean_col     = corr[col].drop(index=col).mean()
            mean_partner = corr[partner].drop(index=partner).mean()
            to_drop.add(col if mean_col > mean_partner else partner)

    kept = [c for c in X.columns if c not in to_drop]
    print(f"Correlation filter (threshold={CORR_THRESHOLD}): "
          f"removed {len(to_drop)}, kept {len(kept)}")
    return X[kept], corr


def _rf_importance(X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
    print(f"Fitting RF (n_estimators={RF_N_ESTIMATORS}) for importance ranking...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    return pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_variance_distribution(X_full: pd.DataFrame, X_after_var: pd.DataFrame):
    variances = X_full.var()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(variances, bins=80, color="#4C72B0", edgecolor="none")
    axes[0].axvline(VARIANCE_THRESHOLD, color="#DD4444", linestyle="--",
                    label=f"threshold = {VARIANCE_THRESHOLD}")
    axes[0].set_xlabel("Variance")
    axes[0].set_ylabel("Number of features")
    axes[0].set_title(f"Feature variance distribution (n={len(X_full.columns)})")
    axes[0].legend()

    removed = set(X_full.columns) - set(X_after_var.columns)
    kept_var = variances[X_after_var.columns]
    axes[1].hist(kept_var, bins=60, color="#55A868", edgecolor="none")
    axes[1].set_xlabel("Variance")
    axes[1].set_ylabel("Number of features")
    axes[1].set_title(f"After variance filter (kept {len(X_after_var.columns)}, "
                      f"removed {len(removed)})")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_variance_distribution.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_correlation_heatmap(corr: pd.DataFrame, title: str, filename: str):
    """Clustermap of the absolute Pearson correlation matrix."""
    n = corr.shape[0]
    # With many features, hide individual tick labels
    show_labels = n <= 60

    figsize = (max(10, n * 0.07), max(8, n * 0.07))
    g = sns.clustermap(
        corr,
        cmap="coolwarm",
        vmin=0, vmax=1,
        figsize=figsize,
        xticklabels=show_labels,
        yticklabels=show_labels,
        linewidths=0 if n > 60 else 0.3,
        cbar_kws={"shrink": 0.5, "label": "|Pearson r|"},
    )
    g.fig.suptitle(title, y=1.01, fontsize=11)
    path = os.path.join(OUT_DIR, filename)
    g.savefig(path, dpi=FIGURE_DPI)
    plt.close(g.fig)
    print(f"  Saved: {path}")


def _plot_correlation_histogram(corr_before: pd.DataFrame, corr_after: pd.DataFrame):
    """Distribution of pairwise |r| values before and after filtering."""
    def _upper_triangle(corr):
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        return corr.values[mask]

    vals_before = _upper_triangle(corr_before)
    corr_after_full = corr_after.corr(method="pearson").abs()
    vals_after  = _upper_triangle(corr_after_full)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    axes[0].hist(vals_before, bins=80, color="#4C72B0", edgecolor="none")
    axes[0].axvline(CORR_THRESHOLD, color="#DD4444", linestyle="--",
                    label=f"threshold = {CORR_THRESHOLD}")
    axes[0].set_xlabel("|Pearson r|")
    axes[0].set_ylabel("Pairs")
    axes[0].set_title(f"Before filtering (n={corr_before.shape[0]} features)")
    axes[0].legend()

    axes[1].hist(vals_after, bins=60, color="#55A868", edgecolor="none")
    axes[1].axvline(CORR_THRESHOLD, color="#DD4444", linestyle="--",
                    label=f"threshold = {CORR_THRESHOLD}")
    axes[1].set_xlabel("|Pearson r|")
    axes[1].set_ylabel("Pairs")
    axes[1].set_title(f"After filtering (n={corr_after.shape[0]} features)")
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_correlation_histogram.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_feature_importance(importance: pd.Series):
    top = importance.head(TOP_N_IMPORTANCE)
    # Color by feature group
    def _color(name):
        if name.startswith("AAC_"):   return "#4C72B0"
        if name.startswith("DPC_"):   return "#DD8452"
        if name.startswith("CTD_"):   return "#55A868"
        return "#8172B2"

    colors = [_color(n) for n in top.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top)), top.values[::-1], color=colors[::-1], edgecolor="none")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1], fontsize=8)
    ax.set_xlabel("Mean decrease in impurity (normalized)")
    ax.set_title(f"Top {TOP_N_IMPORTANCE} features by RF importance")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="AAC"),
        Patch(facecolor="#DD8452", label="DPC"),
        Patch(facecolor="#55A868", label="CTD"),
        Patch(facecolor="#8172B2", label="GlobalDesc"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_feature_importance.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_feature_group_summary(X_full, X_final):
    """Bar chart showing how many features each group contributes before/after."""
    groups = ["GlobalDesc", "AAC", "DPC", "CTD"]

    def _count(cols, prefix):
        if prefix == "GlobalDesc":
            return sum(1 for c in cols if not c.startswith(("AAC_","DPC_","CTD_")))
        return sum(1 for c in cols if c.startswith(f"{prefix}_"))

    before = [_count(X_full.columns, g) for g in groups]
    after  = [_count(X_final.columns, g) for g in groups]

    x = np.arange(len(groups))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, before, w, label="Before selection", color="#4C72B0", edgecolor="none")
    ax.bar(x + w/2, after,  w, label="After selection",  color="#55A868", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Number of features")
    ax.set_title("Feature counts per group before and after selection")
    ax.legend()
    for xi, (b, a) in zip(x, zip(before, after)):
        ax.text(xi - w/2, b + 1, str(b), ha="center", va="bottom", fontsize=8)
        ax.text(xi + w/2, a + 1, str(a), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_feature_group_summary.png")
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report_lines = ["Feature Selection Report", "=" * 60, ""]

    # 1. Load
    X, y = _load_data()
    n_initial = X.shape[1]
    report_lines += [
        f"Initial features          : {n_initial}",
        f"Samples (total)           : {X.shape[0]}",
        f"  Positive (AMP)          : {int(y.sum())}",
        f"  Negative (non-AMP)      : {int((y == 0).sum())}",
        "",
    ]

    # 2. Variance filter
    print("\n--- Step 1: Variance filter ---")
    X_var = _variance_filter(X)
    n_after_var = X_var.shape[1]
    report_lines += [
        f"Variance threshold        : {VARIANCE_THRESHOLD}",
        f"Removed by variance filter: {n_initial - n_after_var}",
        f"Features after variance   : {n_after_var}",
        "",
    ]

    print("  Plotting variance distribution...")
    _plot_variance_distribution(X, X_var)

    # 3. Correlation filter
    print("\n--- Step 2: Correlation filter ---")
    print("  Computing correlation matrix before filtering...")
    corr_before = X_var.corr(method="pearson").abs()

    print("  Plotting correlation heatmap (before)...")
    _plot_correlation_heatmap(
        corr_before,
        f"Absolute Pearson correlation — before filtering (n={n_after_var})",
        "fig_correlation_heatmap_before.png",
    )

    X_final, corr_before_ret = _correlation_filter(X_var)
    n_final = X_final.shape[1]
    report_lines += [
        f"Correlation threshold     : {CORR_THRESHOLD}",
        f"Removed by corr filter    : {n_after_var - n_final}",
        f"Features after corr filter: {n_final}",
        "",
    ]

    print("  Plotting correlation heatmap (after)...")
    corr_after = X_final.corr(method="pearson").abs()
    _plot_correlation_heatmap(
        corr_after,
        f"Absolute Pearson correlation — after filtering (n={n_final})",
        "fig_correlation_heatmap_after.png",
    )

    print("  Plotting correlation histogram...")
    _plot_correlation_histogram(corr_before_ret, X_final)

    # 4. RF importance
    print("\n--- Step 3: RF importance ranking ---")
    X_train, _, y_train, _ = train_test_split(
        X_final, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    importance = _rf_importance(X_train_sc, y_train)
    _plot_feature_importance(importance)

    # 5. Group summary
    print("  Plotting feature group summary...")
    _plot_feature_group_summary(X, X_final)

    # 6. Save selected features
    with open(SELECTED_FEATURES_PATH, "w") as f:
        for feat in X_final.columns:
            f.write(feat + "\n")
    print(f"\nSelected features saved -> {SELECTED_FEATURES_PATH}  ({n_final} features)")

    # 7. Report
    report_lines += [
        f"Final selected features   : {n_final}",
        "",
        "Top 20 features by RF importance:",
    ]
    for i, (feat, val) in enumerate(importance.head(20).items(), 1):
        report_lines.append(f"  {i:>2}. {feat:<40} {val:.6f}")

    report_lines += [
        "",
        "Feature group breakdown (final):",
    ]
    groups = {"GlobalDesc": [], "AAC": [], "DPC": [], "CTD": []}
    for feat in X_final.columns:
        if feat.startswith("AAC_"):   groups["AAC"].append(feat)
        elif feat.startswith("DPC_"): groups["DPC"].append(feat)
        elif feat.startswith("CTD_"): groups["CTD"].append(feat)
        else:                         groups["GlobalDesc"].append(feat)
    for g, feats in groups.items():
        report_lines.append(f"  {g:<12}: {len(feats)}")

    txt_path = os.path.join(OUT_DIR, "feature_selection_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Report saved -> {txt_path}")
    print("\n--- Feature analysis complete ---")


if __name__ == "__main__":
    main()
