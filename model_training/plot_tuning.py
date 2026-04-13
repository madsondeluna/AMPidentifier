# model_training/plot_tuning.py
#
# Generates publication-quality figures for tuning and evaluation results.
# Style: Nature Publishing Group (NPG) aesthetics — minimal, no grid,
#        sentence-case titles, legends outside axes, NPG colour palette.
#
# Memory strategy: model inference is delegated to collect_outputs.py,
# which runs each model in an isolated subprocess and saves .npz files.
# This script only loads the .npz files to generate figures.
#
# Figures generated in model_training/tuned_model/figures/:
#   fig01_roc_curves.png
#   fig02_confusion_matrices.png
#   fig03_calibration.png
#   fig04_feature_importance.png
#   fig05_cv_score_distribution.png
#   fig06_top10_candidates.png
#   fig07_hyperparam_rf.png  ...  fig10_hyperparam_xgb.png
#   fig11_metrics_comparison.png
#   fig12_precision_recall.png
#   fig13_det_curves.png
#   fig14_threshold_sensitivity.png
#
# Run from project root:
#   python -m model_training.plot_tuning

import gc
import os
import subprocess
import sys
import warnings

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from scipy.ndimage import uniform_filter1d
from scipy.stats import mode as scipy_mode
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    det_curve, f1_score, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve, auc,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Nature NPG rcParams
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":           8,
    "axes.titlesize":      9,
    "axes.labelsize":      8,
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
    "legend.fontsize":     7,
    "legend.frameon":      False,
    "legend.borderpad":    0.4,
    "figure.dpi":          300,
    "savefig.dpi":         300,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.spines.left":    True,
    "axes.spines.bottom":  True,
    "axes.grid":           False,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.size":    3,
    "ytick.major.size":    3,
    "xtick.minor.size":    1.5,
    "ytick.minor.size":    1.5,
    "xtick.major.width":   0.6,
    "ytick.major.width":   0.6,
    "axes.linewidth":      0.6,
    "lines.linewidth":     1.0,
    "patch.linewidth":     0.4,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
})

# Nature column widths (mm → in): single 89mm = 3.50in; double 183mm = 7.20in
COL1 = 3.50
COL2 = 7.20

# Nature Publishing Group (NPG) palette — ggsci::pal_npg
COLORS = {
    "RF":    "#4DBBD5",   # sky blue
    "SVM":   "#E64B35",   # cinnabar
    "GB":    "#00A087",   # observatory green
    "XGB":   "#3C5488",   # san marino navy
    "MLP":   "#F39B7F",   # tacao salmon
    "STACK": "#8491B4",   # wistful lavender
    "DEEP":  "#7E6148",   # spicy mix brown
}
ALPHA    = 0.85
ENS_CLR  = "#444444"

BEST_MODEL = "GB"   # highest AUC-ROC and MCC across all tuned models

# Distinct linestyles — one per model so colour + style give two visual channels
LINESTYLES = {
    "RF":    "-",
    "SVM":   (0, (5, 1)),         # long dash
    "GB":    "-.",
    "XGB":   ":",
    "MLP":   (0, (3, 1, 1, 1)),   # dash-dot-dot
    "STACK": (0, (1, 1)),         # dense dots
    "DEEP":  "--",
}

# Per-metric colours for threshold-sensitivity plot
METRIC_COLORS = {
    "MCC":       "#2c3e50",
    "F1":        "#c0392b",
    "Precision": "#2980b9",
    "Recall":    "#27ae60",
}

OUTDIR    = "model_training/tuned_model/figures"
TUNED_DIR = "model_training/tuned_model"
NPZ_DIR   = os.path.join(TUNED_DIR, "outputs")

DATA_DIR      = "model_training/data"
SEL_FEAT_PATH = os.path.join(DATA_DIR, "selected_features.txt")

TREE_MODELS     = {"RF", "GB", "XGB"}
CLASSICAL_ORDER = ["RF", "SVM", "GB", "XGB", "MLP", "STACK"]
ALL_MODELS      = CLASSICAL_ORDER + ["DEEP"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def savefig(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    fig.savefig(path)
    print(f"  Saved -> {path}")
    plt.close(fig)
    gc.collect()


def load_selected_features() -> list:
    with open(SEL_FEAT_PATH) as f:
        return [l.strip() for l in f if l.strip()]


def load_cv_results(model_name: str):
    path = os.path.join(TUNED_DIR, f"cv_results_{model_name.lower()}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def rolling_mean(y, w=5):
    return uniform_filter1d(y.astype(float), size=w, mode="nearest")


def _leg_outside(ax, loc="upper left", bbox=(1.02, 1.0), **kw):
    """Place legend outside axes to the right."""
    return ax.legend(loc=loc, bbox_to_anchor=bbox,
                     borderaxespad=0, **kw)


# ---------------------------------------------------------------------------
# Load pre-computed npz outputs (produced by collect_outputs.py)
# ---------------------------------------------------------------------------
def _metrics_from_arrays(proba, y_test, thresh):
    y_pred = (proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "accuracy":    float(accuracy_score(y_test, y_pred)),
        "precision":   float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_test, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1":          float(f1_score(y_test, y_pred, zero_division=0)),
        "mcc":         float(matthews_corrcoef(y_test, y_pred)),
        "auc_roc":     float(roc_auc_score(y_test, proba)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def load_npz_output(name: str) -> dict | None:
    path = os.path.join(NPZ_DIR, f"{name.lower()}_outputs.npz")
    if not os.path.exists(path):
        return None
    data       = np.load(path)
    proba      = data["proba"].astype(np.float64)
    y_test     = data["y_test"].astype(int)
    thresh     = float(data["threshold"][0])
    importance = data["importance"]
    y_pred     = (proba >= thresh).astype(int)
    imp        = importance if importance.size > 0 else None
    return {
        "proba":      proba,
        "y_test":     y_test,
        "threshold":  thresh,
        "y_pred":     y_pred,
        "metrics":    _metrics_from_arrays(proba, y_test, thresh),
        "importance": imp,
        "stds":       np.zeros_like(imp) if imp is not None else None,
    }


def collect_all_outputs(model_names: list) -> dict:
    os.makedirs(NPZ_DIR, exist_ok=True)
    for name in model_names:
        npz_path = os.path.join(NPZ_DIR, f"{name.lower()}_outputs.npz")
        if os.path.exists(npz_path):
            print(f"  {name.upper()}: cached, skipping subprocess.")
            continue
        print(f"  {name.upper()}: collecting outputs...")
        result = subprocess.run(
            [sys.executable, "-m", "model_training.collect_outputs", name.lower()],
        )
        if result.returncode != 0:
            print(f"  {name.upper()}: subprocess failed (exit {result.returncode}).")

    outputs = {}
    for name in model_names:
        out = load_npz_output(name)
        if out is None:
            print(f"  {name.upper()}: no output file, skipping.")
        outputs[name] = out
    return outputs


# ---------------------------------------------------------------------------
# Fig 01 — ROC curves
# ---------------------------------------------------------------------------
def fig01_roc_curves(outputs: dict):
    fig, ax = plt.subplots(figsize=(COL1 + 1.6, COL1))

    for name, out in outputs.items():
        if out is None:
            continue
        fpr, tpr, _ = roc_curve(out["y_test"], out["proba"])
        ra      = auc(fpr, tpr)
        is_best = name == BEST_MODEL
        lw      = 1.8 if is_best else 1.0
        al      = 1.0 if is_best else ALPHA * 0.80
        ls      = LINESTYLES.get(name, "-")
        ax.plot(fpr, tpr, color=COLORS[name], linewidth=lw, linestyle=ls,
                alpha=al, label=f"{name}  AUC={ra:.3f}",
                zorder=5 if is_best else 3)

        if is_best:
            proba  = out["proba"]
            y_test = out["y_test"]
            thresh = out["threshold"]
            y_pred = (proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            op_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            op_tpr = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            ax.scatter([op_fpr], [op_tpr], marker="*", color=COLORS[name],
                       s=55, zorder=6, linewidths=0)

    ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.7, linestyle=":")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves on Held-Out Test Set")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    _leg_outside(ax, title="Model  AUC", title_fontsize=6.5)

    fig.tight_layout()
    fig.subplots_adjust(right=0.74)
    savefig(fig, "fig01_roc_curves.png")


# ---------------------------------------------------------------------------
# Fig 02 — Confusion matrices with per-panel colour scale
# ---------------------------------------------------------------------------
def fig02_confusion_matrices(outputs: dict):
    names = [n for n, o in outputs.items() if o is not None]
    n     = len(names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.35 * ncols, 2.6 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, name in enumerate(names):
        ax     = axes_flat[idx]
        out    = outputs[name]
        y_test = out["y_test"]
        y_pred = out["y_pred"]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        mat   = np.array([[tn, fp], [fn, tp]], dtype=float)

        color = COLORS[name]
        cmap  = mpl.colors.LinearSegmentedColormap.from_list(
            "cm", ["#f8f8f8", color], N=256
        )
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=mat.max(), aspect="equal")

        # Colorbar (thin, right side of each subplot)
        cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
        cbar.ax.tick_params(labelsize=5.5)
        cbar.outline.set_linewidth(0.4)

        cell_labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                tc  = "white" if val > mat.max() * 0.52 else "#222222"
                ax.text(j, i, f"{cell_labels[i][j]}\n{int(val)}",
                        ha="center", va="center", fontsize=7,
                        color=tc, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg.", "Pos."], fontsize=6.5)
        ax.set_yticklabels(["Neg.", "Pos."] if idx % ncols == 0 else ["", ""],
                           fontsize=6.5)
        ax.set_xlabel("Predicted Label", fontsize=6.5)
        if idx % ncols == 0:
            ax.set_ylabel("True Label", fontsize=6.5)

        mcc = matthews_corrcoef(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        ax.set_title(f"{name} - Acc={acc:.3f}, MCC={mcc:.3f}", fontsize=7.5)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Confusion Matrices on Held-Out Test Set", fontsize=9, y=1.01)
    fig.tight_layout()
    savefig(fig, "fig02_confusion_matrices.png")


# ---------------------------------------------------------------------------
# Fig 03 — Calibration curves
# ---------------------------------------------------------------------------
def fig03_calibration(outputs: dict):
    fig, ax = plt.subplots(figsize=(COL1 + 1.6, COL1))

    for name, out in outputs.items():
        if out is None:
            continue
        frac_pos, mean_pred = calibration_curve(
            out["y_test"], out["proba"], n_bins=10
        )
        is_best = name == BEST_MODEL
        lw      = 1.6 if is_best else 1.0
        ls      = LINESTYLES.get(name, "-")
        mrk     = "D" if is_best else "o"
        ms      = 3.5 if is_best else 2.5
        al      = 1.0 if is_best else ALPHA * 0.80
        ax.plot(mean_pred, frac_pos, color=COLORS[name], linewidth=lw,
                linestyle=ls, marker=mrk, markersize=ms, alpha=al,
                label=name, zorder=5 if is_best else 3)

    ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.7,
            linestyle=":", label="Ideal")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Probability Calibration (Reliability Diagram)")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    _leg_outside(ax)

    fig.tight_layout()
    fig.subplots_adjust(right=0.74)
    savefig(fig, "fig03_calibration.png")


# ---------------------------------------------------------------------------
# Fig 04 — Feature importance (RF, GB, XGB) — 3-column layout, no blank panels
# ---------------------------------------------------------------------------
def fig04_feature_importance(outputs: dict):
    feat_names = load_selected_features()
    # Only tree models have genuine feature_importances_ (non-zero arrays)
    target = {n: o for n, o in outputs.items()
              if o is not None
              and o["importance"] is not None
              and np.sum(o["importance"]) > 0}

    if not target:
        print("  Skipping fig04: no importance data available.")
        return

    n     = len(target)
    top_k = 20
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    # Height computed from feature count to avoid excess whitespace
    row_h = max(3.0, top_k * 0.16 + 1.0)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(COL2, row_h * nrows),
                             sharey=False, squeeze=False)
    axes_flat = axes.flatten()

    for idx, (name, out) in enumerate(target.items()):
        ax         = axes_flat[idx]
        color      = COLORS[name]
        importance = out["importance"]
        order      = np.argsort(importance)[-top_k:]
        names_s    = [feat_names[i][:22] for i in order]
        imp_s      = importance[order]

        y_pos = np.arange(len(names_s))
        ax.barh(y_pos, imp_s, color=color, alpha=ALPHA, linewidth=0,
                height=0.72)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names_s, fontsize=5.2)
        ax.set_xlabel("Mean Decrease in Impurity", fontsize=7)
        ax.set_title(f"{name}: Top {top_k} Features")
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.axvline(0, color="#888888", linewidth=0.5)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Feature Importance: Top {top_k} of {len(feat_names)} Selected Features",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(fig, "fig04_feature_importance.png")


# ---------------------------------------------------------------------------
# Fig 05 — CV score distribution
# ---------------------------------------------------------------------------
def fig05_cv_score_distribution():
    cv_names = ["rf", "gb", "svm", "xgb", "mlp", "stack"]
    data = {}
    for m in cv_names:
        cv = load_cv_results(m)
        if cv is not None and "mean_test_score" in cv.columns:
            data[m.upper()] = cv[["mean_test_score", "std_test_score"]]

    if not data:
        print("  Skipping fig05: no cv_results CSVs found.")
        return

    fig, ax = plt.subplots(figsize=(COL2 * 0.6, 2.9))
    positions = np.arange(len(data))
    rng = np.random.default_rng(42)

    for pos, (model, df) in zip(positions, data.items()):
        color  = COLORS[model]
        scores = df["mean_test_score"].values
        stds   = df["std_test_score"].values
        jitter = rng.uniform(-0.13, 0.13, size=len(scores))

        ax.errorbar(np.full_like(scores, pos) + jitter, scores,
                    yerr=stds, fmt="none", ecolor=color, alpha=0.18,
                    elinewidth=0.5, capsize=0)
        ax.scatter(np.full_like(scores, pos) + jitter, scores,
                   color=color, alpha=0.42, s=6, linewidths=0, zorder=3)

        med = np.median(scores)
        ax.hlines(med, pos - 0.28, pos + 0.28,
                  color=color, linewidth=1.8, zorder=4)

        best_idx = np.argmax(scores)
        ax.scatter([pos + jitter[best_idx]], [scores[best_idx]],
                   color=color, s=26, marker="*", zorder=5, linewidths=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(list(data.keys()))
    ax.set_ylabel("CV AUC-ROC (Mean ± SD, 5 Folds)")
    ax.set_title("CV AUC-ROC Distribution Across 50 Hyperparameter Candidates\n"
                 "(Line: Median; Star: Best Candidate)")
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    fig.tight_layout()
    savefig(fig, "fig05_cv_score_distribution.png")


# ---------------------------------------------------------------------------
# Fig 06 — Top 10 candidates per model
# ---------------------------------------------------------------------------
def fig06_top10_candidates():
    cv_names = ["rf", "gb", "svm", "xgb", "mlp", "stack"]
    available = {}
    for m in cv_names:
        cv = load_cv_results(m)
        if cv is not None and "mean_test_score" in cv.columns:
            available[m.upper()] = cv

    if not available:
        print("  Skipping fig06: no cv_results CSVs found.")
        return

    n     = len(available)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(COL2, nrows * 2.5),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, (model, cv) in enumerate(available.items()):
        ax    = axes_flat[idx]
        color = COLORS[model]
        top10 = cv.nlargest(10, "mean_test_score").reset_index(drop=True)
        y_pos = np.arange(len(top10))

        ax.barh(y_pos, top10["mean_test_score"],
                xerr=top10["std_test_score"],
                color=color, alpha=ALPHA * 0.65, linewidth=0,
                error_kw={"elinewidth": 0.6, "capsize": 2, "ecolor": "#666666"},
                height=0.72)
        ax.barh([0], [top10.loc[0, "mean_test_score"]],
                color=color, alpha=ALPHA, linewidth=0, height=0.72)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"#{i+1}" for i in range(len(top10))], fontsize=6.5)
        ax.set_xlabel("CV AUC-ROC", fontsize=7)
        ax.set_title(f"{model}: Top 10 Candidates")
        ax.invert_yaxis()
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        lo = max(0, top10["mean_test_score"].min() - 0.008)
        hi = min(1.0, top10["mean_test_score"].max() + 0.010)
        ax.set_xlim(lo, hi)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Top 10 Hyperparameter Combinations by CV AUC-ROC", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "fig06_top10_candidates.png")


# ---------------------------------------------------------------------------
# Fig 07–10 — Hyperparameter scatter plots with rolling mean
# ---------------------------------------------------------------------------
def fig_hyperparam(model_name, param_cols, log_params, fig_num):
    cv = load_cv_results(model_name)
    if cv is None:
        print(f"  Skipping fig{fig_num:02d}: cv_results_{model_name}.csv not found.")
        return

    color     = COLORS[model_name.upper()]
    score_col = "mean_test_score"
    std_col   = "std_test_score"
    best_idx  = cv[score_col].idxmax()

    numeric_params = []
    for p in param_cols:
        col = f"param_{p}"
        if col in cv.columns:
            vals = pd.to_numeric(cv[col], errors="coerce")
            if vals.notna().sum() >= 5:
                numeric_params.append(p)

    if not numeric_params:
        print(f"  Skipping fig{fig_num:02d}: no numeric params found.")
        return

    n = len(numeric_params)
    fig, axes = plt.subplots(1, n, figsize=(COL1 * min(n, 3), 2.4),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, param in zip(axes_flat, numeric_params):
        col  = f"param_{param}"
        vals = pd.to_numeric(cv[col], errors="coerce")
        mask = vals.notna()
        x    = vals[mask].values
        y    = cv.loc[mask, score_col].values
        yerr = cv.loc[mask, std_col].values

        sort_idx = np.argsort(x)
        x_s, y_s = x[sort_idx], y[sort_idx]

        ax.errorbar(x, y, yerr=yerr, fmt="o", color=color,
                    alpha=0.42, markersize=3.2, linewidth=0,
                    capsize=1.5, elinewidth=0.5, ecolor=color, zorder=2)

        if len(x_s) >= 8:
            w_rm = max(5, len(x_s) // 8)
            y_rm = rolling_mean(y_s, w=w_rm)
            ax.plot(x_s, y_rm, color=color, linewidth=1.3, alpha=0.55, zorder=3)

        if mask[best_idx]:
            ax.scatter([vals[best_idx]], [cv.loc[best_idx, score_col]],
                       color=color, s=38, zorder=5, linewidths=0, marker="*")

        if param in log_params:
            ax.set_xscale("log")

        label = param.replace("_", " ").title()
        ax.set_xlabel(label)
        ax.set_ylabel("CV AUC-ROC")
        ax.set_title(f"{model_name.upper()}: {label} vs. CV AUC-ROC",
                     fontsize=8.5)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Hide unused axes if n < len(axes_flat)
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"{model_name.upper()} Hyperparameter Search "
        f"(Line: Rolling Mean; Star: Selected Optimum)",
        fontsize=9, y=1.06,
    )
    fig.tight_layout()
    savefig(fig, f"fig{fig_num:02d}_hyperparam_{model_name.lower()}.png")


# ---------------------------------------------------------------------------
# Fig 11 — All-model metrics comparison
# ---------------------------------------------------------------------------
def fig11_metrics_comparison(outputs: dict):
    metrics_def = [
        ("accuracy",    "Accuracy"),
        ("precision",   "Precision"),
        ("recall",      "Sensitivity"),
        ("specificity", "Specificity"),
        ("f1",          "F1"),
        ("mcc",         "MCC"),
        ("auc_roc",     "AUC-ROC"),
    ]

    valid   = {n: o for n, o in outputs.items() if o is not None}
    results = {n: o["metrics"] for n, o in valid.items()}
    y_test  = next(iter(valid.values()))["y_test"]

    all_preds = {n: o["y_pred"] for n, o in valid.items()}
    all_prob  = {n: o["proba"]  for n, o in valid.items()}

    if len(all_preds) > 1:
        preds_mat      = np.column_stack(list(all_preds.values()))
        ens_pred, _    = scipy_mode(preds_mat, axis=1, keepdims=True)
        ens_pred       = ens_pred.ravel().astype(int)
        ens_proba      = np.mean(list(all_prob.values()), axis=0)
        tn, fp, fn, tp = confusion_matrix(y_test, ens_pred).ravel()
        results["ENS"] = {
            "accuracy":    float(accuracy_score(y_test, ens_pred)),
            "precision":   float(precision_score(y_test, ens_pred, zero_division=0)),
            "recall":      float(recall_score(y_test, ens_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "f1":          float(f1_score(y_test, ens_pred, zero_division=0)),
            "mcc":         float(matthews_corrcoef(y_test, ens_pred)),
            "auc_roc":     float(roc_auc_score(y_test, ens_proba)),
        }

    model_names = list(results.keys())
    n_models    = len(model_names)
    n_metrics   = len(metrics_def)
    x           = np.arange(n_metrics)
    bar_width   = 0.80 / n_models
    colors      = [COLORS.get(m, ENS_CLR) for m in model_names]

    fig, ax = plt.subplots(figsize=(COL2, 2.9))

    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals      = [results[name][key] for key, _ in metrics_def]
        offset    = (i - (n_models - 1) / 2) * bar_width
        is_best   = name == BEST_MODEL
        alpha     = 1.0 if name == "ENS" else ALPHA
        lw        = 0.7 if is_best else 0
        ec        = "#111111" if is_best else "none"
        ax.bar(x + offset, vals, bar_width,
               color=color, alpha=alpha, linewidth=lw, edgecolor=ec)

        if is_best:
            # Star above the MCC bar
            mcc_xi = next(j for j, (k, _) in enumerate(metrics_def) if k == "mcc")
            ax.annotate(
                "*",
                xy=(x[mcc_xi] + offset, results[name]["mcc"] + 0.003),
                ha="center", va="bottom", fontsize=9,
                color=color, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics_def], fontsize=7)
    ax.set_ylabel("Score")
    ax.set_ylim(0.80, 1.01)
    ax.yaxis.set_major_locator(MaxNLocator(5, prune="upper"))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.axhline(0.90, color="#b22222", linewidth=0.8,
               linestyle="--", zorder=0, alpha=0.7)
    ax.text(-0.5, 0.905, "90%", color="#b22222", fontsize=6, va="bottom")

    legend_patches = [
        mpatches.Patch(color=COLORS.get(m, ENS_CLR),
                       alpha=1.0 if m == "ENS" else ALPHA, label=m)
        for m in model_names
    ]
    ax.legend(handles=legend_patches,
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, fontsize=6.5,
              handlelength=0.9, handletextpad=0.5)

    ax.set_title("Classification Performance: Tuned Models and Majority-Vote Ensemble")
    fig.tight_layout()
    savefig(fig, "fig11_metrics_comparison.png")


# ---------------------------------------------------------------------------
# Fig 12 — Precision-recall curves
# ---------------------------------------------------------------------------
def fig12_precision_recall(outputs: dict):
    fig, ax = plt.subplots(figsize=(COL1 + 1.6, COL1))

    for name, out in outputs.items():
        if out is None:
            continue
        prec, rec, _ = precision_recall_curve(out["y_test"], out["proba"])
        ap      = average_precision_score(out["y_test"], out["proba"])
        is_best = name == BEST_MODEL
        lw      = 1.8 if is_best else 1.0
        al      = 1.0 if is_best else ALPHA * 0.80
        ls      = LINESTYLES.get(name, "-")
        ax.plot(rec, prec, color=COLORS[name], linewidth=lw, linestyle=ls,
                alpha=al, label=f"{name}  AP={ap:.3f}",
                zorder=5 if is_best else 3)

    ax.axhline(0.5, color="#bbbbbb", linewidth=0.7, linestyle=":", label="Random")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0.46, 1.01])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves on Held-Out Test Set")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    _leg_outside(ax, title="Model  AP", title_fontsize=6.5)

    fig.tight_layout()
    fig.subplots_adjust(right=0.74)
    savefig(fig, "fig12_precision_recall.png")


# ---------------------------------------------------------------------------
# Fig 13 — DET curves (normal-deviate scale)
# ---------------------------------------------------------------------------
def fig13_det_curves(outputs: dict):
    from scipy.stats import norm as sp_norm

    fig, ax = plt.subplots(figsize=(COL1 + 1.6, COL1))

    for name, out in outputs.items():
        if out is None:
            continue
        fpr, fnr, _ = det_curve(out["y_test"], out["proba"])
        fpr_nd  = sp_norm.ppf(np.clip(fpr, 5e-4, 1 - 5e-4))
        fnr_nd  = sp_norm.ppf(np.clip(fnr, 5e-4, 1 - 5e-4))
        is_best = name == BEST_MODEL
        lw      = 1.8 if is_best else 1.0
        al      = 1.0 if is_best else ALPHA * 0.80
        ls      = LINESTYLES.get(name, "-")
        ax.plot(fpr_nd, fnr_nd, color=COLORS[name], linewidth=lw,
                linestyle=ls, alpha=al, label=name,
                zorder=5 if is_best else 3)

    lims = np.array([-3.0, 2.0])
    ax.plot(lims, lims, color="#bbbbbb", linewidth=0.7, linestyle=":")

    pct_ticks  = [0.5, 1, 2, 5, 10, 20, 40]
    nd_ticks   = sp_norm.ppf([p / 100 for p in pct_ticks])
    tick_labels = [f"{p}%" for p in pct_ticks]
    ax.set_xticks(nd_ticks)
    ax.set_xticklabels(tick_labels, fontsize=6)
    ax.set_yticks(nd_ticks)
    ax.set_yticklabels(tick_labels, fontsize=6)
    ax.set_xlim(sp_norm.ppf(0.002), sp_norm.ppf(0.45))
    ax.set_ylim(sp_norm.ppf(0.002), sp_norm.ppf(0.45))

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("False Negative Rate")
    ax.set_title("Detection Error Tradeoff (DET) Curves")
    _leg_outside(ax)

    fig.tight_layout()
    fig.subplots_adjust(right=0.74)
    savefig(fig, "fig13_det_curves.png")


# ---------------------------------------------------------------------------
# Fig 14 — Threshold sensitivity (best model: GB)
# ---------------------------------------------------------------------------
def fig14_threshold_sensitivity(outputs: dict, best_model: str = "GB"):
    out = outputs.get(best_model)
    if out is None:
        out = next((v for v in outputs.values() if v is not None), None)
        if out is None:
            print("  Skipping fig14: no model outputs available.")
            return
        best_model = next(n for n, v in outputs.items() if v is not None)

    y_test = out["y_test"]
    proba  = out["proba"]
    color  = COLORS[best_model]

    thresholds = np.linspace(0.01, 0.99, 199)
    records    = {"MCC": [], "F1": [], "Precision": [], "Recall": []}

    for t in thresholds:
        preds = (proba >= t).astype(int)
        records["MCC"].append(matthews_corrcoef(y_test, preds))
        records["F1"].append(f1_score(y_test, preds, zero_division=0))
        records["Precision"].append(precision_score(y_test, preds, zero_division=0))
        records["Recall"].append(recall_score(y_test, preds, zero_division=0))

    # Each metric gets its own colour; linestyles still differentiate shape
    styles = {"MCC":       ("-",  1.4),
              "F1":        ("--", 1.1),
              "Precision": (":",  1.1),
              "Recall":    ("-.", 1.1)}

    fig, ax = plt.subplots(figsize=(COL1 + 1.6, COL1 * 0.88))

    for metric, vals in records.items():
        ls, lw = styles[metric]
        mc = METRIC_COLORS[metric]
        ax.plot(thresholds, vals, color=mc,
                linewidth=lw, linestyle=ls, alpha=0.9, label=metric)

    opt = out["threshold"]
    opt_mcc = records["MCC"][np.argmin(np.abs(thresholds - opt))]
    ax.axvline(opt, color="#666666", linewidth=0.8, linestyle="--", zorder=0)
    ax.scatter([opt], [opt_mcc], color=METRIC_COLORS["MCC"], s=38, zorder=5,
               linewidths=0, marker="*")
    ax.text(opt + 0.02, 0.07, f"t = {opt:.2f}",
            fontsize=6.5, color="#555555", va="bottom")

    ax.set_xlim([0, 1])
    ax.set_ylim([-0.04, 1.04])
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Sensitivity: {best_model}")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    _leg_outside(ax)

    fig.tight_layout()
    fig.subplots_adjust(right=0.76)
    savefig(fig, "fig14_threshold_sensitivity.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("Step 1: Collecting model outputs (isolated subprocesses)...")
    outputs = collect_all_outputs(ALL_MODELS)

    valid = {n: o for n, o in outputs.items() if o is not None}
    if not valid:
        print("No model outputs available. Run collect_outputs.py first.")
        sys.exit(1)

    print(f"\nModels available: {list(valid.keys())}")

    print("\nFig 01: ROC curves...")
    fig01_roc_curves(valid)

    print("\nFig 02: Confusion matrices...")
    fig02_confusion_matrices(valid)

    print("\nFig 03: Calibration curves...")
    fig03_calibration(valid)

    print("\nFig 04: Feature importance...")
    fig04_feature_importance(valid)

    print("\nFig 05: CV score distribution...")
    fig05_cv_score_distribution()

    print("\nFig 06: Top 10 candidates...")
    fig06_top10_candidates()

    print("\nFig 07: RF hyperparameter exploration...")
    fig_hyperparam("rf",
                   ["n_estimators", "max_depth", "min_samples_split", "max_features"],
                   log_params=[], fig_num=7)

    print("\nFig 08: GB hyperparameter exploration...")
    fig_hyperparam("gb",
                   ["n_estimators", "learning_rate", "max_depth", "subsample"],
                   log_params=["learning_rate"], fig_num=8)

    print("\nFig 09: SVM hyperparameter exploration...")
    fig_hyperparam("svm", ["C", "gamma"], log_params=["C", "gamma"], fig_num=9)

    print("\nFig 10: XGBoost hyperparameter exploration...")
    fig_hyperparam("xgb",
                   ["n_estimators", "learning_rate", "max_depth",
                    "reg_alpha", "reg_lambda"],
                   log_params=["learning_rate", "reg_alpha", "reg_lambda"],
                   fig_num=10)

    print("\nFig 11: Metrics comparison...")
    fig11_metrics_comparison(valid)

    print("\nFig 12: Precision-recall curves...")
    fig12_precision_recall(valid)

    print("\nFig 13: DET curves...")
    fig13_det_curves(valid)

    print("\nFig 14: Threshold sensitivity (GB)...")
    fig14_threshold_sensitivity(valid, best_model="GB")

    print(f"\nAll figures saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
