# benchmarking/plot_internal_metrics.py
#
# Grouped bar chart for the internal held-out test set (Table 2),
# following the same NPG aesthetics as plot_benchmark_comparison.py.
#
# Run from project root:
#   python3 -m benchmarking.plot_internal_metrics

import os
import gc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ---------------------------------------------------------------------------
# NPG rcParams
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "legend.frameon":     False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "lines.linewidth":    1.0,
    "patch.linewidth":    0.4,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

COL2  = 7.20
ALPHA = 0.90

# Same teal family as plot_benchmark_comparison.py
COLORS = {
    "Voting": "#005f73",
    "LGBM":   "#0a7d8a",
    "RF":     "#1a9ba1",
    "GB":     "#43b8be",
    "XGB":    "#7dd3d8",
    "SVM":    "#b2e8eb",
}

# Internal held-out test set — Table 2 (fractions; MCC already in [0,1])
DATA = {
    "Voting": [0.929, 0.942, 0.914, 0.944, 0.928, 0.859, 0.977],
    "LGBM":   [0.927, 0.942, 0.911, 0.943, 0.926, 0.855, 0.975],
    "RF":     [0.919, 0.938, 0.897, 0.941, 0.917, 0.839, 0.972],
    "GB":     [0.920, 0.929, 0.909, 0.931, 0.919, 0.839, 0.974],
    "XGB":    [0.922, 0.920, 0.924, 0.919, 0.922, 0.843, 0.974],
    "SVM":    [0.919, 0.918, 0.921, 0.918, 0.919, 0.839, 0.969],
}

METRICS    = ["Accuracy", "Precision", "Sn", "Sp", "F1", "MCC", "AUC-ROC"]
TOOL_ORDER = list(DATA.keys())

OUTDIR  = "benchmarking"
OUTFILE = os.path.join(OUTDIR, "fig_internal_metrics.png")


def _darken(hex_color, factor=0.65):
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4)]
    return "#{:02x}{:02x}{:02x}".format(
        int(r * factor * 255), int(g * factor * 255), int(b * factor * 255)
    )


def main():
    n_met  = len(METRICS)
    n_tool = len(TOOL_ORDER)
    x      = np.arange(n_met, dtype=float)
    width  = 0.75 / n_tool
    offsets = np.linspace(-(n_tool - 1) / 2, (n_tool - 1) / 2, n_tool) * width

    fig, ax = plt.subplots(figsize=(COL2 * 1.2, COL2 * 0.65))

    for i, tool in enumerate(TOOL_ORDER):
        color = COLORS[tool]
        vals  = DATA[tool]
        bar_x = x + offsets[i]

        for j, (bx, v) in enumerate(zip(bar_x, vals)):
            ax.bar(bx, v, width, color=color, alpha=ALPHA, linewidth=0)
            label = f"{v*100:.1f}%" if METRICS[j] != "MCC" else f"{v:.3f}"
            ax.text(
                bx, v - 0.012, label,
                ha="center", va="top",
                fontsize=4.5, rotation=90,
                color="white",
            )

    # Reference lines
    ref_specs = [
        (0.70, "70%", (0, (1, 2)),       0.35),
        (0.80, "80%", (0, (4, 2)),       0.35),
        (0.90, "90%", (0, (6, 2, 1, 2)), 0.35),
    ]
    ref_handles = []
    for yval, label, ls, alpha in ref_specs:
        ax.axhline(yval, color="#aaaaaa", linewidth=0.7,
                   linestyle=ls, alpha=alpha, zorder=0)
        ref_handles.append(
            mlines.Line2D([], [], color="#aaaaaa", linewidth=0.7,
                          linestyle=ls, alpha=alpha, label=label)
        )

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Classification Performance on the Internal Held-Out Test Set")

    # Legend: row 1 models, row 2 reference lines
    model_handles = [
        mpl.patches.Patch(color=COLORS[t], alpha=ALPHA, label=t)
        for t in TOOL_ORDER
    ]

    leg1 = ax.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(model_handles),
        fontsize=6.5, frameon=False,
        handlelength=1.2, columnspacing=0.8,
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=ref_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=len(ref_handles),
        fontsize=6.5, frameon=False,
        handlelength=1.2, columnspacing=0.8,
    )

    fig.subplots_adjust(bottom=0.20)
    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(OUTFILE)
    print(f"Saved -> {OUTFILE}")
    plt.close(fig)
    gc.collect()


if __name__ == "__main__":
    main()
