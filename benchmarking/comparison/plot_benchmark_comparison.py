# benchmarking/plot_benchmark_comparison.py
#
# Generates a grouped bar chart comparing all classifiers from the independent
# benchmark (Table 3), following the NPG style of fig12_metrics_comparison.png.
#
# Run from project root:
#   python3 -m benchmarking.plot_benchmark_comparison

import os
import gc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ---------------------------------------------------------------------------
# NPG rcParams (mirrors plot_tuning.py)
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

COL2 = 7.20   # Nature double-column width (inches)

# ---------------------------------------------------------------------------
# Colour palette
# AMPidentifier: shades of a single teal/blue family (dark → light)
# External tools: shades of a warm brown/orange family
# ---------------------------------------------------------------------------
AMP_COLORS = {
    "AMPidentifier\nVoting": "#005f73",   # darkest teal  (best model)
    "AMPidentifier\nLGBM":   "#0a7d8a",
    "AMPidentifier\nRF":     "#1a9ba1",
    "AMPidentifier\nGB":     "#43b8be",
    "AMPidentifier\nXGB":    "#7dd3d8",
    "AMPidentifier\nSVM":    "#b2e8eb",   # lightest teal
}

CAMPR3_COLORS = {
    "CAMPR3\nRF":    "#7b2d8b",   # deep violet
    "CAMPR3\nSVM":   "#9e4faa",
    "CAMPR3\nDA":    "#bf80c8",
    "CAMPR3\nANN":   "#ddb8e3",   # lightest lavender
}

OTHER_COLORS = {
    "AMPScanner v2": "#E18727",   # amber
    "AMPlify":       "#20854E",   # forest green
    "ampir":         "#BC3C29",   # brick red
    "DBAASP":        "#6F99AD",   # steel blue
    "amPEPpy":       "#7876B1",   # indigo
}

ALPHA = 0.90

# ---------------------------------------------------------------------------
# Benchmark data — Table 3 values (fractions, not percentages)
# NaN for tools with binary-only output (no AUC-ROC)
# ---------------------------------------------------------------------------
DATA = {
    # AMPidentifier group
    "AMPidentifier\nVoting": [0.866, 0.814, 0.949, 0.784, 0.876, 0.742, 0.950],
    "AMPidentifier\nLGBM":   [0.870, 0.822, 0.945, 0.796, 0.879, 0.749, 0.948],
    "AMPidentifier\nRF":     [0.864, 0.815, 0.941, 0.787, 0.874, 0.736, 0.948],
    "AMPidentifier\nGB":     [0.858, 0.805, 0.945, 0.770, 0.869, 0.727, 0.935],
    "AMPidentifier\nXGB":    [0.846, 0.787, 0.948, 0.744, 0.860, 0.707, 0.930],
    "AMPidentifier\nSVM":    [0.841, 0.785, 0.939, 0.742, 0.855, 0.695, 0.943],
    # CAMPR3 group
    "CAMPR3\nRF":            [0.848, 0.803, 0.922, 0.774, 0.858, 0.704, 0.934],
    "CAMPR3\nSVM":           [0.845, 0.814, 0.895, 0.795, 0.853, 0.694, 0.919],
    "CAMPR3\nDA":            [0.824, 0.797, 0.870, 0.779, 0.832, 0.651, 0.909],
    "CAMPR3\nANN":           [0.792, 0.771, 0.832, 0.753, 0.800, 0.586, np.nan],
    # Other tools
    "AMPScanner v2":         [0.854, 0.802, 0.939, 0.769, 0.865, 0.718, 0.936],
    "AMPlify":               [0.837, 0.778, 0.943, 0.730, 0.853, 0.689, 0.932],
    "ampir":                 [0.810, 0.744, 0.945, 0.675, 0.833, 0.644, 0.921],
    "DBAASP":                [0.756, 0.817, 0.641, 0.865, 0.718, 0.521, np.nan],
    "amPEPpy":               [0.729, 0.656, 0.965, 0.493, 0.781, 0.520, 0.934],
}

METRICS    = ["Accuracy", "Precision", "Sn", "Sp", "F1", "MCC", "AUC-ROC"]
TOOL_ORDER   = list(DATA.keys())
N_AMP        = len(AMP_COLORS)
N_AMP_CAMPR3 = N_AMP + len(CAMPR3_COLORS)

OUTDIR  = "benchmarking"
OUTFILE = os.path.join(OUTDIR, "fig_benchmark_comparison.png")


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
    width  = 0.80 / n_tool
    offsets = np.linspace(-(n_tool - 1) / 2, (n_tool - 1) / 2, n_tool) * width

    all_colors = {**AMP_COLORS, **CAMPR3_COLORS, **OTHER_COLORS}

    fig, ax = plt.subplots(figsize=(COL2 * 1.95, COL2 * 0.80))

    for i, tool in enumerate(TOOL_ORDER):
        color = all_colors[tool]
        vals  = DATA[tool]
        bar_x = x + offsets[i]

        for j, (bx, v) in enumerate(zip(bar_x, vals)):
            if np.isnan(v):
                ax.bar(bx, 0.02, width, bottom=0,
                       color="none", edgecolor="#bbbbbb",
                       linewidth=0.5, hatch="///", alpha=0.5)
            else:
                ax.bar(bx, v, width, color=color, alpha=ALPHA, linewidth=0)
                label = f"{v*100:.1f}%" if METRICS[j] != "MCC" else f"{v:.3f}"
                ax.text(
                    bx, v - 0.012, label,
                    ha="center", va="top",
                    fontsize=3.8, rotation=90,
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
    ax.set_title("Classification Performance on the Independent Benchmark")


    # -----------------------------------------------------------------------
    # Legend: three rows, centre-aligned
    #   Row 1: AMPidentifier (6)
    #   Row 2: CAMPR3 (4)
    #   Row 3: AMPScanner v2, AMPlify, ampir, DBAASP, amPEPpy + N/A + refs
    # -----------------------------------------------------------------------
    amp_handles = [
        mpl.patches.Patch(color=AMP_COLORS[t], alpha=ALPHA,
                          label=t.replace("\n", " "))
        for t in AMP_COLORS
    ]
    campr3_handles = [
        mpl.patches.Patch(color=CAMPR3_COLORS[t], alpha=ALPHA,
                          label=t.replace("\n", " "))
        for t in CAMPR3_COLORS
    ]
    other_handles = [
        mpl.patches.Patch(color=OTHER_COLORS[t], alpha=ALPHA,
                          label=t.replace("\n", " "))
        for t in OTHER_COLORS
    ]
    na_handle = mpl.patches.Patch(
        facecolor="none", edgecolor="#bbbbbb",
        linewidth=0.5, hatch="///", label="N/A (binary output)"
    )

    leg1 = ax.legend(
        handles=amp_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(amp_handles),
        fontsize=6.5, frameon=False,
        handlelength=1.2, columnspacing=0.8,
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=campr3_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=len(campr3_handles),
        fontsize=6.5, frameon=False,
        handlelength=1.2, columnspacing=0.8,
    )
    ax.add_artist(leg2)

    leg3 = ax.legend(
        handles=other_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=len(other_handles),
        fontsize=6.5, frameon=False,
        handlelength=1.2, columnspacing=0.8,
    )
    ax.add_artist(leg3)

    ax.legend(
        handles=[na_handle] + ref_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.21),
        ncol=4,
        fontsize=6.5, frameon=False,
        handlelength=1.2, columnspacing=0.8,
    )

    fig.subplots_adjust(bottom=0.27)
    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(OUTFILE)
    print(f"Saved -> {OUTFILE}")
    plt.close(fig)
    gc.collect()


if __name__ == "__main__":
    main()
