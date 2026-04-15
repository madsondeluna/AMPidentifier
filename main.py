# main.py

import argparse
import os
import sys
from amp_identifier.core import run_prediction_pipeline, BOX_W

_TTY = sys.stdout.isatty()

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _TTY else text

def _banner():
    # Box dimensions match the result box in core.py (BOX_W=72, INNER=70)
    inner = BOX_W - 2

    # Visible text (raw, for padding calculation)
    title_raw    = "AMP" + "identifier" + " 2.0"       # 17 chars
    subtitle_raw = "physicochemical feature engineering | ensemble ML"  # 49 chars

    # Colored versions (ANSI codes invisible)
    title_col = (
        _c("1;32", "AMP")
        + _c("1;36", "identifier")
        + _c("2",    " 2.0")
    )
    subtitle_col = _c("2", subtitle_raw)

    top = "\u2554" + "\u2550" * inner + "\u2557"
    bot = "\u255a" + "\u2550" * inner + "\u255d"
    empty = "\u2551" + " " * inner + "\u2551"

    indent = 4
    def _row(raw, colored):
        pad = inner - indent - len(raw)
        return "\u2551" + " " * indent + colored + " " * pad + "\u2551"

    print()
    print(top)
    print(empty)
    print(_row(title_raw,    title_col))
    print(_row(subtitle_raw, subtitle_col))
    print(empty)
    print(bot)
    print()


def main():
    _banner()

    parser = argparse.ArgumentParser(
        description="AMPidentifier 2.0: antimicrobial peptide classification from FASTA input.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        metavar="FASTA",
        help="Path to input FASTA file. Each header must contain a unique sequence ID.",
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        type=str,
        metavar="DIR",
        help="Directory where output files will be written. Created if it does not exist.",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="voting",
        choices=["rf", "svm", "gb", "xgb", "lgbm", "voting"],
        help=(
            "Model to use for prediction (default: voting).\n"
            "  rf     : Random Forest\n"
            "  svm    : Support Vector Machine (RBF kernel)\n"
            "  gb     : Gradient Boosting\n"
            "  xgb    : XGBoost\n"
            "  lgbm   : LightGBM\n"
            "  voting : Soft-voting ensemble of all five models (recommended)\n"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Decision threshold for AMP classification (0.0 to 1.0).\n"
            "If omitted, the MCC-optimized threshold for the selected model is used.\n"
            "  rf     : 0.56\n"
            "  svm    : 0.47\n"
            "  gb     : 0.55\n"
            "  xgb    : 0.48\n"
            "  lgbm   : 0.71\n"
            "  voting : 0.56\n"
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    run_prediction_pipeline(
        input_file=args.input,
        output_dir=args.output_dir,
        internal_model_type=args.model,
        use_ensemble=(args.model == "voting"),
    )


if __name__ == "__main__":
    main()
