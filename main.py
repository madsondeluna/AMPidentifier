# main.py

import argparse
import os
import sys
from amp_identifier.core import run_prediction_pipeline

_USE_COLOR = sys.stdout.isatty()

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

BANNER = (
    "\n"
    + _c("1;36", "  AMPidentifier 2.0") + "\n"
    + _c("2",    "  physicochemical feature engineering | ensemble ML") + "\n"
)


def main():
    print(BANNER)

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
