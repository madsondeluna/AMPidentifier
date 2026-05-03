import argparse
import os
import sys
from amp_identifier.core import run_prediction_pipeline, BOX_W


class _Fmt(argparse.RawTextHelpFormatter):
    def _format_action(self, action):
        invocation = self._format_action_invocation(action)
        lines = [f"  {invocation}"]
        if action.help:
            help_text = self._expand_help(action)
            for line in help_text.splitlines():
                lines.append(f"  {line}" if line.strip() else "")
        lines.append("")
        return "\n".join(lines) + "\n"

_TTY = sys.stdout.isatty()

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _TTY else text

_AMP_COLS = 18
_ART = [
    r"   _   __  __ ___ _    _         _   _  __ _",
    r"  /_\ |  \/  | _ (_)__| |___ _ _| |_(_)/ _(_)___ _ _",
    r" / _ \| |\/| |  _/ / _` / -_) ' \  _| |  _| / -_) '_|",
    r"/_/ \_\_|  |_|_| |_\__,_\___|_||_\__|_|_| |_\___|_|",
]
_SUBTITLE = "antimicrobial peptide classifier  |  v2.0"


def _banner():
    inner   = BOX_W - 2
    art_w   = max(len(l) for l in _ART)
    left_pad = (inner - art_w) // 2

    top   = "╔" + "═" * inner + "╗"
    bot   = "╚" + "═" * inner + "╝"
    empty = "║" + " " * inner + "║"

    def _art_row(raw):
        amp_raw  = raw[:_AMP_COLS]
        rest_raw = raw[_AMP_COLS:]
        colored  = _c("1;32", amp_raw) + _c("1;36", rest_raw)
        right_pad = inner - left_pad - len(raw)
        return "║" + " " * left_pad + colored + " " * max(0, right_pad) + "║"

    def _sub_row(raw):
        sub_left = (inner - len(raw)) // 2
        colored  = _c("2", raw)
        right_pad = inner - sub_left - len(raw)
        return "║" + " " * sub_left + colored + " " * max(0, right_pad) + "║"

    print()
    print(top)
    for line in _ART:
        print(_art_row(line))
    print(empty)
    print(_sub_row(_SUBTITLE))
    print(bot)
    print()


def main():
    _banner()

    parser = argparse.ArgumentParser(
        description="AMPidentifier 2.0: antimicrobial peptide classification from FASTA input.",
        formatter_class=_Fmt,
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
            "\n"
            "         Accuracy  AUC-ROC   MCC \n"
            "rf     :  91.9%%     0.972   0.839  Random Forest\n"
            "svm    :  91.9%%     0.969   0.839  Support Vector Machine (RBF kernel)\n"
            "gb     :  92.0%%     0.974   0.839  Gradient Boosting\n"
            "xgb    :  92.2%%     0.974   0.843  XGBoost\n"
            "lgbm   :  92.7%%     0.975   0.855  LightGBM\n"
            "voting :  92.9%%     0.977   0.859  Soft-voting ensemble (recommended)\n"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Decision threshold for AMP classification (0.0 to 1.0). If omitted, the MCC-optimized threshold for the selected model is used.\n​",
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
        threshold_override=args.threshold,
    )


if __name__ == "__main__":
    main()
