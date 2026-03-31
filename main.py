# main.py

import argparse
import os
from amp_identifier.core import run_prediction_pipeline

# --- ASCII Art Banner ---
# Added to give the tool a visual identity in the terminal.
# The 'r' before the triple quotes (r""") creates a "raw string",
# which prevents backslashes from being interpreted as escape characters.
BANNER = r"""

////////////////////////////////////////////////////////////////////////
//                                                                    //
//                                                                    //
//      _    __  __ ____  _     _            _   _  __ _              //
//     / \  |  \/  |  _ \(_) __| | ___ _ __ | |_(_)/ _(_) ___ _ __    //
//    / _ \ | |\/| | |_) | |/ _` |/ _ \ '_ \| __| | |_| |/ _ \ '__|   //
//   / ___ \| |  | |  __/| | (_| |  __/ | | | |_| |  _| |  __/ |      //
//  /_/   \_\_|  |_|_|   |_|\__,_|\___|_| |_|\__|_|_| |_|\___|_|      //
//                                                                    //
//             ** BETA VERSION - FOR TESTING PURPOSES **              //
//                                                                    //
////////////////////////////////////////////////////////////////////////

"""

def main():
    """
    Main function to parse command-line arguments and run the prediction pipeline.
    """
    # Prints the banner as soon as the tool is executed
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="AMP Identifier: A tool for Antimicrobial Peptide prediction and analysis.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to the input file containing sequences (FASTA format)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        type=str,
        help="Directory where the result files will be saved."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default='rf',
        choices=['rf', 'svm', 'gb', 'xgb'],
        help="Type of internal model to use for prediction. (default: rf)."
    )
    parser.add_argument(
        "--ensemble",
        action='store_true',
        help="Use all internal models and predict by majority vote."
    )
    parser.add_argument(
        "-e", "--external_models",
        nargs='*',
        type=str,
        default=[],
        help="List of paths to external .pkl models for comparison."
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    run_prediction_pipeline(
        input_file=args.input,
        output_dir=args.output_dir,
        internal_model_type=args.model,
        use_ensemble=args.ensemble,
        external_model_paths=args.external_models
    )

if __name__ == "__main__":
    main()

