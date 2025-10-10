# amp_identifier/core.py

import os
import glob
from . import data_io, feature_extraction, prediction, reporting

# Define the location of internal models
MODEL_DIR = "model_training/saved_model"
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")

def run_prediction_pipeline(input_file: str, output_dir: str, internal_model_type: str, use_ensemble: bool, external_model_paths: list):
    """Orchestrates the full prediction pipeline with model selection and ensemble options."""
    print("--- Starting AMP Identification Pipeline ---")

    # Steps 1 & 2: Load sequences and calculate features (unchanged)
    print(f"Step 1: Loading sequences from {input_file}...")
    sequences, seq_ids = data_io.load_fasta_sequences(input_file)
    if not sequences:
        print("No sequences loaded. Exiting pipeline.")
        return
    print(f"Found {len(sequences)} sequences.")

    print("Step 2: Calculating physicochemical features...")
    features_df = feature_extraction.calculate_physicochemical_features(sequences, seq_ids)
    features_df.fillna(0, inplace=True)
    
    features_report_path = os.path.join(output_dir, "physicochemical_features.csv")
    reporting.save_features_report(features_df, features_report_path)
    print(f"Physicochemical features report saved to {features_report_path}")

    # --- NEW: Prediction logic based on user's choice ---
    print("Step 3: Running predictions...")
    
    # Load the scaler for feature normalization
    print("Step 3.1: Loading feature scaler...")
    scaler = prediction.load_scaler(SCALER_PATH)
    if scaler is None:
        print("Warning: Could not load scaler. Predictions may be inaccurate if models were trained with normalized data.")
    
    all_predictions = {}
    
    # --- Ensemble Voting Logic ---
    if use_ensemble:
        print(" -> Mode: Ensemble Voting")
        internal_model_paths = glob.glob(os.path.join(MODEL_DIR, "amp_model_*.pkl"))
        if not internal_model_paths:
            print("Warning: No internal models found for ensemble mode.")
        
        for model_path in internal_model_paths:
            model_name = os.path.splitext(os.path.basename(model_path))[0].replace('amp_model_', '')
            print(f"    -> Using internal model: {model_name}")
            internal_model = prediction.load_model(model_path)
            if internal_model:
                internal_results = prediction.predict_sequences(internal_model, features_df.copy(), scaler)
                all_predictions[f"internal_{model_name}"] = internal_results
    
    # --- Single Internal Model Logic ---
    else:
        print(f" -> Mode: Single Model (type: {internal_model_type.upper()})")
        model_path = os.path.join(MODEL_DIR, f"amp_model_{internal_model_type}.pkl")
        internal_model = prediction.load_model(model_path)
        if internal_model:
            internal_results = prediction.predict_sequences(internal_model, features_df.copy(), scaler)
            all_predictions[f"internal_{internal_model_type}"] = internal_results

    # --- External Model Logic (unchanged) ---
    if external_model_paths:
        for model_path in external_model_paths:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            print(f" -> Comparing with external model: {model_name}")
            external_model = prediction.load_model(model_path)
            if external_model:
                # Note: External models may not use the same scaler, so we pass None
                external_results = prediction.predict_sequences(external_model, features_df.copy(), None)
                all_predictions[f"external_{model_name}"] = external_results
    
    # --- Step 4: Generate Report ---
    if not all_predictions:
        print("No models were successfully loaded. Cannot generate prediction report.")
        print("--- Pipeline Finished ---")
        return
        
    print("Step 4: Generating final comparison report...")
    comparison_report_path = os.path.join(output_dir, "prediction_comparison_report.csv")
    reporting.save_comparison_report(features_df, all_predictions, use_ensemble, comparison_report_path)
    print(f"Comparison report saved to {comparison_report_path}")
    print("--- Pipeline Finished Successfully ---")
    
    # Citation message
    print("\n" + "*"*80)
    print("If this tool supports your research, please cite:")
    print("Luna-Aragão, M. A., da Silva, R. L., Pacífico, J., Santos-Silva, C. A. & Benko-Iseppon, A. M. (2025).")
    print("AMPidentifier: A Python toolkit for predicting antimicrobial peptides using ensemble machine learning.")
    print("GitHub repository: https://github.com/madsondeluna/AMPIdentifier")
    print("*"*80)