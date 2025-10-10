# amp_identifier/core.py

import os
import glob
import pandas as pd
from tqdm import tqdm
from . import data_io, feature_extraction, prediction, reporting

# Define the location of internal models
MODEL_DIR = "model_training/saved_model"
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")


def run_prediction_pipeline(input_file: str, output_dir: str, internal_model_type: str, use_ensemble: bool, external_model_paths: list):
    """Orchestrates the full prediction pipeline with model selection and ensemble options."""
    
    print("\n" + "="*80)
    print("  AMP IDENTIFICATION PIPELINE")
    print("="*80 + "\n")

    # Step 1: Load sequences
    print("━━ Step 1/4: Loading sequences")
    with tqdm(total=1, desc="  Loading FASTA", bar_format='{desc}: {bar} {percentage:3.0f}%', ncols=80) as pbar:
        sequences, seq_ids = data_io.load_fasta_sequences(input_file)
        pbar.update(1)
    
    if not sequences:
        print("  ❌ No sequences loaded. Exiting pipeline.")
        return
    print(f"  ✓ Found {len(sequences)} sequence(s)\n")

    # Step 2: Calculate features
    print("━━ Step 2/4: Extracting features")
    with tqdm(total=len(sequences), desc="  Calculating features", bar_format='{desc}: {bar} {percentage:3.0f}%', ncols=80) as pbar:
        features_df = feature_extraction.calculate_physicochemical_features(sequences, seq_ids)
        features_df.fillna(0, inplace=True)
        pbar.update(len(sequences))
    
    features_report_path = os.path.join(output_dir, "physicochemical_features.csv")
    reporting.save_features_report(features_df, features_report_path)
    print(f"  ✓ Features saved\n")

    # Step 3: Run predictions
    print("━━ Step 3/4: Running predictions")
    print(f"  → Loading feature scaler...")
    scaler = prediction.load_scaler(SCALER_PATH)
    if scaler is None:
        print("  ⚠ Warning: Could not load scaler")
    
    all_predictions = {}
    
    # --- Ensemble Voting Logic ---
    if use_ensemble:
        print(f"  → Mode: Ensemble (RF + SVM + GB)")
        internal_model_paths = glob.glob(os.path.join(MODEL_DIR, "amp_model_*.pkl"))
        
        total_models = len(internal_model_paths)
        
        with tqdm(internal_model_paths, desc="  Processing models", bar_format='{desc}: {bar} {n_fmt}/{total_fmt}', ncols=80) as pbar:
            for model_path in pbar:
                model_name = os.path.splitext(os.path.basename(model_path))[0].replace('amp_model_', '').upper()
                pbar.set_description(f"  Processing {model_name}")
                
                internal_model = prediction.load_model(model_path)
                if internal_model:
                    internal_results = prediction.predict_sequences(internal_model, features_df.copy(), scaler)
                    all_predictions[f"internal_{model_name.lower()}"] = internal_results
        
        print(f"  ✓ {total_models} models processed\n")
    
    # --- Single Internal Model Logic ---
    else:
        print(f"  → Mode: Single model ({internal_model_type.upper()})")
        
        with tqdm(total=1, desc=f"  Loading {internal_model_type.upper()}", bar_format='{desc}: {bar} {percentage:3.0f}%', ncols=80) as pbar:
            model_path = os.path.join(MODEL_DIR, f"amp_model_{internal_model_type}.pkl")
            internal_model = prediction.load_model(model_path)
            if internal_model:
                internal_results = prediction.predict_sequences(internal_model, features_df.copy(), scaler)
                all_predictions[f"internal_{internal_model_type}"] = internal_results
            pbar.update(1)
        
        print(f"  ✓ Model executed\n")

    # --- External Model Logic ---
    if external_model_paths:
        total_ext = len(external_model_paths)
        
        with tqdm(external_model_paths, desc="  External models", bar_format='{desc}: {bar} {n_fmt}/{total_fmt}', ncols=80) as pbar:
            for model_path in pbar:
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                pbar.set_description(f"  Loading {model_name}")
                
                external_model = prediction.load_model(model_path)
                if external_model:
                    external_results = prediction.predict_sequences(external_model, features_df.copy(), None)
                    all_predictions[f"external_{model_name}"] = external_results
        
        print(f"  ✓ {total_ext} external model(s) processed\n")
    
    # Step 4: Generate Report
    if not all_predictions:
        print("  ❌ No models loaded. Cannot generate report.")
        print("="*80 + "\n")
        return
    
    print("━━ Step 4/4: Generating report")
    
    with tqdm(total=1, desc="  Saving results", bar_format='{desc}: {bar} {percentage:3.0f}%', ncols=80) as pbar:
        comparison_report_path = os.path.join(output_dir, "prediction_comparison_report.csv")
        reporting.save_comparison_report(features_df, all_predictions, use_ensemble, comparison_report_path)
        pbar.update(1)
    
    print(f"  ✓ Report saved\n")
    
    # Calculate and display statistics
    print("━━ PREDICTION SUMMARY")
    print()
    
    report_df = pd.read_csv(comparison_report_path)
    total_sequences = len(report_df)
    
    # Determine which column to use for AMP classification
    if use_ensemble and 'ensemble_prediction' in report_df.columns:
        amp_count = int(report_df['ensemble_prediction'].sum())
        prediction_method = "Ensemble Voting"
    elif f'pred_internal_{internal_model_type}' in report_df.columns:
        amp_count = int(report_df[f'pred_internal_{internal_model_type}'].sum())
        prediction_method = f"{internal_model_type.upper()}"
    else:
        pred_cols = [col for col in report_df.columns if col.startswith('pred_')]
        if pred_cols:
            amp_count = int(report_df[pred_cols[0]].sum())
            prediction_method = "Single Model"
        else:
            amp_count = 0
            prediction_method = "Unknown"
    
    non_amp_count = total_sequences - amp_count
    amp_percentage = (amp_count / total_sequences * 100) if total_sequences > 0 else 0
    non_amp_percentage = 100 - amp_percentage
    
    # Create compact visual bar (30 chars total)
    bar_total = 30
    amp_bar_len = int(amp_percentage * bar_total / 100)
    non_amp_bar_len = bar_total - amp_bar_len
    
    print(f"  Sequences analyzed: {total_sequences}")
    print(f"  Method: {prediction_method}")
    print()
    print(f"  🧬 AMPs:     {amp_count:3d} ({amp_percentage:5.1f}%)  [{'█' * amp_bar_len}{'░' * non_amp_bar_len}]")
    print(f"  🚫 Non-AMPs: {non_amp_count:3d} ({non_amp_percentage:5.1f}%)  [{'░' * amp_bar_len}{'█' * non_amp_bar_len}]")
    print()
    print(f"  📄 Results: {comparison_report_path}")
    print()
    print("="*80)

    # Citation message
    print()
    print("If this tool supports your research, please cite:")
    print("Luna-Aragão, M. A., da Silva, R. L., Pacífico, J., Santos-Silva, C. A. & Benko-Iseppon, A. M. (2025).")
    print("AMPidentifier: A Python toolkit for predicting antimicrobial peptides using ensemble machine learning.")
    print("GitHub repository: https://github.com/madsondeluna/AMPIdentifier")
    print("="*80 + "\n")