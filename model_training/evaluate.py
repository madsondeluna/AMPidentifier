# model_training/evaluate.py

import pandas as pd
import joblib
import os
import glob
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef
)

# --- Configuration ---
DATA_DIR = "model_training/data"
MODEL_DIR = "model_training/saved_model"
TEST_FEATURES_PATH = os.path.join(DATA_DIR, "test_features.csv")
TEST_LABELS_PATH = os.path.join(DATA_DIR, "test_labels.csv")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
# --- NEW: Define output report paths ---
TXT_REPORT_PATH = os.path.join(MODEL_DIR, "evaluation_report.txt")
CSV_REPORT_PATH = os.path.join(MODEL_DIR, "evaluation_report.csv")

def main():
    """
    Main function to evaluate all trained models found in the saved_model directory.
    Note: The test features are already normalized (saved from train.py).
    """
    print("--- Evaluating All Trained Models ---")

    # Load the single test dataset (already normalized)
    print("Loading test data (already normalized)...")
    try:
        X_test = pd.read_csv(TEST_FEATURES_PATH)
        y_test = pd.read_csv(TEST_LABELS_PATH).squeeze()
    except FileNotFoundError as e:
        print(f"Error: Could not find test data. Please run train.py first. Details: {e}")
        return

    # --- NEW: Find all model files using a glob pattern ---
    model_paths = glob.glob(os.path.join(MODEL_DIR, "amp_model_*.pkl"))
    if not model_paths:
        print("Error: No trained models found in 'model_training/saved_model/'. Please run train.py first.")
        return
    
    print(f"Found {len(model_paths)} models to evaluate.")
    
    all_metrics = []

    # --- NEW: Loop through each model, load it, and evaluate it ---
    for model_path in model_paths:
        model_name = os.path.splitext(os.path.basename(model_path))[0].replace('amp_model_', '')
        print(f"\n--- Evaluating model: {model_name.upper()} ---")
        
        model = joblib.load(model_path)
        
        # Make predictions (data is already normalized)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        # Store metrics in a dictionary
        metrics_dict = {
            'model': model_name.upper(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'mcc': mcc,
            'auc_roc': auc_roc
        }
        all_metrics.append(metrics_dict)
    
    # --- NEW: Save the consolidated reports ---
    if not all_metrics:
        print("No models were evaluated.")
        return
        
    # Save a detailed text report
    print(f"\nSaving detailed text report to {TXT_REPORT_PATH}...")
    with open(TXT_REPORT_PATH, 'w') as f:
        f.write("########### Consolidated Model Evaluation Report ###########\n")
        for metrics in all_metrics:
            f.write("\n------------------------------------------------------\n")
            f.write(f" MODEL: {metrics['model']}\n")
            f.write("------------------------------------------------------\n")
            f.write(f" Accuracy:    {metrics['accuracy']:.4f}\n")
            f.write(f" Precision:   {metrics['precision']:.4f}\n")
            f.write(f" Recall:      {metrics['recall']:.4f}\n")
            f.write(f" Specificity: {metrics['specificity']:.4f}\n")
            f.write(f" F1-Score:    {metrics['f1_score']:.4f}\n")
            f.write(f" MCC:         {metrics['mcc']:.4f}\n")
            f.write(f" AUC-ROC:     {metrics['auc_roc']:.4f}\n")

    # Save a CSV report for easy comparison (e.g., in Excel)
    print(f"Saving CSV report to {CSV_REPORT_PATH}...")
    report_df = pd.DataFrame(all_metrics)
    report_df.to_csv(CSV_REPORT_PATH, index=False)
    
    print("\n--- All Evaluations Finished Successfully ---")

if __name__ == "__main__":
    main()