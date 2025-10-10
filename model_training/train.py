# model_training/train.py

import os
import random
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from amp_identifier.feature_extraction import calculate_physicochemical_features
from amp_identifier.data_io import load_fasta_sequences

# --- Configuration ---
DATA_DIR = "model_training/data"
OUTPUT_DIR = "model_training/saved_model"
POSITIVE_FILE = os.path.join(DATA_DIR, "positive_sequences.fasta")
NEGATIVE_FILE = os.path.join(DATA_DIR, "negative_sequences.fasta")
TEST_FEATURES_PATH = os.path.join(DATA_DIR, "test_features.csv")
TEST_LABELS_PATH = os.path.join(DATA_DIR, "test_labels.csv")

def generate_dummy_fasta_data():
    # This function remains unchanged...
    if os.path.exists(POSITIVE_FILE) and os.path.exists(NEGATIVE_FILE):
        return
    print("Generating dummy FASTA data for training...")
    os.makedirs(DATA_DIR, exist_ok=True)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    with open(POSITIVE_FILE, "w") as f:
        for i in range(100):
            seq = "".join(random.choices(amino_acids + "KR"*2, k=random.randint(10, 30)))
            f.write(f">positive_seq_{i+1}\n{seq}\n")
    with open(NEGATIVE_FILE, "w") as f:
        for i in range(100):
            seq = "".join(random.choices(amino_acids, k=random.randint(20, 50)))
            f.write(f">negative_seq_{i+1}\n{seq}\n")
    print("Dummy data generated.")

def main():
    """Main function to run the training pipeline for multiple models."""
    print("--- Starting Model Training Pipeline ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_dummy_fasta_data()

    # --- Data Loading and Feature Calculation (same as before) ---
    print("Step 1: Loading sequence data...")
    pos_seqs, pos_ids = load_fasta_sequences(POSITIVE_FILE)
    neg_seqs, neg_ids = load_fasta_sequences(NEGATIVE_FILE)
    sequences, ids = pos_seqs + neg_seqs, pos_ids + neg_ids
    labels = [1] * len(pos_seqs) + [0] * len(neg_seqs)

    print("Step 2: Calculating features...")
    features_df = calculate_physicochemical_features(sequences, ids)
    features_df['label'] = labels
    X = features_df.drop(columns=['ID', 'sequence', 'label']).fillna(0)
    y = features_df['label']

    print("Step 3: Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- NEW: Normalize features using StandardScaler ---
    print("Step 3.1: Normalizing features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Save the scaler for later use in predictions
    scaler_path = os.path.join(OUTPUT_DIR, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save test data once for all models to be evaluated on the same set
    X_test_scaled.to_csv(TEST_FEATURES_PATH, index=False)
    y_test.to_frame().to_csv(TEST_LABELS_PATH, index=False)
    print("Test data saved for evaluation script.")

    # --- NEW: Define multiple models to train ---
    models = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'svm': SVC(probability=True, random_state=42, class_weight='balanced'),
        'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # --- NEW: Loop to train and save each model ---
    for model_name, model in models.items():
        print(f"\n--- Training {model_name.upper()} model ---")
        model.fit(X_train_scaled, y_train)
        print(f"Model {model_name.upper()} training complete.")
        
        model_path = os.path.join(OUTPUT_DIR, f"amp_model_{model_name}.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    
    print("\n--- All Training Pipelines Finished Successfully ---")

if __name__ == "__main__":
    main()