# amp_identifier/prediction.py

import pandas as pd
import joblib
import os
from typing import Optional

def load_model(model_path: str):
    """
    Loads a .pkl model from a specified path.

    Returns the loaded model object or None if an error occurs.
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

def load_scaler(scaler_path: str):
    """
    Loads a StandardScaler from a specified path.

    Returns the loaded scaler object or None if an error occurs.
    """
    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded successfully from {scaler_path}")
        return scaler
    except FileNotFoundError:
        print(f"Error: Scaler file not found at '{scaler_path}'")
        return None
    except Exception as e:
        print(f"An error occurred while loading the scaler: {e}")
        return None

def predict_sequences(model, features_df: pd.DataFrame, scaler=None) -> Optional[pd.DataFrame]:
    """
    Performs predictions using a loaded model and a features DataFrame.

    Args:
        model: The loaded machine learning model object.
        features_df (pd.DataFrame): DataFrame containing the sequence features.
        scaler: Optional StandardScaler to normalize features before prediction.

    Returns:
        A DataFrame with sequences and their predictions, or None on failure.
    """
    if model is None:
        return None
    
    try:
        # Get the feature names the model was trained on
        model_features = model.feature_names_in_
    except AttributeError:
        print("\nWarning: Cannot determine feature names from the model.")
        print("The model was likely trained with an older scikit-learn version.")
        print("Assuming the feature order is correct, but this is risky.\n")
        # Fallback: use all columns from the features_df except ID and sequence
        model_features = [col for col in features_df.columns if col not in ['ID', 'sequence']]

    # Ensure all required features are present in the dataframe
    if not all(feature in features_df.columns for feature in model_features):
        print("Error: The input data is missing one or more features the model was trained on.")
        return None
    
    # Select and reorder columns to match the model's training data
    X = features_df[model_features]

    # --- NEW: Apply normalization if scaler is provided ---
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    # Perform prediction
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]  # Probability of the positive class (AMP)

    results_df = pd.DataFrame({
        'ID': features_df['ID'],
        'sequence': features_df['sequence'],
        'prediction': predictions,
        'probability_AMP': probabilities
    })

    return results_df