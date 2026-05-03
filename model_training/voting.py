# model_training/voting.py
#
# Soft-voting ensemble with per-model scaling.
# Each base estimator is paired with a fitted scaler stored internally,
# so predict_proba accepts raw (unscaled) feature matrices.

import numpy as np
import pandas as pd


class VotingEnsemble:
    """Soft-voting ensemble that manages per-model scaling internally.

    Parameters
    ----------
    estimators : list of (name, fitted_estimator)
    scalers    : dict of key -> fitted scaler
    scaler_map : dict of name -> key in scalers
    """

    def __init__(self, estimators, scalers, scaler_map):
        self.estimators = estimators
        self.scalers    = scalers
        self.scaler_map = scaler_map

    def predict_proba(self, X):
        probas = []
        columns = X.columns if hasattr(X, "columns") else None
        index   = X.index   if hasattr(X, "index")   else None
        for name, model in self.estimators:
            key  = self.scaler_map[name]
            arr  = self.scalers[key].transform(X)
            X_sc = (pd.DataFrame(arr, columns=columns, index=index)
                    if columns is not None else arr)
            probas.append(model.predict_proba(X_sc)[:, 1])
        avg = np.mean(probas, axis=0)
        return np.column_stack([1.0 - avg, avg])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
