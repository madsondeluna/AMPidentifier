# model_training/tune.py
#
# Hyperparameter optimization for RF, SVM, and GB classifiers.
#
# Strategy:
#   - RandomizedSearchCV over each model's parameter space
#   - StratifiedKFold(n_splits=5) to preserve class balance in each fold
#   - Scoring: roc_auc (more informative than accuracy for binary classification)
#   - The test set is held out entirely; tuning happens only on training data
#   - Tuned models are saved to model_training/tuned_model/
#
# Run from the project root:
#   python -m model_training.tune            (all models)
#   python -m model_training.tune rf gb      (specific models)
#   python -m model_training.tune svm        (SVM only)
#
# Note: SVM with poly/rbf kernels on ~5k samples is slow. Expect 10-20 min.

import os
import sys
import time
import pandas as pd
import joblib
from scipy.stats import loguniform, randint, uniform

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef
)

from amp_identifier.feature_extraction import calculate_physicochemical_features
from amp_identifier.data_io import load_fasta_sequences

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "model_training/data"
TUNED_DIR = "model_training/tuned_model"
POSITIVE_FILE = os.path.join(DATA_DIR, "positive_sequences.fasta")
NEGATIVE_FILE = os.path.join(DATA_DIR, "negative_sequences.fasta")

RANDOM_STATE = 42   # must match train.py to reproduce the same split
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER = 50         # random combinations per model
SCORING = "roc_auc"
N_JOBS = -1         # use all available cores


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_and_prepare():
    """
    Reproduce the exact train/test split used in train.py (same RANDOM_STATE
    and stratify=y) so that the tuned models are evaluated on the same held-out
    test set as the baseline models.
    """
    print("Loading sequences...")
    pos_seqs, pos_ids = load_fasta_sequences(POSITIVE_FILE)
    neg_seqs, neg_ids = load_fasta_sequences(NEGATIVE_FILE)

    sequences = pos_seqs + neg_seqs
    ids = pos_ids + neg_ids
    labels = [1] * len(pos_seqs) + [0] * len(neg_seqs)

    print(f"  Positive: {len(pos_seqs)}, Negative: {len(neg_seqs)}")

    print("Extracting features...")
    features_df = calculate_physicochemical_features(sequences, ids)
    features_df["label"] = labels

    X = features_df.drop(columns=["ID", "sequence", "label"]).fillna(0)
    y = features_df["label"]

    print(f"  Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Fit scaler on training data only; transform both sets
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    print(f"  Train set: {X_train_sc.shape[0]} | Test set: {X_test_sc.shape[0]}")
    return X_train_sc, X_test_sc, y_train, y_test


# ---------------------------------------------------------------------------
# Parameter search spaces
# ---------------------------------------------------------------------------
def get_search_spaces():
    """
    Parameter distributions for RandomizedSearchCV.

    RF:
      - n_estimators, max_depth, min_samples_split/leaf, max_features

    SVM:
      - C (log-uniform), gamma (categorical + log-scale values), kernel
      - Note: poly kernel can be significantly slower than rbf/linear

    GB:
      - GradientBoostingClassifier does not support class_weight; not needed
        here because the training set is balanced (1:1 ratio).
      - n_estimators, learning_rate (log-uniform), max_depth, subsample,
        min_samples_split/leaf
    """
    return {
        "rf": {
            "model": RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS
            ),
            "params": {
                "n_estimators": randint(100, 600),
                "max_depth": [None, 10, 20, 30, 40],
                "min_samples_split": randint(2, 15),
                "min_samples_leaf": randint(1, 8),
                "max_features": ["sqrt", "log2", 0.3, 0.5],
            },
        },
        "svm": {
            "model": SVC(
                probability=True, class_weight="balanced", random_state=RANDOM_STATE,
                max_iter=5000,  # hard limit per fold; prevents runaway linear+high-C combos
            ),
            "params": {
                # C capped at 100: kernel=linear with C > 100 on normalized data
                # converges very slowly (>17 min/fold at C=375) with negligible gain.
                "C": loguniform(1e-2, 1e2),
                "kernel": ["rbf", "linear", "poly"],
                # gamma applies to rbf/poly; 'scale' and 'auto' are valid for all kernels
                "gamma": ["scale", "auto", 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            },
        },
        "gb": {
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": randint(100, 500),
                "learning_rate": loguniform(1e-3, 5e-1),
                "max_depth": randint(2, 8),
                "subsample": uniform(0.5, 0.5),  # uniform in [0.5, 1.0]
                "min_samples_split": randint(2, 15),
                "min_samples_leaf": randint(1, 8),
            },
        },
        "xgb": {
            # XGBoost: regularized gradient boosting with L1/L2 penalties on leaf weights.
            # scale_pos_weight=1 because dataset is balanced (1:1).
            # use_label_encoder removed in XGBoost >= 1.6.
            "model": XGBClassifier(
                random_state=RANDOM_STATE,
                scale_pos_weight=1,
                eval_metric="logloss",
                verbosity=0,
                n_jobs=1,  # parallelism handled by RandomizedSearchCV n_jobs
            ),
            "params": {
                "n_estimators": randint(100, 500),
                "learning_rate": loguniform(1e-3, 5e-1),
                "max_depth": randint(2, 8),
                "subsample": uniform(0.5, 0.5),          # row subsampling [0.5, 1.0]
                "colsample_bytree": uniform(0.5, 0.5),   # column subsampling [0.5, 1.0]
                "reg_alpha": loguniform(1e-4, 1e1),      # L1 regularization
                "reg_lambda": loguniform(1e-1, 1e1),     # L2 regularization
                "min_child_weight": randint(1, 10),
            },
        },
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_on_test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(TUNED_DIR, exist_ok=True)

    # Optional: pass model names as arguments, e.g. "rf gb" or "svm"
    requested = [a.lower() for a in sys.argv[1:]]
    valid = {"rf", "svm", "gb", "xgb"}
    if requested:
        invalid = set(requested) - valid
        if invalid:
            print(f"Unknown model(s): {invalid}. Choose from: {valid}")
            sys.exit(1)

    X_train, X_test, y_train, y_test = load_and_prepare()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    spaces = get_search_spaces()

    if requested:
        spaces = {k: v for k, v in spaces.items() if k in requested}

    summary_rows = []
    report_lines = [
        "########### Hyperparameter Tuning Report ###########",
        "",
        f"CV strategy : StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True)",
        f"Scoring     : {SCORING}",
        f"n_iter      : {N_ITER}",
        f"random_state: {RANDOM_STATE}",
        "",
    ]

    for name, cfg in spaces.items():
        print(f"\n{'='*54}")
        print(f" Tuning {name.upper()}  (n_iter={N_ITER}, cv={CV_FOLDS})")
        print(f"{'='*54}")

        search = RandomizedSearchCV(
            estimator=cfg["model"],
            param_distributions=cfg["params"],
            n_iter=N_ITER,
            scoring=SCORING,
            cv=cv,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            verbose=2,
            refit=True,
        )
        t0 = time.time()
        search.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"\n  Elapsed: {elapsed/60:.1f} min")

        best_model = search.best_estimator_
        best_cv_score = search.best_score_
        metrics = evaluate_on_test(best_model, X_test, y_test)

        print(f"\n  Best CV {SCORING}  : {best_cv_score:.4f}")
        print(f"  Test AUC-ROC : {metrics['auc_roc']:.4f}")
        print(f"  Test MCC     : {metrics['mcc']:.4f}")
        print(f"  Best params  : {search.best_params_}")

        model_path = os.path.join(TUNED_DIR, f"amp_model_{name}_tuned.pkl")
        joblib.dump(best_model, model_path)
        print(f"  Saved -> {model_path}")

        # Save cv_results_ for hyperparameter exploration plots
        cv_results_path = os.path.join(TUNED_DIR, f"cv_results_{name}.csv")
        pd.DataFrame(search.cv_results_).to_csv(cv_results_path, index=False)
        print(f"  CV results -> {cv_results_path}")

        # Save per-model result (does not overwrite other models)
        model_result_path = os.path.join(TUNED_DIR, f"result_{name}.csv")
        pd.DataFrame([{
            "model": name.upper(),
            "best_cv_roc_auc": round(best_cv_score, 4),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        }]).to_csv(model_result_path, index=False)
        print(f"  Result     -> {model_result_path}")

        summary_rows.append({
            "model": name.upper(),
            "best_cv_roc_auc": round(best_cv_score, 4),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        })

        # Build text report section
        report_lines += [
            f"{'='*54}",
            f" MODEL: {name.upper()}",
            f"{'='*54}",
            f" Best CV {SCORING}  : {best_cv_score:.4f}",
            f" Test Accuracy   : {metrics['accuracy']:.4f}",
            f" Test Precision  : {metrics['precision']:.4f}",
            f" Test Recall     : {metrics['recall']:.4f}",
            f" Test Specificity: {metrics['specificity']:.4f}",
            f" Test F1         : {metrics['f1']:.4f}",
            f" Test MCC        : {metrics['mcc']:.4f}",
            f" Test AUC-ROC    : {metrics['auc_roc']:.4f}",
            f" TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}",
            "",
            " Best hyperparameters:",
        ]
        for k, v in sorted(search.best_params_.items()):
            report_lines.append(f"   {k}: {v}")
        report_lines.append("")

    # Save text report
    txt_path = os.path.join(TUNED_DIR, "tuning_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nText report -> {txt_path}")

    # Save CSV summary
    csv_path = os.path.join(TUNED_DIR, "tuning_report.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"CSV report  -> {csv_path}")

    print("\n--- Tuning complete ---")


if __name__ == "__main__":
    main()
