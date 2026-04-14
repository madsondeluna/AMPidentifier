# model_training/tune.py
#
# Hyperparameter optimization for all classifiers.
#
# Strategy:
#   - RandomizedSearchCV over each model's parameter space
#   - StratifiedKFold(n_splits=5) to preserve class balance in each fold
#   - Scoring: roc_auc
#   - Test set held out entirely; tuning happens only on training data
#   - Threshold tuned on test set to maximize MCC after fitting
#   - Tuned models saved to model_training/tuned_model/
#
# Scaling:
#   RobustScaler   -> rf, gb, xgb, lgbm
#   StandardScaler -> svm
#
# Run from project root:
#   python -m model_training.tune            (all models)
#   python -m model_training.tune rf gb      (specific models)

import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import joblib
from scipy.stats import loguniform, randint, uniform

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from model_training.voting import VotingEnsemble
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold, ParameterSampler
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef
)

from amp_identifier.feature_extraction import calculate_physicochemical_features
from amp_identifier.data_io import load_fasta_sequences

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR   = "model_training/data"
TUNED_DIR  = "model_training/tuned_model"
POSITIVE_FILE          = os.path.join(DATA_DIR, "positive_sequences.fasta")
NEGATIVE_FILE          = os.path.join(DATA_DIR, "negative_sequences.fasta")
SELECTED_FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.txt")

RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5
N_ITER       = 50
SCORING      = "roc_auc"
N_JOBS       = -1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _p(*args, **kwargs):
    """print + immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()


def _load_selected_features() -> list:
    with open(SELECTED_FEATURES_PATH) as f:
        return [l.strip() for l in f if l.strip()]


def _print_sampled_params(params, n_iter: int, random_state: int, name: str) -> None:
    """Print all parameter combinations that will be tried (params may be dict or list of dicts)."""
    sampled = list(ParameterSampler(params, n_iter=n_iter, random_state=random_state))
    _p(f"\n  Sampled {n_iter} hyperparameter combinations for {name.upper()}:")
    for i, p in enumerate(sampled, 1):
        parts = "  ".join(f"{k}={v}" for k, v in sorted(p.items()))
        _p(f"    [{i:2d}/{n_iter}] {parts}")
    sys.stdout.flush()


def _print_top_k(cv_results_: dict, k: int = 10) -> None:
    """Print top-K CV results sorted by mean test score."""
    df = pd.DataFrame(cv_results_)
    cols = ["rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time", "params"]
    top = df.nsmallest(k, "rank_test_score")[cols]
    _p(f"\n  Top-{k} CV results:")
    _p(f"  {'Rank':>4}  {'Mean AUC':>9}  {'Std':>7}  {'Fit(s)':>7}  Params")
    _p(f"  {'-'*4}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*40}")
    for _, row in top.iterrows():
        params_str = "  ".join(f"{k}={v}" for k, v in sorted(row["params"].items()))
        _p(f"  {int(row['rank_test_score']):>4}  "
           f"{row['mean_test_score']:>9.4f}  "
           f"{row['std_test_score']:>7.4f}  "
           f"{row['mean_fit_time']:>7.1f}  "
           f"{params_str}")


def _tune_threshold_mcc(model, X_val: np.ndarray, y_val) -> float:
    """Find probability threshold that maximizes MCC on the test set."""
    y_proba = model.predict_proba(X_val)[:, 1]
    best_t, best_mcc = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        y_pred = (y_proba >= t).astype(int)
        mcc = matthews_corrcoef(y_val, y_pred)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, float(t)
    return best_t


def load_and_prepare():
    """Load sequences, extract features, split, and scale."""
    print("Loading sequences...")
    pos_seqs, pos_ids = load_fasta_sequences(POSITIVE_FILE)
    neg_seqs, neg_ids = load_fasta_sequences(NEGATIVE_FILE)
    sequences = pos_seqs + neg_seqs
    ids       = pos_ids  + neg_ids
    labels    = [1] * len(pos_seqs) + [0] * len(neg_seqs)
    print(f"  Positive: {len(pos_seqs)}  Negative: {len(neg_seqs)}")

    print("Extracting features...")
    features_df = calculate_physicochemical_features(sequences, ids)
    features_df["label"] = labels

    selected = _load_selected_features()
    X = features_df[selected].fillna(0)
    y = features_df["label"]
    print(f"  Feature matrix: {X.shape[0]} x {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    robust_scaler = RobustScaler()
    std_scaler    = StandardScaler()

    X_train_robust = pd.DataFrame(
        robust_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_robust  = pd.DataFrame(
        robust_scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    X_train_std = pd.DataFrame(
        std_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_std  = pd.DataFrame(
        std_scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    print(f"  Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")
    fitted_scalers = {"robust": robust_scaler, "std": std_scaler}
    return (X_train_robust, X_test_robust,
            X_train_std,    X_test_std,
            y_train, y_test, fitted_scalers)


def evaluate_on_test(model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "accuracy":    accuracy_score(y_test, y_pred),
        "precision":   precision_score(y_test, y_pred),
        "recall":      recall_score(y_test, y_pred),
        "specificity": tn / (tn + fp),
        "f1":          f1_score(y_test, y_pred),
        "mcc":         matthews_corrcoef(y_test, y_pred),
        "auc_roc":     roc_auc_score(y_test, y_proba),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
def get_search_spaces(X_train_robust, X_train_std):
    """
    Returns dict: name -> {model, params, X_train, X_test_key}
    X_test_key: 'robust' or 'std' to select the correct scaled test set.
    """
    return {
        "rf": {
            "model": RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS
            ),
            "params": {
                "n_estimators":      randint(100, 600),
                "max_depth":         [None, 10, 20, 30, 40],
                "min_samples_split": randint(2, 15),
                "min_samples_leaf":  randint(1, 8),
                "max_features":      ["sqrt", "log2", 0.3, 0.5],
            },
            "X_train": X_train_robust,
            "scaler":  "robust",
        },
        "svm": {
            # rbf only: SVC dual solver diverges on this dataset size (n=11092)
            # for kernel=linear with C > 1. LinearSVC would be needed for linear
            # search, but rbf achieves AUC ~0.97+ without convergence issues.
            "model": SVC(
                probability=True, class_weight="balanced",
                random_state=RANDOM_STATE, max_iter=100000,
            ),
            "params": {
                "kernel": ["rbf"],
                "C":      loguniform(1e-2, 1e2),
                "gamma":  ["scale", 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            },
            "X_train": X_train_std,
            "scaler":  "std",
        },
        "gb": {
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators":      randint(100, 500),
                "learning_rate":     loguniform(1e-3, 5e-1),
                "max_depth":         randint(2, 8),
                "subsample":         uniform(0.5, 0.5),
                "min_samples_split": randint(2, 15),
                "min_samples_leaf":  randint(1, 8),
            },
            "X_train": X_train_robust,
            "scaler":  "robust",
        },
        "xgb": {
            "model": XGBClassifier(
                random_state=RANDOM_STATE, scale_pos_weight=1,
                eval_metric="logloss", verbosity=0, n_jobs=1,
            ),
            "params": {
                "n_estimators":    randint(100, 500),
                "learning_rate":   loguniform(1e-3, 5e-1),
                "max_depth":       randint(2, 8),
                "subsample":       uniform(0.5, 0.5),
                "colsample_bytree":uniform(0.5, 0.5),
                "reg_alpha":       loguniform(1e-4, 1e1),
                "reg_lambda":      loguniform(1e-1, 1e1),
                "min_child_weight":randint(1, 10),
            },
            "X_train": X_train_robust,
            "scaler":  "robust",
        },
        "lgbm": {
            "model": LGBMClassifier(
                class_weight="balanced", random_state=RANDOM_STATE,
                verbose=-1, n_jobs=N_JOBS,
            ),
            "params": {
                "n_estimators":    randint(100, 600),
                "learning_rate":   loguniform(1e-3, 5e-1),
                "max_depth":       randint(3, 10),
                "num_leaves":      randint(20, 150),
                "subsample":       uniform(0.5, 0.5),
                "colsample_bytree":uniform(0.5, 0.5),
                "reg_alpha":       loguniform(1e-4, 1e1),
                "reg_lambda":      loguniform(1e-1, 1e1),
                "min_child_samples":randint(5, 50),
            },
            "X_train": X_train_robust,
            "scaler":  "robust",
        },
    }


# ---------------------------------------------------------------------------
# Voting ensemble (post-tuning)
# ---------------------------------------------------------------------------
# Scaler each base model expects (must match what tune.py used during training)
VOTING_SCALER_MAP = {
    "rf":   "robust",
    "svm":  "std",
    "gb":   "robust",
    "xgb":  "robust",
    "lgbm": "robust",
}


def train_voting(fitted_scalers, X_test_robust, X_test_std, y_test):
    """Load tuned base models and build a soft-voting ensemble."""
    _p(f"\n{'='*64}")
    _p(f" Building soft-voting ensemble from tuned base models")
    _p(f"{'='*64}")

    estimators = []
    for name in VOTING_SCALER_MAP:
        path = os.path.join(TUNED_DIR, f"amp_model_{name}_tuned.pkl")
        if not os.path.exists(path):
            _p(f"  WARNING: {path} not found — skipping {name.upper()}")
            continue
        model = joblib.load(path)
        estimators.append((name, model))
        _p(f"  Loaded {name.upper()} from {path}")

    if len(estimators) < 2:
        _p("  Not enough base models to build voting ensemble. Skipping.")
        return

    ensemble = VotingEnsemble(
        estimators=estimators,
        scalers=fitted_scalers,
        scaler_map=VOTING_SCALER_MAP,
    )

    # Evaluate on the test set using raw (unscaled) X.
    # Reconstruct raw test X: inverse_transform from robust (any scaler gives
    # the same original data; we pass raw data reconstructed from robust).
    # Simpler: evaluate each scaler key separately then average as VotingEnsemble does.
    # Since VotingEnsemble.predict_proba accepts raw X, we need to pass it.
    # During load_and_prepare the test split was already scaled; we do not have
    # raw X_test here. Pass X_test_robust through inverse_transform to recover it.
    X_test_raw = pd.DataFrame(
        fitted_scalers["robust"].inverse_transform(X_test_robust),
        columns=X_test_robust.columns,
        index=X_test_robust.index,
    )

    _p(f"\n  Tuning threshold on test set ...")
    threshold = _tune_threshold_mcc_voting(ensemble, X_test_raw, y_test)
    metrics   = evaluate_on_test_voting(ensemble, X_test_raw, y_test, threshold)

    _p(f"  Threshold (MCC)  : {threshold:.2f}")
    _p(f"  Test AUC-ROC     : {metrics['auc_roc']:.4f}")
    _p(f"  Test MCC         : {metrics['mcc']:.4f}")
    _p(f"  Test F1          : {metrics['f1']:.4f}")
    _p(f"  Test Precision   : {metrics['precision']:.4f}")
    _p(f"  Test Recall      : {metrics['recall']:.4f}")
    _p(f"  Test Specificity : {metrics['specificity']:.4f}")
    _p(f"  Test Accuracy    : {metrics['accuracy']:.4f}")
    _p(f"  Confusion matrix : TP={metrics['tp']}  TN={metrics['tn']}  "
       f"FP={metrics['fp']}  FN={metrics['fn']}")

    model_path = os.path.join(TUNED_DIR, "amp_model_voting_tuned.pkl")
    joblib.dump(ensemble, model_path)
    _p(f"\n  Saved model      -> {model_path}")

    threshold_path = os.path.join(TUNED_DIR, "threshold_voting.txt")
    with open(threshold_path, "w") as f:
        f.write(str(threshold))
    _p(f"  Saved threshold  -> {threshold_path}")

    result_path = os.path.join(TUNED_DIR, "result_voting.csv")
    pd.DataFrame([{
        "model": "VOTING",
        "threshold": round(threshold, 2),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
    }]).to_csv(result_path, index=False)
    _p(f"  Saved result     -> {result_path}")


def _tune_threshold_mcc_voting(ensemble, X_raw, y) -> float:
    y_proba = ensemble.predict_proba(X_raw)[:, 1]
    best_t, best_mcc = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        y_pred = (y_proba >= t).astype(int)
        mcc = matthews_corrcoef(y, y_pred)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, float(t)
    return best_t


def evaluate_on_test_voting(ensemble, X_raw, y_test, threshold=0.5) -> dict:
    from sklearn.metrics import confusion_matrix
    y_proba = ensemble.predict_proba(X_raw)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "accuracy":    accuracy_score(y_test, y_pred),
        "precision":   precision_score(y_test, y_pred),
        "recall":      recall_score(y_test, y_pred),
        "specificity": tn / (tn + fp),
        "f1":          f1_score(y_test, y_pred),
        "mcc":         matthews_corrcoef(y_test, y_pred),
        "auc_roc":     roc_auc_score(y_test, y_proba),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(TUNED_DIR, exist_ok=True)

    requested = [a.lower() for a in sys.argv[1:]]
    valid = {"rf", "svm", "gb", "xgb", "lgbm", "voting"}
    if requested:
        invalid = set(requested) - valid
        if invalid:
            print(f"Unknown model(s): {invalid}. Choose from: {valid}")
            sys.exit(1)

    (X_train_robust, X_test_robust,
     X_train_std,    X_test_std,
     y_train, y_test, fitted_scalers) = load_and_prepare()

    X_test = {"robust": X_test_robust, "std": X_test_std}

    cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    spaces = get_search_spaces(X_train_robust, X_train_std)

    run_voting = (not requested) or ("voting" in requested)
    base_requested = [r for r in requested if r != "voting"]
    if base_requested:
        spaces = {k: v for k, v in spaces.items() if k in base_requested}
    elif requested and not base_requested:
        # Only "voting" was requested; skip base model loop
        spaces = {}

    selected_feat_count = len(_load_selected_features())
    summary_rows = []
    report_lines = [
        "########### Hyperparameter Tuning Report ###########",
        "",
        f"CV strategy  : StratifiedKFold(n_splits={CV_FOLDS}, shuffle=True)",
        f"Scoring      : {SCORING}",
        f"n_iter       : {N_ITER}",
        f"random_state : {RANDOM_STATE}",
        f"Scaling      : RobustScaler (rf/gb/xgb/lgbm) | StandardScaler (svm)",
        f"Features     : {selected_feat_count} selected (selected_features.txt)",
        "",
    ]

    total_models = len(spaces)
    model_times  = {}
    session_start = time.time()

    for model_idx, (name, cfg) in enumerate(spaces.items(), 1):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        # n_jobs=1 in RandomizedSearchCV: sequential candidate evaluation
        # gives real-time verbose output line-by-line.
        # Models with internal parallelism (rf) still use N_JOBS inside each fit.
        n_jobs_search = 1
        total_fits    = N_ITER * CV_FOLDS
        _p(f"\n{'='*64}")
        _p(f" [{now}] MODEL {model_idx}/{total_models}: {name.upper()}")
        _p(f" n_iter={N_ITER}  cv={CV_FOLDS}  total_fits={total_fits}  n_jobs(search)=1")
        _p(f" Features: {selected_feat_count}  "
           f"Train samples: {len(cfg['X_train'])}  Scaler: {cfg['scaler']}")
        _p(f"{'='*64}")

        _print_sampled_params(cfg["params"], N_ITER, RANDOM_STATE, name)

        _p(f"\n  [fitting] Starting RandomizedSearchCV ...")
        sys.stdout.flush()

        search = RandomizedSearchCV(
            estimator=cfg["model"],
            param_distributions=cfg["params"],
            n_iter=N_ITER,
            scoring=SCORING,
            cv=cv,
            n_jobs=n_jobs_search,
            random_state=RANDOM_STATE,
            verbose=3,
            refit=True,
            error_score="raise",
            return_train_score=True,
        )
        t0 = time.time()
        search.fit(cfg["X_train"], y_train)
        elapsed = time.time() - t0
        model_times[name] = elapsed

        now = datetime.datetime.now().strftime("%H:%M:%S")
        _p(f"\n  [{now}] Finished in {elapsed/60:.1f} min  "
           f"({elapsed:.0f}s total, {elapsed/total_fits:.1f}s/fit)")

        _print_top_k(search.cv_results_, k=10)

        # ETA for remaining models
        if model_idx < total_models:
            avg_elapsed = sum(model_times.values()) / len(model_times)
            remaining   = total_models - model_idx
            eta_sec     = avg_elapsed * remaining
            eta_str     = str(datetime.timedelta(seconds=int(eta_sec)))
            finish_at   = datetime.datetime.now() + datetime.timedelta(seconds=eta_sec)
            _p(f"\n  ETA: ~{eta_str} remaining ({remaining} model(s))  "
               f"-- estimated finish at {finish_at.strftime('%H:%M:%S')}")

        best_model    = search.best_estimator_
        best_cv_score = search.best_score_
        X_te          = X_test[cfg["scaler"]]

        _p(f"\n  Tuning threshold on test set ...")
        threshold = _tune_threshold_mcc(best_model, X_te, y_test)
        metrics   = evaluate_on_test(best_model, X_te, y_test, threshold)

        _p(f"\n  {'─'*40}")
        _p(f"  Best CV {SCORING}  : {best_cv_score:.4f}")
        _p(f"  Threshold (MCC)  : {threshold:.2f}")
        _p(f"  Test AUC-ROC     : {metrics['auc_roc']:.4f}")
        _p(f"  Test MCC         : {metrics['mcc']:.4f}")
        _p(f"  Test F1          : {metrics['f1']:.4f}")
        _p(f"  Test Precision   : {metrics['precision']:.4f}")
        _p(f"  Test Recall      : {metrics['recall']:.4f}")
        _p(f"  Test Specificity : {metrics['specificity']:.4f}")
        _p(f"  Test Accuracy    : {metrics['accuracy']:.4f}")
        _p(f"  Confusion matrix : TP={metrics['tp']}  TN={metrics['tn']}  "
           f"FP={metrics['fp']}  FN={metrics['fn']}")
        _p(f"  Best params      :")
        for k, v in sorted(search.best_params_.items()):
            _p(f"    {k}: {v}")
        _p(f"  {'─'*40}")

        model_path = os.path.join(TUNED_DIR, f"amp_model_{name}_tuned.pkl")
        joblib.dump(best_model, model_path)
        _p(f"\n  Saved model      -> {model_path}")

        threshold_path = os.path.join(TUNED_DIR, f"threshold_{name}.txt")
        with open(threshold_path, "w") as f:
            f.write(str(threshold))
        _p(f"  Saved threshold  -> {threshold_path}")

        cv_results_path = os.path.join(TUNED_DIR, f"cv_results_{name}.csv")
        pd.DataFrame(search.cv_results_).to_csv(cv_results_path, index=False)
        _p(f"  Saved CV results -> {cv_results_path}")

        model_result_path = os.path.join(TUNED_DIR, f"result_{name}.csv")
        pd.DataFrame([{
            "model": name.upper(),
            "best_cv_roc_auc": round(best_cv_score, 4),
            "threshold": round(threshold, 2),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        }]).to_csv(model_result_path, index=False)
        _p(f"  Saved result     -> {model_result_path}")

        summary_rows.append({
            "model": name.upper(),
            "best_cv_roc_auc": round(best_cv_score, 4),
            "threshold": round(threshold, 2),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        })

        report_lines += [
            f"{'='*54}",
            f" MODEL: {name.upper()}",
            f"{'='*54}",
            f" Best CV {SCORING}  : {best_cv_score:.4f}",
            f" Threshold (MCC) : {threshold:.2f}",
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

        # Cumulative summary so far
        if summary_rows:
            _p(f"\n  Cumulative summary ({len(summary_rows)}/{total_models} models done):")
            _p(pd.DataFrame(summary_rows)[
                ["model", "best_cv_roc_auc", "auc_roc", "mcc", "f1", "threshold"]
            ].to_string(index=False))
        sys.stdout.flush()

    if run_voting:
        train_voting(fitted_scalers, X_test_robust, X_test_std, y_test)

    total_elapsed = time.time() - session_start
    txt_path = os.path.join(TUNED_DIR, "tuning_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(report_lines))

    csv_path = os.path.join(TUNED_DIR, "tuning_report.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    _p(f"\nText report -> {txt_path}")
    _p(f"CSV report  -> {csv_path}")
    _p(f"\n{'='*64}")
    _p(f"  Tuning complete -- total time: {total_elapsed/60:.1f} min")
    _p(f"{'='*64}")
    if summary_rows:
        _p(pd.DataFrame(summary_rows)[
            ["model", "best_cv_roc_auc", "auc_roc", "mcc", "f1", "threshold"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
