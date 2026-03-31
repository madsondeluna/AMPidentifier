# AMPidentifier
> A Tool for Antimicrobial Peptide (AMP) Prediction and Fast Physicochemical Assessment

```
////////////////////////////////////////////////////////////////////////
//                                                                    //
//                                                                    //
//      _    __  __ ____  _     _            _   _  __ _              //
//     / \  |  \/  |  _ \(_) __| | ___ _ __ | |_(_)/ _(_) ___ _ __    //
//    / _ \ | |\/| | |_) | |/ _` |/ _ \ '_ \| __| | |_| |/ _ \ '__|   //
//   / ___ \| |  | |  __/| | (_| |  __/ | | | |_| |  _| |  __/ |      //
//  /_/   \_\_|  |_|_|   |_|\__,_|\___|_| |_|\__|_|_| |_|\___|_|      //
//                                                                    //
//                                                                    //
////////////////////////////////////////////////////////////////////////

```

## Table of Contents

- [AMPidentifier](#ampidentifier)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Key Updates](#key-updates)
    - [Feature Improved](#feature-improved)
  - [Tool Workflow](#tool-workflow)
    - [Workflow Steps](#workflow-steps)
    - [Key Characteristics](#key-characteristics)
    - [Quick Links Map](#quick-links-map)
  - [Key Features](#key-features)
  - [Installation](#installation)
  - [Quick Test](#quick-test)
  - [How to Use (CLI)](#how-to-use-cli)
    - [Arguments](#arguments)
    - [Examples](#examples)
  - [Pre-Trained Internal Models](#pre-trained-internal-models)
    - [Performance Summary](#performance-summary)
  - [Ensemble Mode Performance](#ensemble-mode-performance)
  - [Outputs](#outputs)
  - [Training Your Own Models](#training-your-own-models)
  - [Project Structure](#project-structure)
    - [Key Components](#key-components)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
    - [Optimization Objective](#optimization-objective)
    - [Cross-Validation Protocol](#cross-validation-protocol)
    - [Search Spaces](#search-spaces)
    - [Tuning Results and Figures](#tuning-results-and-figures)
    - [Final Performance Metrics](#final-performance-metrics)
  - [Contributors](#contributors)
    - [Lead Developer](#lead-developer)
    - [Collaborators](#collaborators)
    - [Advisory Team](#advisory-team)
    - [Quick Reference (tabular)](#quick-reference-tabular)
  - [Funding \& Acknowledgments](#funding--acknowledgments)
  - [Intellectual Property](#intellectual-property)
  - [Contributing](#contributing)
    - [Reporting Issues](#reporting-issues)
      - [Reporting a Bug](#reporting-a-bug)
      - [Suggesting Features or Improvements](#suggesting-features-or-improvements)
    - [Feature Requests \& Roadmap](#feature-requests--roadmap)
    - [Code of Conduct](#code-of-conduct)
  - [How to Cite](#how-to-cite)

---

## About

The **AMPidentifier** is a Python tool for predicting and analyzing Antimicrobial Peptides (AMPs) from amino-acid sequences. It leverages a set of pre-trained Machine Learning models and offers flexible prediction modes, including an ensemble voting system, to provide robust results.

**Unlike web servers or closed-source tools**, AMPidentifier operates as a **fully open and modular framework**. It includes pre-trained models (Random Forest, SVM, Gradient Boosting, and XGBoost) that work both **individually** and in **ensemble mode**. Users can also **integrate external models** (`.pkl` files) to expand their analyses and compare different approaches side-by-side.

Beyond classification, AMPidentifier computes and exports dozens of physicochemical descriptors for each sequence (via `modlamp`) and bundles them into a detailed report.

---

## Key Updates

### Feature Improved
- **XGBoost added**: Regularized gradient boosting (L1/L2 penalties) included as a fourth internal model
- **Hyperparameter optimization**: All models tuned via RandomizedSearchCV with StratifiedKFold(5) cross-validation, scored by AUC-ROC
- **Improved Accuracy**: Random Forest model achieves 88.45% accuracy (was lower without normalization)
- **Better SVM Performance**: SVM benefits significantly from normalized features
- **Consistent Predictions**: Scaler ensures reproducible results across runs


## Tool Workflow

<p align="center">
  <img src="/img/workflow.svg" alt="AMPidentifier Workflow Diagram"/>
</p>

The AMPidentifier pipeline follows a modular workflow that processes peptide sequences through feature extraction and machine learning-based classification:

### Workflow Steps

1. **Input FASTA File**
   - Users provide amino acid sequences in standard FASTA format
   - Multiple sequences can be processed in a single run

2. **AMPidentifier CLI (`main.py`)**
   - Command-line interface serving as the entry point
   - Orchestrates the entire prediction pipeline
   - Handles user arguments and configuration

3. **Parallel Processing Branches**

   **Branch A: Feature Extraction**
   - Computes physicochemical descriptors using `modlamp` library
   - Applies StandardScaler normalization (essential for model performance)
   - Generates `physicochemical_features.csv` with detailed sequence properties
   - These features serve as input for the prediction models

   **Branch B: Model Selection**
   - Users choose one of three prediction strategies:
     - **Single Model**: Select one algorithm (RF, SVM, GB, or XGB)
     - **Ensemble Mode**: Combines all four models through majority voting (recommended)
     - **External Models**: Load custom `.pkl` models for comparison

4. **Model Inference**
   - Applies selected model(s) to normalized features
   - Four internal models available:
     - **RF**: Random Forest (best single-model performance)
     - **SVM**: Support Vector Machine
     - **GB**: Gradient Boosting
     - **XGB**: XGBoost (regularized gradient boosting with L1/L2 penalties)
   - Optional: External models can be included for benchmarking

5. **Output Generation**
   - `prediction_comparison_report.csv`: Contains classification results
     - AMP vs non-AMP predictions
     - Confidence scores per model
     - Side-by-side model comparison
     - Consensus prediction (in ensemble mode)

### Key Characteristics

- **Modular Design**: Each component operates independently and can be used separately
- **Flexible Model Selection**: Supports single models, ensemble voting, and external model integration
- **Normalized Features**: StandardScaler ensures consistent and optimal model performance
- **Comprehensive Output**: Both feature tables and prediction reports are generated for downstream analysis

---

### Quick Links Map

| Step / Artifact                         | See Section                               |
|---------------------------------------- |-------------------------------------------|
| Input FASTA                             | [Arguments](#arguments)                    |
| CLI usage                               | [How to Use (CLI)](#how-to-use-cli)        |
| Physicochemical feature generation      | [Key Features](#key-features)              |
| Model selection / flags                 | [Arguments](#arguments)                    |
| Internal models overview                | [Pre-Trained Internal Models](#pre-trained-internal-models) |
| Outputs (features.csv, predictions.csv) | [Outputs](#outputs)                        |



---

## Key Features

- **Multiple Internal Models:** Four pre-trained ML models (Random Forest, Gradient Boosting, SVM, XGBoost).
- **Ensemble Voting:** Majority vote across all four internal models to improve robustness.
- **Model Selection:** Choose a specific internal model on demand.
- **External Model Comparison:** Load external `.pkl` models for side-by-side comparison.
- **Feature Generation:** Compute and export an extensive set of physicochemical descriptors.

---

## Installation

We recommend using a virtual environment.

```bash
git clone https://github.com/madsondeluna/AMPIdentifier.git
cd AMPIdentifier

# Create the environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Test

Run a quick prediction using the sample data shipped with the repository:

```bash
python3 main.py --input data-for-tests/sample_sequences.fasta --output_dir ./test_results --ensemble
```

If no errors occur and `test_results` is created with output files, your installation is working.

---

## How to Use (CLI)

The entry point is `main.py`.

---

<p align="center">
  <img src="/img/logo-use2.png" alt="AMPidentifer in use on terminal"/>
</p>

---

### Arguments

| Argument               | Description                                                                 | Required | Default |
|------------------------|-----------------------------------------------------------------------------|:--------:|:-------:|
| `-i, --input`          | Path to the input FASTA file                                                |   Yes    |   -     |
| `-o, --output_dir`     | Path to the output directory                                                |   Yes    |   -     |
| `-m, --model`          | Internal model to use: `rf`, `svm`, `gb`, `xgb`                            |    No    |  `rf`   |
| `--ensemble`           | Enable majority-vote ensemble across all four internal models               |    No    |  Flag   |
| `-e, --external_models`| One or more paths to external `.pkl` models for comparison (comma-separated)|    No    |   -     |

### Examples

Single-model (Random Forest, default):
```bash
python3 main.py --input my_sequences.fasta --output_dir ./results_rf
```

Ensemble voting:
```bash
python3 main.py --input my_sequences.fasta --output_dir ./results_ensemble --ensemble
```

Compare SVM with an external model:
```bash
python3 main.py --input my_sequences.fasta --output_dir ./compare_svm --model svm --external_models /path/to/my_model.pkl
```

---

## Pre-Trained Internal Models

All four models were optimized via `RandomizedSearchCV` with `StratifiedKFold(5)` cross-validation (scoring: AUC-ROC, n\_iter=50) and evaluated on a held-out test set (20% split, n=530 per class). See [Hyperparameter Optimization](#hyperparameter-optimization) for full methodology.

### Performance Summary

Best values per metric are in **bold**. The Ensemble column applies majority voting across all four tuned models.

| Metric | RF | SVM | GB | XGB | **Ensemble** |
|---|:---:|:---:|:---:|:---:|:---:|
| Accuracy | 0.8898 | 0.8698 | 0.8891 | 0.8883 | **0.8951** |
| Precision | 0.8940 | 0.8571 | 0.8926 | 0.8924 | **0.9093** |
| Sensitivity (Recall) | 0.8845 | **0.8875** | 0.8845 | 0.8830 | 0.8777 |
| Specificity | 0.8951 | 0.8521 | 0.8936 | 0.8936 | **0.9125** |
| F1-Score | 0.8892 | 0.8721 | 0.8886 | 0.8877 | **0.8932** |
| MCC | 0.7797 | 0.7401 | 0.7781 | 0.7766 | **0.7907** |
| AUC-ROC | 0.9510 | 0.9360 | 0.9534 | **0.9540** | 0.9583 |
| Best CV AUC-ROC | 0.9532 | 0.9436 | 0.9538 | 0.9527 | - |

**Recommended:** Use **Ensemble Mode** (`--ensemble`). The ensemble achieves the best accuracy, precision, specificity, F1, and MCC across all configurations.

**Single-model default:** Random Forest (`-m rf`) offers the best individual precision-recall balance.

---

## Ensemble Mode Performance

The confusion matrix below summarizes the ensemble vote (majority rule, all four tuned models) on the held-out test set (n = 2,650 sequences; 1,325 per class).

|  | **Predicted: 0 (Non-AMP)** | **Predicted: 1 (AMP)** | **Total** |
|:---|:---:|:---:|:---:|
| **Actual: 0 (Non-AMP)** | TN = 1,209 (91.25%) | FP = 116 (8.75%) | 1,325 |
| **Actual: 1 (AMP)** | FN = 162 (12.23%) | TP = 1,163 (87.77%) | 1,325 |
| **Predicted total** | 1,371 | 1,279 | **2,650** |

**Ensemble metrics (tuned models):**

| Accuracy | Precision | Sensitivity | Specificity | F1-Score | MCC | AUC-ROC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 89.51% | 90.93% | 87.77% | 91.25% | 89.32% | 0.7907 | 0.9583 |

The ensemble reduces false positives relative to individual models (FP = 116 vs. up to 196 for SVM alone), making it the preferred configuration for AMP candidate prioritization where laboratory follow-up is resource-constrained.

---

## Outputs

- `physicochemical_features.csv`: Detailed table of computed descriptors.
- `prediction_comparison_report.csv`: Final predictions, including a column for each model used.

---

## Training Your Own Models

Use the scripts under `model_training/`, especially `train.py`, to build and evaluate models on your datasets.

---

## Project Structure

```text
AMPidentifier/
├── .gitignore                  # Instruct Git to ignore files (e.g., virtual env)
├── LICENSE                     # Software license (e.g., MIT)
├── README.md                   # Main project documentation
├── requirements.txt            # Python dependencies
├── main.py                     # CLI entry point for end users
│
├── amp_identifier/             # Main application package
│   ├── __init__.py             # Makes this directory a Python package
│   ├── core.py                 # Orchestrates the main prediction workflow
│   ├── data_io.py              # Input readers (e.g., FASTA)
│   ├── feature_extraction.py   # Physicochemical descriptor computation
│   ├── prediction.py           # Load .pkl models and run inference
│   └── reporting.py            # Generate .csv reports
│
├── normalization-info/         # Documentation about StandardScaler implementation
│   ├── README.md               # Index of normalization documentation
│   ├── normalization_impact_report.md  # Technical report (English)
│   ├── resumo_normalizacao.md          # Executive summary (Portuguese)
│   ├── quick_start_normalized.md       # Quick start guide
│   ├── changelog.md            # Complete changelog
│   └── verify_normalization.py # Verification script
│
├── data-for-tests/             # Example data for quick tests
│   ├── sequences_to_predict.fasta      # Multi-FASTA with example sequences
│   └── results_ensemble/               # Example output directory
│       ├── physicochemical_features.csv
│       └── prediction_comparison_report.csv
│
├── model_training/             # Isolated module for training and evaluation
│   ├── __init__.py             # Package initializer
│   ├── train.py                # Train ML models with StandardScaler normalization
│   ├── evaluate.py             # Evaluate trained models and compute metrics
│   │
│   ├── data/                   # Training/testing data
│   │   ├── positive_sequences.fasta  # Positive (AMP) sequences for training
│   │   ├── negative_sequences.fasta  # Negative (non-AMP) sequences for training
│   │   ├── test_features.csv         # (Generated) Normalized test-set features
│   │   └── test_labels.csv           # (Generated) Test-set labels
│   │
│   └── saved_model/            # Trained artifacts and evaluation outputs
│       ├── feature_scaler.pkl        # (Generated) StandardScaler (REQUIRED)
│       ├── amp_model_rf.pkl          # (Generated) Random Forest model
│       ├── amp_model_svm.pkl         # (Generated) SVM model
│       ├── amp_model_gb.pkl          # (Generated) Gradient Boosting model
│       ├── amp_model_xgb.pkl         # (Generated) XGBoost model
│       ├── evaluation_report.txt     # (Generated) Detailed text report
│       └── evaluation_report.csv     # (Generated) Comparative CSV report
│
├── benchmarking/               # Benchmarking datasets and results
│   ├── base/                   # Base datasets for benchmarking
│   └── results/                # Benchmark results and comparisons
│
├── img/                        # Images directory
│   └── logo-use.png            # Terminal usage screenshot
│
└── tests/                      # Unit tests to ensure code quality
    ├── __init__.py             # Package initializer
    └── test_prediction.py      # Tests for prediction functions
```

### Key Components

- **Modular Design**: Each component is independent and can be used separately or as part of the full pipeline.
- **Pre-trained Models**: Four models (RF, SVM, GB, XGB) ready to use individually or in ensemble mode.
- **External Model Support**: Users can load their own `.pkl` models for comparison and extended analysis.

---

## Hyperparameter Optimization

All internal models were subjected to rigorous hyperparameter search using `RandomizedSearchCV` with `StratifiedKFold` cross-validation. The complete procedure is implemented in `model_training/tune.py`.

### Optimization Objective

The optimization criterion was the Area Under the Receiver Operating Characteristic Curve (AUC-ROC), which measures discrimination capacity across all decision thresholds:

$$\text{AUC-ROC} = \int_0^1 \text{TPR}(t)\,d\,\text{FPR}(t)$$

where $\text{TPR}(t) = \frac{\text{TP}}{\text{TP}+\text{FN}}$ (sensitivity) and $\text{FPR}(t) = \frac{\text{FP}}{\text{FP}+\text{TN}}$ (1 - specificity) at threshold $t$.

AUC-ROC was preferred over accuracy because it is threshold-independent and more informative for binary classification.

### Cross-Validation Protocol

Hyperparameter search was performed exclusively on the training partition ($N_\text{train} = 2{,}120$, 80% of the dataset); the test set ($N_\text{test} = 530$ per class) was held out entirely. For each of the $n_\text{iter} = 50$ randomly sampled configurations $\theta$, the cross-validation score was:

$$\hat{s}(\theta) = \frac{1}{K} \sum_{k=1}^{K} \text{AUC-ROC}\!\left(f_\theta^{(k)},\, \mathcal{D}_k^\text{val}\right), \quad K = 5$$

`StratifiedKFold` preserves the class ratio in every fold:

$$n_k^+ = n_k^- = \left\lfloor \frac{N_\text{train}}{2K} \right\rfloor \approx 212 \text{ samples per class per fold}$$

The optimal configuration is selected as:

$$\theta^* = \arg\max_{\theta \in \Theta_\text{random}} \hat{s}(\theta)$$

The final model is retrained on the full training set with $\theta^*$ (`refit=True`).

### Search Spaces

#### Random Forest (RF)

RF aggregates $T$ decision trees by majority vote, each grown on a bootstrap sample considering $m \leq p$ random features per split:

$$\hat{y}_\text{RF}(x) = \text{mode}\left\{h_t(x)\right\}_{t=1}^{T}$$

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| `n_estimators` ($T$) | $\mathcal{U}_\mathbb{Z}$ | $[100,\ 600]$ |
| `max_depth` | Discrete | $\{\text{None},\ 10,\ 20,\ 30,\ 40\}$ |
| `min_samples_split` | $\mathcal{U}_\mathbb{Z}$ | $[2,\ 15]$ |
| `min_samples_leaf` | $\mathcal{U}_\mathbb{Z}$ | $[1,\ 8]$ |
| `max_features` ($m$) | Discrete | $\{\sqrt{p},\ \log_2 p,\ 0.3p,\ 0.5p\}$ |

#### Support Vector Machine (SVM)

The SVM finds the maximum-margin separating hyperplane with slack variables $\xi_i$:

$$\min_{w,b,\xi}\; \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i \quad \text{s.t.}\quad y_i\!\left(w^\top\phi(x_i)+b\right) \geq 1 - \xi_i,\quad \xi_i \geq 0$$

For the RBF kernel: $K(x_i, x_j) = \exp\!\left(-\gamma\|x_i - x_j\|^2\right)$.

$C$ was bounded to $[10^{-2},\ 10^2]$; values above $10^2$ with `kernel=linear` caused convergence times exceeding 17 min/fold with negligible gain. `max_iter=5000` was set as a hard limit per fold.

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| $C$ | $\log\mathcal{U}$ | $[10^{-2},\ 10^{2}]$ |
| `kernel` | Discrete | $\{\text{rbf},\ \text{linear},\ \text{poly}\}$ |
| $\gamma$ | Discrete | $\{\text{scale},\ \text{auto},\ 10^{-4},\ 10^{-3},\ 10^{-2},\ 10^{-1},\ 1\}$ |

#### Gradient Boosting (GB)

GB constructs an additive model by fitting successive trees to the negative gradient of the loss $\mathcal{L}$:

$$F_T(x) = F_0(x) + \sum_{t=1}^{T} \nu \cdot h_t(x)$$

where $\nu$ (learning rate) controls shrinkage and each $h_t$ is the least-squares fit to the residuals $g_i = -\partial\mathcal{L}/\partial F_{t-1}(x_i)$.

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| `n_estimators` ($T$) | $\mathcal{U}_\mathbb{Z}$ | $[100,\ 500]$ |
| `learning_rate` ($\nu$) | $\log\mathcal{U}$ | $[10^{-3},\ 5\times10^{-1}]$ |
| `max_depth` | $\mathcal{U}_\mathbb{Z}$ | $[2,\ 8]$ |
| `subsample` | $\mathcal{U}$ | $[0.5,\ 1.0]$ |
| `min_samples_split` | $\mathcal{U}_\mathbb{Z}$ | $[2,\ 15]$ |
| `min_samples_leaf` | $\mathcal{U}_\mathbb{Z}$ | $[1,\ 8]$ |

#### XGBoost (XGB)

XGBoost extends gradient boosting with explicit L1 ($\alpha$) and L2 ($\lambda$) regularization on leaf weights $w_j$:

$$\mathcal{L}_\text{XGB} = \mathcal{L} + \sum_{t=1}^{T}\!\left[\lambda\sum_j w_j^2 + \alpha\sum_j |w_j|\right]$$

Row (`subsample`) and column (`colsample_bytree`) subsampling further reduce variance.

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| `n_estimators` ($T$) | $\mathcal{U}_\mathbb{Z}$ | $[100,\ 500]$ |
| `learning_rate` ($\nu$) | $\log\mathcal{U}$ | $[10^{-3},\ 5\times10^{-1}]$ |
| `max_depth` | $\mathcal{U}_\mathbb{Z}$ | $[2,\ 8]$ |
| `subsample` | $\mathcal{U}$ | $[0.5,\ 1.0]$ |
| `colsample_bytree` | $\mathcal{U}$ | $[0.5,\ 1.0]$ |
| `reg_alpha` ($\alpha$, L1) | $\log\mathcal{U}$ | $[10^{-4},\ 10]$ |
| `reg_lambda` ($\lambda$, L2) | $\log\mathcal{U}$ | $[10^{-1},\ 10]$ |
| `min_child_weight` | $\mathcal{U}_\mathbb{Z}$ | $[1,\ 10]$ |

### Tuning Results and Figures

#### ROC Curves

<p align="center">
  <img src="model_training/tuned_model/figures/fig01_roc_curves.png" alt="ROC curves for all tuned classifiers" width="380"/>
</p>

All four tuned models achieve AUC-ROC above 0.93. XGBoost (0.954) and GB (0.953) lead, followed by RF (0.951). SVM achieves 0.936, reflecting the more constrained decision boundary of kernel methods on this feature set. The close grouping of curves confirms that all classifiers successfully capture the physicochemical signal that distinguishes AMPs from non-AMPs.

#### Confusion Matrices

<p align="center">
  <img src="model_training/tuned_model/figures/fig02_confusion_matrices.png" alt="Confusion matrices on held-out test set" width="700"/>
</p>

RF, GB, and XGB show similar TP/TN counts (~1,170/1,184) with balanced FP/FN rates. SVM exhibits a higher FP count (196), reflecting its tendency to favour sensitivity over specificity with the selected kernel parameters. MCC and accuracy values annotated above each panel quantify model quality.

#### Probability Calibration

<p align="center">
  <img src="model_training/tuned_model/figures/fig03_calibration.png" alt="Probability calibration curves" width="380"/>
</p>

Calibration curves (reliability diagrams) assess whether $P(\hat{y}=1 \mid \hat{p}=p) \approx p$. All models track the diagonal reasonably well, meaning that `predict_proba` outputs can be interpreted as AMP probabilities. Slight under-confidence (curves above the diagonal) at intermediate ranges is a known property of boosting classifiers.

#### Feature Importance

<p align="center">
  <img src="model_training/tuned_model/figures/fig04_feature_importance.png" alt="Physicochemical feature importance per classifier" width="700"/>
</p>

Tree-based models (RF, GB, XGB) report Gini impurity-based importance; SVM uses permutation importance on the test set (AUC-ROC reduction, 10 repeats). **Charge** and **Charge Density** rank as the most discriminative features across all classifiers, consistent with the electrostatic mechanism of AMP-membrane interaction. **Boman Index** and **Hydrophobicity Ratio** emerge as secondary predictors. SVM's permutation scores also highlight **Length** and **Molecular Weight**, capturing structural constraints that complement the charge-centric signal captured by tree ensembles.

#### CV Score Distribution

<p align="center">
  <img src="model_training/tuned_model/figures/fig05_cv_score_distribution.png" alt="CV AUC-ROC distribution across 50 candidates" width="380"/>
</p>

The strip plot shows the mean CV AUC-ROC ($\pm$SD over 5 folds) for all 50 randomly sampled configurations per model. The horizontal line marks the median and the star marks $\theta^*$. GB and XGB display narrower, higher distributions, reflecting lower sensitivity to individual hyperparameter choices. SVM shows a broader spread, confirming greater hyperparameter sensitivity. The selected optimum consistently lies in the upper tail.

#### Top 10 Candidates

<p align="center">
  <img src="model_training/tuned_model/figures/fig06_top10_candidates.png" alt="Top 10 hyperparameter combinations by CV AUC-ROC" width="700"/>
</p>

The top 10 hyperparameter combinations per model ranked by mean CV AUC-ROC; the full-opacity bar indicates $\theta^*$. The tight score clustering for RF, GB, and XGB (range < 0.01) indicates a flat optimum landscape where the models are robust to moderate hyperparameter variation. SVM shows a slightly wider spread (~0.03), consistent with its broader search distribution.

#### Hyperparameter Search Spaces

The four panels below show, for each model, how key hyperparameters relate to CV AUC-ROC across all 50 evaluated configurations. The star marks $\theta^*$.

**Random Forest:**

<p align="center">
  <img src="model_training/tuned_model/figures/fig07_hyperparam_rf.png" alt="RF hyperparameter search" width="700"/>
</p>

`n_estimators` above ~250 shows saturating returns. `max_depth=None` (fully grown trees) achieves competitive scores when combined with conservative `min_samples_leaf` values, confirming that ensemble averaging provides sufficient implicit regularization.

**Gradient Boosting:**

<p align="center">
  <img src="model_training/tuned_model/figures/fig08_hyperparam_gb.png" alt="GB hyperparameter search" width="700"/>
</p>

Low-to-moderate `learning_rate` ($\nu \in [0.01,\ 0.1]$) combined with higher `n_estimators` achieves the best results, consistent with the shrinkage-depth trade-off in boosting. `max_depth` between 3 and 5 is preferred.

**Support Vector Machine:**

<p align="center">
  <img src="model_training/tuned_model/figures/fig09_hyperparam_svm.png" alt="SVM hyperparameter search" width="700"/>
</p>

$C$ in the range $[1,\ 10]$ achieves the best CV scores with the RBF kernel. At very low $C$ the model underfits; at very high $C$ convergence slows with marginal gain. $\gamma \leq 10^{-2}$ or `scale` is preferred.

**XGBoost:**

<p align="center">
  <img src="model_training/tuned_model/figures/fig10_hyperparam_xgb.png" alt="XGB hyperparameter search" width="700"/>
</p>

The L2 regularization term ($\lambda$) shows a moderate positive correlation with CV AUC-ROC at intermediate values, confirming that explicit regularization benefits this dataset size. The optimal `learning_rate` falls around 0.05–0.15, consistent with GB.

### Final Performance Metrics

After selecting $\theta^*$ for each model and evaluating on the held-out test set, the ensemble combines all four tuned models by majority vote (AUC-ROC uses mean predicted probabilities).

<p align="center">
  <img src="model_training/tuned_model/figures/fig11_metrics_comparison.png" alt="Classification performance across tuned models and ensemble" width="700"/>
</p>

| Metric | RF | SVM | GB | XGB | **Ensemble** |
|---|:---:|:---:|:---:|:---:|:---:|
| Accuracy | 0.8898 | 0.8698 | 0.8891 | 0.8883 | **0.8951** |
| Precision | 0.8940 | 0.8571 | 0.8926 | 0.8924 | **0.9093** |
| Sensitivity (Recall) | 0.8845 | **0.8875** | 0.8845 | 0.8830 | 0.8777 |
| Specificity | 0.8951 | 0.8521 | 0.8936 | 0.8936 | **0.9125** |
| F1-Score | 0.8892 | 0.8721 | 0.8886 | 0.8877 | **0.8932** |
| MCC | 0.7797 | 0.7401 | 0.7781 | 0.7766 | **0.7907** |
| AUC-ROC | 0.9510 | 0.9360 | 0.9534 | **0.9540** | 0.9583 |
| Best CV AUC-ROC | 0.9532 | 0.9436 | 0.9538 | 0.9527 | - |

The ensemble systematically outperforms all individual models on accuracy, precision, specificity, F1, and MCC. The slight reduction in sensitivity relative to SVM reflects the ensemble's bias toward reducing false positives, which is the preferred trade-off for wet-lab candidate prioritization.

## Contributors

### Lead Developer

- **Madson A. de Luna Aragão** - PhD Candidate in Bioinformatics, UFMG  
  Belo Horizonte, Minas Gerais, Brazil  
  **Responsibilities:** project lead, software architecture, ML pipelines, documentation.  
  **Contacts:** madsondeluna@gmail.com 

### Collaborators

- **Rafael L. da Silva** - Masters Student, UFPE - Collaborator  
  **Contributions:** data preprocessing, pipeline testing, literature review.

### Advisory Team

- **Ana M. Benko‑Iseppon, PhD** - Principal Investigator, UFPE - Advisor  
  **Contributions:** scientific supervision, study design, biological validation.

- **João Pacífico, PhD** - Principal Investigator, UPE - Co‑Advisor  
  **Contributions:** computational analysis review, dataset curation, evaluation protocol, reproducibility.

- **Carlos A. dos Santos-Silva, PhD** - Professor, CESMAC - Co‑Advisor  
  **Contributions:** structural biology expertise, evaluation protocol, benchmarking strategy, reproducibility.

---

### Quick Reference (tabular)

| Name                       | Role / Responsibilities                                   | Affiliation | Location         |
|----------------------------|------------------------------------------------------------|-------------|------------------|
| Madson A. de Luna-Aragão, MSc  | Lead developer; architecture; ML; docs                     | UFMG        | Belo Horizonte, BR |
| Rafael L. da Silva, BSc        | Collaborator; preprocessing; pipeline testing; lit. review | UFPE        | Recife, BR       |
| Ana M. Benko‑Iseppon, PhD | Advisor; study design; review, validation                  | UFPE        | Recife, BR       |
| João Pacífico, PhD        | Co-Advisor; computational review; evaluation       | UPE         | Petrolina, BR       |
| Carlos A. dos Santos-Silva, PhD      | Co‑Advisor; pipeline testing, review    | CESMAC        | Maceió, BR       |


---

## Funding & Acknowledgments

- **Principal Holder:** This software is officially registered under the **UFPE** - Universidade Federal de Pernambuco (Federal University of Pernambuco, Brazil).
- This research was supported by **FACEPE** - Fundação de Amparo à Pesquisa do Estado de Pernambuco (Brazil).
- We acknowledge the **PPGGBM** - Programa de Pós-Graduação em Genética e Biologia Molecular (Graduate Program in Genetics and Molecular Biology) at UFPE for institutional support.

---

## Contributing

### Reporting Issues

#### Reporting a Bug

When reporting a bug, please include:

1. **Clear Title**: Brief description of the problem
2. **Environment Details**:
   - Operating System (macOS, Linux, Windows)
   - Python version (`python3 --version`)
   - AMPidentifier version/commit
3. **Steps to Reproduce**:
   - Exact commands you ran
   - Input files (if possible, share a minimal example)
4. **Expected vs Actual Behavior**:
   - What you expected to happen
   - What actually happened
5. **Error Messages**:
   - Full error traceback
   - Log files (if applicable)

**Example Bug Report:**
```
Title: "Ensemble mode fails with external models on macOS"

Environment:
- macOS 14.2
- Python 3.11.5
- Commit: abc123

Steps to reproduce:
1. Run: python3 main.py --input test.fasta --output_dir ./out --ensemble --external_models custom.pkl
2. Error occurs during model loading

Expected: All models should load and run ensemble prediction
Actual: KeyError when loading external model

Error message:
KeyError: 'feature_names'
[full traceback here]
```

#### Suggesting Features or Improvements

When suggesting a new feature:

1. **Clear Title**: Concise feature description
2. **Use Case**: Explain why this feature would be useful
3. **Proposed Solution**: Describe how you envision it working
4. **Alternatives**: Any alternative approaches you've considered
5. **Additional Context**: Examples, references, or mockups

**Example Feature Request:**
```
Title: "Add support for CSV input format"

Use Case:
Many users have peptide sequences in CSV files with additional metadata.
Supporting CSV input would eliminate the need for format conversion.

Proposed Solution:
Add a --format flag:
python3 main.py --input sequences.csv --format csv --output_dir ./results

CSV should have columns: id, sequence, [optional metadata]

Alternatives:
- Provide a conversion script (less convenient)
- Support Excel files directly (more complex)

Additional Context:
Similar tools like ToolX support CSV input via pandas.
```

### Feature Requests & Roadmap

We're constantly working to improve AMPidentifier. Some areas we're exploring:

- **Activity-specific models**: Separate models for antibacterial, antifungal, and antiviral peptides
- **Deep learning integration**: Support for transformer-based models
- **Web interface**: Browser-based GUI for easier access
- **API endpoint**: RESTful API for programmatic access
- **Additional descriptors**: Integration with more feature calculation libraries

If you have ideas for other features, please open an issue with the tag `enhancement`!

### Code of Conduct

- Be respectful and constructive
- Provide clear and detailed information
- Focus on the problem, not the person
- Help create a welcoming environment for all contributors

---

## Intellectual Property

- This tool is **officially registered** with the **INPI** - Instituto Nacional da Propriedade Industrial (Brazilian National Institute of Industrial Property).
- **Registration Number:** BR 51 2025 005859-4
- **Registration Date:** November 18, 2025
- **Title:** AMPidentifier: A modular python toolkit for predicting antimicrobial peptides using ensemble machine learning
- **Registered Authors:** Madson A. de Luna Aragão, Rafael L. da Silva, João Pacífico, Carlos A. dos Santos-Silva, Ana M. Benko-Iseppon
- All rights reserved. Usage and distribution are subject to the project license terms.

---

## How to Cite

If this tool or its outputs support your research, please cite the repository:

```text
Luna-Aragão, M. A., da Silva, R. L., Pacífico, J., Santos-Silva, C. A. & Benko‑Iseppon, A. M. (2025). AMPidentifier: A Python toolkit for predicting antimicrobial peptides using ensemble machine learning and physicochemical descriptors. GitHub repository. https://github.com/madsondeluna/AMPIdentifier
```
