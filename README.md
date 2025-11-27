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
  - [Benchmarking (Using the Ensemble Mode) - Real Data](#benchmarking-using-the-ensemble-mode---real-data)
    - [Table Explanation](#table-explanation)
    - [The four central quadrants represent the classification results:](#the-four-central-quadrants-represent-the-classification-results)
      - [Understanding Type I and Type II Errors](#understanding-type-i-and-type-ii-errors)
  - [Outputs](#outputs)
  - [Training Your Own Models](#training-your-own-models)
  - [Project Structure](#project-structure)
    - [Key Components](#key-components)
  - [Comparative Benchmarking](#comparative-benchmarking)
    - [Benchmark Dataset Description](#benchmark-dataset-description)
    - [Comparison with State-of-the-Art Tools](#comparison-with-state-of-the-art-tools)
  - [Contributors](#contributors)
    - [Lead Developer](#lead-developer)
    - [Collaborators](#collaborators)
    - [Advisory Team](#advisory-team)
    - [Quick Reference (tabular)](#quick-reference-tabular)
  - [Funding \& Acknowledgments](#funding--acknowledgments)
  - [Intellectual Property](#intellectual-property)
  - [How to Cite](#how-to-cite)

---

## About

The **AMPidentifier** is a Python tool for predicting and analyzing Antimicrobial Peptides (AMPs) from amino-acid sequences. It leverages a set of pre-trained Machine Learning models with **StandardScaler normalization** and offers flexible prediction modes, including an ensemble voting system, to provide robust results.

**Unlike web servers or closed-source tools**, AMPidentifier operates as a **fully open and modular framework**. It includes pre-trained models (Random Forest, SVM, Gradient Boosting) that work both **individually** and in **ensemble mode**. Users can also **integrate external models** (`.pkl` files) to expand their analyses and compare different approaches side-by-side.

Beyond classification, AMPidentifier computes and exports dozens of physicochemical descriptors for each sequence (via `modlamp`) and bundles them into a detailed report.

---

## Key Updates

### Feature Improved 
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
     - **Single Model**: Select one algorithm (RF, SVM, or GB)
     - **Ensemble Mode**: Combines all three models through majority voting (recommended)
     - **External Models**: Load custom `.pkl` models for comparison

4. **Model Inference**
   - Applies selected model(s) to normalized features
   - Three internal models available:
     - **RF**: Random Forest (best single-model performance)
     - **SVM**: Support Vector Machine
     - **GB**: Gradient Boosting
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

- **Multiple Internal Models:** Three pre-trained ML models (Random Forest, Gradient Boosting, SVM).
- **Ensemble Voting:** Majority vote across internal models to improve robustness.
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
| `-m, --model`          | Internal model to use: `rf`, `svm`, `gb`                                    |    No    |  `rf`   |
| `--ensemble`           | Enable majority-vote ensemble across all internal models                    |    No    |  Flag   |
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

Three models are distributed and evaluated on the same dataset for fair comparison. All models are trained with **StandardScaler normalization** for optimal performance.

### Performance Summary

Best values per metric are in **bold**.

| Metric         | Random Forest (RF) | Support Vector Machine (SVM) | Gradient Boosting (GB) |
|----------------|--------------------:|------------------------------:|-----------------------:|
| Accuracy       | **0.8845**         | 0.8740                        | 0.8585                 |
| Precision      | **0.8910**         | 0.8880                        | 0.8665                 |
| Recall         | **0.8762**         | 0.8558                        | 0.8475                 |
| Specificity    | **0.8928**         | 0.8921                        | 0.8694                 |
| F1-Score       | **0.8836**         | 0.8716                        | 0.8569                 |
| MCC            | **0.7692**         | 0.7484                        | 0.7172                 |
| AUC-ROC        | **0.9503**         | 0.9356                        | 0.9289                 |

**Best Practice (Recommended by the authors):** Use **Ensemble Mode** (`--ensemble`)
- Combines all three models through majority voting
- Leverages the strengths of each algorithm
- Provides more robust and reliable predictions
- Supported by literature as the best approach for AMP classification
- All models have excellent metrics (>85% accuracy, >0.92 AUC-ROC)

**For Single Model Usage:** Random Forest (RF)
- Best overall performance across all metrics
- Highest accuracy (88.45%) and AUC-ROC (0.9503)
- Excellent balance between sensitivity and specificity

---

## Benchmarking (Using the Ensemble Mode) - Real Data 

**Performance with Normalized Models (StandardScaler)**

|                                | **Predicted: 0** (Negative) | **Predicted: 1** (Positive) | **Actual Total** |
| :----------------------------- | :-------------------------: | :-------------------------: | :--------------: |
| **Actual: 0** (Negative Dataset) | **TN = 1179** (88.98%)     | **FP = 146** (11.02%)       |       1325       |
| **Actual: 1** (Positive Dataset) | **FN = 186** (14.04%)      | **TP = 1139** (85.96%)      |       1325       |
| **Predicted Total**             |            1365             |            1285             |    **2650**      |

**Ensemble Performance Metrics:**
- **Accuracy:** 87.47%
- **Sensitivity (Recall):** 85.96%
- **Specificity:** 88.98%

### Table Explanation

This table is a confusion matrix, a fundamental tool for evaluating the performance of a classification model. It compares the actual values from your data with the predictions made by the model.

- **Rows (Actual):** Represent the true class of each sample.  
  - *Actual: 0*: Samples that are truly negative (from your "negative dataset").  
  - *Actual: 1*: Samples that are truly positive (from your "positive dataset").  

- **Columns (Predicted):** Represent the class that the model assigned to each sample.  
  - *Predicted: 0*: Samples that the model classified as negative.  
  - *Predicted: 1*: Samples that the model classified as positive.  

---

### The four central quadrants represent the classification results:

- **TN (True Negative):**  
  - Value: 1179  
  - Meaning: The ensemble correctly predicted 1179 samples as negative, and they were indeed negative.  
  - The rate of 88.98% (1179/1325) represents the model's **specificity**.  

- **FP (False Positive):**  
  - Value: 146  
  - Meaning: The ensemble incorrectly predicted 146 samples as positive when they were actually negative.  
  - This is also known as a **Type I Error**.  

- **FN (False Negative):**  
  - Value: 186  
  - Meaning: The ensemble incorrectly predicted 186 samples as negative when they were actually positive.  
  - This is also known as a **Type II Error**.  

- **TP (True Positive):**  
  - Value: 1139  
  - Meaning: The ensemble correctly predicted 1139 samples as positive, and they were indeed positive.  
  - The rate of 85.96% (1139/1325) represents the model's **sensitivity (or recall)**.  

#### Understanding Type I and Type II Errors

In statistical hypothesis testing and machine learning classification:

- **Type I Error (False Positive - FP):**  
  - **Definition:** Rejecting a true null hypothesis; predicting positive when the actual class is negative.
  - **In AMP context:** Classifying a non-AMP peptide as an AMP.
  - **Consequence:** Wasted resources (time, money, lab work) investigating peptides that don't have antimicrobial activity.
  - **Control:** Reducing Type I errors increases **specificity** but may increase Type II errors.

- **Type II Error (False Negative - FN):**  
  - **Definition:** Failing to reject a false null hypothesis; predicting negative when the actual class is positive.
  - **In AMP context:** Classifying a true AMP peptide as non-AMP.
  - **Consequence:** Missing potentially valuable antimicrobial peptides that could be therapeutic candidates.
  - **Control:** Reducing Type II errors increases **sensitivity** but may increase Type I errors.

**Trade-off in AMPidentifier:**  
The ensemble model is calibrated to minimize False Positives (Type I errors) while maintaining good sensitivity. This is preferable for AMP screening because:
- High confidence in positive predictions (low FP rate of 11.02%)
- Efficient use of laboratory resources
- Some true AMPs may be missed (FN rate of 14.04%), but can be recovered in subsequent screening rounds

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
- **Pre-trained Models**: Three models (RF, SVM, GB) ready to use individually or in ensemble mode.
- **External Model Support**: Users can load their own `.pkl` models for comparison and extended analysis.

---

## Comparative Benchmarking

AMPidentifier was designed to provide a robust, open-source alternative to existing AMP prediction tools. To ensure transparency and reproducibility, we benchmark AMPidentifier against other widely-used AMP classifiers, including both web-based servers and command-line tools.

This comparison uses a standardized test dataset from the independent benchmark study by **Zulfiqar et al. (2024)**: "Machine Learning-Assisted Prediction and Generation of Antimicrobial Peptides" ([Small Sci. 2024, 2400579](https://onlinelibrary.wiley.com/doi/10.1002/smsc.202400579)). All tools are evaluated on the same dataset to ensure fair and unbiased comparison across multiple performance metrics. 

**Important:** AMPidentifier will be evaluated using its **ensemble mode** (default recommended configuration), which combines predictions from all three internal models (Random Forest, SVM, Gradient Boosting) through majority voting. This represents the tool's best practice configuration as recommended by the authors and supported by literature as the optimal approach for AMP classification.

The goal is to position AMPidentifier within the current landscape of AMP prediction tools and highlight its strengths in ensemble learning and modular design.

### Benchmark Dataset Description

The benchmark dataset used for comparative evaluation is organized by antimicrobial activity type and contains balanced positive (AMP) and negative (non-AMP) sequences:

| Activity Type    | Positive Sequences | Negative Sequences | Total Sequences |
|------------------|-------------------:|-------------------:|----------------:|
| Antibacterial    | 984                | 984                | 1,968           |
| Antifungal       | 1,384              | 1,384              | 2,768           |
| Antiviral        | 739                | 739                | 1,478           |
| **Total**        | **3,107**          | **3,107**          | **6,214**       |

All sequences are stored in `benchmarking/base/` directory as FASTA files (`bacterial_pos.fasta`, `bacterial_neg.fasta`, `fungal_pos.fasta`, `fungal_neg.fasta`, `viral_pos.fasta`, `viral_neg.fasta`).

**Dataset Source:** Independent test set from Zulfiqar et al. (2024), [DOI: 10.1002/smsc.202400579](https://onlinelibrary.wiley.com/doi/10.1002/smsc.202400579)

### Comparison with State-of-the-Art Tools

| Tool                | Acc (%) | Antibacterial | Antifungal | Antiviral | Type       | Open Source | Modular | Models Available | Ensemble | External Models |
|---------------------|--------:|:-------------:|:----------:|:---------:|------------|:-----------:|:-------:|:----------------:|:--------:|:---------------:|
| **AMPidentifier**   | -       | -             | -          | -         | CLI/Local  | Yes         | Yes     | RF, SVM, GB      | Yes      | Yes             |
| AMPScanner v2       | -       | -             | -          | -         | Web        | -           | No      | -                | -        | No              |
| iAMP-2L             | -       | -             | -          | -         | Web        | -           | No      | -                | -        | No              |
| CAMPR3              | -       | -             | -          | -         | Web        | -           | Yes     | SVM, RF, ANN, DA | -        | No              |
| AMPlify             | -       | -             | -          | -         | CLI/Web    | -           | -       | -                | -        | -               |
| AMPDiscover         | -       | -             | -          | -         | CLI        | -           | -       | -                | -        | -               |

**Column Descriptions:**
- **Acc (%):** Overall accuracy percentage on the benchmark dataset, calculated as (TP + TN) / (TP + TN + FP + FN) × 100, where TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives
- **Antibacterial:** Accuracy on antibacterial peptide subset, calculated as correctly predicted antibacterial sequences / total antibacterial sequences × 100
- **Antifungal:** Accuracy on antifungal peptide subset, calculated as correctly predicted antifungal sequences / total antifungal sequences × 100
- **Antiviral:** Accuracy on antiviral peptide subset, calculated as correctly predicted antiviral sequences / total antiviral sequences × 100
- **Type:** Deployment format (Web-based server, CLI tool, or Local application)
- **Open Source:** Publicly available source code (Yes/No)
- **Modular:** Can individual models be used separately? (Yes/No)
- **Models Available:** Machine learning algorithms available in the tool (RF = Random Forest, SVM = Support Vector Machine, GB = Gradient Boosting, ANN = Artificial Neural Network, DA = Discriminant Analysis)
- **Ensemble:** Supports ensemble/voting prediction across multiple models? (Yes/No)
- **External Models:** Allows integration of user-provided custom models (.pkl)? (Yes/No)

**Benchmarking Status:**

Comparative benchmarking experiments against other AMP prediction tools are currently **in progress**. Performance metrics (accuracy values) will be updated as experiments are completed using the standardized independent test set described above. All tools will be evaluated under identical conditions to ensure fair and unbiased comparison.

**Note:** Complete validation results for AMPidentifier's internal models (Random Forest, SVM, Gradient Boosting) are already available in the [Pre-Trained Internal Models](#pre-trained-internal-models) section, including detailed performance metrics (Accuracy, Precision, Recall, Specificity, F1-Score, MCC, AUC-ROC) and confusion matrix analysis in the [Benchmarking (Using the Ensemble Mode)](#benchmarking-using-the-ensemble-mode---real-data) section.

---

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
