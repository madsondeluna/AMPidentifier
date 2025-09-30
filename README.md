# AMPidentifier: A Tool for Antimicrobial Peptide (AMP) Prediction and Physicochemical Assessment

```python 

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

The **AMPidentifier** is a Python tool for predicting and analyzing Antimicrobial Peptides (AMPs) from amino-acid sequences. It leverages a set of pre-trained Machine Learning models and offers flexible prediction modes, including an ensemble voting system, to provide robust results.

Beyond classification, AMPidentifier computes and exports dozens of physicochemical descriptors for each sequence (via `modlamp`) and bundles them into a detailed report.

---


## Tool Workflow 

- [Input: FASTA file](#arguments)
  - processed by [AMPidentifier CLI](#how-to-use-cli)
    - → [Physicochemical Feature Extraction](#key-features)
      - produces [features.csv](#outputs)
    - → [Model Inference](#how-to-use-cli)
      - via [Model Selection](#arguments)
        - run a [Single Internal Model (RF, SVM, GB)](#pre-trained-internal-models)
        - or enable [Ensemble Mode (Voting)](#arguments)
        - or add [External Model Comparison](#arguments)
      - produces [predictions.csv](#outputs)

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
python -m venv venv

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
python main.py \
  --input data-for-tests/sample_sequences.fasta \
  --output_dir ./test_results \
  --ensemble
```

If no errors occur and `test_results` is created with output files, your installation is working.

---

## How to Use (CLI)

The entry point is `main.py`.

### Arguments

| Argument               | Description                                                                 | Required | Default |
|------------------------|-----------------------------------------------------------------------------|:--------:|:-------:|
| `-i, --input`          | Path to the input FASTA file                                                |   Yes    |   —     |
| `-o, --output_dir`     | Path to the output directory                                                |   Yes    |   —     |
| `-m, --model`          | Internal model to use: `rf`, `svm`, `gb`                                    |    No    |  `rf`   |
| `--ensemble`           | Enable majority-vote ensemble across all internal models                    |    No    |  Flag   |
| `-e, --external_models`| One or more paths to external `.pkl` models for comparison (comma-separated)|    No    |   —     |

### Examples

Single-model (Random Forest, default):
```bash
python main.py --input my_sequences.fasta --output_dir ./results_rf
```

Ensemble voting:
```bash
python main.py --input my_sequences.fasta --output_dir ./results_ensemble --ensemble
```

Compare SVM with an external model:
```bash
python main.py \
  --input my_sequences.fasta \
  --output_dir ./compare_svm \
  --model svm \
  --external_models /path/to/my_model.pkl
```

---

## Pre-Trained Internal Models

Three models are distributed and evaluated on the same dataset for fair comparison.

### Performance Summary

Best values per metric are in **bold**.

| Metric         | Random Forest (RF) | Gradient Boosting (GB) | Support Vector Machine (SVM) |
|----------------|--------------------:|-----------------------:|------------------------------:|
| Accuracy       | **0.8838**         | 0.8585                 | 0.5940                        |
| Precision      | **0.8903**         | 0.8665                 | 0.5828                        |
| Recall         | 0.8755             | 0.8475                 | **0.6611**                    |
| Specificity    | **0.8921**         | 0.8694                 | 0.5268                        |
| F1-Score       | **0.8828**         | 0.8569                 | 0.6195                        |
| MCC            | **0.7677**         | 0.7172                 | 0.1896                        |
| AUC-ROC        | **0.9503**         | 0.9289                 | 0.6377                        |

---

## Benchmarking (Using the Ensemble Mode) - Real Data 

|                                | **Predicted: 0** (Negative) | **Predicted: 1** (Positive) | **Actual Total** |
| :----------------------------- | :-------------------------: | :-------------------------: | :--------------: |
| **Actual: 0** (Negative Dataset) | **TN = 720** (73.25%)       | **FP = 263** (26.75%)       |       983        |
| **Actual: 1** (Positive Dataset) | **FN = 91** (9.26%)         | **TP = 892** (90.74%)       |       983        |
| **Predicted Total**             |             811             |            1155             |    **1966**      |

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
  - Value: 720  
  - Meaning: The model correctly predicted 720 samples as negative, and they were indeed negative.  
  - The rate of 73.25% (720/983) represents the model's **specificity**.  

- **FP (False Positive):**  
  - Value: 263  
  - Meaning: The model incorrectly predicted 263 samples as positive when they were actually negative.  
  - This is also known as a **Type I Error**.  

- **FN (False Negative):**  
  - Value: 91  
  - Meaning: The model incorrectly predicted 91 samples as negative when they were actually positive.  
  - This is also known as a **Type II Error**.  

- **TP (True Positive):**  
  - Value: 892  
  - Meaning: The model correctly predicted 892 samples as positive, and they were indeed positive.  
  - The rate of 90.74% (892/983) represents the model's **sensitivity (or recall)**.  

---

## Outputs

- `physicochemical_features.csv`: Detailed table of computed descriptors.
- `prediction_comparison_report.csv`: Final predictions, including a column for each model used.

---

## Training Your Own Models

Use the scripts under `model_training/`, especially `train.py`, to build and evaluate models on your datasets.

---

## Project Layout (Proposed)

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
├── data-for-tests/             # Example data for quick tests
│   └── sample_sequences.fasta  # Multi-FASTA with example sequences
│
├── model_training/             # Isolated module for training and evaluation
│   ├── __init__.py             # Package initializer
│   ├── train.py                # Train ML models
│   ├── evaluate.py             # Evaluate trained models and compute metrics
│   │
│   ├── data/                   # Training/testing data
│   │   ├── positive_sequences.fasta  # Positive (AMP) sequences for training
│   │   ├── negative_sequences.fasta  # Negative (non-AMP) sequences for training
│   │   ├── test_features.csv         # (Generated) Test-set features
│   │   └── test_labels.csv           # (Generated) Test-set labels
│   │
│   └── saved_model/            # Trained artifacts and evaluation outputs
│       ├── amp_model_rf.pkl          # (Generated) Random Forest model
│       ├── amp_model_svm.pkl         # (Generated) SVM model
│       ├── amp_model_gb.pkl          # (Generated) Gradient Boosting model
│       ├── evaluation_report.txt     # (Generated) Detailed text report
│       └── evaluation_report.csv     # (Generated) Comparative CSV report
│
├── img/                        # Images directory
└── tests/                      # Unit tests to ensure code quality
    ├── __init__.py             # Package initializer
    └── test_prediction.py      # Tests for prediction functions
```

---


## Contributors

### Lead Developer

- **Madson A. de Luna Aragão** — PhD Candidate in Bioinformatics, UFMG  
  Belo Horizonte, Minas Gerais, Brazil  
  **Responsibilities:** project lead, software architecture, ML pipelines, documentation.  
  **Contacts:** madsondeluna@gmail.com 

### Collaborators

- **Rafael L. da Silva** — Masters Student, UFPE — Collaborator  
  **Contributions:** data preprocessing, pipeline testing, literature review.

### Advisory Team

- **Ana M. Benko‑Iseppon, PhD** — Principal Investigator, UFPE — Advisor  
  **Contributions:** scientific supervision, study design, biological validation.

- **João Pacífico, PhD** — Principal Investigator, UPE — Co‑Advisor  
  **Contributions:** computational analysis review, dataset curation, evaluation protocol, reproducibility.

- **Carlos A. dos Santos-Silva, PhD** — Professor, CESMAC — Co‑Advisor  
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

- This research was supported by **FACEPE** — Fundação de Amparo à Pesquisa do Estado de Pernambuco (Brazil).
- We thank our advisory and collaborator team for scientific guidance and technical feedback.

---

## Intellectual Property

- This tool is **under submission for registration** with the **INPI** — Instituto Nacional da Propriedade Industrial (Brazilian National Institute of Industrial Property).
- All rights reserved. Usage and distribution are subject to the project license terms.

---

## How to Cite

If this tool or its outputs support your research, please cite the repository:

```text
Luna-Aragão, M. A., da Silva, R. L., Pacífico, J., Santos-Silva, C. A. & Benko‑Iseppon, A. M. (2025). AMPidentifier: A Python toolkit for predicting antimicrobial peptides using ensemble machine learning and physicochemical descriptors. GitHub repository. https://github.com/madsondeluna/AMPIdentifier
```
