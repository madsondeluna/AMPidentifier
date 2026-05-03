# AMPidentifier — Google Colab and Jupyter Notebook Guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/madsondeluna/AMPidentifier/blob/main/AMPidentifier_Colab_Guide.ipynb)

Open the notebook directly in Google Colab by clicking the badge above, or download `AMPidentifier_Colab_Guide.ipynb` to run locally in Jupyter.

## What the notebook covers

- **Installation**: `pip install ampidentifier` and model file setup via sparse checkout
- **Input format**: FASTA requirements and an example with 8 sequences
- **CLI flags**: full reference for `-i`, `-o`, `-m`, and `--threshold`
- **Model selection**: six models with accuracy, AUC-ROC, and MCC
- **Custom threshold**: guidance on sensitivity vs. specificity trade-offs
- **Output**: column descriptions and pandas-based filtering
- **Visualization**: horizontal bar chart of AMP probabilities
- **Batch comparison**: run all six models and merge results into a single DataFrame
- **Python API**: direct call to `run_prediction_pipeline` for pipeline integration

## Quick start

```bash
pip install ampidentifier
```

Then open the notebook in Colab or Jupyter and run the cells in order. The setup section downloads the trained model files automatically.

## CLI entry point

After `pip install ampidentifier`, the `ampidentifier2` command is available:

```bash
ampidentifier2 -i sequences.fasta -o results/
```

| Flag | Short | Required | Default | Description |
|---|---|---|---|---|
| `--input` | `-i` | Yes | — | Input FASTA file |
| `--output_dir` | `-o` | Yes | — | Output directory |
| `--model` | `-m` | No | `voting` | Model: `rf`, `svm`, `gb`, `xgb`, `lgbm`, `voting` |
| `--threshold` | — | No | MCC-optimized | Decision threshold (0.0–1.0) |

## Output format

Each run produces `predictions_{model}.csv` with columns:

| Column | Description |
|---|---|
| `ID` | Sequence identifier from the FASTA header |
| `sequence` | Amino acid sequence |
| `probability_AMP` | Predicted AMP probability (0.0–1.0) |
| `prediction` | `1` = AMP, `0` = non-AMP |
| `label` | `AMP` or `non-AMP` |

## Model performance

| Model | Accuracy | AUC-ROC | MCC |
|---|---|---|---|
| Voting ensemble (recommended) | 92.9% | 0.977 | 0.859 |
| LightGBM | 92.7% | 0.975 | 0.855 |
| XGBoost | 92.2% | 0.974 | 0.843 |
| Gradient Boosting | 92.0% | 0.974 | 0.839 |
| Random Forest | 91.9% | 0.972 | 0.839 |
| SVM (RBF kernel) | 91.9% | 0.969 | 0.839 |

Metrics from the internal test set. On the independent benchmark (n=4,736): voting ensemble AUC-ROC 0.950, MCC 0.742, Sensitivity 94.9%, Specificity 78.4%.
