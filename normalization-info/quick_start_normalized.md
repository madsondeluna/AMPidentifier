# Quick Start Guide - Normalized Models

## Implementation Complete

The StandardScaler normalization has been successfully implemented across all pipeline stages.

## How to Use

### 1. Single Model Prediction
```bash
python main.py -i input.fasta -o output_dir -m rf
```
Models available: `rf` (Random Forest), `svm` (SVM), `gb` (Gradient Boosting)

### 2. Ensemble Prediction (Recommended)
```bash
python main.py -i input.fasta -o output_dir --ensemble
```

### 3. With External Models
```bash
python main.py -i input.fasta -o output_dir --ensemble -e external_model1.pkl external_model2.pkl
```

## What Changed

### Automatic Normalization
- Features are automatically normalized using the saved scaler
- Scaler is loaded from: `model_training/saved_model/feature_scaler.pkl`
- All internal models use the same normalization

### Training
```bash
python -m model_training.train
```
- Features are normalized with StandardScaler
- Scaler is saved for use in predictions
- Models are trained on normalized data

### Evaluation
```bash
python -m model_training.evaluate
```
- Uses pre-normalized test data
- Calculates metrics correctly

## Verification

Run the verification script to check the implementation:
```bash
python verify_normalization.py
```

Expected output:
- Scaler found and loaded
- Data appears normalized (mean ≈ 0, std ≈ 1)
- 3 models found (RF, SVM, GB)
- Transformation is reversible
- Metrics calculated correctly

## Model Performance

| Model | Accuracy | F1-Score | AUC-ROC | Recommendation |
|-------|----------|----------|---------|----------------|
| RF    | 88.45%   | 0.8836   | 0.9503  | Best overall |
| SVM   | 87.40%   | 0.8716   | 0.9356  | Good         |
| GB    | 85.85%   | 0.8569   | 0.9289  | Good         |

**Recommended**: Use `--ensemble` for best results

## Important Notes

1. **Scaler Dependency**: The scaler file is required for predictions
   - Location: `model_training/saved_model/feature_scaler.pkl`
   - Always keep it with the models

2. **Backward Compatibility**: Old models (without normalization) are not compatible

3. **External Models**: External models won't use the scaler (may have their own normalization)

## Key Files

```
model_training/saved_model/
├── feature_scaler.pkl          # StandardScaler (REQUIRED)
├── amp_model_rf.pkl            # Random Forest model
├── amp_model_svm.pkl           # SVM model
├── amp_model_gb.pkl            # Gradient Boosting model
├── evaluation_report.txt       # Detailed metrics
└── evaluation_report.csv       # Metrics for analysis
```

## Best Practices

1. Use ensemble mode for important predictions
2. Random Forest for single model predictions
3. Always keep scaler.pkl with model files
4. Run verification after training new models
5. Version control the scaler with models

## Documentation

- **Detailed Report**: `NORMALIZATION_IMPACT_REPORT.md`
- **Portuguese Summary**: `RESUMO_NORMALIZACAO.md`
- **Verification Script**: `verify_normalization.py`

## Troubleshooting

### Error: Scaler file not found
```
Error: Scaler file not found at 'model_training/saved_model/feature_scaler.pkl'
```
**Solution**: Run `python -m model_training.train` to generate the scaler

### Warning: Cannot determine feature names
```
Warning: Cannot determine feature names from the model.
```
**Solution**: This is expected for older models. The code handles this automatically.

## Next Steps

1. Validate on independent datasets
2. Compare with published benchmarks
3. Consider hyperparameter tuning
4. Evaluate feature importance
5. Document usage protocols

---

**Version**: AMPidentifier
**Date**: October 10, 2025  
**Status**: Production Ready
