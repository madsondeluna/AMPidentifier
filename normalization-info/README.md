# Normalization Documentation

This folder contains all documentation related to the StandardScaler normalization implementation in AMPidentifier v2.0.

## 📚 Documentation Files

### 1. [normalization_impact_report.md](normalization_impact_report.md)
**Detailed Technical Report (English)**
- Complete implementation details
- Performance comparison before/after normalization
- Technical analysis of changes
- Prediction comparison for test sequences

### 2. [resumo_normalizacao.md](resumo_normalizacao.md)
**Executive Summary (Portuguese)**
- Resumo executivo da implementação
- Alterações realizadas
- Resultados dos modelos
- Recomendações de uso

### 3. [quick_start_normalized.md](quick_start_normalized.md)
**Quick Start Guide**
- How to use normalized models
- Command examples
- Best practices
- Troubleshooting tips

### 4. [changelog.md](changelog.md)
**Complete Changelog**
- Detailed list of all changes
- Breaking changes
- Migration guide
- Version history

### 5. [verify_normalization.py](verify_normalization.py)
**Verification Script**
- Automated verification tool
- Checks scaler implementation
- Validates normalized data
- Tests model compatibility

## 🎯 Quick Links

### For Users
- **Getting Started**: See [quick_start_normalized.md](quick_start_normalized.md)
- **Portuguese Guide**: See [resumo_normalizacao.md](resumo_normalizacao.md)

### For Developers
- **Technical Details**: See [normalization_impact_report.md](normalization_impact_report.md)
- **Changes Log**: See [changelog.md](changelog.md)

### For Testing
- **Verify Installation**: Run `python normalization-info/verify_normalization.py`

## 🚀 Usage Examples

### Run Verification Script
```bash
python normalization-info/verify_normalization.py
```

### Quick Prediction
```bash
# Ensemble mode (recommended)
python main.py -i input.fasta -o output_dir --ensemble

# Single model
python main.py -i input.fasta -o output_dir -m rf
```

## 📊 Key Results

### Model Performance (with normalization)

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| RF    | 88.45%   | 0.8836   | 0.9503  |
| SVM   | 87.40%   | 0.8716   | 0.9356  |
| GB    | 85.85%   | 0.8569   | 0.9289  |

## ⚠️ Important Notes

1. **Scaler Required**: The `feature_scaler.pkl` file is required for all predictions
2. **Location**: Found at `model_training/saved_model/feature_scaler.pkl`
3. **Backward Compatibility**: Old models (without normalization) are not compatible
4. **Automatic**: Normalization is applied automatically in the pipeline

## 📖 Documentation Index

```
normalization-info/
├── README.md (this file)
├── normalization_impact_report.md  # Technical report
├── resumo_normalizacao.md          # Portuguese summary
├── quick_start_normalized.md       # Quick start guide
├── changelog.md                    # Complete changelog
└── verify_normalization.py         # Verification script
```

---

**Version**: AMPidentifier v2.0  
**Date**: October 10, 2025  
**Status**: ✅ Production Ready
