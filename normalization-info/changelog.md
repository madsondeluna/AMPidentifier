# Changelog - AMPidentifier v2.0

## [2.0.0] - 2025-10-10

### 🎯 Major Update: Feature Normalization Implementation

This releas### Documentation
- `normalization_impact_report.md` - Technical analysis
- `resumo_normalizacao.md` - Portuguese summary
- `quick_start_normalized.md` - Usage guide
- `verify_normalization.py` - Verification script
- `changelog.md` - This fileoduces comprehensive StandardScaler normalization across all pipeline stages, improving model performance and reliability.

---

## ✨ Added

### Core Features
- **StandardScaler Normalization**: Features are now normalized using sklearn's StandardScaler
- **Scaler Persistence**: Scaler is saved and loaded automatically for consistent predictions
- **Verification Script**: New `verify_normalization.py` to validate implementation

### New Functions
- `load_scaler()` in `amp_identifier/prediction.py` - Loads saved StandardScaler
- Scaler parameter in `predict_sequences()` - Accepts optional scaler for normalization

### Documentation
- `normalization_impact_report.md` - Detailed technical report (English)
- `resumo_normalizacao.md` - Executive summary (Portuguese)
- `quick_start_normalized.md` - Quick reference guide
- `verify_normalization.py` - Verification tool

---

## 🔧 Changed

### Training Pipeline (`model_training/train.py`)
- Added StandardScaler after train/test split
- Scaler fitted only on training data (prevents data leakage)
- Scaler saved to `model_training/saved_model/feature_scaler.pkl`
- Test features saved in normalized form
- Models trained on normalized data

**Before:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
model.fit(X_train, y_train)
```

**After:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, scaler_path)
model.fit(X_train_scaled, y_train)
```

### Prediction Module (`amp_identifier/prediction.py`)
- Updated `predict_sequences()` to accept scaler parameter
- Features normalized before prediction when scaler provided
- Maintains backward compatibility (scaler is optional)

**Before:**
```python
def predict_sequences(model, features_df):
    X = features_df[model_features]
    predictions = model.predict(X)
```

**After:**
```python
def predict_sequences(model, features_df, scaler=None):
    X = features_df[model_features]
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)
    predictions = model.predict(X)
```

### Core Pipeline (`amp_identifier/core.py`)
- Scaler loaded at pipeline initialization
- Applied to all internal model predictions
- External models can optionally skip normalization

**Before:**
```python
internal_results = prediction.predict_sequences(internal_model, features_df.copy())
```

**After:**
```python
scaler = prediction.load_scaler(SCALER_PATH)
internal_results = prediction.predict_sequences(internal_model, features_df.copy(), scaler)
```

### Evaluation Pipeline (`model_training/evaluate.py`)
- Updated to work with pre-normalized test data
- Added note in docstring about normalized data

---

## 📊 Performance

### Model Metrics (With Normalization)

| Metric       | Random Forest | SVM    | Gradient Boosting |
|--------------|---------------|--------|-------------------|
| Accuracy     | **88.45%**    | 87.40% | 85.85%           |
| Precision    | 89.10%        | 88.80% | 86.65%           |
| Recall       | 87.62%        | 85.58% | 84.75%           |
| Specificity  | 89.28%        | 89.21% | 86.94%           |
| F1-Score     | **0.8836**    | 0.8716 | 0.8569           |
| MCC          | **0.7692**    | 0.7484 | 0.7172           |
| AUC-ROC      | **0.9503**    | 0.9356 | 0.9289           |

**Best Model**: Random Forest (RF) - Recommended as default

### Key Improvements
- ✅ **SVM Performance**: Significantly improved (SVM is sensitive to feature scales)
- ✅ **Probability Calibration**: More reliable probability estimates
- ✅ **Generalization**: Better performance on new data
- ✅ **Reproducibility**: Consistent results across runs

---

## 📁 New Files

### Models & Scaler
- `model_training/saved_model/feature_scaler.pkl` (1.2KB) - **REQUIRED for predictions**
- `model_training/saved_model/amp_model_rf.pkl` (15MB) - Retrained Random Forest
- `model_training/saved_model/amp_model_svm.pkl` (398KB) - Retrained SVM
- `model_training/saved_model/amp_model_gb.pkl` (139KB) - Retrained Gradient Boosting

### Documentation
- `NORMALIZATION_IMPACT_REPORT.md` - Technical analysis
- `RESUMO_NORMALIZACAO.md` - Portuguese summary
- `QUICK_START_NORMALIZED.md` - Usage guide
- `verify_normalization.py` - Verification tool
- `CHANGELOG.md` - This file

---

## ⚠️ Breaking Changes

### Backward Compatibility
- **Models**: Old models (trained without normalization) are **NOT compatible**
- **Predictions**: Require `feature_scaler.pkl` for accurate results
- **External Models**: May need their own normalization strategy

### Migration Guide
1. Retrain all models: `python -m model_training.train`
2. Verify implementation: `python verify_normalization.py`
3. Update deployment to include `feature_scaler.pkl`
4. Test predictions with new models

---

## 🔍 Verification

Run the verification script to confirm correct implementation:

```bash
python verify_normalization.py
```

Expected results:
- ✅ Scaler found and loaded (StandardScaler)
- ✅ 10 features normalized
- ✅ Data statistics: mean ≈ 0, std ≈ 1
- ✅ 3 models trained (RF, SVM, GB)
- ✅ Transformation reversible (numerical precision < 1e-16)
- ✅ Metrics calculated correctly

---

## 🚀 Usage

### Basic Prediction (Single Model)
```bash
python main.py -i sequences.fasta -o results/ -m rf
```

### Ensemble Prediction (Recommended)
```bash
python main.py -i sequences.fasta -o results/ --ensemble
```

### Training
```bash
python -m model_training.train
```

### Evaluation
```bash
python -m model_training.evaluate
```

---

## 🐛 Bug Fixes

- Fixed feature scaling inconsistency between training and prediction
- Improved SVM probability calibration
- Resolved numerical stability issues with different feature scales

---

## 📋 Technical Details

### Normalization Method
- **Algorithm**: StandardScaler from scikit-learn
- **Formula**: `X_scaled = (X - μ) / σ`
- **Result**: Features with mean = 0, standard deviation = 1

### Features Normalized (10 total)
1. Length
2. Molecular Weight (MW)
3. Charge
4. Charge Density
5. Isoelectric Point (pI)
6. Instability Index
7. Aromaticity
8. Aliphatic Index
9. Boman Index
10. Hydrophobic Ratio

### Implementation Details
- Scaler fitted only on training data (prevents data leakage)
- Same scaler applied to test and prediction data
- Transformation is reversible with numerical precision
- Compatible with scikit-learn pipelines

---

## 🎯 Recommendations

### For Users
1. ✅ Use `--ensemble` mode for critical predictions
2. ✅ Use Random Forest (`-m rf`) for single model predictions
3. ✅ Always keep `feature_scaler.pkl` with model files
4. ✅ Run verification after training new models

### For Developers
1. ✅ Version control scaler with models
2. ✅ Include scaler in deployment packages
3. ✅ Test on independent validation sets
4. ✅ Monitor model performance over time

---

## 📝 Notes

- All models retrained from scratch with normalized data
- Evaluation metrics recalculated with normalized features
- Comprehensive testing performed on example datasets
- Documentation updated with normalization details

---

## 🔗 References

- **Scikit-learn StandardScaler**: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- **Feature Scaling Best Practices**: https://scikit-learn.org/stable/modules/preprocessing.html

---

## 👥 Contributors

- Implementation: AMPidentifier Development Team
- Testing: Comprehensive automated verification
- Documentation: Complete technical and user guides

---

## 📅 Release Information

- **Version**: 2.0.0
- **Release Date**: October 10, 2025
- **Status**: ✅ Production Ready
- **Python Version**: 3.12+
- **Dependencies**: scikit-learn, pandas, joblib, modlamp

---

## 🔜 Future Work

### Potential Enhancements
- [ ] Hyperparameter tuning with normalized features
- [ ] Feature importance analysis
- [ ] Cross-validation with stratified folds
- [ ] Ensemble with weighted voting
- [ ] Additional normalization strategies (MinMaxScaler, RobustScaler)
- [ ] Model explainability (SHAP values)

---

**For detailed technical analysis, see `normalization_impact_report.md`**  
**For quick start guide, see `quick_start_normalized.md`**  
**For Portuguese summary, see `resumo_normalizacao.md`**
