# Model Explainability Report

**Generated:** 2025-12-29 02:52:27

---

## Executive Summary

This report provides comprehensive explainability analysis for the three AMP prediction models using SHAP (SHapley Additive exPlanations). SHAP values provide a unified measure of feature importance and show how each feature contributes to individual predictions.

### Models Analyzed

1. **Random Forest (RF)** - Ensemble of decision trees
2. **Support Vector Machine (SVM)** - Kernel-based classifier
3. **Gradient Boosting (GB)** - Sequential ensemble method

---

## What is SHAP?

SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain machine learning model predictions. It assigns each feature an importance value for a particular prediction, showing how much each feature contributed to moving the prediction away from the base value.

### Key Advantages

- **Model-agnostic:** Works with any machine learning model
- **Theoretically sound:** Based on Shapley values from cooperative game theory
- **Consistent:** Features that contribute more receive higher importance
- **Local and global:** Explains individual predictions and overall model behavior

---

## RF Model Analysis

### Top 10 Most Important Features

| Rank | Feature | Mean Abs SHAP | Mean SHAP |
|------|---------|---------------|----------|
| 1 | Charge | 0.108923 | -0.009541 |
| 2 | ChargeDensity | 0.094883 | -0.001007 |
| 3 | Aromaticity | 0.067022 | -0.000269 |
| 4 | Length | 0.058349 | 0.005366 |
| 5 | pI | 0.042694 | 0.010414 |
| 6 | MW | 0.039448 | -0.000279 |
| 7 | AliphaticInd | 0.030609 | 0.002265 |
| 8 | BomanInd | 0.025823 | -0.000236 |
| 9 | InstabilityInd | 0.023903 | -0.000266 |
| 10 | HydrophRatio | 0.023463 | -0.001045 |

### Visualizations

1. **Summary Plot:** `rf_summary_plot.png`
   - Shows distribution of SHAP values for each feature
   - Color indicates feature value (red = high, blue = low)

2. **Bar Plot:** `rf_bar_plot.png`
   - Shows mean absolute SHAP values (overall feature importance)

3. **Waterfall Plots:** `rf_waterfall_sample_*.png`
   - Shows how features contribute to individual predictions
   - Demonstrates model decision-making process

4. **Dependence Plots:** `rf_dependence_*.png`
   - Shows relationship between feature values and SHAP values
   - Reveals non-linear relationships and interactions

---

## SVM Model Analysis

### Top 10 Most Important Features

| Rank | Feature | Mean Abs SHAP | Mean SHAP |
|------|---------|---------------|----------|
| 1 | Length | 0.188007 | 0.000000 |
| 2 | ChargeDensity | 0.162526 | -0.000000 |
| 3 | MW | 0.148283 | -0.000000 |
| 4 | Charge | 0.090759 | 0.000000 |
| 5 | pI | 0.086373 | -0.000000 |
| 6 | Aromaticity | 0.074497 | 0.000000 |
| 7 | BomanInd | 0.041612 | -0.000000 |
| 8 | AliphaticInd | 0.034796 | -0.000000 |
| 9 | HydrophRatio | 0.031705 | -0.000000 |
| 10 | InstabilityInd | 0.022707 | 0.000000 |

### Visualizations

1. **Summary Plot:** `svm_summary_plot.png`
   - Shows distribution of SHAP values for each feature
   - Color indicates feature value (red = high, blue = low)

2. **Bar Plot:** `svm_bar_plot.png`
   - Shows mean absolute SHAP values (overall feature importance)

3. **Waterfall Plots:** `svm_waterfall_sample_*.png`
   - Shows how features contribute to individual predictions
   - Demonstrates model decision-making process

4. **Dependence Plots:** `svm_dependence_*.png`
   - Shows relationship between feature values and SHAP values
   - Reveals non-linear relationships and interactions

---

## GB Model Analysis

### Top 10 Most Important Features

| Rank | Feature | Mean Abs SHAP | Mean SHAP |
|------|---------|---------------|----------|
| 1 | Charge | 0.981482 | -0.076160 |
| 2 | ChargeDensity | 0.519698 | -0.036644 |
| 3 | Length | 0.498726 | 0.019399 |
| 4 | MW | 0.494013 | -0.014953 |
| 5 | Aromaticity | 0.463763 | 0.003312 |
| 6 | pI | 0.432532 | 0.080873 |
| 7 | BomanInd | 0.231262 | -0.005363 |
| 8 | InstabilityInd | 0.178053 | 0.014661 |
| 9 | AliphaticInd | 0.167262 | 0.009981 |
| 10 | HydrophRatio | 0.084373 | -0.000737 |

### Visualizations

1. **Summary Plot:** `gb_summary_plot.png`
   - Shows distribution of SHAP values for each feature
   - Color indicates feature value (red = high, blue = low)

2. **Bar Plot:** `gb_bar_plot.png`
   - Shows mean absolute SHAP values (overall feature importance)

3. **Waterfall Plots:** `gb_waterfall_sample_*.png`
   - Shows how features contribute to individual predictions
   - Demonstrates model decision-making process

4. **Dependence Plots:** `gb_dependence_*.png`
   - Shows relationship between feature values and SHAP values
   - Reveals non-linear relationships and interactions

---

## Model Comparison

The comparison plot (`models_comparison.png`) shows how feature importance varies across the three models. This helps identify:

- **Consensus features:** Important across all models
- **Model-specific features:** Important for specific model types
- **Robustness:** Features consistently important are more reliable

---

## Interpreting SHAP Values

### SHAP Value Interpretation

- **Positive SHAP value:** Feature pushes prediction toward positive class (AMP)
- **Negative SHAP value:** Feature pushes prediction toward negative class (non-AMP)
- **Magnitude:** Larger absolute value = stronger influence

### Summary Plot Interpretation

- **Vertical axis:** Features ranked by importance
- **Horizontal axis:** SHAP value (impact on prediction)
- **Color:** Feature value (red = high, blue = low)
- **Density:** Distribution of SHAP values across samples

---

## Conclusion

This explainability analysis demonstrates that our AMP prediction models are **not black boxes**. Through SHAP analysis, we can:

1. Identify which physicochemical features are most important for predictions
2. Understand how each feature contributes to individual predictions
3. Validate that models use biologically relevant features
4. Build trust in model predictions through transparency

The comprehensive visualizations and feature importance rankings provide clear insights into model decision-making, making the models interpretable and trustworthy for antimicrobial peptide prediction.

---

## Files Generated

### Per-Model Files

For each model (RF, SVM, GB):
- `{model}_summary_plot.png` - SHAP summary plot
- `{model}_bar_plot.png` - Feature importance bar plot
- `{model}_waterfall_sample_*.png` - Individual prediction explanations
- `{model}_dependence_*.png` - Feature dependence plots
- `{model}_feature_importance.csv` - Feature importance table

### Comparison Files

- `models_comparison.png` - Feature importance across models
- `models_comparison.csv` - Comparison data table
- `EXPLAINABILITY_REPORT.md` - This report

