# Model Explainability with SHAP

## Overview

This module provides comprehensive explainability analysis for the three AMP prediction models using SHAP (SHapley Additive exPlanations). The goal is to demonstrate that our models are **not black boxes** and to understand which physicochemical features contribute most to predictions.

---

## What is SHAP?

SHAP (SHapley Additive exPlanations) is a unified framework for interpreting machine learning model predictions based on game theory (Shapley values). It provides:

- **Feature importance**: Which features matter most globally
- **Individual explanations**: How features contribute to specific predictions
- **Consistency**: Features that contribute more receive higher importance
- **Model-agnostic**: Works with any machine learning model

### Key Advantages

1. **Theoretically sound**: Based on cooperative game theory
2. **Consistent**: Features with larger contributions get higher importance
3. **Local and global**: Explains both individual predictions and overall behavior
4. **Interpretable**: Clear visual representations

---

## Models Analyzed

This analysis covers all three AMP prediction models:

1. **Random Forest (RF)** - Ensemble of decision trees
2. **Support Vector Machine (SVM)** - Kernel-based classifier  
3. **Gradient Boosting (GB)** - Sequential ensemble method

---

## Running the Analysis

### Prerequisites

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Required packages:
- `shap>=0.42.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.12.0`

### Quick Start

```bash
# From project root directory
./scripts/run_explainability_analysis.sh
```

Or run directly with Python:

```bash
python3 -m model_training.explainability
```

### Expected Runtime

- **Random Forest**: ~1-2 minutes
- **Gradient Boosting**: ~1-2 minutes
- **SVM**: ~5-10 minutes (slower due to KernelExplainer)

Total: ~10-15 minutes

---

## Generated Outputs

All outputs are saved to `model_training/explainability_reports/`

### Per-Model Visualizations

For each model (RF, SVM, GB), the following are generated:

#### 1. Summary Plot (`{model}_summary_plot.png`)

**What it shows:**
- Distribution of SHAP values for each feature
- Features ranked by importance (top to bottom)
- Color indicates feature value (red = high, blue = low)

**Interpretation:**
- Features at the top are most important
- Wide distributions indicate varied impact across samples
- Color patterns show how feature values affect predictions

#### 2. Bar Plot (`{model}_bar_plot.png`)

**What it shows:**
- Mean absolute SHAP values for each feature
- Overall feature importance ranking

**Interpretation:**
- Longer bars = more important features
- Simple ranking of global feature importance

#### 3. Waterfall Plots (`{model}_waterfall_sample_*.png`)

**What it shows:**
- How each feature contributes to a specific prediction
- Starting from base value to final prediction

**Interpretation:**
- Red bars push prediction toward positive class (AMP)
- Blue bars push prediction toward negative class (non-AMP)
- Bar length shows contribution magnitude

#### 4. Dependence Plots (`{model}_dependence_*.png`)

**What it shows:**
- Relationship between feature values and SHAP values
- Interaction effects with other features

**Interpretation:**
- Shows non-linear relationships
- Reveals feature interactions
- Validates biological relevance

### Comparison Visualizations

#### Model Comparison Plot (`models_comparison.png`)

**What it shows:**
- Feature importance across all three models
- Top 15 most important features

**Interpretation:**
- Consensus features: Important across all models
- Model-specific features: Important for certain model types
- Robustness: Consistently important features are more reliable

### Data Tables

#### Feature Importance Tables (`{model}_feature_importance.csv`)

Contains for each model:
- Feature ranking
- Mean absolute SHAP value
- Mean SHAP value (directional)

#### Comparison Table (`models_comparison.csv`)

Feature importance values across all three models for easy comparison.

### Comprehensive Report

#### EXPLAINABILITY_REPORT.md

A detailed Markdown report containing:
- Executive summary
- SHAP methodology explanation
- Per-model analysis with top features
- Interpretation guidelines
- Model comparison insights
- Conclusions about model transparency

---

## Interpreting SHAP Values

### SHAP Value Meaning

- **Positive SHAP value**: Feature pushes prediction toward positive class (AMP)
- **Negative SHAP value**: Feature pushes prediction toward negative class (non-AMP)
- **Magnitude**: Larger absolute value = stronger influence

### Summary Plot Colors

- **Red**: High feature value
- **Blue**: Low feature value
- **Purple**: Medium feature value

### Example Interpretation

If a feature has:
- High SHAP value when red (high feature value) → High values predict AMP
- Low SHAP value when blue (low feature value) → Low values predict non-AMP
- This indicates a positive correlation with AMP prediction

---

## Use Cases

### 1. Model Validation

Verify that models use biologically relevant features:
- Are charge-related features important? (Expected for AMPs)
- Are hydrophobicity features significant? (Expected for membrane interaction)
- Do models rely on meaningful patterns?

### 2. Feature Engineering

Identify which features matter most:
- Focus on important features for model improvement
- Remove or combine less important features
- Create new features based on important patterns

### 3. Trust and Transparency

Demonstrate model interpretability:
- Show stakeholders how models make decisions
- Build confidence in predictions
- Identify potential biases or issues

### 4. Scientific Insights

Discover biological patterns:
- Which physicochemical properties define AMPs?
- How do different models prioritize features?
- Are there unexpected important features?

---

## Technical Details

### SHAP Explainers Used

#### TreeExplainer (RF and GB)
- Fast and exact for tree-based models
- Uses tree structure for efficient computation
- Provides exact Shapley values

#### KernelExplainer (SVM)
- Model-agnostic approach
- Uses sampling for approximation
- Slower but works for any model

### Computational Considerations

- **Memory**: Requires loading models and test data
- **Time**: SVM analysis is slower due to KernelExplainer
- **Sampling**: SVM uses 100 samples for efficiency
- **Parallelization**: Tree explainers use multiple cores

---

## Troubleshooting

### Common Issues

#### 1. Memory Error

**Problem**: Out of memory during SHAP calculation

**Solution**:
```python
# Reduce sample size in explainability.py
X_sample = shap.sample(X_test, 50)  # Reduce from 100
```

#### 2. Slow SVM Analysis

**Problem**: SVM explainer takes too long

**Solution**:
```python
# Reduce background samples
background_sample = shap.sample(X_background, 50)  # Reduce from 100
```

#### 3. Import Error

**Problem**: Cannot import shap

**Solution**:
```bash
pip install shap>=0.42.0 --upgrade
```

#### 4. Matplotlib Backend Error

**Problem**: Display issues with plots

**Solution**:
```python
# Add to explainability.py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

---

## Extending the Analysis

### Adding New Visualizations

To add custom SHAP plots, modify `explainability.py`:

```python
def plot_custom_analysis(shap_values, X_data, model_name):
    """Add your custom SHAP visualization here."""
    # Your code here
    pass
```

### Analyzing Specific Samples

To explain specific predictions:

```python
# In explainability.py
sample_idx = 42  # Your sample of interest
shap.force_plot(explainer.expected_value, shap_values[sample_idx], X_test.iloc[sample_idx])
```

### Comparing Subgroups

To analyze different subgroups:

```python
# Split by prediction confidence
high_conf = predictions > 0.9
low_conf = predictions < 0.1

# Analyze separately
shap_values_high = explainer.shap_values(X_test[high_conf])
shap_values_low = explainer.shap_values(X_test[low_conf])
```

---

## References

### SHAP Documentation
- Official docs: https://shap.readthedocs.io/
- GitHub: https://github.com/slundberg/shap
- Paper: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"

### Related Work
- Shapley values: Shapley (1953) "A value for n-person games"
- TreeSHAP: Lundberg et al. (2020) "From local explanations to global understanding"
- KernelSHAP: Lundberg & Lee (2017) NIPS

---

## Best Practices

### 1. Always Check Feature Importance

Before trusting a model, verify:
- Important features make biological sense
- No single feature dominates (overfitting)
- Consistent patterns across models

### 2. Examine Individual Predictions

For critical predictions:
- Use waterfall plots to understand decision
- Check if explanation aligns with expectation
- Verify feature values are reasonable

### 3. Compare Across Models

- Look for consensus features
- Investigate model-specific patterns
- Use ensemble of explanations for robustness

### 4. Document Findings

- Save all visualizations
- Record important insights
- Share reports with stakeholders

---

## Integration with Main Pipeline

The explainability analysis integrates seamlessly:

```bash
# 1. Train models
python3 -m model_training.train

# 2. Evaluate models
python3 -m model_training.evaluate

# 3. Generate explainability reports
./scripts/run_explainability_analysis.sh

# 4. Review reports
cat model_training/explainability_reports/EXPLAINABILITY_REPORT.md
```

---

## Conclusion

This explainability module provides comprehensive transparency for AMP prediction models. Through SHAP analysis, we demonstrate that:

1. Models are interpretable and transparent
2. Predictions are based on meaningful features
3. Decision-making process is understandable
4. Models can be trusted for scientific applications

The generated reports serve as evidence that our models are **not black boxes** but rather transparent, interpretable tools for antimicrobial peptide prediction.
