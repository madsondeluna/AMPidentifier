# model_training/explainability.py

"""
Model Explainability Module using SHAP (SHapley Additive exPlanations)

This module provides comprehensive explainability analysis for the three AMP prediction models:
- Random Forest (RF)
- Support Vector Machine (SVM)
- Gradient Boosting (GB)

It generates various SHAP visualizations and reports to demonstrate that the models
are not black boxes and to understand which features contribute most to predictions.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime

# Configuration
OUTPUT_DIR = "model_training/saved_model"
EXPLAINABILITY_DIR = "model_training/explainability_reports"
TEST_FEATURES_PATH = "model_training/data/test_features.csv"
TEST_LABELS_PATH = "model_training/data/test_labels.csv"

# Create output directory
os.makedirs(EXPLAINABILITY_DIR, exist_ok=True)

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_models_and_data():
    """Load trained models, scaler, and test data."""
    print("Loading models and data...")
    
    # Load models
    models = {
        'rf': joblib.load(os.path.join(OUTPUT_DIR, "amp_model_rf.pkl")),
        'svm': joblib.load(os.path.join(OUTPUT_DIR, "amp_model_svm.pkl")),
        'gb': joblib.load(os.path.join(OUTPUT_DIR, "amp_model_gb.pkl"))
    }
    
    # Load scaler
    scaler = joblib.load(os.path.join(OUTPUT_DIR, "feature_scaler.pkl"))
    
    # Load test data
    X_test = pd.read_csv(TEST_FEATURES_PATH)
    y_test = pd.read_csv(TEST_LABELS_PATH)['label']
    
    print(f"Loaded {len(models)} models and {len(X_test)} test samples")
    print(f"Features: {X_test.shape[1]}")
    
    return models, scaler, X_test, y_test


def create_shap_explainer(model, model_name, X_background):
    """
    Create appropriate SHAP explainer for each model type.
    
    Args:
        model: Trained model
        model_name: Name of the model ('rf', 'svm', 'gb')
        X_background: Background data for SHAP
    
    Returns:
        SHAP explainer object
    """
    print(f"Creating SHAP explainer for {model_name.upper()}...")
    
    if model_name == 'rf':
        # TreeExplainer for Random Forest
        explainer = shap.TreeExplainer(model)
    elif model_name == 'gb':
        # TreeExplainer for Gradient Boosting
        explainer = shap.TreeExplainer(model)
    elif model_name == 'svm':
        # KernelExplainer for SVM (slower but works for any model)
        # Use a subset of background data for efficiency
        background_sample = shap.sample(X_background, min(100, len(X_background)))
        explainer = shap.KernelExplainer(model.predict_proba, background_sample)
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    return explainer


def calculate_shap_values(explainer, X_test, model_name):
    """Calculate SHAP values for test set."""
    print(f"Calculating SHAP values for {model_name.upper()}...")
    
    if model_name == 'svm':
        # For SVM, use a smaller sample for efficiency
        X_sample = shap.sample(X_test, min(100, len(X_test)))
        shap_values = explainer.shap_values(X_sample)
        return shap_values, X_sample
    else:
        # For tree-based models, calculate for all test samples
        shap_values = explainer.shap_values(X_test)
        return shap_values, X_test


def plot_summary_plot(shap_values, X_data, model_name, feature_names):
    """
    Generate SHAP summary plot (beeswarm plot).
    Shows the distribution of SHAP values for each feature.
    """
    print(f"Generating summary plot for {model_name.upper()}...")
    
    plt.figure(figsize=(12, 8))
    
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        # TreeExplainer format: (n_samples, n_features, n_classes)
        shap_values_plot = shap_values[:, :, 1]
    else:
        shap_values_plot = shap_values
    
    shap.summary_plot(
        shap_values_plot,
        X_data,
        feature_names=feature_names,
        show=False,
        max_display=20
    )
    
    plt.title(f'SHAP Summary Plot - {model_name.upper()} Model', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(EXPLAINABILITY_DIR, f'{model_name}_summary_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to {output_path}")


def plot_bar_plot(shap_values, X_data, model_name, feature_names):
    """
    Generate SHAP bar plot.
    Shows mean absolute SHAP values for each feature.
    """
    print(f"Generating bar plot for {model_name.upper()}...")
    
    plt.figure(figsize=(12, 8))
    
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        # TreeExplainer format: (n_samples, n_features, n_classes)
        shap_values_plot = shap_values[:, :, 1]
    else:
        shap_values_plot = shap_values
    
    shap.summary_plot(
        shap_values_plot,
        X_data,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20
    )
    
    plt.title(f'SHAP Feature Importance - {model_name.upper()} Model', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(EXPLAINABILITY_DIR, f'{model_name}_bar_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Bar plot saved to {output_path}")


def plot_waterfall_examples(shap_values, X_data, model_name, feature_names, n_examples=3):
    """
    Generate waterfall plots for individual predictions.
    Shows how each feature contributes to a specific prediction.
    """
    print(f"Generating waterfall plots for {model_name.upper()}...")
    
    # Handle different SHAP value formats
    # TreeExplainer returns shape (n_samples, n_features, n_classes) for binary classification
    # KernelExplainer returns list of arrays for each class
    
    if isinstance(shap_values, list):
        # KernelExplainer format: list of arrays, one per class
        # Use positive class (index 1)
        shap_values_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        # TreeExplainer format: (n_samples, n_features, n_classes)
        # Extract positive class (index 1) for all samples
        shap_values_plot = shap_values[:, :, 1]
    else:
        # Already in correct format
        shap_values_plot = shap_values
    
    # Select examples: first, middle, and last
    indices = [0, len(X_data) // 2, len(X_data) - 1]
    indices = indices[:n_examples]
    
    for idx, sample_idx in enumerate(indices):
        plt.figure(figsize=(12, 8))
        
        # Get SHAP values for this sample (should be 1D array of length n_features)
        sample_shap_values = shap_values_plot[sample_idx]
        
        # Get feature values for this sample
        if isinstance(X_data, pd.DataFrame):
            sample_features = X_data.iloc[sample_idx].values
        else:
            sample_features = X_data[sample_idx]
        
        # Create explanation object for waterfall plot
        explanation = shap.Explanation(
            values=sample_shap_values,
            base_values=0,  # Will be set by SHAP
            data=sample_features,
            feature_names=feature_names
        )
        
        shap.waterfall_plot(explanation, show=False, max_display=15)
        
        plt.title(f'SHAP Waterfall Plot - {model_name.upper()} Model (Sample {sample_idx})', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(EXPLAINABILITY_DIR, f'{model_name}_waterfall_sample_{idx+1}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Waterfall plots saved for {n_examples} samples")


def plot_dependence_plots(shap_values, X_data, model_name, feature_names, top_n=5):
    """
    Generate SHAP dependence plots for top features.
    Shows how feature values affect SHAP values.
    """
    print(f"Generating dependence plots for {model_name.upper()}...")
    
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        # TreeExplainer format: (n_samples, n_features, n_classes)
        shap_values_plot = shap_values[:, :, 1]
    else:
        shap_values_plot = shap_values
    
    # Get top features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]
    
    for idx in top_features_idx:
        feature_name = feature_names[idx]
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            idx,
            shap_values_plot,
            X_data,
            feature_names=feature_names,
            show=False
        )
        
        plt.title(f'SHAP Dependence Plot - {model_name.upper()} Model\nFeature: {feature_name}',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Sanitize feature name for filename
        safe_feature_name = feature_name.replace('/', '_').replace(' ', '_')
        output_path = os.path.join(EXPLAINABILITY_DIR, 
                                   f'{model_name}_dependence_{safe_feature_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Dependence plots saved for top {top_n} features")


def generate_feature_importance_table(shap_values, feature_names, model_name):
    """
    Generate a table with feature importance rankings.
    """
    print(f"Generating feature importance table for {model_name.upper()}...")
    
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        # TreeExplainer format: (n_samples, n_features, n_classes)
        shap_values_plot = shap_values[:, :, 1]
    else:
        shap_values_plot = shap_values
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap,
        'Mean_SHAP': shap_values_plot.mean(axis=0)
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Mean_Abs_SHAP', ascending=False)
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    
    # Reorder columns
    importance_df = importance_df[['Rank', 'Feature', 'Mean_Abs_SHAP', 'Mean_SHAP']]
    
    # Save to CSV
    output_path = os.path.join(EXPLAINABILITY_DIR, f'{model_name}_feature_importance.csv')
    importance_df.to_csv(output_path, index=False, float_format='%.6f')
    
    print(f"Feature importance table saved to {output_path}")
    
    return importance_df


def generate_comparison_plot(all_importances, model_names):
    """
    Generate a comparison plot of feature importance across all models.
    """
    print("Generating model comparison plot...")
    
    # Get top 15 features across all models
    all_features = set()
    for importance_df in all_importances.values():
        all_features.update(importance_df.head(15)['Feature'].tolist())
    
    # Create comparison DataFrame
    comparison_data = []
    for feature in all_features:
        row = {'Feature': feature}
        for model_name, importance_df in all_importances.items():
            feature_row = importance_df[importance_df['Feature'] == feature]
            if not feature_row.empty:
                row[model_name.upper()] = feature_row['Mean_Abs_SHAP'].values[0]
            else:
                row[model_name.upper()] = 0
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['Max_Importance'] = comparison_df[['RF', 'SVM', 'GB']].max(axis=1)
    comparison_df = comparison_df.sort_values('Max_Importance', ascending=False).head(15)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(comparison_df))
    width = 0.25
    
    ax.bar(x - width, comparison_df['RF'], width, label='Random Forest', alpha=0.8)
    ax.bar(x, comparison_df['SVM'], width, label='SVM', alpha=0.8)
    ax.bar(x + width, comparison_df['GB'], width, label='Gradient Boosting', alpha=0.8)
    
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute SHAP Value', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Comparison Across Models', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Feature'], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(EXPLAINABILITY_DIR, 'models_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {output_path}")
    
    # Save comparison table
    comparison_df.to_csv(os.path.join(EXPLAINABILITY_DIR, 'models_comparison.csv'), 
                        index=False, float_format='%.6f')


def generate_markdown_report(all_importances, model_names):
    """
    Generate a comprehensive Markdown report.
    """
    print("Generating Markdown report...")
    
    report_path = os.path.join(EXPLAINABILITY_DIR, 'EXPLAINABILITY_REPORT.md')
    
    with open(report_path, 'w') as f:
        f.write("# Model Explainability Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report provides comprehensive explainability analysis for the three AMP prediction models ")
        f.write("using SHAP (SHapley Additive exPlanations). SHAP values provide a unified measure of feature ")
        f.write("importance and show how each feature contributes to individual predictions.\n\n")
        
        f.write("### Models Analyzed\n\n")
        f.write("1. **Random Forest (RF)** - Ensemble of decision trees\n")
        f.write("2. **Support Vector Machine (SVM)** - Kernel-based classifier\n")
        f.write("3. **Gradient Boosting (GB)** - Sequential ensemble method\n\n")
        
        f.write("---\n\n")
        
        f.write("## What is SHAP?\n\n")
        f.write("SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain machine learning ")
        f.write("model predictions. It assigns each feature an importance value for a particular prediction, ")
        f.write("showing how much each feature contributed to moving the prediction away from the base value.\n\n")
        
        f.write("### Key Advantages\n\n")
        f.write("- **Model-agnostic:** Works with any machine learning model\n")
        f.write("- **Theoretically sound:** Based on Shapley values from cooperative game theory\n")
        f.write("- **Consistent:** Features that contribute more receive higher importance\n")
        f.write("- **Local and global:** Explains individual predictions and overall model behavior\n\n")
        
        f.write("---\n\n")
        
        # Feature importance for each model
        for model_name in model_names:
            f.write(f"## {model_name.upper()} Model Analysis\n\n")
            
            importance_df = all_importances[model_name]
            
            f.write(f"### Top 10 Most Important Features\n\n")
            f.write("| Rank | Feature | Mean Abs SHAP | Mean SHAP |\n")
            f.write("|------|---------|---------------|----------|\n")
            
            for _, row in importance_df.head(10).iterrows():
                f.write(f"| {int(row['Rank'])} | {row['Feature']} | ")
                f.write(f"{row['Mean_Abs_SHAP']:.6f} | {row['Mean_SHAP']:.6f} |\n")
            
            f.write("\n")
            f.write(f"### Visualizations\n\n")
            f.write(f"1. **Summary Plot:** `{model_name}_summary_plot.png`\n")
            f.write(f"   - Shows distribution of SHAP values for each feature\n")
            f.write(f"   - Color indicates feature value (red = high, blue = low)\n\n")
            
            f.write(f"2. **Bar Plot:** `{model_name}_bar_plot.png`\n")
            f.write(f"   - Shows mean absolute SHAP values (overall feature importance)\n\n")
            
            f.write(f"3. **Waterfall Plots:** `{model_name}_waterfall_sample_*.png`\n")
            f.write(f"   - Shows how features contribute to individual predictions\n")
            f.write(f"   - Demonstrates model decision-making process\n\n")
            
            f.write(f"4. **Dependence Plots:** `{model_name}_dependence_*.png`\n")
            f.write(f"   - Shows relationship between feature values and SHAP values\n")
            f.write(f"   - Reveals non-linear relationships and interactions\n\n")
            
            f.write("---\n\n")
        
        f.write("## Model Comparison\n\n")
        f.write("The comparison plot (`models_comparison.png`) shows how feature importance varies ")
        f.write("across the three models. This helps identify:\n\n")
        f.write("- **Consensus features:** Important across all models\n")
        f.write("- **Model-specific features:** Important for specific model types\n")
        f.write("- **Robustness:** Features consistently important are more reliable\n\n")
        
        f.write("---\n\n")
        
        f.write("## Interpreting SHAP Values\n\n")
        f.write("### SHAP Value Interpretation\n\n")
        f.write("- **Positive SHAP value:** Feature pushes prediction toward positive class (AMP)\n")
        f.write("- **Negative SHAP value:** Feature pushes prediction toward negative class (non-AMP)\n")
        f.write("- **Magnitude:** Larger absolute value = stronger influence\n\n")
        
        f.write("### Summary Plot Interpretation\n\n")
        f.write("- **Vertical axis:** Features ranked by importance\n")
        f.write("- **Horizontal axis:** SHAP value (impact on prediction)\n")
        f.write("- **Color:** Feature value (red = high, blue = low)\n")
        f.write("- **Density:** Distribution of SHAP values across samples\n\n")
        
        f.write("---\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("This explainability analysis demonstrates that our AMP prediction models are **not black boxes**. ")
        f.write("Through SHAP analysis, we can:\n\n")
        f.write("1. Identify which physicochemical features are most important for predictions\n")
        f.write("2. Understand how each feature contributes to individual predictions\n")
        f.write("3. Validate that models use biologically relevant features\n")
        f.write("4. Build trust in model predictions through transparency\n\n")
        
        f.write("The comprehensive visualizations and feature importance rankings provide clear insights ")
        f.write("into model decision-making, making the models interpretable and trustworthy for ")
        f.write("antimicrobial peptide prediction.\n\n")
        
        f.write("---\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("### Per-Model Files\n\n")
        f.write("For each model (RF, SVM, GB):\n")
        f.write("- `{model}_summary_plot.png` - SHAP summary plot\n")
        f.write("- `{model}_bar_plot.png` - Feature importance bar plot\n")
        f.write("- `{model}_waterfall_sample_*.png` - Individual prediction explanations\n")
        f.write("- `{model}_dependence_*.png` - Feature dependence plots\n")
        f.write("- `{model}_feature_importance.csv` - Feature importance table\n\n")
        
        f.write("### Comparison Files\n\n")
        f.write("- `models_comparison.png` - Feature importance across models\n")
        f.write("- `models_comparison.csv` - Comparison data table\n")
        f.write("- `EXPLAINABILITY_REPORT.md` - This report\n\n")
    
    print(f"Markdown report saved to {report_path}")


def main():
    """Main function to run complete explainability analysis."""
    print("=" * 80)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 80)
    print()
    
    # Load models and data
    models, scaler, X_test, y_test = load_models_and_data()
    feature_names = X_test.columns.tolist()
    
    # Store all importance DataFrames for comparison
    all_importances = {}
    
    # Process each model
    for model_name, model in models.items():
        print("\n" + "=" * 80)
        print(f"Processing {model_name.upper()} Model")
        print("=" * 80)
        
        # Create explainer
        explainer = create_shap_explainer(model, model_name, X_test)
        
        # Calculate SHAP values
        shap_values, X_sample = calculate_shap_values(explainer, X_test, model_name)
        
        # Generate visualizations
        plot_summary_plot(shap_values, X_sample, model_name, feature_names)
        plot_bar_plot(shap_values, X_sample, model_name, feature_names)
        plot_waterfall_examples(shap_values, X_sample, model_name, feature_names, n_examples=3)
        plot_dependence_plots(shap_values, X_sample, model_name, feature_names, top_n=5)
        
        # Generate feature importance table
        importance_df = generate_feature_importance_table(shap_values, feature_names, model_name)
        all_importances[model_name] = importance_df
        
        print(f"\n{model_name.upper()} analysis complete!")
    
    # Generate comparison visualizations
    print("\n" + "=" * 80)
    print("Generating Model Comparisons")
    print("=" * 80)
    generate_comparison_plot(all_importances, list(models.keys()))
    
    # Generate comprehensive report
    generate_markdown_report(all_importances, list(models.keys()))
    
    print("\n" + "=" * 80)
    print("EXPLAINABILITY ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll reports saved to: {EXPLAINABILITY_DIR}")
    print("\nGenerated files:")
    print("  - SHAP summary plots (3)")
    print("  - Feature importance bar plots (3)")
    print("  - Waterfall plots (9 total, 3 per model)")
    print("  - Dependence plots (15 total, 5 per model)")
    print("  - Feature importance tables (3 CSV files)")
    print("  - Model comparison plot and table")
    print("  - Comprehensive Markdown report")
    print()


if __name__ == "__main__":
    main()
