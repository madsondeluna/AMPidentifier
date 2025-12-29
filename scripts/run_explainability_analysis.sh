#!/bin/bash

# Script to run SHAP explainability analysis for AMP prediction models
# This script generates comprehensive explainability reports for all three models

set -e

echo "=========================================="
echo "AMP Model Explainability Analysis"
echo "=========================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "model_training/explainability.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Check if models exist
if [ ! -f "model_training/saved_model/amp_model_rf.pkl" ]; then
    echo "Error: Models not found. Please train models first using:"
    echo "  python3 -m model_training.train"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import shap" 2>/dev/null || {
    echo "SHAP not installed. Installing dependencies..."
    pip3 install -r requirements.txt
}

echo ""
echo "Starting SHAP explainability analysis..."
echo "This may take several minutes depending on your system."
echo ""

# Run explainability analysis
python3 -m model_training.explainability

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Reports generated in: model_training/explainability_reports/"
echo ""
echo "Generated files:"
echo "  - EXPLAINABILITY_REPORT.md (comprehensive report)"
echo "  - Summary plots (3 files)"
echo "  - Bar plots (3 files)"
echo "  - Waterfall plots (9 files)"
echo "  - Dependence plots (15 files)"
echo "  - Feature importance tables (3 CSV files)"
echo "  - Model comparison plot and table"
echo ""
echo "To view the report:"
echo "  cat model_training/explainability_reports/EXPLAINABILITY_REPORT.md"
echo ""
