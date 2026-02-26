# -*- coding: utf-8 -*-
"""
Main Pipeline: Explainable AI for Credit Risk Assessment
==========================================================
Orchestrates the complete analysis in 7 sequential phases:
  1. Data Loading & Quality Validation
  2. Exploratory Data Analysis (EDA)
  3. Bias Analysis, Null Handling & Preprocessing
  4. Model Training (Random Forest + Logistic Regression)
  5. Model Evaluation & Comparison
  6. Explainability Analysis (SHAP + LIME)
  7. Summary Report Generation

Dataset: German Credit Data (Kaggle / UCI Machine Learning Repository)
Models: Random Forest (black box) vs Logistic Regression (interpretable)
XAI Techniques: SHAP (global + local) + LIME (local)

Author: Jefferson Velasquez, Frank Macias, Jorge Murillo
Course: Machine Learning — Master's in AI
Assignment: Explainability (XAI) & Ethical AI
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processor import CreditDataProcessor
from src.models.engine import CreditModelEngine
from src.explainability.xai_engine import ExplainabilityEngine
from src.utils.visualizer import ResultsVisualizer


def main():
    print("=" * 70)
    print("  EXPLAINABLE AI (XAI) FOR CREDIT RISK ASSESSMENT")
    print("  Random Forest vs Regresión Logística — SHAP & LIME Analysis")
    print("  Dataset: German Credit Data (Kaggle / UCI)")
    print("=" * 70)

    # --- Configuration ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(project_root, "assets")
    dataset_path = os.path.join(project_root, "src", "data", "kaggle", "german_credit_data.csv")

    os.makedirs(assets_dir, exist_ok=True)

    print(f"\nProject Root: {project_root}")
    print(f"Dataset:      {dataset_path}")
    print(f"Assets:       {assets_dir}")

    if not os.path.exists(dataset_path):
        print(f"\n❌ ERROR: Dataset not found at {dataset_path}")
        print("   Place the German Credit CSV from Kaggle in src/data/kaggle/")
        return

    # =====================================================================
    # PHASE 1: Data Loading & Quality Validation
    # =====================================================================
    print("\n" + "=" * 70)
    print("  PHASE 1: DATA LOADING & QUALITY VALIDATION")
    print("=" * 70)

    processor = CreditDataProcessor(dataset_path)
    df = processor.load_data()
    processor.validate_quality()


if __name__ == "__main__":
    main()
