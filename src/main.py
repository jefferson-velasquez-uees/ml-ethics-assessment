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

Author: Jefferson Velasquez
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
    quality_report = processor.validate_quality()

    # =====================================================================
    # PHASE 2: Target Generation & Exploratory Data Analysis (EDA)
    # =====================================================================
    print("\n" + "=" * 70)
    print("  PHASE 2: TARGET GENERATION & EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # Generate target variable (Risk) from UCI study correlations
    processor.generate_target_variable()

    visualizer = ResultsVisualizer(output_dir=assets_dir)
    visualizer.plot_target_distribution(processor.df)
    visualizer.plot_age_distribution_by_risk(processor.df)

    # =====================================================================
    # PHASE 3: Bias Analysis, Null Handling & Preprocessing
    # =====================================================================
    print("\n" + "=" * 70)
    print("  PHASE 3: BIAS ANALYSIS, NULL HANDLING & PREPROCESSING")
    print("=" * 70)

    processor.analyze_bias_indicators()

    # Handle nulls BEFORE preprocessing (key decision documented)
    processor.handle_nulls(strategy='category')

    processor.preprocess()
    X_train, X_test, y_train, y_test = processor.split()

    # Create DataFrames with feature names (needed for SHAP/LIME)
    X_train_df = processor.get_feature_dataframe(X_train)
    X_test_df = processor.get_feature_dataframe(X_test)

    # Correlation matrix (post-encoding)
    visualizer.plot_correlation_matrix(processor.df_processed)

    # =====================================================================
    # PHASE 4: Model Training
    # =====================================================================
    print("\n" + "=" * 70)
    print("  PHASE 4: MODEL TRAINING")
    print("=" * 70)

    engine = CreditModelEngine()
    engine.train_all(X_train, y_train)

    # =====================================================================
    # PHASE 5: Model Evaluation & Comparison
    # =====================================================================
    print("\n" + "=" * 70)
    print("  PHASE 5: MODEL EVALUATION & COMPARISON")
    print("=" * 70)

    results = engine.evaluate_all(X_test, y_test, processor.feature_names)
    cv_results = engine.cross_validate(processor.X, processor.y)

    # Visualize evaluation
    visualizer.plot_confusion_matrices(results, y_test)
    visualizer.plot_metrics_comparison(results)

    # Extract interpretable insights from both models
    lr_coefs = engine.get_logistic_coefficients(processor.feature_names)
    rf_importance = engine.get_rf_feature_importance(processor.feature_names)

    # =====================================================================
    # PHASE 6: EXPLAINABILITY ANALYSIS (SHAP + LIME)
    # =====================================================================
    print("\n" + "=" * 70)
    print("  PHASE 6: EXPLAINABILITY — SHAP + LIME")
    print("=" * 70)

    xai = ExplainabilityEngine(output_dir=assets_dir)

    # --- 6.1 SHAP Global Analysis (Random Forest) ---
    print("\n--- 6.1 SHAP Global Analysis ---")
    xai.compute_shap_values(
        model=engine.models['Random Forest'],
        X_train_df=X_train_df,
        X_test_df=X_test_df,
        model_name="Random Forest"
    )
    xai.plot_shap_summary(X_test_df)
    xai.plot_shap_bar(X_test_df)

    # --- 6.2 SHAP Individual Explanations (2 examples) ---
    print("\n--- 6.2 SHAP Individual Explanations ---")
    rf_predictions = results['Random Forest']['predictions']

    # Find one correct "bad" prediction and one correct "good" prediction
    bad_correct_idx = None
    good_correct_idx = None

    for i in range(len(y_test)):
        if y_test[i] == 1 and rf_predictions[i] == 1 and bad_correct_idx is None:
            bad_correct_idx = i
        if y_test[i] == 0 and rf_predictions[i] == 0 and good_correct_idx is None:
            good_correct_idx = i
        if bad_correct_idx is not None and good_correct_idx is not None:
            break

    # Explain both cases
    if bad_correct_idx is not None:
        print(f"\n   Case 1: Correctly classified as BAD RISK (sample #{bad_correct_idx})")
        xai.plot_shap_individual(X_test_df, bad_correct_idx,
                                 actual_label=1, predicted_label=1)
    else:
        print("   ⚠️  No correctly classified 'bad' sample found")

    if good_correct_idx is not None:
        print(f"\n   Case 2: Correctly classified as GOOD CREDIT (sample #{good_correct_idx})")
        xai.plot_shap_individual(X_test_df, good_correct_idx,
                                 actual_label=0, predicted_label=0)
    else:
        print("   ⚠️  No correctly classified 'good' sample found")

    # --- 6.3 LIME Analysis ---
    print("\n--- 6.3 LIME Analysis ---")

    # Identify categorical feature indices for LIME
    cat_indices = [processor.feature_names.index(c) for c in processor.categorical_cols
                   if c in processor.feature_names]

    xai.setup_lime(
        X_train=X_train,
        feature_names=processor.feature_names,
        categorical_features_indices=cat_indices
    )

    # LIME explanations for the same 2 samples as SHAP (for comparison)
    if bad_correct_idx is not None:
        xai.explain_lime_instance(
            model=engine.models['Random Forest'],
            X_instance=X_test[bad_correct_idx],
            index_label=bad_correct_idx,
            actual=1, predicted=1,
            num_features=9
        )

    if good_correct_idx is not None:
        xai.explain_lime_instance(
            model=engine.models['Random Forest'],
            X_instance=X_test[good_correct_idx],
            index_label=good_correct_idx,
            actual=0, predicted=0,
            num_features=9
        )

    # --- 6.4 Comparison Visualization ---
    print("\n--- 6.4 Method Comparison ---")
    shap_ranking = xai.get_shap_feature_ranking(X_test_df)
    visualizer.plot_importance_comparison(shap_ranking, lr_coefs, rf_importance)

    # =====================================================================
    # PHASE 7: SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("  PHASE 7: SUMMARY")
    print("=" * 70)

    print("\n📊 Generated Visualizations:")
    for f in sorted(os.listdir(assets_dir)):
        if f.endswith('.png'):
            print(f"   • {f}")

    print(f"\n📋 KEY FINDINGS:")
    print(f"   Dataset: 1000 applicants, 9 features, {quality_report['total_nulls']} null values")
    print(f"   Class balance: 70% good / 30% bad")
    for name, res in results.items():
        print(f"   {name}: Accuracy={res['accuracy']:.3f} | F1={res['f1']:.3f} | AUC={res['roc_auc']:.3f}")

    print(f"\n   Top 3 features (SHAP): {shap_ranking['feature'].head(3).tolist()}")
    print(f"   Top 3 features (LR):   {lr_coefs['feature'].head(3).tolist()}")
    print(f"   Top 3 features (RF):   {rf_importance['feature'].head(3).tolist()}")

    print("\n✅ ANALYSIS COMPLETE")
    print("   All visualizations saved in the 'assets/' directory.")
    print("   Refer to README.md for the interpretive analysis and ethical reflection.")


if __name__ == "__main__":
    main()
