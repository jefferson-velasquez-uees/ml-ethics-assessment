# -*- coding: utf-8 -*-
"""
Explainability Engine: SHAP + LIME
====================================
Implements the core XAI techniques for the project:
1. SHAP (SHapley Additive exPlanations) — Global + Local explanations
2. LIME (Local Interpretable Model-agnostic Explanations) — Local explanations

Both are model-agnostic but we apply them to Random Forest (black box)
and compare insights with Logistic Regression coefficients (white box).
"""

import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class ExplainabilityEngine:
    """
    Wraps SHAP and LIME to produce global and local explanations.
    All visualizations are saved to the assets directory.
    """

    def __init__(self, output_dir: str = "assets"):
        self.output_dir = output_dir
        self.shap_values_rf = None
        self.shap_explainer_rf = None
        self.lime_explainer = None

    # =========================================================================
    # SHAP ANALYSIS
    # =========================================================================

    def compute_shap_values(self, model, X_train_df, X_test_df, model_name="Random Forest"):
        """
        Compute SHAP values using TreeExplainer (fast for RF) or KernelExplainer.
        """
        print(f"\n🔮 Computing SHAP values for {model_name}...")

        if model_name == "Random Forest":
            # TreeExplainer is exact and fast for tree-based models
            self.shap_explainer_rf = shap.TreeExplainer(model)
            self.shap_values_rf = self.shap_explainer_rf.shap_values(X_test_df)
            print(f"   ✅ TreeExplainer computed for {X_test_df.shape[0]} samples")

            # For binary classification, shap_values may be:
            # - list [class_0, class_1] (old format)
            # - 3D array (n_samples, n_features, n_classes) (new format)
            if isinstance(self.shap_values_rf, list):
                self.shap_values_positive = self.shap_values_rf[1]
            elif hasattr(self.shap_values_rf, 'ndim') and self.shap_values_rf.ndim == 3:
                self.shap_values_positive = self.shap_values_rf[:, :, 1]
            else:
                self.shap_values_positive = self.shap_values_rf
        else:
            # KernelExplainer for non-tree models (slower but universal)
            background = shap.sample(X_train_df, 100)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_vals = explainer.shap_values(X_test_df.iloc[:50])  # subset for speed
            self.shap_values_positive = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
            print(f"   ✅ KernelExplainer computed for 50 samples")

        return self.shap_values_positive

    def plot_shap_summary(self, X_test_df):
        """
        SHAP Summary Plot (Global): Shows which features matter most
        and how their values affect predictions.
        """
        print("   📊 Generating SHAP Summary Plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values_positive,
            X_test_df,
            show=False,
            plot_size=(12, 8)
        )
        plt.title("SHAP Summary Plot — Random Forest\n(Impacto global de cada variable en la predicción de riesgo crediticio)",
                  fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        path = f"{self.output_dir}/shap_summary_plot.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {path}")

    def plot_shap_bar(self, X_test_df):
        """
        SHAP Bar Plot (Global): Mean absolute SHAP values per feature.
        """
        print("   📊 Generating SHAP Bar Plot...")
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            self.shap_values_positive,
            X_test_df,
            plot_type="bar",
            show=False,
            plot_size=(10, 7)
        )
        plt.title("SHAP Feature Importance (Mean |SHAP|)\nRandom Forest — Variables con mayor impacto global",
                  fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        path = f"{self.output_dir}/shap_bar_importance.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {path}")

    def plot_shap_individual(self, X_test_df, index, actual_label, predicted_label):
        """
        SHAP Waterfall Plot for a SINGLE prediction (Local explanation).
        Handles both binary classification formats from TreeExplainer.
        """
        print(f"   📊 Generating SHAP Individual Explanation (sample #{index})...")

        plt.figure(figsize=(12, 6))

        # Handle different SHAP output formats for binary classification
        if isinstance(self.shap_values_rf, list):
            # Old format: list of arrays [class_0_values, class_1_values]
            sv = self.shap_values_rf[1][index]
            bv = self.shap_explainer_rf.expected_value[1]
        elif self.shap_values_rf.ndim == 3:
            # New format: 3D array (n_samples, n_features, n_classes)
            sv = self.shap_values_rf[index, :, 1]
            bv = self.shap_explainer_rf.expected_value[1]
        else:
            sv = self.shap_values_rf[index]
            bv = self.shap_explainer_rf.expected_value

        explanation = shap.Explanation(
            values=sv,
            base_values=bv,
            data=X_test_df.iloc[index].values,
            feature_names=X_test_df.columns.tolist()
        )

        shap.waterfall_plot(explanation, show=False)

        label_map = {0: 'GOOD (Bajo Riesgo)', 1: 'BAD (Alto Riesgo)'}
        plt.title(
            f"Explicación Individual — Muestra #{index}\n"
            f"Real: {label_map[actual_label]} | Predicción: {label_map[predicted_label]}",
            fontsize=11, fontweight='bold', pad=15
        )
        plt.tight_layout()
        path = f"{self.output_dir}/shap_waterfall_sample_{index}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {path}")

    # =========================================================================
    # LIME ANALYSIS
    # =========================================================================

    def setup_lime(self, X_train, feature_names, categorical_features_indices=None):
        """
        Initialize LIME explainer with training data distribution.
        """
        print("\n🍋 Setting up LIME Explainer...")

        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=['Good Credit', 'Bad Credit'],
            mode='classification',
            categorical_features=categorical_features_indices,
            random_state=42
        )
        print("   ✅ LIME Explainer initialized")

    def explain_lime_instance(self, model, X_instance, index_label, actual, predicted,
                               num_features=10):
        """
        Generate LIME explanation for a single instance.
        Returns the explanation object and saves the plot.
        """
        print(f"\n   🍋 LIME Explanation for sample #{index_label}...")

        explanation = self.lime_explainer.explain_instance(
            X_instance,
            model.predict_proba,
            num_features=num_features,
            num_samples=5000
        )

        # Save as image
        fig = explanation.as_pyplot_figure()
        fig.set_size_inches(12, 6)

        label_map = {0: 'GOOD (Bajo Riesgo)', 1: 'BAD (Alto Riesgo)'}
        fig.suptitle(
            f"LIME — Explicación Local — Muestra #{index_label}\n"
            f"Real: {label_map[actual]} | Predicción: {label_map[predicted]}",
            fontsize=11, fontweight='bold'
        )
        fig.tight_layout()
        path = f"{self.output_dir}/lime_explanation_sample_{index_label}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ Saved: {path}")

        return explanation

    # =========================================================================
    # COMPARISON UTILITIES
    # =========================================================================

    def get_shap_feature_ranking(self, X_test_df):
        """Return a DataFrame ranking features by mean absolute SHAP value."""
        mean_abs_shap = np.abs(self.shap_values_positive).mean(axis=0)
        ranking = pd.DataFrame({
            'feature': X_test_df.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        return ranking
