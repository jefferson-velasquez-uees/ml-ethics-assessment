# -*- coding: utf-8 -*-
"""
Visualizer: Charts & Comparison Plots
=======================================
Generates all non-SHAP/LIME visualizations:
- Confusion matrices
- Model comparison charts
- Coefficient vs SHAP importance comparison
- EDA plots
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


class ResultsVisualizer:
    """Generates and saves all project visualizations."""

    def __init__(self, output_dir: str = "assets"):
        self.output_dir = output_dir
        # Set consistent style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    # --- EDA Plots ---

    def plot_target_distribution(self, df):
        """Bar plot of credit risk distribution."""
        fig, ax = plt.subplots(figsize=(8, 5))
        # Support both column names
        target_col = 'Risk' if 'Risk' in df.columns else 'credit_risk'
        counts = df[target_col].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.5)

        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{val}\n({val/len(df):.0%})', ha='center', va='bottom', fontweight='bold')

        ax.set_title('Distribución de la Variable Objetivo\n(Riesgo Crediticio)',
                     fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Clasificación de Riesgo')
        ax.set_ylabel('Número de Solicitantes')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(['Good (Bajo Riesgo)', 'Bad (Alto Riesgo)'])
        fig.tight_layout()
        path = f"{self.output_dir}/target_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ Saved: {path}")

    def plot_age_distribution_by_risk(self, df):
        """Overlapping histograms of age by credit risk."""
        fig, ax = plt.subplots(figsize=(10, 5))
        target_col = 'Risk' if 'Risk' in df.columns else 'credit_risk'
        age_col = 'Age' if 'Age' in df.columns else 'age'

        # Map string labels to 0/1 if needed
        risk_vals = df[target_col]
        if risk_vals.dtype == 'object':
            label_map = {'good': 0, 'bad': 1}
            risk_vals = risk_vals.map(label_map)

        for risk, color, label in [(0, '#2ecc71', 'Good'), (1, '#e74c3c', 'Bad')]:
            subset = df.loc[risk_vals == risk, age_col]
            ax.hist(subset, bins=20, alpha=0.6, color=color, label=label, edgecolor='white')

        ax.set_title('Distribución de Edad por Riesgo Crediticio\n(¿Existe sesgo por edad?)',
                     fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Edad')
        ax.set_ylabel('Frecuencia')
        ax.legend(title='Riesgo')
        fig.tight_layout()
        path = f"{self.output_dir}/age_distribution_by_risk.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ Saved: {path}")

    def plot_correlation_matrix(self, df_numeric):
        """Correlation heatmap of numeric features."""
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = df_numeric.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, ax=ax, square=True, linewidths=0.5)
        ax.set_title('Matriz de Correlación\n(Variables numéricas del dataset de crédito)',
                     fontsize=13, fontweight='bold', pad=15)
        fig.tight_layout()
        path = f"{self.output_dir}/correlation_matrix.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ Saved: {path}")

    # --- Model Evaluation Plots ---

    def plot_confusion_matrices(self, results, y_test):
        """Side-by-side confusion matrices for both models."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, (name, res) in zip(axes, results.items()):
            cm = res['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Real')

        fig.suptitle('Matrices de Confusión — Comparación de Modelos',
                     fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        path = f"{self.output_dir}/confusion_matrices_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ Saved: {path}")

    def plot_metrics_comparison(self, results):
        """Grouped bar chart comparing key metrics across models."""
        metrics_list = []
        for name, res in results.items():
            metrics_list.append({
                'Modelo': name,
                'Accuracy': res['accuracy'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'F1-Score': res['f1'],
                'ROC-AUC': res['roc_auc'] if res['roc_auc'] else 0
            })

        df_metrics = pd.DataFrame(metrics_list)
        df_melted = df_metrics.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df_metrics.columns) - 1)
        width = 0.35

        for i, (_, row) in enumerate(df_metrics.iterrows()):
            values = row.drop('Modelo').values.astype(float)
            bars = ax.bar(x + i * width, values, width, label=row['Modelo'], edgecolor='white')
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(df_metrics.columns[1:], fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Valor de la Métrica')
        ax.set_title('Comparación de Rendimiento: Random Forest vs Regresión Logística',
                     fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper right')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        fig.tight_layout()
        path = f"{self.output_dir}/metrics_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ Saved: {path}")

    # --- XAI Comparison Plots ---

    def plot_importance_comparison(self, shap_ranking, lr_coefs, rf_importance):
        """
        Side-by-side comparison of feature rankings from 3 methods:
        SHAP, Logistic Regression coefficients, and RF Gini importance.
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        # 1. SHAP ranking
        top_shap = shap_ranking.head(10)
        axes[0].barh(top_shap['feature'], top_shap['mean_abs_shap'], color='#3498db')
        axes[0].set_title('SHAP\n(Mean |SHAP Value|)', fontsize=11, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Importancia SHAP')

        # 2. Logistic Regression coefficients
        top_lr = lr_coefs.head(10)
        colors_lr = ['#e74c3c' if c > 0 else '#2ecc71' for c in top_lr['coefficient']]
        axes[1].barh(top_lr['feature'], top_lr['abs_coefficient'], color=colors_lr)
        axes[1].set_title('Regresión Logística\n(|Coeficiente|)', fontsize=11, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].set_xlabel('|Coeficiente|')

        # 3. RF Gini importance
        top_rf = rf_importance.head(10)
        axes[2].barh(top_rf['feature'], top_rf['importance'], color='#27ae60')
        axes[2].set_title('Random Forest\n(Gini Importance)', fontsize=11, fontweight='bold')
        axes[2].invert_yaxis()
        axes[2].set_xlabel('Importancia Gini')

        fig.suptitle('Comparación de Explicaciones: ¿Qué variables importan según cada método?',
                     fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        path = f"{self.output_dir}/importance_comparison_3methods.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ Saved: {path}")
