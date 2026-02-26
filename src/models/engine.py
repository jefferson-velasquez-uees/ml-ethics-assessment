# -*- coding: utf-8 -*-
"""
Model Engine: Random Forest vs Logistic Regression
====================================================
Trains, evaluates, and compares a "black box" (Random Forest) against
an "interpretable" (Logistic Regression) model.
Core thesis: performance vs transparency trade-off.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import cross_val_score


class CreditModelEngine:
    """
    Manages training and evaluation of both models.
    Stores trained models for downstream explainability analysis.
    """

    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',  # Handle 70/30 imbalance
                n_jobs=-1
            ),
            'Regresión Logística': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
                solver='lbfgs'
            )
        }
        self.results = {}

    def train_all(self, X_train, y_train):
        """Train all models and store them."""
        print("\n🏋️ MODEL TRAINING")
        for name, model in self.models.items():
            print(f"\n   Training {name}...")
            model.fit(X_train, y_train)
            print(f"   ✅ {name} trained successfully")

    def evaluate_all(self, X_test, y_test, feature_names=None):
        """
        Evaluate all trained models. Returns structured results dict.
        """
        print("\n📈 MODEL EVALUATION")

        for name, model in self.models.items():
            y_pred = model.predict(X_test)

            # Probabilities for ROC-AUC
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                y_proba = None
                roc_auc = None

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred,
                'probabilities': y_proba,
                'report': classification_report(y_test, y_pred, output_dict=True)
            }

            self.results[name] = metrics

            print(f"\n   === {name} ===")
            print(f"   Accuracy:  {metrics['accuracy']:.3f}")
            print(f"   Precision: {metrics['precision']:.3f}")
            print(f"   Recall:    {metrics['recall']:.3f}")
            print(f"   F1-Score:  {metrics['f1']:.3f}")
            if roc_auc:
                print(f"   ROC-AUC:   {roc_auc:.3f}")
            print(f"   Confusion Matrix:\n{metrics['confusion_matrix']}")

        return self.results

    def cross_validate(self, X, y, cv=5):
        """Cross-validation for robustness check."""
        print("\n🔄 CROSS-VALIDATION (k=5)")
        cv_results = {}

        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            cv_results[name] = {
                'mean_f1': scores.mean(),
                'std_f1': scores.std(),
                'scores': scores
            }
            print(f"   {name}: F1 = {scores.mean():.3f} (±{scores.std():.3f})")

        return cv_results

    def get_logistic_coefficients(self, feature_names):
        """
        Extract and rank logistic regression coefficients.
        This IS the interpretability advantage of Logistic Regression.
        """
        lr_model = self.models['Regresión Logística']
        coefs = lr_model.coef_[0]

        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefs,
            'abs_coefficient': np.abs(coefs),
            'direction': ['↑ Increases Risk' if c > 0 else '↓ Decreases Risk' for c in coefs]
        }).sort_values('abs_coefficient', ascending=False)

        print("\n📋 LOGISTIC REGRESSION COEFFICIENTS (Interpretability)")
        print(coef_df.to_string(index=False))
        return coef_df

    def get_rf_feature_importance(self, feature_names):
        """Extract Random Forest built-in feature importances (Gini-based)."""
        rf_model = self.models['Random Forest']
        importances = rf_model.feature_importances_

        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print("\n🌲 RANDOM FOREST FEATURE IMPORTANCE (Gini)")
        print(imp_df.to_string(index=False))
        return imp_df
