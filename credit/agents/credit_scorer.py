"""
Credit Scoring Agent - Predicts default probability using ML models
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from .base import ApplicationData, BaseAgent

logger = logging.getLogger(__name__)


class CreditScoringAgent(BaseAgent):
    """
    Credit scoring agent that uses ML models to predict default probability.
    Supports multiple models: Logistic Regression, LightGBM, Neural Network.
    """

    def __init__(
        self, agent_id: str = "credit_scorer", config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(agent_id, config)
        self.model = None

        self.model_type = self.config.get("model_type", "lightgbm")
        self.feature_names = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list):
        """Train the credit scoring model"""
        self.feature_names = feature_names

        if self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression

            self.model = LogisticRegression(random_state=42, max_iter=1000)

        elif self.model_type == "lightgbm":
            import lightgbm as lgb

            self.model = lgb.LGBMClassifier(
                random_state=42, n_estimators=100, learning_rate=0.05, max_depth=6
            )

        elif self.model_type == "xgboost":
            import xgboost as xgb

            self.model = xgb.XGBClassifier(
                random_state=42, n_estimators=100, learning_rate=0.05, max_depth=6
            )

        elif self.model_type == "neural_net":
            from sklearn.neural_network import MLPClassifier

            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32), random_state=42, max_iter=500
            )

        self.logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        self.logger.info("Model training complete")

    def process(self, application: ApplicationData) -> Dict[str, Any]:
        """
        Process application and return credit score and default probability.
        """
        if self.model is None:
            self.logger.warning("Model not trained, returning default scores")
            return {
                "credit_score": 650,
                "probability_default": 0.20,
                "confidence": 0.5,
                "features": {},
            }

        # Extract features from application
        features = self._extract_features(application)
        feature_vector = self._features_to_vector(features)

        # Predict
        prob_default = self.model.predict_proba(feature_vector)[0, 1]
        credit_score = self._probability_to_credit_score(prob_default)

        # Get feature importance
        feature_importance = self._get_feature_importance()

        return {
            "credit_score": int(credit_score),
            "probability_default": float(prob_default),
            "confidence": 0.85,  # Can be refined with calibration
            "features": features,
            "feature_importance": feature_importance,
        }

    def _extract_features(self, application: ApplicationData) -> Dict[str, float]:
        """Extract features from application data"""
        financial = application.financial_info
        applicant = application.applicant_info
        credit_hist = application.credit_history or {}

        features = {
            # Financial features
            "annual_income": financial.get("annual_income", 50000),
            "debt_to_income": financial.get("debt_to_income_ratio", 0.3),
            "loan_amount": financial.get("loan_amount", 10000),
            "loan_to_income": financial.get("loan_amount", 10000)
            / max(financial.get("annual_income", 50000), 1),
            # Credit history features
            "credit_lines_open": credit_hist.get("credit_lines_open", 5),
            "total_credit_limit": credit_hist.get("total_credit_limit", 50000),
            "credit_utilization": credit_hist.get("credit_utilization", 0.3),
            "delinquencies": credit_hist.get("delinquencies_2y", 0),
            "inquiries": credit_hist.get("inquiries_6m", 1),
            "accounts_age_months": credit_hist.get("oldest_account_months", 60),
            # Applicant features
            "age": applicant.get("age", 35),
            "employment_length_years": applicant.get("employment_length", 5),
            "homeownership_own": 1 if applicant.get("home_ownership") == "OWN" else 0,
            "homeownership_rent": 1 if applicant.get("home_ownership") == "RENT" else 0,
        }

        return features

    def _features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array in correct order"""
        if not self.feature_names:
            self.feature_names = list(features.keys())

        vector = np.array([features.get(name, 0.0) for name in self.feature_names])
        return vector.reshape(1, -1)

    def _probability_to_credit_score(self, prob_default: float) -> float:
        """Convert default probability to FICO-like credit score (300-850)"""
        # Inverse sigmoid-like mapping
        # Low PD -> High score, High PD -> Low score
        score = 850 - (prob_default * 550)
        return max(300, min(850, score))

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.model is None or not self.feature_names:
            return {}

        try:
            if hasattr(self.model, "feature_importances_"):
                # Tree-based models
                importance = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                # Linear models
                importance = np.abs(self.model.coef_[0])
            else:
                return {}

            return dict(zip(self.feature_names, importance.tolist()))
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            return {}
