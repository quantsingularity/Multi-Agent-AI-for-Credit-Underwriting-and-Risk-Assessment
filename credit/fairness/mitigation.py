"""
Fairness Agent and Bias Mitigation Strategies
Implements pre-processing, in-training, and post-processing fairness methods.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FairnessAgent:
    """
    Agent responsible for fairness monitoring and bias mitigation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.protected_attributes = self.config.get(
            "protected_attributes", ["sex", "race"]
        )
        self.fairness_threshold = self.config.get("fairness_threshold", 0.05)
        self.mitigation_strategy = self.config.get("mitigation_strategy", "reweighing")

    def process(
        self,
        predictions: np.ndarray,
        sensitive_features: pd.DataFrame,
        y_true: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Analyze fairness metrics for predictions.

        Args:
            predictions: Model predictions (0/1 or probabilities)
            sensitive_features: DataFrame with protected attributes
            y_true: True labels (optional, for equalized odds)

        Returns:
            Dictionary with fairness metrics and pass/fail status
        """
        metrics = {}

        for attr in self.protected_attributes:
            if attr not in sensitive_features.columns:
                continue

            attr_metrics = self._compute_group_metrics(
                predictions, sensitive_features[attr], y_true
            )
            metrics[attr] = attr_metrics

        # Overall fairness status
        passed = all(
            m.get("demographic_parity_diff", 0) <= self.fairness_threshold
            for m in metrics.values()
        )

        return {
            "passed": passed,
            "metrics": metrics,
            "threshold": self.fairness_threshold,
        }

    def _compute_group_metrics(
        self,
        predictions: np.ndarray,
        sensitive_attr: pd.Series,
        y_true: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Compute fairness metrics for a sensitive attribute"""

        if np.issubdtype(predictions.dtype, np.floating):
            pred_binary = (predictions >= 0.5).astype(int)
        else:
            pred_binary = predictions

        # Get unique groups
        groups = sensitive_attr.unique()
        if len(groups) < 2:
            return {"error": "Less than 2 groups"}

        # Compute approval rates per group
        approval_rates = {}
        for group in groups:
            mask = sensitive_attr == group
            approval_rates[str(group)] = pred_binary[mask].mean()

        # Demographic parity difference (max difference between groups)
        rates = list(approval_rates.values())
        dp_diff = max(rates) - min(rates)

        # Disparate impact (min/max ratio)
        disparate_impact = min(rates) / max(rates) if max(rates) > 0 else 1.0

        metrics = {
            "demographic_parity_diff": dp_diff,
            "disparate_impact": disparate_impact,
            "approval_rates": approval_rates,
        }

        # Equalized odds (if true labels available)
        if y_true is not None:
            eo_metrics = self._compute_equalized_odds(
                pred_binary, y_true, sensitive_attr
            )
            metrics.update(eo_metrics)

        return metrics

    def _compute_equalized_odds(
        self, predictions: np.ndarray, y_true: np.ndarray, sensitive_attr: pd.Series
    ) -> Dict[str, float]:
        """Compute equalized odds metrics"""
        groups = sensitive_attr.unique()

        tpr_by_group = {}  # True positive rate
        fpr_by_group = {}  # False positive rate

        for group in groups:
            mask = sensitive_attr == group
            y_true_group = y_true[mask]
            pred_group = predictions[mask]

            # TPR = TP / (TP + FN)
            if y_true_group.sum() > 0:
                tpr = pred_group[y_true_group == 1].mean()
            else:
                tpr = 0.0

            # FPR = FP / (FP + TN)
            if (y_true_group == 0).sum() > 0:
                fpr = pred_group[y_true_group == 0].mean()
            else:
                fpr = 0.0

            tpr_by_group[str(group)] = tpr
            fpr_by_group[str(group)] = fpr

        # Equalized odds difference
        tpr_values = list(tpr_by_group.values())
        fpr_values = list(fpr_by_group.values())

        eo_diff = max(
            max(tpr_values) - min(tpr_values), max(fpr_values) - min(fpr_values)
        )

        return {
            "equalized_odds_diff": eo_diff,
            "tpr_by_group": tpr_by_group,
            "fpr_by_group": fpr_by_group,
        }


class ReweighingMitigator:
    """
    Pre-processing bias mitigation via sample reweighing.
    Implements Kamiran & Calders (2012) reweighing algorithm.
    """

    def __init__(self):
        self.weights_ = None

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, sensitive_feature: pd.Series
    ) -> "ReweighingMitigator":
        """
        Compute sample weights to balance outcomes across protected groups.

        Args:
            X: Feature matrix
            y: Labels
            sensitive_feature: Protected attribute

        Returns:
            Self
        """
        n = len(y)
        weights = np.ones(n)

        # Get unique groups
        groups = sensitive_feature.unique()

        # Overall statistics
        p_y1 = y.mean()
        p_y0 = 1 - p_y1

        for group in groups:
            mask = sensitive_feature == group
            p_group = mask.mean()

            # Conditional probabilities
            p_y1_given_group = y[mask].mean()
            p_y0_given_group = 1 - p_y1_given_group

            # Expected probabilities (if independent)
            expected_p_y1_group = p_y1 * p_group
            expected_p_y0_group = p_y0 * p_group

            # Reweighing factors
            if p_y1_given_group > 0:
                w_y1 = expected_p_y1_group / (p_y1_given_group * p_group)
            else:
                w_y1 = 1.0

            if p_y0_given_group > 0:
                w_y0 = expected_p_y0_group / (p_y0_given_group * p_group)
            else:
                w_y0 = 1.0

            # Assign weights
            weights[mask & (y == 1)] = w_y1
            weights[mask & (y == 0)] = w_y0

        self.weights_ = weights
        logger.info(
            f"Computed reweighing factors: mean={weights.mean():.3f}, std={weights.std():.3f}"
        )

        return self

    def transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Return X and weights"""
        if self.weights_ is None:
            raise ValueError("Must call fit() first")
        return X, self.weights_

    def fit_transform(
        self, X: pd.DataFrame, y: np.ndarray, sensitive_feature: pd.Series
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Fit and transform in one step"""
        self.fit(X, y, sensitive_feature)
        return self.transform(X)


class ThresholdOptimizer:
    """
    Post-processing bias mitigation via group-specific thresholds.
    Optimizes thresholds to satisfy fairness constraints.
    """

    def __init__(self, constraint: str = "demographic_parity"):
        """
        Args:
            constraint: "demographic_parity" or "equalized_odds"
        """
        self.constraint = constraint
        self.thresholds_ = {}

    def fit(
        self, y_pred_proba: np.ndarray, y_true: np.ndarray, sensitive_feature: pd.Series
    ) -> "ThresholdOptimizer":
        """
        Find optimal thresholds for each group.

        Args:
            y_pred_proba: Predicted probabilities
            y_true: True labels
            sensitive_feature: Protected attribute

        Returns:
            Self
        """
        groups = sensitive_feature.unique()

        if self.constraint == "demographic_parity":
            # Equalize approval rates across groups
            overall_approval_rate = (y_pred_proba >= 0.5).mean()

            for group in groups:
                mask = sensitive_feature == group
                probs = y_pred_proba[mask]

                # Find threshold that gives overall approval rate
                threshold = np.percentile(probs, (1 - overall_approval_rate) * 100)
                self.thresholds_[str(group)] = threshold

        elif self.constraint == "equalized_odds":
            # Equalize TPR and FPR across groups
            # Simplified: use same threshold but could optimize per group
            for group in groups:
                self.thresholds_[str(group)] = 0.5

        logger.info(f"Optimized thresholds: {self.thresholds_}")
        return self

    def predict(
        self, y_pred_proba: np.ndarray, sensitive_feature: pd.Series
    ) -> np.ndarray:
        """
        Apply group-specific thresholds.

        Args:
            y_pred_proba: Predicted probabilities
            sensitive_feature: Protected attribute

        Returns:
            Binary predictions
        """
        predictions = np.zeros(len(y_pred_proba), dtype=int)

        for group, threshold in self.thresholds_.items():
            mask = sensitive_feature == group
            predictions[mask] = (y_pred_proba[mask] >= threshold).astype(int)

        return predictions


def compute_fairness_metrics(
    y_pred: np.ndarray, y_true: np.ndarray, sensitive_features: pd.DataFrame
) -> Dict[str, Any]:
    """
    Standalone function to compute all fairness metrics.
    """
    agent = FairnessAgent(
        config={"protected_attributes": list(sensitive_features.columns)}
    )
    return agent.process(y_pred, sensitive_features, y_true)


if __name__ == "__main__":
    # Test fairness metrics
    np.random.seed(42)
    n = 1000

    # Generate test data with bias
    sex = np.random.choice(["M", "F"], n)
    y_true = np.random.binomial(1, 0.3, n)

    # Biased predictions (approve males more)
    y_pred_proba = (
        y_true.astype(float) + 0.2 * (sex == "M") + np.random.normal(0, 0.1, n)
    )
    y_pred_proba = y_pred_proba.clip(0, 1)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    sensitive_df = pd.DataFrame({"sex": sex})

    # Test fairness agent
    agent = FairnessAgent(config={"protected_attributes": ["sex"]})
    results = agent.process(y_pred, sensitive_df, y_true)

    print("Fairness Analysis:")
    print(f"Passed: {results['passed']}")
    print(f"Metrics: {results['metrics']}")

    # Test reweighing
    reweigher = ReweighingMitigator()
    X_dummy = pd.DataFrame({"feature": np.random.randn(n)})
    X_reweighed, weights = reweigher.fit_transform(X_dummy, y_true, pd.Series(sex))
    print(f"\nReweighing: mean weight = {weights.mean():.3f}")

    # Test threshold optimizer
    optimizer = ThresholdOptimizer(constraint="demographic_parity")
    optimizer.fit(y_pred_proba, y_true, pd.Series(sex))
    y_pred_fair = optimizer.predict(y_pred_proba, pd.Series(sex))

    results_fair = agent.process(y_pred_fair, sensitive_df, y_true)
    print("\nAfter threshold optimization:")
    print(f"Passed: {results_fair['passed']}")
    print(f"Metrics: {results_fair['metrics']}")
