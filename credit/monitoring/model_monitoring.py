"""
Model Monitoring System
Implements comprehensive monitoring for production credit models including:
- Performance tracking (AUC, precision, recall over time)
- Fairness drift detection across demographic groups
- Data quality checks and distribution shifts
- Model degradation alerts
"""

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceSnapshot:
    """Snapshot of model performance at a point in time"""

    timestamp: str
    n_samples: int
    auc_roc: float
    precision: float
    recall: float
    f1_score: float
    approval_rate: float
    default_rate: float
    avg_credit_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FairnessSnapshot:
    """Snapshot of fairness metrics at a point in time"""

    timestamp: str
    protected_attribute: str
    demographic_parity_diff: float
    equalized_odds_diff: float
    disparate_impact: float
    approval_rates_by_group: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualitySnapshot:
    """Snapshot of data quality metrics"""

    timestamp: str
    n_samples: int
    missing_rate: float
    feature_means: Dict[str, float]
    feature_stds: Dict[str, float]
    outlier_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelMonitor:
    """
    Comprehensive model monitoring system for production credit models.
    """

    def __init__(
        self,
        model_id: str,
        baseline_metrics: Optional[Dict[str, float]] = None,
        alert_thresholds: Optional[Dict[str, float]] = None,
        window_size: int = 1000,
    ):
        """
        Args:
            model_id: Unique identifier for the model
            baseline_metrics: Baseline performance metrics from validation
            alert_thresholds: Thresholds for triggering alerts
            window_size: Number of recent predictions to monitor
        """
        self.model_id = model_id
        self.baseline_metrics = baseline_metrics or {}
        self.window_size = window_size

        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "auc_drop": 0.05,  # Alert if AUC drops by more than 0.05
            "fairness_drift": 0.10,  # Alert if fairness metric changes by 0.10
            "data_quality_outlier_rate": 0.15,  # Alert if >15% outliers
            "missing_rate": 0.10,  # Alert if >10% missing values
            "performance_degradation": 0.10,  # Alert if any metric drops >10%
        }

        # Rolling windows for recent data
        self.predictions_window = deque(maxlen=window_size)
        self.true_labels_window = deque(maxlen=window_size)
        self.features_window = deque(maxlen=window_size)
        self.sensitive_features_window = deque(maxlen=window_size)

        # Historical snapshots
        self.performance_history: List[ModelPerformanceSnapshot] = []
        self.fairness_history: List[FairnessSnapshot] = []
        self.data_quality_history: List[DataQualitySnapshot] = []

        # Alerts
        self.active_alerts: List[Dict[str, Any]] = []

        logger.info(f"Initialized ModelMonitor for model {model_id}")

    def log_prediction(
        self,
        prediction: float,
        features: Dict[str, float],
        sensitive_features: Dict[str, Any],
        true_label: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ):
        """
        Log a single prediction for monitoring.

        Args:
            prediction: Model prediction (probability)
            features: Feature values used for prediction
            sensitive_features: Protected attributes (sex, race, etc.)
            true_label: Ground truth label (if available)
            timestamp: Prediction timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        self.predictions_window.append(
            {"prediction": prediction, "timestamp": timestamp, "true_label": true_label}
        )
        self.features_window.append(features)
        self.sensitive_features_window.append(sensitive_features)

        if true_label is not None:
            self.true_labels_window.append(true_label)

    def compute_performance_metrics(self) -> Optional[ModelPerformanceSnapshot]:
        """
        Compute current performance metrics from rolling window.
        Requires ground truth labels to be available.
        """
        if len(self.true_labels_window) < 50:
            logger.warning("Insufficient labeled data for performance metrics")
            return None

        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        # Get recent predictions with labels
        recent_preds = []
        recent_labels = []

        for pred_info in list(self.predictions_window):
            if pred_info["true_label"] is not None:
                recent_preds.append(pred_info["prediction"])
                recent_labels.append(pred_info["true_label"])

        if len(recent_labels) < 50:
            return None

        y_true = np.array(recent_labels)
        y_pred_proba = np.array(recent_preds)
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)

        # Compute metrics
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            recall = recall_score(y_true, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)

            snapshot = ModelPerformanceSnapshot(
                timestamp=datetime.now().isoformat(),
                n_samples=len(y_true),
                auc_roc=float(auc),
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1),
                approval_rate=float(y_pred_binary.mean()),
                default_rate=float(y_true.mean()),
                avg_credit_score=float(850 - y_pred_proba.mean() * 550),
            )

            self.performance_history.append(snapshot)

            # Check for performance degradation
            self._check_performance_degradation(snapshot)

            return snapshot

        except Exception as e:
            logger.error(f"Error computing performance metrics: {e}")
            return None

    def compute_fairness_metrics(
        self, protected_attribute: str = "sex"
    ) -> Optional[FairnessSnapshot]:
        """
        Compute fairness metrics from rolling window.
        """
        if len(self.predictions_window) < 100:
            logger.warning("Insufficient data for fairness metrics")
            return None

        # Extract recent data
        predictions = [p["prediction"] for p in list(self.predictions_window)]
        sensitive_features = list(self.sensitive_features_window)

        if not sensitive_features or protected_attribute not in sensitive_features[0]:
            logger.warning(f"Protected attribute '{protected_attribute}' not available")
            return None

        # Get true labels if available
        true_labels = None
        if len(self.true_labels_window) >= len(predictions):
            true_labels = np.array(list(self.true_labels_window)[-len(predictions) :])

        from fairness.mitigation import FairnessAgent

        sens_df = pd.DataFrame(sensitive_features)
        fairness_agent = FairnessAgent(
            config={"protected_attributes": [protected_attribute]}
        )

        result = fairness_agent.process(np.array(predictions), sens_df, true_labels)

        metrics = result["metrics"].get(protected_attribute, {})

        snapshot = FairnessSnapshot(
            timestamp=datetime.now().isoformat(),
            protected_attribute=protected_attribute,
            demographic_parity_diff=metrics.get("demographic_parity_diff", 0.0),
            equalized_odds_diff=metrics.get("equalized_odds_diff", 0.0),
            disparate_impact=metrics.get("disparate_impact", 1.0),
            approval_rates_by_group=metrics.get("approval_rates", {}),
        )

        self.fairness_history.append(snapshot)

        # Check for fairness drift
        self._check_fairness_drift(snapshot)

        return snapshot

    def compute_data_quality_metrics(self) -> Optional[DataQualitySnapshot]:
        """
        Compute data quality metrics from rolling window.
        """
        if len(self.features_window) < 50:
            logger.warning("Insufficient data for quality metrics")
            return None

        # Convert to DataFrame
        features_df = pd.DataFrame(list(self.features_window))

        # Compute statistics
        feature_means = features_df.mean().to_dict()
        feature_stds = features_df.std().to_dict()
        missing_rate = features_df.isnull().sum().sum() / (
            features_df.shape[0] * features_df.shape[1]
        )

        # Detect outliers using IQR method
        outliers = 0
        for col in features_df.select_dtypes(include=[np.number]).columns:
            Q1 = features_df[col].quantile(0.25)
            Q3 = features_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers += (
                (features_df[col] < (Q1 - 3 * IQR))
                | (features_df[col] > (Q3 + 3 * IQR))
            ).sum()

        outlier_rate = outliers / (features_df.shape[0] * features_df.shape[1])

        snapshot = DataQualitySnapshot(
            timestamp=datetime.now().isoformat(),
            n_samples=len(features_df),
            missing_rate=float(missing_rate),
            feature_means=feature_means,
            feature_stds=feature_stds,
            outlier_rate=float(outlier_rate),
        )

        self.data_quality_history.append(snapshot)

        # Check for data quality issues
        self._check_data_quality(snapshot)

        return snapshot

    def _check_performance_degradation(self, current: ModelPerformanceSnapshot):
        """Check for performance degradation and raise alerts"""
        if not self.baseline_metrics:
            return

        baseline_auc = self.baseline_metrics.get("auc", current.auc_roc)
        auc_drop = baseline_auc - current.auc_roc

        if auc_drop > self.alert_thresholds["auc_drop"]:
            self._raise_alert(
                alert_type="PERFORMANCE_DEGRADATION",
                severity="HIGH",
                message=f"AUC dropped by {auc_drop:.4f} (from {baseline_auc:.4f} to {current.auc_roc:.4f})",
                details={
                    "baseline_auc": baseline_auc,
                    "current_auc": current.auc_roc,
                    "drop": auc_drop,
                },
            )

        # Check other metrics
        for metric in ["precision", "recall", "f1_score"]:
            baseline_value = self.baseline_metrics.get(metric, getattr(current, metric))
            current_value = getattr(current, metric)
            drop = baseline_value - current_value

            if drop > self.alert_thresholds["performance_degradation"]:
                self._raise_alert(
                    alert_type="PERFORMANCE_DEGRADATION",
                    severity="MEDIUM",
                    message=f"{metric.upper()} dropped by {drop:.4f}",
                    details={
                        f"baseline_{metric}": baseline_value,
                        f"current_{metric}": current_value,
                        "drop": drop,
                    },
                )

    def _check_fairness_drift(self, current: FairnessSnapshot):
        """Check for fairness drift"""
        if len(self.fairness_history) < 2:
            return

        # Compare to previous snapshot
        previous = self.fairness_history[-2]

        if previous.protected_attribute != current.protected_attribute:
            return

        dp_drift = abs(
            current.demographic_parity_diff - previous.demographic_parity_diff
        )
        eo_drift = abs(current.equalized_odds_diff - previous.equalized_odds_diff)

        if dp_drift > self.alert_thresholds["fairness_drift"]:
            self._raise_alert(
                alert_type="FAIRNESS_DRIFT",
                severity="HIGH",
                message=f"Demographic parity drift of {dp_drift:.4f} detected for {current.protected_attribute}",
                details={
                    "protected_attribute": current.protected_attribute,
                    "previous_dp_diff": previous.demographic_parity_diff,
                    "current_dp_diff": current.demographic_parity_diff,
                    "drift": dp_drift,
                },
            )

        if eo_drift > self.alert_thresholds["fairness_drift"]:
            self._raise_alert(
                alert_type="FAIRNESS_DRIFT",
                severity="HIGH",
                message=f"Equalized odds drift of {eo_drift:.4f} detected for {current.protected_attribute}",
                details={
                    "protected_attribute": current.protected_attribute,
                    "previous_eo_diff": previous.equalized_odds_diff,
                    "current_eo_diff": current.equalized_odds_diff,
                    "drift": eo_drift,
                },
            )

    def _check_data_quality(self, current: DataQualitySnapshot):
        """Check for data quality issues"""
        if current.missing_rate > self.alert_thresholds["missing_rate"]:
            self._raise_alert(
                alert_type="DATA_QUALITY",
                severity="MEDIUM",
                message=f"High missing rate: {current.missing_rate:.2%}",
                details={"missing_rate": current.missing_rate},
            )

        if current.outlier_rate > self.alert_thresholds["data_quality_outlier_rate"]:
            self._raise_alert(
                alert_type="DATA_QUALITY",
                severity="MEDIUM",
                message=f"High outlier rate: {current.outlier_rate:.2%}",
                details={"outlier_rate": current.outlier_rate},
            )

        # Check for distribution shift
        if len(self.data_quality_history) >= 2:
            previous = self.data_quality_history[-2]

            for feature, current_mean in current.feature_means.items():
                if feature in previous.feature_means:
                    prev_mean = previous.feature_means[feature]
                    prev_std = previous.feature_stds.get(feature, 1.0)

                    if prev_std > 0:
                        z_score = abs(current_mean - prev_mean) / prev_std

                        if z_score > 3:  # 3-sigma rule
                            self._raise_alert(
                                alert_type="DISTRIBUTION_SHIFT",
                                severity="MEDIUM",
                                message=f"Significant distribution shift detected for feature '{feature}'",
                                details={
                                    "feature": feature,
                                    "previous_mean": prev_mean,
                                    "current_mean": current_mean,
                                    "z_score": float(z_score),
                                },
                            )

    def _raise_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Raise a monitoring alert"""
        alert = {
            "model_id": self.model_id,
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "details": details or {},
            "status": "active",
        }

        self.active_alerts.append(alert)
        logger.warning(f"[{severity}] {alert_type}: {message}")

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring dashboard data.
        """
        dashboard = {
            "model_id": self.model_id,
            "last_updated": datetime.now().isoformat(),
            "monitoring_window_size": self.window_size,
            "current_window_size": len(self.predictions_window),
            "performance": {
                "current": (
                    self.performance_history[-1].to_dict()
                    if self.performance_history
                    else None
                ),
                "baseline": self.baseline_metrics,
                "history": [
                    p.to_dict() for p in self.performance_history[-10:]
                ],  # Last 10 snapshots
            },
            "fairness": {
                "current": (
                    [f.to_dict() for f in self.fairness_history[-2:]]
                    if self.fairness_history
                    else []
                ),
                "history": [f.to_dict() for f in self.fairness_history[-10:]],
            },
            "data_quality": {
                "current": (
                    self.data_quality_history[-1].to_dict()
                    if self.data_quality_history
                    else None
                ),
                "history": [d.to_dict() for d in self.data_quality_history[-10:]],
            },
            "alerts": {
                "active": [a for a in self.active_alerts if a["status"] == "active"],
                "total_count": len(self.active_alerts),
                "by_severity": self._count_alerts_by_severity(),
                "by_type": self._count_alerts_by_type(),
            },
        }

        return dashboard

    def _count_alerts_by_severity(self) -> Dict[str, int]:
        """Count active alerts by severity"""
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for alert in self.active_alerts:
            if alert["status"] == "active":
                counts[alert["severity"]] = counts.get(alert["severity"], 0) + 1
        return counts

    def _count_alerts_by_type(self) -> Dict[str, int]:
        """Count active alerts by type"""
        counts = {}
        for alert in self.active_alerts:
            if alert["status"] == "active":
                alert_type = alert["alert_type"]
                counts[alert_type] = counts.get(alert_type, 0) + 1
        return counts

    def acknowledge_alert(self, alert_index: int):
        """Acknowledge and resolve an alert"""
        if 0 <= alert_index < len(self.active_alerts):
            self.active_alerts[alert_index]["status"] = "acknowledged"
            self.active_alerts[alert_index][
                "acknowledged_at"
            ] = datetime.now().isoformat()
            logger.info(f"Alert {alert_index} acknowledged")

    def save_monitoring_report(self, output_path: str):
        """Save monitoring report to JSON file"""
        dashboard = self.get_monitoring_dashboard()

        with open(output_path, "w") as f:
            json.dump(dashboard, f, indent=2)

        logger.info(f"Saved monitoring report to {output_path}")

    def get_performance_trend(
        self, metric: str = "auc_roc", lookback_periods: int = 10
    ) -> Dict[str, Any]:
        """
        Get performance trend for a specific metric.

        Args:
            metric: Metric name ('auc_roc', 'precision', 'recall', 'f1_score')
            lookback_periods: Number of historical periods to analyze

        Returns:
            Dictionary with trend analysis
        """
        if not self.performance_history:
            return {"status": "insufficient_data"}

        recent_history = self.performance_history[-lookback_periods:]
        values = [getattr(snapshot, metric) for snapshot in recent_history]
        timestamps = [snapshot.timestamp for snapshot in recent_history]

        # Compute trend
        if len(values) >= 2:
            trend_slope = (values[-1] - values[0]) / len(values)
            trend_direction = (
                "improving"
                if trend_slope > 0
                else "degrading" if trend_slope < 0 else "stable"
            )
        else:
            trend_slope = 0.0
            trend_direction = "unknown"

        return {
            "metric": metric,
            "current_value": values[-1] if values else None,
            "mean_value": float(np.mean(values)),
            "std_value": float(np.std(values)),
            "min_value": float(np.min(values)),
            "max_value": float(np.max(values)),
            "trend_direction": trend_direction,
            "trend_slope": float(trend_slope),
            "timestamps": timestamps,
            "values": values,
        }


if __name__ == "__main__":
    # Demo model monitoring
    logger.info("Model Monitoring System Demo")

    # Initialize monitor
    monitor = ModelMonitor(
        model_id="credit_model_v1",
        baseline_metrics={
            "auc": 0.85,
            "precision": 0.75,
            "recall": 0.70,
            "f1_score": 0.72,
        },
        window_size=500,
    )

    # Simulate predictions
    np.random.seed(42)
    for i in range(600):
        # Simulate gradual performance degradation
        degradation_factor = 1.0 - (i / 600) * 0.15

        # Generate synthetic prediction and features
        true_default_prob = np.random.beta(2, 5)
        prediction = true_default_prob * degradation_factor + np.random.normal(0, 0.05)
        prediction = np.clip(prediction, 0, 1)

        true_label = int(true_default_prob > 0.5)

        features = {
            "credit_score": np.random.normal(680, 80),
            "debt_to_income": np.random.beta(2, 5),
            "credit_utilization": np.random.beta(2, 5),
            "inquiries": np.random.poisson(2),
        }

        sensitive_features = {
            "sex": np.random.choice(["M", "F"]),
            "race": np.random.choice(["White", "Black", "Hispanic", "Asian"]),
        }

        monitor.log_prediction(prediction, features, sensitive_features, true_label)

        # Compute metrics every 100 predictions
        if (i + 1) % 100 == 0:
            perf = monitor.compute_performance_metrics()
            fairness = monitor.compute_fairness_metrics("sex")
            quality = monitor.compute_data_quality_metrics()

            if perf:
                print(
                    f"\nPrediction {i+1}: AUC={perf.auc_roc:.4f}, Precision={perf.precision:.4f}"
                )

    # Get dashboard
    dashboard = monitor.get_monitoring_dashboard()
    print("\n" + "=" * 80)
    print("MONITORING DASHBOARD")
    print("=" * 80)
    print(f"Active Alerts: {len(dashboard['alerts']['active'])}")
    print(f"Alerts by Severity: {dashboard['alerts']['by_severity']}")
    print(f"Alerts by Type: {dashboard['alerts']['by_type']}")

    # Print some alerts
    for alert in dashboard["alerts"]["active"][:5]:
        print(f"\n[{alert['severity']}] {alert['alert_type']}: {alert['message']}")

    report_path = "monitoring_report.json"
    monitor.save_monitoring_report(report_path)
    print(f"\nMonitoring report saved to {report_path}")
