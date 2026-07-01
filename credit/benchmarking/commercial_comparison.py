"""
Commercial Credit Scoring Benchmark Module
Compares ML-based credit underwriting against FICO and VantageScore.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from benchmark comparison"""

    model_name: str
    auc_roc: float
    precision: float
    recall: float
    f1_score: float
    approval_rate: float
    default_rate_approved: float
    revenue_per_loan: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "auc_roc": self.auc_roc,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "approval_rate": self.approval_rate,
            "default_rate_approved": self.default_rate_approved,
            "revenue_per_loan": self.revenue_per_loan,
        }


class CommercialCreditBenchmark:
    """
    Benchmark credit underwriting models against commercial credit scores.
    Compares ML models with FICO, VantageScore, and other traditional systems.
    """

    def __init__(self):
        # Industry benchmark thresholds
        self.fico_thresholds = {
            "very_poor": (300, 579),
            "fair": (580, 669),
            "good": (670, 739),
            "very_good": (740, 799),
            "exceptional": (800, 850),
        }

        self.vantage_thresholds = {
            "very_poor": (300, 499),
            "poor": (500, 600),
            "fair": (601, 660),
            "good": (661, 780),
            "excellent": (781, 850),
        }

        # Typical approval thresholds
        self.approval_thresholds = {
            "fico": 620,  # Common subprime threshold
            "vantage": 600,
            "ml_model": 0.5,  # Probability threshold
        }

    def simulate_fico_score(self, features: pd.DataFrame) -> np.ndarray:
        """
        Simulate FICO score based on credit features.

        FICO Score Factors:
        - Payment History (35%)
        - Amounts Owed (30%)
        - Length of Credit History (15%)
        - New Credit (10%)
        - Credit Mix (10%)
        """
        n = len(features)

        def _get_col(col: str, fallback: np.ndarray) -> np.ndarray:
            if col in features.columns:
                return features[col].values
            return fallback

        credit_util = _get_col("credit_utilization", np.random.beta(2, 5, n))
        delinquencies = _get_col("delinquencies_2y", np.random.poisson(0.3, n))
        inquiries = _get_col("inquiries_6m", np.random.poisson(1.2, n))
        account_age = _get_col("oldest_account_months", np.random.exponential(80, n))
        credit_lines = _get_col("credit_lines_open", np.random.poisson(8, n))

        # FICO score simulation (simplified)
        base_score = 850

        # Payment history penalty (35%)
        payment_penalty = delinquencies * 80

        # Amounts owed penalty (30%)
        utilization_penalty = credit_util * 200

        # Length of history bonus (15%)
        history_bonus = np.minimum(account_age / 240, 1.0) * 100

        # New credit penalty (10%)
        inquiry_penalty = inquiries * 15

        # Credit mix bonus (10%)
        mix_bonus = np.minimum(credit_lines / 15, 1.0) * 50

        fico_score = (
            base_score
            - payment_penalty
            - utilization_penalty
            + history_bonus
            - inquiry_penalty
            + mix_bonus
            + np.random.normal(0, 20, n)
        )  # Random variation

        return np.clip(fico_score, 300, 850)

    def simulate_vantage_score(self, features: pd.DataFrame) -> np.ndarray:
        """
        Simulate VantageScore based on credit features.

        VantageScore Factors:
        - Payment History (40%)
        - Age and Type of Credit (21%)
        - Credit Utilization (20%)
        - Total Balances (11%)
        - Recent Credit (5%)
        - Available Credit (3%)
        """
        # Similar to FICO but different weightings
        fico_scores = self.simulate_fico_score(features)

        # VantageScore typically correlates highly with FICO but differs
        vantage_scores = fico_scores + np.random.normal(0, 15, len(features))

        return np.clip(vantage_scores, 300, 850)

    def make_fico_decision(self, fico_scores: np.ndarray) -> np.ndarray:
        """Make approval decisions based on FICO scores"""
        return (fico_scores >= self.approval_thresholds["fico"]).astype(int)

    def make_vantage_decision(self, vantage_scores: np.ndarray) -> np.ndarray:
        """Make approval decisions based on VantageScore"""
        return (vantage_scores >= self.approval_thresholds["vantage"]).astype(int)

    def benchmark_models(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        ml_predictions: np.ndarray,
        ml_model_name: str = "ML Model",
    ) -> Dict[str, BenchmarkResult]:
        """
        Comprehensive benchmark comparing ML model with commercial scores.

        Args:
            X_test: Test features
            y_test: True labels (1 = default, 0 = paid)
            ml_predictions: ML model probability predictions
            ml_model_name: Name of ML model

        Returns:
            Dictionary of benchmark results for each scoring method
        """
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        results = {}

        # 1. ML Model
        ml_binary = (ml_predictions >= 0.5).astype(int)
        results[ml_model_name] = BenchmarkResult(
            model_name=ml_model_name,
            auc_roc=float(roc_auc_score(y_test, ml_predictions)),
            precision=float(precision_score(y_test, ml_binary, zero_division=0)),
            recall=float(recall_score(y_test, ml_binary, zero_division=0)),
            f1_score=float(f1_score(y_test, ml_binary, zero_division=0)),
            approval_rate=float((ml_binary == 0).mean()),  # 0 = approve
            default_rate_approved=(
                float(y_test[ml_binary == 0].mean())
                if (ml_binary == 0).sum() > 0
                else 0
            ),
            revenue_per_loan=self._calculate_revenue(y_test, ml_binary),
        )

        # 2. FICO Score
        fico_scores = self.simulate_fico_score(X_test)
        fico_decisions = self.make_fico_decision(fico_scores)
        fico_binary = (
            1 - fico_decisions
        )  # Convert to default prediction (1 = deny, 0 = approve)

        results["FICO"] = BenchmarkResult(
            model_name="FICO",
            auc_roc=float(
                roc_auc_score(y_test, 850 - fico_scores)
            ),  # Invert for default probability
            precision=float(precision_score(y_test, fico_binary, zero_division=0)),
            recall=float(recall_score(y_test, fico_binary, zero_division=0)),
            f1_score=float(f1_score(y_test, fico_binary, zero_division=0)),
            approval_rate=float(fico_decisions.mean()),
            default_rate_approved=(
                float(y_test[fico_decisions == 1].mean())
                if (fico_decisions == 1).sum() > 0
                else 0
            ),
            revenue_per_loan=self._calculate_revenue(y_test, fico_binary),
        )

        # 3. VantageScore
        vantage_scores = self.simulate_vantage_score(X_test)
        vantage_decisions = self.make_vantage_decision(vantage_scores)
        vantage_binary = 1 - vantage_decisions

        results["VantageScore"] = BenchmarkResult(
            model_name="VantageScore",
            auc_roc=float(roc_auc_score(y_test, 850 - vantage_scores)),
            precision=float(precision_score(y_test, vantage_binary, zero_division=0)),
            recall=float(recall_score(y_test, vantage_binary, zero_division=0)),
            f1_score=float(f1_score(y_test, vantage_binary, zero_division=0)),
            approval_rate=float(vantage_decisions.mean()),
            default_rate_approved=(
                float(y_test[vantage_decisions == 1].mean())
                if (vantage_decisions == 1).sum() > 0
                else 0
            ),
            revenue_per_loan=self._calculate_revenue(y_test, vantage_binary),
        )

        return results

    def _calculate_revenue(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        interest_rate: float = 0.15,
        avg_loan_amount: float = 15000,
        loss_given_default: float = 0.75,
    ) -> float:
        """
        Calculate expected revenue per loan.

        Args:
            y_true: True labels (1 = default, 0 = paid)
            predictions: Binary predictions (1 = deny, 0 = approve)
            interest_rate: Annual interest rate
            avg_loan_amount: Average loan amount
            loss_given_default: Percentage of loan lost on default

        Returns:
            Expected revenue per loan
        """
        # Only calculate for approved loans
        approved_mask = predictions == 0

        if approved_mask.sum() == 0:
            return 0.0

        n_approved = approved_mask.sum()
        n_defaults = y_true[approved_mask].sum()
        n_paid = n_approved - n_defaults

        # Revenue from successful loans
        revenue_success = n_paid * (avg_loan_amount * interest_rate * 3)  # 3-year term

        # Loss from defaults
        loss_defaults = n_defaults * (avg_loan_amount * loss_given_default)

        # Net revenue per loan
        net_revenue = (revenue_success - loss_defaults) / n_approved

        return float(net_revenue)

    def get_ml_advantages(self) -> Dict[str, List[str]]:
        """
        Document advantages of ML approach over traditional scoring.
        """
        return {
            "data_utilization": [
                "Can incorporate alternative data sources (utility payments, rent, employment)",
                "Uses non-linear relationships between features",
                "Automatically discovers feature interactions",
                "Handles missing data more flexibly",
            ],
            "performance": [
                "Higher AUC-ROC (typically 0.02-0.05 improvement)",
                "Better calibration of default probabilities",
                "More granular risk segmentation",
                "Improved precision at the margin",
            ],
            "fairness": [
                "Can explicitly optimize for fairness constraints",
                "Transparent bias detection and mitigation",
                "Auditable decision logic",
                "Adaptable to regulatory requirements",
            ],
            "business_value": [
                "Risk-based pricing optimization",
                "Lower default rates in approved population",
                "Higher approval rates for creditworthy applicants",
                "Faster adaptation to market changes",
            ],
            "explainability": [
                "SHAP values for feature importance",
                "Counterfactual explanations",
                "Decision path visualization",
                "Regulatory-compliant adverse action reasons",
            ],
        }

    def get_fico_limitations(self) -> Dict[str, List[str]]:
        """
        Document limitations of traditional FICO scoring.
        """
        return {
            "thin_file_problem": [
                "Cannot score ~26 million 'credit invisible' Americans",
                "Requires 6+ months of credit history",
                "Limited utility for young adults and immigrants",
            ],
            "static_model": [
                "Updated infrequently (FICO 8 from 2009, FICO 10 from 2020)",
                "Slow to incorporate new data sources",
                "Fixed feature weightings",
            ],
            "fairness_concerns": [
                "Historical biases embedded in training data",
                "Proxy discrimination through correlated features",
                "No explicit fairness constraints",
                "Opaque methodology",
            ],
            "business_constraints": [
                "One-size-fits-all threshold",
                "Limited customization for specific loan types",
                "No integration with alternative data",
                "High licensing costs",
            ],
        }

    def generate_comparison_report(
        self, benchmark_results: Dict[str, BenchmarkResult]
    ) -> str:
        """Generate comprehensive comparison report"""
        report = "=" * 80 + "\n"
        report += "COMMERCIAL CREDIT SCORING BENCHMARK COMPARISON\n"
        report += "=" * 80 + "\n\n"

        # Performance comparison table
        report += "PERFORMANCE METRICS\n"
        report += "-" * 80 + "\n"
        report += f"{'Model':<20} {'AUC':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'Approval %':<12}\n"
        report += "-" * 80 + "\n"

        for model_name, result in benchmark_results.items():
            report += (
                f"{model_name:<20} {result.auc_roc:<10.4f} {result.precision:<12.4f} "
            )
            report += f"{result.recall:<10.4f} {result.f1_score:<10.4f} {result.approval_rate*100:<12.1f}\n"

        report += "\n"

        # Business metrics
        report += "BUSINESS IMPACT\n"
        report += "-" * 80 + "\n"
        report += (
            f"{'Model':<20} {'Default Rate (Approved)':<25} {'Revenue per Loan':<20}\n"
        )
        report += "-" * 80 + "\n"

        for model_name, result in benchmark_results.items():
            report += f"{model_name:<20} {result.default_rate_approved*100:<25.2f}% "
            report += f"${result.revenue_per_loan:<20,.2f}\n"

        report += "\n"

        # Key insights
        report += "KEY INSIGHTS\n"
        report += "-" * 80 + "\n"

        ml_model = list(benchmark_results.values())[0]
        fico_model = benchmark_results.get("FICO")

        if fico_model:
            auc_improvement = (
                (ml_model.auc_roc - fico_model.auc_roc) / fico_model.auc_roc * 100
            )
            revenue_improvement = (
                (ml_model.revenue_per_loan - fico_model.revenue_per_loan)
                / abs(fico_model.revenue_per_loan)
                * 100
            )

            report += f"✓ ML model achieves {auc_improvement:+.1f}% improvement in AUC over FICO\n"
            report += f"✓ Revenue per loan improved by {revenue_improvement:+.1f}%\n"
            report += f"✓ Default rate in approved population: {ml_model.default_rate_approved:.1%} (ML) vs {fico_model.default_rate_approved:.1%} (FICO)\n"

        report += "\n" + "=" * 80 + "\n"

        return report


if __name__ == "__main__":
    # Demo benchmark
    print("Commercial Credit Scoring Benchmark")
    print("=" * 80)

    # Generate synthetic test data
    np.random.seed(42)
    n_test = 1000

    X_test = pd.DataFrame(
        {
            "credit_utilization": np.random.beta(2, 5, n_test),
            "delinquencies_2y": np.random.poisson(0.3, n_test),
            "inquiries_6m": np.random.poisson(1.2, n_test),
            "oldest_account_months": np.random.exponential(80, n_test),
            "credit_lines_open": np.random.poisson(8, n_test),
        }
    )

    y_test = np.random.binomial(1, 0.2, n_test)
    ml_predictions = np.random.beta(2, 5, n_test)

    # Run benchmark
    benchmark = CommercialCreditBenchmark()
    results = benchmark.benchmark_models(
        X_test, y_test, ml_predictions, "LightGBM Agent"
    )

    # Generate report
    report = benchmark.generate_comparison_report(results)
    print(report)

    # ML advantages
    print("\nML APPROACH ADVANTAGES:")
    print("-" * 80)
    advantages = benchmark.get_ml_advantages()
    for category, points in advantages.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for point in points:
            print(f"  • {point}")
