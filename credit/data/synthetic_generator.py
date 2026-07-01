"""
Synthetic Loan Application Data Generator
Generates realistic, deterministic synthetic loan applications for experiments.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generates deterministic synthetic loan application data with realistic distributions.
    Includes demographic features for fairness testing.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)
        self.seed = random_seed

    def generate_applications(
        self, n_samples: int = 1000, default_rate: float = 0.20
    ) -> pd.DataFrame:
        """
        Generate synthetic loan applications with realistic feature distributions.

        Args:
            n_samples: Number of applications to generate
            default_rate: Target default rate (label imbalance)

        Returns:
            DataFrame with application features and labels
        """
        logger.info(f"Generating {n_samples} synthetic applications (seed={self.seed})")

        # Demographics (for fairness testing)
        age = self.rng.normal(38, 12, n_samples).clip(18, 80)
        sex = self.rng.choice(["M", "F"], n_samples, p=[0.52, 0.48])
        race = self.rng.choice(
            ["White", "Black", "Hispanic", "Asian", "Other"],
            n_samples,
            p=[0.60, 0.13, 0.18, 0.06, 0.03],
        )

        # Employment & Income
        employment_length = self.rng.exponential(5, n_samples).clip(0, 40)

        # Income varies by demographics (introduces potential bias)
        base_income = self.rng.lognormal(10.8, 0.6, n_samples)
        # Add systematic income gap (for fairness testing)
        income_multiplier = np.ones(n_samples)
        income_multiplier[sex == "F"] *= 0.92  # Gender pay gap
        income_multiplier[race == "Black"] *= 0.88  # Racial income gap
        income_multiplier[race == "Hispanic"] *= 0.90
        annual_income = (base_income * income_multiplier).clip(15000, 300000)

        # Loan details
        loan_amount = self.rng.lognormal(9.5, 0.8, n_samples).clip(1000, 40000)
        loan_purpose = self.rng.choice(
            [
                "debt_consolidation",
                "credit_card",
                "home_improvement",
                "major_purchase",
                "other",
            ],
            n_samples,
            p=[0.35, 0.25, 0.15, 0.15, 0.10],
        )

        # Credit history
        credit_lines_open = self.rng.poisson(8, n_samples).clip(1, 30)
        total_credit_limit = self.rng.lognormal(10.5, 0.9, n_samples).clip(5000, 200000)
        revolving_balance = self.rng.beta(2, 5, n_samples) * total_credit_limit
        credit_utilization = (revolving_balance / total_credit_limit).clip(0, 1)

        delinquencies_2y = self.rng.poisson(0.3, n_samples).clip(0, 10)
        inquiries_6m = self.rng.poisson(1.2, n_samples).clip(0, 10)
        oldest_account_months = self.rng.exponential(80, n_samples).clip(12, 480)

        # Home ownership
        home_ownership = self.rng.choice(
            ["RENT", "OWN", "MORTGAGE", "OTHER"], n_samples, p=[0.35, 0.12, 0.50, 0.03]
        )

        # Debt-to-income ratio
        monthly_debt = (
            revolving_balance * 0.03 + loan_amount * 0.02
        )  # Approx monthly payments
        monthly_income = annual_income / 12
        debt_to_income = (monthly_debt / monthly_income).clip(0, 1)

        # Generate default label based on risk factors
        default_prob = self._calculate_default_probability(
            credit_utilization=credit_utilization,
            delinquencies=delinquencies_2y,
            debt_to_income=debt_to_income,
            inquiries=inquiries_6m,
            loan_to_income=loan_amount / annual_income,
        )

        # Adjust to target default rate
        default_threshold = np.percentile(default_prob, (1 - default_rate) * 100)
        loan_status = (default_prob > default_threshold).astype(int)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "application_id": [
                    f"APP_{self.seed}_{i:06d}" for i in range(n_samples)
                ],
                "age": age.astype(int),
                "sex": sex,
                "race": race,
                "employment_length": employment_length,
                "annual_income": annual_income,
                "loan_amount": loan_amount,
                "loan_purpose": loan_purpose,
                "debt_to_income_ratio": debt_to_income,
                "credit_lines_open": credit_lines_open.astype(int),
                "total_credit_limit": total_credit_limit,
                "revolving_balance": revolving_balance,
                "credit_utilization": credit_utilization,
                "delinquencies_2y": delinquencies_2y.astype(int),
                "inquiries_6m": inquiries_6m.astype(int),
                "oldest_account_months": oldest_account_months.astype(int),
                "home_ownership": home_ownership,
                "default_probability": default_prob,
                "loan_status": loan_status,  # 0 = paid, 1 = default
            }
        )

        logger.info(
            f"Generated {n_samples} applications with {loan_status.sum()} defaults ({loan_status.mean():.2%})"
        )

        return df

    def _calculate_default_probability(
        self,
        credit_utilization,
        delinquencies,
        debt_to_income,
        inquiries,
        loan_to_income,
    ) -> np.ndarray:
        """
        Calculate base default probability from risk factors.
        Uses realistic logistic-like function.
        """
        # Risk score (higher = more risky)
        risk_score = (
            credit_utilization * 2.0
            + delinquencies * 0.3
            + debt_to_income * 1.5
            + inquiries * 0.1
            + loan_to_income * 1.0
            + self.rng.normal(0, 0.2, len(credit_utilization))  # Random noise
        )

        # Convert to probability via sigmoid
        default_prob = 1 / (
            1 + np.exp(-risk_score + 2)
        )  # Shift to get reasonable range
        return default_prob.clip(0.01, 0.99)

    def generate_application_objects(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to ApplicationData-compatible dictionaries.
        """
        applications = []

        for idx, row in df.iterrows():
            app = {
                "application_id": row["application_id"],
                "applicant_info": {
                    "age": int(row["age"]),
                    "sex": row["sex"],
                    "race": row["race"],
                    "employment_length": float(row["employment_length"]),
                    "home_ownership": row["home_ownership"],
                },
                "financial_info": {
                    "annual_income": float(row["annual_income"]),
                    "loan_amount": float(row["loan_amount"]),
                    "loan_purpose": row["loan_purpose"],
                    "debt_to_income_ratio": float(row["debt_to_income_ratio"]),
                },
                "credit_history": {
                    "credit_lines_open": int(row["credit_lines_open"]),
                    "total_credit_limit": float(row["total_credit_limit"]),
                    "credit_utilization": float(row["credit_utilization"]),
                    "delinquencies_2y": int(row["delinquencies_2y"]),
                    "inquiries_6m": int(row["inquiries_6m"]),
                    "oldest_account_months": int(row["oldest_account_months"]),
                },
                "documents": [],  # Placeholder for document processing
                "metadata": {
                    "true_default_prob": float(row["default_probability"]),
                    "true_label": int(row["loan_status"]),
                },
            }
            applications.append(app)

        return applications


def validate_synthetic_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate synthetic data quality and realism.
    """
    validation = {
        "n_samples": len(df),
        "default_rate": df["loan_status"].mean(),
        "missing_values": df.isnull().sum().sum(),
        "income_range": (df["annual_income"].min(), df["annual_income"].max()),
        "loan_amount_range": (df["loan_amount"].min(), df["loan_amount"].max()),
        "demographics": {
            "sex_distribution": df["sex"].value_counts().to_dict(),
            "race_distribution": df["race"].value_counts().to_dict(),
        },
        "credit_utilization_mean": df["credit_utilization"].mean(),
        "debt_to_income_mean": df["debt_to_income_ratio"].mean(),
    }

    logger.info(f"Data validation: {validation}")
    return validation


if __name__ == "__main__":
    # Test data generation
    generator = SyntheticDataGenerator(random_seed=42)
    df = generator.generate_applications(n_samples=1000)

    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())

    validation = validate_synthetic_data(df)
    print("\nValidation:")
    print(validation)
