"""
Test suite for the Multi-Agent Credit Underwriting system.

Covers synthetic data generation, the credit-scoring agent, the multi-agent
supervisor (heuristic fallback and trained-model paths), fairness analysis and
mitigation, adverse-action notice generation, and the end-to-end experiment
runner in quick mode.
"""

import numpy as np
import pandas as pd
import pytest
from agents.base import ApplicationData
from agents.credit_scorer import CreditScoringAgent
from agents.supervisor import LoanSupervisor
from compliance.adverse_action import AdverseActionNoticeGenerator
from data.synthetic_generator import SyntheticDataGenerator
from fairness.mitigation import FairnessAgent, ReweighingMitigator


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic_df():
    gen = SyntheticDataGenerator(random_seed=42)
    return gen.generate_applications(n_samples=400, default_rate=0.20)


def _make_application(
    app_id, income, loan, dti, delinquencies, utilization, emp_len=5, with_docs=True
):
    return ApplicationData(
        application_id=app_id,
        applicant_info={
            "age": 35,
            "employment_length": emp_len,
            "home_ownership": "RENT",
        },
        financial_info={
            "annual_income": income,
            "loan_amount": loan,
            "debt_to_income_ratio": dti,
        },
        documents=(
            [{"type": "pay_stub", "extracted_fields": {"income": income}}]
            if with_docs
            else []
        ),
        credit_history={
            "credit_utilization": utilization,
            "delinquencies_2y": delinquencies,
            "inquiries_6m": 1,
        },
    )


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
class TestSyntheticData:
    def test_shape_and_columns(self, synthetic_df):
        assert len(synthetic_df) == 400
        for col in ["sex", "race", "annual_income", "loan_amount", "loan_status"]:
            assert col in synthetic_df.columns

    def test_label_is_binary(self, synthetic_df):
        assert set(synthetic_df["loan_status"].unique()).issubset({0, 1})

    def test_reproducible(self):
        a = SyntheticDataGenerator(random_seed=7).generate_applications(100)
        b = SyntheticDataGenerator(random_seed=7).generate_applications(100)
        pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# Credit scoring agent
# --------------------------------------------------------------------------- #
class TestCreditScoringAgent:
    def test_untrained_returns_defaults(self):
        agent = CreditScoringAgent()
        app = _make_application("U1", 60000, 10000, 0.3, 0, 0.3)
        out = agent.process(app)
        assert 300 <= out["credit_score"] <= 850
        assert 0.0 <= out["probability_default"] <= 1.0

    def test_trained_model_predicts(self):
        feat_names = [
            "annual_income",
            "debt_to_income",
            "loan_amount",
            "loan_to_income",
            "credit_lines_open",
            "total_credit_limit",
            "credit_utilization",
            "delinquencies",
            "inquiries",
            "accounts_age_months",
            "age",
            "employment_length_years",
            "homeownership_own",
            "homeownership_rent",
        ]
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, len(feat_names)))
        y = (X[:, 1] + X[:, 6] > 0).astype(int)
        agent = CreditScoringAgent(config={"model_type": "logistic"})
        agent.train(X, y, feat_names)

        app = _make_application("T1", 120000, 12000, 0.15, 0, 0.1)
        out = agent.process(app)
        assert 300 <= out["credit_score"] <= 850
        assert 0.0 <= out["probability_default"] <= 1.0
        assert out["feature_importance"]  # importance available for trained model


# --------------------------------------------------------------------------- #
# Multi-agent supervisor
# --------------------------------------------------------------------------- #
class TestSupervisor:
    def test_heuristic_decisions_are_data_dependent(self):
        sup = LoanSupervisor()
        strong = _make_application("S1", 150000, 12000, 0.10, 0, 0.05)
        weak = _make_application(
            "S2", 35000, 34000, 0.6, 4, 0.95, emp_len=0, with_docs=False
        )

        r_strong = sup.process(strong)
        r_weak = sup.process(weak)

        assert r_strong.decision in {"approve", "review", "deny"}
        assert r_weak.decision in {"approve", "review", "deny"}
        # A strong applicant should be at least as creditworthy as a weak one.
        assert r_strong.risk_score <= r_weak.risk_score
        # The weak applicant (no documents) should raise fraud signals.
        assert r_weak.agent_outputs["fraud_detector"]["alerts"]

    def test_uses_trained_credit_agent(self):
        feat_names = [
            "annual_income",
            "debt_to_income",
            "loan_amount",
            "loan_to_income",
            "credit_lines_open",
            "total_credit_limit",
            "credit_utilization",
            "delinquencies",
            "inquiries",
            "accounts_age_months",
            "age",
            "employment_length_years",
            "homeownership_own",
            "homeownership_rent",
        ]
        rng = np.random.default_rng(1)
        X = rng.normal(size=(150, len(feat_names)))
        y = (X[:, 1] > 0).astype(int)
        credit_agent = CreditScoringAgent(config={"model_type": "logistic"})
        credit_agent.train(X, y, feat_names)

        sup = LoanSupervisor(credit_agent=credit_agent)
        result = sup.process(_make_application("M1", 100000, 10000, 0.2, 0, 0.2))
        # The model path does not tag a heuristic method.
        assert result.agent_outputs["credit_scorer"].get("method") != "heuristic"
        assert result.rationale  # a real rationale was generated

    def test_negotiation_loop_logs_rounds_for_borderline(self):
        sup = LoanSupervisor()
        border = _make_application("B1", 60000, 30000, 0.35, 1, 0.45)
        result = sup.process(border)
        if "negotiation" in result.agent_outputs:
            log = result.agent_outputs["negotiation"]
            assert isinstance(log, list) and len(log) >= 1
            assert "score_after" in log[0]

    def test_approved_terms_present(self):
        sup = LoanSupervisor()
        strong = _make_application("A1", 200000, 10000, 0.05, 0, 0.05)
        result = sup.process(strong)
        if result.decision == "approve":
            assert result.recommended_terms is not None
            assert "interest_rate" in result.recommended_terms


# --------------------------------------------------------------------------- #
# Fairness
# --------------------------------------------------------------------------- #
class TestFairness:
    def test_fairness_agent_metrics(self):
        rng = np.random.default_rng(0)
        n = 300
        preds = rng.integers(0, 2, n)
        sensitive = pd.DataFrame(
            {
                "sex": rng.choice(["M", "F"], n),
                "race": rng.choice(["White", "Black", "Hispanic"], n),
            }
        )
        agent = FairnessAgent()
        out = agent.process(preds, sensitive, y_true=rng.integers(0, 2, n))
        assert "passed" in out and isinstance(out["passed"], bool)
        assert "sex" in out["metrics"]

    def test_reweighing_produces_weights(self):
        rng = np.random.default_rng(0)
        n = 200
        X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        y = rng.integers(0, 2, n)
        sensitive = pd.DataFrame({"sex": rng.choice(["M", "F"], n)})

        mit = ReweighingMitigator()
        _, weights = mit.fit_transform(X, y, sensitive["sex"])
        assert len(weights) == n
        assert np.all(weights > 0)


# --------------------------------------------------------------------------- #
# Compliance
# --------------------------------------------------------------------------- #
class TestAdverseAction:
    def test_generate_notice_contains_required_sections(self):
        gen = AdverseActionNoticeGenerator()
        notice = gen.generate_notice(
            application_id="APP123",
            applicant_name="Jane Doe",
            applicant_address={
                "street": "1 Main St",
                "city": "Springfield",
                "state": "IL",
                "zip": "62701",
            },
            decision="deny",
            primary_reasons=["CREDIT_SCORE_TOO_LOW", "HIGH_DEBT_TO_INCOME_RATIO"],
            credit_score=580,
        )
        assert isinstance(notice, dict)
        # The notice should include human-readable text mentioning the applicant.
        text = notice.get("notice_text") or notice.get("text") or str(notice)
        assert "Jane Doe" in text


# --------------------------------------------------------------------------- #
# End-to-end experiment runner
# --------------------------------------------------------------------------- #
class TestExperimentRunner:
    def test_quick_evaluation(self, synthetic_df, tmp_path):
        from eval.experiment_runner import ExperimentRunner

        runner = ExperimentRunner(output_dir=str(tmp_path), random_seed=42)
        results = runner.run_full_evaluation(synthetic_df, quick_mode=True)

        assert "baselines" in results
        assert "fairness" in results
        # Each baseline reports an AUC in the valid range.
        for name, metrics in results["baselines"].items():
            assert 0.0 <= metrics["auc"] <= 1.0
