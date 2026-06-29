"""
Loan Supervisor Agent - Orchestrates the underwriting workflow
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import ApplicationData, BaseAgent, DecisionResult

logger = logging.getLogger(__name__)


class LoanSupervisor(BaseAgent):
    """
    Supervisor agent that orchestrates the entire underwriting workflow.
    Delegates tasks to specialized agents and makes final decisions.
    """

    def __init__(
        self,
        agent_id: str = "supervisor",
        config: Optional[Dict[str, Any]] = None,
        credit_agent: Optional["BaseAgent"] = None,
        fraud_agent: Optional["BaseAgent"] = None,
    ):
        super().__init__(agent_id, config)
        self.decision_threshold_approve = self.config.get("threshold_approve", 0.7)
        self.decision_threshold_deny = self.config.get("threshold_deny", 0.3)
        self.human_review_threshold = self.config.get("human_review_threshold", 0.5)
        self.max_negotiation_rounds = self.config.get("max_negotiation_rounds", 3)

        # Optional real specialized agents. When a credit_agent is supplied the
        # supervisor uses its trained model for scoring; otherwise it falls back
        # to a transparent heuristic computed from the application data (never a
        # hard-coded constant).
        self.credit_agent = credit_agent
        self.fraud_agent = fraud_agent

        # Income-verification tolerance (fractional difference allowed between
        # reported and document-extracted income).
        self.income_tolerance = self.config.get("income_tolerance", 0.15)

    def process(self, application: ApplicationData) -> DecisionResult:
        """
        Main orchestration logic for processing a loan application.

        Steps:
        1. Parse documents
        2. Verify income and employment
        3. Calculate credit score
        4. Assess fraud risk
        5. Check fairness constraints
        6. Generate explanation
        7. Make final decision with human review gating if needed
        """
        self.logger.info(f"Processing application: {application.application_id}")

        agent_outputs = {}
        evidence = []

        # Step 1: Document processing (delegated to document processor)
        doc_result = self._delegate_document_processing(application)
        agent_outputs["document_processor"] = doc_result
        evidence.append(
            {
                "agent": "document_processor",
                "confidence": doc_result.get("confidence", 1.0),
                "summary": doc_result.get("summary", "Documents processed"),
            }
        )

        # Step 2: Income verification (delegated to income verifier)
        income_result = self._delegate_income_verification(application, doc_result)
        agent_outputs["income_verifier"] = income_result
        evidence.append(
            {
                "agent": "income_verifier",
                "confidence": income_result.get("confidence", 1.0),
                "verified": income_result.get("verified", False),
            }
        )

        # Step 3: Credit scoring (delegated to credit scoring agent)
        credit_result = self._delegate_credit_scoring(
            application, doc_result, income_result
        )
        agent_outputs["credit_scorer"] = credit_result
        evidence.append(
            {
                "agent": "credit_scorer",
                "score": credit_result.get("credit_score", 0),
                "pd": credit_result.get("probability_default", 0.5),
            }
        )

        # Step 4: Fraud & Risk assessment (delegated to fraud agent)
        fraud_result = self._delegate_fraud_detection(
            application, doc_result, income_result
        )
        agent_outputs["fraud_detector"] = fraud_result
        evidence.append(
            {
                "agent": "fraud_detector",
                "fraud_score": fraud_result.get("fraud_score", 0),
                "alerts": fraud_result.get("alerts", []),
            }
        )

        # Step 5: Fairness check (delegated to fairness agent)
        fairness_result = self._delegate_fairness_check(application, credit_result)
        agent_outputs["fairness_checker"] = fairness_result

        # Step 6: Aggregate scores and make preliminary decision
        aggregate_score = self._aggregate_scores(
            credit_result, fraud_result, income_result
        )

        preliminary_decision = self._make_preliminary_decision(aggregate_score)
        self.logger.debug(
            f"Preliminary decision for {application.application_id}: {preliminary_decision}"
        )

        # Step 7: Negotiation loop for borderline cases
        if self._is_borderline(aggregate_score):
            final_score, negotiation_log = self._negotiation_loop(
                application, aggregate_score, agent_outputs
            )
            agent_outputs["negotiation"] = negotiation_log
        else:
            final_score = aggregate_score

        # Step 8: Final decision with human review gating
        decision, confidence = self._make_final_decision(final_score, fairness_result)

        # Step 9: Generate explanation (delegated to explanation agent)
        explanation_result = self._delegate_explanation_generation(
            application, decision, agent_outputs, evidence
        )

        # Construct final decision result
        result = DecisionResult(
            application_id=application.application_id,
            decision=decision,
            confidence=confidence,
            risk_score=1.0 - final_score,
            recommended_terms=self._generate_terms(decision, final_score),
            rationale=explanation_result.get("rationale", ""),
            evidence=evidence,
            agent_outputs=agent_outputs,
            fairness_metrics=fairness_result.get("metrics", {}),
        )

        self.logger.info(
            f"Decision for {application.application_id}: {decision} (confidence: {confidence:.3f})"
        )
        return result

    def _delegate_document_processing(
        self, application: ApplicationData
    ) -> Dict[str, Any]:
        """Delegate to DocumentProcessor agent.

        Derives a real confidence from the documents actually attached to the
        application and extracts an income figure from them when available.
        """
        documents = application.documents or []
        n_docs = len(documents)

        # Confidence grows with the number of supporting documents and is
        # reduced when none are present.
        confidence = min(0.5 + 0.15 * n_docs, 0.98) if n_docs else 0.4

        # Prefer an income value found on the documents (e.g. a pay stub);
        # otherwise fall back to the reported financial info.
        doc_income = None
        for doc in documents:
            if isinstance(doc, dict):
                fields = doc.get("extracted_fields", doc)
                if isinstance(fields, dict) and "income" in fields:
                    doc_income = fields["income"]
                    break
        if doc_income is None:
            doc_income = application.financial_info.get("annual_income", 0)

        employment_verified = any(
            isinstance(d, dict)
            and str(d.get("type", "")).lower() in {"pay_stub", "w2", "employment"}
            for d in documents
        ) or bool(application.applicant_info.get("employment_length"))

        return {
            "confidence": round(confidence, 3),
            "n_documents": n_docs,
            "summary": (
                f"{n_docs} document(s) processed"
                if n_docs
                else "No documents attached; using reported values"
            ),
            "extracted_fields": {
                "income": doc_income,
                "employment_verified": employment_verified,
            },
        }

    def _delegate_income_verification(
        self, application: ApplicationData, doc_result: Dict
    ) -> Dict[str, Any]:
        """Delegate to IncomeVerifier agent.

        Verifies reported income against the document-extracted income and
        computes a debt-to-income ratio used downstream.
        """
        reported = float(application.financial_info.get("annual_income", 0) or 0)
        extracted = float(
            doc_result.get("extracted_fields", {}).get("income", reported) or 0
        )

        if reported <= 0:
            verified, confidence, diff = False, 0.4, 1.0
        else:
            diff = abs(reported - extracted) / reported
            verified = diff <= self.income_tolerance
            # Confidence is highest when reported and extracted income agree.
            confidence = round(max(0.5, 1.0 - diff), 3)

        monthly_debt = float(
            application.financial_info.get("monthly_debt_payments", 0) or 0
        )
        dti = application.financial_info.get("debt_to_income_ratio")
        if dti is None and reported > 0:
            dti = (monthly_debt * 12) / reported
        dti = float(dti) if dti is not None else 0.3

        return {
            "verified": verified,
            "confidence": confidence,
            "reported_income": reported,
            "verified_income": extracted,
            "income_discrepancy": round(diff, 3),
            "debt_to_income_ratio": round(dti, 3),
        }

    def _delegate_credit_scoring(
        self, application: ApplicationData, doc_result: Dict, income_result: Dict
    ) -> Dict[str, Any]:
        """Delegate to the CreditScoring agent.

        Uses a real trained ``CreditScoringAgent`` when one was provided to the
        supervisor; otherwise computes a transparent heuristic probability of
        default from the application's financials and credit history.
        """
        if self.credit_agent is not None:
            try:
                return self.credit_agent.process(application)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(
                    f"Credit agent failed ({exc}); falling back to heuristic"
                )

        return self._heuristic_credit_score(application, income_result)

    def _heuristic_credit_score(
        self, application: ApplicationData, income_result: Dict
    ) -> Dict[str, Any]:
        """Transparent, data-driven fallback credit score (no trained model)."""
        financial = application.financial_info
        credit_hist = application.credit_history or {}

        income = float(financial.get("annual_income", 50000) or 50000)
        loan_amount = float(financial.get("loan_amount", 10000) or 10000)
        loan_to_income = loan_amount / max(income, 1.0)
        dti = float(income_result.get("debt_to_income_ratio", 0.3))
        utilization = float(credit_hist.get("credit_utilization", 0.3))
        delinquencies = float(credit_hist.get("delinquencies_2y", 0))
        inquiries = float(credit_hist.get("inquiries_6m", 1))

        # Log-odds style accumulation of risk factors -> probability of default.
        risk = (
            -1.5
            + 1.8 * dti
            + 1.2 * loan_to_income
            + 1.5 * utilization
            + 0.6 * delinquencies
            + 0.15 * inquiries
        )
        prob_default = 1.0 / (1.0 + np.exp(-risk))
        prob_default = float(min(max(prob_default, 0.01), 0.99))
        credit_score = int(max(300, min(850, 850 - prob_default * 550)))

        return {
            "credit_score": credit_score,
            "probability_default": prob_default,
            "confidence": 0.7,
            "method": "heuristic",
            "features": {
                "debt_to_income": dti,
                "loan_to_income": round(loan_to_income, 3),
                "credit_utilization": utilization,
                "delinquencies_2y": delinquencies,
            },
        }

    def _delegate_fraud_detection(
        self, application: ApplicationData, doc_result: Dict, income_result: Dict
    ) -> Dict[str, Any]:
        """Delegate to FraudDetector agent.

        Uses a real fraud agent when provided; otherwise applies transparent
        rule-based fraud signals derived from the application.
        """
        if self.fraud_agent is not None:
            try:
                return self.fraud_agent.process(application)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(
                    f"Fraud agent failed ({exc}); falling back to rules"
                )

        alerts: List[str] = []
        score = 0.0

        # Signal 1: reported vs document income mismatch.
        discrepancy = float(income_result.get("income_discrepancy", 0.0))
        if discrepancy > self.income_tolerance:
            alerts.append("income_discrepancy")
            score += min(0.4, discrepancy)

        # Signal 2: implausible loan-to-income ratio.
        income = float(application.financial_info.get("annual_income", 1) or 1)
        loan_amount = float(application.financial_info.get("loan_amount", 0) or 0)
        if income > 0 and loan_amount / income > 1.0:
            alerts.append("high_loan_to_income")
            score += 0.2

        # Signal 3: missing supporting documents.
        if doc_result.get("n_documents", 0) == 0:
            alerts.append("no_supporting_documents")
            score += 0.15

        # Signal 4: very short employment paired with a large loan.
        emp_len = float(application.applicant_info.get("employment_length", 5) or 5)
        if emp_len < 1 and loan_amount > 0.5 * income:
            alerts.append("short_employment_large_loan")
            score += 0.15

        score = float(min(score, 1.0))
        confidence = round(0.7 + 0.3 * (1.0 - score), 3)
        return {"fraud_score": score, "alerts": alerts, "confidence": confidence}

    def _delegate_fairness_check(
        self, application: ApplicationData, credit_result: Dict
    ) -> Dict[str, Any]:
        """Delegate to FairnessAgent.

        Individual-application fairness is a pass-through here (true fairness is
        a population-level property handled by ``fairness/mitigation.py``).
        Precomputed population metrics can be injected via ``config['fairness']``.
        """
        metrics = self.config.get(
            "fairness", {"demographic_parity_diff": None, "equalized_odds_diff": None}
        )
        passed = self.config.get("fairness_passed", True)
        return {"passed": passed, "metrics": metrics, "level": "population"}

    def _delegate_explanation_generation(
        self,
        application: ApplicationData,
        decision: str,
        agent_outputs: Dict,
        evidence: List,
    ) -> Dict[str, Any]:
        """Delegate to ExplanationAgent.

        Builds a human-readable rationale from the actual agent outputs.
        """
        credit = agent_outputs.get("credit_scorer", {})
        income = agent_outputs.get("income_verifier", {})
        fraud = agent_outputs.get("fraud_detector", {})

        parts = [f"Decision: {decision.upper()}."]
        if "credit_score" in credit:
            parts.append(
                f"Credit score {credit['credit_score']} "
                f"(PD={credit.get('probability_default', 0):.2%})."
            )
        if income:
            parts.append(
                "Income verified." if income.get("verified") else "Income unverified."
            )
        if fraud:
            alerts = fraud.get("alerts", [])
            parts.append(
                "No fraud signals."
                if not alerts
                else f"Fraud signals: {', '.join(alerts)}."
            )

        return {"rationale": " ".join(parts), "key_factors": evidence}

    def _aggregate_scores(
        self, credit_result: Dict, fraud_result: Dict, income_result: Dict
    ) -> float:
        """Aggregate scores from different agents into single score"""
        # Weighted combination
        credit_weight = 0.5
        fraud_weight = 0.3
        income_weight = 0.2

        credit_score_normalized = 1.0 - credit_result.get("probability_default", 0.5)
        fraud_score_normalized = 1.0 - fraud_result.get("fraud_score", 0.0)
        income_score = 1.0 if income_result.get("verified", False) else 0.5

        aggregate = (
            credit_score_normalized * credit_weight
            + fraud_score_normalized * fraud_weight
            + income_score * income_weight
        )

        return aggregate

    def _make_preliminary_decision(self, score: float) -> str:
        """Make preliminary decision based on aggregate score"""
        if score >= self.decision_threshold_approve:
            return "approve"
        elif score <= self.decision_threshold_deny:
            return "deny"
        else:
            return "review"

    def _is_borderline(self, score: float) -> bool:
        """
        Check if case is borderline and needs negotiation.

        """
        return self.decision_threshold_deny < score < self.decision_threshold_approve

    def _negotiation_loop(
        self, application: ApplicationData, initial_score: float, agent_outputs: Dict
    ) -> tuple:
        """Negotiation loop for borderline cases.

        Borderline applications are re-examined: strong corroborating evidence
        (verified income, no fraud alerts, low default probability) nudges the
        score up, while contradicting evidence nudges it down. The loop stops as
        soon as the case leaves the borderline band or the score stabilises.
        """
        score = float(initial_score)
        log = []

        income = agent_outputs.get("income_verifier", {})
        fraud = agent_outputs.get("fraud_detector", {})
        credit = agent_outputs.get("credit_scorer", {})

        for round_num in range(self.max_negotiation_rounds):
            adjustment = 0.0
            reasons = []

            if income.get("verified"):
                adjustment += 0.03
                reasons.append("income_verified")
            else:
                adjustment -= 0.03
                reasons.append("income_unverified")

            fraud_score = float(fraud.get("fraud_score", 0.0))
            if fraud_score > 0.2:
                adjustment -= 0.05
                reasons.append("fraud_risk")
            elif not fraud.get("alerts"):
                adjustment += 0.02
                reasons.append("no_fraud")

            pd = float(credit.get("probability_default", 0.5))
            if pd < 0.1:
                adjustment += 0.03
                reasons.append("low_pd")
            elif pd > 0.3:
                adjustment -= 0.03
                reasons.append("high_pd")

            new_score = float(min(max(score + adjustment, 0.0), 1.0))
            log.append(
                {
                    "round": round_num + 1,
                    "score_before": round(score, 4),
                    "adjustment": round(adjustment, 4),
                    "score_after": round(new_score, 4),
                    "reasons": reasons,
                }
            )

            score = new_score

            # Stop when out of the borderline band or the score has stabilised.
            if not self._is_borderline(score) or abs(adjustment) < 1e-6:
                break

        return score, log

    def _make_final_decision(self, score: float, fairness_result: Dict) -> tuple:
        """Make final decision with human review gating"""
        confidence = abs(score - 0.5) * 2  # Convert to 0-1 confidence

        # Check if human review is needed
        if self._is_borderline(score) or not fairness_result.get("passed", True):
            decision = "review"
            confidence = min(confidence, 0.6)  # Cap confidence for review cases
        elif score >= self.decision_threshold_approve:
            decision = "approve"
        elif score <= self.decision_threshold_deny:
            decision = "deny"
        else:
            decision = "review"

        return decision, confidence

    def _generate_terms(self, decision: str, score: float) -> Optional[Dict[str, Any]]:
        """Generate recommended loan terms for approved applications"""
        if decision != "approve":
            return None

        # Risk-based pricing
        base_rate = 0.05
        risk_premium = (1.0 - score) * 0.10

        return {
            "interest_rate": base_rate + risk_premium,
            "term_months": 36,
            "max_amount": 50000 * score,
        }
