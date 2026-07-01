"""
Adverse Action Notice Generator - Regulation B Compliance
Automatically generates adverse action notices when credit is denied or offered on less favorable terms.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AdverseActionNoticeGenerator:
    """
    Generates adverse action notices compliant with Regulation B (Equal Credit Opportunity Act).

    Required by 12 CFR § 1002.9 when:
    - Credit application is denied
    - Credit terms are less favorable than requested
    - Credit account is closed
    """

    def __init__(self, creditor_info: Optional[Dict[str, str]] = None):
        self.creditor_info = creditor_info or {
            "name": "Example Financial Institution",
            "address": "123 Main Street, Suite 100",
            "city_state_zip": "New York, NY 10001",
            "phone": "1-800-555-0123",
            "website": "www.example-financial.com",
        }

    def generate_notice(
        self,
        application_id: str,
        applicant_name: str,
        applicant_address: Dict[str, str],
        decision: str,
        primary_reasons: List[str],
        credit_score: Optional[int] = None,
        credit_bureau_info: Optional[Dict[str, str]] = None,
        decision_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate adverse action notice.

        Args:
            application_id: Unique application identifier
            applicant_name: Full name of applicant
            applicant_address: Dictionary with street, city, state, zip
            decision: "deny", "counteroffer", or "review"
            primary_reasons: List of specific reasons for adverse action (max 4 most important)
            credit_score: Credit score used in decision (if applicable)
            credit_bureau_info: Information about credit bureau(s) used
            decision_date: Date of decision (defaults to now)

        Returns:
            Dictionary containing the notice content and metadata
        """
        if decision_date is None:
            decision_date = datetime.now()

        # Validate reasons
        valid_reasons = self._get_valid_reason_codes()
        for reason in primary_reasons:
            if reason not in valid_reasons:
                logger.warning(f"Reason code '{reason}' not in standard reason list")

        # Build notice content
        notice = {
            "notice_id": f"ADV-{application_id}-{decision_date.strftime('%Y%m%d')}",
            "application_id": application_id,
            "notice_date": decision_date.strftime("%B %d, %Y"),
            "notice_type": "Adverse Action Notice",
            "applicant": {"name": applicant_name, "address": applicant_address},
            "creditor": self.creditor_info,
            "decision": self._format_decision(decision),
            "reasons": self._format_reasons(primary_reasons),
            "credit_score_disclosure": self._format_credit_score_disclosure(
                credit_score, credit_bureau_info
            ),
            "applicant_rights": self._get_applicant_rights(),
            "content": self._generate_notice_text(
                applicant_name, decision, primary_reasons, credit_score, decision_date
            ),
        }

        logger.info(
            f"Generated adverse action notice {notice['notice_id']} for {applicant_name}"
        )
        return notice

    def _format_decision(self, decision: str) -> Dict[str, str]:
        """Format decision section"""
        decision_map = {
            "deny": {
                "action": "DENIED",
                "description": "Your application for credit has been denied.",
            },
            "counteroffer": {
                "action": "COUNTEROFFER",
                "description": "Credit has been extended with terms different from those you requested.",
            },
            "review": {
                "action": "INCOMPLETE",
                "description": "Your application is incomplete. Additional information is required.",
            },
        }
        return decision_map.get(decision, decision_map["deny"])

    def _format_reasons(self, reasons: List[str]) -> List[Dict[str, str]]:
        """Format and expand reason codes"""
        reason_descriptions = self._get_reason_descriptions()

        formatted_reasons = []
        for i, reason in enumerate(
            reasons[:4], 1
        ):  # Regulation B requires top 4 reasons
            formatted_reasons.append(
                {
                    "rank": i,
                    "code": reason,
                    "description": reason_descriptions.get(reason, reason),
                }
            )

        return formatted_reasons

    def _format_credit_score_disclosure(
        self, credit_score: Optional[int], credit_bureau_info: Optional[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Format credit score disclosure (required by FACTA)"""
        if credit_score is None:
            return None

        if credit_bureau_info is None:
            credit_bureau_info = {
                "name": "Example Credit Bureau",
                "address": "P.O. Box 1000, Atlanta, GA 30301",
                "phone": "1-800-555-1000",
                "website": "www.examplecreditbureau.com",
            }

        return {
            "score": credit_score,
            "score_range": "300-850",
            "date_pulled": datetime.now().strftime("%B %d, %Y"),
            "key_factors": [
                "Number of accounts with delinquency",
                "Amount owed on accounts",
                "Length of credit history",
                "Number of recent credit inquiries",
            ],
            "credit_bureau": credit_bureau_info,
            "notice": "This credit score was used in evaluating your application.",
        }

    def _get_applicant_rights(self) -> Dict[str, Any]:
        """Get applicant rights disclosure"""
        return {
            "ecoa_rights": [
                "You have the right to a written statement of specific reasons for the denial within 60 days.",
                "The federal Equal Credit Opportunity Act prohibits creditors from discriminating against credit applicants on the basis of race, color, religion, national origin, sex, marital status, age, or because you receive public assistance.",
                "If you believe you have been discriminated against, you should send a complaint to:",
            ],
            "complaint_agencies": [
                {
                    "agency": "Consumer Financial Protection Bureau",
                    "address": "P.O. Box 4503, Iowa City, IA 52244",
                    "website": "www.consumerfinance.gov",
                },
                {
                    "agency": "Federal Trade Commission",
                    "address": "Consumer Response Center, 600 Pennsylvania Avenue NW, Washington, DC 20580",
                    "website": "www.ftc.gov",
                },
            ],
            "fcra_rights": [
                "You have the right to obtain a free copy of your credit report from the credit bureau(s) listed above.",
                "You have the right to dispute inaccurate information in your credit report.",
                "You may contact the credit bureau directly to request your report.",
            ],
        }

    def _generate_notice_text(
        self,
        applicant_name: str,
        decision: str,
        reasons: List[str],
        credit_score: Optional[int],
        decision_date: datetime,
    ) -> str:
        """Generate complete notice text"""
        reason_descriptions = self._get_reason_descriptions()

        text = f"""
ADVERSE ACTION NOTICE
Date: {decision_date.strftime("%B %d, %Y")}

Dear {applicant_name},

Thank you for your recent credit application. We regret to inform you that your application has been {self._format_decision(decision)['action']}.

{self._format_decision(decision)['description']}

PRINCIPAL REASON(S) FOR THIS DECISION:

"""
        for i, reason in enumerate(reasons[:4], 1):
            text += f"{i}. {reason_descriptions.get(reason, reason)}\n"

        if credit_score:
            text += f"""
CREDIT SCORE DISCLOSURE:

Your credit score: {credit_score} (Range: 300-850)
Date: {datetime.now().strftime("%B %d, %Y")}

The credit score used in evaluating your application is from a credit report obtained from a consumer reporting agency. Key factors that adversely affected your credit score are listed in the disclosure provided with this notice.

"""

        text += """
YOUR RIGHTS UNDER THE EQUAL CREDIT OPPORTUNITY ACT:

The federal Equal Credit Opportunity Act prohibits creditors from discriminating against credit applicants on the basis of race, color, religion, national origin, sex, marital status, age (provided the applicant has the capacity to enter into a binding contract); because all or part of the applicant's income derives from any public assistance program; or because the applicant has in good faith exercised any right under the Consumer Credit Protection Act.

If you have any questions regarding this decision, please contact us at the phone number or address listed below.

Sincerely,
"""
        text += f"{self.creditor_info['name']}\n"
        text += f"{self.creditor_info['address']}\n"
        text += f"{self.creditor_info['city_state_zip']}\n"
        text += f"Phone: {self.creditor_info['phone']}\n"

        return text

    def _get_valid_reason_codes(self) -> List[str]:
        """Get list of valid adverse action reason codes"""
        return [
            "CREDIT_SCORE_TOO_LOW",
            "INSUFFICIENT_CREDIT_HISTORY",
            "DELINQUENT_CREDIT_OBLIGATIONS",
            "TOO_MANY_RECENT_INQUIRIES",
            "HIGH_DEBT_TO_INCOME_RATIO",
            "BANKRUPTCY_HISTORY",
            "FORECLOSURE_HISTORY",
            "INSUFFICIENT_INCOME",
            "UNSTABLE_EMPLOYMENT",
            "HIGH_CREDIT_UTILIZATION",
            "TOO_MANY_OPEN_ACCOUNTS",
            "RECENT_DELINQUENCY",
            "COLLECTION_ACCOUNTS",
            "CHARGE_OFFS",
            "INSUFFICIENT_COLLATERAL",
            "UNABLE_TO_VERIFY_INCOME",
            "UNABLE_TO_VERIFY_EMPLOYMENT",
            "INCOMPLETE_APPLICATION",
            "TOO_SHORT_CREDIT_HISTORY",
            "TOO_FEW_CREDIT_ACCOUNTS",
        ]

    def _get_reason_descriptions(self) -> Dict[str, str]:
        """Get human-readable descriptions for reason codes"""
        return {
            "CREDIT_SCORE_TOO_LOW": "Credit score below minimum requirement",
            "INSUFFICIENT_CREDIT_HISTORY": "Insufficient credit history to assess creditworthiness",
            "DELINQUENT_CREDIT_OBLIGATIONS": "Delinquent past or present credit obligations with others",
            "TOO_MANY_RECENT_INQUIRIES": "Too many recent inquiries for credit",
            "HIGH_DEBT_TO_INCOME_RATIO": "Debt-to-income ratio exceeds acceptable threshold",
            "BANKRUPTCY_HISTORY": "Bankruptcy in credit history",
            "FORECLOSURE_HISTORY": "Foreclosure or repossession in credit history",
            "INSUFFICIENT_INCOME": "Income insufficient for amount requested",
            "UNSTABLE_EMPLOYMENT": "Length or stability of employment",
            "HIGH_CREDIT_UTILIZATION": "Proportion of credit used compared to credit available",
            "TOO_MANY_OPEN_ACCOUNTS": "Number of established credit accounts",
            "RECENT_DELINQUENCY": "Recent delinquencies on credit accounts",
            "COLLECTION_ACCOUNTS": "Accounts in collection",
            "CHARGE_OFFS": "Charge-offs on previous credit accounts",
            "INSUFFICIENT_COLLATERAL": "Insufficient collateral for secured loan",
            "UNABLE_TO_VERIFY_INCOME": "Unable to verify income",
            "UNABLE_TO_VERIFY_EMPLOYMENT": "Unable to verify employment",
            "INCOMPLETE_APPLICATION": "Application incomplete or missing required information",
            "TOO_SHORT_CREDIT_HISTORY": "Length of credit history too short",
            "TOO_FEW_CREDIT_ACCOUNTS": "Limited number of credit accounts",
        }

    def generate_sample_notices(self) -> List[Dict[str, Any]]:
        """Generate sample adverse action notices for demonstration"""
        samples = []

        # Sample 1: Credit score too low
        samples.append(
            self.generate_notice(
                application_id="APP_42_000001",
                applicant_name="John Smith",
                applicant_address={
                    "street": "456 Oak Avenue",
                    "city": "Springfield",
                    "state": "IL",
                    "zip": "62701",
                },
                decision="deny",
                primary_reasons=[
                    "CREDIT_SCORE_TOO_LOW",
                    "DELINQUENT_CREDIT_OBLIGATIONS",
                    "HIGH_DEBT_TO_INCOME_RATIO",
                    "TOO_MANY_RECENT_INQUIRIES",
                ],
                credit_score=580,
                credit_bureau_info={
                    "name": "Experian",
                    "address": "P.O. Box 9554, Allen, TX 75013",
                    "phone": "1-888-397-3742",
                    "website": "www.experian.com",
                },
            )
        )

        # Sample 2: Insufficient credit history
        samples.append(
            self.generate_notice(
                application_id="APP_42_000002",
                applicant_name="Maria Garcia",
                applicant_address={
                    "street": "789 Pine Street, Apt 3B",
                    "city": "Los Angeles",
                    "state": "CA",
                    "zip": "90012",
                },
                decision="deny",
                primary_reasons=[
                    "INSUFFICIENT_CREDIT_HISTORY",
                    "TOO_SHORT_CREDIT_HISTORY",
                    "TOO_FEW_CREDIT_ACCOUNTS",
                ],
                credit_score=None,  # No score available due to thin file
            )
        )

        # Sample 3: Counteroffer with different terms
        samples.append(
            self.generate_notice(
                application_id="APP_42_000003",
                applicant_name="Robert Johnson",
                applicant_address={
                    "street": "321 Elm Drive",
                    "city": "Austin",
                    "state": "TX",
                    "zip": "73301",
                },
                decision="counteroffer",
                primary_reasons=[
                    "HIGH_CREDIT_UTILIZATION",
                    "RECENT_DELINQUENCY",
                    "HIGH_DEBT_TO_INCOME_RATIO",
                ],
                credit_score=650,
                credit_bureau_info={
                    "name": "TransUnion",
                    "address": "P.O. Box 1000, Chester, PA 19016",
                    "phone": "1-800-916-8800",
                    "website": "www.transunion.com",
                },
            )
        )

        return samples


def generate_notice_pdf(notice: Dict[str, Any], output_path: str) -> str:
    """
    Generate PDF version of adverse action notice.

    Args:
        notice: Notice dictionary from generate_notice()
        output_path: Path to save PDF

    Returns:
        Path to generated PDF
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=colors.HexColor("#1a1a1a"),
            spaceAfter=30,
            alignment=1,  # Center
        )
        story.append(Paragraph("ADVERSE ACTION NOTICE", title_style))
        story.append(Spacer(1, 0.2 * inch))

        # Notice content
        content = notice["content"].replace("\n", "<br/>")
        story.append(Paragraph(content, styles["Normal"]))

        # Build PDF
        doc.build(story)
        logger.info(f"Generated PDF adverse action notice: {output_path}")
        return output_path

    except ImportError:
        logger.warning("reportlab not installed. PDF generation skipped.")
        return output_path


if __name__ == "__main__":
    # Demo usage
    generator = AdverseActionNoticeGenerator()

    # Generate sample notices
    samples = generator.generate_sample_notices()

    print("=" * 80)
    print("SAMPLE ADVERSE ACTION NOTICES")
    print("=" * 80)

    for i, notice in enumerate(samples, 1):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i}: {notice['decision']['action']}")
        print(f"{'='*80}")
        print(notice["content"])
        print(f"\nNotice ID: {notice['notice_id']}")
        print(f"Application ID: {notice['application_id']}")
        print(f"Reasons: {[r['description'] for r in notice['reasons']]}")
