"""
Generate publication-ready figures from experimental results.
"""

import logging as _lg

# --- keep run output readable: suppress benign third-party noise (auto-added) ---
import os as _os
import warnings as _w

_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
for _m in (
    r".*does not have valid feature names.*",
    r".*tight_layout.*",
    r".*Gym has been unmaintained.*",
    r".*not wrapped with a ``Monitor``.*",
):
    _w.filterwarnings("ignore", message=_m)
_w.filterwarnings("ignore", category=DeprecationWarning)
_w.filterwarnings("ignore", category=FutureWarning)
try:
    from sklearn.exceptions import ConvergenceWarning as _CW

    _w.filterwarnings("ignore", category=_CW)
except Exception:
    pass
for _n in (
    "matplotlib",
    "PIL",
    "urllib3",
    "yfinance",
    "tensorflow",
    "absl",
    "gym",
    "gymnasium",
    "shap",
    "numba",
    "h5py",
):
    _lg.getLogger(_n).setLevel(_lg.ERROR)


def _silence_tqdm():
    try:
        import tqdm.std as _tstd

        _orig = _tstd.tqdm.__init__

        def _init(self, *a, **k):
            k["disable"] = True
            _orig(self, *a, **k)

        _tstd.tqdm.__init__ = _init
        try:
            from tqdm import auto as _ta

            if _ta.tqdm is not _tstd.tqdm:
                _o2 = _ta.tqdm.__init__

                def _init2(self, *a, **k):
                    k["disable"] = True
                    _o2(self, *a, **k)

                _ta.tqdm.__init__ = _init2
        except Exception:
            pass
    except Exception:
        pass


_silence_tqdm()
# --- end output cleanup ---

# --- matplotlib/seaborn compatibility shim (auto-added) ---
# Old seaborn (<0.12) calls matplotlib.cm.register_cmap / get_cmap, removed in
# matplotlib 3.9+. Restore them so seaborn imports and runs on modern matplotlib
# without requiring a seaborn upgrade. No-op on already-compatible versions.
try:
    import matplotlib as _mpl_compat
    import matplotlib.cm as _mpl_cm_compat

    if not hasattr(_mpl_cm_compat, "register_cmap"):

        def _compat_register_cmap(name=None, cmap=None, **_kw):
            if cmap is None and not isinstance(name, str):
                cmap, name = name, getattr(name, "name", None)
            _mpl_compat.colormaps.register(cmap, name=name, force=True)

        _mpl_cm_compat.register_cmap = _compat_register_cmap

    if not hasattr(_mpl_cm_compat, "get_cmap"):

        def _compat_get_cmap(name=None, lut=None):
            _c = (
                _mpl_compat.colormaps[name]
                if isinstance(name, str)
                else (
                    name
                    if name is not None
                    else _mpl_compat.colormaps[_mpl_compat.rcParams["image.cmap"]]
                )
            )
            return _c.resampled(lut) if lut else _c

        _mpl_cm_compat.get_cmap = _compat_get_cmap
except Exception:
    pass
# --- end compatibility shim ---

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Non-interactive backend
import argparse
import json
import logging
from pathlib import Path

import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


def load_results(results_dir: str = "results") -> dict:
    """Load experimental results"""
    results_path = Path(results_dir) / "metrics_summary.json"
    with open(results_path, "r") as f:
        return json.load(f)


def generate_system_architecture(output_dir: str = "figures"):
    """Generate system architecture diagram using Graphviz"""
    try:
        from graphviz import Digraph

        dot = Digraph(comment="Multi-Agent Credit Underwriting System")
        dot.attr(rankdir="TB", size="10,8")
        dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue")

        # Supervisor
        dot.node("supervisor", "Loan Supervisor\\nOrchestrator", fillcolor="gold")

        # Agent layer
        dot.node("doc_proc", "Document\\nProcessor", fillcolor="lightgreen")
        dot.node("income_ver", "Income\\nVerifier", fillcolor="lightgreen")
        dot.node("credit_score", "Credit Scoring\\nAgent", fillcolor="lightgreen")
        dot.node("fraud_det", "Fraud Detection\\nAgent", fillcolor="lightgreen")
        dot.node("fairness", "Fairness\\nAgent", fillcolor="lightcoral")
        dot.node("explain", "Explanation &\\nAudit Agent", fillcolor="lightyellow")

        # Data sources
        dot.node("input", "Loan Application", shape="parallelogram", fillcolor="white")
        dot.node(
            "output", "Decision +\\nRationale", shape="parallelogram", fillcolor="white"
        )

        # Edges
        dot.edge("input", "supervisor")
        dot.edge("supervisor", "doc_proc", label="parse docs")
        dot.edge("supervisor", "income_ver", label="verify income")
        dot.edge("supervisor", "credit_score", label="score")
        dot.edge("supervisor", "fraud_det", label="fraud check")
        dot.edge("supervisor", "fairness", label="fairness check")
        dot.edge("supervisor", "explain", label="explain")

        # Return paths
        dot.edge("doc_proc", "supervisor", style="dashed")
        dot.edge("income_ver", "supervisor", style="dashed")
        dot.edge("credit_score", "supervisor", style="dashed")
        dot.edge("fraud_det", "supervisor", style="dashed")
        dot.edge("fairness", "supervisor", style="dashed", color="red")
        dot.edge("explain", "output", style="dashed")
        dot.edge("supervisor", "output")

        output_path = Path(output_dir) / "system_architecture"
        dot.render(output_path, format="svg", cleanup=True)
        logger.info(f"Generated {output_path}.svg")

    except (ImportError, Exception) as e:
        logger.warning(f"Could not generate architecture diagram: {e}")
        logger.warning(
            "Graphviz not available or 'dot' executable not found, skipping architecture diagram"
        )


def generate_roc_pr_curves(results: dict, output_dir: str = "figures"):
    """Generate ROC and PR curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    ax = axes[0]
    models_to_plot = ["dummy", "logistic", "decision_tree", "lightgbm"]
    colors = ["gray", "blue", "green", "red"]

    for model_name, color in zip(models_to_plot, colors):
        if model_name in results["baselines"]:
            metrics = results["baselines"][model_name]
            if "roc_curve" in metrics:
                fpr = metrics["roc_curve"]["fpr"]
                tpr = metrics["roc_curve"]["tpr"]
                auc = metrics["auc"]
                ax.plot(
                    fpr,
                    tpr,
                    label=f"{model_name.title()} (AUC={auc:.3f})",
                    color=color,
                    linewidth=2,
                )

    ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Default Prediction", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # PR Curve
    ax = axes[1]
    for model_name, color in zip(models_to_plot, colors):
        if model_name in results["baselines"]:
            metrics = results["baselines"][model_name]
            if "pr_curve" in metrics:
                precision = metrics["pr_curve"]["precision"]
                recall = metrics["pr_curve"]["recall"]
                ap = metrics["avg_precision"]
                ax.plot(
                    recall,
                    precision,
                    label=f"{model_name.title()} (AP={ap:.3f})",
                    color=color,
                    linewidth=2,
                )

    # Baseline (default rate)
    default_rate = results["metadata"]["default_rate_test"]
    ax.axhline(
        y=default_rate,
        color="k",
        linestyle="--",
        label=f"Random (AP={default_rate:.3f})",
        linewidth=1,
    )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "roc_pr_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated {output_path}")


def generate_fairness_tradeoffs(results: dict, output_dir: str = "figures"):
    """Generate fairness-performance tradeoff plot"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect data points
    methods = []
    aucs = []
    dp_diffs = []

    # Baseline (no mitigation)
    if "baseline" in results["fairness"]:
        methods.append("Baseline\\n(No Mitigation)")
        aucs.append(results["fairness"]["baseline"]["model_performance"]["auc"])
        dp_diff = results["fairness"]["baseline"]["metrics"]["sex"].get(
            "demographic_parity_diff", 0
        )
        dp_diffs.append(dp_diff)

    # Reweighing
    if "reweighing" in results["fairness"]:
        methods.append("Reweighing\\n(Pre-processing)")
        aucs.append(results["fairness"]["reweighing"]["model_performance"]["auc"])
        dp_diff = results["fairness"]["reweighing"]["metrics"]["sex"].get(
            "demographic_parity_diff", 0
        )
        dp_diffs.append(dp_diff)

    # Threshold optimization
    if "threshold_opt" in results["fairness"]:
        methods.append("Threshold Opt\\n(Post-processing)")
        aucs.append(results["fairness"]["threshold_opt"]["model_performance"]["auc"])
        dp_diff = results["fairness"]["threshold_opt"]["metrics"]["sex"].get(
            "demographic_parity_diff", 0
        )
        dp_diffs.append(dp_diff)

    # Scatter plot
    colors = ["red", "orange", "green"]
    for i, (method, auc, dp) in enumerate(zip(methods, aucs, dp_diffs)):
        ax.scatter(
            dp, auc, s=200, c=colors[i], alpha=0.7, edgecolors="black", linewidth=2
        )
        ax.annotate(
            method,
            (dp, auc),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc=colors[i], alpha=0.3),
        )

    # Add fairness threshold line
    threshold = 0.05
    ax.axvline(
        x=threshold,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Fairness Threshold ({threshold})",
    )

    ax.set_xlabel("Demographic Parity Difference (Sex)", fontsize=12, fontweight="bold")
    ax.set_ylabel("AUC (Model Performance)", fontsize=12, fontweight="bold")
    ax.set_title("Fairness-Performance Tradeoffs", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # Set reasonable limits
    ax.set_xlim(-0.01, max(dp_diffs) * 1.2)
    ax.set_ylim(min(aucs) * 0.95, max(aucs) * 1.02)

    plt.tight_layout()
    output_path = Path(output_dir) / "fairness_tradeoffs.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated {output_path}")


def generate_explainability_example(output_dir: str = "figures"):
    """Generate example explainability visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    # Title
    fig.text(
        0.5,
        0.95,
        "Example: Loan Application Decision with Explanation",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )

    # Application info box
    app_text = """
    APPLICATION ID: APP_42_000123
    
    Applicant Information:
    • Age: 35 years
    • Employment: 5 years (verified)
    • Annual Income: $75,000
    • Requested Loan: $15,000
    • Purpose: Debt Consolidation
    """

    fig.text(
        0.05,
        0.70,
        app_text,
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    # Credit factors box
    credit_text = """
    Credit Profile:
    • Credit Score: 680 (Good)
    • Credit Utilization: 35%
    • Delinquencies (2y): 0
    • Credit History: 8 years
    • Open Accounts: 6
    """

    fig.text(
        0.55,
        0.70,
        credit_text,
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    )

    # Feature importance bars
    features = [
        "Credit Score",
        "Income",
        "Debt-to-Income",
        "Credit Util.",
        "Employment",
    ]
    importance = [0.35, 0.25, 0.20, 0.15, 0.05]
    colors_bar = ["green" if imp > 0.2 else "orange" for imp in importance]

    ax_inset = fig.add_axes([0.1, 0.35, 0.35, 0.25])
    ax_inset.barh(features, importance, color=colors_bar, alpha=0.7)
    ax_inset.set_xlabel("Feature Importance", fontsize=10)
    ax_inset.set_title("Key Decision Factors", fontsize=11, fontweight="bold")
    ax_inset.grid(axis="x", alpha=0.3)

    # Agent scores box
    agent_text = """
    Agent Assessments:
    
    ✓ Document Processor: 95% confidence
    ✓ Income Verifier: Verified
    ✓ Credit Scoring: PD = 15% (Low Risk)
    ✓ Fraud Detection: 2% risk (Clear)
    ✓ Fairness Check: Passed
    """

    fig.text(
        0.55,
        0.35,
        agent_text,
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.3),
    )

    # Decision box
    decision_text = """
    FINAL DECISION: APPROVED
    
    Confidence: 85%
    Risk Score: 0.15
    
    Recommended Terms:
    • Interest Rate: 6.5% APR
    • Term: 36 months
    • Monthly Payment: $458
    
    Rationale:
    Strong credit history, verified income, low default
    probability. Debt consolidation purpose aligns with
    risk profile. No fairness concerns identified.
    """

    fig.text(
        0.05,
        0.02,
        decision_text,
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    # Audit trail note
    audit_text = """
    Audit Trail: #AUD_20260101_123456
    All evidence logged for regulatory review
    """

    fig.text(
        0.55,
        0.10,
        audit_text,
        fontsize=8,
        family="monospace",
        style="italic",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3),
    )

    plt.tight_layout()
    output_path = Path(output_dir) / "explainability_example.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated {output_path}")


def generate_orchestration_sequence(output_dir: str = "figures"):
    """Generate sequence diagram for agent orchestration"""
    try:
        from graphviz import Digraph

        dot = Digraph(comment="Orchestration Sequence")
        dot.attr(rankdir="LR", size="12,6")
        dot.attr("node", shape="box", style="filled", fillcolor="lightblue")

        # Time steps
        dot.node(
            "t0", "Application\\nReceived", shape="ellipse", fillcolor="lightgreen"
        )
        dot.node("t1", "Parse\\nDocuments")
        dot.node("t2", "Verify\\nIncome")
        dot.node("t3", "Score\\nCredit")
        dot.node("t4", "Check\\nFraud")
        dot.node("t5", "Fairness\\nReview", fillcolor="lightcoral")
        dot.node("t6", "Borderline?\\nNegotiate", shape="diamond", fillcolor="yellow")
        dot.node("t7", "Generate\\nExplanation")
        dot.node("t8", "Final\\nDecision", shape="ellipse", fillcolor="lightgreen")

        # Sequential flow
        dot.edge("t0", "t1", label="1")
        dot.edge("t1", "t2", label="2")
        dot.edge("t2", "t3", label="3")
        dot.edge("t3", "t4", label="4")
        dot.edge("t4", "t5", label="5")
        dot.edge("t5", "t6", label="6")
        dot.edge("t6", "t7", label="No")
        dot.edge("t6", "t3", label="Yes\\n(revise)", style="dashed", color="red")
        dot.edge("t7", "t8", label="7")

        output_path = Path(output_dir) / "orchestration_sequence"
        dot.render(output_path, format="svg", cleanup=True)
        logger.info(f"Generated {output_path}.svg")

    except (ImportError, Exception) as e:
        logger.warning(f"Could not generate sequence diagram: {e}")
        logger.warning(
            "Graphviz not available or 'dot' executable not found, skipping sequence diagram"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures from experimental results"
    )
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument(
        "--output-dir", default="figures", help="Output directory for figures"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Skip expensive visualizations"
    )
    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Generating publication-ready figures...")

    # Load results
    try:
        results = load_results(args.results_dir)
    except FileNotFoundError:
        logger.error(f"Results not found in {args.results_dir}. Run experiments first.")
        return

    # Generate figures
    logger.info("1/5: System architecture diagram...")
    generate_system_architecture(args.output_dir)

    logger.info("2/5: Orchestration sequence diagram...")
    generate_orchestration_sequence(args.output_dir)

    logger.info("3/5: ROC and PR curves...")
    generate_roc_pr_curves(results, args.output_dir)

    logger.info("4/5: Fairness tradeoffs...")
    generate_fairness_tradeoffs(results, args.output_dir)

    logger.info("5/5: Explainability example...")
    generate_explainability_example(args.output_dir)

    logger.info(f"All figures generated in {args.output_dir}/")

    # List generated files
    figures = list(Path(args.output_dir).glob("*"))
    print("\nGenerated figures:")
    for fig in sorted(figures):
        print(f"  • {fig.name}")


if __name__ == "__main__":
    main()
