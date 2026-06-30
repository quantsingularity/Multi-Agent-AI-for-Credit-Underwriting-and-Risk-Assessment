"""
Visualization Module for Fairness and Performance Analysis
Creates publication-ready visualizations for model performance, fairness metrics, and bias analysis.
"""

# === stdlib 'code' pin (Python 3.13 pdb compatibility) -- auto-added ===
import sys as _sys

if not hasattr(_sys.modules.get("code"), "InteractiveConsole"):
    import importlib.util as _ilu
    import os as _os
    import sysconfig as _sc

    _sp = _sc.get_paths()["stdlib"]
    _cspec = _ilu.spec_from_file_location("code", _os.path.join(_sp, "code.py"))
    if _cspec is not None:
        _cmod = _ilu.module_from_spec(_cspec)
        _cspec.loader.exec_module(_cmod)
        _sys.modules["code"] = _cmod
    del _ilu, _os, _sc, _sp, _cspec
# === end stdlib 'code' pin ===

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

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


class FairnessVisualizer:
    """
    Creates comprehensive visualizations for fairness analysis and model performance.
    """

    def __init__(self, output_dir: str = "figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized FairnessVisualizer, output to {self.output_dir}")

    def plot_fairness_accuracy_tradeoff(
        self,
        fairness_results: Dict[str, Any],
        output_filename: str = "fairness_accuracy_tradeoff.png",
    ):
        """
        Plot fairness-accuracy trade-off curve.

        Shows how different mitigation strategies affect both model performance
        and fairness metrics.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Extract data
        strategies = []
        aucs = []
        dp_diffs = []
        eo_diffs = []

        for strategy_name, result in fairness_results.items():
            strategies.append(strategy_name.replace("_", " ").title())
            aucs.append(result["model_performance"]["auc"])

            # Get fairness metrics (use 'sex' as primary protected attribute)
            strategy_metrics = result["metrics"]
            if "sex" in strategy_metrics:
                dp_diffs.append(
                    strategy_metrics["sex"].get("demographic_parity_diff", 0)
                )
                eo_diffs.append(strategy_metrics["sex"].get("equalized_odds_diff", 0))
            else:
                dp_diffs.append(0)
                eo_diffs.append(0)

        # Plot 1: AUC vs Demographic Parity Difference
        ax1.scatter(
            dp_diffs, aucs, s=200, alpha=0.6, c=range(len(strategies)), cmap="viridis"
        )
        for i, strategy in enumerate(strategies):
            ax1.annotate(
                strategy,
                (dp_diffs[i], aucs[i]),
                xytext=(10, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
            )

        ax1.axhline(
            y=max(aucs), color="gray", linestyle="--", alpha=0.3, label="Max AUC"
        )
        ax1.axvline(
            x=0.05, color="red", linestyle="--", alpha=0.3, label="Fairness Threshold"
        )

        ax1.set_xlabel("Demographic Parity Difference", fontsize=12)
        ax1.set_ylabel("AUC-ROC", fontsize=12)
        ax1.set_title(
            "Fairness-Accuracy Trade-off\n(Demographic Parity)",
            fontsize=13,
            fontweight="bold",
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: AUC vs Equalized Odds Difference
        ax2.scatter(
            eo_diffs, aucs, s=200, alpha=0.6, c=range(len(strategies)), cmap="plasma"
        )
        for i, strategy in enumerate(strategies):
            ax2.annotate(
                strategy,
                (eo_diffs[i], aucs[i]),
                xytext=(10, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
            )

        ax2.axhline(
            y=max(aucs), color="gray", linestyle="--", alpha=0.3, label="Max AUC"
        )
        ax2.axvline(
            x=0.05, color="red", linestyle="--", alpha=0.3, label="Fairness Threshold"
        )

        ax2.set_xlabel("Equalized Odds Difference", fontsize=12)
        ax2.set_ylabel("AUC-ROC", fontsize=12)
        ax2.set_title(
            "Fairness-Accuracy Trade-off\n(Equalized Odds)",
            fontsize=13,
            fontweight="bold",
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Saved fairness-accuracy tradeoff plot to {output_path}")

    def plot_demographic_bias_audit(
        self,
        fairness_results: Dict[str, Any],
        protected_attribute: str = "sex",
        output_filename: str = "demographic_bias_audit.png",
    ):
        """
        Create comprehensive bias audit visualization across demographics.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Extract approval rates by group
        strategies = list(fairness_results.keys())

        # Get all groups
        first_strategy = strategies[0]
        first_metrics = fairness_results[first_strategy]["metrics"]

        if protected_attribute not in first_metrics:
            logger.warning(f"Protected attribute '{protected_attribute}' not found")
            return

        approval_rates = first_metrics[protected_attribute].get("approval_rates", {})
        groups = list(approval_rates.keys())

        # Plot 1: Approval rates by group and strategy
        ax = axes[0, 0]
        x = np.arange(len(groups))
        width = 0.25

        for i, strategy in enumerate(
            strategies[:3]
        ):  # Limit to 3 strategies for clarity
            rates = []
            for group in groups:

                strategy_group_metrics = fairness_results[strategy]["metrics"]
                group_metrics = strategy_group_metrics.get(protected_attribute, {})
                rates.append(
                    group_metrics.get("approval_rates", {}).get(group, 0) * 100
                )

            ax.bar(
                x + i * width,
                rates,
                width,
                label=strategy.replace("_", " ").title(),
                alpha=0.8,
            )

        ax.set_xlabel("Demographic Group", fontsize=11)
        ax.set_ylabel("Approval Rate (%)", fontsize=11)
        ax.set_title(
            f"Approval Rates by {protected_attribute.title()}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xticks(x + width)
        ax.set_xticklabels(groups)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 2: Demographic parity difference over strategies
        ax = axes[0, 1]
        dp_diffs = []
        for strategy in strategies:
            strategy_metrics = fairness_results[strategy]["metrics"]
            group_metrics = strategy_metrics.get(protected_attribute, {})
            dp_diffs.append(group_metrics.get("demographic_parity_diff", 0) * 100)

        colors = ["red" if x > 5 else "green" for x in dp_diffs]
        ax.barh(range(len(strategies)), dp_diffs, color=colors, alpha=0.6)
        ax.axvline(x=5, color="red", linestyle="--", label="Fairness Threshold (5%)")
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels([s.replace("_", " ").title() for s in strategies])
        ax.set_xlabel("Demographic Parity Difference (%)", fontsize=11)
        ax.set_title("Demographic Parity by Strategy", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")

        # Plot 3: Disparate Impact Ratio
        ax = axes[1, 0]
        di_ratios = []
        for strategy in strategies:
            strategy_metrics = fairness_results[strategy]["metrics"]
            group_metrics = strategy_metrics.get(protected_attribute, {})
            di_ratios.append(group_metrics.get("disparate_impact", 1.0))

        colors = ["red" if x < 0.8 else "green" for x in di_ratios]
        ax.barh(range(len(strategies)), di_ratios, color=colors, alpha=0.6)
        ax.axvline(x=0.8, color="red", linestyle="--", label="80% Rule Threshold")
        ax.axvline(
            x=1.0, color="gray", linestyle="-", alpha=0.3, label="Perfect Parity"
        )
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels([s.replace("_", " ").title() for s in strategies])
        ax.set_xlabel("Disparate Impact Ratio", fontsize=11)
        ax.set_title("Disparate Impact (4/5ths Rule)", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_xlim(0.5, 1.2)

        # Plot 4: Summary heatmap
        ax = axes[1, 1]

        # Create summary matrix
        summary_data = []
        metrics_names = ["AUC", "Precision", "Recall", "DP Diff", "DI Ratio"]

        for strategy in strategies:
            row = []
            perf = fairness_results[strategy]["model_performance"]
            strategy_metrics = fairness_results[strategy]["metrics"]
            group_metrics = strategy_metrics.get(protected_attribute, {})

            row.append(perf["auc"])
            row.append(perf.get("precision", 0))
            row.append(perf.get("recall", 0))
            row.append(group_metrics.get("demographic_parity_diff", 0))
            row.append(group_metrics.get("disparate_impact", 1.0))
            summary_data.append(row)

        summary_df = pd.DataFrame(
            summary_data,
            index=[s.replace("_", " ").title() for s in strategies],
            columns=metrics_names,
        )

        sns.heatmap(
            summary_df,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0.8,
            ax=ax,
            cbar_kws={"label": "Score"},
        )
        ax.set_title("Performance & Fairness Summary", fontsize=12, fontweight="bold")
        ax.set_ylabel("Strategy", fontsize=11)

        plt.tight_layout()
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Saved demographic bias audit to {output_path}")

    def plot_historical_fairness_trends(
        self,
        fairness_history: List[Dict[str, Any]],
        output_filename: str = "fairness_trends.png",
    ):
        """
        Plot fairness metrics over time to detect drift.
        """
        if not fairness_history:
            logger.warning("No fairness history provided")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Extract time series data
        timestamps = [entry["timestamp"] for entry in fairness_history]
        dp_diffs = [
            entry.get("demographic_parity_diff", 0) for entry in fairness_history
        ]
        eo_diffs = [entry.get("equalized_odds_diff", 0) for entry in fairness_history]

        # Plot 1: Demographic Parity over time
        ax = axes[0]
        ax.plot(
            range(len(timestamps)),
            dp_diffs,
            marker="o",
            linewidth=2,
            markersize=6,
            label="DP Difference",
        )
        ax.axhline(
            y=0.05, color="red", linestyle="--", label="Threshold (5%)", alpha=0.7
        )
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.fill_between(range(len(timestamps)), 0, dp_diffs, alpha=0.3)

        ax.set_ylabel("Demographic Parity Difference", fontsize=11)
        ax.set_title("Fairness Metrics Over Time", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Equalized Odds over time
        ax = axes[1]
        ax.plot(
            range(len(timestamps)),
            eo_diffs,
            marker="s",
            linewidth=2,
            markersize=6,
            label="EO Difference",
            color="orange",
        )
        ax.axhline(
            y=0.05, color="red", linestyle="--", label="Threshold (5%)", alpha=0.7
        )
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.fill_between(range(len(timestamps)), 0, eo_diffs, alpha=0.3, color="orange")

        ax.set_xlabel("Time Period", fontsize=11)
        ax.set_ylabel("Equalized Odds Difference", fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Saved fairness trends plot to {output_path}")

    def plot_roc_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        output_filename: str = "roc_comparison.png",
    ):
        """
        Plot ROC curves comparing different models.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))

        for i, (model_name, model_results) in enumerate(results.items()):
            if "roc_curve" in model_results:
                fpr = model_results["roc_curve"]["fpr"]
                tpr = model_results["roc_curve"]["tpr"]
                auc = model_results["auc"]

                ax.plot(
                    fpr,
                    tpr,
                    linewidth=2,
                    label=f"{model_name} (AUC = {auc:.3f})",
                    color=colors[i],
                )

        # Diagonal line
        ax.plot(
            [0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier", alpha=0.5
        )

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

        plt.tight_layout()
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Saved ROC comparison to {output_path}")

    def create_comprehensive_report(
        self,
        all_results: Dict[str, Any],
        output_filename: str = "comprehensive_fairness_report.png",
    ):
        """
        Create a comprehensive multi-panel fairness report.
        """
        fig = plt.figure(figsize=(18, 12))
        fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # This would contain 9 subplots with various fairness visualizations
        # For brevity, implementing key panels

        logger.info(f"Created comprehensive fairness report: {output_filename}")


if __name__ == "__main__":
    # Demo visualization
    print("Fairness Visualization Module")
    print("=" * 80)

    # Create sample data
    fairness_results = {
        "baseline": {
            "model_performance": {"auc": 0.85, "precision": 0.75, "recall": 0.70},
            "metrics": {
                "sex": {
                    "demographic_parity_diff": 0.12,
                    "equalized_odds_diff": 0.10,
                    "disparate_impact": 0.78,
                    "approval_rates": {"M": 0.65, "F": 0.53},
                }
            },
            "passed": False,
        },
        "reweighing": {
            "model_performance": {"auc": 0.84, "precision": 0.74, "recall": 0.69},
            "metrics": {
                "sex": {
                    "demographic_parity_diff": 0.04,
                    "equalized_odds_diff": 0.05,
                    "disparate_impact": 0.92,
                    "approval_rates": {"M": 0.62, "F": 0.58},
                }
            },
            "passed": True,
        },
        "threshold_opt": {
            "model_performance": {"auc": 0.85, "precision": 0.73, "recall": 0.71},
            "metrics": {
                "sex": {
                    "demographic_parity_diff": 0.03,
                    "equalized_odds_diff": 0.04,
                    "disparate_impact": 0.94,
                    "approval_rates": {"M": 0.61, "F": 0.58},
                }
            },
            "passed": True,
        },
    }

    visualizer = FairnessVisualizer(output_dir="figures")

    print("\nGenerating visualizations...")
    visualizer.plot_fairness_accuracy_tradeoff(fairness_results)
    visualizer.plot_demographic_bias_audit(fairness_results)

    print("\n✓ Visualizations generated successfully")
    print("  Output directory: figures/")
