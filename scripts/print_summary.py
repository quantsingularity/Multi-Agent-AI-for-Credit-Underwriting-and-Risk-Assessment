"""Print a clean, readable summary of the credit-underwriting quick run."""

import json
import os

WIDTH = 60


def _row(label, value, indent=2):
    dots = max(2, (WIDTH - indent - 12) - len(label))
    print(f"{' ' * indent}{label} {'.' * dots} {value}")


def main():
    path = os.path.join("results", "metrics_summary.json")
    with open(path) as f:
        results = json.load(f)

    meta = results["metadata"]
    print()
    print("=" * WIDTH)
    print("CREDIT UNDERWRITING - QUICK RUN SUMMARY".center(WIDTH))
    print("=" * WIDTH)

    print("\nDataset")
    _row("Train / test", f"{meta['n_train']} / {meta['n_test']}")
    _row("Default rate (test)", f"{meta['default_rate_test']:.1%}")

    print("\nModel performance (AUC)")
    baselines = results["baselines"]
    best = max(baselines, key=lambda k: baselines[k]["auc"])
    for name, metrics in sorted(baselines.items(), key=lambda kv: -kv[1]["auc"]):
        mark = "  <- best" if name == best else ""
        _row(name, f"{metrics['auc']:.4f}{mark}")

    for phase, key in (
        ("Fairness (baseline)", "baseline"),
        ("Fairness (after mitigation)", "reweighing"),
    ):
        block = results["fairness"].get(key, {})
        print(f"\n{phase}")
        _row("Passed", str(block.get("passed", "n/a")))
        for attr, m in block.get("metrics", {}).items():
            if "demographic_parity_diff" in m:
                _row(f"{attr} DP diff", f"{m['demographic_parity_diff']:.4f}")

    print()
    print("=" * WIDTH)
    print("Results: results/metrics_summary.json   Figures: figures/")
    print("=" * WIDTH)


if __name__ == "__main__":
    main()
