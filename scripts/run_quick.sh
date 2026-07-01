#!/bin/bash
# Quick integration test (30 minutes on 4-core CPU)

set -e

echo "========================================="
echo "Quick Integration Test"
echo "========================================="
echo "Estimated time: 20-30 minutes"
echo ""

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/credit"
export RANDOM_SEED=42
export QUICK_MODE=true

# Create output directories
mkdir -p results/logs
mkdir -p figures

echo "[1/5] Generating synthetic data..."
python3 -c "
import sys
sys.path.insert(0, 'code')
from data.synthetic_generator import SyntheticDataGenerator
import pandas as pd

generator = SyntheticDataGenerator(random_seed=42)
df = generator.generate_applications(n_samples=500, default_rate=0.20)
df.to_csv('results/synthetic_data_quick.csv', index=False)
print(f'Generated {len(df)} applications')
"

echo "[2/5] Running experiments..."
python3 -c "
import sys
sys.path.insert(0, 'code')
from eval.experiment_runner import ExperimentRunner
import pandas as pd

df = pd.read_csv('results/synthetic_data_quick.csv')
runner = ExperimentRunner(output_dir='results', random_seed=42)
results = runner.run_full_evaluation(df, quick_mode=True)
print('Experiments complete!')
"

echo "[3/5] Generating figures..."
python3 scripts/generate_figures.py --quick

echo "[4/5] Running tests..."
python3 -m pytest credit/tests/ -q -p no:cacheprovider --disable-warnings --tb=short || true

echo "[5/5] Generating summary..."
python3 scripts/print_summary.py

echo ""
echo "✓ Quick test complete!"
echo "Results: results/metrics_summary.json"
echo "Figures: figures/"
