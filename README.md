# PACD

## Testing & Baselines

### PACD-IVAPCI pipeline

Run the end-to-end pipeline:

```bash
python run_pacd_ivapci_pipeline.py --data sachs_data.csv --output results/pacd
```

Key CLI options:

- `--alpha`: CI significance level
- `--max-k`: maximum conditioning set size
- `--estimator`: `ivapci`, `pacd`, or `simple`
- `--epochs`: training epochs for IVAPCI/PACD-T
- `--baseline-conds`: baseline conditions used for intervention evidence
- `--effect-threshold`: fixed pruning threshold (defaults to quantile-based)
- `--effect-quantile`: quantile-based pruning threshold

### PC baseline

If you have `causal-learn` available, run the PC baseline:

```bash
python run_pc_baseline.py --data sachs_data.csv --output results/pc --alpha 0.001 --max-k 3
```

The PC baseline exports:

- `skeleton.csv`: undirected edges
- `cpdag.csv`: directed/undirected edges from the learned CPDAG
- `pc_graph.json`: JSON representation for downstream comparison

If `causal-learn` is not installed, the script will print an installation hint and exit.

### Synthetic benchmark (PACD vs PC)

Generate synthetic scenarios and compare PACD skeletons with PC:

```bash
python run_synthetic_benchmark.py --output results/synthetic --n 1000 --alpha 0.001 --max-k 3
```

Outputs:
- per-scenario CSV data
- ground-truth edges (`*_truth.json`)
- summary metrics (`summary.json`)
