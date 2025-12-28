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
