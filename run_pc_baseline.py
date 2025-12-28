#!/usr/bin/env python3
"""
PC baseline runner
=================

Runs the PC algorithm using causal-learn and saves skeleton/CPDAG outputs.

Usage:
    python run_pc_baseline.py --data sachs_data.csv --output results/pc
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import importlib.util
import numpy as np
import pandas as pd


def _require_causal_learn() -> None:
    if importlib.util.find_spec("causallearn") is None:
        message = (
            "causal-learn is required for PC baseline.\n"
            "Install with: pip install causal-learn"
        )
        raise SystemExit(message)


def _run_pc(data: np.ndarray, alpha: float, max_k: int):
    _require_causal_learn()
    from causallearn.search.ConstraintBased.PC import pc

    return pc(data, alpha=alpha, stable=True, uc_rule=0, uc_priority=0, max_k=max_k)


def _graph_edges(graph, var_names: List[str]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    skeleton = []
    cpdag = []
    node_count = graph.get_num_nodes()
    for i in range(node_count):
        for j in range(i + 1, node_count):
            edge = graph.get_edge(graph.nodes[i], graph.nodes[j])
            if edge is None:
                continue
            skeleton.append((var_names[i], var_names[j]))
            cpdag.append((var_names[i], var_names[j], edge.get_endpoint1().name + edge.get_endpoint2().name))
    return skeleton, cpdag


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PC baseline (causal-learn)")
    parser.add_argument("--data", "-d", required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", default="./results/pc", help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.001, help="CI significance level")
    parser.add_argument("--max-k", type=int, default=3, help="Maximum conditioning set size")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Missing input file: {data_path}")

    df = pd.read_csv(data_path)
    var_names = list(df.columns)
    X = df.values.astype(float)

    result = _run_pc(X, alpha=args.alpha, max_k=args.max_k)
    skeleton, cpdag = _graph_edges(result.G, var_names)

    os.makedirs(args.output, exist_ok=True)
    pd.DataFrame(skeleton, columns=["node_i", "node_j"]).to_csv(
        os.path.join(args.output, "skeleton.csv"), index=False
    )
    pd.DataFrame(cpdag, columns=["node_i", "node_j", "endpoints"]).to_csv(
        os.path.join(args.output, "cpdag.csv"), index=False
    )

    with open(os.path.join(args.output, "pc_graph.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "nodes": var_names,
                "skeleton": skeleton,
                "cpdag": cpdag,
                "alpha": args.alpha,
                "max_k": args.max_k,
            },
            handle,
            indent=2,
        )

    print(f"Saved PC baseline to {args.output}/")


if __name__ == "__main__":
    main()
