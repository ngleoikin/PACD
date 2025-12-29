#!/usr/bin/env python3
"""
Directional structure + IVAPCI effect pipeline
=============================================

Use PC or PACD to orient edges, then estimate edge effects with IVAPCI.

Usage:
  python run_direction_ivapci_pipeline.py --data sachs_data.csv --output results/dir_ivapci --direction pacd
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import importlib.util
import numpy as np
import pandas as pd

from model_wrapper import estimate_ate_ivapci, is_ivapci_available, load_ivapci
from pacd_structure_learning import PACDStructureConfig, PACDStructureLearner


def _run_pc(data: np.ndarray, alpha: float, max_k: int):
    if importlib.util.find_spec("causallearn") is None:
        raise SystemExit("causal-learn is required for PC direction pipeline.")
    from causallearn.search.ConstraintBased.PC import pc

    return pc(data, alpha=alpha, stable=True, uc_rule=0, uc_priority=0, max_k=max_k)


def _directed_edges_from_pc(graph, var_names: List[str]) -> List[Dict]:
    edges = []
    node_count = graph.get_num_nodes()
    for i in range(node_count):
        for j in range(i + 1, node_count):
            edge = graph.get_edge(graph.nodes[i], graph.nodes[j])
            if edge is None:
                continue
            endpoints = edge.get_endpoint1().name + edge.get_endpoint2().name
            if endpoints == "TAIL-ARROW":
                source, target = var_names[i], var_names[j]
            elif endpoints == "ARROW-TAIL":
                source, target = var_names[j], var_names[i]
            else:
                source, target = var_names[i], var_names[j]
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "direction_confidence": "pc",
                    "endpoints": endpoints,
                }
            )
    return edges


def _estimate_ivapci_for_edge(
    data: pd.DataFrame,
    source: str,
    target: str,
    device: str,
    epochs: int,
    n_bootstrap: int,
) -> Dict:
    covariates = [c for c in data.columns if c not in [source, target]]
    X_all = data[covariates].values.astype(np.float32) if covariates else np.zeros((len(data), 1), dtype=np.float32)
    A = (data[source].values > np.median(data[source].values)).astype(np.float32)
    Y = data[target].values.astype(np.float32)

    result = estimate_ate_ivapci(
        X_all,
        A,
        Y,
        x_dim=0,
        w_dim=0,
        z_dim=X_all.shape[1],
        epochs=epochs,
        device=device,
        n_bootstrap=n_bootstrap,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Direction + IVAPCI pipeline")
    parser.add_argument("--data", "-d", required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", default="./results/dir_ivapci", help="Output directory")
    parser.add_argument("--direction", choices=["pc", "pacd"], default="pacd", help="Direction method")
    parser.add_argument("--alpha", type=float, default=0.001, help="CI significance level")
    parser.add_argument("--max-k", type=int, default=3, help="Maximum conditioning set size")
    parser.add_argument("--epochs", type=int, default=80, help="IVAPCI training epochs")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--n-bootstrap", type=int, default=50, help="Bootstrap samples")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise SystemExit("No numeric columns found.")

    var_names = list(numeric_df.columns)
    data = numeric_df

    if args.direction == "pacd":
        config = PACDStructureConfig(alpha=args.alpha, max_k=args.max_k)
        learner = PACDStructureLearner(config)
        result = learner.learn(data.values, var_names)
        directed_edges = result["directed_edges"]
    else:
        pc_result = _run_pc(data.values, args.alpha, args.max_k)
        directed_edges = _directed_edges_from_pc(pc_result.G, var_names)

    if not is_ivapci_available():
        _, _, err = load_ivapci()
        raise SystemExit(f"IVAPCI is not available: {err}")

    edges_with_effects = []
    for edge in directed_edges:
        source = edge["source"]
        target = edge["target"]
        effect = _estimate_ivapci_for_edge(
            data, source, target, device=args.device, epochs=args.epochs, n_bootstrap=args.n_bootstrap
        )
        edge_out = {
            **edge,
            "tau": effect["ate"],
            "se": effect["se"],
            "ci": effect["ci"],
            "p_value": effect["p_value"],
            "method": "ivapci",
        }
        edges_with_effects.append(edge_out)

    os.makedirs(args.output, exist_ok=True)
    pd.DataFrame(edges_with_effects).to_csv(os.path.join(args.output, "edge_effects.csv"), index=False)
    with open(os.path.join(args.output, "edge_effects.json"), "w", encoding="utf-8") as handle:
        json.dump({"edges": edges_with_effects, "method": args.direction}, handle, indent=2)

    print(f"Saved edge effects to {args.output}/")


if __name__ == "__main__":
    main()
