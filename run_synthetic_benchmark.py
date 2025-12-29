#!/usr/bin/env python3
"""
Synthetic benchmark runner
==========================

Generate synthetic scenarios and compare PACD skeleton vs PC baseline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import importlib.util
import numpy as np
import pandas as pd

from pacd_structure_learning import PACDStructureConfig, PACDStructureLearner


def _maybe_run_pc(data: np.ndarray, alpha: float, max_k: int):
    if importlib.util.find_spec("causallearn") is None:
        return None
    from causallearn.search.ConstraintBased.PC import pc
    return pc(data, alpha=alpha, stable=True, uc_rule=0, uc_priority=0, max_k=max_k)


def _edges_from_pc(graph, var_names: List[str]) -> List[Tuple[str, str]]:
    edges = []
    node_count = graph.get_num_nodes()
    for i in range(node_count):
        for j in range(i + 1, node_count):
            edge = graph.get_edge(graph.nodes[i], graph.nodes[j])
            if edge is not None:
                edges.append((var_names[i], var_names[j]))
    return edges


def _score_skeleton(true_edges: List[Tuple[str, str]], pred_edges: List[Tuple[str, str]]) -> Dict:
    true_set = {tuple(sorted(edge)) for edge in true_edges}
    pred_set = {tuple(sorted(edge)) for edge in pred_edges}
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    shd = fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": shd,
    }


def _scenario_linear(n: int, rng: np.random.Generator, n_vars: int):
    x1 = rng.normal(size=n)
    x2 = 0.8 * x1 + rng.normal(scale=0.5, size=n)
    x3 = 0.6 * x2 + rng.normal(scale=0.5, size=n)
    x4 = 0.4 * x1 + 0.5 * x3 + rng.normal(scale=0.5, size=n)
    base = [x1, x2, x3, x4]
    extra = []
    for k in range(4, n_vars):
        parent = base[k % len(base)]
        extra.append(0.6 * parent + rng.normal(scale=0.6, size=n))
    data = np.column_stack(base + extra)
    edges = [("X1", "X2"), ("X2", "X3"), ("X1", "X4"), ("X3", "X4")]
    for idx in range(5, n_vars + 1):
        parent_idx = (idx - 1) % 4 + 1
        edges.append((f"X{parent_idx}", f"X{idx}"))
    return data, edges


def _scenario_nonlinear(n: int, rng: np.random.Generator, n_vars: int):
    x1 = rng.normal(size=n)
    x2 = np.tanh(x1) + rng.normal(scale=0.3, size=n)
    x3 = np.sin(x2) + rng.normal(scale=0.3, size=n)
    x4 = x1 * x3 + rng.normal(scale=0.5, size=n)
    base = [x1, x2, x3, x4]
    extra = []
    for k in range(4, n_vars):
        parent = base[k % len(base)]
        extra.append(np.tanh(parent) + rng.normal(scale=0.5, size=n))
    data = np.column_stack(base + extra)
    edges = [("X1", "X2"), ("X2", "X3"), ("X1", "X4"), ("X3", "X4")]
    for idx in range(5, n_vars + 1):
        parent_idx = (idx - 1) % 4 + 1
        edges.append((f"X{parent_idx}", f"X{idx}"))
    return data, edges


def _scenario_heteroscedastic(n: int, rng: np.random.Generator, n_vars: int):
    x1 = rng.normal(size=n)
    noise = rng.normal(scale=0.2 + 0.3 * np.abs(x1), size=n)
    x2 = 0.7 * x1 + noise
    x3 = 0.6 * x2 + rng.normal(scale=0.4, size=n)
    base = [x1, x2, x3]
    extra = []
    for k in range(3, n_vars):
        parent = base[k % len(base)]
        extra.append(0.5 * parent + rng.normal(scale=0.5, size=n))
    data = np.column_stack(base + extra)
    edges = [("X1", "X2"), ("X2", "X3")]
    for idx in range(4, n_vars + 1):
        parent_idx = (idx - 1) % 3 + 1
        edges.append((f"X{parent_idx}", f"X{idx}"))
    return data, edges


def _scenario_measurement_error(n: int, rng: np.random.Generator, n_vars: int):
    x1_true = rng.normal(size=n)
    x1 = x1_true + rng.normal(scale=0.2, size=n)
    x2 = 0.9 * x1_true + rng.normal(scale=0.5, size=n)
    x3 = 0.6 * x2 + rng.normal(scale=0.5, size=n)
    base = [x1, x2, x3]
    extra = []
    for k in range(3, n_vars):
        parent = base[k % len(base)]
        extra.append(0.7 * parent + rng.normal(scale=0.6, size=n))
    data = np.column_stack(base + extra)
    edges = [("X1", "X2"), ("X2", "X3")]
    for idx in range(4, n_vars + 1):
        parent_idx = (idx - 1) % 3 + 1
        edges.append((f"X{parent_idx}", f"X{idx}"))
    return data, edges


SCENARIOS = {
    "linear": _scenario_linear,
    "nonlinear": _scenario_nonlinear,
    "hetero": _scenario_heteroscedastic,
    "measurement": _scenario_measurement_error,
}


SCENARIO_SETS = {
    "core": ["linear", "nonlinear", "hetero", "measurement"],
    "stress": ["nonlinear", "hetero", "measurement"],
    "all": ["linear", "nonlinear", "hetero", "measurement", "multienv_soft"],
    "multienv": ["multienv_soft"],
}


def _scenario_multienv_soft(n: int, rng: np.random.Generator, n_vars: int, n_envs: int = 5):
    base_data, edges = _scenario_linear(n, rng, n_vars)
    envs = rng.integers(0, n_envs, size=n)
    data = base_data.copy()
    for env in range(n_envs):
        idx = envs == env
        scale = 1.0 + 0.15 * env
        data[idx, 0] = data[idx, 0] * scale + rng.normal(scale=0.2, size=idx.sum())
        data[idx, 1] = data[idx, 1] + 0.2 * env
    return data, edges, envs


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic benchmark for PACD vs PC")
    parser.add_argument("--output", "-o", default="./results/synthetic", help="Output directory")
    parser.add_argument("--n", type=int, default=1000, help="Sample size per scenario")
    parser.add_argument("--n-vars", type=int, default=12, help="Number of variables")
    parser.add_argument(
        "--scenario-set",
        choices=sorted(SCENARIO_SETS.keys()),
        default="core",
        help="Scenario set to run",
    )
    parser.add_argument("--alpha", type=float, default=0.001, help="CI significance level")
    parser.add_argument("--max-k", type=int, default=3, help="Maximum conditioning set size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    results = []
    scenario_names = SCENARIO_SETS[args.scenario_set]
    for name in scenario_names:
        if name == "multienv_soft":
            data, true_edges, envs = _scenario_multienv_soft(args.n, rng, args.n_vars)
        else:
            generator = SCENARIOS[name]
            data, true_edges = generator(args.n, rng, args.n_vars)
            envs = None

        var_names = [f"X{i+1}" for i in range(data.shape[1])]

        pacd_config = PACDStructureConfig(
            alpha=args.alpha,
            max_k=args.max_k,
            ci_method="pearson",
            use_nonparanormal=True,
        )
        learner = PACDStructureLearner(pacd_config)
        pacd_result = learner.learn(data, var_names)
        pacd_score = _score_skeleton(true_edges, pacd_result["skeleton"])

        pc_score = None
        pc_env_score = None
        pc_available = importlib.util.find_spec("causallearn") is not None
        if pc_available:
            pc_result = _maybe_run_pc(data, args.alpha, args.max_k)
            pc_edges = _edges_from_pc(pc_result.G, var_names)
            pc_score = _score_skeleton(true_edges, pc_edges)
            if envs is not None:
                env_edges = set()
                for env in np.unique(envs):
                    env_data = data[envs == env]
                    env_result = _maybe_run_pc(env_data, args.alpha, args.max_k)
                    env_edges.update(_edges_from_pc(env_result.G, var_names))
                pc_env_score = _score_skeleton(true_edges, list(env_edges))

        results.append(
            {
                "scenario": name,
                "n": args.n,
                "n_vars": args.n_vars,
                "pacd": pacd_score,
                "pc": pc_score,
                "pc_env": pc_env_score,
                "pc_available": pc_available,
            }
        )

        pd.DataFrame(data, columns=var_names).to_csv(
            output_dir / f"{name}.csv", index=False
        )
        with open(output_dir / f"{name}_truth.json", "w", encoding="utf-8") as handle:
            json.dump({"edges": true_edges}, handle, indent=2)
        truth_graph = {
            "nodes": var_names,
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "weight": 1.0,
                    "ci": None,
                    "p_value": None,
                    "direction_confidence": "truth",
                    "is_mediated": False,
                    "sepset": [],
                    "intervention": {},
                    "robustness": {},
                }
                for source, target in true_edges
            ],
            "pruned_edges": [],
            "sepsets": {},
            "meta": {"truth": True, "scenario": name},
        }
        with open(output_dir / f"{name}_truth_graph.json", "w", encoding="utf-8") as handle:
            json.dump(truth_graph, handle, indent=2)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved synthetic benchmark outputs to {output_dir}")


if __name__ == "__main__":
    main()
