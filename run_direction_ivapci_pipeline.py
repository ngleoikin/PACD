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
from typing import Dict, List, Optional, Tuple

import importlib.util
import numpy as np
import pandas as pd

from model_wrapper import estimate_ate_ivapci, is_ivapci_available, load_ivapci
from pacd_structure_learning import (
    MPCDConfig,
    MPCDStructureLearner,
    PACDStructureConfig,
    PACDStructureLearner,
)
from s3cdo_structure_learning import S3CDOConfig, S3CDOStructureLearner
from run_pacd_ivapci_pipeline import InterventionDirector, PipelineConfig


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
                undirected = False
            elif endpoints == "ARROW-TAIL":
                source, target = var_names[j], var_names[i]
                undirected = False
            else:
                source, target = var_names[i], var_names[j]
                undirected = True
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "direction_confidence": "pc_directed" if not undirected else "pc_undirected",
                    "endpoints": endpoints,
                    "undirected": undirected,
                }
            )
    return edges


def _print_edges(label: str, edges: List[Dict]) -> None:
    print(f"[{label}] edges = {len(edges)}")
    for edge in edges:
        marker = " (undirected)" if edge.get("undirected") else ""
        print(f"  - {edge['source']} -> {edge['target']}{marker}")


def _default_intervention_map(
    cond_values: List[str], baseline_conds: List[str], var_names: List[str]
) -> Dict[str, Dict[str, List[str]]]:
    return {
        cond: {"targets": var_names}
        for cond in cond_values
        if cond not in baseline_conds
    }


def _apply_intervention_evidence(
    data: pd.DataFrame,
    directed_edges: List[Dict],
    sepsets: Dict[str, List[str]],
    baseline_conds: List[str],
    intervention_map: Dict[str, Dict[str, List[str]]],
) -> List[Dict]:
    config = PipelineConfig(baseline_conditions=tuple(baseline_conds))
    director = InterventionDirector(config, intervention_map)
    updated_edges = []
    for edge in directed_edges:
        v1, v2 = edge["source"], edge["target"]
        sepset = sepsets.get(f"{v1}|{v2}", sepsets.get(f"{v2}|{v1}", []))
        evidence = director.get_direction_evidence(data, v1, v2, sepset)
        reverse_evidence = director.get_direction_evidence(data, v2, v1, sepset)
        score_f = evidence["y_shift"]["z"] + evidence["residual_shift"]["z"]
        score_r = reverse_evidence["y_shift"]["z"] + reverse_evidence["residual_shift"]["z"]

        if score_r - score_f > 0.5:
            source, target = v2, v1
            chosen = reverse_evidence
            reverse = evidence
            final_conf = "high"
            score = score_r
            reverse_score = score_f
        elif score_f - score_r > 0.5:
            source, target = v1, v2
            chosen = evidence
            reverse = reverse_evidence
            final_conf = "high"
            score = score_f
            reverse_score = score_r
        else:
            source, target = v1, v2
            chosen = evidence
            reverse = reverse_evidence
            final_conf = edge.get("direction_confidence", "pacd")
            score = score_f
            reverse_score = score_r

        updated_edges.append(
            {
                **edge,
                "source": source,
                "target": target,
                "direction_confidence": final_conf,
                "intervention_effect": chosen["y_shift"]["effect"],
                "intervention_p": chosen["y_shift"]["p_value"],
                "intervention_residual_p": chosen["residual_shift"]["p_value"],
                "intervention_score": score,
                "intervention_reverse_score": reverse_score,
                "intervention_valid_envs": [env["env"] for env in chosen["valid_envs"]],
                "intervention_reverse_valid_envs": [
                    env["env"] for env in reverse["valid_envs"]
                ],
            }
        )
    return updated_edges


def _estimate_ivapci_for_edge(
    data: pd.DataFrame,
    source: str,
    target: str,
    device: str,
    epochs: int,
    n_bootstrap: int,
    progress: Optional[Dict[str, int]] = None,
) -> Dict:
    env_cols = []
    if "COND" in data.columns:
        env_cols = list(pd.get_dummies(data["COND"], prefix="E", drop_first=True).columns)

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    covariates = [c for c in numeric_cols if c not in [source, target]]
    X_env = (
        pd.get_dummies(data["COND"], prefix="E", drop_first=True).values.astype(np.float32)
        if env_cols
        else np.zeros((len(data), 0), dtype=np.float32)
    )
    X_cov = (
        data[covariates].values.astype(np.float32)
        if covariates
        else np.zeros((len(data), 1), dtype=np.float32)
    )
    X_all = np.concatenate([X_env, X_cov], axis=1) if X_env.size else X_cov
    A = (data[source].values > np.median(data[source].values)).astype(np.float32)
    Y = data[target].values.astype(np.float32)

    if X_all.shape[1] == 0:
        X_all = np.zeros((len(data), 1), dtype=np.float32)

    d_all = X_all.shape[1]
    if d_all < 3:
        extra = np.repeat(X_all[:, [-1]], 3 - d_all, axis=1)
        X_all = np.concatenate([X_all, extra], axis=1)
        d_all = X_all.shape[1]

    x_dim = max(1, X_env.shape[1])
    remaining = d_all - x_dim
    if remaining < 2:
        x_dim = max(1, d_all - 2)
        remaining = d_all - x_dim
    w_dim = max(1, remaining // 2)
    z_dim = remaining - w_dim
    if z_dim < 1:
        z_dim = 1
        w_dim = max(1, remaining - z_dim)

    result = estimate_ate_ivapci(
        X_all,
        A,
        Y,
        x_dim=x_dim,
        w_dim=w_dim,
        z_dim=z_dim,
        epochs=epochs,
        device=device,
        n_bootstrap=n_bootstrap,
        progress=progress,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Direction + IVAPCI pipeline")
    parser.add_argument("--data", "-d", required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", default="./results/dir_ivapci", help="Output directory")
    parser.add_argument(
        "--direction",
        choices=["pc", "pacd", "mpcd", "s3cdo"],
        default="pacd",
        help="Direction method",
    )
    parser.add_argument(
        "--s3cdo-top-m",
        type=int,
        default=8,
        help="S3C-DO screening top-m neighbors",
    )
    parser.add_argument(
        "--s3cdo-ci-method",
        choices=["spearman", "pearson"],
        default="spearman",
        help="S3C-DO CI method (spearman/pearson)",
    )
    parser.add_argument(
        "--s3cdo-use-nonparanormal",
        action="store_true",
        help="Enable nonparanormal transform for S3C-DO",
    )
    parser.add_argument(
        "--s3cdo-ci-perm-samples",
        type=int,
        default=200,
        help="S3C-DO permutation samples for Spearman CI",
    )
    parser.add_argument(
        "--s3cdo-auto-fix-perm-resolution",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-fix alpha to permutation resolution for S3C-DO Spearman CI",
    )
    parser.add_argument(
        "--s3cdo-collider-rule",
        choices=["naive", "cpc", "majority"],
        default="cpc",
        help="S3C-DO collider rule (naive/cpc/majority)",
    )
    parser.add_argument(
        "--s3cdo-collider-majority-threshold",
        type=float,
        default=0.5,
        help="S3C-DO majority rule threshold",
    )
    parser.add_argument(
        "--s3cdo-fallback-sepset-search",
        action="store_true",
        help="Enable fallback sepset search for unshielded triples",
    )
    parser.add_argument(
        "--s3cdo-fallback-max-k",
        type=int,
        default=None,
        help="Fallback max-k for sepset search",
    )
    parser.add_argument(
        "--s3cdo-bootstrap",
        type=int,
        default=0,
        help="Bootstrap runs for S3C-DO stability (0 to disable)",
    )
    parser.add_argument(
        "--s3cdo-bootstrap-threshold",
        type=float,
        default=0.95,
        help="Skeleton stability threshold for S3C-DO bootstrap",
    )
    parser.add_argument(
        "--s3cdo-dir-threshold",
        type=float,
        default=0.95,
        help="Direction stability threshold for S3C-DO bootstrap",
    )
    parser.add_argument(
        "--mpcd-m-grid",
        default="",
        help="MPCD scales, comma-separated (e.g. 2,3,4,5)",
    )
    parser.add_argument(
        "--mpcd-stability-tau",
        type=float,
        default=0.6,
        help="MPCD stability threshold",
    )
    parser.add_argument("--alpha", type=float, default=0.001, help="CI significance level")
    parser.add_argument("--max-k", type=int, default=3, help="Maximum conditioning set size")
    parser.add_argument("--epochs", type=int, default=80, help="IVAPCI training epochs")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--n-bootstrap", type=int, default=50, help="Bootstrap samples")
    parser.add_argument(
        "--only-structure",
        action="store_true",
        help="Only run structure learning and save directed edges without IVAPCI",
    )
    parser.add_argument(
        "--baseline-conds",
        default="",
        help="Comma-separated baseline COND values for intervention evidence (e.g. env_0)",
    )
    parser.add_argument(
        "--intervention",
        default=None,
        help="JSON file for intervention map (optional)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise SystemExit("No numeric columns found.")

    var_names = list(numeric_df.columns)
    data = numeric_df

    if args.direction in {"pacd", "mpcd"}:
        config = PACDStructureConfig(alpha=args.alpha, max_k=args.max_k)
        if args.direction == "mpcd":
            m_grid = [
                int(item.strip())
                for item in args.mpcd_m_grid.split(",")
                if item.strip()
            ]
            mpcd_config = MPCDConfig(
                m_grid=m_grid or None,
                stability_tau=args.mpcd_stability_tau,
                base_config=config,
            )
            learner = MPCDStructureLearner(mpcd_config)
        else:
            learner = PACDStructureLearner(config)
        result = learner.learn(data.values, var_names)
        directed_edges = result["directed_edges"]
        if "COND" in df.columns:
            baseline_conds = [
                item.strip()
                for item in args.baseline_conds.split(",")
                if item.strip()
            ]
            if not baseline_conds:
                baseline_conds = list(PipelineConfig().baseline_conditions)
            if args.intervention:
                with open(args.intervention, "r", encoding="utf-8") as handle:
                    intervention_map = json.load(handle)
            else:
                intervention_map = _default_intervention_map(
                    sorted(df["COND"].astype(str).unique()),
                    baseline_conds,
                    var_names,
                )
            directed_edges = _apply_intervention_evidence(
                df,
                directed_edges,
                result.get("sepsets", {}),
                baseline_conds,
                intervention_map,
            )
    else:
        if args.direction == "s3cdo":
            s3_cfg = S3CDOConfig(
                top_m=args.s3cdo_top_m,
                alpha=args.alpha,
                max_k=args.max_k,
                ci_method=args.s3cdo_ci_method,
                use_nonparanormal=args.s3cdo_use_nonparanormal,
                ci_perm_samples=args.s3cdo_ci_perm_samples,
                auto_fix_perm_resolution=args.s3cdo_auto_fix_perm_resolution,
                collider_rule=args.s3cdo_collider_rule,
                collider_majority_threshold=args.s3cdo_collider_majority_threshold,
                fallback_sepset_search=args.s3cdo_fallback_sepset_search,
                fallback_max_k=args.s3cdo_fallback_max_k,
            )
            s3_learner = S3CDOStructureLearner(s3_cfg)
            s3_result = s3_learner.learn(data.values, var_names)
            directed_edges = s3_result["directed_edges"]
            sepsets = s3_result.get("sepsets", {})
            if args.s3cdo_bootstrap > 0:
                rng = np.random.default_rng(0)
                n_samples = data.shape[0]
                skeleton_counts: Dict[Tuple[str, str], int] = {}
                dir_counts: Dict[Tuple[str, str], int] = {}
                for _ in range(args.s3cdo_bootstrap):
                    indices = rng.integers(0, n_samples, size=n_samples)
                    boot_data = data.values[indices]
                    boot_learner = S3CDOStructureLearner(s3_cfg)
                    boot_result = boot_learner.learn(boot_data, var_names)
                    for u, v in boot_result.get("skeleton", []):
                        key = (u, v) if u < v else (v, u)
                        skeleton_counts[key] = skeleton_counts.get(key, 0) + 1
                    for edge in boot_result.get("directed_edges", []):
                        if edge.get("orientation_method") == "undirected":
                            continue
                        key = (edge["source"], edge["target"])
                        dir_counts[key] = dir_counts.get(key, 0) + 1

                boot_total = max(1, args.s3cdo_bootstrap)
                filtered_edges = []
                for edge in directed_edges:
                    key = tuple(sorted((edge["source"], edge["target"])))
                    sk_freq = skeleton_counts.get(key, 0) / boot_total
                    edge["skeleton_stability"] = sk_freq
                    dir_freq = None
                    if edge.get("orientation_method") != "undirected":
                        dir_freq = dir_counts.get((edge["source"], edge["target"]), 0) / boot_total
                        edge["dir_stability"] = dir_freq
                    else:
                        edge["dir_stability"] = None

                    if sk_freq < args.s3cdo_bootstrap_threshold:
                        continue
                    if (
                        edge.get("orientation_method") != "undirected"
                        and (dir_freq or 0.0) < args.s3cdo_dir_threshold
                    ):
                        edge["orientation_method"] = "undirected"
                        edge["direction_confidence"] = "undecided"
                    filtered_edges.append(edge)
                directed_edges = filtered_edges
        else:
            pc_result = _run_pc(data.values, args.alpha, args.max_k)
            directed_edges = _directed_edges_from_pc(pc_result.G, var_names)
            sepsets = {}

        if "COND" in df.columns:
            baseline_conds = [
                item.strip()
                for item in args.baseline_conds.split(",")
                if item.strip()
            ]
            if not baseline_conds:
                baseline_conds = list(PipelineConfig().baseline_conditions)
            if args.intervention:
                with open(args.intervention, "r", encoding="utf-8") as handle:
                    intervention_map = json.load(handle)
            else:
                intervention_map = _default_intervention_map(
                    sorted(df["COND"].astype(str).unique()),
                    baseline_conds,
                    var_names,
                )
            directed_edges = _apply_intervention_evidence(
                df,
                directed_edges,
                sepsets,
                baseline_conds,
                intervention_map,
            )

    _print_edges(args.direction.upper(), directed_edges)
    if args.only_structure:
        os.makedirs(args.output, exist_ok=True)
        pd.DataFrame(directed_edges).to_csv(
            os.path.join(args.output, "directed_edges.csv"), index=False
        )
        with open(
            os.path.join(args.output, "directed_edges.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump({"edges": directed_edges, "method": args.direction}, handle, indent=2)
        print(f"Saved directed edges to {args.output}/")
        return

    total_edges = len(directed_edges)
    total_trains = total_edges * (1 + max(args.n_bootstrap, 0))
    print(
        f"[IVAPCI] will estimate {total_edges} edges, "
        f"{total_trains} total training runs (1 + n_bootstrap per edge)"
    )

    if not is_ivapci_available():
        _, _, err = load_ivapci()
        raise SystemExit(f"IVAPCI is not available: {err}")

    edges_with_effects = []
    for idx, edge in enumerate(directed_edges, start=1):
        source = edge["source"]
        target = edge["target"]
        print(f"[IVAPCI] {source} -> {target} ...")
        effect = _estimate_ivapci_for_edge(
            df,
            source,
            target,
            device=args.device,
            epochs=args.epochs,
            n_bootstrap=args.n_bootstrap,
            progress={
                "edge_index": idx,
                "edge_total": total_edges,
                "base_scenario": (idx - 1) * (1 + max(args.n_bootstrap, 0)) + 1,
                "total_scenarios": total_trains,
            },
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
