#!/usr/bin/env python3
"""
PACD-IVAPCI 完整集成流水线
==========================

流程:
1. PACD 结构学习
   - 非参数CI检验 → 骨架
   - 9环境干预 → 定向 + 删边证据
   - 保存分离集（谁解释了谁）

2. IVAPCI 效应估计
   - 每条边: 直接效应 τ_direct
   - 中介对照: τ_total vs τ_direct
   - 输出: 效应值、CI、稳健性指标

3. 回灌剪枝
   - 剪掉弱边 (|τ| < threshold)
   - 剪掉间接边 (τ_direct ≈ 0 但 τ_total ≠ 0)
   - 标注边权重

使用:
    python run_pacd_ivapci_pipeline.py --data sachs_data.csv --output results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, norm, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge

warnings.filterwarnings("ignore")


# ============================================================
# 配置
# ============================================================


@dataclass
class PipelineConfig:
    """流水线配置"""

    # Step 1: PACD 骨架学习
    alpha_ci: float = 0.001
    max_k: int = 3
    use_nonparanormal: bool = True

    # Step 2: 干预定向
    intervention_alpha: float = 0.05
    min_intervention_effect: float = 0.1

    # Step 3: 效应估计
    estimator: str = "ivapci"
    ivapci_epochs: int = 80
    ivapci_device: str = "cpu"
    n_bootstrap: int = 50

    # Step 4: 回灌剪枝
    effect_threshold: float = 5.0
    indirect_ratio: float = 0.3
    p_threshold: float = 0.05

    # 输出
    top_k_ivapci: int = 30


# ============================================================
# 模型导入
# ============================================================


def import_models() -> Dict:
    """导入PACD和IVAPCI"""
    result = {"pacd": False, "ivapci": False}

    try:
        from model_wrapper import (
            create_ivapci_estimator,
            estimate_ate_ivapci,
            is_ivapci_available,
            estimate_ate_pacd,
            is_pacd_available,
        )
        result["pacd"] = is_pacd_available()
        result["ivapci"] = is_ivapci_available()
        result["estimate_pacd"] = estimate_ate_pacd if result["pacd"] else None
        result["estimate_ivapci"] = estimate_ate_ivapci if result["ivapci"] else None
        result["create_ivapci"] = create_ivapci_estimator if result["ivapci"] else None
    except ImportError as exc:
        print(f"⚠️ model_wrapper导入失败: {exc}")

    return result


# ============================================================
# Step 1: PACD 结构学习 (使用 pacd_structure_learning.py)
# ============================================================


def import_pacd_structure():
    """导入PACD结构学习模块"""
    try:
        from pacd_structure_learning import (
            PACDStructureLearner,
            PACDStructureConfig,
        )
        return PACDStructureLearner, PACDStructureConfig, True
    except ImportError:
        return None, None, False


class PACDSkeletonLearner:
    """
    PACD骨架学习器

    优先使用 pacd_structure_learning.py 中的 p-adic 实现
    如果不可用，回退到简化版
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.skeleton_: Set[Tuple[int, int]] = set()
        self.sepsets_: Dict[Tuple[int, int], List[int]] = {}
        self.edge_pvalues_: Dict[Tuple[int, int], float] = {}

        self._PACDLearner, self._PACDConfig, self._use_pacd = import_pacd_structure()

    def nonparanormal_transform(self, X: np.ndarray) -> np.ndarray:
        """Gaussian Copula变换 (回退用)"""
        n, d = X.shape
        X_transformed = np.zeros_like(X, dtype=float)
        for j in range(d):
            ranks = stats.rankdata(X[:, j])
            delta = 1.0 / (4.0 * n**0.25)
            ranks = np.clip(ranks / (n + 1), delta, 1 - delta)
            X_transformed[:, j] = stats.norm.ppf(ranks)
        return X_transformed

    def partial_correlation_test(
        self, X: np.ndarray, i: int, j: int, S: List[int]
    ) -> Tuple[float, float]:
        """偏相关检验 (回退用)"""
        n = X.shape[0]

        if not S:
            if self.config.use_nonparanormal:
                r, _ = pearsonr(X[:, i], X[:, j])
            else:
                r, _ = spearmanr(X[:, i], X[:, j])
            z = 0.5 * np.log((1 + r + 1e-10) / (1 - r + 1e-10))
            se = 1.0 / np.sqrt(n - 3)
            p_value = 2 * (1 - norm.cdf(abs(z) / se))
            return r, p_value

        X_S = X[:, list(S)]
        model_i = Ridge(alpha=0.01).fit(X_S, X[:, i])
        res_i = X[:, i] - model_i.predict(X_S)
        model_j = Ridge(alpha=0.01).fit(X_S, X[:, j])
        res_j = X[:, j] - model_j.predict(X_S)

        if self.config.use_nonparanormal:
            r, _ = pearsonr(res_i, res_j)
        else:
            r, _ = spearmanr(res_i, res_j)
        z = 0.5 * np.log((1 + r + 1e-10) / (1 - r + 1e-10))
        df = n - len(S) - 3
        se = 1.0 / np.sqrt(max(df, 1))
        p_value = 2 * (1 - norm.cdf(abs(z) / se))

        return r, p_value

    def learn_skeleton(self, X: np.ndarray, var_names: List[str]) -> Dict:
        """
        骨架学习

        优先使用 p-adic PACD，否则回退到简化版
        """
        if self._use_pacd:
            print("  使用 p-adic PACD 结构学习")
            pacd_config = self._PACDConfig(
                p=2,
                m=4,
                alpha=self.config.alpha_ci,
                max_k=self.config.max_k,
                ci_method="pearson" if self.config.use_nonparanormal else "spearman",
                use_nonparanormal=self.config.use_nonparanormal,
            )
            learner = self._PACDLearner(pacd_config)
            result = learner.learn(X, var_names)

            self.sepsets_ = {}
            var_idx = {name: i for i, name in enumerate(var_names)}
            for key, sep in result["sepsets"].items():
                parts = key.split("|")
                if len(parts) == 2:
                    i = var_idx.get(parts[0])
                    j = var_idx.get(parts[1])
                    if i is not None and j is not None:
                        self.sepsets_[(i, j)] = [
                            var_idx[s] for s in sep if s in var_idx
                        ]

            return {
                "skeleton": [
                    (edge["source"], edge["target"])
                    for edge in result["directed_edges"]
                ],
                "skeleton_indices": result["skeleton"]["skeleton_indices"],
                "sepsets": result["sepsets"],
                "edge_strengths": {},
                "n_edges": result["n_edges"],
                "directed_edges": result["directed_edges"],
            }

        print("  使用简化版骨架学习 (p-adic 模块不可用)")
        return self._learn_skeleton_simple(X, var_names)

    def _learn_skeleton_simple(self, X: np.ndarray, var_names: List[str]) -> Dict:
        """简化版骨架学习"""
        _, d = X.shape

        if self.config.use_nonparanormal:
            X = self.nonparanormal_transform(X)

        edges = {(i, j) for i in range(d) for j in range(i + 1, d)}

        for k in range(self.config.max_k + 1):
            to_remove = []

            for (i, j) in list(edges):
                neighbors_i = {
                    v
                    for (a, b) in edges
                    if a == i or b == i
                    for v in (a, b)
                    if v not in (i, j)
                }
                neighbors_j = {
                    v
                    for (a, b) in edges
                    if a == j or b == j
                    for v in (a, b)
                    if v not in (i, j)
                }
                neighbors = neighbors_i | neighbors_j

                if len(neighbors) < k:
                    continue

                for S in combinations(neighbors, k):
                    S_list = list(S)
                    _, p = self.partial_correlation_test(X, i, j, S_list)

                    if p > self.config.alpha_ci:
                        to_remove.append((i, j))
                        self.sepsets_[(i, j)] = S_list
                        self.sepsets_[(j, i)] = S_list
                        break

                if (i, j) not in to_remove:
                    _, p = self.partial_correlation_test(X, i, j, [])
                    self.edge_pvalues_[(i, j)] = p

            for e in to_remove:
                edges.discard(e)

        self.skeleton_ = edges

        result = {
            "skeleton": [(var_names[i], var_names[j]) for (i, j) in edges],
            "skeleton_indices": list(edges),
            "sepsets": {},
            "edge_strengths": {},
            "n_edges": len(edges),
        }

        for (i, j), S in self.sepsets_.items():
            key = f"{var_names[i]}|{var_names[j]}"
            result["sepsets"][key] = [var_names[s] for s in S]

        for (i, j), p in self.edge_pvalues_.items():
            key = f"{var_names[i]}--{var_names[j]}"
            result["edge_strengths"][key] = 1 - p

        return result


# ============================================================
# Step 2: 干预定向
# ============================================================


class InterventionDirector:
    """
    利用9个干预环境进行定向
    - 比较干预前后的效应
    - 增强删边证据
    """

    def __init__(self, config: PipelineConfig, intervention_map: Dict):
        self.config = config
        self.intervention_map = intervention_map
        self.baseline_conditions = ["CD3CD28", "CD3CD28+ICAM2"]
        self.direction_evidence_: Dict[str, Dict] = {}

    def get_intervention_effect(
        self, data: pd.DataFrame, source: str, target: str
    ) -> Dict:
        """
        计算干预source对target的效应
        """
        int_envs = [
            condition
            for condition, info in self.intervention_map.items()
            if source in info.get("targets", [])
        ]

        if not int_envs:
            return {
                "effect": 0,
                "p_value": 1.0,
                "n_environments": 0,
                "evidence": "none",
            }

        baseline = data[data["COND"].isin(self.baseline_conditions)]
        if len(baseline) < 20:
            return {
                "effect": 0,
                "p_value": 1.0,
                "n_environments": 0,
                "evidence": "insufficient",
            }

        effects = []
        p_values = []

        for env in int_envs:
            int_data = data[data["COND"] == env]
            if len(int_data) < 20:
                continue

            try:
                _, p = mannwhitneyu(
                    baseline[target], int_data[target], alternative="two-sided"
                )
                effect = np.mean(int_data[target]) - np.mean(baseline[target])
                effects.append(effect)
                p_values.append(p)
            except Exception:
                continue

        if not effects:
            return {
                "effect": 0,
                "p_value": 1.0,
                "n_environments": 0,
                "evidence": "none",
            }

        avg_effect = np.mean(effects)
        combined_p = 1 - stats.chi2.cdf(
            -2 * np.sum(np.log(np.array(p_values) + 1e-10)),
            2 * len(p_values),
        )

        evidence = (
            "strong"
            if combined_p < 0.01
            else ("moderate" if combined_p < 0.05 else "weak")
        )

        return {
            "effect": float(avg_effect),
            "p_value": float(combined_p),
            "n_environments": len(effects),
            "evidence": evidence,
        }

    def orient_edges(
        self, data: pd.DataFrame, skeleton: List[Tuple[str, str]]
    ) -> List[Dict]:
        """对骨架中的边进行定向"""
        directed_edges = []

        for (v1, v2) in skeleton:
            effect_1to2 = self.get_intervention_effect(data, v1, v2)
            effect_2to1 = self.get_intervention_effect(data, v2, v1)

            if effect_1to2["evidence"] != "none" and effect_2to1["evidence"] == "none":
                source, target = v1, v2
                confidence = "high"
                evidence = effect_1to2
            elif (
                effect_2to1["evidence"] != "none"
                and effect_1to2["evidence"] == "none"
            ):
                source, target = v2, v1
                confidence = "high"
                evidence = effect_2to1
            elif (
                effect_1to2["evidence"] != "none"
                and effect_2to1["evidence"] != "none"
            ):
                if abs(effect_1to2["effect"]) > abs(effect_2to1["effect"]):
                    source, target = v1, v2
                    evidence = effect_1to2
                else:
                    source, target = v2, v1
                    evidence = effect_2to1
                confidence = "medium"
            else:
                source, target = (v1, v2) if v1 < v2 else (v2, v1)
                confidence = "low"
                evidence = {
                    "effect": 0,
                    "p_value": 1.0,
                    "n_environments": 0,
                    "evidence": "none",
                }

            directed_edges.append(
                {
                    "source": source,
                    "target": target,
                    "direction_confidence": confidence,
                    "intervention_effect": evidence["effect"],
                    "intervention_p": evidence["p_value"],
                    "intervention_evidence": evidence["evidence"],
                    "n_intervention_envs": evidence["n_environments"],
                }
            )

            self.direction_evidence_[f"{source}->{target}"] = evidence

        return directed_edges


# ============================================================
# Step 3: IVAPCI 效应估计
# ============================================================


class IVAPCIEffectEstimator:
    """效应估计器"""

    def __init__(self, config: PipelineConfig, models: Dict):
        self.config = config
        self.models = models

        self._pacdt_available = False
        try:
            from model_wrapper import is_pacd_available, estimate_ate_pacd

            if is_pacd_available():
                self._pacdt_available = True
                self._estimate_pacd = estimate_ate_pacd
        except ImportError:
            pass

    def estimate_direct_effect(
        self,
        data: pd.DataFrame,
        source: str,
        target: str,
        mediators: List[str],
        all_vars: List[str],
        use_pacd: bool = False,
    ) -> Dict:
        """估计直接效应（控制中介变量后）"""
        n = len(data)

        A = (data[source].values > np.median(data[source].values)).astype(np.float32)
        Y = data[target].values.astype(np.float32)

        if use_pacd and self._pacdt_available:
            try:
                X_all = data[all_vars].values.astype(np.float32)
                result = self._estimate_pacd(
                    X_all,
                    A,
                    Y,
                    epochs=150,
                    device=self.config.ivapci_device,
                    n_bootstrap=self.config.n_bootstrap,
                )
                return {
                    "tau": result["ate"],
                    "se": result["se"],
                    "ci": result["ci"],
                    "p_value": result["p_value"],
                    "method": "pacd-t",
                }
            except Exception as exc:
                print(f"    PACD-T失败: {exc}")

        x_vars = [source]
        w_vars = mediators if mediators else []
        z_vars = [v for v in all_vars if v not in [source, target] + w_vars]

        if not z_vars:
            z_vars = [source]

        X_block = data[x_vars].values.astype(np.float32)
        W_block = (
            data[w_vars].values.astype(np.float32)
            if w_vars
            else np.zeros((n, 1), dtype=np.float32)
        )
        Z_block = data[z_vars].values.astype(np.float32)

        V_all = np.hstack([X_block, W_block, Z_block])

        if self.models.get("estimate_ivapci"):
            try:
                result = self.models["estimate_ivapci"](
                    V_all,
                    A,
                    Y,
                    x_dim=X_block.shape[1],
                    w_dim=W_block.shape[1],
                    z_dim=Z_block.shape[1],
                    epochs=self.config.ivapci_epochs,
                    device=self.config.ivapci_device,
                    n_bootstrap=self.config.n_bootstrap,
                )
                return {
                    "tau": result["ate"],
                    "se": result["se"],
                    "ci": result["ci"],
                    "p_value": result["p_value"],
                    "method": "ivapci",
                    "diagnostics": result.get("diagnostics", {}),
                }
            except Exception:
                pass

        return self._simple_dr_estimate(data, source, target, w_vars)

    def estimate_total_effect(
        self,
        data: pd.DataFrame,
        source: str,
        target: str,
        all_vars: List[str],
        use_pacd: bool = False,
    ) -> Dict:
        """估计总效应（不控制中介）"""
        return self.estimate_direct_effect(
            data, source, target, [], all_vars, use_pacd=use_pacd
        )

    def _simple_dr_estimate(
        self, data: pd.DataFrame, source: str, target: str, covariates: List[str]
    ) -> Dict:
        """简化DR估计"""
        n = len(data)
        A = (data[source].values > np.median(data[source].values)).astype(int)
        Y = data[target].values

        if covariates:
            W = data[covariates].values
            try:
                ps_model = LogisticRegression(max_iter=1000, solver="lbfgs")
                ps_model.fit(W, A)
                ps = np.clip(ps_model.predict_proba(W)[:, 1], 0.05, 0.95)
            except Exception:
                ps = np.full(n, 0.5)
        else:
            ps = np.full(n, 0.5)

        w1 = A / ps
        w0 = (1 - A) / (1 - ps)
        mu1 = np.sum(w1 * Y) / (np.sum(w1) + 1e-10)
        mu0 = np.sum(w0 * Y) / (np.sum(w0) + 1e-10)
        tau = mu1 - mu0

        taus = []
        for _ in range(self.config.n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            A_b, Y_b, ps_b = A[idx], Y[idx], ps[idx]
            w1_b = A_b / ps_b
            w0_b = (1 - A_b) / (1 - ps_b)
            tau_b = np.sum(w1_b * Y_b) / (np.sum(w1_b) + 1e-10) - np.sum(
                w0_b * Y_b
            ) / (np.sum(w0_b) + 1e-10)
            taus.append(tau_b)

        se = np.std(taus)
        ci = [np.percentile(taus, 2.5), np.percentile(taus, 97.5)]
        p_value = 2 * (1 - norm.cdf(abs(tau) / (se + 1e-10)))

        return {
            "tau": float(tau),
            "se": float(se),
            "ci": [float(ci[0]), float(ci[1])],
            "p_value": float(p_value),
            "method": "simple_dr",
        }

    def mediation_analysis(
        self,
        data: pd.DataFrame,
        source: str,
        target: str,
        potential_mediators: List[str],
        all_vars: List[str],
        use_pacd: bool = False,
    ) -> Dict:
        """中介分析"""
        total = self.estimate_total_effect(
            data, source, target, all_vars, use_pacd=use_pacd
        )

        if potential_mediators:
            direct = self.estimate_direct_effect(
                data,
                source,
                target,
                potential_mediators,
                all_vars,
                use_pacd=use_pacd,
            )
        else:
            direct = total

        tau_total = total["tau"]
        tau_direct = direct["tau"]
        tau_indirect = tau_total - tau_direct

        indirect_ratio = abs(tau_indirect) / abs(tau_total) if abs(tau_total) > 1e-6 else 0
        is_mediated = indirect_ratio > (1 - self.config.indirect_ratio)

        return {
            "tau_total": tau_total,
            "tau_direct": tau_direct,
            "tau_indirect": tau_indirect,
            "indirect_ratio": indirect_ratio,
            "is_mediated": is_mediated,
            "total_result": total,
            "direct_result": direct,
            "mediators_controlled": potential_mediators,
        }


# ============================================================
# Step 4: 回灌剪枝
# ============================================================


class GraphPruner:
    """根据IVAPCI结果剪枝"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pruned_edges_: List[Dict] = []
        self.pruning_reasons_: Dict[str, str] = {}

    def prune(self, edges: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """剪枝"""
        kept = []
        pruned = []

        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            key = f"{source}->{target}"

            tau = edge.get("tau_direct", edge.get("tau_total", 0))
            p_value = edge.get("p_direct", edge.get("p_total", 1.0))
            is_mediated = edge.get("is_mediated", False)
            indirect_ratio = edge.get("indirect_ratio", 0)

            prune_reason = None

            if abs(tau) < self.config.effect_threshold:
                prune_reason = (
                    f"weak_effect (|τ|={abs(tau):.2f} < {self.config.effect_threshold})"
                )
            elif p_value > self.config.p_threshold:
                prune_reason = (
                    f"not_significant (p={p_value:.4f} > {self.config.p_threshold})"
                )
            elif is_mediated and indirect_ratio > (1 - self.config.indirect_ratio):
                prune_reason = f"indirect_edge (ratio={indirect_ratio:.2f})"

            if prune_reason:
                edge["prune_reason"] = prune_reason
                pruned.append(edge)
                self.pruning_reasons_[key] = prune_reason
            else:
                kept.append(edge)

        self.pruned_edges_ = pruned
        return kept, pruned


# ============================================================
# 主流水线
# ============================================================


class PACDIVAPCIPipeline:
    """完整的PACD-IVAPCI流水线"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.models = import_models()

        self.skeleton_learner = PACDSkeletonLearner(self.config)
        self.effect_estimator = IVAPCIEffectEstimator(self.config, self.models)
        self.pruner = GraphPruner(self.config)

        self.skeleton_result_ = None
        self.directed_edges_ = []
        self.effect_results_ = []
        self.final_edges_ = []
        self.pruned_edges_ = []

    def run(
        self,
        data: pd.DataFrame,
        intervention_map: Dict,
        var_names: Optional[List[str]] = None,
    ) -> Dict:
        """运行完整流水线"""
        if var_names is None:
            var_names = [c for c in data.columns if c not in ["COND", "INT"]]

        print("=" * 70)
        print("PACD-IVAPCI 完整流水线")
        print("=" * 70)
        print(f"数据: {len(data)} 样本, {len(var_names)} 变量")
        print(f"效应估计器: {self.config.estimator.upper()}")
        print(f"IVAPCI: {'✓ 可用' if self.models.get('ivapci') else '✗ 不可用'}")
        print(f"PACD-T: {'✓ 可用' if self.models.get('pacd') else '✗ 不可用'}")

        print("\n" + "─" * 50)
        print("[Step 1] PACD 结构学习")
        print("─" * 50)

        X = data[var_names].values
        self.skeleton_result_ = self.skeleton_learner.learn_skeleton(X, var_names)

        print(f"  ✓ 发现 {self.skeleton_result_['n_edges']} 条边")
        print(f"  ✓ 保存 {len(self.skeleton_result_['sepsets'])} 个分离集")

        pacd_directed = self.skeleton_result_.get("directed_edges", None)

        print("\n" + "─" * 50)
        print("[Step 2] 干预定向 (9环境)")
        print("─" * 50)

        if "COND" in data.columns and intervention_map:
            director = InterventionDirector(self.config, intervention_map)

            if pacd_directed:
                print("  结合 p-adic 方向识别 + 干预证据")
                self.directed_edges_ = []

                for edge in pacd_directed:
                    v1, v2 = edge["source"], edge["target"]

                    int_evidence = director.get_intervention_effect(data, v1, v2)
                    int_evidence_rev = director.get_intervention_effect(data, v2, v1)

                    pacd_conf = edge.get("direction_confidence", "low")

                    if int_evidence["evidence"] in ["strong", "moderate"]:
                        final_conf = "high"
                    elif int_evidence_rev["evidence"] in ["strong", "moderate"]:
                        v1, v2 = v2, v1
                        final_conf = "high"
                    else:
                        final_conf = pacd_conf

                    self.directed_edges_.append(
                        {
                            "source": v1,
                            "target": v2,
                            "direction_confidence": final_conf,
                            "pacd_confidence": pacd_conf,
                            "intervention_effect": int_evidence["effect"],
                            "intervention_p": int_evidence["p_value"],
                            "intervention_evidence": int_evidence["evidence"],
                            "n_intervention_envs": int_evidence["n_environments"],
                            "sepset": edge.get("sepset", []),
                        }
                    )
            else:
                self.directed_edges_ = director.orient_edges(
                    data, self.skeleton_result_["skeleton"]
                )

            high_conf = sum(
                1 for e in self.directed_edges_ if e["direction_confidence"] == "high"
            )
            med_conf = sum(
                1
                for e in self.directed_edges_
                if e["direction_confidence"] == "medium"
            )
            print(f"  ✓ 高置信定向: {high_conf} 条")
            print(f"  ✓ 中置信定向: {med_conf} 条")
        else:
            if pacd_directed:
                self.directed_edges_ = pacd_directed
                print("  ⚠️ 无干预信息，使用 p-adic 方向识别")
            else:
                self.directed_edges_ = [
                    {
                        "source": v1 if v1 < v2 else v2,
                        "target": v2 if v1 < v2 else v1,
                        "direction_confidence": "low",
                        "intervention_effect": 0,
                        "intervention_evidence": "none",
                        "sepset": self.skeleton_result_["sepsets"].get(
                            f"{v1}|{v2}",
                            self.skeleton_result_["sepsets"].get(f"{v2}|{v1}", []),
                        ),
                    }
                    for (v1, v2) in self.skeleton_result_["skeleton"]
                ]
                print("  ⚠️ 无干预信息，使用默认定向")

        print("\n" + "─" * 50)
        print("[Step 3] IVAPCI 效应估计")
        print("─" * 50)

        sorted_edges = sorted(
            self.directed_edges_,
            key=lambda e: abs(e.get("intervention_effect", 0)),
            reverse=True,
        )

        self.effect_results_ = []

        for idx, edge in enumerate(sorted_edges):
            source = edge["source"]
            target = edge["target"]

            potential_mediators = [
                e["target"]
                for e in self.directed_edges_
                if e["source"] == source and e["target"] != target
            ]

            use_deep = idx < self.config.top_k_ivapci
            use_pacd = self.config.estimator == "pacd"

            if use_deep:
                if self.config.estimator == "pacd":
                    method_hint = "(PACD-T)"
                elif self.config.estimator == "ivapci":
                    method_hint = "(IVAPCI)"
                else:
                    method_hint = "(DR)"
            else:
                method_hint = "(DR)"

            print(f"  [{idx + 1}/{len(sorted_edges)}] {source} → {target} {method_hint}")

            mediation = self.effect_estimator.mediation_analysis(
                data,
                source,
                target,
                potential_mediators,
                var_names,
                use_pacd=(use_deep and use_pacd),
            )

            result = {
                **edge,
                "tau_total": mediation["tau_total"],
                "tau_direct": mediation["tau_direct"],
                "tau_indirect": mediation["tau_indirect"],
                "indirect_ratio": mediation["indirect_ratio"],
                "is_mediated": mediation["is_mediated"],
                "se_total": mediation["total_result"]["se"],
                "se_direct": mediation["direct_result"]["se"],
                "ci_total": mediation["total_result"]["ci"],
                "ci_direct": mediation["direct_result"]["ci"],
                "p_total": mediation["total_result"]["p_value"],
                "p_direct": mediation["direct_result"]["p_value"],
                "method": mediation["direct_result"]["method"],
                "mediators_controlled": mediation["mediators_controlled"],
                "sepset": self.skeleton_result_["sepsets"].get(
                    f"{source}|{target}",
                    self.skeleton_result_["sepsets"].get(f"{target}|{source}", []),
                ),
            }

            self.effect_results_.append(result)

            sig = (
                "***"
                if result["p_direct"] < 0.001
                else ("**" if result["p_direct"] < 0.01 else "")
            )
            med_flag = " [间接]" if result["is_mediated"] else ""
            print(f"      τ_direct={result['tau_direct']:>7.2f} {sig}{med_flag}")

        print("\n" + "─" * 50)
        print("[Step 4] 回灌剪枝")
        print("─" * 50)

        self.final_edges_, self.pruned_edges_ = self.pruner.prune(self.effect_results_)

        print(f"  ✓ 保留 {len(self.final_edges_)} 条边")
        print(f"  ✗ 剪掉 {len(self.pruned_edges_)} 条边")

        if self.pruned_edges_:
            print("\n  剪枝原因统计:")
            reasons = {}
            for edge in self.pruned_edges_:
                reason = edge.get("prune_reason", "unknown").split()[0]
                reasons[reason] = reasons.get(reason, 0) + 1
            for reason, count in reasons.items():
                print(f"    - {reason}: {count}")

        print("\n" + "=" * 70)
        print("完成!")
        print("=" * 70)
        print(f"  骨架边数: {self.skeleton_result_['n_edges']}")
        print(f"  最终边数: {len(self.final_edges_)}")
        print(f"  剪枝边数: {len(self.pruned_edges_)}")

        return {
            "skeleton": self.skeleton_result_,
            "directed_edges": self.directed_edges_,
            "effect_results": self.effect_results_,
            "final_edges": self.final_edges_,
            "pruned_edges": self.pruned_edges_,
        }

    def save_results(self, output_dir: str) -> None:
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)

        df_final = pd.DataFrame(self.final_edges_)
        df_final.to_csv(f"{output_dir}/final_edges.csv", index=False)

        df_pruned = pd.DataFrame(self.pruned_edges_)
        df_pruned.to_csv(f"{output_dir}/pruned_edges.csv", index=False)

        viz_data = {
            "nodes": list(
                set([e["source"] for e in self.final_edges_]
                    + [e["target"] for e in self.final_edges_])
            ),
            "edges": [
                {
                    "source": e["source"],
                    "target": e["target"],
                    "weight": e["tau_direct"],
                    "ci": e["ci_direct"],
                    "p_value": e["p_direct"],
                    "direction_confidence": e.get("direction_confidence", "medium"),
                    "is_mediated": e.get("is_mediated", False),
                    "sepset": e.get("sepset", []),
                }
                for e in self.final_edges_
            ],
            "pruned_edges": [
                {
                    "source": e["source"],
                    "target": e["target"],
                    "reason": e.get("prune_reason", "unknown"),
                }
                for e in self.pruned_edges_
            ],
            "sepsets": self.skeleton_result_["sepsets"],
        }

        with open(f"{output_dir}/causal_graph.json", "w", encoding="utf-8") as handle:
            json.dump(viz_data, handle, indent=2, default=str)

        self._write_report(f"{output_dir}/analysis_report.md")

        print(f"\n结果已保存到: {output_dir}/")
        print("  - final_edges.csv: 最终因果边")
        print("  - pruned_edges.csv: 被剪掉的边")
        print("  - causal_graph.json: 可视化数据")
        print("  - analysis_report.md: 分析报告")

    def _write_report(self, filepath: str) -> None:
        """生成Markdown报告"""
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write("# PACD-IVAPCI 因果发现报告\n\n")

            handle.write("## 1. 概要\n\n")
            handle.write(f"- 骨架边数: {self.skeleton_result_['n_edges']}\n")
            handle.write(f"- 最终边数: {len(self.final_edges_)}\n")
            handle.write(f"- 剪枝边数: {len(self.pruned_edges_)}\n\n")

            handle.write("## 2. Top 10 因果效应\n\n")
            handle.write("| 排名 | 因果边 | τ_direct | CI | p-value | 方向置信 |\n")
            handle.write("|------|--------|----------|-------|---------|----------|\n")

            sorted_edges = sorted(
                self.final_edges_, key=lambda e: abs(e["tau_direct"]), reverse=True
            )
            for i, edge in enumerate(sorted_edges[:10], 1):
                sig = "***" if edge["p_direct"] < 0.001 else ""
                handle.write(
                    f"| {i} | {edge['source']} → {edge['target']} | "
                    f"{edge['tau_direct']:.2f} {sig} | "
                    f"[{edge['ci_direct'][0]:.1f}, {edge['ci_direct'][1]:.1f}] | "
                    f"{edge['p_direct']:.4f} | {edge.get('direction_confidence', '-')} |\n"
                )

            handle.write("\n## 3. 被剪掉的边\n\n")
            if self.pruned_edges_:
                handle.write("| 边 | 原因 |\n")
                handle.write("|-----|------|\n")
                for edge in self.pruned_edges_:
                    handle.write(
                        f"| {edge['source']} → {edge['target']} | "
                        f"{edge.get('prune_reason', '-')} |\n"
                    )
            else:
                handle.write("无\n")

            handle.write("\n## 4. 分离集 (条件独立证据)\n\n")
            handle.write("以下变量对在给定分离集后条件独立:\n\n")
            for key, sepset in list(self.skeleton_result_["sepsets"].items())[:10]:
                if sepset:
                    handle.write(f"- `{key}` | {', '.join(sepset)}\n")


# ============================================================
# 命令行入口
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="PACD-IVAPCI 完整流水线")
    parser.add_argument("--data", "-d", required=True, help="输入CSV文件")
    parser.add_argument("--output", "-o", default="./results", help="输出目录")
    parser.add_argument("--intervention", "-i", default=None, help="干预映射JSON")

    parser.add_argument("--alpha", type=float, default=0.001, help="CI检验显著性")
    parser.add_argument("--max-k", type=int, default=3, help="最大条件集大小")

    parser.add_argument(
        "--estimator",
        choices=["ivapci", "pacd", "simple"],
        default="ivapci",
        help="效应估计器 (ivapci/pacd/simple)",
    )
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--device", default="cpu", help="计算设备")
    parser.add_argument("--top-k", type=int, default=30, help="深度学习估计的边数")

    parser.add_argument("--effect-threshold", type=float, default=5.0, help="效应阈值")
    parser.add_argument("--p-threshold", type=float, default=0.05, help="显著性阈值")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"错误: 找不到 {args.data}")
        sys.exit(1)

    print(f"加载数据: {args.data}")
    data = pd.read_csv(args.data)
    print(f"数据维度: {data.shape}")

    intervention_map = {}
    if args.intervention and os.path.exists(args.intervention):
        with open(args.intervention, encoding="utf-8") as handle:
            intervention_map = json.load(handle)
        print(f"加载干预映射: {len(intervention_map)} 个条件")

    config = PipelineConfig(
        alpha_ci=args.alpha,
        max_k=args.max_k,
        estimator=args.estimator,
        ivapci_epochs=args.epochs,
        ivapci_device=args.device,
        top_k_ivapci=args.top_k,
        effect_threshold=args.effect_threshold,
        p_threshold=args.p_threshold,
    )

    pipeline = PACDIVAPCIPipeline(config)
    pipeline.run(data, intervention_map)
    pipeline.save_results(args.output)


if __name__ == "__main__":
    main()
