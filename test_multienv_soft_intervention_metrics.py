#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-environment (Sachs-like 9 条件) + 结构学习评估扩展指标
========================================================

你提的“把指标从 skeleton 扩展到：
- 方向准确率
- 边权误差 |τ_hat - τ_true|
- 间接边识别 AUC
并且加入 Sachs 那种 9 环境软干预(soft intervention)
”——这个脚本就是一个可直接跑的合成基准。

核心思想
--------
1) 用一个 11 节点 (Raf, Mek, Plcg, PIP2, PIP3, Erk, Akt, PKA, PKC, P38, Jnk) 的 SCM 生成数据
2) 生成 9 个环境:
   - CD3CD28 (baseline)
   - CD3CD28+ICAM2 (baseline2 / 轻微激活)
   - CD3CD28+U0126      (MEK 抑制)
   - CD3CD28+AktInhib   (AKT 抑制)
   - CD3CD28+G0076      (PKC 抑制)
   - CD3CD28+Psitect    (PLCγ 抑制, 这里用作示例)
   - CD3CD28+LY         (PI3K 抑制, 这里等效为 PIP3 生成受抑)
   - PMA                (PKC 激活)
   - B2CAMP             (PKA 激活)
   这些都是“软干预”：不把变量 do() 固定到常数，而是改变其结构方程中的强度/偏置/噪声。
3) 用你的 PACD-IVAPCI 流水线(可选)输出：
   skeleton + 干预定向 + (τ_total, τ_direct, indirect_ratio)
4) 计算扩展指标:
   - skeleton: precision/recall/F1/SHD
   - direction: orient_acc / orient_recall
   - edge weight: MAE(|τ_hat - τ_true|)
   - indirect AUC: 用 indirect_ratio 作为 score，区分“真间接边(存在祖先路径但无直接边)” vs “无因果路径”

运行
----
# 放在你 IVAPCI 项目根目录(能 import run_pacd_ivapci_pipeline.py)：
python test_multienv_soft_intervention_metrics.py --n 2000 --seed 42 --use_pipeline 1 --estimator simple

# 如果你想用 IVAPCI / PACD-T 做 τ 估计（会更慢）：
python test_multienv_soft_intervention_metrics.py --use_pipeline 1 --estimator ivapci --ivapci_epochs 80 --top_k 20

备注
----
- τ_true 的定义：对每条真边 X->Y，在 baseline 环境下做 do(X=q75) vs do(X=q25)
  的 Monte-Carlo 差 (E[Y|do(X=hi)]-E[Y|do(X=lo)])。

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


# ------------------------------
# 1) Ground-truth DAG / SCM
# ------------------------------

NODES = ["Plcg", "PIP2", "PIP3", "PKA", "PKC", "Raf", "Mek", "Erk", "Akt", "Jnk", "P38"]

# 一个“合理且可控”的 11 节点 DAG（示例用）
TRUE_EDGES: List[Tuple[str, str]] = [
    ("Plcg", "PIP2"),
    ("PIP2", "PIP3"),
    ("Plcg", "PKC"),
    ("PKC", "Raf"),
    ("PKA", "Raf"),   # 抑制作用用负系数表达
    ("Raf", "Mek"),
    ("Mek", "Erk"),
    ("PIP3", "Akt"),
    ("PKC", "Jnk"),
    ("PKC", "P38"),
    ("Erk", "P38"),
]

TRUE_SKELETON: Set[Tuple[str, str]] = set((min(a, b), max(a, b)) for a, b in TRUE_EDGES)


@dataclass
class EnvCfg:
    """soft intervention config"""
    name: str
    stim: float = 1.0  # 全局刺激强度（CD3/CD28 类）
    scale: Dict[str, float] = None   # 对某些节点整体缩放
    bias: Dict[str, float] = None    # 对某些节点加入偏置
    edge_scale: Dict[Tuple[str, str], float] = None  # 对某条边的系数缩放

    def __post_init__(self):
        self.scale = self.scale or {}
        self.bias = self.bias or {}
        self.edge_scale = self.edge_scale or {}


def simulate_scm(n: int, env: EnvCfg, seed: int, do: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    生成一个环境的数据。
    do: {var: constant} 表示 do(var=constant)，用于计算 tau_true。
    """
    rng = np.random.default_rng(seed)
    do = do or {}

    # exogenous noises
    eps = {v: rng.normal(0, 1.0, size=n).astype(np.float32) for v in NODES}

    # helper: edge coefficient with env scaling
    def coef(src: str, tgt: str, base: float) -> float:
        return float(base) * float(env.edge_scale.get((src, tgt), 1.0))

    V: Dict[str, np.ndarray] = {}

    # Topological order consistent with TRUE_EDGES
    # Plcg
    if "Plcg" in do:
        V["Plcg"] = np.full(n, float(do["Plcg"]), dtype=np.float32)
    else:
        V["Plcg"] = (0.9 * env.stim + eps["Plcg"] * 0.8).astype(np.float32)

    # PIP2
    if "PIP2" in do:
        V["PIP2"] = np.full(n, float(do["PIP2"]), dtype=np.float32)
    else:
        V["PIP2"] = (coef("Plcg", "PIP2", 0.9) * V["Plcg"] + 0.2 * env.stim + eps["PIP2"] * 0.8).astype(np.float32)

    # PIP3
    if "PIP3" in do:
        V["PIP3"] = np.full(n, float(do["PIP3"]), dtype=np.float32)
    else:
        lin = coef("PIP2", "PIP3", 1.0) * V["PIP2"] + 0.2 * env.stim + eps["PIP3"] * 0.8
        V["PIP3"] = (np.tanh(lin)).astype(np.float32)

    # PKA
    if "PKA" in do:
        V["PKA"] = np.full(n, float(do["PKA"]), dtype=np.float32)
    else:
        V["PKA"] = (0.4 * env.stim + eps["PKA"] * 0.8).astype(np.float32)

    # PKC
    if "PKC" in do:
        V["PKC"] = np.full(n, float(do["PKC"]), dtype=np.float32)
    else:
        lin = coef("Plcg", "PKC", 1.0) * V["Plcg"] + 0.2 * env.stim + eps["PKC"] * 0.8
        V["PKC"] = np.tanh(lin).astype(np.float32)

    # Raf
    if "Raf" in do:
        V["Raf"] = np.full(n, float(do["Raf"]), dtype=np.float32)
    else:
        lin = coef("PKC", "Raf", 1.0) * V["PKC"] + coef("PKA", "Raf", -0.9) * V["PKA"] + 0.1 * env.stim + eps["Raf"] * 0.8
        V["Raf"] = np.tanh(lin).astype(np.float32)

    # Mek
    if "Mek" in do:
        V["Mek"] = np.full(n, float(do["Mek"]), dtype=np.float32)
    else:
        lin = coef("Raf", "Mek", 1.2) * V["Raf"] + 0.05 * env.stim + eps["Mek"] * 0.8
        V["Mek"] = np.tanh(lin).astype(np.float32)

    # Erk
    if "Erk" in do:
        V["Erk"] = np.full(n, float(do["Erk"]), dtype=np.float32)
    else:
        lin = coef("Mek", "Erk", 1.3) * V["Mek"] + 0.05 * env.stim + eps["Erk"] * 0.8
        V["Erk"] = np.tanh(lin).astype(np.float32)

    # Akt
    if "Akt" in do:
        V["Akt"] = np.full(n, float(do["Akt"]), dtype=np.float32)
    else:
        lin = coef("PIP3", "Akt", 1.2) * V["PIP3"] + 0.05 * env.stim + eps["Akt"] * 0.8
        V["Akt"] = np.tanh(lin).astype(np.float32)

    # Jnk
    if "Jnk" in do:
        V["Jnk"] = np.full(n, float(do["Jnk"]), dtype=np.float32)
    else:
        lin = coef("PKC", "Jnk", 1.0) * V["PKC"] + 0.1 * env.stim + eps["Jnk"] * 0.9
        V["Jnk"] = np.tanh(lin).astype(np.float32)

    # P38
    if "P38" in do:
        V["P38"] = np.full(n, float(do["P38"]), dtype=np.float32)
    else:
        lin = coef("PKC", "P38", 0.9) * V["PKC"] + coef("Erk", "P38", 0.8) * V["Erk"] + 0.05 * env.stim + eps["P38"] * 0.9
        V["P38"] = np.tanh(lin).astype(np.float32)

    # Apply env node-level scale/bias
    for v in NODES:
        if v in do:
            continue
        if v in env.scale:
            V[v] = (V[v] * float(env.scale[v])).astype(np.float32)
        if v in env.bias:
            V[v] = (V[v] + float(env.bias[v])).astype(np.float32)

    df = pd.DataFrame({v: V[v] for v in NODES})
    df["COND"] = env.name
    df["INT"] = 1 if env.name not in ("CD3CD28", "CD3CD28+ICAM2") else 0
    return df


def build_9_envs() -> Tuple[List[EnvCfg], Dict]:
    """
    返回：
    - envs: 9 环境配置
    - intervention_map: 给你的流水线 Step2(干预定向) 用：cond -> targets
    """
    envs = [
        EnvCfg("CD3CD28", stim=1.0),
        EnvCfg("CD3CD28+ICAM2", stim=1.1, bias={"Plcg": 0.15, "PKC": 0.10}),  # 轻微增强上游
        EnvCfg("CD3CD28+U0126", stim=1.0, scale={"Mek": 0.2}, edge_scale={("Mek", "Erk"): 0.2}),
        EnvCfg("CD3CD28+AktInhib", stim=1.0, scale={"Akt": 0.2}),
        EnvCfg("CD3CD28+G0076", stim=1.0, scale={"PKC": 0.2}),
        EnvCfg("CD3CD28+Psitect", stim=1.0, scale={"Plcg": 0.25}),
        EnvCfg("CD3CD28+LY", stim=1.0, edge_scale={("PIP2", "PIP3"): 0.25}, scale={"PIP3": 0.5}),
        EnvCfg("PMA", stim=1.0, bias={"PKC": 0.6}),
        EnvCfg("B2CAMP", stim=1.0, bias={"PKA": 0.6}),
    ]

    intervention_map = {
        "CD3CD28": {"targets": []},
        "CD3CD28+ICAM2": {"targets": ["Plcg", "PKC"]},
        "CD3CD28+U0126": {"targets": ["Mek"]},
        "CD3CD28+AktInhib": {"targets": ["Akt"]},
        "CD3CD28+G0076": {"targets": ["PKC"]},
        "CD3CD28+Psitect": {"targets": ["Plcg"]},
        "CD3CD28+LY": {"targets": ["PIP3"]},
        "PMA": {"targets": ["PKC"]},
        "B2CAMP": {"targets": ["PKA"]},
    }
    return envs, intervention_map


# ------------------------------
# 2) Ground truth helpers
# ------------------------------

def compute_ancestors(edges: List[Tuple[str, str]], nodes: List[str]) -> Dict[Tuple[str, str], bool]:
    """reachability: is src ancestor of tgt? (including length>=1 path)"""
    idx = {v: i for i, v in enumerate(nodes)}
    d = len(nodes)
    reach = np.zeros((d, d), dtype=bool)
    for a, b in edges:
        reach[idx[a], idx[b]] = True
    # Floyd-Warshall
    for k in range(d):
        reach = reach | (reach[:, [k]] & reach[[k], :])
    return {(a, b): bool(reach[idx[a], idx[b]]) for a in nodes for b in nodes if a != b}


def tau_true_for_edge(src: str, tgt: str, n_mc: int, base_env: EnvCfg, seed: int, qlo: float, qhi: float) -> float:
    """τ_true = E[Y|do(X=qhi)] - E[Y|do(X=qlo)] under baseline env"""
    df_hi = simulate_scm(n_mc, base_env, seed=seed + 11, do={src: qhi})
    df_lo = simulate_scm(n_mc, base_env, seed=seed + 29, do={src: qlo})
    return float(df_hi[tgt].mean() - df_lo[tgt].mean())


# ------------------------------
# 3) Metrics
# ------------------------------

def skeleton_metrics(pred_edges_undirected: Set[Tuple[str, str]], true_edges_undirected: Set[Tuple[str, str]]) -> Dict[str, float]:
    tp = len(pred_edges_undirected & true_edges_undirected)
    fp = len(pred_edges_undirected - true_edges_undirected)
    fn = len(true_edges_undirected - pred_edges_undirected)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    shd = fp + fn
    return {"precision": prec, "recall": rec, "f1": f1, "shd": float(shd), "tp": tp, "fp": fp, "fn": fn}


def direction_metrics(pred_directed: Set[Tuple[str, str]], true_directed: Set[Tuple[str, str]]) -> Dict[str, float]:
    true_skel = set((min(a, b), max(a, b)) for a, b in true_directed)
    eligible = {(a, b) for (a, b) in pred_directed if (min(a, b), max(a, b)) in true_skel}
    correct = sum(1 for e in eligible if e in true_directed)
    orient_acc = correct / len(eligible) if eligible else 0.0
    orient_recall = correct / len(true_directed) if true_directed else 0.0
    return {"orient_acc": orient_acc, "orient_recall": orient_recall, "n_oriented_on_true_skel": len(eligible)}


def edge_weight_mae(pred_edge_taus: Dict[Tuple[str, str], float], true_edge_taus: Dict[Tuple[str, str], float]) -> Dict[str, float]:
    common = [e for e in pred_edge_taus.keys() if e in true_edge_taus]
    if not common:
        return {"mae": float("nan"), "n": 0}
    errs = [abs(float(pred_edge_taus[e]) - float(true_edge_taus[e])) for e in common]
    return {"mae": float(np.mean(errs)), "n": len(common)}


def indirect_auc(pred_edges: List[Dict], reach: Dict[Tuple[str, str], bool], true_directed: Set[Tuple[str, str]]) -> Dict[str, float]:
    """
    label=1: 有祖先路径 src=>tgt 且 (src,tgt) 不是直接边
    label=0: src 不是 tgt 的祖先 (no causal path)
    score=indirect_ratio (越大越像间接)
    """
    if roc_auc_score is None:
        return {"auc": float("nan"), "n": 0, "note": "sklearn not available"}

    ys, scores = [], []
    for e in pred_edges:
        src, tgt = e["source"], e["target"]
        if (src, tgt) in true_directed:
            continue
        label = 1 if reach.get((src, tgt), False) else 0
        score = float(e.get("indirect_ratio", 0.0))
        ys.append(label)
        scores.append(score)

    if len(set(ys)) < 2:
        return {"auc": float("nan"), "n": len(ys), "note": "only one class in labels"}
    return {"auc": float(roc_auc_score(ys, scores)), "n": len(ys)}


# ------------------------------
# 4) Runner
# ------------------------------

def run_pipeline_if_available(df_long: pd.DataFrame, intervention_map: Dict, estimator: str, ivapci_epochs: int, top_k: int):
    """
    使用你现成的 PACD-IVAPCI 流水线（run_pacd_ivapci_pipeline.py）。
    """
    try:
        from run_pacd_ivapci_pipeline import PACDIVAPCIPipeline, PipelineConfig
    except Exception as e:
        return None, f"import pipeline failed: {e}"

    cfg = PipelineConfig()
    cfg.estimator = estimator
    cfg.ivapci_epochs = int(ivapci_epochs)
    cfg.top_k_ivapci = int(top_k)
    cfg.alpha_ci = 0.01
    cfg.max_k = 3
    cfg.effect_threshold = 0.0
    cfg.p_threshold = 1.0

    pipe = PACDIVAPCIPipeline(cfg)
    _ = pipe.run(df_long, intervention_map, var_names=NODES)
    return pipe, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="每个环境样本量")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_pipeline", type=int, default=1, help="1=用你的PACD-IVAPCI流水线; 0=只生成数据并计算tau_true")
    ap.add_argument("--estimator", type=str, default="simple", choices=["simple", "ivapci", "pacd"], help="流水线中 Step3 的估计器选择")
    ap.add_argument("--ivapci_epochs", type=int, default=80)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--n_mc_tau", type=int, default=8000, help="计算tau_true时的MC样本量")
    args = ap.parse_args()

    envs, intervention_map = build_9_envs()

    # 生成 9 环境 long data
    dfs = []
    for i, env in enumerate(envs):
        dfs.append(simulate_scm(args.n, env, seed=args.seed + 1000 * i))
    df_long = pd.concat(dfs, ignore_index=True)

    # baseline 用于定义 q25/q75
    base_env = envs[0]
    df_base = simulate_scm(5000, base_env, seed=args.seed + 999)
    quantiles = {}
    for v in NODES:
        qlo = float(np.quantile(df_base[v].values, 0.25))
        qhi = float(np.quantile(df_base[v].values, 0.75))
        quantiles[v] = (qlo, qhi)

    # 计算每条真边的 tau_true
    true_edge_tau = {}
    for (src, tgt) in TRUE_EDGES:
        qlo, qhi = quantiles[src]
        true_edge_tau[(src, tgt)] = tau_true_for_edge(src, tgt, n_mc=args.n_mc_tau, base_env=base_env, seed=args.seed, qlo=qlo, qhi=qhi)

    reach = compute_ancestors(TRUE_EDGES, NODES)

    print("=" * 72)
    print("Multi-env soft intervention benchmark (Sachs-like 9 conditions)")
    print("=" * 72)
    print(f"n per env: {args.n}  | total: {len(df_long)} | estimator: {args.estimator}")
    print(f"true edges: {len(TRUE_EDGES)} | nodes: {len(NODES)} | n_mc_tau: {args.n_mc_tau}")
    print("-" * 72)

    if not args.use_pipeline:
        print("use_pipeline=0: 只生成数据与 tau_true（不跑结构学习）")
        return

    pipe, err = run_pipeline_if_available(df_long, intervention_map, args.estimator, args.ivapci_epochs, args.top_k)
    if err is not None:
        print(f"⚠️ {err}")
        print("提示：把本脚本放在能 import run_pacd_ivapci_pipeline.py 的目录里再跑。")
        return

    pred_edges = pipe.effect_results_ if getattr(pipe, "effect_results_", None) is not None else []
    pred_dir = {(e["source"], e["target"]) for e in pred_edges}
    pred_skel = set((min(a, b), max(a, b)) for (a, b) in pred_dir)

    m_skel = skeleton_metrics(pred_skel, TRUE_SKELETON)
    true_dir = set(TRUE_EDGES)
    m_dir = direction_metrics(pred_dir, true_dir)

    pred_tau = {(e["source"], e["target"]): float(e.get("tau_direct", e.get("tau_total", 0.0))) for e in pred_edges}
    m_w = edge_weight_mae(pred_tau, true_edge_tau)

    m_auc = indirect_auc(pred_edges, reach, true_dir)

    print("Skeleton metrics:")
    print(f"  precision={m_skel['precision']:.3f}  recall={m_skel['recall']:.3f}  f1={m_skel['f1']:.3f}  SHD={m_skel['shd']:.1f}  (tp={m_skel['tp']}, fp={m_skel['fp']}, fn={m_skel['fn']})")
    print("Direction metrics (evaluated on edges that lie on true skeleton):")
    print(f"  orient_acc={m_dir['orient_acc']:.3f}  orient_recall={m_dir['orient_recall']:.3f}  n_oriented_on_true_skel={m_dir['n_oriented_on_true_skel']}")
    print("Edge weight error (MAE on correctly oriented true edges):")
    print(f"  mae={m_w['mae']:.4f}  n={m_w['n']}")
    print("Indirect-edge identification AUC (predicted non-direct edges only):")
    note = m_auc.get("note", "")
    if note:
        print(f"  auc={m_auc['auc']}  n={m_auc['n']}  note={note}")
    else:
        print(f"  auc={m_auc['auc']:.3f}  n={m_auc['n']}")

    print("-" * 72)
    print("Sample of τ_true (baseline do(q75)-do(q25)) for some true edges:")
    for e in TRUE_EDGES[:6]:
        print(f"  {e[0]} -> {e[1]} : tau_true={true_edge_tau[e]:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
