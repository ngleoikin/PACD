"""IVAPCI v3.3 (TheoryComplete + Weak-IV Adaptive): hierarchical encoder + theorem-aware extras + weak-IV self-adaptation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from . import BaseCausalEstimator
from .ivapci_theory_diagnostics import (
    TheoremComplianceDiagnostics,
    TheoremDiagnosticsConfig,
)


# =============================================================================
# Weak-IV Adaptive Theory Implementation (新增)
# =============================================================================

@dataclass
class WeakIVAdaptiveConfig:
    """弱IV自适应配置 (基于理论文档)"""
    # === 开关 ===
    enabled: bool = True               # 是否启用弱IV自适应
    
    # === 强度指标参数 ===
    ema_rho: float = 0.9               # EMA平滑系数
    safety_mix_omega: float = 0.6      # 安全融合系数 ω ∈ [0.5, 0.8]
    xi_min: float = 1e-4               # 数值下限
    
    # === 模式检测阈值 (带滞回) ===
    xi_enter_weak: float = 0.50        # 进入弱IV阈值 (对应 AUC < 0.75)
    xi_exit_weak: float = 0.55         # 退出弱IV阈值 (对应 AUC > 0.775)
    warmup_epochs: int = 20            # warmup期间不切换模式
    
    # === Exclusion日程参数 (γ_adv_z) ===
    gamma_z_max: float = 0.3           # 上界
    gamma_z_0: float = 0.1             # 基础系数
    alpha: float = 2.0                 # 弱IV放大系数 ∈ [1, 3]
    
    # === Relevance保护日程参数 (λ_cons_z) ===
    lambda_cons_max: float = 0.15      # 上界
    lambda_cons_0: float = 0.03        # 基础系数
    beta: float = 1.0                  # 弱IV放大系数 ∈ [0.5, 2]
    
    # === HSIC日程参数 ===
    lambda_hsic_max: float = 0.1       # 上界
    lambda_hsic_0: float = 0.01        # 基础系数
    eta: float = 0.5                   # log衰减系数
    
    # === 弱IV特化损失参数 ===
    # ExclPenalty
    excl_threshold: float = 0.02       # ℓ₀: 泄漏阈值
    excl_k: float = 0.1                # k_excl
    excl_cap: float = 3.0              # c_excl: 上界
    xi_floor: float = 0.02             # 防止除零
    
    # RelevPenalty  
    relev_threshold: float = 0.05      # κ₀: 相关性损失阈值
    relev_k: float = 0.05              # k_rel
    
    # 损失权重
    lambda_excl: float = 0.5           # ExclPenalty权重
    lambda_relev: float = 0.3          # RelevPenalty权重


class WeakIVAdaptiveScheduler:
    """弱IV自适应调度器 (基于理论文档 Theorem 1-2)
    
    核心理论:
        |Bias_weak| ≲ c₁ * (δ_excl / ξ) + c₂ * δ_rep
        
    当ξ(IV强度)下降时，排除限制误差被1/ξ放大，需要成对调参。
    """
    
    def __init__(
        self, 
        n_samples: int,
        xi_raw: Optional[float] = None,
        config: Optional[WeakIVAdaptiveConfig] = None
    ):
        self.n = n_samples
        self.config = config or WeakIVAdaptiveConfig()
        
        # 强度指标
        self.xi_raw = xi_raw
        self.xi_rep_ema = 0.0
        self.xi_eff = self.config.xi_min
        
        # 模式状态
        self.weak_iv_mode = False
        self.mode_switch_count = 0
        
        # 历史记录
        self.history = {
            'xi_eff': [],
            'weak_iv_mode': [],
            'gamma_z_adaptive': [],
            'lambda_hsic_adaptive': [],
        }
    
    @staticmethod
    def compute_strength_from_auc(auc: float) -> float:
        """AUC → 统一强度 [0, 1]"""
        return 2 * (max(auc, 1 - auc) - 0.5)
    
    def compute_xi_raw_from_data(self, Z: np.ndarray, A: np.ndarray) -> float:
        """计算场景固有IV强度 ξ_raw（更稳健：K-fold AUC）"""
        try:
            # 小样本/类别不平衡下，单次拟合AUC波动较大：用Stratified K-fold均值更稳
            n = int(len(A))
            if n < 30:
                # 极小样本时避免CV不稳定：退化为一次拟合
                probe = LogisticRegression(max_iter=500, solver='lbfgs')
                probe.fit(Z, A)
                proba = probe.predict_proba(Z)[:, 1]
                auc = roc_auc_score(A, proba)
                return self.compute_strength_from_auc(float(auc))

            n_folds = int(getattr(self.config, "xi_raw_cv_folds", 3) or 3)
            n_folds = int(np.clip(n_folds, 2, 5))
            cv = StratifiedKFold(n_splits=min(n_folds, max(2, n // 20)), shuffle=True, random_state=0)

            aucs = []
            for tr_idx, te_idx in cv.split(Z, A):
                # 某些极端切分可能导致测试集只有一个类别，跳过该fold
                if len(np.unique(A[te_idx])) < 2:
                    continue
                probe = LogisticRegression(max_iter=500, solver='lbfgs')
                probe.fit(Z[tr_idx], A[tr_idx])
                proba = probe.predict_proba(Z[te_idx])[:, 1]
                aucs.append(float(roc_auc_score(A[te_idx], proba)))

            if aucs:
                auc = float(np.mean(aucs))
            else:
                # 退化为一次拟合
                probe = LogisticRegression(max_iter=500, solver='lbfgs')
                probe.fit(Z, A)
                proba = probe.predict_proba(Z)[:, 1]
                auc = float(roc_auc_score(A, proba))

            return self.compute_strength_from_auc(auc)
        except Exception:
            # 🔑 安全修复：异常时返回保守的弱IV值，避免误判成"不弱"
            return self.config.xi_min  # 默认弱IV，更安全
    
    def update_xi_eff(self, xi_rep: float) -> float:
        """更新有效强度 ξ_eff (EMA + 安全融合)"""
        cfg = self.config
        
        # EMA平滑
        self.xi_rep_ema = cfg.ema_rho * self.xi_rep_ema + (1 - cfg.ema_rho) * xi_rep
        
        # 安全融合
        if self.xi_raw is not None:
            xi_eff = max(self.xi_rep_ema, cfg.safety_mix_omega * self.xi_raw)
        else:
            xi_eff = self.xi_rep_ema
        
        self.xi_eff = max(xi_eff, cfg.xi_min)
        return self.xi_eff
    
    def detect_weak_iv_mode(self, epoch: int, xi_for_mode: Optional[float] = None) -> bool:
        """弱IV模式检测 (带滞回阈值)"""
        cfg = self.config
        
        if epoch < cfg.warmup_epochs:
            return self.weak_iv_mode

        xi = float(self.xi_eff if xi_for_mode is None else xi_for_mode)

        if not self.weak_iv_mode:
            if xi < cfg.xi_enter_weak:
                self.weak_iv_mode = True
                self.mode_switch_count += 1
        else:
            if xi > cfg.xi_exit_weak:
                self.weak_iv_mode = False
                self.mode_switch_count += 1
        
        return self.weak_iv_mode
    
    def compute_gamma_z_adaptive(self) -> float:
        """计算自适应的 γ_adv_z (Exclusion强度)
        
        γ_adv_z(n, ξ) = min(γ_max, γ₀ * n^(-1/2) * [1 + α * (1-ξ)/ξ])
        """
        cfg = self.config
        xi = max(self.xi_eff, cfg.xi_min)
        
        base = cfg.gamma_z_0 * (self.n ** -0.5)
        weak_iv_factor = 1 + cfg.alpha * (1 - xi) / xi
        
        return min(cfg.gamma_z_max, base * weak_iv_factor)
    
    def compute_lambda_cons_adaptive(self) -> float:
        """计算自适应的 λ_consistency (Relevance保护)"""
        cfg = self.config
        xi = max(self.xi_eff, cfg.xi_min)
        
        base = cfg.lambda_cons_0 * (self.n ** -0.5)
        weak_iv_factor = 1 + cfg.beta * (1 - xi) / xi
        
        return min(cfg.lambda_cons_max, base * weak_iv_factor)
    
    def compute_lambda_hsic_adaptive(self) -> float:
        """计算自适应的 λ_hsic
        
        λ_hsic(n, ξ) = min(λ_max, λ₀ * log(1+n)/√n * [1 + η * log(1/ξ)])
        """
        cfg = self.config
        xi = max(self.xi_eff, cfg.xi_min)
        
        base = cfg.lambda_hsic_0 * np.log1p(self.n) / np.sqrt(self.n)
        weak_iv_factor = 1 + cfg.eta * np.log(1 / xi)
        
        return min(cfg.lambda_hsic_max, base * weak_iv_factor)
    
    def compute_excl_penalty(self, L_ZY: float) -> float:
        """计算排除限制增强惩罚 (ExclPenalty)
        
        ExclPenalty = s_excl(ξ) * max(0, L_ZY - ℓ₀)
        s_excl(ξ) = min(c_excl, k_excl / max(ξ, ξ_floor))
        """
        cfg = self.config
        xi = max(self.xi_eff, cfg.xi_floor)
        
        s_excl = min(cfg.excl_cap, cfg.excl_k / xi)
        penalty = s_excl * max(0, L_ZY - cfg.excl_threshold)
        
        return cfg.lambda_excl * penalty
    
    def compute_relev_penalty(self, L_ZA: float) -> float:
        """计算相关性保护惩罚 (RelevPenalty)"""
        cfg = self.config
        xi = max(self.xi_eff, cfg.xi_floor)
        
        if L_ZA <= cfg.relev_threshold:
            return 0.0
        
        s_rel = cfg.relev_k / xi
        penalty = s_rel * L_ZA
        
        return cfg.lambda_relev * penalty
    
    def step(
        self, 
        epoch: int, 
        diagnostics: Dict[str, float]
    ) -> Dict[str, float]:
        """每个epoch调用，返回自适应参数"""
        
        # 1. 从诊断中获取表示强度
        if 'rep_auc_z_to_a' in diagnostics:
            auc = diagnostics['rep_auc_z_to_a']
            xi_rep = self.compute_strength_from_auc(auc)
        else:
            xi_rep = self.xi_rep_ema
        
        # 2. 更新有效强度
        xi_eff = self.update_xi_eff(xi_rep)
        
        # 3. 检测模式
        weak_iv_mode = self.detect_weak_iv_mode(epoch, xi_for_mode=self.xi_rep_ema)
        
        # 4. 计算自适应参数
        gamma_z_adaptive = self.compute_gamma_z_adaptive()
        lambda_cons_adaptive = self.compute_lambda_cons_adaptive()
        lambda_hsic_adaptive = self.compute_lambda_hsic_adaptive()
        
        # 5. 计算弱IV特化损失
        L_ZY = diagnostics.get('rep_exclusion_leakage_r2_cond', diagnostics.get('rep_exclusion_leakage_r2', 0.0))
        L_ZA = diagnostics.get('consistency_loss_z', 0.0)
        
        excl_penalty = self.compute_excl_penalty(L_ZY) if weak_iv_mode else 0.0
        relev_penalty = self.compute_relev_penalty(L_ZA) if weak_iv_mode else 0.0
        
        # 6. 记录历史
        self.history['xi_eff'].append(xi_eff)
        self.history['weak_iv_mode'].append(weak_iv_mode)
        self.history['gamma_z_adaptive'].append(gamma_z_adaptive)
        self.history['lambda_hsic_adaptive'].append(lambda_hsic_adaptive)
        
        return {
            'xi_eff': xi_eff,
            'weak_iv_mode': weak_iv_mode,
            'gamma_z_adaptive': gamma_z_adaptive,
            'lambda_cons_adaptive': lambda_cons_adaptive,
            'lambda_hsic_adaptive': lambda_hsic_adaptive,
            'excl_penalty': excl_penalty,
            'relev_penalty': relev_penalty,
            'weak_iv_loss': excl_penalty + relev_penalty,
        }


# =============================================================================
# Original IVAPCI v3.3 Code (with Weak-IV Integration)
# =============================================================================

def _info_aware_standardize(
    train: np.ndarray,
    min_std: float = 1e-2,
    low_var_min_std: float = 1e-4,
    clip_value: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize while *clamping* near-constant dimensions."""
    train = np.asarray(train, dtype=np.float32)
    mean = train.mean(axis=0, keepdims=True, dtype=np.float32)
    std = train.std(axis=0, keepdims=True, dtype=np.float32)
    protected = (std < min_std).squeeze(0)
    std_clamped = np.where(std < low_var_min_std, np.maximum(std, low_var_min_std), std).astype(np.float32)
    standardized = ((train - mean) / std_clamped).astype(np.float32)
    if clip_value is not None:
        standardized = np.clip(standardized, -clip_value, clip_value).astype(np.float32)
    return standardized, mean.astype(np.float32), std_clamped, protected


def _effective_sample_size(weights: np.ndarray) -> float:
    if weights.size == 0:
        return 0.0
    num = float(np.sum(weights) ** 2)
    den = float(np.sum(weights**2) + 1e-12)
    return num / den if den > 0 else 0.0


def _compute_residual(target: torch.Tensor, condition: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """Project out ``condition`` from ``target``."""
    if target.shape[0] != condition.shape[0]:
        min_n = min(int(target.shape[0]), int(condition.shape[0]))
        target = target[:min_n]
        condition = condition[:min_n]
    condition_c = condition - condition.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    ct_c = condition_c.T @ condition_c / max(1, condition_c.shape[0])
    i_eye = torch.eye(ct_c.shape[0], device=condition.device, dtype=condition.dtype)
    ct_t = condition_c.T @ target_c / max(1, condition_c.shape[0])
    try:
        beta = torch.linalg.solve(ct_c + ridge * i_eye, ct_t)
    except RuntimeError:
        beta = torch.zeros(condition_c.shape[1], target_c.shape[1], device=condition.device, dtype=condition.dtype)
    target_res = target_c - condition_c @ beta
    return target_res


class TrueConditionalAdversary(nn.Module):
    """Conditional adversary via concat+detach."""
    def __init__(self, target_dim: int, condition_dim: int, hidden: Sequence[int], *, loss: str = "bce"):
        super().__init__()
        self.discriminator = _mlp(target_dim + condition_dim, hidden, 1)
        self.loss = loss.lower()

    def forward(self, target: torch.Tensor, condition: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([target, condition.detach()], dim=1)
        logits = self.discriminator(combined).squeeze(-1)
        if self.loss == "bce":
            return nn.functional.binary_cross_entropy_with_logits(logits, label)
        return nn.functional.mse_loss(logits, label)


def compute_ess_by_group(e_hat: np.ndarray, A: np.ndarray) -> Tuple[float, float]:
    eps = 1e-8
    treated_mask = A == 1
    control_mask = A == 0
    if treated_mask.sum() > 0:
        w_treated = 1.0 / np.clip(e_hat[treated_mask], eps, 1 - eps)
        ess_treated = _effective_sample_size(w_treated) / treated_mask.sum()
    else:
        ess_treated = 0.0
    if control_mask.sum() > 0:
        w_control = 1.0 / np.clip(1 - e_hat[control_mask], eps, 1 - eps)
        ess_control = _effective_sample_size(w_control) / control_mask.sum()
    else:
        ess_control = 0.0
    return ess_treated, ess_control


def find_optimal_clip_threshold(
    e_raw: np.ndarray, A: np.ndarray, ess_target: float,
    min_clip: float, max_clip: float, n_search: int = 20,
) -> Tuple[float, dict]:
    candidates = np.linspace(min_clip, max_clip, n_search)
    best_clip = max_clip
    best_min_ess = 0.0
    for clip_val in candidates:
        e_clipped = np.clip(e_raw, clip_val, 1 - clip_val)
        ess_t, ess_c = compute_ess_by_group(e_clipped, A)
        min_ess = min(ess_t, ess_c)
        if min_ess >= ess_target:
            best_clip = clip_val
            best_min_ess = min_ess
            break
        if min_ess > best_min_ess:
            best_clip = clip_val
            best_min_ess = min_ess
    e_final = np.clip(e_raw, best_clip, 1 - best_clip)
    ess_t, ess_c = compute_ess_by_group(e_final, A)
    eps = 1e-8
    w_ate = A / np.clip(e_final, eps, 1 - eps) + (1 - A) / np.clip(1 - e_final, eps, 1 - eps)
    overall_ess = _effective_sample_size(w_ate)
    overall_ess_ratio = overall_ess / len(A) if len(A) else 0.0
    stats = {
        "clip_threshold": float(best_clip),
        "ess_treated": float(ess_t),
        "ess_control": float(ess_c),
        "ess_min": float(min(ess_t, ess_c)),
        "ess_overall_ratio": float(overall_ess_ratio),
        "frac_clipped": float(np.mean((e_raw < best_clip) | (e_raw > 1 - best_clip))),
    }
    return float(best_clip), stats


def adaptive_ipw_cap_by_quantile(
    weights: np.ndarray, quantile: float, min_cap: float, max_cap: float,
) -> float:
    if weights.size == 0:
        return min_cap
    cap_candidate = float(np.quantile(np.abs(weights), quantile))
    return float(np.clip(cap_candidate, min_cap, max_cap))


def _apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_f = np.asarray(x, dtype=np.float32)
    return (x_f - mean) / std


class InformationLossMonitor:
    """Reconstruction MSE / variance as a lightweight ΔI proxy."""
    @staticmethod
    def estimate(model: "IVAPCIv33TheoryHierEstimator", V_all: np.ndarray) -> Dict[str, float]:
        V_std = _apply_standardize(V_all.astype(np.float32), model._v_mean, model._v_std)
        with torch.no_grad():
            V_t = torch.from_numpy(V_std).to(model.device)
            tx, tw, tz, tn = model._encode_blocks(V_t)
            recon = model.decoder(torch.cat([tx, tw, tz, tn], dim=1)).cpu().numpy()
        recon_mse = float(np.mean((recon - V_std) ** 2))
        var = float(np.var(V_std) + 1e-8)
        return {"recon_mse": recon_mse, "info_loss_proxy": recon_mse / var}


def enhanced_training_monitor(estimator: "IVAPCIv33TheoryHierEstimator", epoch: int, losses: Dict[str, float]) -> None:
    """Lightweight training-time monitor."""
    if epoch % 10 != 0:
        return
    diag = getattr(estimator, "training_diagnostics", {}) or {}
    cfg = getattr(estimator, "config", None)
    estimand = str(getattr(cfg, "estimand", "ate")).lower().strip() if cfg else "ate"
    is_proximal = estimand in {"proximal_dr", "proximal", "proximal_bridge"}
    
    warnings = []
    w_auc = float(diag.get("rep_auc_w_to_a", 0.5))
    # ✅ proximal 下不报 W 独立性警告（W→A 相关性是桥方程需要的信号）
    if abs(w_auc - 0.5) > 0.1 and not is_proximal:
        warnings.append(f"⚠️  W 独立性差: AUC={w_auc:.3f}")
    z_r2 = float(diag.get("rep_exclusion_leakage_r2", 0.0))
    if z_r2 > 0.15 and not is_proximal:
        warnings.append(f"⚠️  Z 泄露: R²={z_r2:.3f}")
    iv_strength = float(diag.get("iv_relevance_abs_corr", 0.0))
    if iv_strength < 0.15:
        warnings.append(f"⚠️  弱 IV: corr={iv_strength:.3f}")
    # 弱IV自适应状态
    weak_iv_scheduler = getattr(estimator, "_weak_iv_scheduler", None)
    if weak_iv_scheduler is not None and weak_iv_scheduler.weak_iv_mode:
        warnings.append(f"🔧 弱IV自适应模式: ξ_eff={weak_iv_scheduler.xi_eff:.3f}")
    # proximal_dr 诊断 (v8.9: 显示真正的 min-max 对抗训练指标)
    if is_proximal:
        critic_loss = float(diag.get("q_bridge_critic_loss", 0.0))
        adv_loss = float(diag.get("q_bridge_adv_loss", 0.0))
        q_mean = float(diag.get("q_pred_mean", 0.0))
        q_std = float(diag.get("q_pred_std", 0.0))
        f_mean = float(diag.get("f_critic_mean", 0.0))
        f_std = float(diag.get("f_critic_std", 0.0))
        h_hsic = float(diag.get("bridge_hsic_epoch", 0.0))
        if critic_loss != 0 or adv_loss != 0:
            warnings.append(f"📊 Proximal DR: critic={critic_loss:.4f}, adv={adv_loss:.4f}, q={q_mean:.2f}±{q_std:.2f}, f={f_mean:.2f}±{f_std:.2f}")
    if warnings:
        print(f"\nEpoch {epoch} 诊断:")
        for w in warnings:
            print(f"  {w}")
        print()


# NN building blocks
def _mlp(input_dim: int, hidden: Sequence[int], out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class _GroupEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        if hidden:
            self.body = _mlp(input_dim, hidden, hidden[-1], dropout=dropout)
            self.out = nn.Linear(hidden[-1], out_dim)
        else:
            self.body = nn.Identity()
            self.out = nn.Linear(input_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x))


class _Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int], out_dim: int):
        super().__init__()
        self.net = _mlp(input_dim, hidden, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _AClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int]):
        super().__init__()
        self.net = _mlp(input_dim, hidden, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t).squeeze(-1)


class _YRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int]):
        super().__init__()
        self.net = _mlp(input_dim + 1, hidden, 1)

    def forward(self, t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([t, a.unsqueeze(-1)], dim=1)).squeeze(-1)


class _BridgeHead(nn.Module):
    """Outcome bridge head: h(W,A,X) = mu(W,X) + A * tau(W,X).
    
    Satisfies the bridge equation: E[Y - h(W,A,X) | A,X,Z] = 0
    
    This structure explicitly models the treatment effect tau(W,X),
    preventing it from being drowned out by baseline features.
    """
    def __init__(self, input_dim: int, hidden: Sequence[int]):
        super().__init__()
        # Baseline outcome network: mu(W,X)
        self.mu_net = _mlp(input_dim, hidden, 1)
        # Treatment effect network: tau(W,X)
        self.tau_net = _mlp(input_dim, hidden, 1)
    
    def forward(self, t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mu = self.mu_net(t).squeeze(-1)
        tau = self.tau_net(t).squeeze(-1)
        return mu + a * tau
    
    def get_tau(self, t: torch.Tensor) -> torch.Tensor:
        """Get the treatment effect tau(W,X) directly."""
        return self.tau_net(t).squeeze(-1)


class _QBridgeHead(nn.Module):
    """Treatment bridge head: q(Z,A,X).
    
    Satisfies the bridge equation: E[q(Z,a,X) | W,X,A=a] = 1/P(A=a|W,X)
    
    For binary treatment:
    - q(Z,1,X): should satisfy E[q|W,X,A=1] = 1/e(W,X)
    - q(Z,0,X): should satisfy E[q|W,X,A=0] = 1/(1-e(W,X))
    
    This is used in the DR correction term of the Proximal estimator.
    """
    def __init__(self, z_dim: int, x_dim: int, hidden: Sequence[int]):
        super().__init__()
        # q(Z, a=1, X) network
        self.q1_net = _mlp(z_dim + x_dim, hidden, 1)
        # q(Z, a=0, X) network  
        self.q0_net = _mlp(z_dim + x_dim, hidden, 1)
    
    def forward(self, z: torch.Tensor, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Compute q(Z,A,X) for given treatment values."""
        zx = torch.cat([z, x], dim=1)
        q1 = self.q1_net(zx).squeeze(-1)
        q0 = self.q0_net(zx).squeeze(-1)
        # Return q1 for treated, q0 for control
        return a * q1 + (1 - a) * q0
    
    def get_q1(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Get q(Z,1,X) for treated."""
        return self.q1_net(torch.cat([z, x], dim=1)).squeeze(-1)
    
    def get_q0(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Get q(Z,0,X) for control."""
        return self.q0_net(torch.cat([z, x], dim=1)).squeeze(-1)


class _QBridgeCritic(nn.Module):
    """Critic network for q-bridge adversarial moment training.
    
    The critic tries to predict q_residual = q(Z,X) - target from (W,X).
    If the moment condition E[q - target | W,X] = 0 is satisfied,
    the critic should not be able to predict the residual.
    
    Training:
    1. Critic step: max E[(critic(W,X) - residual)^2] (critic learns to predict residual)
    2. Q-bridge step: min E[critic(W,X)^2] (q learns to make residual unpredictable)
    """
    def __init__(self, w_dim: int, x_dim: int, hidden: Sequence[int] = (64, 32)):
        super().__init__()
        self.net = _mlp(w_dim + x_dim, hidden, 1)
    
    def forward(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Predict residual from (W,X)."""
        return self.net(torch.cat([w, x], dim=1)).squeeze(-1)


class _Gate(nn.Module):
    """Sample-dependent gate: g = sigmoid(W * ctx + b), initialized mostly-closed."""

    def __init__(self, ctx_dim: int, out_dim: int, init_bias: float = -6.0):
        super().__init__()
        self.lin = nn.Linear(ctx_dim, out_dim)
        nn.init.zeros_(self.lin.weight)
        nn.init.constant_(self.lin.bias, float(init_bias))

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(ctx))


class _YAdversary(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int]):
        super().__init__()
        self.net = _mlp(input_dim, hidden, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t).squeeze(-1)


# Orthogonality penalties
def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=0, keepdim=True)


def _offdiag_corr_penalty(blocks: list[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    pen = torch.zeros((), device=blocks[0].device)
    centered = [_center(b) for b in blocks]
    norms = [torch.sqrt(torch.mean(b ** 2, dim=0, keepdim=True) + eps) for b in centered]
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            bi = centered[i] / norms[i]
            bj = centered[j] / norms[j]
            corr = (bi.T @ bj) / bi.shape[0]
            pen = pen + torch.mean(corr ** 2)
    return pen


def _conditional_orthogonal_penalty(blocks: list[torch.Tensor], cond: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    pen = torch.zeros((), device=blocks[0].device)
    C = _center(cond)
    CtC = C.T @ C / C.shape[0]
    I = torch.eye(CtC.shape[0], device=C.device)
    Ct = C.T / C.shape[0]
    resid = []
    for B in blocks:
        Bc = _center(B)
        Beta = torch.linalg.solve(CtC + ridge * I, Ct @ Bc)
        resid.append(Bc - C @ Beta)
    return _offdiag_corr_penalty(resid)


def _padic_ultrametric_loss(Uc: torch.Tensor, num_triplets: int = 128) -> torch.Tensor:
    if Uc.shape[0] < 3:
        return torch.zeros((), device=Uc.device)
    n = Uc.shape[0]
    t = min(num_triplets, max(1, n // 2))
    idx = torch.randint(0, n, (t, 3), device=Uc.device)
    u_i, u_j, u_k = Uc[idx[:, 0]], Uc[idx[:, 1]], Uc[idx[:, 2]]
    d_ij = torch.norm(u_i - u_j, dim=1)
    d_jk = torch.norm(u_j - u_k, dim=1)
    d_ik = torch.norm(u_i - u_k, dim=1)
    viol = torch.relu(d_ik - torch.maximum(d_ij, d_jk))
    return torch.mean(viol ** 2)


def _rbf_hsic(x: torch.Tensor, y: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    """Compute HSIC with RBF kernels for independence penalty."""
    if x.shape[0] < 4 or y.shape[0] < 4:
        return torch.zeros((), device=x.device)
    n = x.shape[0]
    if sigma is None:
        with torch.no_grad():
            dist2 = torch.pdist(x, p=2).pow(2)
            sigma = torch.sqrt(torch.median(dist2) + 1e-6)
        sigma = float(sigma.item()) if sigma is not None else 1.0
    gamma = 1.0 / (2 * max(sigma, 1e-6) ** 2)
    Kx = torch.exp(-gamma * torch.cdist(x, x).pow(2))
    dist2_y = torch.pdist(y, p=2).pow(2)
    sigma_y = torch.sqrt(torch.median(dist2_y) + 1e-6)
    gamma_y = 1.0 / (2 * max(float(sigma_y.item()), 1e-6) ** 2)
    Ky = torch.exp(-gamma_y * torch.cdist(y, y).pow(2))
    H = torch.eye(n, device=x.device) - torch.ones((n, n), device=x.device) / n
    Kc = H @ Kx @ H
    Lc = H @ Ky @ H
    hsic = torch.trace(Kc @ Lc) / ((n - 1) ** 2)
    return hsic



# =============================================================================
# Scenario-aware Robustness (Heavy-tail / Misaligned-proxies / Weak-overlap)
# =============================================================================

class ScenarioType(Enum):
    """Problem scenario types (optional)."""
    DEFAULT = "default"
    HEAVY_TAIL = "heavy_tail"
    MISALIGNED_PROXIES = "misaligned_proxies"
    WEAK_OVERLAP = "weak_overlap"


def detect_scenario_type(scenario_name: Optional[str]) -> ScenarioType:
    """Detect scenario type from a scenario name string."""
    if not scenario_name:
        return ScenarioType.DEFAULT
    name_lower = str(scenario_name).lower().replace("_", "-")
    if "heavy-tail" in name_lower or "heavytail" in name_lower:
        return ScenarioType.HEAVY_TAIL
    if "misaligned" in name_lower:
        return ScenarioType.MISALIGNED_PROXIES
    if "weak-overlap" in name_lower or "weakoverlap" in name_lower:
        return ScenarioType.WEAK_OVERLAP
    return ScenarioType.DEFAULT


@dataclass
class ScenarioSpecificConfig:
    """Scenario-specific robustness knobs (defaults are no-op)."""

    # Heavy-tail
    use_huber_loss: bool = False
    huber_delta: float = 1.0
    winsorize_y: bool = False
    winsorize_quantile: float = 0.01

    # Misaligned proxies
    lambda_ortho_boost: float = 1.0
    lambda_hsic_boost: float = 1.0
    gamma_adv_w_boost: float = 1.0
    lambda_alignment: float = 0.0  # proxy alignment loss weight


    # Gated routing (Misaligned fix)
    gate_reg_mult: float = 1.0  # <1 => easier to open gates

    # Weak overlap
    lambda_overlap_boost: float = 1.0
    overlap_margin_boost: float = 1.0
    use_focal_loss: bool = False
    focal_gamma: float = 2.0


def get_scenario_config(scenario_type: ScenarioType) -> ScenarioSpecificConfig:
    """Return defaults for a detected scenario type.
    
    最终配置 (v6.3):
    - HEAVY_TAIL: Huber + Winsorize + 严格裁剪 (Delta: +0.74 → -1.94)
    - MISALIGNED_PROXIES: Proximal Bridge + 分离式 BridgeHead (Delta: +1.14 → +0.36)
    - WEAK_OVERLAP: ATO + IPW裁剪 (Delta: +0.59 → +0.46)
    """
    if scenario_type == ScenarioType.HEAVY_TAIL:
        return ScenarioSpecificConfig(
            use_huber_loss=True,
            huber_delta=1.0,
            winsorize_y=True,
            winsorize_quantile=0.01,
            lambda_ortho_boost=1.5,
        )
    if scenario_type == ScenarioType.MISALIGNED_PROXIES:
        # v5: 采用“门控软路由”修复主线；其它 boost 保持 1.0，避免硬约束破坏信息
        return ScenarioSpecificConfig(gate_reg_mult=0.2)
    if scenario_type == ScenarioType.WEAK_OVERLAP:
        # v6: ATO + IPW裁剪 (estimand 在 apply_scenario_adjustments 设置)
        return ScenarioSpecificConfig(
            lambda_overlap_boost=10.0,   # 保持 v2
            overlap_margin_boost=3.0,    # 保持 v2
            use_focal_loss=True,
            focal_gamma=3.0,             # 保持 v2
        )
    return ScenarioSpecificConfig()


def _winsorize_1d(y: np.ndarray, q: float) -> np.ndarray:
    """Winsorize a 1D array at [q, 1-q] quantiles."""
    q = float(np.clip(q, 0.0, 0.49))
    if q <= 0:
        return y
    lo = np.quantile(y, q)
    hi = np.quantile(y, 1.0 - q)
    return np.clip(y, lo, hi)


class HuberLoss(nn.Module):
    """Mean Huber loss (robust to outliers)."""
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = float(delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        abs_diff = torch.abs(diff)
        delta = self.delta
        quadratic = 0.5 * diff ** 2
        linear = delta * (abs_diff - 0.5 * delta)
        return torch.where(abs_diff <= delta, quadratic, linear).mean()


class FocalBCELoss(nn.Module):
    """Binary focal loss on logits (for weak-overlap / class imbalance)."""
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = self._bce(logits, target)
        p = torch.sigmoid(logits)
        pt = target * p + (1.0 - target) * (1.0 - p)
        w = (1.0 - pt).pow(self.gamma)
        return (w * bce).mean()


def proxy_alignment_loss(tx: torch.Tensor, tw: torch.Tensor, tz: Optional[torch.Tensor] = None, ridge: float = 1e-3) -> torch.Tensor:
    """Encourage proxy representations to have aligned directions (misaligned-proxies helper)."""
    def _center(x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=0, keepdim=True)

    def _pair_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = _center(a); b = _center(b)
        n = a.shape[0]
        if n < 4:
            return torch.zeros((), device=a.device)
        cov = (a.T @ b) / n  # da x db
        va = (a.T @ a) / n + ridge * torch.eye(a.shape[1], device=a.device)
        vb = (b.T @ b) / n + ridge * torch.eye(b.shape[1], device=b.device)
        try:
            x = torch.linalg.solve(va, cov)          # da x db
            y = torch.linalg.solve(vb, cov.T)        # db x da
            corr = torch.trace(x @ y) / max(1.0, float(min(a.shape[1], b.shape[1])))
            return corr
        except RuntimeError:
            return torch.zeros((), device=a.device)

    corr = _pair_corr(tx, tw)
    k = 1
    if tz is not None and tz.numel() > 0:
        corr = corr + _pair_corr(tx, tz)
        corr = corr + _pair_corr(tw, tz)
        k = 3
    corr = corr / float(k)
    # minimize negative correlation -> maximize alignment
    return -corr


# =============================================================================
# Config (with Weak-IV Adaptive)
# =============================================================================

@dataclass
class IVAPCIV33TheoryConfig:
    """Theory-informed configuration with Weak-IV Adaptive support."""

    x_dim: int = 0
    w_dim: int = 0
    z_dim: int = 0

    latent_x_dim: int = 4
    latent_w_dim: int = 4
    latent_z_dim: int = 2
    latent_n_dim: int = 4

    # === Gated soft routing (Misaligned-proxies fix) ===
    enable_gated_routing: bool = True
    gate_init_bias: float = -6.0  # sigmoid(bias); more negative => gates closed by default
    lambda_gate: float = 0.01     # L1-like penalty on gate openness

    enc_x_hidden: Sequence[int] = (128, 64)
    enc_w_hidden: Sequence[int] = (256, 128, 64)
    enc_z_hidden: Sequence[int] = (32, 16)  # 搜索时固定值
    enc_n_hidden: Sequence[int] = (128, 64)
    dropout_z: float = 0.30  # 搜索时固定值

    dec_hidden: Sequence[int] = (128, 128)

    a_hidden: Sequence[int] = (64, 32)
    y_hidden: Sequence[int] = (128, 64)
    adv_a_hidden: Sequence[int] = (64,)
    adv_y_hidden: Sequence[int] = (64,)
    gamma_adv_w_cond: float = 0.284414  # optimized
    gamma_adv_z_cond: float = 0.1
    gamma_adv_n_cond: float = 0.08

    lambda_hsic: float = 0.064533  # optimized
    hsic_max_samples: int = 256

    lambda_recon: float = 1.0
    lambda_a: float = 0.1
    lambda_y: float = 0.5
    lambda_ortho: float = 0.0125  # 动态调整值 for n=500
    lambda_cond_ortho: float = 1e-3
    lambda_consistency: float = 0.10  # 动态调整值 for n=500
    ridge_alpha: float = 1e-2
    standardize_nuisance: bool = True
    gamma_adv_w: float = 0.325339  # optimized
    gamma_adv_z: float = 0.18  # 搜索时固定值（关键！）
    gamma_adv_n: float = 0.1
    adv_steps: int = 3
    adv_steps_min: int = 1
    adv_steps_max: int = 5
    adv_steps_dynamic: bool = True
    adv_warmup_epochs: int = 10
    adv_ramp_epochs: int = 30
    gamma_padic: float = 1e-3

    min_std: float = 0.004  # 动态调整值 for n=500
    low_var_min_std: float = 1e-3
    std_clip: Optional[float] = 10.0
    use_noise_in_latent: bool = True

    lr_main: float = 1e-3
    lr_adv: float = 1.5e-3
    batch_size: int = 128
    epochs_pretrain: int = 36   # 动态调整值 for n=500
    epochs_main: int = 110      # 动态调整值 for n=500
    val_frac: float = 0.1
    early_stopping_patience: int = 18
    early_stopping_min_delta: float = 0.0

    n_splits_dr: int = 5
    clip_prop: float = 0.01
    clip_prop_adaptive_max: float = 0.06
    clip_prop_radr: float = 1e-2
    propensity_logreg_C: float = 0.5
    propensity_shrinkage: float = 0.02
    ipw_cap: float = 15.0
    ipw_cap_quantile: float = 0.995
    ipw_cap_high: float = 100.0
    ipw_cap_radr: Optional[float] = None
    adaptive_ipw: bool = True
    ess_target: float = 0.55

    # === Estimand switch for weak-overlap (estimation stage) ===
    # 'ate' | 'trimmed_ate' | 'ato' | 'proximal_bridge'
    estimand: str = "ate"
    trim_prop: float = 0.05  # for trimmed_ate: keep samples with e in [trim_prop, 1-trim_prop]

    # === Proximal bridge (M3) ===
    # If estimand='proximal_bridge' or 'proximal_dr', the estimator learns bridge functions.
    # Outcome bridge h(W,A,X): E[Y - h(W,A,X) | A,X,Z] = 0
    # Treatment bridge q(Z,A,X): E[q(Z,a,X) | W,X,A=a] = 1/P(A=a|W,X)
    lambda_bridge: float = 1.0  # v6.3: 从0.5增加到1.0，加强bridge训练
    lambda_bridge_hsic: float = 0.05  # v6.1: 从0.1降低到0.05，减少对treatment effect的压制
    bridge_hidden: Sequence[int] = (256, 128, 64)  # v6.3: 增加容量
    bridge_hsic_max_samples: int = 256
    bridge_detach_cond: bool = True  # detach (tx,tz) in HSIC to avoid distorting encoders
    bridge_warmup_epochs: int = 10
    bridge_hsic_by_treatment: bool = True  # v8.1: HSIC computed within A groups (closer to conditional-moment)
    
    # === Treatment bridge q (M3 complete) ===
    # q-bridge satisfies: E[q(Z,a,X) | W,X,A=a] = 1/e(W,X) or 1/(1-e(W,X))
    enable_q_bridge: bool = True  # v8: 启用 treatment bridge
    q_bridge_hidden: Sequence[int] = (128, 64)
    q_critic_hidden: Sequence[int] = (64, 32)  # v8.6: critic 网络
    lambda_q_bridge: float = 0.5  # q-bridge 训练权重
    lambda_q_bridge_adv: float = 1.0  # v8.6: 对抗损失权重
    lambda_q_bridge_mean: float = 0.1  # v8.6: 均值惩罚（较小）
    lambda_q_bridge_l2: float = 1e-3  # v8.6: L2 稳定
    lambda_q_bridge_hsic: float = 0.02  # v8.1: HSIC on (q - target) vs (W,X) (deprecated, use adversarial)
    q_critic_steps: int = 3  # v8.6: 每步 critic 训练次数
    q_bridge_hsic_max_samples: int = 256
    q_bridge_prop_clip: Tuple[float, float] = (0.01, 0.99)
    q_bridge_target_clip: Tuple[float, float] = (0.5, 5.0)  # v8.6: 更保守的 clip 范围
    q_bridge_warmup_epochs: int = 10  # v8.6: 更早开始训练
    
    # === Cross-fitting for Proximal DR ===
    enable_cross_fitting: bool = False  # v8.7: 启用真正的交叉拟合
    cross_fitting_folds: int = 3  # K 折交叉拟合（3折平衡速度和效果）
    crossfit_nuisance_epochs: int = 80  # v8.8: 每折 nuisance heads 训练轮数
    crossfit_warm_start: bool = True  # v8.8: 用全局训练的 heads 初始化每折
    
    # === Proximal DR estimand ===
    # 'proximal_bridge': plug-in E[h1-h0] (v6.3)
    # 'proximal_dr': full DR formula with q-bridge (v8)
    # estimand 可选: 'ate', 'ato', 'trimmed_ate', 'proximal_bridge', 'proximal_dr'

    lambda_overlap: float = 0.02
    overlap_margin: float = 0.05
    overlap_warmup_epochs: int = 10

    cond_adv_warmup_epochs: int = 10
    cond_adv_ramp_epochs: int = 20
    monitor_batch_size: int = 512
    monitor_every: int = 1
    monitor_ema: float = 0.8
    overlap_boost: float = 2.0
    ess_target_train: Optional[float] = None

    seed: int = 42
    device: str = "cpu"

    cond_ortho_warmup_epochs: int = 15

    n_samples_hint: Optional[int] = None
    adaptive: bool = True

    # 修复的参数
    target_w_auc: float = 0.519117  # optimized
    target_z_r2: float = 0.10
    lambda_hsic_w_a: float = 0.033206  # optimized
    
    use_scheduler_feedback: bool = True
    disable_gamma_auto_scale: bool = True
    
    # === 新增: 弱IV自适应配置 ===
    weak_iv_adaptive: WeakIVAdaptiveConfig = field(default_factory=WeakIVAdaptiveConfig)
    
    # === 新增: 场景感知配置 ===
    scenario_name: Optional[str] = None
    scenario_type: ScenarioType = field(default=ScenarioType.DEFAULT)
    scenario_config: ScenarioSpecificConfig = field(default_factory=ScenarioSpecificConfig)

    def apply_theorem45_defaults(self) -> "IVAPCIV33TheoryConfig":
        n = self.n_samples_hint
        if (not self.adaptive) or (n is None) or (n <= 0):
            return self
        base = max(6, int(2 * np.log1p(n)))
        self.latent_x_dim = max(2, base // 3)
        self.latent_w_dim = max(2, base // 3)
        self.latent_z_dim = max(2, base // 6)
        self.latent_n_dim = max(2, base // 3)

        ortho_scale = 0.005 * np.sqrt(np.log1p(n))
        self.lambda_ortho = float(max(self.lambda_ortho, ortho_scale))
        self.lambda_consistency = 0.04 * np.sqrt(np.log1p(n))

        self.gamma_padic = 0.001 * np.sqrt(np.log1p(n))
        self.min_std = max(1e-4, 1e-2 / np.sqrt(np.log1p(n)))
        self.low_var_min_std = min(self.low_var_min_std, self.min_std * 0.1)

        self.epochs_pretrain = int(min(80, 30 + 10 * np.log1p(n / 500)))
        self.epochs_main = int(min(220, 100 + 15 * np.log1p(n / 500)))
        
        if getattr(self, "ess_target_train", None) is None:
            r = float(np.log1p(n) / max(1e-6, np.log1p(2000.0)))
            self.ess_target_train = float(np.clip(0.20 + 0.15 * r, 0.20, 0.45))
        self.ess_target = float(np.clip(self.ess_target_train + 0.05, 0.25, 0.55))

        self.clip_prop = float(min(self.clip_prop, 0.01))
        self.clip_prop_adaptive_max = float(max(self.clip_prop_adaptive_max, 0.05))

        if n <= 800:
            self.adv_steps = int(max(self.adv_steps_min, min(self.adv_steps_max, max(2, self.adv_steps))))
        else:
            self.adv_steps = int(max(self.adv_steps_min, min(self.adv_steps_max, self.adv_steps)))
        return self

    def apply_scenario_adjustments(self) -> "IVAPCIV33TheoryConfig":
        """根据 scenario_name 检测场景类型并应用特定调整。
        
        最终配置 (v6.3):
        - HEAVY_TAIL: Huber损失 + Winsorize + 更严格的裁剪 (Delta: +0.74 → -1.94)
        - MISALIGNED_PROXIES: Proximal Bridge + 分离式 BridgeHead (Delta: +1.14 → +0.36)
        - WEAK_OVERLAP: ATO + IPW裁剪 (Delta: +0.59 → +0.46)
        """
        if self.scenario_name:
            self.scenario_type = detect_scenario_type(self.scenario_name)
            self.scenario_config = get_scenario_config(self.scenario_type)
            
            # 场景特定参数覆盖
            if self.scenario_type == ScenarioType.HEAVY_TAIL:
                # Heavy-tail: 保持原来的标准 DR 方法 + Huber损失 + Winsorize
                # 不使用 proximal 方法，因为 Heavy-tail 不是 proximal 场景
                self.std_clip = 5.0  # 从10.0降低
                self.ipw_cap = 8.0   # 从15.0降低
                self.early_stopping_patience = 25  # 增加耐心防止过早停止
                
            elif self.scenario_type == ScenarioType.MISALIGNED_PROXIES:
                # v9.0: Use plug-in (proximal_bridge) for Misaligned scenario
                #
                # Experiments v8.5-v8.9 showed:
                # - plug-in: RMSE=1.08, negative bias, stable
                # - DR: RMSE=1.35-1.78, positive bias, unstable
                #
                # The q-bridge correction often "over-corrects" in misaligned scenarios.
                # Therefore, use the more robust plug-in estimator.
                self.estimand = "proximal_bridge"
                
                # Q-bridge configuration (trained but not used for final estimate)
                self.enable_q_bridge = True
                self.q_bridge_hidden = (128, 64)
                self.q_critic_hidden = (64, 32)
                self.lambda_q_bridge = 0.5
                self.lambda_q_bridge_adv = 1.0
                self.lambda_q_bridge_mean = 0.1
                self.lambda_q_bridge_l2 = 1e-3
                self.q_critic_steps = 3
                self.q_bridge_warmup_epochs = 10
                self.q_bridge_target_clip = (0.5, 5.0)
                
                # Outcome bridge configuration
                self.lambda_bridge = 1.0
                self.lambda_bridge_hsic = 0.2
                self.bridge_hidden = (256, 128, 64)
                self.bridge_warmup_epochs = 10
                
                # Cross-fitting disabled for plug-in
                self.enable_cross_fitting = False
                
                # ✅ proximal: don't purge proxy signals
                self.gamma_adv_w = 0.0
                self.gamma_adv_z = 0.0
                self.gamma_adv_n = 0.0
                self.gamma_adv_w_cond = 0.0
                self.gamma_adv_z_cond = 0.0
                self.gamma_adv_n_cond = 0.0
                self.lambda_hsic = 0.0
                self.lambda_ortho = 0.0
                self.lambda_cond_ortho = 0.0
                self.lambda_hsic_w_a = 0.0
                
                # Disable scheduler feedback and weak-IV adaptive
                self.use_scheduler_feedback = False
                self.weak_iv_adaptive.enabled = False
                self.enable_gated_routing = False
                
            elif self.scenario_type == ScenarioType.WEAK_OVERLAP:
                # v6: ATO + IPW裁剪
                # 自动切换到 ato estimand
                self.estimand = "ato"
                self.clip_prop = 0.10
                self.clip_prop_adaptive_max = 0.20
                self.propensity_shrinkage = 0.15
                self.ess_target = min(self.ess_target * 2.0, 0.9)
                self.ipw_cap = 5.0
                self.ipw_cap_quantile = 0.98
                
        return self


def adaptive_regularization_schedule(
    epoch: int, total_epochs: int, diagnostics: dict, config: IVAPCIV33TheoryConfig,
) -> tuple[float, float, float]:
    """Dynamic adjustment of adversarial and HSIC weights.
    
    NOTE: In proximal_dr mode, we skip all adjustments and return 0s,
    because proximal identification needs to preserve proxy↔treatment/outcome associations.
    """
    # ✅ proximal_dr 模式下直接返回 0，不做任何调度
    estimand = str(getattr(config, "estimand", "ate")).lower().strip()
    if estimand in {"proximal_dr", "proximal", "proximal_bridge"}:
        return 0.0, 0.0, 0.0
    
    # ✅ 如果显式关闭了调度反馈，也返回基础值（不 boost）
    if not getattr(config, "use_scheduler_feedback", True):
        return config.gamma_adv_w, config.gamma_adv_z, config.lambda_hsic
    
    base_w = config.gamma_adv_w
    base_z = config.gamma_adv_z
    base_hsic = config.lambda_hsic

    if epoch < total_epochs * 0.3:
        phase_scale = 0.7
    elif epoch < total_epochs * 0.7:
        phase_scale = 1.0
    else:
        phase_scale = 1.2

    info_loss = diagnostics.get("info_loss_proxy", 0.015)
    if info_loss > 0.018:
        info_scale = 0.8
    elif info_loss < 0.014:
        info_scale = 1.1
    else:
        info_scale = 1.0

    w_auc = diagnostics.get("rep_auc_w_to_a", 0.5)
    w_dev = abs(w_auc - 0.5)
    if w_dev > 0.18:
        w_boost = 1.5
    elif w_dev > 0.12:
        w_boost = 1.2
    else:
        w_boost = 1.0

    z_r2 = diagnostics.get("rep_exclusion_leakage_r2", 0.0)
    if z_r2 > 0.15:
        z_boost = 1.3
    elif z_r2 > 0.10:
        z_boost = 1.1
    else:
        z_boost = 1.0

    gamma_w = base_w * phase_scale * info_scale * w_boost
    gamma_z = base_z * phase_scale * z_boost
    lambda_h = base_hsic * phase_scale * info_scale

    gamma_w = float(np.clip(gamma_w, 0.05, 0.4))
    gamma_z = float(np.clip(gamma_z, 0.05, 0.3))
    lambda_h = float(np.clip(lambda_h, 0.005, 0.10))

    return gamma_w, gamma_z, lambda_h


class SmartAdversarialScheduler:
    """Warmup + cosine ramp + feedback scheduler for adversarial strengths."""

    def __init__(
        self,
        base_gamma_w: float,
        base_gamma_z: float,
        warmup_epochs: int = 10,
        ramp_epochs: int = 30,
        target_w_auc: float = 0.5,
        target_z_r2: float = 0.1,
        use_feedback: bool = True,
    ) -> None:
        self.base_gamma_w = base_gamma_w
        self.base_gamma_z = base_gamma_z
        self.warmup = warmup_epochs
        self.ramp = ramp_epochs
        self.target_w_auc = target_w_auc
        self.target_z_r2 = target_z_r2
        self.use_feedback = use_feedback

    def step(self, epoch: int, total_epochs: int, diagnostics: dict) -> tuple[float, float]:
        if epoch < self.warmup:
            return 0.0, 0.0

        if epoch < self.warmup + self.ramp:
            prog = (epoch - self.warmup) / max(1, self.ramp)
            ramp_factor = 0.5 * (1 - np.cos(np.pi * prog))
        else:
            ramp_factor = 1.0

        base_w = self.base_gamma_w * ramp_factor
        base_z = self.base_gamma_z * ramp_factor

        if self.use_feedback:
            w_auc_val = diagnostics.get("rep_auc_w_to_a", None)
            z_r2_val = diagnostics.get("rep_exclusion_leakage_r2", None)

            # 如果诊断值缺失，不要默认触发“降权”(0.9)，避免把Bayes/网格给的超参隐式改掉
            if w_auc_val is None:
                w_mult = 1.0
            else:
                w_auc = float(w_auc_val)
                w_dev = abs(w_auc - self.target_w_auc)
                if w_dev > 0.15:
                    w_mult = 1.3
                elif w_dev > 0.08:
                    w_mult = 1.1
                elif w_dev < 0.05:
                    w_mult = 0.9
                else:
                    w_mult = 1.0

            if z_r2_val is None:
                z_mult = 1.0
            else:
                z_r2 = float(z_r2_val)
                if z_r2 > 0.18:
                    z_mult = 1.3
                elif z_r2 > 0.12:
                    z_mult = 1.1
                elif z_r2 < 0.08:
                    z_mult = 0.9
                else:
                    z_mult = 1.0
        else:
            w_mult = 1.0
            z_mult = 1.0

        gamma_w = float(np.clip(base_w * w_mult, 0.0, self.base_gamma_w * 2.0))
        gamma_z = float(np.clip(base_z * z_mult, 0.0, self.base_gamma_z * 2.0))
        return gamma_w, gamma_z


# =============================================================================
# Estimator (with Weak-IV Adaptive Integration)
# =============================================================================

class IVAPCIv33TheoryHierEstimator(BaseCausalEstimator):
    """Hierarchical encoder with theorem-1-5 components + Weak-IV Adaptive."""

    def __init__(self, config: Optional[IVAPCIV33TheoryConfig] = None):
        self.config = config or IVAPCIV33TheoryConfig()
        if self.config.n_samples_hint is not None:
            self.config.apply_theorem45_defaults()
        # 应用场景安全覆盖（如 std_clip/ipw_cap/clip_prop 等）
        self.config.apply_scenario_adjustments()
        self.device = torch.device(self.config.device)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self._is_fit = False
        self._protected_mask: Optional[np.ndarray] = None
        self._weak_iv_flag: bool = False
        self.training_diagnostics: Dict[str, float] = {}
        self.info_monitor = InformationLossMonitor()
        self._weak_iv_scheduler: Optional[WeakIVAdaptiveScheduler] = None

    def _identifiability_checks(self, V_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        """Pre-training identifiability checks."""
        cfg = self.config
        n = len(A)

        # IV strength check
        z_slice = slice(cfg.x_dim + cfg.w_dim, cfg.x_dim + cfg.w_dim + cfg.z_dim)
        Z_block = V_all[:, z_slice]
        correlations = []
        for j in range(Z_block.shape[1]):
            corr = np.abs(np.corrcoef(Z_block[:, j], A)[0, 1])
            if not np.isnan(corr):
                correlations.append(corr)
        max_corr = max(correlations) if correlations else 0.0
        self.training_diagnostics["iv_relevance_abs_corr"] = float(max_corr)

        # Weak IV detection
        if max_corr < 0.15:
            self._weak_iv_flag = True
            self.training_diagnostics["weak_iv_warning"] = (
                "First-stage F < 10: weak IV detected; using conservative DR."
            )
        self.training_diagnostics["weak_iv_flag"] = self._weak_iv_flag

        # First-stage F-statistic proxy
        try:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(Z_block, A)
            r2 = lr.score(Z_block, A)
            k = Z_block.shape[1]
            f_stat = (r2 / k) / ((1 - r2) / max(1, n - k - 1))
            self.training_diagnostics["iv_first_stage_f"] = float(f_stat)
        except Exception:
            pass

    def _compute_representation_diagnostics(
        self,
        txn: np.ndarray, twn: np.ndarray, tzn: np.ndarray,
        A: np.ndarray, Y: np.ndarray
    ) -> None:
        """Post-fit representation diagnostics."""
        try:
            n = len(A)
            a_vals = np.unique(A)
            is_binary01 = bool(a_vals.size == 2 and np.all(np.isin(a_vals, [0, 1])))
            A01 = A.astype(int) if is_binary01 else None
            strat = A01 if (A01 is not None and np.min(np.bincount(A01)) >= 2) else None
            idx = np.arange(n)
            idx_tr, idx_te = train_test_split(idx, test_size=0.3, random_state=0, stratify=strat)

            def _auc(feat: np.ndarray) -> float:
                if A01 is None or feat.size == 0:
                    return float("nan")
                lr = LogisticRegression(max_iter=1000)
                lr.fit(feat[idx_tr], A01[idx_tr])
                p = lr.predict_proba(feat[idx_te])[:, 1]
                return float(roc_auc_score(A01[idx_te], p))

            def _r2(feat: np.ndarray) -> float:
                if feat.size == 0 or np.var(Y[idx_te]) == 0:
                    return float("nan")
                reg = Ridge(alpha=1.0)
                reg.fit(feat[idx_tr], Y[idx_tr])
                pred = reg.predict(feat[idx_te])
                return float(r2_score(Y[idx_te], pred))

            auc_z_a = _auc(tzn) if tzn.ndim == 2 else _auc(tzn.reshape(-1, 1))
            auc_w_a = _auc(twn) if twn.ndim == 2 else _auc(twn.reshape(-1, 1))

            feat_xw_a = np.column_stack([A, txn, twn])
            feat_z_a = np.column_stack([A, tzn])

            r2_xw_a_y = _r2(feat_xw_a)
            r2_z_a_y = _r2(feat_z_a)

            self.training_diagnostics.update({
                "rep_auc_z_to_a": auc_z_a,
                "rep_auc_w_to_a": auc_w_a,
                "rep_r2_xw_a_to_y": r2_xw_a_y,
                "rep_r2_z_a_to_y": r2_z_a_y,
                "rep_exclusion_leakage_r2": float(r2_z_a_y) if np.isfinite(r2_z_a_y) else float("nan"),
                "rep_exclusion_leakage_gap": float(r2_z_a_y - r2_xw_a_y)
                    if (np.isfinite(r2_z_a_y) and np.isfinite(r2_xw_a_y)) else float("nan"),
            })
        except Exception:
            return

    @staticmethod
    def _toggle_requires_grad(modules: Iterable[nn.Module], flag: bool) -> None:
        for mod in modules:
            for p in mod.parameters():
                p.requires_grad = flag

    def _build(self, d_all: int) -> None:
        cfg = self.config
        if not (cfg.x_dim and cfg.w_dim and cfg.z_dim):
            raise ValueError("IVAPCI v3.3 requires x_dim, w_dim, z_dim")
        if cfg.x_dim + cfg.w_dim + cfg.z_dim != d_all:
            raise ValueError("x_dim + w_dim + z_dim must equal input dimension")

        self._block_slices = (
            slice(0, cfg.x_dim),
            slice(cfg.x_dim, cfg.x_dim + cfg.w_dim),
            slice(cfg.x_dim + cfg.w_dim, cfg.x_dim + cfg.w_dim + cfg.z_dim),
        )

        self.enc_x = _GroupEncoder(cfg.x_dim, cfg.enc_x_hidden, cfg.latent_x_dim).to(self.device)
        self.enc_w = _GroupEncoder(cfg.w_dim, cfg.enc_w_hidden, cfg.latent_w_dim).to(self.device)
        self.enc_z = _GroupEncoder(cfg.z_dim, cfg.enc_z_hidden, cfg.latent_z_dim, dropout=cfg.dropout_z).to(self.device)
        self.enc_n = _GroupEncoder(d_all, cfg.enc_n_hidden, cfg.latent_n_dim).to(self.device)

        total_lat = cfg.latent_x_dim + cfg.latent_w_dim + cfg.latent_z_dim + cfg.latent_n_dim
        self.decoder = _Decoder(total_lat, cfg.dec_hidden, d_all).to(self.device)

        a_in_dim = cfg.latent_x_dim + cfg.latent_z_dim + (cfg.latent_w_dim if cfg.enable_gated_routing else 0)
        y_in_dim = cfg.latent_x_dim + cfg.latent_w_dim + (cfg.latent_z_dim if cfg.enable_gated_routing else 0)
        self.a_head = _AClassifier(a_in_dim, cfg.a_hidden).to(self.device)
        self.y_head = _YRegressor(y_in_dim, cfg.y_hidden).to(self.device)

        # Dedicated propensity head e(W,X) for treatment-bridge targets (proximal DR)
        # v9.0: Use RAW (W,X) instead of latent - critical for correct conditioning!
        w_raw_dim = self._block_slices[1].stop - self._block_slices[1].start
        x_raw_dim = self._block_slices[0].stop - self._block_slices[0].start
        z_raw_dim = self._block_slices[2].stop - self._block_slices[2].start
        
        # Keep latent version for backward compatibility
        self.a_wx_head = _AClassifier(cfg.latent_x_dim + cfg.latent_w_dim, cfg.a_hidden).to(self.device)
        # v9.0: RAW version - this is what proximal theory actually requires
        self.a_wx_raw_head = _AClassifier(w_raw_dim + x_raw_dim, cfg.a_hidden).to(self.device)

        # Proximal outcome bridge head: h(W,A,X) = mu(W,X) + A*tau(W,X)
        # Satisfies: E[Y - h(W,A,X) | A,X,Z] = 0
        bridge_in_dim = cfg.latent_x_dim + cfg.latent_w_dim
        self.bridge_head = _BridgeHead(bridge_in_dim, cfg.bridge_hidden).to(self.device)
        
        # Proximal treatment bridge head: q(Z,A,X)
        # Satisfies: E[q(Z,a,X) | W,X,A=a] = 1/P(A=a|W,X)
        # v9.0: Use RAW (Z,X) for q-bridge to match the conditioning space
        if getattr(cfg, "enable_q_bridge", True):
            # Keep latent version for backward compatibility
            self.q_bridge_head = _QBridgeHead(
                cfg.latent_z_dim, 
                cfg.latent_x_dim, 
                getattr(cfg, "q_bridge_hidden", (128, 64))
            ).to(self.device)
            # v9.0: RAW version - matches the theoretical bridge equation
            self.q_bridge_raw_head = _QBridgeHead(
                z_raw_dim,
                x_raw_dim,
                getattr(cfg, "q_bridge_hidden", (128, 64))
            ).to(self.device)
            # v9.0: Critic with A input - enables per-treatment-group moment conditions
            # E[q(Z,1,X) - 1/e | W,X,A=1] = 0 and E[q(Z,0,X) - 1/(1-e) | W,X,A=0] = 0
            self.q_bridge_critic = _QBridgeCritic(
                w_raw_dim + x_raw_dim + 1,  # +1 for treatment indicator A
                1,  # dummy x_dim (not used separately)
                getattr(cfg, "q_critic_hidden", (64, 32))
            ).to(self.device)
        else:
            self.q_bridge_head = None
            self.q_bridge_raw_head = None
            self.q_bridge_critic = None

        # Gates (soft routing): allow tx-dependent interactions to flow across routes
        if cfg.enable_gated_routing:
            self.gate_aw = _Gate(cfg.latent_x_dim, cfg.latent_w_dim, init_bias=cfg.gate_init_bias).to(self.device)
            self.gate_yz = _Gate(cfg.latent_x_dim, cfg.latent_z_dim, init_bias=cfg.gate_init_bias).to(self.device)
        else:
            self.gate_aw = None
            self.gate_yz = None

        self.a_from_z = _AClassifier(cfg.latent_z_dim, (32,)).to(self.device)
        self.y_from_w = _YRegressor(cfg.latent_w_dim, (64, 32)).to(self.device)

        self.adv_w = _AClassifier(cfg.latent_w_dim, cfg.adv_a_hidden).to(self.device)
        self.adv_n = _AClassifier(cfg.latent_n_dim, cfg.adv_a_hidden).to(self.device)
        self.adv_z = _YAdversary(cfg.latent_z_dim, cfg.adv_y_hidden).to(self.device)
        self.adv_w_cond = TrueConditionalAdversary(
            cfg.latent_w_dim, cfg.latent_x_dim, cfg.adv_a_hidden, loss="bce"
        ).to(self.device)
        self.adv_z_cond = TrueConditionalAdversary(
            cfg.latent_z_dim, cfg.latent_x_dim + 1, cfg.adv_y_hidden, loss="mse"
        ).to(self.device)
        self.adv_n_cond = TrueConditionalAdversary(
            cfg.latent_n_dim, cfg.latent_x_dim + cfg.latent_w_dim + cfg.latent_z_dim, cfg.adv_a_hidden, loss="bce"
        ).to(self.device)

        main_params = (
            list(self.enc_x.parameters()) + list(self.enc_w.parameters())
            + list(self.enc_z.parameters()) + list(self.enc_n.parameters())
            + list(self.decoder.parameters()) + list(self.a_head.parameters())
            + list(self.y_head.parameters()) + list(self.a_from_z.parameters())
            + list(self.y_from_w.parameters())
            + list(self.bridge_head.parameters())
            + (list(self.gate_aw.parameters()) + list(self.gate_yz.parameters()) if cfg.enable_gated_routing else [])
        )
        self.main_opt = torch.optim.Adam(main_params, lr=cfg.lr_main)
        self.main_sched = ReduceLROnPlateau(self.main_opt, mode="min", factor=0.5, patience=10)

        adv_params = (
            list(self.adv_w.parameters()) + list(self.adv_z.parameters())
            + list(self.adv_n.parameters()) + list(self.adv_w_cond.parameters())
            + list(self.adv_z_cond.parameters()) + list(self.adv_n_cond.parameters())
        )
        self.adv_opt = torch.optim.Adam(adv_params, lr=cfg.lr_adv)
        
        # Q-bridge optimizer (separate, starts later)
        # v9.0: Include both latent and raw heads
        if self.q_bridge_head is not None:
            q_params = list(self.q_bridge_head.parameters())
            if hasattr(self, 'q_bridge_raw_head') and self.q_bridge_raw_head is not None:
                q_params += list(self.q_bridge_raw_head.parameters())
            self.q_bridge_opt = torch.optim.Adam(q_params, lr=cfg.lr_main)
            # Critic optimizer for adversarial moment training
            self.q_critic_opt = torch.optim.Adam(
                list(self.q_bridge_critic.parameters()),
                lr=cfg.lr_main * 2  # Critic typically needs higher LR
            )
        else:
            self.q_bridge_opt = None
            self.q_critic_opt = None

        # Propensity optimizer for e_hat(W,X) used by q-bridge targets
        # v9.0: Use RAW (W,X) version
        prop_params = list(self.a_wx_head.parameters())
        if hasattr(self, 'a_wx_raw_head') and self.a_wx_raw_head is not None:
            prop_params += list(self.a_wx_raw_head.parameters())
        self.prop_wx_opt = torch.optim.Adam(
            prop_params, lr=cfg.lr_main
        ) if getattr(cfg, "enable_q_bridge", False) else None
            
        self.dtype = next(self.enc_x.parameters()).dtype

    def _split_blocks_tensor(self, V_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            V_t[:, self._block_slices[0]],
            V_t[:, self._block_slices[1]],
            V_t[:, self._block_slices[2]],
        )

    def _encode_blocks(self, V_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_dtype = next(self.enc_x.parameters()).dtype
        if V_t.dtype != target_dtype:
            V_t = V_t.to(dtype=target_dtype)
        x_part, w_part, z_part = self._split_blocks_tensor(V_t)
        tx = self.enc_x(x_part)
        tw = self.enc_w(w_part)
        tz = self.enc_z(z_part)
        tn = self.enc_n(V_t)
        return tx, tw, tz, tn

    def _quick_monitor(self, V_std: np.ndarray, A: np.ndarray, Y_std: np.ndarray, max_n: int) -> Dict[str, float]:
        """Lightweight in-training diagnostics."""
        n = int(A.shape[0])
        if n < 8:
            return {"rep_auc_w_to_a": 0.5, "rep_exclusion_leakage_r2": 0.0, "overlap_ess_min": 0.0, "info_loss_proxy": 0.02, "rep_auc_z_to_a": 0.5}

        k = int(min(max_n, n))
        idx = np.random.choice(n, size=k, replace=False)
        vb = torch.from_numpy(V_std[idx]).to(self.device, dtype=self.dtype)
        ab = torch.from_numpy(A[idx]).to(self.device, dtype=self.dtype)
        yb = torch.from_numpy(Y_std[idx]).to(self.device, dtype=self.dtype)

        modules = [self.enc_x, self.enc_w, self.enc_z, self.enc_n, self.decoder, self.a_head, self.adv_w, self.adv_z, self.a_from_z]
        train_states = [m.training for m in modules]
        for m in modules:
            m.eval()

        with torch.no_grad():
            tx, tw, tz, tn = self._encode_blocks(vb)
            w_logits = self.adv_w(tw).view(-1).cpu().numpy()
            z_pred = self.adv_z(tz).view(-1).cpu().numpy()
            e_hat = torch.sigmoid(self.a_head(self._compose_a_t(tx, tw, tz))).view(-1).cpu().numpy()
            recon = self.decoder(torch.cat([tx, tw, tz, tn], dim=1))
            info_loss = float(torch.mean((recon - vb) ** 2).detach().cpu().item())
            
            # 🔑 关键修复：计算 rep_auc_z_to_a 用于在线 ξ_rep 更新
            z_to_a_logits = torch.sigmoid(self.a_from_z(tz)).view(-1).cpu().numpy()

            try:
                y_np = yb.cpu().numpy()
                tx_np = tx.cpu().numpy()
                tw_np = tw.cpu().numpy()
                tz_np = tz.cpu().numpy()
                cov = np.column_stack([np.ones_like(y_np), ab.cpu().numpy(), tx_np, tw_np])
                coef, *_ = np.linalg.lstsq(cov, y_np, rcond=None)
                y_resid = y_np - cov @ coef
                lr_coef, *_ = np.linalg.lstsq(tz_np, y_resid, rcond=None)
                y_resid_hat = tz_np @ lr_coef
                ss_res = float(np.sum((y_resid - y_resid_hat) ** 2))
                ss_tot = float(np.sum((y_resid - y_resid.mean()) ** 2) + 1e-12)
                z_r2_cond = 1.0 - ss_res / ss_tot
            except Exception:
                z_r2_cond = float("nan")

        for m, state in zip(modules, train_states):
            m.train(mode=state)

        try:
            w_auc = float(roc_auc_score(ab.cpu().numpy(), w_logits))
        except Exception:
            w_auc = float("nan")
        
        # 🔑 计算 Z→A 的 AUC (rep_auc_z_to_a)
        try:
            z_to_a_auc = float(roc_auc_score(ab.cpu().numpy(), z_to_a_logits))
        except Exception:
            z_to_a_auc = 0.5  # 默认随机

        z_r2 = float(r2_score(yb.cpu().numpy(), z_pred)) if np.var(z_pred) > 0 else float("nan")
        e_hat = np.clip(e_hat, 1e-6, 1 - 1e-6)
        ess_t, ess_c = compute_ess_by_group(e_hat, ab.cpu().numpy())
        ess_min = float(min(ess_t, ess_c))

        return {
            "rep_auc_w_to_a": w_auc,
            "rep_auc_z_to_a": z_to_a_auc,  # 🔑 新增：用于在线 ξ_rep 更新
            "rep_exclusion_leakage_r2": z_r2,
            "gate_aw_mean": float(torch.sigmoid(self.gate_aw.lin.bias).mean().detach().cpu().item()) if (getattr(self.config, "enable_gated_routing", False) and getattr(self, "gate_aw", None) is not None) else 0.0,
            "gate_yz_mean": float(torch.sigmoid(self.gate_yz.lin.bias).mean().detach().cpu().item()) if (getattr(self.config, "enable_gated_routing", False) and getattr(self, "gate_yz", None) is not None) else 0.0,
            "rep_exclusion_leakage_r2_cond": z_r2_cond,
            "overlap_ess_min": ess_min,
            "info_loss_proxy": info_loss,
        }

    # === Gated soft routing helpers ===
    def _compose_a_t(self, tx: torch.Tensor, tw: torch.Tensor, tz: torch.Tensor) -> torch.Tensor:
        """Input to a_head: [tx, tz] plus tx-gated tw (if enabled)."""
        if getattr(self.config, "enable_gated_routing", False) and (self.gate_aw is not None):
            g_aw = self.gate_aw(tx)
            return torch.cat([tx, tz, g_aw * tw], dim=1)
        return torch.cat([tx, tz], dim=1)

    def _compose_y_t(self, tx: torch.Tensor, tw: torch.Tensor, tz: torch.Tensor) -> torch.Tensor:
        """Input to y_head: [tx, tw] plus tx-gated tz (if enabled)."""
        if getattr(self.config, "enable_gated_routing", False) and (self.gate_yz is not None):
            g_yz = self.gate_yz(tx)
            return torch.cat([tx, tw, g_yz * tz], dim=1)
        return torch.cat([tx, tw], dim=1)

    def _gate_penalty(self, tx: torch.Tensor) -> torch.Tensor:
        """L1-like penalty to keep gates mostly closed (prevents harming non-misaligned scenarios)."""
        if getattr(self.config, "enable_gated_routing", False) and (self.gate_aw is not None) and (self.gate_yz is not None):
            g_aw = self.gate_aw(tx)
            g_yz = self.gate_yz(tx)
            return g_aw.mean() + g_yz.mean()
        return torch.zeros((), device=tx.device)

    def fit(self, V_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> None:
        V_all = np.asarray(V_all, dtype=np.float32)
        A = np.asarray(A, dtype=np.float32).reshape(-1)
        Y = np.asarray(Y, dtype=np.float32).reshape(-1)
        # 可选：Heavy-tail 场景下对 Y 做 winsorize（先于标准化）
        if getattr(self.config, 'scenario_config', None) is not None and self.config.scenario_config.winsorize_y:
            Y = _winsorize_1d(Y, self.config.scenario_config.winsorize_quantile)
        n, d_all = V_all.shape
        cfg = self.config

        self._weak_iv_flag = False

        if cfg.n_samples_hint is None:
            cfg.n_samples_hint = int(n)
        # 总是调用 apply_theorem45_defaults() 以确保参数一致性
        if cfg.adaptive:
            cfg.apply_theorem45_defaults()

        cfg.apply_scenario_adjustments()

        self._identifiability_checks(V_all, A, Y)
        # 记录场景信息（如果启用）
        self.training_diagnostics["scenario_type"] = str(getattr(cfg, 'scenario_type', ScenarioType.DEFAULT).value)

        # === 弱IV自适应调度器初始化 ===
        if cfg.weak_iv_adaptive.enabled:
            z_slice = slice(cfg.x_dim + cfg.w_dim, cfg.x_dim + cfg.w_dim + cfg.z_dim)
            Z_block = V_all[:, z_slice]
            
            self._weak_iv_scheduler = WeakIVAdaptiveScheduler(
                n_samples=n,
                config=cfg.weak_iv_adaptive
            )
            
            # 计算场景固有IV强度
            xi_raw = self._weak_iv_scheduler.compute_xi_raw_from_data(Z_block, A)
            self._weak_iv_scheduler.xi_raw = xi_raw
            self.training_diagnostics["weak_iv_xi_raw"] = xi_raw
            
            if xi_raw < cfg.weak_iv_adaptive.xi_enter_weak:
                print(f"🔍 检测到弱IV场景: ξ_raw={xi_raw:.4f} < {cfg.weak_iv_adaptive.xi_enter_weak}")
                print(f"   将启用弱IV自适应调整")

        # Train/val split
        uniq, counts = np.unique(A, return_counts=True)
        stratify_arr = None
        if uniq.size > 1:
            test_size = max(1, int(round(cfg.val_frac * n)))
            if test_size >= uniq.size and counts.min() >= 2:
                stratify_arr = A
        tr_idx, va_idx = train_test_split(np.arange(n), test_size=cfg.val_frac, random_state=cfg.seed, stratify=stratify_arr)
        V_tr, V_va = V_all[tr_idx], V_all[va_idx]
        A_tr, A_va = A[tr_idx], A[va_idx]
        Y_tr, Y_va = Y[tr_idx], Y[va_idx]

        V_tr_std, self._v_mean, self._v_std, protected = _info_aware_standardize(
            V_tr, min_std=cfg.min_std, low_var_min_std=cfg.low_var_min_std, clip_value=cfg.std_clip
        )
        self._protected_mask = protected
        V_va_std = _apply_standardize(V_va, self._v_mean, self._v_std)
        if cfg.std_clip is not None:
            V_tr_std = np.clip(V_tr_std, -cfg.std_clip, cfg.std_clip)
            V_va_std = np.clip(V_va_std, -cfg.std_clip, cfg.std_clip)

        Y_tr_std, self._y_mean, self._y_std, _ = _info_aware_standardize(
            Y_tr.reshape(-1, 1), min_std=1e-6, low_var_min_std=1e-6, clip_value=cfg.std_clip
        )
        Y_tr_std = Y_tr_std.squeeze(1)
        Y_va_std = _apply_standardize(Y_va.reshape(-1, 1), self._y_mean, self._y_std).squeeze(1)

        self._build(d_all)
        target_dtype = next(self.enc_x.parameters()).dtype

        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()

        # Scenario-aware robust loss functions
        sc = getattr(cfg, 'scenario_config', ScenarioSpecificConfig())
        huber_loss_fn = HuberLoss(delta=sc.huber_delta).to(self.device)
        focal_bce_fn = FocalBCELoss(gamma=sc.focal_gamma).to(self.device)

        tr_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(V_tr_std), torch.from_numpy(A_tr), torch.from_numpy(Y_tr_std)
            ),
            batch_size=cfg.batch_size, shuffle=True,
        )
        va_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(V_va_std), torch.from_numpy(A_va), torch.from_numpy(Y_va_std)
            ),
            batch_size=cfg.batch_size, shuffle=False,
        )

        self._adv_scheduler = SmartAdversarialScheduler(
            base_gamma_w=cfg.gamma_adv_w,
            base_gamma_z=cfg.gamma_adv_z,
            warmup_epochs=cfg.adv_warmup_epochs,
            ramp_epochs=cfg.adv_ramp_epochs,
            target_w_auc=getattr(cfg, 'target_w_auc', 0.52),
            target_z_r2=getattr(cfg, 'target_z_r2', 0.10),
            use_feedback=getattr(cfg, 'use_scheduler_feedback', True),
        )

        # Stage 0: reconstruction pretraining
        for _ in range(cfg.epochs_pretrain):
            self.enc_x.train(); self.enc_w.train(); self.enc_z.train(); self.enc_n.train(); self.decoder.train()
            for vb, _, _ in tr_loader:
                vb = vb.to(self.device)
                tx, tw, tz, tn = self._encode_blocks(vb)
                recon = self.decoder(torch.cat([tx, tw, tz, tn], dim=1))
                loss = mse(recon, vb)
                self.main_opt.zero_grad(); loss.backward(); self.main_opt.step()

        best_val = float("inf")
        patience = 0
        best_state = None

        # Stage 1: full training with Weak-IV Adaptive
        for epoch in range(cfg.epochs_main):
            # Diagnostics
            if cfg.monitor_every and cfg.monitor_every > 0 and (epoch % cfg.monitor_every) == 0:
                diag_now = self._quick_monitor(V_tr_std, A_tr, Y_tr_std, max_n=cfg.monitor_batch_size)
                if not hasattr(self, "_diag_ema"):
                    self._diag_ema = dict(diag_now)
                else:
                    ema = float(cfg.monitor_ema)
                    for k, v in diag_now.items():
                        prev = self._diag_ema.get(k, v)
                        self._diag_ema[k] = float(ema * prev + (1.0 - ema) * v)
                self.training_diagnostics.update(self._diag_ema)

            # === 弱IV自适应调度 ===
            weak_iv_params = None
            weak_iv_loss_weight = 0.0
            weak_iv_excl_penalty = 0.0
            weak_iv_relev_penalty = 0.0

            lambda_hsic_adaptive = cfg.lambda_hsic
            lambda_cons_adaptive = cfg.lambda_consistency

            if self._weak_iv_scheduler is not None and cfg.weak_iv_adaptive.enabled:
                weak_iv_params = self._weak_iv_scheduler.step(epoch, self.training_diagnostics)

                if weak_iv_params.get('weak_iv_mode', False):
                    # 在弱IV模式下应用自适应参数
                    lambda_hsic_adaptive = float(weak_iv_params.get('lambda_hsic_adaptive', cfg.lambda_hsic))
                    lambda_cons_adaptive = float(weak_iv_params.get('lambda_cons_adaptive', cfg.lambda_consistency))

                    # 弱IV特化损失（拆开记录，便于诊断）
                    weak_iv_excl_penalty = float(weak_iv_params.get('excl_penalty', 0.0))
                    weak_iv_relev_penalty = float(weak_iv_params.get('relev_penalty', 0.0))
                    weak_iv_loss_weight = float(weak_iv_excl_penalty + weak_iv_relev_penalty)

                    # 记录诊断，供benchmark/summary聚合
                    self.training_diagnostics["weak_iv_mode"] = True
                    self.training_diagnostics["weak_iv_xi_eff"] = float(weak_iv_params.get('xi_eff', 0.0))
                    self.training_diagnostics["weak_iv_gamma_z"] = float(weak_iv_params.get('gamma_z_adaptive', 0.0))
                    self.training_diagnostics["weak_iv_lambda_hsic"] = float(lambda_hsic_adaptive)
                    self.training_diagnostics["weak_iv_lambda_cons"] = float(lambda_cons_adaptive)
                    self.training_diagnostics["weak_iv_excl_penalty"] = float(weak_iv_excl_penalty)
                    self.training_diagnostics["weak_iv_relev_penalty"] = float(weak_iv_relev_penalty)
                    self.training_diagnostics["weak_iv_loss"] = float(weak_iv_loss_weight)

                    # 每20个epoch打印状态
                    if epoch % 20 == 0:
                        print(
                            f"  📊 弱IV自适应 (epoch {epoch}): ξ_eff={self.training_diagnostics['weak_iv_xi_eff']:.3f}, "
                            f"λ_hsic={lambda_hsic_adaptive:.4f}, λ_cons={lambda_cons_adaptive:.4f}, "
                            f"excl={weak_iv_excl_penalty:.4e}, relev={weak_iv_relev_penalty:.4e}"
                        )
                else:
                    self.training_diagnostics["weak_iv_mode"] = False

            self.enc_x.train(); self.enc_w.train(); self.enc_z.train(); self.enc_n.train()
            self.decoder.train(); self.a_head.train(); self.y_head.train()
            self.a_from_z.train(); self.y_from_w.train()
            self.adv_w.train(); self.adv_z.train(); self.adv_n.train()
            self.adv_w_cond.train(); self.adv_z_cond.train(); self.adv_n_cond.train()

            _, _, lambda_hsic_ep = adaptive_regularization_schedule(
                epoch, cfg.epochs_main, getattr(self, "training_diagnostics", {}), cfg
            )
            
            # 在弱IV模式下使用自适应HSIC
            if self._weak_iv_scheduler is not None and self._weak_iv_scheduler.weak_iv_mode:
                lambda_hsic_ep = lambda_hsic_adaptive

            # 记录最终使用的正则权重，便于summary聚合/排错
            self.training_diagnostics["lambda_hsic_use"] = float(lambda_hsic_ep)
            
            gamma_w_use, gamma_z_use = self._adv_scheduler.step(
                epoch, cfg.epochs_main, getattr(self, "training_diagnostics", {})
            )
            
            # ✅ proximal_dr 模式下跳过所有对抗调度（保留 proxy 信号）
            estimand_str = str(getattr(cfg, "estimand", "ate")).lower().strip()
            if estimand_str in {"proximal_dr", "proximal", "proximal_bridge"}:
                gamma_w_use = 0.0
                gamma_z_use = 0.0
                gamma_n_use = 0.0
                gamma_w_cond = 0.0
                gamma_z_cond = 0.0
                gamma_n_cond = 0.0
                cond_factor = 0.0  # ✅ 修复：也定义 cond_factor
                adv_steps_ep = 0
            else:
                # 🔑 关键修复：在弱IV模式下应用 gamma_z_adaptive (成对调参的核心)
                if self._weak_iv_scheduler is not None and self._weak_iv_scheduler.weak_iv_mode:
                    gamma_z_adaptive = weak_iv_params.get("gamma_z_adaptive", gamma_z_use)
                    gamma_z_use = max(gamma_z_use, gamma_z_adaptive)  # 取较大值以增强排除限制
                
                # 计算 factor 时避免除零
                w_factor = gamma_w_use / max(cfg.gamma_adv_w, 1e-8) if cfg.gamma_adv_w > 1e-8 else 0.0
                z_factor = gamma_z_use / max(cfg.gamma_adv_z, 1e-8) if cfg.gamma_adv_z > 1e-8 else 0.0
                cond_factor = 0.0
                if epoch >= cfg.cond_adv_warmup_epochs:
                    cond_factor = float(min(1.0, (epoch - cfg.cond_adv_warmup_epochs + 1) / max(1, cfg.cond_adv_ramp_epochs)))
                gamma_n_use = cfg.gamma_adv_n * w_factor
                gamma_w_cond = cfg.gamma_adv_w_cond * w_factor * cond_factor
                gamma_z_cond = cfg.gamma_adv_z_cond * z_factor * cond_factor
                gamma_n_cond = cfg.gamma_adv_n_cond * w_factor * cond_factor

                adv_steps_ep = cfg.adv_steps
                leak_boost = 1.0  # ✅ 默认值
                if cfg.adv_steps_dynamic:
                    diag_now = getattr(self, "training_diagnostics", {}) or {}
                    w_auc = float(diag_now.get("rep_auc_w_to_a", 0.5))
                    w_auc_eff = max(w_auc, 1.0 - w_auc)
                    z_r2 = float(diag_now.get("rep_exclusion_leakage_r2_cond", diag_now.get("rep_exclusion_leakage_r2", 0.0)))
                    if w_auc_eff > 0.58 or z_r2 > 0.16:
                        adv_steps_ep += 1
                    if w_auc_eff > 0.64 or z_r2 > 0.22:
                        adv_steps_ep += 1
                    adv_steps_ep = int(np.clip(adv_steps_ep, cfg.adv_steps_min, cfg.adv_steps_max))

                    if w_auc_eff > 0.64:
                        leak_boost = 1.2
                    elif w_auc_eff > 0.58:
                        leak_boost = 1.1
                    gamma_w_use *= leak_boost
                    gamma_w_cond *= leak_boost
                gamma_n_use *= leak_boost

            epoch_loss = 0.0
            epoch_batches = 0
            epoch_bridge_mse = 0.0
            epoch_bridge_hsic = 0.0
            epoch_bridge_batches = 0
            
            # 在弱IV模式下使用自适应consistency
            lambda_consistency_use = lambda_cons_adaptive if (
                self._weak_iv_scheduler is not None and self._weak_iv_scheduler.weak_iv_mode
            ) else cfg.lambda_consistency

            self.training_diagnostics["lambda_consistency_use"] = float(lambda_consistency_use)
            
            for vb, ab, yb in tr_loader:
                vb = vb.to(self.device, dtype=target_dtype)
                ab = ab.to(self.device, dtype=target_dtype)
                yb = yb.to(self.device, dtype=target_dtype)

                # Adversary update
                if adv_steps_ep > 0 and (gamma_w_use > 0 or gamma_z_use > 0 or gamma_n_use > 0):
                    for _ in range(adv_steps_ep):
                        with torch.no_grad():
                            tx_d, tw_d, tz_d, tn_d = self._encode_blocks(vb)
                        adv_loss = bce(self.adv_w(tw_d), ab) + bce(self.adv_n(tn_d), ab) + mse(self.adv_z(tz_d), yb)
                        if cond_factor > 0:
                            adv_loss = adv_loss + self.adv_w_cond(tw_d, tx_d, ab)
                            adv_loss = adv_loss + self.adv_z_cond(tz_d, torch.cat([tx_d, ab.unsqueeze(1)], dim=1), yb)
                            adv_loss = adv_loss + self.adv_n_cond(tn_d, torch.cat([tx_d, tw_d, tz_d], dim=1), ab)
                        self.adv_opt.zero_grad(); adv_loss.backward(); self.adv_opt.step()

                # Main update
                self._toggle_requires_grad(
                    [self.adv_w, self.adv_n, self.adv_z, self.adv_w_cond, self.adv_z_cond, self.adv_n_cond], False
                )
                tx, tw, tz, tn = self._encode_blocks(vb)
                recon = self.decoder(torch.cat([tx, tw, tz, tn], dim=1))
                logits_a = self.a_head(self._compose_a_t(tx, tw, tz))
                y_pred = self.y_head(self._compose_y_t(tx, tw, tz), ab)

                # === Proximal bridge (HSIC moment restriction) ===
                bridge_mse = torch.zeros((), device=self.device)
                bridge_hsic = torch.zeros((), device=self.device)
                bridge_loss = torch.zeros((), device=self.device)
                q_bridge_loss = torch.zeros((), device=self.device)
                
                estimand_str = str(getattr(cfg, "estimand", "ate")).lower().strip()
                if estimand_str in {"proximal_bridge", "proximal", "bridge", "proximal_dr"}:
                    # h(W,A,X) uses [tx, tw] with treatment a appended inside the head
                    y_bridge = self.bridge_head(torch.cat([tx, tw], dim=1), ab)
                    bridge_mse = mse(y_bridge, yb)
                    # residual independence from (A,X,Z)
                    r = (yb - y_bridge).unsqueeze(1)
                    cond_tx = tx.detach() if getattr(cfg, "bridge_detach_cond", True) else tx
                    cond_tz = tz.detach() if getattr(cfg, "bridge_detach_cond", True) else tz
                    cond_noa = torch.cat([cond_tx, cond_tz], dim=1)

                    # subsample for HSIC
                    max_n = int(getattr(cfg, "bridge_hsic_max_samples", 256) or 256)
                    by_a = bool(getattr(cfg, "bridge_hsic_by_treatment", True))
                    bridge_hsic = torch.tensor(0.0, device=r.device)
                    if by_a:
                        hs = []
                        for mask in [(ab > 0.5), (ab <= 0.5)]:
                            if mask.sum() < 8:
                                continue
                            r_a = r[mask]
                            r_a = r_a - r_a.mean(dim=0, keepdim=True)
                            c_a = cond_noa[mask]
                            if (max_n > 0) and (r_a.shape[0] > max_n):
                                idx_b = torch.randperm(r_a.shape[0], device=r.device)[:max_n]
                                r_a = r_a[idx_b]
                                c_a = c_a[idx_b]
                            hs.append(_rbf_hsic(r_a, c_a))
                        if len(hs) > 0:
                            bridge_hsic = sum(hs) / len(hs)
                    else:
                        cond = torch.cat([ab.unsqueeze(1).float(), cond_noa], dim=1)
                        r_c = r - r.mean(dim=0, keepdim=True)
                        if (max_n > 0) and (r.shape[0] > max_n):
                            idx_b = torch.randperm(r.shape[0], device=r.device)[:max_n]
                            r_h = r_c[idx_b]
                            cond_h = cond[idx_b]
                        else:
                            r_h = r_c
                            cond_h = cond
                        bridge_hsic = _rbf_hsic(r_h, cond_h)
                    # warmup ramp to avoid destabilizing early training
                    warm = int(getattr(cfg, "bridge_warmup_epochs", 10) or 0)
                    ramp = 1.0 if (warm <= 0) else float(min(1.0, max(0.0, (epoch - warm) / max(1.0, 10.0))))
                    bridge_loss = float(getattr(cfg, "lambda_bridge", 0.5)) * (bridge_mse + float(getattr(cfg, "lambda_bridge_hsic", 0.1)) * bridge_hsic) * ramp
                    epoch_bridge_mse += float(bridge_mse.item())
                    epoch_bridge_hsic += float(bridge_hsic.item())
                    epoch_bridge_batches += 1
                    
                    # === Q-bridge training (Treatment Bridge) with Adversarial Moment Constraint ===
                    # v9.0: CRITICAL FIX - Use RAW inputs throughout!
                    #
                    # The bridge equation is defined on OBSERVED proxies:
                    #   E[q(Z,a,X) | W,X,A=a] = 1/P(A=a|W,X)
                    # 
                    # Previous versions (v8.6-v8.9) had a fatal bug: e/q used latent (tw,tx,tz)
                    # but critic used raw (w_raw, x_raw). This "torn conditioning space" caused
                    # the min-max game to find the wrong equilibrium (systematic positive bias).
                    #
                    # v9.0 fixes:
                    # 1. e(W,X) uses raw: a_wx_raw_head(w_raw, x_raw)
                    # 2. q(Z,X) uses raw: q_bridge_raw_head(z_raw, x_raw, ab)
                    # 3. Critic includes A: enables per-group moment conditions
                    if self.q_bridge_head is not None and estimand_str == "proximal_dr":
                        q_warm = int(getattr(cfg, "q_bridge_warmup_epochs", 15) or 0)
                        if epoch >= q_warm:
                            # --- (1) Extract ALL RAW variables ---
                            x_raw = vb[:, self._block_slices[0]]
                            w_raw = vb[:, self._block_slices[1]]
                            z_raw = vb[:, self._block_slices[2]]
                            
                            # --- (2) Estimate propensity e(W,X) using RAW inputs ---
                            e_clip_lo, e_clip_hi = getattr(cfg, "q_bridge_prop_clip", (0.01, 0.99))
                            wx_raw = torch.cat([w_raw, x_raw], dim=1)
                            logits_prop_raw = self.a_wx_raw_head(wx_raw)
                            prop_loss = bce(logits_prop_raw, ab)
                            if self.prop_wx_opt is not None:
                                self.prop_wx_opt.zero_grad()
                                prop_loss.backward()
                                self.prop_wx_opt.step()
                            with torch.no_grad():
                                e_wx = torch.sigmoid(logits_prop_raw).clamp(float(e_clip_lo), float(e_clip_hi))

                            # --- (3) Target for q-bridge: 1/e for treated, 1/(1-e) for control ---
                            t_clip_lo, t_clip_hi = getattr(cfg, "q_bridge_target_clip", (0.5, 5.0))
                            target_q = ab / e_wx + (1 - ab) / (1 - e_wx)
                            target_q = target_q.clamp(float(t_clip_lo), float(t_clip_hi))

                            # --- (4) q prediction using RAW (Z,X) ---
                            q_pred = self.q_bridge_raw_head(z_raw, x_raw, ab)
                            q_pred_clipped = q_pred.clamp(float(t_clip_lo), float(t_clip_hi))
                            
                            # Residual
                            q_resid = q_pred_clipped - target_q

                            # --- (5) ADVERSARIAL MOMENT TRAINING (Min-Max) with A in critic ---
                            # Moment conditions are PER TREATMENT GROUP:
                            #   E[q(Z,1,X) - 1/e | W,X,A=1] = 0
                            #   E[q(Z,0,X) - 1/(1-e) | W,X,A=0] = 0
                            # Including A in critic helps enforce these separately
                            
                            n_critic_steps = int(getattr(cfg, "q_critic_steps", 3))
                            # Critic input: (W, X, A) concatenated
                            critic_input = torch.cat([w_raw, x_raw, ab.unsqueeze(1)], dim=1)
                            
                            for _ in range(n_critic_steps):
                                # Critic maximizes: E[f * resid] - 0.5 * E[f^2]
                                f_critic = self.q_bridge_critic(critic_input.detach(), torch.zeros(1, device=self.device))  # dummy second arg
                                critic_loss = -(f_critic * q_resid.detach()).mean() + 0.5 * (f_critic ** 2).mean()
                                if self.q_critic_opt is not None:
                                    self.q_critic_opt.zero_grad()
                                    critic_loss.backward()
                                    self.q_critic_opt.step()
                            
                            # Q step: minimize E[f * resid]
                            q_pred_new = self.q_bridge_raw_head(z_raw, x_raw, ab)
                            q_pred_new_clipped = q_pred_new.clamp(float(t_clip_lo), float(t_clip_hi))
                            q_resid_new = q_pred_new_clipped - target_q.detach()
                            
                            f_fixed = self.q_bridge_critic(critic_input.detach(), torch.zeros(1, device=self.device)).detach()
                            
                            # Q-bridge loss
                            lambda_q_adv = float(getattr(cfg, "lambda_q_bridge_adv", 1.0))
                            q_adv_loss = (f_fixed * q_resid_new).mean()
                            
                            lambda_q_mean = float(getattr(cfg, "lambda_q_bridge_mean", 0.1))
                            lambda_q_l2 = float(getattr(cfg, "lambda_q_bridge_l2", 1e-3))
                            q_mean_pen = (q_resid_new.mean() ** 2)
                            q_l2_pen = (q_pred_new ** 2).mean()
                            
                            q_bridge_loss = lambda_q_adv * q_adv_loss + lambda_q_mean * q_mean_pen + lambda_q_l2 * q_l2_pen
                            
                            if self.q_bridge_opt is not None:
                                self.q_bridge_opt.zero_grad()
                                (float(getattr(cfg, "lambda_q_bridge", 0.5)) * q_bridge_loss).backward()
                                self.q_bridge_opt.step()

                            # --- (6) Quality diagnostics for adaptive shrinkage ---
                            frac_clipped = float(((q_pred < t_clip_lo) | (q_pred > t_clip_hi)).float().mean().item())
                            moment_violation = float((-(f_fixed * q_resid_new).mean() + 0.5 * (f_fixed ** 2).mean()).abs().item())

                            # --- diagnostics ---
                            try:
                                self.training_diagnostics["prop_wx_loss"] = float(prop_loss.detach().cpu().item())
                                self.training_diagnostics["q_bridge_mse"] = float(mse(q_pred_clipped.detach(), target_q.detach()).cpu().item())
                                self.training_diagnostics["q_bridge_mean_pen"] = float(q_mean_pen.detach().cpu().item())
                                self.training_diagnostics["q_bridge_l2_pen"] = float(q_l2_pen.detach().cpu().item())
                                self.training_diagnostics["q_bridge_critic_loss"] = float(critic_loss.detach().cpu().item())
                                self.training_diagnostics["q_bridge_adv_loss"] = float(q_adv_loss.detach().cpu().item())
                                self.training_diagnostics["q_pred_mean"] = float(q_pred_clipped.mean().detach().cpu().item())
                                self.training_diagnostics["q_pred_std"] = float(q_pred_clipped.std().detach().cpu().item())
                                self.training_diagnostics["f_critic_mean"] = float(f_fixed.mean().detach().cpu().item())
                                self.training_diagnostics["f_critic_std"] = float(f_fixed.std().detach().cpu().item())
                                self.training_diagnostics["q_frac_clipped"] = frac_clipped
                                self.training_diagnostics["q_moment_violation"] = moment_violation
                            except Exception:
                                pass

                cons_a = bce(self.a_from_z(tz), ab)
                cons_y = mse(self.y_from_w(tw, ab), yb)
                consistency = cons_a + cons_y

                # --- Weak-IV diagnostics: store Z→A consistency loss for relev-penalty (EMA-smoothed) ---
                try:
                    cons_a_val = float(cons_a.detach().cpu().item())
                    if not hasattr(self, "_cons_a_ema"):
                        self._cons_a_ema = cons_a_val
                    else:
                        ema_rho = float(getattr(cfg, "monitor_ema", 0.95))
                        self._cons_a_ema = float(ema_rho * self._cons_a_ema + (1.0 - ema_rho) * cons_a_val)
                    self.training_diagnostics["consistency_loss_z"] = float(self._cons_a_ema)
                except Exception:
                    pass

                tx_det = tx.detach()
                tw_det = tw.detach()
                tz_det = tz.detach()

                ortho = _offdiag_corr_penalty([tx, tw, tz, tn])
                warmup = cfg.cond_ortho_warmup_epochs
                cond_weight = 0.0
                if cfg.lambda_cond_ortho > 0 and epoch >= warmup:
                    ramp = min(1.0, (epoch - warmup + 1) / max(1, warmup))
                    cond_weight = float(cfg.lambda_cond_ortho * ramp)
                cond_ortho = torch.zeros((), device=self.device)
                if cond_weight > 0:
                    cond_ortho_wz = _conditional_orthogonal_penalty([tw, tz], tx, ridge=cfg.ridge_alpha)
                    tn_cond = torch.cat([tx_det, tw_det, tz_det], dim=1)
                    tn_res = _compute_residual(tn, tn_cond, ridge=cfg.ridge_alpha)
                    cond_ortho_n = _offdiag_corr_penalty([tn_res, tx_det, tw_det, tz_det])
                    cond_ortho = cond_ortho_wz + cond_ortho_n
                    
                adv_w_logits = self.adv_w(tw)
                adv_n_logits = self.adv_n(tn)
                adv_z_pred = self.adv_z(tz)
                adv_w_cond_loss = adv_z_cond_loss = adv_n_cond_loss = None
                if cond_factor > 0:
                    adv_w_cond_loss = self.adv_w_cond(tw, tx_det, ab)
                    adv_z_cond_loss = self.adv_z_cond(tz, torch.cat([tx_det, ab.unsqueeze(1)], dim=1), yb)
                    adv_n_cond_loss = self.adv_n_cond(tn, torch.cat([tx_det, tw_det, tz_det], dim=1), ab)

                hsic_pen = torch.zeros((), device=self.device)
                if lambda_hsic_ep > 0:
                    if tx.shape[0] > cfg.hsic_max_samples:
                        idx = torch.randperm(tx.shape[0], device=self.device)[:cfg.hsic_max_samples]
                        tx_h, tw_h, tz_h, tn_h = tx[idx], tw[idx], tz[idx], tn[idx]
                    else:
                        tx_h, tw_h, tz_h, tn_h = tx, tw, tz, tn
                    hsic_pen = (
                        _rbf_hsic(tx_h, tw_h) + _rbf_hsic(tw_h, tz_h) + _rbf_hsic(tx_h, tn_h)
                        + _rbf_hsic(tw_h, tn_h) + _rbf_hsic(tz_h, tn_h)
                    )
                    lambda_hsic_w_a = getattr(cfg, 'lambda_hsic_w_a', 0.0)
                    if lambda_hsic_w_a > 0:
                        ab_h = ab[idx] if tx.shape[0] > cfg.hsic_max_samples else ab
                        ab_2d = ab_h.unsqueeze(1).float()
                        hsic_w_a = _rbf_hsic(tw_h, ab_2d)
                        hsic_pen = hsic_pen + lambda_hsic_w_a * hsic_w_a

                lambda_overlap_ep = float(cfg.lambda_overlap)
                if cfg.lambda_overlap > 0:
                    diag_now = getattr(self, "training_diagnostics", {}) or {}
                    ess_min = diag_now.get("overlap_ess_min")
                    ess_target_train = cfg.ess_target_train or 0.0
                    if ess_min is not None and ess_target_train > 0:
                        gap = max(0.0, float(ess_target_train) - float(ess_min))
                        scale = 1.0 + float(cfg.overlap_boost) * gap / max(1e-6, float(ess_target_train))
                        lambda_overlap_ep = float(np.clip(cfg.lambda_overlap * scale, 0.0, cfg.lambda_overlap * 3.0))

                overlap_pen = torch.zeros((), device=self.device)
                if lambda_overlap_ep > 0 and epoch >= cfg.overlap_warmup_epochs:
                    prob_a = torch.sigmoid(logits_a)
                    sc = getattr(cfg, 'scenario_config', ScenarioSpecificConfig())
                    m = float(cfg.overlap_margin) * float(getattr(sc, 'overlap_margin_boost', 1.0))
                    base_pen = torch.mean(torch.relu(m - prob_a) ** 2 + torch.relu(prob_a - (1 - m)) ** 2)
                    # Weak-overlap: add extra penalty on extreme propensities
                    if getattr(cfg, 'scenario_type', ScenarioType.DEFAULT) == ScenarioType.WEAK_OVERLAP:
                        extreme = torch.mean(torch.relu(0.01 - prob_a) ** 2 + torch.relu(prob_a - 0.99) ** 2)
                        overlap_pen = base_pen + 10.0 * extreme
                    else:
                        overlap_pen = base_pen

                # Scenario-aware main losses
                sc = getattr(cfg, 'scenario_config', ScenarioSpecificConfig())
                a_loss_main = focal_bce_fn(logits_a, ab) if sc.use_focal_loss else bce(logits_a, ab)
                y_loss_main = huber_loss_fn(y_pred, yb) if sc.use_huber_loss else mse(y_pred, yb)
                lambda_ortho_use = float(cfg.lambda_ortho) * float(sc.lambda_ortho_boost)
                lambda_hsic_use = float(lambda_hsic_ep) * float(sc.lambda_hsic_boost)
                lambda_overlap_use = float(lambda_overlap_ep) * float(sc.lambda_overlap_boost)
                align_pen = torch.zeros((), device=self.device)
                if float(sc.lambda_alignment) > 0:
                    align_pen = proxy_alignment_loss(tx, tw, tz)

                loss_main = (
                    cfg.lambda_recon * mse(recon, vb)
                    + cfg.lambda_a * a_loss_main
                    + cfg.lambda_y * y_loss_main
                    + lambda_consistency_use * consistency
                    + lambda_ortho_use * ortho
                    + cond_weight * cond_ortho
                    + cfg.gamma_padic * _padic_ultrametric_loss(torch.cat([tx, tz], dim=1))
                    + lambda_hsic_use * hsic_pen
                    + lambda_overlap_use * overlap_pen
                    + float(sc.lambda_alignment) * align_pen
                    + cfg.lambda_gate * float(sc.gate_reg_mult) * self._gate_penalty(tx)
                    - gamma_w_use * bce(adv_w_logits, ab)
                    - gamma_n_use * bce(adv_n_logits, ab)
                    - gamma_z_use * mse(adv_z_pred, yb)
                )
                if cond_factor > 0:
                    if adv_w_cond_loss is not None:
                        loss_main = loss_main - gamma_w_cond * adv_w_cond_loss
                    if adv_z_cond_loss is not None:
                        loss_main = loss_main - gamma_z_cond * adv_z_cond_loss
                    if adv_n_cond_loss is not None:
                        loss_main = loss_main - gamma_n_cond * adv_n_cond_loss
                
                                # add proximal-bridge loss (if enabled)
                if bridge_loss is not None and bridge_loss.numel() == 1:
                    loss_main = loss_main + bridge_loss

# === 添加弱IV特化损失（明确拆分） ===
                if weak_iv_params is not None and weak_iv_params.get('weak_iv_mode', False):
                    if (weak_iv_excl_penalty > 0.0) or (weak_iv_relev_penalty > 0.0):
                        loss_main = loss_main + weak_iv_excl_penalty + weak_iv_relev_penalty

                self.main_opt.zero_grad(); loss_main.backward(); self.main_opt.step()
                self._toggle_requires_grad(
                    [self.adv_w, self.adv_n, self.adv_z, self.adv_w_cond, self.adv_z_cond, self.adv_n_cond], True
                )
                epoch_loss += float(loss_main.item())
                epoch_batches += 1

                        # record proximal-bridge epoch diagnostics
            if str(getattr(cfg, "estimand", "ate")).lower().strip() in {"proximal_bridge", "proximal", "bridge"}:
                if epoch_bridge_batches > 0:
                    self.training_diagnostics["bridge_mse_epoch"] = float(epoch_bridge_mse / epoch_bridge_batches)
                    self.training_diagnostics["bridge_hsic_epoch"] = float(epoch_bridge_hsic / epoch_bridge_batches)
                    self.training_diagnostics["bridge_batches"] = float(epoch_bridge_batches)

# Validation
            self.enc_x.eval(); self.enc_w.eval(); self.enc_z.eval(); self.enc_n.eval()
            self.decoder.eval(); self.a_head.eval(); self.y_head.eval()
            vals = []
            with torch.no_grad():
                for vb, ab, yb in va_loader:
                    vb = vb.to(self.device, dtype=target_dtype)
                    ab = ab.to(self.device, dtype=target_dtype)
                    yb = yb.to(self.device, dtype=target_dtype)
                    tx, tw, tz, tn = self._encode_blocks(vb)
                    recon = self.decoder(torch.cat([tx, tw, tz, tn], dim=1))
                    logits_a = self.a_head(self._compose_a_t(tx, tw, tz))
                    y_pred = self.y_head(self._compose_y_t(tx, tw, tz), ab)
                    loss_val = cfg.lambda_recon * mse(recon, vb) + cfg.lambda_a * bce(logits_a, ab) + cfg.lambda_y * mse(y_pred, yb)
                    vals.append(float(loss_val.item()))
            mean_val = float(np.mean(vals)) if vals else float("inf")
            self.main_sched.step(mean_val)

            # Early stopping
            if mean_val < best_val - cfg.early_stopping_min_delta:
                best_val = mean_val
                patience = 0
                best_state = {k: v.state_dict() for k, v in {
                    "enc_x": self.enc_x, "enc_w": self.enc_w, "enc_z": self.enc_z, "enc_n": self.enc_n,
                    "decoder": self.decoder, "a_head": self.a_head, "y_head": self.y_head,
                    "a_from_z": self.a_from_z, "y_from_w": self.y_from_w,
                    "adv_w": self.adv_w, "adv_z": self.adv_z, "adv_n": self.adv_n,
                    "adv_w_cond": self.adv_w_cond, "adv_z_cond": self.adv_z_cond, "adv_n_cond": self.adv_n_cond,
                }.items()}
            else:
                patience += 1
                if patience >= cfg.early_stopping_patience:
                    break

            # Training monitor
            enhanced_training_monitor(self, epoch, {"main": epoch_loss / max(epoch_batches, 1), "val": mean_val})

        # Restore best state
        if best_state is not None:
            self.enc_x.load_state_dict(best_state["enc_x"])
            self.enc_w.load_state_dict(best_state["enc_w"])
            self.enc_z.load_state_dict(best_state["enc_z"])
            self.enc_n.load_state_dict(best_state["enc_n"])
            self.decoder.load_state_dict(best_state["decoder"])
            self.a_head.load_state_dict(best_state["a_head"])
            self.y_head.load_state_dict(best_state["y_head"])
            self.a_from_z.load_state_dict(best_state["a_from_z"])
            self.y_from_w.load_state_dict(best_state["y_from_w"])
            self.adv_w.load_state_dict(best_state["adv_w"])
            self.adv_z.load_state_dict(best_state["adv_z"])
            self.adv_n.load_state_dict(best_state["adv_n"])
            self.adv_w_cond.load_state_dict(best_state["adv_w_cond"])
            self.adv_z_cond.load_state_dict(best_state["adv_z_cond"])
            self.adv_n_cond.load_state_dict(best_state["adv_n_cond"])

        # Final diagnostics
        self.enc_x.eval(); self.enc_w.eval(); self.enc_z.eval(); self.enc_n.eval()
        self.adv_w.eval(); self.adv_n.eval(); self.adv_z.eval()
        with torch.no_grad():
            V_t = torch.from_numpy(_apply_standardize(V_all, self._v_mean, self._v_std)).to(self.device)
            tx, tw, tz, tn = self._encode_blocks(V_t)
            txn, twn, tzn = tx.cpu().numpy(), tw.cpu().numpy(), tz.cpu().numpy()
            
            # 计算 adversary 诊断
            adv_w_probs = torch.sigmoid(self.adv_w(tw)).cpu().numpy()
            adv_n_probs = torch.sigmoid(self.adv_n(tn)).cpu().numpy()
            adv_z_pred = self.adv_z(tz).cpu().numpy()
        
        # 计算 adv_w_acc, adv_n_acc, adv_z_r2
        adv_w_acc = float(((adv_w_probs > 0.5) == (A == 1)).mean()) if adv_w_probs.size else np.nan
        adv_n_acc = float(((adv_n_probs > 0.5) == (A == 1)).mean()) if adv_n_probs.size else np.nan
        
        # adv_z 是在标准化 Y 上训练的，需要转换回原始尺度
        if hasattr(self, "_y_std") and hasattr(self, "_y_mean"):
            y_scale = float(self._y_std.squeeze())
            y_mean = float(self._y_mean.squeeze())
            adv_z_pred_raw = adv_z_pred * y_scale + y_mean
        else:
            adv_z_pred_raw = adv_z_pred
        adv_z_r2 = float(r2_score(Y, adv_z_pred_raw)) if np.var(Y) > 0 else np.nan
        
        self.training_diagnostics.update({
            "adv_w_acc": adv_w_acc,
            "adv_n_acc": adv_n_acc,
            "adv_z_r2": adv_z_r2,
        })
        
        self._compute_representation_diagnostics(txn, twn, tzn, A, Y)
        self.training_diagnostics["min_std"] = float(cfg.min_std)
        self.training_diagnostics["protected_features"] = int(np.sum(self._protected_mask)) if self._protected_mask is not None else 0
        self._is_fit = True

    def get_latent(self, V: np.ndarray) -> np.ndarray:
        V = np.asarray(V, dtype=np.float32)
        V_std = _apply_standardize(V, self._v_mean, self._v_std)
        V_t = torch.from_numpy(V_std).to(self.device)
        self.enc_x.eval(); self.enc_w.eval(); self.enc_z.eval(); self.enc_n.eval()
        with torch.no_grad():
            tx, tw, tz, tn = self._encode_blocks(V_t)
            if self.config.use_noise_in_latent:
                return torch.cat([tx, tw, tz, tn], dim=1).cpu().numpy()
            return torch.cat([tx, tw, tz], dim=1).cpu().numpy()

    def effect(self, V: np.ndarray) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("Model not fitted")
        U = self.get_latent(V)
        return self._simple_effect(U)

    def _simple_effect(self, U: np.ndarray) -> np.ndarray:
        cfg = self.config
        dx, dw = cfg.latent_x_dim, cfg.latent_w_dim
        tx = U[:, :dx]
        tw = U[:, dx:dx + dw]
        self.y_head.eval()
        with torch.no_grad():
            tx_t = torch.from_numpy(tx.astype(np.float32)).to(self.device)
            tw_t = torch.from_numpy(tw.astype(np.float32)).to(self.device)
            ones = torch.ones(tx_t.shape[0], device=self.device)
            zeros = torch.zeros(tx_t.shape[0], device=self.device)
            t1_std = self.y_head(self._compose_y_t(tx_t, tw_t, tz_t), ones).cpu().numpy()
            t0_std = self.y_head(self._compose_y_t(tx_t, tw_t, tz_t), zeros).cpu().numpy()
        t1 = self._destandardize_y(t1_std)
        t0 = self._destandardize_y(t0_std)
        return t1 - t0

    def _destandardize_y(self, y_std: np.ndarray) -> np.ndarray:
        return y_std * self._y_std.squeeze() + self._y_mean.squeeze()

    def ate(self, V: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        U = self.get_latent(V)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        return self._dr_ate(U, A, Y)

    def _head_features(self, U: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        dx, dw, dz = cfg.latent_x_dim, cfg.latent_w_dim, cfg.latent_z_dim
        tx = U[:, :dx]
        tw = U[:, dx:dx + dw]
        tz = U[:, dx + dw:dx + dw + dz]
        self.a_head.eval(); self.y_head.eval()
        with torch.no_grad():
            tx_t = torch.from_numpy(tx.astype(np.float32)).to(self.device)
            tw_t = torch.from_numpy(tw.astype(np.float32)).to(self.device)
            tz_t = torch.from_numpy(tz.astype(np.float32)).to(self.device)
            a_t = torch.from_numpy(A.astype(np.float32)).to(self.device)
            s_logits = self.a_head(self._compose_a_t(tx_t, tw_t, tz_t)).cpu().numpy()
            t_obs_std = self.y_head(self._compose_y_t(tx_t, tw_t, tz_t), a_t).cpu().numpy()
            ones = torch.ones(tx_t.shape[0], device=self.device)
            zeros = torch.zeros(tx_t.shape[0], device=self.device)
            t1_std = self.y_head(self._compose_y_t(tx_t, tw_t, tz_t), ones).cpu().numpy()
            t0_std = self.y_head(self._compose_y_t(tx_t, tw_t, tz_t), zeros).cpu().numpy()
        t_obs = self._destandardize_y(t_obs_std)
        t1 = self._destandardize_y(t1_std)
        t0 = self._destandardize_y(t0_std)
        return s_logits, t_obs, t1, t0

    def _split_latent(self, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        cfg = self.config
        dx, dw, dz = cfg.latent_x_dim, cfg.latent_w_dim, cfg.latent_z_dim
        tx = U[:, :dx]
        tw = U[:, dx:dx + dw]
        tz = U[:, dx + dw:dx + dw + dz]
        tn = U[:, dx + dw + dz:] if cfg.use_noise_in_latent and U.shape[1] > dx + dw + dz else None
        return tx, tw, tz, tn

    def _dr_ate(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        cfg = self.config
        U = np.asarray(U)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        weak_iv = bool(getattr(self, "_weak_iv_flag", False))
        dr_mode = "weak_iv_conservative" if weak_iv else "default"
        stats: Dict[str, list] = {k: [] for k in ["e_min", "e_max", "e_q01", "e_q05", "e_q95", "e_q99",
                                                    "clip_used", "cap_used", "overlap_score", "frac_e_clipped",
                                                    "clip_opt_overall_ess_ratio", "clip_opt_min_ess_ratio",
                                                    "ipw_abs_max_raw", "ipw_abs_max_postclip_precap",
                                                    "ipw_abs_max_capped", "frac_ipw_capped", "frac_ipw_capped_raw"]}

        def _quantiles(x: np.ndarray) -> Dict[str, float]:
            if x.size == 0:
                return {"min": np.nan, "max": np.nan, "q01": np.nan, "q05": np.nan, "q95": np.nan, "q99": np.nan}
            return {"min": float(np.min(x)), "max": float(np.max(x)),
                    "q01": float(np.quantile(x, 0.01)), "q05": float(np.quantile(x, 0.05)),
                    "q95": float(np.quantile(x, 0.95)), "q99": float(np.quantile(x, 0.99))}

        kf = KFold(n_splits=cfg.n_splits_dr, shuffle=True, random_state=cfg.seed)
        psi = np.zeros_like(Y, dtype=float)
        # reset overlap-weighted accumulators (ATO)
        self._ato_num_denom = [0.0, 0.0]
        clip = float(max(cfg.clip_prop_radr, cfg.clip_prop))
        ipw_cap = cfg.ipw_cap_radr if getattr(cfg, "ipw_cap_radr", None) is not None else cfg.ipw_cap
        if weak_iv and ipw_cap is not None:
            ipw_cap = min(float(ipw_cap), 10.0)

        for tr, te in kf.split(U):
            A_tr, A_te = A[tr], A[te]
            Y_tr, Y_te = Y[tr], Y[te]
            s_tr, t_obs_tr, t1_tr, t0_tr = self._head_features(U[tr], A_tr)
            s_te, t_obs_te, t1_te, t0_te = self._head_features(U[te], A_te)
            tx_tr, tw_tr, tz_tr, tn_tr = self._split_latent(U[tr])
            tx_te, tw_te, tz_te, tn_te = self._split_latent(U[te])

            s_tr_clipped = np.clip(s_tr, -10, 10)
            s_te_clipped = np.clip(s_te, -10, 10)
            prop_feats_tr = [s_tr_clipped.reshape(-1, 1), tx_tr, tz_tr]
            prop_feats_te = [s_te_clipped.reshape(-1, 1), tx_te, tz_te]
            if tn_tr is not None:
                prop_feats_tr.append(tn_tr)
                prop_feats_te.append(tn_te)
            Xp_tr = np.column_stack(prop_feats_tr)
            Xp_te = np.column_stack(prop_feats_te)
            if cfg.standardize_nuisance:
                scaler_prop = StandardScaler().fit(Xp_tr)
                Xp_tr = scaler_prop.transform(Xp_tr)
                Xp_te = scaler_prop.transform(Xp_te)
            if np.unique(A_tr).size < 2:
                e_raw = np.full_like(A_te, float(np.mean(A_tr)) if len(A_tr) else 0.5, dtype=float)
            else:
                prop = LogisticRegression(max_iter=2000, solver="lbfgs", C=cfg.propensity_logreg_C)
                prop.fit(Xp_tr, A_tr)
                e_raw = prop.predict_proba(Xp_te)[:, 1]
            if cfg.propensity_shrinkage > 0:
                prior = float(np.mean(A_tr)) if len(A_tr) else 0.5
                e_raw = (1 - cfg.propensity_shrinkage) * e_raw + cfg.propensity_shrinkage * prior

            qs = _quantiles(e_raw)
            clip_use = clip
            # 自适应裁剪：在 [clip, clip_prop_adaptive_max] 内搜索最小 clip 使 ESS 达标
            if cfg.clip_prop_adaptive_max is not None and cfg.clip_prop_adaptive_max > clip_use and cfg.ess_target is not None and cfg.ess_target > 0:
                clip_opt, clip_stats = find_optimal_clip_threshold(
                    e_raw=e_raw,
                    A=A_te,
                    ess_target=float(cfg.ess_target),
                    min_clip=float(clip_use),
                    max_clip=float(cfg.clip_prop_adaptive_max),
                    n_search=25,
                )
                clip_use = float(clip_opt)
                # 记录折内裁剪信息（在 fold mean 聚合）
                stats.setdefault("clip_opt_overall_ess_ratio", []).append(float(clip_stats.get("overall_ess_ratio", np.nan)))
                stats.setdefault("clip_opt_min_ess_ratio", []).append(float(clip_stats.get("min_ess_ratio", np.nan)))
            cap_use = ipw_cap
            e_hat = np.clip(e_raw, clip_use, 1 - clip_use)
            stats["clip_used"].append(float(clip_use))
            stats["cap_used"].append(float(cap_use) if cap_use is not None else np.nan)
            eps = max(clip_use, 0.05)
            overlap_score = float(np.mean((e_raw > eps) & (e_raw < 1 - eps))) if e_raw.size else np.nan
            stats["overlap_score"].append(overlap_score)
            stats["frac_e_clipped"].append(float(np.mean((e_raw < clip_use) | (e_raw > 1 - clip_use))))
            for k, v in qs.items():
                stats[f"e_{k}"].append(v)

            def _outcome_features(a_vec, tx_block, tw_block, t_val, tn_block):
                feats = [a_vec.reshape(-1, 1), tx_block, tw_block, t_val.reshape(-1, 1),
                         (a_vec * t_val).reshape(-1, 1), (a_vec[:, None] * tx_block), (a_vec[:, None] * tw_block)]
                if tn_block is not None:
                    feats.append(tn_block)
                    feats.append(a_vec[:, None] * tn_block)
                return np.column_stack(feats)

            Xo_tr = _outcome_features(A_tr, tx_tr, tw_tr, t_obs_tr, tn_tr)
            Xo_te = _outcome_features(A_te, tx_te, tw_te, t_obs_te, tn_te)
            if cfg.standardize_nuisance:
                scaler_out = StandardScaler().fit(Xo_tr)
                Xo_tr = scaler_out.transform(Xo_tr)
                Xo_te = scaler_out.transform(Xo_te)

            if len(A_tr) == 0:
                out_model: Ridge | DummyRegressor = DummyRegressor(strategy="mean")
                out_model.fit(np.zeros((1, Xo_tr.shape[1])), [0.0])
            else:
                out_model = Ridge(alpha=cfg.ridge_alpha)
                out_model.fit(Xo_tr, Y_tr)

            X1 = _outcome_features(np.ones_like(A_te), tx_te, tw_te, t1_te, tn_te)
            X0 = _outcome_features(np.zeros_like(A_te), tx_te, tw_te, t0_te, tn_te)
            if cfg.standardize_nuisance:
                X1 = scaler_out.transform(X1)
                X0 = scaler_out.transform(X0)

            m_hat = out_model.predict(Xo_te)
            m1 = out_model.predict(X1)
            m0 = out_model.predict(X0)

            eps_raw = 1e-6
            e_safe = np.clip(e_raw, eps_raw, 1 - eps_raw)
            w_raw = (A_te - e_safe) / (e_safe * (1 - e_safe))
            stats["ipw_abs_max_raw"].append(float(np.max(np.abs(w_raw))) if w_raw.size else np.nan)
            w = (A_te - e_hat) / (e_hat * (1 - e_hat))
            stats["ipw_abs_max_postclip_precap"].append(float(np.max(np.abs(w))) if w.size else np.nan)

            if ipw_cap and ipw_cap > 0:
                cap_val = float(ipw_cap)
                if cfg.ipw_cap_quantile is not None and 0 < cfg.ipw_cap_quantile < 1 and w.size:
                    cap_val = min(cap_val, float(np.quantile(np.abs(w), cfg.ipw_cap_quantile)))
                w_capped = np.clip(w, -cap_val, cap_val)
            else:
                w_capped = w
            stats["ipw_abs_max_capped"].append(float(np.max(np.abs(w_capped))) if w_capped.size else np.nan)
            if ipw_cap and ipw_cap > 0:
                stats["frac_ipw_capped"].append(float(np.mean(np.abs(w) >= float(ipw_cap))) if w.size else np.nan)
                stats["frac_ipw_capped_raw"].append(float(np.mean(np.abs(w_raw) >= float(ipw_cap))) if w_raw.size else np.nan)
            else:
                stats["frac_ipw_capped"].append(np.nan)
                stats["frac_ipw_capped_raw"].append(np.nan)

            estimand = str(getattr(cfg, "estimand", "ate")).lower().strip()
            # === Estimand switch ===
            if estimand in {"trimmed_ate", "trimmed-ate", "trimmed"}:
                trim = float(max(getattr(cfg, "trim_prop", 0.05), clip_use))
                mask = (e_raw > trim) & (e_raw < 1 - trim)
                psi_fold = (m1 - m0) + w_capped * (Y_te - m_hat)
                psi_fold = np.where(mask, psi_fold, np.nan)
                psi[te] = psi_fold
                stats.setdefault("trim_prop", []).append(trim)
                stats.setdefault("frac_trimmed", []).append(float(np.mean(~mask)))
            elif estimand in {"ato", "overlap", "overlap_weighted"}:
                # Overlap-weighted target (ATO): weights proportional to e*(1-e)
                h = e_hat * (1 - e_hat)
                cap_val = float(ipw_cap) if (ipw_cap is not None and ipw_cap > 0) else None
                inv_e = 1.0 / np.maximum(e_hat, 1e-6)
                inv_1me = 1.0 / np.maximum(1.0 - e_hat, 1e-6)
                if cap_val is not None:
                    inv_e = np.minimum(inv_e, cap_val)
                    inv_1me = np.minimum(inv_1me, cap_val)
                psi_fold = (m1 - m0) + (A_te * inv_e) * (Y_te - m1) - ((1 - A_te) * inv_1me) * (Y_te - m0)
                # Store for diagnostics; final aggregation is weighted
                psi[te] = psi_fold
                stats.setdefault("ato_den", []).append(float(np.mean(h)))
                stats.setdefault("ato_h_min", []).append(float(np.min(h)) if h.size else np.nan)
                stats.setdefault("ato_h_max", []).append(float(np.max(h)) if h.size else np.nan)
                # accumulate numerator/denominator for weighted mean via training_diagnostics later
                if not hasattr(self, "_ato_num_denom"):
                    self._ato_num_denom = [0.0, 0.0]
                self._ato_num_denom[0] += float(np.nansum(h * psi_fold))
                self._ato_num_denom[1] += float(np.nansum(h))
            else:
                # Default ATE
                psi[te] = m1 - m0 + w_capped * (Y_te - m_hat)

        if not hasattr(self, "training_diagnostics"):
            self.training_diagnostics = {}
        if stats:
            agg = {k: float(np.nanmean(v)) for k, v in stats.items() if v}
            self.training_diagnostics.update({f"dr_{k}": v for k, v in agg.items()})
            self.training_diagnostics["dr_mode"] = dr_mode
            self.training_diagnostics["dr_estimand"] = str(getattr(cfg, "estimand", "ate"))

        estimand_final = str(getattr(cfg, "estimand", "ate")).lower().strip()
        if estimand_final in {"trimmed_ate", "trimmed-ate", "trimmed"}:
            return float(np.nanmean(psi))
        if estimand_final in {"ato", "overlap", "overlap_weighted"}:
            num_denom = getattr(self, "_ato_num_denom", None)
            if num_denom is None or (num_denom[1] <= 0):
                return float(np.nan)
            # record denom for diagnostics
            self.training_diagnostics["dr_ato_denom"] = float(num_denom[1])
            return float(num_denom[0] / num_denom[1])
        return float(np.mean(psi))

    
    def _proximal_bridge_ate(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        """Proximal outcome-bridge estimand using learned h(W,A,X) = mu(W,X) + A*tau(W,X).

        With separated BridgeHead, we can directly get tau = E[tau(W,X)].
        """
        cfg = self.config
        if not hasattr(self, "bridge_head") or self.bridge_head is None:
            raise RuntimeError("bridge_head is not initialized; set cfg.estimand='proximal_bridge' before fit().")
        U = np.asarray(U)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        tx, tw, tz, _tn = self._split_latent(U)
        t_bridge = np.concatenate([tx, tw], axis=1)
        self.bridge_head.eval()
        with torch.no_grad():
            tb = torch.tensor(t_bridge, device=self.device, dtype=torch.float32)
            
            # Use separated structure: directly get tau from tau_net
            if hasattr(self.bridge_head, 'get_tau'):
                tau_vals = self.bridge_head.get_tau(tb).detach().cpu().numpy().reshape(-1)
                tau = float(np.mean(tau_vals))
                self.training_diagnostics["bridge_tau_std"] = float(np.std(tau_vals))
            else:
                # Fallback to h(1) - h(0) for compatibility
                a1 = torch.ones((tb.shape[0],), device=self.device, dtype=torch.float32)
                a0 = torch.zeros((tb.shape[0],), device=self.device, dtype=torch.float32)
                h1 = self.bridge_head(tb, a1).detach().cpu().numpy().reshape(-1)
                h0 = self.bridge_head(tb, a0).detach().cpu().numpy().reshape(-1)
                tau = float(np.mean(h1 - h0))

            # diagnostics: residual HSIC with (A,X,Z) on full sample (approx)
            try:
                a_t = torch.tensor(A, device=self.device, dtype=torch.float32)
                y_t = torch.tensor(Y, device=self.device, dtype=torch.float32)
                h_obs = self.bridge_head(tb, a_t)
                r = (y_t - h_obs).unsqueeze(1)
                cond = torch.cat([a_t.unsqueeze(1), torch.tensor(tx, device=self.device, dtype=torch.float32),
                                  torch.tensor(tz, device=self.device, dtype=torch.float32)], dim=1)
                max_n = int(getattr(cfg, "bridge_hsic_max_samples", 256) or 256)
                if (max_n > 0) and (r.shape[0] > max_n):
                    idx = torch.randperm(r.shape[0], device=r.device)[:max_n]
                    r_h = r[idx]
                    cond_h = cond[idx]
                else:
                    r_h, cond_h = r, cond
                hs = float(_rbf_hsic(r_h, cond_h).item())
                self.training_diagnostics["bridge_resid_hsic_full"] = hs
            except Exception:
                pass

        return tau

    def _proximal_dr_ate(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        """Full Proximal Doubly-Robust estimator (Theorem 4.3).
        
        Uses both outcome bridge h(W,A,X) and treatment bridge q(Z,A,X):
        
        符号约定（与代码一致）：
        - q₁(Z,X): 满足 E[q₁|W,X,A=1] = 1/e(W,X)，用于 A=1 的校正
        - q₀(Z,X): 满足 E[q₀|W,X,A=0] = 1/(1-e(W,X))，用于 A=0 的校正
        
        τ_ATE = E[h₁(W,X) - h₀(W,X) 
                + A·q₁(Z,X)·(Y - h₁(W,X)) 
                - (1-A)·q₀(Z,X)·(Y - h₀(W,X))]
        
        This achieves Neyman orthogonality: bias is O(||h-h*|| * ||q-q*||).
        """
        cfg = self.config
        if not hasattr(self, "bridge_head") or self.bridge_head is None:
            raise RuntimeError("bridge_head is not initialized; set cfg.estimand='proximal_dr' before fit().")
        if not hasattr(self, "q_bridge_head") or self.q_bridge_head is None:
            # Fall back to plug-in estimator if q-bridge not available
            return self._proximal_bridge_ate(U, A, Y)
        
        U = np.asarray(U)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        n = len(A)
        
        tx, tw, tz, _tn = self._split_latent(U)
        t_bridge = np.concatenate([tx, tw], axis=1)
        
        self.bridge_head.eval()
        self.q_bridge_head.eval()
        
        with torch.no_grad():
            tb = torch.tensor(t_bridge, device=self.device, dtype=torch.float32)
            tx_t = torch.tensor(tx, device=self.device, dtype=torch.float32)
            tz_t = torch.tensor(tz, device=self.device, dtype=torch.float32)
            a_t = torch.tensor(A, device=self.device, dtype=torch.float32)
            y_t = torch.tensor(Y, device=self.device, dtype=torch.float32)
            
            # h₁(W,X) and h₀(W,X) from outcome bridge
            a1 = torch.ones((n,), device=self.device, dtype=torch.float32)
            a0 = torch.zeros((n,), device=self.device, dtype=torch.float32)
            h1 = self.bridge_head(tb, a1)
            h0 = self.bridge_head(tb, a0)
            
            # q₁(Z,X) and q₀(Z,X) from treatment bridge
            q1 = self.q_bridge_head.get_q1(tz_t, tx_t)
            q0 = self.q_bridge_head.get_q0(tz_t, tx_t)
            
            # Clamp q values for stability
            # Clamp q values for stability (use same range as training)
            t_clip_lo, t_clip_hi = getattr(cfg, "q_bridge_target_clip", (0.5, 5.0))
            q1 = q1.clamp(t_clip_lo, t_clip_hi)
            q0 = q0.clamp(t_clip_lo, t_clip_hi)
            
            # Residuals
            r1 = y_t - h1  # Y - h₁(W,X)
            r0 = y_t - h0  # Y - h₀(W,X)
            
            # ✅ 正确的 Proximal DR formula:
            # q₁ 用于 A=1 的校正，q₀ 用于 A=0 的校正
            psi = (h1 - h0) + a_t * q1 * r1 - (1 - a_t) * q0 * r0
            
            tau = float(psi.mean().item())
            
            # Diagnostics
            self.training_diagnostics["proximal_dr_h_term"] = float((h1 - h0).mean().item())
            self.training_diagnostics["proximal_dr_q1_term"] = float((a_t * q1 * r1).mean().item())
            self.training_diagnostics["proximal_dr_q0_term"] = float(((1 - a_t) * q0 * r0).mean().item())
            self.training_diagnostics["proximal_dr_q1_mean"] = float(q1.mean().item())
            self.training_diagnostics["proximal_dr_q0_mean"] = float(q0.mean().item())
            
            # Standard error estimate (influence function variance)
            psi_centered = psi - psi.mean()
            se = float(torch.sqrt((psi_centered ** 2).mean() / n).item())
            self.training_diagnostics["proximal_dr_se"] = se
            
        return tau

    def _proximal_dr_ate_crossfit(self, V_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        """Proximal DR with TRUE K-fold cross-fitting for Neyman orthogonality.
        
        Cross-fitting procedure:
        1. Split data into K folds
        2. For each fold k, RETRAIN h^{-k}, q^{-k}, e^{-k} on complement (encoder frozen)
        3. Compute DR estimate on fold k using h^{-k}, q^{-k}
        4. Average across folds
        
        This ensures the nuisance estimators are independent of the evaluation sample,
        achieving √n-consistency under ||h-h*|| * ||q-q*|| = o_p(n^{-1/2}).
        
        Key insight: We freeze the encoder (representation learning) and only retrain
        the lightweight bridge heads on each fold. This is much cheaper than full retraining
        while still providing the orthogonality benefits.
        """
        cfg = self.config
        K = int(getattr(cfg, "cross_fitting_folds", 3))  # Use 3 folds for speed
        n = len(A)
        
        if not getattr(cfg, "enable_cross_fitting", True) or K <= 1:
            # Fall back to non-cross-fitted estimate
            U = self.get_latent(V_all)
            return self._proximal_dr_ate(U, A, Y)
        
        from sklearn.model_selection import KFold
        
        # Get latent representations using frozen encoder
        U_all = self.get_latent(V_all)
        tx_all, tw_all, tz_all, _tn_all = self._split_latent(U_all)
        
        # Extract raw W, X for conditioning
        V_t = torch.tensor(V_all, device=self.device, dtype=torch.float32)
        x_raw_all = V_t[:, self._block_slices[0]].cpu().numpy()
        w_raw_all = V_t[:, self._block_slices[1]].cpu().numpy()
        
        kf = KFold(n_splits=K, shuffle=True, random_state=cfg.seed)
        
        # Store influence function values for each sample
        psi_all = np.zeros(n)
        fold_taus = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(V_all)):
            # --- Train new nuisance heads on training fold ---
            fold_h, fold_q, fold_e, fold_critic = self._train_fold_nuisances(
                tx_all[train_idx], tw_all[train_idx], tz_all[train_idx],
                x_raw_all[train_idx], w_raw_all[train_idx],
                A[train_idx], Y[train_idx],
                fold_idx=fold_idx
            )
            
            # --- Evaluate on test fold ---
            psi_fold = self._compute_dr_influence(
                tx_all[test_idx], tw_all[test_idx], tz_all[test_idx],
                A[test_idx], Y[test_idx],
                fold_h, fold_q, fold_e
            )
            
            psi_all[test_idx] = psi_fold
            fold_taus.append(float(np.mean(psi_fold)))
        
        tau = float(np.mean(psi_all))
        
        # Diagnostics
        self.training_diagnostics["proximal_dr_crossfit_folds"] = K
        self.training_diagnostics["proximal_dr_fold_taus"] = fold_taus
        self.training_diagnostics["proximal_dr_fold_std"] = float(np.std(fold_taus))
        self.training_diagnostics["proximal_dr_psi_std"] = float(np.std(psi_all))
        
        return tau
    
    def _train_fold_nuisances(self, tx_tr, tw_tr, tz_tr, x_raw_tr, w_raw_tr, A_tr, Y_tr, fold_idx=0):
        """Train nuisance heads (h-bridge, q-bridge, propensity) on a single fold.
        
        Encoder is frozen; only the lightweight heads are trained.
        This is much faster than full model training.
        
        v8.8: Warm-start option - initialize fold heads from globally trained heads.
        This reduces underfitting bias while maintaining cross-fitting variance reduction.
        """
        cfg = self.config
        device = self.device
        n_tr = len(A_tr)
        
        # Create heads for this fold
        bridge_in_dim = cfg.latent_x_dim + cfg.latent_w_dim
        fold_h = _BridgeHead(bridge_in_dim, cfg.bridge_hidden).to(device)
        
        fold_q = _QBridgeHead(
            cfg.latent_z_dim, cfg.latent_x_dim, 
            getattr(cfg, "q_bridge_hidden", (128, 64))
        ).to(device)
        
        fold_e = _AClassifier(cfg.latent_x_dim + cfg.latent_w_dim, cfg.a_hidden).to(device)
        
        w_raw_dim = self._block_slices[1].stop - self._block_slices[1].start
        x_raw_dim = self._block_slices[0].stop - self._block_slices[0].start
        fold_critic = _QBridgeCritic(w_raw_dim, x_raw_dim, getattr(cfg, "q_critic_hidden", (64, 32))).to(device)
        
        # v8.8: Warm-start from globally trained heads
        if getattr(cfg, "crossfit_warm_start", True):
            try:
                # Copy weights from global heads
                fold_h.load_state_dict(self.bridge_head.state_dict())
                if self.q_bridge_head is not None:
                    fold_q.load_state_dict(self.q_bridge_head.state_dict())
                if self.a_wx_head is not None:
                    fold_e.load_state_dict(self.a_wx_head.state_dict())
                if hasattr(self, 'q_bridge_critic') and self.q_bridge_critic is not None:
                    fold_critic.load_state_dict(self.q_bridge_critic.state_dict())
            except Exception:
                pass  # Fall back to random init if shapes don't match
        
        # Optimizers (use smaller LR for warm-started fine-tuning)
        lr_factor = 0.5 if getattr(cfg, "crossfit_warm_start", True) else 1.0
        h_opt = torch.optim.Adam(fold_h.parameters(), lr=cfg.lr_main * lr_factor)
        q_opt = torch.optim.Adam(fold_q.parameters(), lr=cfg.lr_main * lr_factor)
        e_opt = torch.optim.Adam(fold_e.parameters(), lr=cfg.lr_main * lr_factor)
        critic_opt = torch.optim.Adam(fold_critic.parameters(), lr=cfg.lr_main * 2 * lr_factor)
        
        # Convert to tensors
        tx_t = torch.tensor(tx_tr, device=device, dtype=torch.float32)
        tw_t = torch.tensor(tw_tr, device=device, dtype=torch.float32)
        tz_t = torch.tensor(tz_tr, device=device, dtype=torch.float32)
        x_raw_t = torch.tensor(x_raw_tr, device=device, dtype=torch.float32)
        w_raw_t = torch.tensor(w_raw_tr, device=device, dtype=torch.float32)
        a_t = torch.tensor(A_tr, device=device, dtype=torch.float32)
        y_t = torch.tensor(Y_tr, device=device, dtype=torch.float32)
        tb = torch.cat([tx_t, tw_t], dim=1)
        
        # Training epochs
        n_epochs = int(getattr(cfg, "crossfit_nuisance_epochs", 80))
        batch_size = min(256, n_tr)
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()
        
        for epoch in range(n_epochs):
            # Shuffle indices
            perm = torch.randperm(n_tr, device=device)
            
            for start in range(0, n_tr, batch_size):
                end = min(start + batch_size, n_tr)
                idx = perm[start:end]
                
                # --- (1) Train propensity e(W,X) ---
                logits_e = fold_e(tb[idx]).squeeze(-1)  # 确保是 1D
                loss_e = bce(logits_e, a_t[idx])
                e_opt.zero_grad(); loss_e.backward(); e_opt.step()
                
                # --- (2) Train h-bridge ---
                y_pred_h = fold_h(tb[idx], a_t[idx])
                loss_h = mse(y_pred_h, y_t[idx])
                h_opt.zero_grad(); loss_h.backward(); h_opt.step()
                
                # --- (3) Train q-bridge with adversarial moment (Min-Max) ---
                if epoch >= 5:  # Warmup
                    with torch.no_grad():
                        e_wx = torch.sigmoid(fold_e(tb[idx])).squeeze(-1).clamp(0.01, 0.99)
                    
                    t_clip_lo, t_clip_hi = getattr(cfg, "q_bridge_target_clip", (0.5, 5.0))
                    target_q = a_t[idx] / e_wx + (1 - a_t[idx]) / (1 - e_wx)
                    target_q = target_q.clamp(t_clip_lo, t_clip_hi)
                    
                    q_pred = fold_q(tz_t[idx], tx_t[idx], a_t[idx]).clamp(t_clip_lo, t_clip_hi)
                    q_resid = q_pred - target_q
                    
                    # Critic step: maximize E[f*resid] - 0.5*E[f^2]
                    for _ in range(2):
                        f_critic = fold_critic(w_raw_t[idx], x_raw_t[idx])
                        loss_critic = -(f_critic * q_resid.detach()).mean() + 0.5 * (f_critic ** 2).mean()
                        critic_opt.zero_grad(); loss_critic.backward(); critic_opt.step()
                    
                    # Q-bridge step: minimize E[f*resid] where f is fixed
                    q_pred_new = fold_q(tz_t[idx], tx_t[idx], a_t[idx]).clamp(t_clip_lo, t_clip_hi)
                    q_resid_new = q_pred_new - target_q.detach()
                    f_fixed = fold_critic(w_raw_t[idx], x_raw_t[idx]).detach()
                    
                    # KEY: (f_fixed * q_resid_new).mean() HAS gradient w.r.t. q!
                    q_adv_loss = (f_fixed * q_resid_new).mean()
                    q_mean_pen = (q_resid_new.mean() ** 2)
                    q_l2_pen = (q_pred_new ** 2).mean()
                    
                    loss_q = q_adv_loss + 0.1 * q_mean_pen + 1e-3 * q_l2_pen
                    q_opt.zero_grad(); loss_q.backward(); q_opt.step()
        
        fold_h.eval()
        fold_q.eval()
        fold_e.eval()
        
        return fold_h, fold_q, fold_e, fold_critic
    
    def _compute_dr_influence(self, tx, tw, tz, A, Y, h_head, q_head, e_head):
        """Compute DR influence function values using provided nuisance heads."""
        cfg = self.config
        device = self.device
        n = len(A)
        
        tx_t = torch.tensor(tx, device=device, dtype=torch.float32)
        tw_t = torch.tensor(tw, device=device, dtype=torch.float32)
        tz_t = torch.tensor(tz, device=device, dtype=torch.float32)
        a_t = torch.tensor(A, device=device, dtype=torch.float32)
        y_t = torch.tensor(Y, device=device, dtype=torch.float32)
        tb = torch.cat([tx_t, tw_t], dim=1)
        
        # Use same clip range as training
        t_clip_lo, t_clip_hi = getattr(cfg, "q_bridge_target_clip", (0.5, 5.0))
        
        with torch.no_grad():
            a1 = torch.ones(n, device=device)
            a0 = torch.zeros(n, device=device)
            
            h1 = h_head(tb, a1)
            h0 = h_head(tb, a0)
            
            q1 = q_head.get_q1(tz_t, tx_t).clamp(t_clip_lo, t_clip_hi)
            q0 = q_head.get_q0(tz_t, tx_t).clamp(t_clip_lo, t_clip_hi)
            
            r1 = y_t - h1
            r0 = y_t - h0
            
            # DR formula: psi = (h1 - h0) + A*q1*r1 - (1-A)*q0*r0
            psi = (h1 - h0) + a_t * q1 * r1 - (1 - a_t) * q0 * r0
            
        return psi.cpu().numpy()
    
    def _proximal_safe_dr_ate(self, U: np.ndarray, A: np.ndarray, Y: np.ndarray, V_raw: np.ndarray = None) -> float:
        """Safe Proximal DR with data-driven gating (v9.1).
        
        Instead of hard-coding "use plug-in for misaligned", this computes:
            tau = tau_plugin + alpha * correction
        
        where alpha ∈ [0,1] is determined by diagnostic signals:
        - moment_violation: how well critic can predict residual (high = bad)
        - correction_var: variance of DR correction term (high = unstable)
        - frac_clipped: fraction of e/q values clipped (high = weak overlap/misalignment)
        
        When diagnostics indicate q-bridge is unreliable, alpha → 0 (safe plug-in).
        When diagnostics are good, alpha → 1 (full DR correction).
        """
        cfg = self.config
        if not hasattr(self, "bridge_head") or self.bridge_head is None:
            raise RuntimeError("bridge_head is not initialized")
        
        U = np.asarray(U)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)
        n = len(A)
        
        tx, tw, tz, _tn = self._split_latent(U)
        t_bridge = np.concatenate([tx, tw], axis=1)
        
        self.bridge_head.eval()
        
        with torch.no_grad():
            tb = torch.tensor(t_bridge, device=self.device, dtype=torch.float32)
            a_t = torch.tensor(A, device=self.device, dtype=torch.float32)
            y_t = torch.tensor(Y, device=self.device, dtype=torch.float32)
            
            # h₁(W,X) and h₀(W,X) from outcome bridge
            a1 = torch.ones((n,), device=self.device, dtype=torch.float32)
            a0 = torch.zeros((n,), device=self.device, dtype=torch.float32)
            h1 = self.bridge_head(tb, a1)
            h0 = self.bridge_head(tb, a0)
            
            # Plug-in estimate (always computed)
            tau_plugin = float((h1 - h0).mean().item())
            
            # Check if q-bridge is available
            has_q = (hasattr(self, 'q_bridge_raw_head') and self.q_bridge_raw_head is not None) or \
                    (hasattr(self, 'q_bridge_head') and self.q_bridge_head is not None)
            
            if not has_q:
                # No q-bridge, return plug-in
                self.training_diagnostics["safe_dr_alpha"] = 0.0
                self.training_diagnostics["safe_dr_tau_plugin"] = tau_plugin
                self.training_diagnostics["safe_dr_correction"] = 0.0
                return tau_plugin
            
            # Get q values
            t_clip_lo, t_clip_hi = getattr(cfg, "q_bridge_target_clip", (0.5, 5.0))
            
            if V_raw is not None and hasattr(self, 'q_bridge_raw_head') and self.q_bridge_raw_head is not None:
                V_t = torch.tensor(V_raw, device=self.device, dtype=torch.float32)
                x_raw = V_t[:, self._block_slices[0]]
                z_raw = V_t[:, self._block_slices[2]]
                self.q_bridge_raw_head.eval()
                q1 = self.q_bridge_raw_head.get_q1(z_raw, x_raw)
                q0 = self.q_bridge_raw_head.get_q0(z_raw, x_raw)
            else:
                tx_t = torch.tensor(tx, device=self.device, dtype=torch.float32)
                tz_t = torch.tensor(tz, device=self.device, dtype=torch.float32)
                self.q_bridge_head.eval()
                q1 = self.q_bridge_head.get_q1(tz_t, tx_t)
                q0 = self.q_bridge_head.get_q0(tz_t, tx_t)
            
            # Compute clipping fraction BEFORE clipping
            frac_clipped_q1 = float(((q1 < t_clip_lo) | (q1 > t_clip_hi)).float().mean().item())
            frac_clipped_q0 = float(((q0 < t_clip_lo) | (q0 > t_clip_hi)).float().mean().item())
            frac_clipped = (frac_clipped_q1 + frac_clipped_q0) / 2
            
            # Clip q values
            q1 = q1.clamp(t_clip_lo, t_clip_hi)
            q0 = q0.clamp(t_clip_lo, t_clip_hi)
            
            # Residuals
            r1 = y_t - h1
            r0 = y_t - h0
            
            # DR correction term (per-sample)
            correction_per_sample = a_t * q1 * r1 - (1 - a_t) * q0 * r0
            correction = float(correction_per_sample.mean().item())
            correction_var = float(correction_per_sample.var().item())
            correction_std = float(correction_per_sample.std().item())
            
            # === Compute gating alpha based on diagnostics ===
            # 
            # Three signals of unreliable q-bridge:
            # 1. moment_violation: critic can predict residual (from training diagnostics)
            # 2. correction_var: high variance in correction term
            # 3. frac_clipped: many q values hit clip bounds
            
            # Get moment violation from training (if available)
            moment_violation = float(self.training_diagnostics.get("q_moment_violation", 0.0))
            
            # Normalize diagnostics to [0,1] scores (higher = worse)
            # These thresholds are calibrated from v8.5-v9.0 experiments
            score_moment = min(1.0, moment_violation / 0.5)  # 0.5 = "very bad"
            score_var = min(1.0, correction_std / (abs(tau_plugin) + 0.1))  # relative to effect size
            score_clip = min(1.0, frac_clipped / 0.3)  # 30% clipped = very bad
            
            # Combined "unreliability" score
            unreliability = max(score_moment, score_var * 0.5, score_clip)
            
            # Alpha: how much to trust the DR correction
            # unreliability=0 → alpha=1 (full DR)
            # unreliability=1 → alpha=0 (pure plug-in)
            alpha = max(0.0, min(1.0, 1.0 - unreliability))
            
            # Final estimate: safe blend
            tau = tau_plugin + alpha * correction
            
            # Store diagnostics
            self.training_diagnostics["safe_dr_alpha"] = alpha
            self.training_diagnostics["safe_dr_tau_plugin"] = tau_plugin
            self.training_diagnostics["safe_dr_correction"] = correction
            self.training_diagnostics["safe_dr_correction_std"] = correction_std
            self.training_diagnostics["safe_dr_frac_clipped"] = frac_clipped
            self.training_diagnostics["safe_dr_score_moment"] = score_moment
            self.training_diagnostics["safe_dr_score_var"] = score_var
            self.training_diagnostics["safe_dr_score_clip"] = score_clip
            self.training_diagnostics["safe_dr_unreliability"] = unreliability
            
        return tau
        
    def estimate_ate(self, V_all: np.ndarray, A: np.ndarray, Y: np.ndarray) -> float:
        """Estimate the treatment effect using the trained model.

        - estimand='ate'/'trimmed_ate'/'ato': DR-based effect on learned latent U.
        - estimand='proximal_bridge': plug-in proximal bridge effect using learned h(W,A,X).
        - estimand='proximal_dr': full Proximal DR with h-bridge and q-bridge.
        - estimand='proximal_safe_dr': Safe DR with data-driven gating (v9.1).
        """
        U = self.get_latent(V_all)
        estimand = str(getattr(self.config, "estimand", "ate")).lower().strip()
        
        if estimand in {"proximal_dr"}:
            # Full Proximal DR with both bridges
            if getattr(self.config, "enable_cross_fitting", True):
                return self._proximal_dr_ate_crossfit(V_all, A, Y)
            return self._proximal_dr_ate(U, A, Y)
        elif estimand in {"proximal_safe_dr", "safe_dr", "safe_proximal_dr"}:
            # v9.1: Safe DR with data-driven gating
            return self._proximal_safe_dr_ate(U, A, Y, V_raw=V_all)
        elif estimand in {"proximal_bridge", "proximal", "bridge"}:
            # Plug-in estimator using only h-bridge
            return self._proximal_bridge_ate(U, A, Y)
        else:
            # Standard DR-based estimators
            return self._dr_ate(U, A, Y)

    def get_training_diagnostics(self) -> Dict[str, float]:
        """Return training diagnostics dictionary."""
        return dict(self.training_diagnostics)


# Alias for RADR compatibility
class IVAPCIv33TheoryHierRADREstimator(IVAPCIv33TheoryHierEstimator):
    """Alias with RADR naming convention."""
    pass


__all__ = [
    "IVAPCIV33TheoryConfig",
    "WeakIVAdaptiveConfig",
    "WeakIVAdaptiveScheduler",
    "IVAPCIv33TheoryHierEstimator",
    "IVAPCIv33TheoryHierRADREstimator",
]
