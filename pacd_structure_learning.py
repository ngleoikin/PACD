"""
PACD 结构学习模块
=================

基于 PACD 理论实现:
1. p-adic 特征映射 (Section 2)
2. 因果方向识别 (Theorem 3.1) - 通过预测风险比较
3. 条件独立性检验 (Theorem 4.1) - p-adic 特征上的CI检验
4. 骨架学习 (PC算法 + p-adic CI)
5. 稳定性增强 (Bootstrap + 集成规则)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


@dataclass
class PACDStructureConfig:
    """PACD结构学习配置"""

    # p-adic 参数
    p: int = 2
    m: int = 4

    # CI检验参数
    alpha: float = 0.001
    max_k: int = 3
    ci_method: str = "spearman"
    use_nonparanormal: bool = False

    # CI 检验实现细节（更贴近定理：交叉拟合残差化 + 正则化）
    ci_crossfit: bool = True
    ci_ridge_alpha: float = 0.1

    # 方向识别参数
    direction_threshold: float = 0.1
    n_folds: int = 5

    direction_ridge_alpha: float = 0.1
    direction_use_risk_bounds: bool = True

    # 稳定性与集成参数
    n_bootstrap: int = 20
    edge_freq_threshold: float = 0.6
    residual_dependence_weight: float = 0.5

    # 结构定向一致性（PC: v-structure + Meek rules）
    orient_vstructures: bool = True
    use_meek_rules: bool = True
    ensure_acyclic: bool = True
    store_min_pvalue: bool = True
    cache_padic_features: bool = True

    # 敏感性分析（可选）：对 m / alpha 的小网格扰动，观测边与方向稳定性
    run_sensitivity: bool = False
    sensitivity_m_grid: Optional[List[int]] = None
    sensitivity_alpha_grid: Optional[List[float]] = None

    # 其他
    seed: int = 42


class PAdicFeatureMapper:
    """
    p-adic 特征映射 (Section 2)

    将连续变量 x 映射为有限精度的 p-adic 截断：
        φ_{p,m}(x) = Σ_{k=1}^{m} a_k p^{-k}
    其中 a_k = ⌊p^k x⌋ mod p，为 p 进制小数点后的第 k 位。

    说明：
    - 输入会按列 min-max 归一化到 [0,1]；
    - 支持 fit/transform 以复用归一化参数；
    - transform 内部采用向量化实现，避免对每个标量做 Python 循环。
    """

    def __init__(self, p: int = 2, m: int = 4):
        self.p = int(p)
        self.m = int(m)
        self._min: Optional[np.ndarray] = None
        self._range: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PAdicFeatureMapper":
        X = np.asarray(X)
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1.0
        self._min = X_min
        self._range = X_range
        return self

    def _normalize(self, X: np.ndarray, cols: Optional[List[int]] = None) -> np.ndarray:
        X = np.asarray(X)
        if self._min is None or self._range is None:
            # 未 fit：使用当前数据本身的 min-max（与旧实现兼容）
            X_min = X.min(axis=0, keepdims=True)
            X_max = X.max(axis=0, keepdims=True)
            X_range = X_max - X_min
            X_range[X_range == 0] = 1.0
            return np.clip((X - X_min) / X_range, 0.0, 1.0)

        if cols is not None:
            X_min = self._min[:, cols]
            X_range = self._range[:, cols]
        else:
            # 维度匹配则直接用；否则退化为局部 min-max（不覆盖已 fit 参数）
            if X.shape[1] == self._min.shape[1]:
                X_min = self._min
                X_range = self._range
            else:
                X_min = X.min(axis=0, keepdims=True)
                X_max = X.max(axis=0, keepdims=True)
                X_range = X_max - X_min
                X_range[X_range == 0] = 1.0

        return np.clip((X - X_min) / X_range, 0.0, 1.0)

    def transform(self, X: np.ndarray, cols: Optional[List[int]] = None) -> np.ndarray:
        """对数据矩阵进行 p-adic 变换；若已 fit，可通过 cols 选取对应列的归一化参数。"""
        Xn = self._normalize(X, cols=cols)
        # φ(x) = Σ_{k=1..m} digit_k(x) * p^{-k}
        p = self.p
        m = self.m
        out = np.zeros_like(Xn, dtype=float)
        # digits: floor(x * p^k) mod p
        for k in range(1, m + 1):
            scaled = np.floor(Xn * (p**k)).astype(int)
            digit = np.mod(scaled, p)
            out += digit * (p ** (-k))
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class PAdicCITest:
    """基于 p-adic 特征的条件独立性检验 (Theorem 4.1)"""

    def __init__(self, config: PACDStructureConfig):
        self.config = config
        self.mapper = PAdicFeatureMapper(config.p, config.m)
        # 可选：在 learn_skeleton() 前预计算 Φ(X) 以避免 PC 循环中重复 transform
        self._phi_all: Optional[np.ndarray] = None

    def set_cache(self, phi_all: Optional[np.ndarray]) -> None:
        """设置/清除全量 Φ(X) 缓存（shape: [n, d]）"""
        self._phi_all = phi_all

    def test_partial_correlation(
        self,
        X: np.ndarray,
        i: int,
        j: int,
        S: List[int],
    ) -> Tuple[bool, float, float]:
        """基于 p-adic 特征的偏相关检验"""
        n = X.shape[0]

        if not S:
            if self.config.ci_method == "pearson":
                r, _ = pearsonr(X[:, i], X[:, j])
            else:
                r, _ = spearmanr(X[:, i], X[:, j])
            z = 0.5 * np.log((1 + r + 1e-10) / (1 - r + 1e-10))
            se = 1.0 / np.sqrt(n - 3)
            p_value = 2 * (1 - norm.cdf(abs(z) / se))
            return p_value > self.config.alpha, r, p_value

        X_S = X[:, S]
        if self._phi_all is not None and getattr(self.config, "cache_padic_features", True):
            # 直接切片全量缓存（列选择与 transform(X_S) 等价，因为 min/max 按列计算）
            Phi_S = self._phi_all[:, S]
        else:
            Phi_S = self.mapper.transform(X_S)
        # 残差化：使用 Φ(S) 作为控制变量（Theorem 4.1 的实现更稳健形式：交叉拟合残差化）
        if getattr(self.config, "ci_crossfit", True) and n >= 2 * max(5, self.config.n_folds):
            res_i = np.empty(n, dtype=float)
            res_j = np.empty(n, dtype=float)
            kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.seed)
            for tr, te in kf.split(Phi_S):
                mi = Ridge(alpha=self.config.ci_ridge_alpha)
                mi.fit(Phi_S[tr], X[tr, i])
                res_i[te] = X[te, i] - mi.predict(Phi_S[te])
                mj = Ridge(alpha=self.config.ci_ridge_alpha)
                mj.fit(Phi_S[tr], X[tr, j])
                res_j[te] = X[te, j] - mj.predict(Phi_S[te])
        else:
            mi = Ridge(alpha=self.config.ci_ridge_alpha)
            mi.fit(Phi_S, X[:, i])
            res_i = X[:, i] - mi.predict(Phi_S)
            mj = Ridge(alpha=self.config.ci_ridge_alpha)
            mj.fit(Phi_S, X[:, j])
            res_j = X[:, j] - mj.predict(Phi_S)

        if self.config.ci_method == "pearson":
            r, _ = pearsonr(res_i, res_j)
        else:
            r, _ = spearmanr(res_i, res_j)
        z = 0.5 * np.log((1 + r + 1e-10) / (1 - r + 1e-10))
        df = max(n - len(S) - 3, 1)
        se = 1.0 / np.sqrt(df)
        p_value = 2 * (1 - norm.cdf(abs(z) / se))

        return p_value > self.config.alpha, r, p_value


class PAdicDirectionIdentifier:
    """基于 p-adic 特征的因果方向识别 (Theorem 3.1)"""

    def __init__(self, config: PACDStructureConfig):
        self.config = config
        self.mapper = PAdicFeatureMapper(config.p, config.m)

    def compute_prediction_risk(self, X: np.ndarray, Y: np.ndarray) -> float:
        """计算 E[|Y - h(Φ(X))|²] 的估计"""
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        self.mapper.fit(X)
        Phi_X = self.mapper.transform(X)
        kf = KFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.seed,
        )
        mse_scores = []
        for train_idx, test_idx in kf.split(Phi_X):
            model = Ridge(alpha=self.config.direction_ridge_alpha)
            model.fit(Phi_X[train_idx], Y[train_idx])
            y_pred = model.predict(Phi_X[test_idx])
            mse_scores.append(mean_squared_error(Y[test_idx], y_pred))
        return float(np.mean(mse_scores))

    def compute_prediction_risk_with_bounds(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """
        计算预测风险，并给出一个与定理直觉一致的误差界（粗略版）。

        返回包含：
          - risk_estimate: 交叉拟合 MSE
          - sigma_eps: 残差标准差估计
          - L_est: 线性模型系数范数（粗估 Lipschitz 常数）
          - lower_bound / upper_bound: σ^2 与 σ^2 + L^2 p^{-2m}
        """
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        self.mapper.fit(X)
        Phi_X = self.mapper.transform(X)
        kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.seed)
        y_hat = np.empty_like(Y, dtype=float)
        for tr, te in kf.split(Phi_X):
            model = Ridge(alpha=self.config.direction_ridge_alpha)
            model.fit(Phi_X[tr], Y[tr])
            y_hat[te] = model.predict(Phi_X[te])
        resid = Y - y_hat
        risk_est = float(np.mean(resid**2))
        sigma_eps = float(np.std(resid))

        # 用全样本线性拟合的系数范数粗估 Lipschitz 常数
        model_full = Ridge(alpha=self.config.direction_ridge_alpha)
        model_full.fit(Phi_X, Y)
        coef = getattr(model_full, "coef_", np.array([0.0]))
        L_est = float(np.linalg.norm(np.ravel(coef), ord=2))

        lower = sigma_eps**2
        upper = lower + (L_est**2) * (self.config.p ** (-2 * self.config.m))
        return {
            "risk_estimate": risk_est,
            "sigma_eps": sigma_eps,
            "L_est": L_est,
            "lower_bound": lower,
            "upper_bound": upper,
        }

    def compute_residual_dependence(self, X: np.ndarray, Y: np.ndarray) -> float:
        """估计残差与输入的相关性（越小越符合因果方向）"""
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        self.mapper.fit(X)
        Phi_X = self.mapper.transform(X)
        model = Ridge(alpha=self.config.direction_ridge_alpha)
        model.fit(Phi_X, Y)
        residuals = Y - model.predict(Phi_X)
        corr, _ = spearmanr(residuals, X[:, 0])
        return float(abs(corr))

    def identify_direction(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        var_name_x: str = "X",
        var_name_y: str = "Y",
        residual_weight: float = 0.0,
    ) -> Dict:
        """识别 X 和 Y 之间的因果方向"""
        risk_xy = self.compute_prediction_risk(X, Y)
        risk_yx = self.compute_prediction_risk(Y, X)
        risk_xy_bounds = None
        risk_yx_bounds = None
        if getattr(self.config, "direction_use_risk_bounds", True):
            try:
                risk_xy_bounds = self.compute_prediction_risk_with_bounds(X, Y)
                risk_yx_bounds = self.compute_prediction_risk_with_bounds(Y, X)
            except Exception:
                risk_xy_bounds = None
                risk_yx_bounds = None

        dep_xy = self.compute_residual_dependence(X, Y)
        dep_yx = self.compute_residual_dependence(Y, X)

        score_xy = risk_xy + residual_weight * dep_xy
        score_yx = risk_yx + residual_weight * dep_yx
        score_diff = score_yx - score_xy

        if score_diff > self.config.direction_threshold:
            direction = f"{var_name_x}->{var_name_y}"
            source, target = var_name_x, var_name_y
            confidence = min(abs(score_diff) / (score_xy + 1e-10), 1.0)
        elif score_diff < -self.config.direction_threshold:
            direction = f"{var_name_y}->{var_name_x}"
            source, target = var_name_y, var_name_x
            confidence = min(abs(score_diff) / (score_yx + 1e-10), 1.0)
        else:
            direction = f"{var_name_x}--{var_name_y}"
            source, target = var_name_x, var_name_y
            confidence = 0.0

        return {
            "direction": direction,
            "source": source,
            "target": target,
            "confidence": float(confidence),
            "risk_xy": float(risk_xy),
            "risk_yx": float(risk_yx),
            "risk_xy_bounds": risk_xy_bounds,
            "risk_yx_bounds": risk_yx_bounds,
            "risk_diff": float(risk_yx - risk_xy),
            "residual_dep_xy": float(dep_xy),
            "residual_dep_yx": float(dep_yx),
        }


class PACDStructureLearner:
    """PACD 结构学习器 (PC + p-adic CI)"""

    def __init__(self, config: Optional[PACDStructureConfig] = None):
        self.config = config or PACDStructureConfig()
        self.ci_tester = PAdicCITest(self.config)
        self.direction_identifier = PAdicDirectionIdentifier(self.config)

        self.skeleton_: Set[Tuple[int, int]] = set()
        self.sepsets_: Dict[Tuple[int, int], List[int]] = {}
        self.directions_: Dict[Tuple[int, int], Dict] = {}
        self.edge_pvalues_: Dict[Tuple[int, int], float] = {}
        self.edge_pvalues_sep_: Dict[Tuple[int, int], float] = {}

    def _neighbors(self, edges: Set[Tuple[int, int]], node: int, exclude: int) -> Set[int]:
        neighbors: Set[int] = set()
        for (a, b) in edges:
            if a == node and b != exclude:
                neighbors.add(b)
            elif b == node and a != exclude:
                neighbors.add(a)
        return neighbors

    def learn_skeleton(self, X: np.ndarray) -> Set[Tuple[int, int]]:
        """PC算法学习骨架 (使用 p-adic CI检验)"""
        _, d = X.shape
        if self.config.use_nonparanormal:
            X = self._nonparanormal_transform(X)
        # 预计算 Φ(X) 以加速 PC 循环中的条件独立性检验（可关闭）
        if getattr(self.config, "cache_padic_features", True):
            try:
                self.ci_tester.mapper.fit(X)
                phi_all = self.ci_tester.mapper.transform(X)
            except Exception:
                phi_all = None
            self.ci_tester.set_cache(phi_all)
        else:
            self.ci_tester.set_cache(None)

        edges = {(i, j) for i in range(d) for j in range(i + 1, d)}

        for k in range(self.config.max_k + 1):
            to_remove = []
            for (i, j) in list(edges):
                neighbors = self._neighbors(edges, i, j) | self._neighbors(edges, j, i)
                if len(neighbors) < k:
                    continue
                for S in combinations(neighbors, k):
                    is_indep, _, p = self.ci_tester.test_partial_correlation(X, i, j, list(S))
                    if getattr(self.config, "store_min_pvalue", True):
                        prev = self.edge_pvalues_.get((i, j), 1.0)
                        self.edge_pvalues_[(i, j)] = float(min(prev, p))
                    if is_indep:
                        to_remove.append((i, j))
                        self.sepsets_[(i, j)] = list(S)
                        self.sepsets_[(j, i)] = list(S)
                        self.edge_pvalues_sep_[(i, j)] = float(p)
                        self.edge_pvalues_sep_[(j, i)] = float(p)
                        break
                if (i, j) not in to_remove:
                    _, _, p = self.ci_tester.test_partial_correlation(X, i, j, [])
                    if getattr(self.config, "store_min_pvalue", True):
                        prev = self.edge_pvalues_.get((i, j), 1.0)
                        self.edge_pvalues_[(i, j)] = float(min(prev, p))
                    else:
                        self.edge_pvalues_[(i, j)] = float(p)
            for e in to_remove:
                edges.discard(e)

        self.skeleton_ = edges
        # 清除缓存，避免下次调用误用旧数据
        self.ci_tester.set_cache(None)
        return edges

    def _nonparanormal_transform(self, X: np.ndarray) -> np.ndarray:
        """Gaussian Copula变换"""
        n, d = X.shape
        X_transformed = np.zeros_like(X, dtype=float)
        for j in range(d):
            ranks = np.argsort(np.argsort(X[:, j])) + 1
            delta = 1.0 / (4.0 * n**0.25)
            ranks = np.clip(ranks / (n + 1), delta, 1 - delta)
            X_transformed[:, j] = norm.ppf(ranks)
        return X_transformed

    # --------- 全局定向：v-structure + Meek rules + 风险补完 ---------
    def _is_adjacent(
        self, a: int, b: int, undirected: Set[Tuple[int, int]], directed: Set[Tuple[int, int]]
    ) -> bool:
        if a == b:
            return True
        u, v = (a, b) if a < b else (b, a)
        if (u, v) in undirected:
            return True
        return (a, b) in directed or (b, a) in directed

    def _orient_edge(
        self, src: int, dst: int, undirected: Set[Tuple[int, int]], directed: Set[Tuple[int, int]]
    ) -> bool:
        """尝试定向 src->dst；若已存在 dst->src 则不做变更"""
        if (dst, src) in directed:
            return False
        if getattr(self.config, "ensure_acyclic", True) and self._would_create_cycle(
            directed, src, dst
        ):
            return False
        u, v = (src, dst) if src < dst else (dst, src)
        undirected.discard((u, v))
        if (src, dst) not in directed:
            directed.add((src, dst))
            return True
        return False

    def _would_create_cycle(self, directed: Set[Tuple[int, int]], src: int, dst: int) -> bool:
        """若加入 src->dst 是否会形成环（小图上用 DFS 即可）"""
        # 若已存在 dst => ... => src 路径，则 src->dst 会成环
        stack = [dst]
        visited = set()
        adj = {}
        for a, b in directed:
            adj.setdefault(a, set()).add(b)
        while stack:
            cur = stack.pop()
            if cur == src:
                return True
            if cur in visited:
                continue
            visited.add(cur)
            for nxt in adj.get(cur, ()):
                if nxt not in visited:
                    stack.append(nxt)
        return False

    def _orient_vstructures(
        self,
        d: int,
        undirected: Set[Tuple[int, int]],
        directed: Set[Tuple[int, int]],
        edge_meta: Dict[Tuple[int, int], str],
    ) -> None:
        if not getattr(self.config, "orient_vstructures", True):
            return
        # i - k - j, 且 i 与 j 不相邻；若 k 不在 sepset(i,j)，则 i->k<-j
        for k in range(d):
            nbrs = [n for n in range(d) if n != k and self._is_adjacent(k, n, undirected, directed)]
            for i, j in combinations(nbrs, 2):
                if self._is_adjacent(i, j, undirected, directed):
                    continue
                sepset = self.sepsets_.get((i, j)) or self.sepsets_.get((j, i)) or []
                if k not in sepset:
                    if self._orient_edge(i, k, undirected, directed):
                        edge_meta[(i, k)] = "v-structure"
                    if self._orient_edge(j, k, undirected, directed):
                        edge_meta[(j, k)] = "v-structure"

    def _apply_meek_rules(
        self,
        d: int,
        undirected: Set[Tuple[int, int]],
        directed: Set[Tuple[int, int]],
        edge_meta: Dict[Tuple[int, int], str],
    ) -> None:
        if not getattr(self.config, "use_meek_rules", True):
            return
        changed = True
        while changed:
            changed = False
            # R1: a->b, b-c, a not adj c  => b->c
            dir_list = list(directed)
            und_list = list(undirected)
            for a, b in dir_list:
                for (u, v) in und_list:
                    if u == b:
                        c = v
                    elif v == b:
                        c = u
                    else:
                        continue
                    if c == a:
                        continue
                    if not self._is_adjacent(a, c, undirected, directed):
                        if self._orient_edge(b, c, undirected, directed):
                            edge_meta[(b, c)] = "meek-R1"
                            changed = True

            # R2: a-b 且存在 a->c->b => a->b（对称检查）
            und_list = list(undirected)
            for (u, v) in und_list:
                # u->c->v
                for c in range(d):
                    if (u, c) in directed and (c, v) in directed:
                        if self._orient_edge(u, v, undirected, directed):
                            edge_meta[(u, v)] = "meek-R2"
                            changed = True
                        break
                    if (v, c) in directed and (c, u) in directed:
                        if self._orient_edge(v, u, undirected, directed):
                            edge_meta[(v, u)] = "meek-R2"
                            changed = True
                        break

    def _orient_remaining_by_risk(
        self,
        X: np.ndarray,
        var_names: List[str],
        undirected: Set[Tuple[int, int]],
        directed: Set[Tuple[int, int]],
        edge_meta: Dict[Tuple[int, int], str],
        residual_weight: float,
    ) -> Tuple[Dict[Tuple[int, int], Dict], Dict[Tuple[int, int], Dict]]:
        """对剩余未定向边用风险比较补完；返回这些边的 risk 结果"""
        risk_results: Dict[Tuple[int, int], Dict] = {}
        risk_cache: Dict[Tuple[int, int], Dict] = {}
        for (u, v) in sorted(undirected):
            res = self.direction_identifier.identify_direction(
                X[:, u], X[:, v], var_names[u], var_names[v], residual_weight=residual_weight
            )
            risk_cache[(u, v)] = res
            # undecided 直接跳过（保持无向，不输出为有向边）
            if "--" in res.get("direction", ""):
                continue
            src_name, dst_name = res["source"], res["target"]
            src = u if src_name == var_names[u] else v
            dst = v if src == u else u
            if self._orient_edge(src, dst, undirected, directed):
                edge_meta[(src, dst)] = "risk"
                risk_results[(src, dst)] = res
        return risk_results, risk_cache

    def _orient_edges_global(
        self, X: np.ndarray, var_names: List[str], residual_weight: float = 0.0
    ) -> List[Dict]:
        _, d = X.shape
        undirected: Set[Tuple[int, int]] = {tuple(sorted(e)) for e in self.skeleton_}
        directed: Set[Tuple[int, int]] = set()
        edge_meta: Dict[Tuple[int, int], str] = {}

        # 1) v-structures
        self._orient_vstructures(d, undirected, directed, edge_meta)
        # 2) Meek rules propagation
        self._apply_meek_rules(d, undirected, directed, edge_meta)
        # 3) risk-based completion (optional)
        risk_results, risk_cache = self._orient_remaining_by_risk(
            X, var_names, undirected, directed, edge_meta, residual_weight=residual_weight
        )

        directed_edges: List[Dict] = []
        for (src, dst) in sorted(directed):
            method = edge_meta.get((src, dst), "rule")
            # sepset 取无向对的存储（若存在）
            sepset_idx = self.sepsets_.get((src, dst)) or self.sepsets_.get((dst, src)) or []
            edge = {
                "source": var_names[src],
                "target": var_names[dst],
                "orientation_method": method,
                "sepset": [var_names[s] for s in sepset_idx],
            }
            if method == "risk" and (src, dst) in risk_results:
                rr = risk_results[(src, dst)]
                conf = float(rr.get("confidence", 0.0))
                edge.update(
                    {
                        "risk_diff": float(rr.get("risk_diff", 0.0)),
                        "confidence": conf,
                        "direction_confidence": (
                            "high" if conf > 0.5 else ("medium" if conf > 0.2 else "low")
                        ),
                    }
                )
            else:
                edge.update(
                    {
                        "risk_diff": None,
                        "confidence": 1.0,
                        "direction_confidence": "rule",
                    }
                )
            directed_edges.append(edge)
        # 对剩余无向边：保留为“未决定”输出（不强行定向），但仍给出 risk/dep 指标供人工判断
        for (u, v) in sorted(undirected):
            sepset_idx = self.sepsets_.get((u, v)) or self.sepsets_.get((v, u)) or []
            rr = risk_cache.get(
                (u, v),
                self.direction_identifier.identify_direction(
                    X[:, u], X[:, v], var_names[u], var_names[v], residual_weight=residual_weight
                ),
            )
            directed_edges.append(
                {
                    "source": var_names[u],
                    "target": var_names[v],
                    "orientation_method": "undirected",
                    "direction": rr.get("direction"),
                    "risk_diff": float(rr.get("risk_diff", 0.0)),
                    "confidence": float(rr.get("confidence", 0.0)),
                    "direction_confidence": "undecided",
                    "sepset": [var_names[s] for s in sepset_idx],
                }
            )

        return directed_edges

    def orient_edges(self, X: np.ndarray, var_names: List[str]) -> List[Dict]:
        """全局一致定向：v-structure + Meek rules + 风险补完"""
        return self._orient_edges_global(X, var_names, residual_weight=0.0)

    def learn(self, X: np.ndarray, var_names: List[str]) -> Dict:
        """完整结构学习"""
        skeleton = self.learn_skeleton(X)
        directed = self.orient_edges(X, var_names)
        sensitivity = None
        if getattr(self.config, "run_sensitivity", False):
            sensitivity = self.sensitivity_analysis(X, var_names)

        sepsets_named = {
            f"{var_names[i]}|{var_names[j]}": [var_names[s] for s in S]
            for (i, j), S in self.sepsets_.items()
        }
        directed_count = sum(
            1 for edge in directed if edge.get("orientation_method") != "undirected"
        )
        undirected_count = len(directed) - directed_count
        return {
            "skeleton": [(var_names[i], var_names[j]) for (i, j) in skeleton],
            "skeleton_indices": list(skeleton),
            "directed_edges": directed,
            "sepsets": sepsets_named,
            "n_edges": directed_count,
            "n_undirected": undirected_count,
            "sensitivity_analysis": sensitivity,
        }

    def sensitivity_analysis(self, X: np.ndarray, var_names: List[str]) -> Dict:
        """
        对关键超参做小网格扰动，统计边/方向的稳定性。

        这对应附件观点中的“敏感性分析”：如果轻微改变 m 或正则化强度，
        得到的骨架/方向是否保持一致。
        """
        cfg = self.config
        m0 = int(cfg.m)
        a0 = float(getattr(cfg, "ci_ridge_alpha", 0.1))

        m_grid = cfg.sensitivity_m_grid
        if m_grid is None:
            m_grid = sorted({max(1, m0 - 1), m0, m0 + 1})
        a_grid = cfg.sensitivity_alpha_grid
        if a_grid is None:
            a_grid = [max(1e-6, a0 / 2), a0, a0 * 2]

        edge_counts: Dict[Tuple[str, str], int] = {}
        dir_counts: Dict[Tuple[str, str], int] = {}
        total = 0
        for m in m_grid:
            for a in a_grid:
                cfg2 = replace(cfg, m=int(m), ci_ridge_alpha=float(a), direction_ridge_alpha=float(a))
                learner = PACDStructureLearner(cfg2)
                out = learner.learn(X, var_names)
                total += 1
                for e in out.get("skeleton", []):
                    k = tuple(sorted(e))
                    edge_counts[k] = edge_counts.get(k, 0) + 1
                for e in out.get("directed_edges", []):
                    if e.get("orientation_method") == "undirected":
                        continue
                    k = (e["source"], e["target"])
                    dir_counts[k] = dir_counts.get(k, 0) + 1

        edge_freq = [
            {"node1": k[0], "node2": k[1], "freq": v / max(1, total)}
            for k, v in edge_counts.items()
        ]
        direction_freq = [
            {"source": k[0], "target": k[1], "freq": v / max(1, total)}
            for k, v in dir_counts.items()
        ]
        edge_freq.sort(key=lambda x: (-x["freq"], x["node1"], x["node2"]))
        direction_freq.sort(key=lambda x: (-x["freq"], x["source"], x["target"]))
        return {
            "grid": {"m": m_grid, "alpha": a_grid},
            "n_runs": total,
            "edge_frequency": edge_freq,
            "direction_frequency": direction_freq,
        }


class PACDEnsembleStructureLearner(PACDStructureLearner):
    """
    集成式结构学习器:
    - Bootstrap 稳定性选择
    - 风险 + 残差依赖度联合定向
    """

    def __init__(self, config: Optional[PACDStructureConfig] = None):
        super().__init__(config)
        # 复用同一个 RNG，确保 bootstrap 抽样在同一 seed 下可复现但每次不同
        self._rng = np.random.default_rng(self.config.seed)

    def _bootstrap_indices(self, n: int) -> np.ndarray:
        return self._rng.integers(0, n, size=n)

    def _compute_sepsets_for_graph(
        self, X: np.ndarray, edges: Set[Tuple[int, int]]
    ) -> None:
        n, d = X.shape
        self.sepsets_.clear()
        self.edge_pvalues_sep_.clear()
        if getattr(self.config, "cache_padic_features", True):
            try:
                self.ci_tester.mapper.fit(X)
                phi_all = self.ci_tester.mapper.transform(X)
            except Exception:
                phi_all = None
            self.ci_tester.set_cache(phi_all)
        else:
            self.ci_tester.set_cache(None)

        for k in range(self.config.max_k + 1):
            for i in range(d):
                for j in range(i + 1, d):
                    if (i, j) in edges or (i, j) in self.sepsets_:
                        continue
                    neighbors = self._neighbors(edges, i, j) | self._neighbors(edges, j, i)
                    if len(neighbors) < k:
                        continue
                    for S in combinations(neighbors, k):
                        is_indep, _, p = self.ci_tester.test_partial_correlation(X, i, j, list(S))
                        if is_indep:
                            self.sepsets_[(i, j)] = list(S)
                            self.sepsets_[(j, i)] = list(S)
                            self.edge_pvalues_sep_[(i, j)] = float(p)
                            self.edge_pvalues_sep_[(j, i)] = float(p)
                            break

        self.ci_tester.set_cache(None)

    def learn_skeleton(self, X: np.ndarray) -> Set[Tuple[int, int]]:
        n, d = X.shape
        edge_counts = np.zeros((d, d), dtype=int)
        base_edges = set()

        for _ in range(self.config.n_bootstrap):
            idx = self._bootstrap_indices(n)
            learner = PACDStructureLearner(self.config)
            edges = learner.learn_skeleton(X[idx])
            base_edges |= edges
            for (i, j) in edges:
                edge_counts[i, j] += 1
                edge_counts[j, i] += 1

        threshold = int(np.ceil(self.config.edge_freq_threshold * self.config.n_bootstrap))
        stable_edges = {
            (i, j)
            for (i, j) in base_edges
            if edge_counts[i, j] >= threshold
        }

        self.skeleton_ = stable_edges
        # 基于稳定骨架补跑 sepset，避免与 bootstrap 路径不一致
        self._compute_sepsets_for_graph(X, stable_edges)
        return stable_edges

    def orient_edges(self, X: np.ndarray, var_names: List[str]) -> List[Dict]:
        """集成版本的全局一致定向：规则优先，风险补完（带残差依赖度）"""
        edges = self._orient_edges_global(
            X, var_names, residual_weight=self.config.residual_dependence_weight
        )
        # 为风险补完的边补充残差依赖度信息（规则定向的边用 None）
        for e in edges:
            if e.get("orientation_method") == "risk":
                src = var_names.index(e["source"])
                dst = var_names.index(e["target"])
                # 重新计算一次（小规模开销可接受）；也可后续做缓存
                dep_xy = self.direction_identifier.compute_residual_dependence(X[:, src], X[:, dst])
                dep_yx = self.direction_identifier.compute_residual_dependence(X[:, dst], X[:, src])
                e["residual_dep_xy"] = float(dep_xy)
                e["residual_dep_yx"] = float(dep_yx)
            else:
                e["residual_dep_xy"] = None
                e["residual_dep_yx"] = None
        return edges


def demo() -> None:
    np.random.seed(42)
    n = 1000
    X1 = np.random.randn(n)
    X2 = 0.8 * X1 + np.random.randn(n) * 0.5
    X3 = 0.5 * X1 + 0.6 * X2 + np.random.randn(n) * 0.3

    data = np.column_stack([X1, X2, X3])
    var_names = ["X1", "X2", "X3"]

    config = PACDStructureConfig(p=2, m=4, alpha=0.01)
    learner = PACDEnsembleStructureLearner(config)
    result = learner.learn(data, var_names)

    print("\n结果:")
    print(f"  边数: {result['n_edges']}")
    for edge in result["directed_edges"]:
        print(
            f"  {edge['source']} -> {edge['target']} "
            f"(conf={edge['direction_confidence']})"
        )


if __name__ == "__main__":
    demo()
